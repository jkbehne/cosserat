#include <cosserat/simulation/simulation_graph.hpp>

#include <ranges>
#include <variant>

#include <spdlog/spdlog.h>

#include <cosserat/math/indexing.hpp>

#include <cosserat/utils/assertions.hpp>

namespace cosserat::simulation {

using utils::nice_assert;

JointEdge::JointEdge(
    const std::string& name1,
    const std::string& name2,
    Eigen::Index element_one_idx_,
    Eigen::Index element_two_idx_,
    bool joint_bottom_one,
    bool joint_bottom_two,
    physics::JointVariant joint_
) : body1_name(name1),
    body2_name(name2),
    element_one_idx(element_one_idx_),
    element_two_idx(element_two_idx_),
    node_one_idx(),
    node_two_idx(),
    joint(std::move(joint_))
{
    nice_assert(element_one_idx >= 0, "Element indices must be >= zero");
    nice_assert(element_two_idx >= 0, "Element indices must be >= zero");
    node_one_idx = joint_bottom_one ? element_one_idx : element_one_idx + 1;
    node_two_idx = joint_bottom_two ? element_two_idx : element_two_idx + 1;
}

// Closes the graph. Everything below this point is registration, which may
// only happen before finalizing; everything above is per-step work, which may
// only happen after.
void SimulationGraph::finalize()
{
    nice_assert(not finalized, "System has already been finalized");
    nice_assert(not m_bodies.empty(), "Can't finalize a system with no bodies");
    if (m_contacts.empty()) spdlog::warn("No contacts declared. All bodies are not solid");

    for (const auto& value : m_bodies | std::views::values)
    {
        m_final_systems.push_back(value);
    }
    finalized = true;
    spdlog::info("Finalized simulation");
}

std::vector<typename SimulationGraph::SubSystemType>& SimulationGraph::final_systems()
{
    nice_assert(finalized, "finalized must be true before final_systems() can be called");
    return m_final_systems;
}

// Phase 1 and 4 of a step. Runs immediately after each half kinematic step,
// so a pinned node is put back before anything reads its position: the
// internal-force pass that follows would otherwise see a configuration the
// boundary condition was supposed to forbid.
void SimulationGraph::constrain_values(double time)
{
    spdlog::debug("Constraining values at simulation time {}", time);
    for (const auto& key : std::views::keys(m_constraints))
    {
        nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
        for (auto& constraint : m_constraints.at(key))
        {
            auto visitor = [&](auto& arg) {physics::constrain_values(constraint, arg, time);};
            std::visit(visitor, m_bodies.at(key).body());
        }
    }
}

// Phase 2 of a step, and the only one where more than one kind of rule fires.
// Runs after every body has computed its internal loads, because rod-to-rod
// contact reads them.
//
// The three groups run joints, then forces, then contacts. Joints and forces
// only add into the external accumulators, so their order relative to one
// another does not change the result. Contacts must come last: rod-to-rod
// contact reads the accumulated external forces to work out how hard the two
// bodies are already being pressed together, and would see zeros if it ran
// first.
void SimulationGraph::synchronize(double time)
{
    spdlog::debug("Snychronizing bodies at simulation time {}", time);

    // --- joints first: they only accumulate, so they may go before forces ---

    for (auto& joint : m_joints)
    {
        const auto& name1 = joint.body1_name;
        const auto& name2 = joint.body2_name;
        nice_assert(m_bodies.contains(name1), "Body name " + name1 + " not found");
        nice_assert(m_bodies.contains(name2), "Body name " + name2 + " not found");
        auto& variant1 = m_bodies.at(name1).body();
        auto& variant2 = m_bodies.at(name2).body();
        auto& joint_variant = joint.joint;
        const auto element_one_idx = joint.element_one_idx;
        const auto element_two_idx = joint.element_two_idx;
        const auto node_one_idx = joint.node_one_idx;
        const auto node_two_idx = joint.node_two_idx;
        std::visit(
            [&](auto& body2)
            {
                std::visit(
                    [&](auto& body1)
                    {
                        physics::apply_forces(
                            joint_variant, body1, node_one_idx, body2, node_two_idx, time
                        );
                        physics::apply_torques(
                            joint_variant, body1, element_one_idx, body2, element_two_idx, time
                        );
                    }, variant1
                );
            }, variant2
        );
    }

    // --- then force and torque rules, likewise purely additive -------------
    for (const auto& key : std::views::keys(m_forces))
    {
        nice_assert(m_bodies.contains(key), "Key " + key + " not found in body map");
        for (auto& force_variant : m_forces.at(key))
        {
            auto visitor = [&](auto& body)
            {
                physics::apply_forces(force_variant, body, time);
                physics::apply_torques(force_variant, body, time);
            };
            std::visit(visitor, m_bodies.at(key).body());
        }
    }

    // --- contacts last: they read what the two groups above accumulated ----
    for (const auto& key : std::views::keys(m_contacts))
    {
        const auto& name1 = key.first;
        const auto& name2 = key.second;
        auto& contact = m_contacts.at(key);
        if (name1 == name2)
        {
            const auto& name = name1;
            nice_assert(m_bodies.contains(name), "Key " + name + " not found in body map");
            std::visit(
                [&](auto& body){physics::apply_contact(contact, body, time);},
                m_bodies.at(name).body()
            );
        }
        else
        {
            nice_assert(m_bodies.contains(name1), "Key " + name1 + " not found in body map");
            nice_assert(m_bodies.contains(name2), "Key " + name2 + " not found in body map");
            auto& variant1 = m_bodies.at(name1).body();
            auto& variant2 = m_bodies.at(name2).body();
            std::visit(
                [&](auto& body2)
                {
                    std::visit(
                        [&](auto& body1)
                        {
                            physics::apply_contact(contact, body1, body2, time);
                        }, variant1
                    );
                }, variant2
            );
        }
    }
}

// Phase 3 of a step, after the rates have been advanced. Damping lives here
// rather than in a phase of its own, matching the reference: a damper scales
// rates that a boundary condition may just have pinned to zero, so the
// constraints run first and pinning wins.
void SimulationGraph::constrain_rates(double time)
{
    spdlog::debug("Constraing rates at simulation time {}", time);

    // --- boundary conditions first, so that pinning survives damping -------
    for (const auto& key : std::views::keys(m_constraints))
    {
        nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
        for (auto& constraint : m_constraints.at(key))
        {
            auto visitor = [&](auto& arg) {physics::constrain_rates(constraint, arg, time);};
            std::visit(visitor, m_bodies.at(key).body());
        }
    }

    // --- then damping, chained onto whatever the constraints left ----------
    for (const auto& key : std::views::keys(m_dampers))
    {
        nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
        for (auto& damper : m_dampers.at(key))
        {
            auto visitor = [&](auto& arg) {physics::dampen_rates(damper, arg, time);};
            std::visit(visitor, m_bodies.at(key).body());
        }
    }
}

// Phase 5, the last thing a step does before the external accumulators are
// cleared. That ordering is deliberate: a diagnostic writing at this point
// sees the loads that produced the step it is recording, where clearing first
// would leave every recorded frame with empty force stacks.
void SimulationGraph::apply_callbacks(double time, std::uint64_t step)
{
    spdlog::debug("Applying call backs for step {} at simulation time {}", step, time);
    for (const auto& key : std::views::keys(m_diagnostics))
    {
        for (auto& diagnostic : m_diagnostics.at(key))
        {
            auto visitor = [&](auto& arg) {make_callback(diagnostic, arg, time, step);};
            nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
            std::visit(visitor, m_bodies.at(key).body());
        }
    }
}

void SimulationGraph::add_body(const std::string& name, BodyVariant body)
{
    nice_assert(not finalized, "Can't add body to finalized simulation");
    nice_assert(not m_bodies.contains(name), "Name (" + name + ") already found in body map");
    spdlog::info("Adding body with name {} to simulation", name);
    m_bodies.insert_or_assign(name, std::move(body));
}

void SimulationGraph::add_constraint(const std::string& name, ConstraintVariant constraint)
{
    nice_assert(not finalized, "Can't add constraint to finalized simulation");
    nice_assert(m_bodies.contains(name),
        "Name (" + name + ") not found in body map. Can't add constraint");
    auto visitor = [&](auto& arg) {physics::validate(constraint, arg);};
    std::visit(visitor, m_bodies.at(name).body());
    spdlog::info("Add constraint to body {}", name);
    if (not m_constraints.contains(name)) m_constraints.insert_or_assign(name, ConstraintVector{});
    m_constraints.at(name).push_back(std::move(constraint));
}

void SimulationGraph::add_connection(
    const std::string& name1, const std::string& name2,
    Eigen::Index element_one_idx, Eigen::Index element_two_idx,
    bool joint_bottom_one, bool joint_bottom_two, JointVariant joint)
{
    nice_assert(not finalized, "Can't add connection to finalized simulation");
    nice_assert(m_bodies.contains(name1),
        "Name (" + name1 + ") not found in body map. Can't add connection");
    nice_assert(m_bodies.contains(name2),
        "Name (" + name2 + ") not found in body map. Can't add connection");

    auto& variant1 = m_bodies.at(name1).body();
    auto visitor = [&](auto& arg2)
    {
        auto nested_visitor = [&](auto& arg1){physics::validate(joint, arg1, arg2);};
        std::visit(nested_visitor, variant1);
    };
    std::visit(visitor, m_bodies.at(name2).body());

    // Returned from the visit rather than assigned into an outer variable, so
    // the compiler can see they are always initialised.
    const Eigen::Index index_one_resolved = std::visit(
        [&](auto& body) { return resolve_index(element_one_idx, body.num_elements()); },
        m_bodies.at(name1).body()
    );
    const Eigen::Index index_two_resolved = std::visit(
        [&](auto& body) { return resolve_index(element_two_idx, body.num_elements()); },
        m_bodies.at(name2).body()
    );

    spdlog::info("Connecting body {} to {}", name1, name2);
    m_joints.push_back(
        JointEdge(name1, name2, index_one_resolved, index_two_resolved,
            joint_bottom_one, joint_bottom_two, std::move(joint)));
}

void SimulationGraph::add_contact(const std::string& name1, const std::string& name2, ContactVariant contact)
{
    nice_assert(not finalized, "Can't add contact to finalized simulation");
    nice_assert(not (name1 == name2), "Use two argument add_contact for self contacts");
    nice_assert(m_bodies.contains(name1),
        "Name (" + name1 + ") not found in body map. Can't add contact");
    nice_assert(m_bodies.contains(name2),
        "Name (" + name2 + ") not found in body map. Can't add contact");
    auto& variant1 = m_bodies.at(name1).body();
    auto visitor = [&](auto& arg2)
    {
        auto nested_visitor = [&](auto& arg1){physics::validate(contact, arg1, arg2);};
        std::visit(nested_visitor, variant1);
    };
    std::visit(visitor, m_bodies.at(name2).body());
    // The rule cares which system is first, but the pair of bodies is one
    // physical interaction however it is written down. Registering it both
    // ways round would apply it twice on every step.
    nice_assert(
        not m_contacts.contains(std::make_pair(name1, name2)),
        "Contact between (" + name1 + ", " + name2 + ") is already declared"
    );
    nice_assert(
        not m_contacts.contains(std::make_pair(name2, name1)),
        "Contact between (" + name1 + ", " + name2 + ") is already declared as ("
            + name2 + ", " + name1 + ")"
    );

    spdlog::info("Initiating contact detection between body {} to {}", name1, name2);
    m_contacts.insert_or_assign(std::make_pair(name1, name2), std::move(contact));
}

void SimulationGraph::add_contact(const std::string& name, ContactVariant contact)
{
    nice_assert(not finalized, "Can't add contact to finalized simulation");
    nice_assert(m_bodies.contains(name),
        "Name (" + name + ") not found in body map. Can't add contact");
    auto visitor = [&](auto& arg) {physics::validate(contact, arg);};
    std::visit(visitor, m_bodies.at(name).body());
    // A self contact keys on the name paired with itself, so both orderings
    // are the same key and one check covers it.
    nice_assert(
        not m_contacts.contains(std::make_pair(name, name)),
        "Self contact for body " + name + " is already declared"
    );

    spdlog::info("Initiating self-contact detection for body {}", name);
    m_contacts.insert_or_assign(std::make_pair(name, name), std::move(contact));
}

void SimulationGraph::add_forcing_to(const std::string& name, ForceVariant force)
{
    nice_assert(not finalized, "Can't add forcing to finalized simulation");
    nice_assert(m_bodies.contains(name),
        "Name (" + name + ") not found in body map. Can't add force");
    auto visitor = [&](auto& arg) {physics::validate(force, arg);};
    std::visit(visitor, m_bodies.at(name).body());
    spdlog::info("Adding forcing to body {}", name);
    if (not m_forces.contains(name)) m_forces.insert_or_assign(name, ForceVector{});
    m_forces.at(name).push_back(std::move(force));
}

void SimulationGraph::dampen(const std::string& name, DampeningVariant damper)
{
    nice_assert(not finalized, "Can't add damping to finalized simulation");
    nice_assert(m_bodies.contains(name),
        "Name (" + name + ") not found in body map. Can't add damping");
    auto visitor = [&](auto& arg) {physics::validate(damper, arg);};
    std::visit(visitor, m_bodies.at(name).body());
    spdlog::info("Adding damper to body {}", name);
    if (not m_dampers.contains(name)) m_dampers.insert_or_assign(name, DampeningVector{});
    m_dampers.at(name).push_back(std::move(damper));
}

void SimulationGraph::collect_diagnostics(const std::string& name, DiagnosticVariant diagnostic)
{
    nice_assert(not finalized, "Can't add diagnostic to finalized simulation");
    nice_assert(m_bodies.contains(name),
        "Name (" + name + ") not found in body map. Can't add diagnostic");
    auto visitor = [&](auto& arg) {validate(diagnostic, arg);};
    std::visit(visitor, m_bodies.at(name).body());
    spdlog::info("Adding diagnostic to body {}", name);
    if (not m_diagnostics.contains(name)) m_diagnostics.insert_or_assign(name, DiagnosticVector{});
    m_diagnostics.at(name).push_back(std::move(diagnostic));
}
}
