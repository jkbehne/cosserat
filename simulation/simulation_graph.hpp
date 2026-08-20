#pragma once

#include <concepts>
#include <ranges>
#include <string>
#include <variant>

#include <Eigen/Core>
#include <spdlog/spdlog.h>
#include <tsl/ordered_map.h>

#include "math/indexing.hpp"

#include "physics/bodies.hpp"
#include "physics/constraints.hpp"
#include "physics/contacts.hpp"
#include "physics/damping.hpp"
#include "physics/forces.hpp"
#include "physics/joints.hpp"

#include "simulation/diagnostics.hpp"

#include "utils/assertions.hpp"

namespace cosserat::simulation {

using utils::nice_assert;

struct JointEdge
{
    public: // Members
        std::string body1_name;
        std::string body2_name;
        Eigen::Index element_one_idx;
        Eigen::Index element_two_idx;
        Eigen::Index node_one_idx;
        Eigen::Index node_two_idx;
        physics::JointVariant joint;

    public: // Methods
        JointEdge(
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
};
using JointVector = std::vector<JointEdge>;

struct ContactEdge
{
    public: // Members
        std::string body1_name;
        std::string body2_name;
        physics::ContactVariant contact;

    public: // Methods
        ContactEdge(
            const std::string& name1,
            const std::string& name2,
            physics::ContactVariant contact_
        ) : contact(std::move(contact_)) {}
};

namespace detail {
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        
        // A simple, reliable hash combination algorithm (e.g., Boost's hash_combine formula)
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};
} // End namespace detail

class SimulationGraph
{
public: // Types
    using ForceVariant = physics::ForceTorqueVariant;
    using DampeningVariant = physics::DamperVariant;
    using ConstraintVariant = physics::ConstraintVariant;
    using JointVariant = physics::JointVariant;
    using BodyVariant = physics::BodyVariantWrapper;

    using ConstraintVector = std::vector<ConstraintVariant>;
    using ForceVector = std::vector<ForceVariant>;
    using DampeningVector = std::vector<DampeningVariant>;
    using DiagnosticVector = std::vector<DiagnosticVariant>;
    using SubSystemType = BodyVariant;
    using ContactMap = tsl::ordered_map<
        std::pair<std::string, std::string>, physics::ContactVariant, detail::PairHash
    >;

private: // Members
    tsl::ordered_map<std::string, BodyVariant> m_bodies;
    tsl::ordered_map<std::string, ConstraintVector> m_constraints;
    tsl::ordered_map<std::string, ForceVector> m_forces;
    tsl::ordered_map<std::string, DampeningVector> m_dampers;
    tsl::ordered_map<std::string, DiagnosticVector> m_diagnostics;
    JointVector m_joints;
    ContactMap m_contacts;

    std::vector<SubSystemType> m_final_systems;

    bool finalized = false;

public: // Interface methods
    // Todo: Need to do something to run constraint at t=0 for initial conditions

    std::vector<SubSystemType>& final_systems()
    {
        nice_assert(finalized, "finalized must be true before final_systems() can be called");
        if (m_contacts.empty()) spdlog::warn("No contacts declared. All bodies are not solid");
        return m_final_systems;
    }

    void finalize()
    {
        nice_assert(not finalized, "System has already been finalized");
        nice_assert(not m_bodies.empty(), "Can't finalize a system with no bodies");

        for (const auto& value : m_bodies | std::views::values)
        {
            m_final_systems.push_back(value);
        }
        finalized = true;
        spdlog::info("Finalized simulation");
    }

    void constrain_values(double time)
    {
        spdlog::debug("Constraining values at simulation time {}", time);
        for (const auto& key : std::views::keys(m_constraints))
        {
            for (auto& constraint : m_constraints[key])
            {
                auto visitor = [&](auto& arg) {physics::constrain_values(constraint, arg, time);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key].body());
            }
        }
    }

    void synchronize(double time)
    {
        spdlog::debug("Snychronizing bodies at simulation time {}", time);

        for (auto& joint : m_joints)
        {
            const auto& name1 = joint.body1_name;
            const auto& name2 = joint.body2_name;
            nice_assert(m_bodies.contains(name1), "Body name " + name1 + " not found");
            nice_assert(m_bodies.contains(name2), "Body name " + name2 + " not found");
            auto& variant1 = m_bodies[name1].body();
            auto& variant2 = m_bodies[name2].body();
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
                        }, variant1
                    );
                }, variant2
            );
        }
    }

    void constrain_rates(double time)
    {
        spdlog::debug("Constraing rates at simulation time {}", time);
        for (const auto& key : std::views::keys(m_constraints))
        {
            for (auto& constraint : m_constraints[key])
            {
                auto visitor = [&](auto& arg) {physics::constrain_rates(constraint, arg, time);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key].body());
            }
        }
    }

    // It's assumed callbacks should never depend on one another - so order shoudln't matter
    // Still, the order for each will match how the diagnostics were added
    void apply_callbacks(double time, std::uint64_t step)
    {
        spdlog::debug("Applying call backs for step {} at simulation time {}", step, time);
        for (const auto& key : std::views::keys(m_diagnostics))
        {
            for (auto& diagnostic : m_diagnostics[key])
            {
                auto visitor = [&](auto& arg) {make_callback(diagnostic, arg, time, step);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key].body());
            }
        }
    }

public: // Building methods
    void add_body(const std::string& name, BodyVariant body)
    {
        nice_assert(not finalized, "Can't add body to finalized simulation");
        nice_assert(not m_bodies.contains(name), "Name (" + name + ") already found in body map");
        spdlog::info("Adding body with name {} to simulation", name);
        auto variant_ptr = std::make_shared<physics::BodyVariant>(std::move(body));
        m_bodies[name] = BodyVariant(std::move(variant_ptr));
    }

    void add_constraint(const std::string& name, ConstraintVariant constraint)
    {
        nice_assert(not finalized, "Can't add constraint to finalized simulation");
        nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add constraint"
        );

        // Check that this particular constraint can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(constraint, arg);};
        std::visit(visitor, m_bodies[name].body());
        spdlog::info("Add constraint to body {}", name);

        if (not m_constraints.contains(name)) m_constraints[name] = {std::move(constraint)};
        else m_constraints[name].push_back(std::move(constraint));
    }

    void add_connection(
        const std::string& name1,
        const std::string& name2,
        Eigen::Index element_one_idx,
        Eigen::Index element_two_idx,
        bool joint_bottom_one,
        bool joint_bottom_two,
        JointVariant joint
    )
    {
        nice_assert(not finalized, "Can't add connection to finalized simulation");
        nice_assert(
            m_bodies.contains(name1),
            "Name (" + name1 + ") not found in body map. Can't add connection"
        );
        nice_assert(
            m_bodies.contains(name2),
            "Name (" + name2 + ") not found in body map. Can't add connection"
        );

        // Check that this particular connection can be connected to these two bodies
        const auto& variant1 = m_bodies[name1].body();
        auto visitor = [&](const auto& arg2)
        {
            auto nested_visitor = [&](const auto& arg1){physics::validate(joint, arg1, arg2);};
            std::visit(nested_visitor, variant1);
        };
        std::visit(visitor, m_bodies[name2].body());

        // Resolve index for each body
        Eigen::Index index_one_resolved;
        std::visit(
            [&](const auto& body)
            {
                const auto num_elements = body.num_elements();
                index_one_resolved = resolve_index(element_one_idx, num_elements);
            }, m_bodies[name1].body()
        );
        Eigen::Index index_two_resolved;
            std::visit(
            [&](const auto& body)
            {
                const auto num_elements = body.num_elements();
                index_two_resolved = resolve_index(element_one_idx, num_elements);
            }, m_bodies[name2].body()
        );

        spdlog::info("Connecting body {} to {}", name1, name2);
        m_joints.push_back(
            JointEdge(
                name1,
                name2,
                index_one_resolved,
                index_two_resolved,
                joint_bottom_one,
                joint_bottom_two,
                std::move(joint)
            )
        );
    }

    void add_forcing_to(const std::string& name, ForceVariant force)
    {
        nice_assert(not finalized, "Can't add forcing to finalized simulation");
        nice_assert(
            m_bodies.contains(name), "Name (" + name + ") not found in body map. Can't add force"
        );

        // Check that this particular force can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(force, arg);};
        std::visit(visitor, m_bodies[name].body());
        spdlog::info("Adding forcing to body {}", name);

        if (not m_forces.contains(name)) m_forces[name] = {std::move(force)};
        else m_forces[name].push_back(std::move(force));
    }

    void dampen(const std::string& name, DampeningVariant damper)
    {
        nice_assert(not finalized, "Can't add damping to finalized simulation");
        nice_assert(
            m_bodies.contains(name), "Name (" + name + ") not found in body map. Can't add damping"
        );

        // Check that this particular damper can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(damper, arg);};
        std::visit(visitor, m_bodies[name].body());
        spdlog::info("Adding damper to body {}", name);

        if (not m_dampers.contains(name)) m_dampers[name] = {std::move(damper)};
        else m_dampers[name].push_back(std::move(damper));
    }

    void collect_diagnostics(const std::string& name, DiagnosticVariant diagnostic)
    {
        nice_assert(not finalized, "Can't add diagnostic to finalized simulation");
        nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add diagnostic"
        );

        // Check that this particular diagnostic can be applied to the particular body
        auto visitor = [&](const auto& arg) {validate(diagnostic, arg);};
        std::visit(visitor, m_bodies[name].body());
        spdlog::info("Adding diagnostic to body {}", name);

        if (not m_diagnostics.contains(name)) m_diagnostics[name] = {std::move(diagnostic)};
        else m_diagnostics[name].push_back(std::move(diagnostic));
    }
};
} // End namespace cosserat::simulation
