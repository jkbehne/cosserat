#pragma once

#include <concepts>
#include <ranges>
#include <string>
#include <unordered_map>
#include <variant>

#include "physics/bodies.hpp"
#include "physics/constraints.hpp"
#include "physics/damping.hpp"
#include "physics/forces.hpp"
#include "physics/joints.hpp"

#include "simulation/diagnostics.hpp"

#include "utils/assertions.hpp"

namespace cosserat::simulation {

using utils::nice_assert;

class SimulationGraph
{
public: // Types
    using ForceVariant = physics::ForceTorqueVariant;
    using DampeningVariant = physics::DamperVariant;
    using ConstraintVariant = physics::ConstraintVariant;
    using JointVariant = physics::JointVariant;
    using BodyVariant = physics::BodyVariant;

    using ConstraintVector = std::vector<ConstraintVariant>;
    using ForceVector = std::vector<ForceVariant>;
    using DampeningVector = std::vector<DampeningVariant>;
    using DiagnosticVector = std::vector<DiagnosticVariant>;
    struct JointEdge
    {
        public: // Members
            std::string body1_name;
            std::string body2_name;
            JointVariant joint;

        public: // Methods
            JointEdge(
                const std::string& name1,
                const std::string& name2,
                JointVariant joint_
            ) : body1_name(name1),
                body2_name(name2),
                joint(std::move(joint_)) {}
    };
     using JointVector = std::vector<JointEdge>;
     using SubSystemType = BodyVariant;

private: // Members
    std::unordered_map<std::string, BodyVariant> m_bodies;
    std::unordered_map<std::string, ConstraintVector> m_constraints;
    std::unordered_map<std::string, ForceVector> m_forces;
    std::unordered_map<std::string, DampeningVector> m_dampers;
    std::unordered_map<std::string, DiagnosticVector> m_diagnostics;
    JointVector m_joints;

public: // Interface methods
    // Todo: Need to do something to run constraint at t=0 for initial conditions

    std::vector<SubSystemType>& final_systems();

    // It's assumed constraints only act on a single body, so it's safe to run constraints for
    // each body in the order in which they're stored in m_constraints, but it's worth noting
    // that constraints for given body will execute in the order in which they were declared.
    // It's on the user to ensure that order makes sense and constraints aren't overwriting
    // one another.
    void constrain_values(double time)
    {
        for (const auto& key : std::views::keys(m_constraints))
        {
            for (auto& constraint : m_constraints[key])
            {
                auto visitor = [&](auto& arg) {physics::constrain_values(constraint, arg, time);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key]);
            }
        }
    }

    void synchronize(double time);

    // It's assumed constraints only act on a single body, so it's safe to run constraints for
    // each body in the order in which they're stored in m_constraints, but it's worth noting
    // that constraints for given body will execute in the order in which they were declared.
    // It's on the user to ensure that order makes sense and constraints aren't overwriting
    // one another.
    void constrain_rates(double time)
    {
        for (const auto& key : std::views::keys(m_constraints))
        {
            for (auto& constraint : m_constraints[key])
            {
                auto visitor = [&](auto& arg) {physics::constrain_rates(constraint, arg, time);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key]);
            }
        }
    }

    void apply_callbacks(double time, std::uint64_t step)
    {
        // It's assumed callbacks should never depend on one another - so order shoudln't matter
        // Still, the order for each will match how the diagnostics were added
        for (const auto& key : std::views::keys(m_diagnostics))
        {
            for (auto& diagnostic : m_diagnostics[key])
            {
                auto visitor = [&](auto& arg) {make_callback(diagnostic, arg, time, step);};
                nice_assert(m_bodies.contains(key), "Couldn't find key " + key + " in body map");
                std::visit(visitor, m_bodies[key]);
            }
        }
    }

public: // Building methods
    void add_body(const std::string& name, BodyVariant body)
    {
        utils::nice_assert(
            not m_bodies.contains(name),
            "Name (" + name + ") already found in body map"
        );
        m_bodies[name] = std::move(body);
    }

    void add_constraint(const std::string& name, ConstraintVariant constraint)
    {
        utils::nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add constraint"
        );

        // Check that this particular constraint can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(constraint, arg);};
        std::visit(visitor, m_bodies[name]);

        if (not m_constraints.contains(name)) m_constraints[name];
        m_constraints[name].push_back(std::move(constraint));
    }

    void add_connection(
        const std::string& name1, const std::string& name2, JointVariant joint
    )
    {
        utils::nice_assert(
            m_bodies.contains(name1),
            "Name (" + name1 + ") not found in body map. Can't add connection"
        );
        utils::nice_assert(
            m_bodies.contains(name2),
            "Name (" + name2 + ") not found in body map. Can't add connection"
        );

        // Check that this particular connection can be connected to these two bodies
        const auto& variant1 = m_bodies[name1];
        auto visitor = [&](const auto& arg2)
        {
            auto nested_visitor = [&](const auto& arg1){physics::validate(joint, arg1, arg2);};
            std::visit(nested_visitor, variant1);
        };
        std::visit(visitor, m_bodies[name2]);

        m_joints.push_back(JointEdge(name1, name2, std::move(joint)));
    }

    void add_forcing_to(const std::string& name, ForceVariant force)
    {
        utils::nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add force"
        );

        // Check that this particular force can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(force, arg);};
        std::visit(visitor, m_bodies[name]);

        if (not m_forces.contains(name)) m_forces[name];
        m_forces[name].push_back(std::move(force));
    }

    void dampen(const std::string& name, DampeningVariant damper)
    {
        utils::nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add damping"
        );

        // Check that this particular damper can be applied to the particular body
        auto visitor = [&](const auto& arg) {physics::validate(damper, arg);};
        std::visit(visitor, m_bodies[name]);

        if (not m_dampers.contains(name)) m_dampers[name];
        m_dampers[name].push_back(std::move(damper));
    }

    void collect_diagnostics(const std::string& name, DiagnosticVariant diagnostic)
    {
        utils::nice_assert(
            m_bodies.contains(name),
            "Name (" + name + ") not found in body map. Can't add diagnostic"
        );

        // Check that this particular diagnostic can be applied to the particular body
        auto visitor = [&](const auto& arg) {validate(diagnostic, arg);};
        std::visit(visitor, m_bodies[name]);

        if (not m_diagnostics.contains(name)) m_diagnostics[name];
        m_diagnostics[name].push_back(std::move(diagnostic));
    }
};
} // End namespace cosserat::simulation
