#pragma once

/**
 * @file simulation_graph.hpp
 * @brief The collection of bodies and rules that make up a simulation.
 *
 * A simulation is a set of named bodies plus the rules attached to them:
 * constraints, forces, dampers, joints, contacts and diagnostics. The graph
 * owns all of it, checks each rule against the body it is attached to as it is
 * registered, and then presents the whole thing to a time stepper through a
 * handful of phase methods.
 *
 * @section graph_phases The phases and the order they run in
 *
 * The stepper drives the graph through four phases per step. Their order is
 * not arbitrary and matches the reference implementation exactly:
 *
 * @verbatim
 *   half kinematic step        positions and frames advance by dt/2
 *   constrain_values(t)        boundary conditions pin configuration
 *   compute_internal_...(t)    each body's internal loads, per body
 *   synchronize(t)             joints, forces, then contacts accumulate
 *   dynamic step               accelerations, then rates advance by dt
 *   constrain_rates(t)         boundary conditions, then damping
 *   half kinematic step        positions and frames advance by dt/2
 *   constrain_values(t)        boundary conditions pin configuration again
 *   apply_callbacks(t, step)   diagnostics observe the finished step
 *   zero out external loads    accumulators cleared, per body
 * @endverbatim
 *
 * Three of those orderings carry real consequences:
 *
 * - **Internal forces are computed before @ref SimulationGraph::synchronize**,
 *   not inside it. Rod-to-rod contact reads a body's internal forces to work
 *   out how hard the two are already being pressed together, so contact would
 *   see stale values if the order were reversed.
 * - **Damping runs inside @ref SimulationGraph::constrain_rates**, after the
 *   rate boundary conditions rather than as a phase of its own. A damper
 *   scales rates that a constraint may just have pinned to zero, and pinning
 *   must win.
 * - **External accumulators are cleared at the very end, after the
 *   callbacks.** Diagnostics therefore see the loads that produced the step
 *   they are recording. Clearing at the top of a step instead would leave every
 *   recorded frame with empty force stacks.
 *
 * Within @ref SimulationGraph::synchronize the order is joints, then forces,
 * then contacts. Joints and forces only accumulate additively, so their
 * relative order does not matter; contact must come last, because it reads the
 * accumulated external forces. The reference registers these in one flat group
 * ordered by however the user declared them, which leaves the same requirement
 * satisfied by accident rather than by construction.
 *
 * @section graph_indices How a joint addresses a body
 *
 * A joint constrains position at a node and orientation at an element, and for
 * a rod those are different domains: a rod with @c n elements has @c n+1
 * nodes, and node @c k sits between elements @c k-1 and @c k. The reference
 * implementation uses one index for both, which happens to work only because
 * its endpoint conventions line up, and silently addresses the wrong element
 * for an interior connection.
 *
 * @ref SimulationGraph::add_connection therefore takes an element index and a
 * flag saying which end of that element the joint attaches to, and derives the
 * node from the pair. Element @c i spans nodes @c i and @c i+1, so the bottom
 * of element @c i is node @c i and its top is node @c i+1.
 *
 * @warning A rigid body has one element and one node, so only the bottom of
 *          element zero is addressable. Passing a top attachment for a rigid
 *          body asks for node 1, which does not exist.
 */

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <tsl/ordered_map.h>

#include <cosserat/physics/bodies.hpp>
#include <cosserat/physics/constraints.hpp>
#include <cosserat/physics/contacts.hpp>
#include <cosserat/physics/damping.hpp>
#include <cosserat/physics/forces.hpp>
#include <cosserat/physics/joints.hpp>

#include <cosserat/simulation/diagnostics.hpp>

namespace cosserat::simulation {

/**
 * @brief A joint together with the two bodies and indices it acts between.
 *
 * Stores both the element indices, which the torque half of a joint uses, and
 * the node indices derived from them, which the force half uses. Resolving
 * them once here keeps the per-step work to a lookup.
 */
struct JointEdge
{
public: // Members
    /** @brief Name of the first body. */
    std::string body1_name;

    /** @brief Name of the second body. */
    std::string body2_name;

    /** @brief Element of the first body whose orientation the joint acts on. */
    Eigen::Index element_one_idx;

    /** @brief Element of the second body whose orientation the joint acts on. */
    Eigen::Index element_two_idx;

    /** @brief Node of the first body the joint pulls on. */
    Eigen::Index node_one_idx;

    /** @brief Node of the second body the joint pulls on. */
    Eigen::Index node_two_idx;

    /** @brief The joint itself. */
    physics::JointVariant joint;

public: // Methods
    /**
     * @brief Builds an edge, deriving each node index from its element.
     *
     * Element @c i spans nodes @c i and @c i+1, so the bottom of an element is
     * the node with the same index and the top is the next one along.
     *
     * @param name1 Name of the first body.
     * @param name2 Name of the second body.
     * @param element_one_idx_ Element of the first body; already resolved, so
     *        it must not be negative.
     * @param element_two_idx_ Element of the second body; likewise resolved.
     * @param joint_bottom_one True if the joint attaches to the bottom of the
     *        first body's element rather than its top.
     * @param joint_bottom_two True if the joint attaches to the bottom of the
     *        second body's element.
     * @param joint_ The joint to apply.
     */
    JointEdge(
        const std::string& name1,
        const std::string& name2,
        Eigen::Index element_one_idx_,
        Eigen::Index element_two_idx_,
        bool joint_bottom_one,
        bool joint_bottom_two,
        physics::JointVariant joint_
    );
};

namespace detail {

/**
 * @brief Hashes a pair, for keying contacts on the two body names.
 *
 * Deliberately order-sensitive, since a contact rule distinguishes its first
 * system from its second. @ref SimulationGraph::add_contact separately rejects
 * a pair already registered the other way round, so the asymmetry never lets
 * one physical interaction be declared twice.
 */
struct PairHash
{
    /**
     * @brief Combines the two element hashes.
     * @tparam T1 First element type.
     * @tparam T2 Second element type.
     * @param p Pair to hash.
     * @return The combined hash.
     */
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

/**
 * @brief Owns the bodies and rules of a simulation and drives them per step.
 *
 * Built in two stages. First every body and rule is registered, each checked
 * against the body it attaches to as it arrives, so an incompatible pairing
 * fails where it was declared rather than midway through a run. Then
 * @ref finalize closes the graph to further additions and flattens the bodies
 * into the list the stepper iterates.
 *
 * The maps preserve insertion order, so a run is reproducible: rules fire in
 * the order they were declared rather than in whatever order a hash table
 * happens to yield.
 */
class SimulationGraph
{
public: // Types
    /** @brief A force or torque rule. */
    using ForceVariant = physics::ForceTorqueVariant;
    /** @brief A damping rule. */
    using DampeningVariant = physics::DamperVariant;
    /** @brief A boundary condition. */
    using ConstraintVariant = physics::ConstraintVariant;
    /** @brief A joint. */
    using JointVariant = physics::JointVariant;
    /** @brief A handle to a body of any kind. */
    using BodyVariant = physics::BodyVariantWrapper;
    /** @brief A contact rule. */
    using ContactVariant = physics::ContactVariant;

    /** @brief All boundary conditions attached to one body. */
    using ConstraintVector = std::vector<ConstraintVariant>;
    /** @brief All force rules attached to one body. */
    using ForceVector = std::vector<ForceVariant>;
    /** @brief All dampers attached to one body. */
    using DampeningVector = std::vector<DampeningVariant>;
    /** @brief All diagnostics attached to one body. */
    using DiagnosticVector = std::vector<DiagnosticVariant>;

    /**
     * @brief Contacts, keyed on the ordered pair of body names.
     *
     * A self contact is the pair of a name with itself. At most one contact
     * exists per pair of bodies; see @ref add_contact.
     */
    using ContactMap = tsl::ordered_map<std::pair<
        std::string, std::string>, ContactVariant, detail::PairHash
    >;
    /** @brief Every joint in the simulation. */
    using JointVector = std::vector<JointEdge>;

    /** @brief The type the stepper advances; required by the solver concept. */
    using SubSystemType = BodyVariant;

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

public: // Solver interface methods
    /**
     * @brief Closes the graph and prepares it for stepping.
     *
     * After this no further bodies or rules may be added, and
     * @ref final_systems becomes available. The flattened list holds handles
     * that share their bodies with the graph, so stepping through the list is
     * seen by every rule.
     *
     * @note Does not fire the diagnostics for the initial state. The reference
     *       implementation does, so its output includes a frame at step zero
     *       and this does not.
     */
    void finalize();

    /**
     * @brief The bodies, flattened for the stepper to iterate.
     * @return Handles to every body, in the order they were added.
     */
    std::vector<SubSystemType>& final_systems();

    /**
     * @brief Applies every boundary condition to configuration.
     *
     * Runs after each half kinematic step, so a pinned node is put back before
     * anything reads its position.
     *
     * @param time Current simulation time.
     */
    void constrain_values(double time);

    /**
     * @brief Accumulates every external load onto the bodies.
     *
     * Joints first, then force rules, then contacts. Contacts come last
     * because they read the external forces the earlier two have accumulated;
     * joints and forces only add, so their order between themselves is
     * immaterial.
     *
     * @param time Current simulation time.
     *
     * @note Must run after each body's @c compute_internal_forces_and_torques,
     *       since rod-to-rod contact reads internal forces.
     */
    void synchronize(double time);

    /**
     * @brief Applies every boundary condition to rates, then every damper.
     *
     * Constraints run first: a damper scales rates that a constraint may have
     * pinned, and pinning must win.
     *
     * @param time Current simulation time.
     */
    void constrain_rates(double time);

    /**
     * @brief Lets every diagnostic observe the finished step.
     *
     * Runs before the external accumulators are cleared, so a diagnostic sees
     * the loads that produced the step it is recording.
     *
     * @param time Current simulation time.
     * @param step Index of the step just completed.
     */
    void apply_callbacks(double time, std::uint64_t step);

public: // Building methods
    /**
     * @brief Adds a body under a name.
     * @param name Name to register it under; must not already be taken.
     * @param body Handle to the body.
     */
    void add_body(const std::string& name, BodyVariant body);

    /**
     * @brief Attaches a boundary condition to a body.
     *
     * The constraint is checked against the body immediately, so an
     * incompatible pairing fails here rather than on the first step.
     *
     * @param name Body to constrain; must already be registered.
     * @param constraint The boundary condition.
     */
    void add_constraint(const std::string& name, ConstraintVariant constraint);

    /**
     * @brief Joins two bodies with a joint.
     *
     * Each element index is resolved against its own body, so a negative index
     * counts back from that body's last element. The node each half of the
     * joint pulls on is derived from the element and the attachment flag; see
     * the file-level notes on indexing.
     *
     * @param name1 First body; must already be registered.
     * @param name2 Second body; must already be registered.
     * @param element_one_idx Element of the first body, negative to count back.
     * @param element_two_idx Element of the second body, negative to count back.
     * @param joint_bottom_one True if the joint attaches at the bottom of the
     *        first body's element rather than its top.
     * @param joint_bottom_two True if the joint attaches at the bottom of the
     *        second body's element.
     * @param joint The joint to apply.
     */
    void add_connection(
        const std::string& name1,
        const std::string& name2,
        Eigen::Index element_one_idx,
        Eigen::Index element_two_idx,
        bool joint_bottom_one,
        bool joint_bottom_two,
        JointVariant joint
    );

    /**
     * @brief Declares contact between two distinct bodies.
     *
     * Order matters to the rule itself: a rod-to-cylinder contact expects the
     * rod first. It does not matter to whether the pair is already taken,
     * though, so declaring both @c (a,b) and @c (b,a) is rejected. Those name
     * one physical interaction, and registering it twice would apply it twice
     * every step.
     *
     * @param name1 First body; must already be registered.
     * @param name2 Second body; must already be registered and must differ
     *        from @p name1.
     * @param contact The contact rule.
     */
    void add_contact(const std::string& name1, const std::string& name2, ContactVariant contact);

    /**
     * @brief Declares contact of a body with itself.
     *
     * @param name Body to check against itself; must already be registered.
     * @param contact The contact rule.
     */
    void add_contact(const std::string& name, ContactVariant contact);

    /**
     * @brief Attaches a force or torque rule to a body.
     * @param name Body to load; must already be registered.
     * @param force The rule.
     */
    void add_forcing_to(const std::string& name, ForceVariant force);

    /**
     * @brief Attaches a damper to a body.
     * @param name Body to damp; must already be registered.
     * @param damper The damper.
     */
    void dampen(const std::string& name, DampeningVariant damper);

    /**
     * @brief Attaches a diagnostic to a body.
     * @param name Body to observe; must already be registered.
     * @param diagnostic The diagnostic.
     */
    void collect_diagnostics(const std::string& name, DiagnosticVariant diagnostic);
};
} // End namespace cosserat::simulation
