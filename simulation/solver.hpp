#pragma once

/**
 * @file solver.hpp
 * @brief The time stepper that advances a whole simulation.
 *
 * A solver owns nothing but a timestep and its place in the run. What it
 * integrates is supplied to it, and so is the decision about when to stop,
 * which keeps the stepper ignorant both of what is being simulated and of why
 * anyone wants it to end.
 *
 * @section solver_ordering The order of a step
 *
 * One step of the position Verlet scheme, in the order it happens:
 *
 * @verbatim
 *   half kinematic step        positions and frames advance by dt/2
 *   constrain_values(t)        boundary conditions pin configuration
 *   compute_internal_...(t)    each body's internal loads, per body
 *   synchronize(t)             joints, forces and contacts accumulate
 *   dynamic step               accelerations, then rates advance by dt
 *   constrain_rates(t)         boundary conditions, then damping
 *   half kinematic step        positions and frames advance by dt/2
 *   constrain_values(t)        boundary conditions pin configuration again
 *   apply_callbacks(t, step)   diagnostics observe the finished step
 *   zero out external loads    accumulators cleared, per body
 * @endverbatim
 *
 * Three of those orderings carry consequences. Internal forces are computed
 * before @c synchronize, because contact between two rods reads them. Damping
 * runs inside @c constrain_rates rather than as a phase of its own, so a
 * damper cannot scale away a rate a constraint has just pinned. And the
 * external accumulators are cleared last, after the callbacks, so a diagnostic
 * records the loads that produced the step it is writing.
 *
 * @section solver_once A solver runs once
 *
 * @ref Solver::full_solve may be called once on any given solver. A solver
 * carries the step count and the origin of its run, and reusing one would
 * either continue an old run under the guise of a new one or throw away the
 * state that made the returned times mean anything. A fresh solver for a fresh
 * run costs nothing and cannot be got wrong.
 */

#include <cmath>
#include <cstdint>
#include <utility>
#include <concepts>
#include <vector>

#include "utils/assertions.hpp"

namespace cosserat::simulation {

template<typename> class SolverTestPeer;

/**
 * @brief A body the stepper knows how to advance.
 * @tparam T Candidate body type.
 */
template<typename T>
concept IntegrableSystem = requires(T obj, double time, double scale)
{
    obj.update_kinematics(time, scale);
    obj.update_dynamics(time, scale);
    obj.update_accelerations(time, scale);
    obj.compute_internal_forces_and_torques(time);
    obj.zero_out_external_forces_and_torques(time);
};

/**
 * @brief A collection of bodies together with the rules coupling them.
 *
 * The stepper drives the four phase methods and leaves the collection to
 * decide what each of them means.
 *
 * @tparam T Candidate collection type.
 */
template<typename T>
concept SystemCollection = requires(T obj, double time, std::uint64_t step)
{
    typename T::SubSystemType;
    requires IntegrableSystem<typename T::SubSystemType>;
    {obj.final_systems()} -> std::same_as<std::vector<typename T::SubSystemType>&>;
    obj.constrain_values(time);
    obj.synchronize(time);
    obj.constrain_rates(time);
    obj.apply_callbacks(time, step);
};

/**
 * @brief Something that decides when a run should end.
 *
 * Asked before every step, and once before the first, so a criterion already
 * satisfied at the start ends the run without integrating anything. Taken by
 * reference, so a criterion accumulating state across a run, a settling test
 * being the obvious case, keeps that state.
 *
 * @tparam T Candidate criterion type.
 * @tparam SystemType The collection being run.
 *
 * @warning Nothing here bounds a run. A criterion that never returns true
 *          never returns at all, so one watching the state rather than the
 *          clock wants a time limit folded into it.
 */
template<typename T, typename SystemType>
concept StopCriterion = requires(T& criterion, SystemType& system, double time)
{
    {criterion(system, time)} -> std::same_as<bool>;
};

/**
 * @brief Advances a collection of bodies through time.
 * @tparam SystemType Any @ref SystemCollection.
 */
template<SystemCollection SystemType>
class Solver
{
    friend class SolverTestPeer<SystemType>;

private: // Types
    using SubSystemType = typename SystemType::SubSystemType;

private: // Members
    std::uint64_t m_current_step;
    double m_initial_time;
    double m_dt;
    bool m_has_run;

public: // Constructor
    /**
     * @brief Builds a solver with a fixed timestep.
     * @param dt Timestep; must be finite and greater than zero.
     */
    explicit Solver(double dt)
    : m_current_step(0), m_initial_time(0.0), m_dt(dt), m_has_run(false)
    {
        utils::nice_assert(
            std::isfinite(m_dt) and m_dt > 0.0, "dt must be finite and greater than 0"
        );
    }

public: // Methods
    /**
     * @brief Integrates until a criterion says to stop.
     *
     * Before any stepping, the configuration and the rates are constrained and
     * the callbacks fired once, so the state the run starts from is recorded
     * and is one the boundary conditions actually permit. The criterion is
     * then consulted before each step, including the first.
     *
     * @tparam Stopper Any @ref StopCriterion for this collection.
     * @param system The collection to advance.
     * @param stop_criterion Asked, before every step, whether to stop.
     * @param start Simulation time to begin from.
     * @return The simulation time reached.
     *
     * @throws Fails an assertion if this solver has already been run.
     *
     * @note The solver marks itself used before stepping rather than after, so
     *       a run that fails part way through cannot be resumed by a second
     *       call. Its step count would no longer describe the run being asked
     *       for.
     */
    template<StopCriterion<SystemType> Stopper>
    double full_solve(SystemType& system, Stopper&& stop_criterion, double start)
    {
        utils::nice_assert(
            not m_has_run, "Solver has already been run. Declare a new solver"
        );
        m_has_run = true;

        m_initial_time = start;
        double simulation_time = start;
        system.constrain_values(simulation_time);
        system.constrain_rates(simulation_time);
        system.apply_callbacks(simulation_time, m_current_step);

        while (not stop_criterion(system, simulation_time))
        {
            simulation_time = step(system, simulation_time);
        }

        return simulation_time;
    }

    /**
     * @brief Integrates from one time to another.
     *
     * A convenience over the criterion form, stopping on the clock alone.
     *
     * @param system The collection to advance.
     * @param start Simulation time to begin from.
     * @param end Simulation time to stop at; must be at least one step past
     *        @p start.
     * @return The simulation time reached. That is @p end exactly when the
     *         interval divides evenly by the timestep, and the first step
     *         beyond it otherwise.
     */
    double full_solve(SystemType& system, double start, double end)
    {
        utils::nice_assert(end - start >= m_dt, "Expected full sim time to be at least one dt");
        // Stopping at end rather than beyond it: asked at exactly end the
        // interval has been covered, and a further step would overshoot.
        auto time_lmbda = [end](SystemType&, double time) {return time >= end;};
        return full_solve(system, time_lmbda, start);
    }

private: // Methods
    /**
     * @brief Advances the whole collection by one timestep.
     *
     * See @ref solver_ordering for what happens, and why in that order.
     *
     * @param system The collection to advance.
     * @param time Simulation time at the start of the step.
     * @return The time after the step, recomputed from the origin and the step
     *         count rather than accumulated, so a long run does not drift.
     */
    double step(SystemType& system, double time)
    {
        double sim_time = time;
        for (auto& sub_system : system.final_systems())
        {
            kinematic_step(sub_system, time);
        }
        sim_time += 0.5 * m_dt;
        system.constrain_values(sim_time);

        for (auto& sub_system : system.final_systems())
        {
            sub_system.compute_internal_forces_and_torques(sim_time);
        }
        system.synchronize(sim_time);

        for (auto& sub_system : system.final_systems())
        {
            dynamics_step(sub_system, sim_time);
        }
        system.constrain_rates(sim_time);

        for (auto& sub_system : system.final_systems())
        {
            kinematic_step(sub_system, sim_time);
        }
        sim_time += 0.5 * m_dt;
        system.constrain_values(sim_time);
        ++m_current_step;
        system.apply_callbacks(sim_time, m_current_step);

        for (auto& sub_system : system.final_systems())
        {
            sub_system.zero_out_external_forces_and_torques(sim_time);
        }
        return m_initial_time + static_cast<double>(m_current_step) * m_dt;
    }

    /**
     * @brief Advances one body's configuration by half a timestep.
     * @param system The body to advance.
     * @param time Simulation time to hand the body.
     */
    void kinematic_step(SubSystemType& system, double time)
    {
        system.update_kinematics(time, 0.5 * m_dt);
    }

    /**
     * @brief Advances one body's rates by a whole timestep.
     *
     * Accelerations first, since the rates are advanced using them.
     *
     * @param system The body to advance.
     * @param time Simulation time to hand the body.
     */
    void dynamics_step(SubSystemType& system, double time)
    {
        system.update_accelerations(time, m_dt);
        system.update_dynamics(time, m_dt);
    }
};
} // End namespace cosserat::simulation
