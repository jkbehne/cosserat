#pragma once

#include <cmath>
#include <cstdint>
#include <concepts>
#include <vector>

#include "utils/assertions.hpp"

namespace cosserat::simulation {

template<typename> class SolverTestPeer;

template<typename T>
concept IntegrableSystem = requires(T obj, double time, double scale)
{
    obj.update_kinematics(time, scale);
    obj.update_dynamics(time, scale);
    obj.update_accelerations(time, scale);
    obj.compute_internal_forces_and_torques(time);
    obj.zero_out_external_forces_and_torques(time);
};

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

template<SystemCollection SystemType>
class Solver
{
    friend class SolverTestPeer<SystemType>;

private: // Types
    using SubSystemType = typename SystemType::SubSystemType;

private: // Members
    std::uint64_t current_step;
    double initial_time;
    double dt;

public: // Constructor
    explicit Solver(double dt_) : current_step(0), initial_time(0.0), dt(dt_)
    {
        utils::nice_assert(
            std::isfinite(dt) and dt > 0.0,
            "dt must be finite and greater than 0"
        );
    }

public: // Methods
    double full_solve(SystemType& system, double start, double end)
    {
        reset();
        initial_time = start;
        const auto full_dt = end - start;
        utils::nice_assert(
            full_dt >= dt, "Expected full sim time to be at least one dt"
        );
        const std::uint64_t num_steps = static_cast<std::uint64_t>(
            std::round(full_dt / dt)
        );
        double time = start;
        system.constrain_values(time);
        system.constrain_rates(time);
        system.apply_callbacks(time, current_step);
        for (std::uint64_t idx = 0; idx < num_steps; ++idx)
        {
            time = step(system, time);
        }
        return time;
    }

    void reset()
    {
        current_step = 0;
        initial_time = 0.0;
    }

    double step(SystemType& system, double time)
    {
        if (current_step == 0) initial_time = time;
        double sim_time = time;
        for (auto& sub_system : system.final_systems())
        {
            kinematic_step(sub_system, time);
        }
        sim_time += 0.5 * dt;
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
        sim_time += 0.5 * dt;
        system.constrain_values(sim_time);
        ++current_step;
        system.apply_callbacks(sim_time, current_step);

        for (auto& sub_system : system.final_systems())
        {
            sub_system.zero_out_external_forces_and_torques(sim_time);
        }
        return initial_time + static_cast<double>(current_step) * dt;
    }

private: // Methods
    void kinematic_step(SubSystemType& system, double time)
    {
        system.update_kinematics(time, 0.5 * dt);
    }

    void dynamics_step(SubSystemType& system, double time)
    {
        system.update_accelerations(time, dt);
        system.update_dynamics(time, dt);
    }
};
} // End namespace math::cosserat
