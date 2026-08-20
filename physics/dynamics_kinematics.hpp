#pragma once

#include <concepts>

#include <Eigen/Core>

#include "math/linalg.hpp"
#include "math/types.hpp"

namespace cosserat::physics::dynamics {

template<typename T>
concept DynamicSystem = requires(T obj)
{
    {obj.mutable_velocities()} -> std::same_as<Vector3DStack&>;
    {obj.mutable_angular_velocities()} -> std::same_as<Vector3DStack&>;
    {obj.accelerations()} -> std::same_as<const Vector3DStack&>;
    {obj.angular_accelerations()} -> std::same_as<const Vector3DStack&>;
};

template<typename T>
concept KinematicSystem = requires(T obj)
{
    {obj.mutable_positions()} -> std::same_as<Vector3DStack&>;
    {obj.mutable_frames()} -> std::same_as<Matrix3DStack&>;
    {obj.velocities()} -> std::same_as<const Vector3DStack&>;
    {obj.angular_velocities()} -> std::same_as<const Vector3DStack&>;
};

template<DynamicSystem SystemType>
void update_dynamics(SystemType& system, double, double scale)
{
    auto& velocities = system.mutable_velocities();
    const auto& accelerations = system.accelerations();
    utils::nice_assert(
        velocities.rows() == accelerations.rows(),
        "velocities and accelerations incompatible sizes"
    );
    velocities += scale * accelerations;

    auto& angular_velocities = system.mutable_angular_velocities();
    const auto angular_accelerations = system.angular_accelerations();
    utils::nice_assert(
        angular_velocities.rows() == angular_accelerations.rows(),
        "angular velocities and angular accelerations incompatible sizes"
    );
    angular_velocities += scale * angular_accelerations;
}

template<KinematicSystem SystemType>
void update_kinematics(SystemType& system, double, double scale)
{
    auto& positions = system.mutable_positions();
    const auto& velocities = system.velocities();
    utils::nice_assert(
        positions.rows() == velocities.rows(), "positions and velocities incompatible sizes"
    );
    positions += scale * velocities;

    const auto& omegas = system.angular_velocities();
    const auto num_rows = omegas.rows();
    Matrix3DStack rotations;
    rotations.reserve(num_rows);
    for (Eigen::Index idx = 0; idx < num_rows; ++idx)
    {
        rotations.push_back(math::rotation_matrix(scale, omegas.row(idx).transpose()));
    }

    auto& frames = system.mutable_frames();
    utils::nice_assert(
        static_cast<Eigen::Index>(frames.size()) == num_rows,
        "Frames and omegas incompatible sizes"
    );
    for (Eigen::Index idx = 0; idx < num_rows; ++idx)
    {
        const auto& rotation = rotations[idx];
        frames[idx] = rotation * frames[idx];
    }
}
} // End namespace cosserat::physics::dynamics
