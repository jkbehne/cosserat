#pragma once

/**
 * @file dynamics_kinematics.hpp
 * @brief The two half-steps a symplectic integrator applies to a system.
 *
 * A symplectic stepper alternates between advancing rates from accelerations
 * and advancing configuration from rates:
 *
 * @f[ \mathbf{v} \mathrel{+}= h\,\mathbf{a}, \qquad
 *     \pmb{\omega} \mathrel{+}= h\,\pmb{\alpha} @f]
 * @f[ \mathbf{x} \mathrel{+}= h\,\mathbf{v}, \qquad
 *     \mathbf{Q} \leftarrow \mathbf{R}(h, \pmb{\omega})\,\mathbf{Q} @f]
 *
 * The scale @f$ h @f$ is the stepper's prefactor for the current substage,
 * which is not necessarily the whole timestep.
 *
 * Both operations are free function templates rather than a variant, because
 * unlike a force or a damper there is only one way to integrate and so nothing
 * to dispatch over. Each is constrained to exactly the accessors it touches,
 * so a system that supports only one half still compiles against that half.
 *
 * This matches PyElastica's @c update_kinematics and @c update_dynamics, whose
 * implementations live in its rod data structures module.
 *
 * @note Frames are advanced by repeated multiplication and are never
 *       re-orthonormalised, so rounding accumulates. It does so slowly: at a
 *       timestep of @c 1e-4 a rod's frames remain orthogonal to roughly
 *       @c 2e-13 after a million steps, which still clears the @c 1e-12
 *       tolerance a rod applies to its own frames. A much longer run, a much
 *       larger timestep, or a tighter orthogonality tolerance would change
 *       that, and the remedy is to periodically re-orthonormalise each frame,
 *       for instance by replacing it with the @c Q factor of its QR
 *       decomposition.
 */

#include <cmath>
#include <concepts>

#include <Eigen/Core>

#include "math/linalg.hpp"
#include "math/types.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics::dynamics {

/**
 * @brief A system whose rates can be advanced from its accelerations.
 * @tparam T System type.
 */
template<typename T>
concept DynamicSystem = requires(T obj)
{
    {obj.mutable_velocities()} -> std::same_as<Vector3DStack&>;
    {obj.mutable_angular_velocities()} -> std::same_as<Vector3DStack&>;
    {obj.accelerations()} -> std::same_as<const Vector3DStack&>;
    {obj.angular_accelerations()} -> std::same_as<const Vector3DStack&>;
};

/**
 * @brief A system whose configuration can be advanced from its rates.
 * @tparam T System type.
 */
template<typename T>
concept KinematicSystem = requires(T obj)
{
    {obj.mutable_positions()} -> std::same_as<Vector3DStack&>;
    {obj.mutable_frames()} -> std::same_as<Matrix3DStack&>;
    {obj.velocities()} -> std::same_as<const Vector3DStack&>;
    {obj.angular_velocities()} -> std::same_as<const Vector3DStack&>;
};

/**
 * @brief Advances translational and angular rates by their accelerations.
 *
 * @f[ \mathbf{v} \mathrel{+}= h\,\mathbf{a}, \qquad
 *     \pmb{\omega} \mathrel{+}= h\,\pmb{\alpha} @f]
 *
 * @tparam SystemType Any @ref DynamicSystem.
 * @param system System to advance.
 * @param time Current simulation time; unused, present so that both half-steps
 *        share one signature.
 * @param scale Integration prefactor for this substage.
 *
 * @note Reads the accelerations a body last computed, so it belongs after that
 *       body's @c update_accelerations.
 */
template<DynamicSystem SystemType>
void update_dynamics(SystemType& system, double time, double scale)
{
    (void)time;

    auto& velocities = system.mutable_velocities();
    const auto& accelerations = system.accelerations();
    utils::nice_assert(
        velocities.rows() == accelerations.rows(),
        "velocities and accelerations incompatible sizes"
    );
    velocities += scale * accelerations;

    auto& angular_velocities = system.mutable_angular_velocities();
    const auto& angular_accelerations = system.angular_accelerations();
    utils::nice_assert(
        angular_velocities.rows() == angular_accelerations.rows(),
        "angular velocities and angular accelerations incompatible sizes"
    );
    angular_velocities += scale * angular_accelerations;
}

/**
 * @brief Advances positions and frames by the current rates.
 *
 * Positions translate along their velocities, and each frame is rotated by
 * @f$ \mathbf{R}(h, \pmb{\omega}) @f$ applied on the left, matching the
 * reference implementation's @c director = R @ director ordering.
 *
 * @f[ \mathbf{x} \mathrel{+}= h\,\mathbf{v}, \qquad
 *     \mathbf{Q}_i \leftarrow \mathbf{R}(h, \pmb{\omega}_i)\,\mathbf{Q}_i @f]
 *
 * An element with a negligible angular velocity is left strictly untouched
 * rather than multiplied by an identity, so a frame that is not turning cannot
 * accumulate rounding error at all. That also keeps the routine usable on the
 * perfectly ordinary states where the angular velocity is exactly zero: every
 * element of a freshly built body, and any element a boundary condition has
 * clamped. Without the guard those cases would trip
 * @ref cosserat::math::rotation_matrix, which rejects an axis too short to
 * define a direction.
 *
 * @tparam SystemType Any @ref KinematicSystem.
 * @param system System to advance.
 * @param time Current simulation time; unused, present so that both half-steps
 *        share one signature.
 * @param scale Integration prefactor for this substage.
 *
 * @note Reads the rates that @ref update_dynamics last wrote.
 */
template<KinematicSystem SystemType>
void update_kinematics(SystemType& system, double time, double scale)
{
    (void)time;

    auto& positions = system.mutable_positions();
    const auto& velocities = system.velocities();
    utils::nice_assert(
        positions.rows() == velocities.rows(), "positions and velocities incompatible sizes"
    );
    positions += scale * velocities;

    const auto& omegas = system.angular_velocities();
    const Eigen::Index num_rows = omegas.rows();

    auto& frames = system.mutable_frames();
    // Checked before any frame is touched, so that a size mismatch cannot
    // leave the system half-advanced.
    utils::nice_assert(
        static_cast<Eigen::Index>(frames.size()) == num_rows,
        "Frames and omegas incompatible sizes"
    );

    for (Eigen::Index idx = 0; idx < num_rows; ++idx)
    {
        const Eigen::Vector3d omega = omegas.row(idx).transpose();
        const double omega_norm = omega.norm();

        // Two ways an element can have nothing to do. First, an angular
        // velocity too small to define an axis, which is exactly what
        // rotation_matrix refuses. Second, a rotation angle -- the scale times
        // the angular speed -- small enough that the rotation is the identity
        // to within the same tolerance. Either way the frame is left alone.
        if (omega_norm <= math::rotation_tolerance) continue;
        if (std::abs(scale) * omega_norm < math::rotation_tolerance) continue;

        frames[idx] = math::rotation_matrix(scale, omega) * frames[idx];
    }
}
} // End namespace cosserat::physics::dynamics
