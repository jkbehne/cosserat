#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "math/linalg.hpp"
#include "math/functions.hpp"
#include "math/types.hpp"

#include "physics/coords.hpp"

#include "utils/assertions.hpp"
#include "utils/basic_concepts.hpp"

namespace cosserat::physics {

// Some useful concepts
template<typename T>
concept ForceableSystem = requires(T sys)
{
    {sys.num_elements()} -> std::convertible_to<std::uint64_t>;
    {sys.mass()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.external_forces()} -> std::same_as<Vector3DStack&>;
};

template<typename T>
concept TorqueableSystem = requires(T sys)
{
    {sys.num_elements()} -> std::convertible_to<std::uint64_t>;
    {sys.frames()} -> std::same_as<const Matrix3DStack&>;
    {sys.external_torques()} -> std::same_as<Vector3DStack&>;
};

template<typename T>
concept ForceableTorqueableSystem = ForceableSystem<T> and TorqueableSystem<T>;

template<IsXYZ GravityCoord>
struct GravityForce
{
public: // Static constexpr members
    static constexpr double gravity = -9.80665;

public: // Methods
    template<ForceableSystem SystemType>
    void apply_forces(SystemType& system, double) const
    {
        const Eigen::VectorXd& mass = system.mass();
        system.external_forces().col(GravityCoord::idx) += gravity * mass;
    }

    template<typename T>
    void apply_torques(T&&, double) const {}
};

struct EndpointForce
{
public: // Static constexpr members
    static constexpr double tolerance = 1e-10;

private: // Members
    Eigen::Vector3d m_first_link_force;
    Eigen::Vector3d m_last_link_force;
    math::LinearRamp m_linear_ramp;

public: // Methods
    EndpointForce(
        Eigen::Vector3d first_link_force_,
        Eigen::Vector3d last_link_force_,
        double onset_time_,
        double ramp_time_,
        std::optional<double> offset_time_
    );

    template<ForceableSystem SystemType>
    void apply_forces(SystemType& system, double time) const
    {
        const auto factor = m_linear_ramp(time);
        const auto num_entries = system.external_forces().rows();
        utils::nice_assert(num_entries > 0, "Need at least one force entry");
        system.external_forces().row(0) += factor * m_first_link_force.transpose();
        system.external_forces().row(num_entries - 1) += factor * m_last_link_force.transpose();
    }

    template<typename T>
    void apply_torques(T&&, double) const {}

    const Eigen::Vector3d& first_link_force() const;
    const Eigen::Vector3d& last_link_force() const;
    const math::LinearRamp& linear_ramp() const;
};

struct UniformForce
{
public: // Static constexpr members
    static constexpr double tolerance = 1e-10;

private: // Members
    Eigen::Vector3d m_force;

public: // Methods
    explicit UniformForce(Eigen::Vector3d force_);

    template<ForceableSystem SystemType>
    void apply_forces(SystemType& system, double) const
    {
        const auto num_elements = system.num_elements();
        utils::nice_assert(num_elements > 0, "Must have at least one element");
        const auto scale = 1.0 / static_cast<double>(num_elements);
        const Eigen::Vector3d scaled_force = scale * m_force;
        system.external_forces().rowwise() += scaled_force.transpose();

        // Because mass is half on first and last element
        system.external_forces().row(0) -= 0.5 * scaled_force.transpose();
        const auto last_idx = system.external_forces().rows() - 1;
        system.external_forces().row(last_idx) -= 0.5 * scaled_force.transpose();
    }

    template<typename T>
    void apply_torques(T&&, double) const {}

    const Eigen::Vector3d& force() const;
};

struct EndpointForceSinusoidal
{
public: // Static constexpr members
    static constexpr double tolerance = 1e-10;

private: // Members
    Eigen::Vector3d m_normal_direction;
    Eigen::Vector3d m_tangent_direction;
    Eigen::Vector3d m_roll_direction;
    double m_first_link_magnitude;
    double m_last_link_magnitude;
    double m_onset_time;

public: // Methods
    EndpointForceSinusoidal(
        Eigen::Vector3d normal_dir,
        Eigen::Vector3d tangent_dir,
        double first_link_mag,
        double last_link_mag,
        double onset_time_
    );

    template<ForceableSystem SystemType>
    void apply_forces(SystemType& system, double time) const
    {
        const auto num_elements = system.external_forces().rows();
        utils::nice_assert(num_elements > 0, "Expected at least one force element");
        if (time < m_onset_time)
        {
            // When time smaller than onset time apply the force in normal
            // direction.
            const Eigen::Vector3d start_force = -2.0 * m_first_link_magnitude * m_normal_direction;
            const Eigen::Vector3d end_force = -2.0 * m_last_link_magnitude * m_normal_direction;
            system.external_forces().row(0) += start_force.transpose();
            system.external_forces().row(num_elements - 1) += end_force.transpose();
        }
        else
        {
            // When time is greater than onset time, forces are applied in normal
            // and roll direction.
            const auto cos_arg = 0.5 * std::numbers::pi * (time - m_onset_time);
            const auto fl_mag = m_first_link_magnitude;
            const auto fl_roll_force = fl_mag * std::cos(cos_arg) * m_roll_direction;
            const auto fl_norm_force = fl_mag * std::sin(cos_arg) * m_normal_direction;
            const Eigen::Vector3d fl_force = fl_roll_force + fl_norm_force;

            const auto ll_mag = m_last_link_magnitude;
            const auto ll_roll_force = ll_mag * std::cos(cos_arg) * m_roll_direction;
            const auto ll_norm_force = ll_mag * std::sin(cos_arg) * m_normal_direction;
            const Eigen::Vector3d ll_force = ll_roll_force + ll_norm_force;

            system.external_forces().row(0) += fl_force.transpose();
            system.external_forces().row(num_elements - 1) += ll_force.transpose();
        }
    }

    template<typename T>
    void apply_torques(T&&, double) const {}

    const Eigen::Vector3d& normal_direction() const;
    const Eigen::Vector3d& tangent_direction() const;
    const Eigen::Vector3d& roll_direction() const;
    double first_link_magnitude() const;
    double last_link_magnitude() const;
    double onset_time() const;
};

struct UniformTorque
{
public: // Static constexpr members
    static constexpr double tolerance = 1e-10;

private: // Members
    Eigen::Vector3d m_torque;

public: // Methods
    explicit UniformTorque(Eigen::Vector3d torque_);

    template<typename T>
    void apply_forces(T&&, double) const {}

    template<TorqueableSystem SystemType>
    void apply_torques(SystemType& system, double) const
    {
        const auto num_elements = system.num_elements();
        utils::nice_assert(num_elements > 0, "Must have at least one element");
        const double scale = 1.0 / static_cast<double>(num_elements);
        const Eigen::VectorXd ones = Eigen::VectorXd::Ones(num_elements);
        const auto repeated = scale * (m_torque * ones.transpose());
        system.external_torques() += math::batched_matrix_vector<3>(
            system.frames(),
            repeated
        );
    }

    const Eigen::Vector3d& torque() const;
};

struct MuscleTorque
{
public: // Static constexpr members
    static constexpr unsigned int spline_degree = 3;
    static constexpr double tolerance = 1e-10;

private: // Members
    // Order matters due to initializer list
    Eigen::Vector3d m_direction;
    double m_angular_frequency;
    double m_wave_number;
    double m_phase_shift;
    math::LinearRamp m_linear_ramp;
    Eigen::VectorXd m_normalized_coords;
    Eigen::VectorXd m_spline;

public: // Methods
    MuscleTorque(
        Eigen::Vector3d direction_,
        double angular_freq,
        double wave_number_,
        double phase_shift_,
        double onset_time,
        double ramp_time,
        std::optional<double> offset_time,
        const Eigen::VectorXd& rest_lengths,
        const Eigen::VectorXd& b_coeffs
    );

    template<typename T>
    void apply_forces(T&&, double) const {}

    template<TorqueableSystem SystemType>
    void apply_torques(SystemType& system, double time) const
    {
        const auto factor = m_linear_ramp(time);
        const auto sine_phase = m_angular_frequency * time + m_phase_shift;
        const auto sine_arg = sine_phase - (m_wave_number * m_normalized_coords).array();
        const Eigen::VectorXd torque_mag = factor * m_spline.array() * sine_arg.array().sin();

        const auto torque = m_direction * torque_mag.reverse().transpose();
        const Vector3DStack torque_product = math::batched_matrix_vector<3>(
            system.frames(), torque
        );
        const auto num_torques = torque_product.rows();
        system.external_torques().bottomRows(num_torques - 1) +=
            torque_product.bottomRows(num_torques - 1);
        const Vector3DStack torque_mismatch = math::batched_matrix_vector<3>(
            system.frames(),
            torque.rightCols(num_torques - 1),
            true /* ignore size mismatch */
        );
        system.external_torques().topRows(num_torques - 1)
            -= torque_mismatch.topRows(num_torques - 1);
    }

    const Eigen::Vector3d& direction() const;
    double angular_frequency() const;
    double wave_number() const;
    double phase_shift() const;
    const math::LinearRamp& linear_ramp() const;
    const Eigen::VectorXd& normalized_coords() const;
    const Eigen::VectorXd& spline() const;

public: // Static methods
    static Eigen::VectorXd determine_spline(
        const Eigen::VectorXd& b_coeffs, const Eigen::VectorXd& s_coords
    );
};

using GravityForceX = GravityForce<XTag>;
using GravityForceY = GravityForce<YTag>;
using GravityForceZ = GravityForce<ZTag>;
using ForceTorqueVariant = std::variant<
    GravityForceX,
    GravityForceY,
    GravityForceZ,
    EndpointForce,
    UniformForce,
    EndpointForceSinusoidal,
    UniformTorque,
    MuscleTorque
>;

template<typename BodyType>
void validate(ForceTorqueVariant& force_var, BodyType& system)
{
    std::visit([&](auto& force)
    {
        const double time = 0.0;
        if constexpr (
            not requires {
                force.apply_forces(system, time);
                force.apply_torques(system, time);
            }
        ) utils::nice_assert(false, "Force is incompatible with this system");
    }, force_var);
}

template<typename BodyType>
void apply_forces(ForceTorqueVariant& force_var, BodyType& system, double time)
{
    std::visit([&](auto& force)
    {
        if constexpr (requires {force.apply_forces(system, time);})
        {
            force.apply_forces(system, time);
        }
        else utils::nice_assert(false, "Force is incompatible with this system");
    }, force_var);
}

template<typename BodyType>
void apply_torques(ForceTorqueVariant& force_var, BodyType& system, double time)
{
    std::visit([&](auto& force)
    {
        if constexpr (requires {force.apply_torques(system, time);})
        {
            force.apply_torques(system, time);
        }
        else utils::nice_assert(false, "Force is incompatible with this system");
    }
    , force_var);
}
} // End namespace cosserat::physics
