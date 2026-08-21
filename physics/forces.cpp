#include "physics/forces.hpp"

#include <unsupported/Eigen/Splines>

namespace cosserat::physics {

EndpointForce::EndpointForce(
    Eigen::Vector3d first_link_force_,
    Eigen::Vector3d last_link_force_,
    double onset_time_,
    double ramp_time_,
    std::optional<double> offset_time_
) : m_first_link_force(first_link_force_),
    m_last_link_force(last_link_force_),
    m_linear_ramp(
        math::LinearRamp(onset_time_, ramp_time_, offset_time_)
    )
{
    utils::nice_assert(
        m_first_link_force.norm() > tolerance
        or m_last_link_force.norm() > tolerance,
        "Both first and last link forces are nearly zero"
    );
}

const Eigen::Vector3d& EndpointForce::first_link_force() const
{
    return m_first_link_force;
}
const Eigen::Vector3d& EndpointForce::last_link_force() const
{
    return m_last_link_force;
}
const math::LinearRamp& EndpointForce::linear_ramp() const
{
    return m_linear_ramp;
}

UniformForce::UniformForce(Eigen::Vector3d force_) : m_force(force_)
{
    utils::nice_assert(
        m_force.norm() > tolerance, "Force is nearly zero"
    );
}

const Eigen::Vector3d& UniformForce::force() const {return m_force;}

EndpointForceSinusoidal::EndpointForceSinusoidal(
    Eigen::Vector3d normal_dir,
    Eigen::Vector3d tangent_dir,
    double first_link_mag,
    double last_link_mag,
    double onset_time_
) : m_normal_direction(normal_dir),
    m_tangent_direction(tangent_dir),
    m_roll_direction(normal_dir.cross(tangent_dir)),
    m_first_link_magnitude(first_link_mag),
    m_last_link_magnitude(last_link_mag),
    m_onset_time(onset_time_)
{
    utils::nice_assert(
        math::is_unit_vector(m_normal_direction, tolerance),
        "Expected normal_direction to be unit vector"
    );
    utils::nice_assert(
        math::is_unit_vector(m_tangent_direction, tolerance),
        "Expected tangent_direction to be unit vector"
    );
    utils::nice_assert(
        math::is_unit_vector(m_roll_direction, tolerance),
        "Normal and tangent vectors aren't orthogonal"
    );
    utils::nice_assert(
        std::isfinite(m_first_link_magnitude) and m_first_link_magnitude >= 0.0,
        "Expected first_link_magnitude to be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_last_link_magnitude) and m_last_link_magnitude >= 0.0,
        "Expected last_link_magnitude to be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_onset_time), "Onset time must be finite"
    );
}

const Eigen::Vector3d& EndpointForceSinusoidal::normal_direction() const
{
    return m_normal_direction;
}

const Eigen::Vector3d& EndpointForceSinusoidal::tangent_direction() const
{
    return m_tangent_direction;
}

const Eigen::Vector3d& EndpointForceSinusoidal::roll_direction() const
{
    return m_roll_direction;
}

double EndpointForceSinusoidal::first_link_magnitude() const
{
    return m_first_link_magnitude;
}

double EndpointForceSinusoidal::last_link_magnitude() const
{
    return m_last_link_magnitude;
}

double EndpointForceSinusoidal::onset_time() const
{
    return m_onset_time;
}

UniformTorque::UniformTorque(Eigen::Vector3d torque_) : m_torque(torque_)
{
    utils::nice_assert(
        m_torque.norm() > tolerance, "Norm of torque is nearly zero"
    );
}

const Eigen::Vector3d& UniformTorque::torque() const
{
    return m_torque;
}

MuscleTorque::MuscleTorque(
    Eigen::Vector3d direction_,
    double angular_freq,
    double wave_number_,
    double phase_shift_,
    double onset_time,
    double ramp_time,
    std::optional<double> offset_time,
    const Eigen::VectorXd& rest_lengths,
    const Eigen::VectorXd& b_coeffs
) : m_direction(direction_),
    m_angular_frequency(angular_freq),
    m_wave_number(wave_number_),
    m_phase_shift(phase_shift_),
    m_linear_ramp(
        math::LinearRamp(onset_time, ramp_time, offset_time)
    ),
    m_normalized_coords(math::to_normalized_coords(rest_lengths)),
    m_spline(determine_spline(b_coeffs, m_normalized_coords))
{
    utils::nice_assert(
        math::is_unit_vector(m_direction, tolerance),
        "Expected direction to be a unit vector"
    );
    utils::nice_assert(
        std::isfinite(m_angular_frequency) and m_angular_frequency > 0.0,
        "angular_frequency must be finite and greater than 0"
    );
    utils::nice_assert(
        std::isfinite(m_wave_number) and m_wave_number > 0.0,
        "wave_number must be finite and greater than zero"
    );
    utils::nice_assert(
        std::isfinite(m_phase_shift), "phase_shift must be finite"
    );
}

const Eigen::Vector3d& MuscleTorque::direction() const {return m_direction;}
double MuscleTorque::angular_frequency() const {return m_angular_frequency;}
double MuscleTorque::wave_number() const {return m_wave_number;}
double MuscleTorque::phase_shift() const {return m_phase_shift;}
const math::LinearRamp& MuscleTorque::linear_ramp() const {return m_linear_ramp;}
const Eigen::VectorXd& MuscleTorque::normalized_coords() const {return m_normalized_coords;}
const Eigen::VectorXd& MuscleTorque::spline() const {return m_spline;}

// Static methods
Eigen::VectorXd MuscleTorque::determine_spline(
    const Eigen::VectorXd& b_coeffs, const Eigen::VectorXd& s_coords
)
{
    if (b_coeffs.size() == 0) return Eigen::VectorXd::Zero(s_coords.size());

    utils::nice_assert(
        b_coeffs.size() >= 4, "b_coeffs should have at least 4 entries"
    );
    const Eigen::Index m = b_coeffs.size();
    Eigen::VectorXd knots(m + spline_degree + 1);
    knots.head(spline_degree + 1).setConstant(0.0);
    knots.tail(spline_degree + 1).setConstant(1.0);
    for (Eigen::Index i = 0; i < m - Eigen::Index(spline_degree) - 1; ++i)
    {
        knots(spline_degree + 1 + i) = double(i + 1) / double(m - spline_degree);
    }
    Eigen::Spline<double, 1, spline_degree> spline(knots, b_coeffs.transpose());

    Eigen::VectorXd result(s_coords.size());
    for (Eigen::Index idx = 0; idx < s_coords.size(); ++idx)
    {
        result(idx) = spline(s_coords(idx))(0, 0);
    }
    return result;
}
} // End namespace cosserat::physics
