/**
 * @file joints.cpp
 * @brief Non-template implementations for the joints.
 */

#include "physics/joints.hpp"

#include <cmath>

namespace cosserat::physics {

namespace {

/** @brief Fails unless a coefficient is finite and non-negative. */
void assert_coefficient(double value, const char* message)
{
    utils::nice_assert(std::isfinite(value) and value >= 0.0, message);
}

} // End anonymous namespace

// ---------------------------------------------------------------------------
// FreeJoint
// ---------------------------------------------------------------------------

FreeJoint::FreeJoint(double stiffness, double damping)
    : m_stiffness(stiffness), m_damping(damping)
{
    assert_coefficient(m_stiffness, "stiffness must be finite and non-negative");
    assert_coefficient(m_damping, "damping must be finite and non-negative");
}

double FreeJoint::stiffness() const {return m_stiffness;}
double FreeJoint::damping() const {return m_damping;}

// ---------------------------------------------------------------------------
// HingeJoint
// ---------------------------------------------------------------------------

HingeJoint::HingeJoint(
    double stiffness,
    double damping,
    double rotational_stiffness,
    Eigen::Vector3d normal_direction
) : m_free_joint(stiffness, damping),
    m_rotational_stiffness(rotational_stiffness),
    m_normal_direction(std::move(normal_direction))
{
    assert_coefficient(
        m_rotational_stiffness,
        "rotational_stiffness must be finite and non-negative"
    );
    utils::nice_assert(
        m_normal_direction.array().isFinite().all(),
        "normal_direction must contain only finite values"
    );

    // Normalised rather than required to be unit, matching the reference.
    const double length = m_normal_direction.norm();
    utils::nice_assert(
        length > joint_direction_tolerance,
        "normal_direction is too short to define a plane normal"
    );
    m_normal_direction /= length;
}

double HingeJoint::stiffness() const {return m_free_joint.stiffness();}
double HingeJoint::damping() const {return m_free_joint.damping();}
double HingeJoint::rotational_stiffness() const {return m_rotational_stiffness;}
const Eigen::Vector3d& HingeJoint::normal_direction() const
{
    return m_normal_direction;
}

// ---------------------------------------------------------------------------
// FixedJoint
// ---------------------------------------------------------------------------

FixedJoint::FixedJoint(
    double stiffness,
    double damping,
    double rotational_stiffness,
    double rotational_damping,
    Eigen::Matrix3d rest_rotation_matrix
) : m_free_joint(stiffness, damping),
    m_rotational_stiffness(rotational_stiffness),
    m_rotational_damping(rotational_damping),
    m_rest_rotation_matrix(std::move(rest_rotation_matrix))
{
    assert_coefficient(
        m_rotational_stiffness,
        "rotational_stiffness must be finite and non-negative"
    );
    assert_coefficient(
        m_rotational_damping,
        "rotational_damping must be finite and non-negative"
    );
    utils::nice_assert(
        m_rest_rotation_matrix.array().isFinite().all(),
        "rest_rotation_matrix must contain only finite values"
    );

    // Stricter than the reference, which only checks the shape. A rest
    // rotation that is not a rotation yields torques that look plausible but
    // are meaningless, so it's worth rejecting
    const Eigen::Matrix3d residual =
        m_rest_rotation_matrix * m_rest_rotation_matrix.transpose()
        - Eigen::Matrix3d::Identity();
    utils::nice_assert(
        residual.cwiseAbs().maxCoeff() < joint_rotation_tolerance,
        "rest_rotation_matrix must be orthonormal"
    );
    utils::nice_assert(
        std::abs(m_rest_rotation_matrix.determinant() - 1.0)
            < joint_rotation_tolerance,
        "rest_rotation_matrix must have unit determinant"
    );
}

double FixedJoint::stiffness() const {return m_free_joint.stiffness();}
double FixedJoint::damping() const {return m_free_joint.damping();}
double FixedJoint::rotational_stiffness() const {return m_rotational_stiffness;}
double FixedJoint::rotational_damping() const {return m_rotational_damping;}
const Eigen::Matrix3d& FixedJoint::rest_rotation_matrix() const
{
    return m_rest_rotation_matrix;
}
} // End namespace cosserat::physics
