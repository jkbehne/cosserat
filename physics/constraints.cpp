/**
 * @file constraints.cpp
 * @brief Non-template implementations for the boundary conditions.
 */

#include "physics/constraints.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <numbers>

namespace cosserat::physics {

namespace {

/** @brief Fails unless every entry of a stack of vectors is finite. */
void assert_finite(const Vector3DStack& stack, const char* message)
{
    utils::nice_assert(stack.array().isFinite().all(), message);
}

/** @brief Fails unless every entry of a stack of matrices is finite. */
void assert_finite(const Matrix3DStack& stack, const char* message)
{
    for (const Eigen::Matrix3d& entry : stack)
    {
        utils::nice_assert(entry.array().isFinite().all(), message);
    }
}

/** @brief Converts a boolean selector to a 1.0 / 0.0 mask. */
Eigen::Array3d selector_mask(const std::array<bool, 3>& selector)
{
    Eigen::Array3d mask;
    mask << (selector[0] ? 1.0 : 0.0),
            (selector[1] ? 1.0 : 0.0),
            (selector[2] ? 1.0 : 0.0);
    return mask;
}

} // End anonymous namespace

Eigen::Index resolve_index(std::int64_t index, Eigen::Index count)
{
    utils::nice_assert(count > 0, "Cannot index into an empty stack");

    const Eigen::Index resolved = (index < 0)
        ? count + static_cast<Eigen::Index>(index)
        : static_cast<Eigen::Index>(index);

    utils::nice_assert(
        resolved >= 0 and resolved < count, "Constraint index is out of range"
    );
    return resolved;
}

Eigen::Matrix3d rotation_matrix(double scale, const Eigen::Vector3d& axis)
{
    utils::nice_assert(std::isfinite(scale), "Rotation scale must be finite");
    utils::nice_assert(
        axis.array().isFinite().all(), "Rotation axis must be finite"
    );

    // The reference implementation takes the angle from the axis length and
    // normalises the axis afterwards, so a unit axis makes scale the angle.
    const double length = axis.norm();
    utils::nice_assert(
        length > rotation_tolerance,
        "Rotation axis is too short to define a direction"
    );

    const double angle = scale * length;
    if (std::abs(angle) < rotation_tolerance) return Eigen::Matrix3d::Identity();

    // Transpose because the reference convention is the transpose of the
    // textbook Rodrigues matrix; see the header warning.
    const Eigen::Vector3d unit_axis = axis / length;
    return Eigen::AngleAxisd(angle, unit_axis).toRotationMatrix().transpose();
}

// ---------------------------------------------------------------------------
// OneEndFixedBoundaryCondition
// ---------------------------------------------------------------------------

OneEndFixedBoundaryCondition::OneEndFixedBoundaryCondition(
    Eigen::Vector3d fixed_position, Eigen::Matrix3d fixed_directors
) : m_fixed_position(std::move(fixed_position)),
    m_fixed_directors(std::move(fixed_directors))
{
    utils::nice_assert(
        m_fixed_position.array().isFinite().all(),
        "fixed_position must contain only finite values"
    );
    utils::nice_assert(
        m_fixed_directors.array().isFinite().all(),
        "fixed_directors must contain only finite values"
    );
}

const Eigen::Vector3d& OneEndFixedBoundaryCondition::fixed_position() const
{
    return m_fixed_position;
}
const Eigen::Matrix3d& OneEndFixedBoundaryCondition::fixed_directors() const
{
    return m_fixed_directors;
}

// ---------------------------------------------------------------------------
// GeneralConstraint
// ---------------------------------------------------------------------------

GeneralConstraint::GeneralConstraint(
    std::vector<std::int64_t> position_indices,
    Vector3DStack fixed_positions,
    std::vector<std::int64_t> director_indices,
    std::array<bool, 3> translational_selector,
    std::array<bool, 3> rotational_selector
) : m_position_indices(std::move(position_indices)),
    m_fixed_positions(std::move(fixed_positions)),
    m_director_indices(std::move(director_indices)),
    m_translational_selector(selector_mask(translational_selector)),
    m_rotational_selector(selector_mask(rotational_selector))
{
    utils::nice_assert(
        m_fixed_positions.rows()
            == static_cast<Eigen::Index>(m_position_indices.size()),
        "Expected one fixed position per constrained node index"
    );
    assert_finite(m_fixed_positions, "fixed_positions must contain only finite values");
    utils::nice_assert(
        not m_position_indices.empty() or not m_director_indices.empty(),
        "Constraint has no node or element indices and would do nothing"
    );
}

const std::vector<std::int64_t>& GeneralConstraint::position_indices() const
{
    return m_position_indices;
}
const Vector3DStack& GeneralConstraint::fixed_positions() const
{
    return m_fixed_positions;
}
const std::vector<std::int64_t>& GeneralConstraint::director_indices() const
{
    return m_director_indices;
}
const Eigen::Array3d& GeneralConstraint::translational_selector() const
{
    return m_translational_selector;
}
const Eigen::Array3d& GeneralConstraint::rotational_selector() const
{
    return m_rotational_selector;
}

// ---------------------------------------------------------------------------
// FixedConstraint
// ---------------------------------------------------------------------------

FixedConstraint::FixedConstraint(
    std::vector<std::int64_t> position_indices,
    Vector3DStack fixed_positions,
    std::vector<std::int64_t> director_indices,
    Matrix3DStack fixed_directors
) : m_position_indices(std::move(position_indices)),
    m_fixed_positions(std::move(fixed_positions)),
    m_director_indices(std::move(director_indices)),
    m_fixed_directors(std::move(fixed_directors))
{
    utils::nice_assert(
        m_fixed_positions.rows()
            == static_cast<Eigen::Index>(m_position_indices.size()),
        "Expected one fixed position per clamped node index"
    );
    utils::nice_assert(
        m_fixed_directors.size() == m_director_indices.size(),
        "Expected one fixed director per clamped element index"
    );
    assert_finite(m_fixed_positions, "fixed_positions must contain only finite values");
    assert_finite(m_fixed_directors, "fixed_directors must contain only finite values");
    utils::nice_assert(
        not m_position_indices.empty() or not m_director_indices.empty(),
        "Constraint has no node or element indices and would do nothing"
    );
}

const std::vector<std::int64_t>& FixedConstraint::position_indices() const
{
    return m_position_indices;
}
const Vector3DStack& FixedConstraint::fixed_positions() const
{
    return m_fixed_positions;
}
const std::vector<std::int64_t>& FixedConstraint::director_indices() const
{
    return m_director_indices;
}
const Matrix3DStack& FixedConstraint::fixed_directors() const
{
    return m_fixed_directors;
}

// ---------------------------------------------------------------------------
// HelicalBucklingBoundaryCondition
// ---------------------------------------------------------------------------

HelicalBucklingBoundaryCondition::HelicalBucklingBoundaryCondition(
    Eigen::Vector3d position_start,
    Eigen::Vector3d position_end,
    Eigen::Matrix3d director_start,
    Eigen::Matrix3d director_end,
    double twisting_time,
    double slack,
    double number_of_rotations
) : m_twisting_time(twisting_time),
    m_slack(slack),
    m_number_of_rotations(number_of_rotations)
{
    utils::nice_assert(
        std::isfinite(m_twisting_time) and m_twisting_time > 0.0,
        "twisting_time must be finite and greater than zero"
    );
    utils::nice_assert(std::isfinite(m_slack), "slack must be finite");
    utils::nice_assert(
        std::isfinite(m_number_of_rotations), "number_of_rotations must be finite"
    );
    utils::nice_assert(
        position_start.array().isFinite().all()
        and position_end.array().isFinite().all(),
        "End positions must contain only finite values"
    );
    utils::nice_assert(
        director_start.array().isFinite().all()
        and director_end.array().isFinite().all(),
        "End directors must contain only finite values"
    );

    const Eigen::Vector3d separation = position_end - position_start;
    utils::nice_assert(
        separation.norm() > 0.0, "End positions must be distinct"
    );
    const Eigen::Vector3d direction = separation / separation.norm();

    // Each end supplies half the total rotation and half the total slack, so
    // the two together deliver the requested twist and shortening.
    const double angular_speed =
        (2.0 * m_number_of_rotations * std::numbers::pi / m_twisting_time) / 2.0;
    const double shrink_speed = m_slack / (m_twisting_time * 2.0);

    m_final_start_position = position_start + 0.5 * m_slack * direction;
    m_final_end_position = position_end - 0.5 * m_slack * direction;

    m_angular_velocity = angular_speed * direction;
    m_shrink_velocity = shrink_speed * direction;

    const double theta = m_number_of_rotations * std::numbers::pi;
    m_final_start_directors = rotation_matrix(theta, direction) * director_start;
    m_final_end_directors = rotation_matrix(-theta, direction) * director_end;
}

double HelicalBucklingBoundaryCondition::twisting_time() const
{
    return m_twisting_time;
}
double HelicalBucklingBoundaryCondition::slack() const {return m_slack;}
double HelicalBucklingBoundaryCondition::number_of_rotations() const
{
    return m_number_of_rotations;
}
const Eigen::Vector3d& HelicalBucklingBoundaryCondition::final_start_position() const
{
    return m_final_start_position;
}
const Eigen::Vector3d& HelicalBucklingBoundaryCondition::final_end_position() const
{
    return m_final_end_position;
}
const Eigen::Vector3d& HelicalBucklingBoundaryCondition::angular_velocity() const
{
    return m_angular_velocity;
}
const Eigen::Vector3d& HelicalBucklingBoundaryCondition::shrink_velocity() const
{
    return m_shrink_velocity;
}
const Eigen::Matrix3d& HelicalBucklingBoundaryCondition::final_start_directors() const
{
    return m_final_start_directors;
}
const Eigen::Matrix3d& HelicalBucklingBoundaryCondition::final_end_directors() const
{
    return m_final_end_directors;
}
} // End namespace cosserat::physics
