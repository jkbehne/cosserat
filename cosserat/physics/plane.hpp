#pragma once

/**
 * @file plane.hpp
 * @brief An infinite plane, as a contact surface.
 *
 * A plane is not a body: it has no mass, no state and no dynamics, and nothing
 * ever integrates it. It exists so that contact rules have something to push
 * against, which is why it lives here rather than alongside the rods and rigid
 * bodies.
 *
 * @note This is a placeholder sized to what the contact rules actually read.
 *       If a fuller surface type appears later, only the two accessors below
 *       need to survive for @ref cosserat::physics::RodPlaneContact and its
 *       relatives to keep working.
 */

#include <Eigen/Core>

namespace cosserat::physics {

/**
 * @brief An infinite plane through a point with a given normal.
 *
 * The normal picks out the half-space a body is pushed back into: a body on
 * the side the normal points toward is out of contact, and one that has
 * crossed to the other side is penetrating.
 */
class Plane
{
public: // Static constexpr members
    /** @brief Tolerance for the unit-length check on the normal. */
    static constexpr double tolerance = 1e-12;

private: // Members
    Eigen::Vector3d m_origin;
    Eigen::Vector3d m_normal;

public: // Methods
    /**
     * @brief Builds a plane.
     * @param origin Any point on the plane.
     * @param normal Unit vector normal to the plane.
     */
    Plane(const Eigen::Vector3d& origin, const Eigen::Vector3d& normal);

    /**
     * @brief Signed distance from the plane to a point.
     *
     * Positive on the side the normal points toward.
     *
     * @param point Point to measure.
     * @return The signed distance.
     */
    double signed_distance(const Eigen::Vector3d& point) const;

    /** @brief A point on the plane. */
    const Eigen::Vector3d& origin() const;

    /** @brief Unit normal to the plane. */
    const Eigen::Vector3d& normal() const;
};
} // End namespace cosserat::physics
