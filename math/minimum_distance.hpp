#pragma once

/**
 * @file minimum_distance.hpp
 * @brief Closest-approach geometry and broad-phase pruning for contact.
 *
 * Contact between two bodies reduces to two questions: could they possibly be
 * touching, and if so how far apart are they at their closest approach. This
 * header answers both.
 *
 * Every shape is treated as a capsule, a straight segment swollen by a radius.
 * A rod element is a segment from one node to the next; a rigid cylinder is a
 * segment along its axis; a sphere is a degenerate segment of zero length.
 * Narrow-phase contact is then the distance between two segments compared
 * against the sum of their radii, which is what
 * @ref minimum_distance_segment_segment and
 * @ref minimum_distance_segment_point compute.
 *
 * The broad phase is a single axis-aligned box per body. There is no
 * hierarchy, no spatial hash and no sweep: the box only rules out pairs that
 * cannot possibly touch, and everything surviving it goes through the full
 * element-by-element test. That matches the reference implementation, which is
 * likewise brute force with one bounding-box rejection.
 *
 * This is a port of PyElastica's @c contact_utils module.
 */

#include <Eigen/Core>

#include "math/types.hpp"

namespace cosserat::math {

/**
 * @brief Closest approach between two segments.
 *
 * The two contact points are the points on each segment realising the minimum
 * distance, and @ref distance_vector runs from the first to the second.
 */
struct MinimumDistanceResult
{
public: // Members
    /**
     * @brief Vector from the closest point on the first segment to the closest
     *        point on the second.
     */
    Eigen::Vector3d distance_vector;

    /** @brief Closest point on the second segment. */
    Eigen::Vector3d contact_point_two;

    /**
     * @brief Closest point on the first segment.
     *
     * @warning The reference implementation returns @c x1 - t*e1 here where
     *          the closest point is @c x1 + t*e1, so its value for this field
     *          is not the contact point at all. Only the rod-to-cylinder
     *          kernel reads a contact point, and it reads
     *          @ref contact_point_two, so the error never surfaces there. This
     *          port returns the correct point; see the tests, which pin both
     *          the correct value and the fact that it differs from the
     *          reference.
     */
    Eigen::Vector3d contact_point_one;
};

/**
 * @brief An axis-aligned bounding box.
 */
struct AxisAlignedBox
{
public: // Members
    /** @brief Corner with the smallest coordinate on every axis. */
    Eigen::Vector3d lower;

    /** @brief Corner with the largest coordinate on every axis. */
    Eigen::Vector3d upper;
};

/**
 * @brief Closest approach between two segments.
 *
 * Each segment runs from a start point along an edge vector, so the first
 * spans @c x1 to @c x1 + e1 and the second @c x2 to @c x2 + e2. The
 * unconstrained minimiser is found first, and only if it falls outside either
 * segment are the four endpoint cases tried and the best kept. Segments close
 * to parallel are detected up front, since the unconstrained solution is
 * singular there.
 *
 * @param first_start Start of the first segment.
 * @param first_edge Edge vector of the first segment; must not be degenerate.
 * @param second_start Start of the second segment.
 * @param second_edge Edge vector of the second segment; must not be
 *        degenerate.
 * @return The closest approach.
 */
MinimumDistanceResult minimum_distance_segment_segment(
    const Eigen::Vector3d& first_start,
    const Eigen::Vector3d& first_edge,
    const Eigen::Vector3d& second_start,
    const Eigen::Vector3d& second_edge
);

/**
 * @brief Closest approach between a segment and a point.
 *
 * Used for a sphere, whose capsule is a segment of zero length.
 *
 * @param segment_start Start of the segment.
 * @param segment_edge Edge vector of the segment; must not be degenerate.
 * @param point The point.
 * @return The closest approach, whose @c contact_point_two is the point
 *         itself.
 */
MinimumDistanceResult minimum_distance_segment_point(
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_edge,
    const Eigen::Vector3d& point
);

/**
 * @brief Whether two boxes fail to overlap on at least one axis.
 * @param first First box.
 * @param second Second box.
 * @return True when the boxes are disjoint, so the pair can be skipped.
 */
bool boxes_not_intersecting(
    const AxisAlignedBox& first, const AxisAlignedBox& second
);

/**
 * @brief Bounding box around a rod, padded for its thickness and elements.
 *
 * The padding is the largest radius plus the largest element length, applied
 * on every axis. That is looser than necessary, which is deliberate: it is one
 * cheap number rather than a per-element box.
 *
 * @param positions Node positions.
 * @param radii Element radii.
 * @param lengths Element lengths.
 * @return The padded box.
 */
AxisAlignedBox bounding_box_rod(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths
);

/**
 * @brief Bounding box around a cylinder.
 *
 * The cylinder's half-extents are expressed in its own frame and rotated into
 * the lab, then taken absolutely.
 *
 * @warning This reproduces a bug in the reference implementation and does not
 *          actually bound the cylinder. Taking the absolute value after
 *          summing the rotated extents lets opposing components cancel, where
 *          a correct box needs the absolute value inside the sum. For a
 *          cylinder of radius 0.2 and length 2 tilted half a radian, the
 *          half-extent along x comes out as 0.304 against a correct 0.655, and
 *          the cylinder's own end face lies outside its own box. Since this
 *          box is only used to reject pairs, the effect is that a tilted
 *          cylinder can have real contacts silently pruned away. The fix is to
 *          take the absolute value of the rotation before multiplying; it is
 *          not applied here so that results match the reference.
 *
 * @param center Centre of the cylinder.
 * @param frame Body frame; its third row is the axis.
 * @param radius Cylinder radius.
 * @param length Cylinder length.
 * @return The box.
 */
AxisAlignedBox bounding_box_cylinder(
    const Eigen::Vector3d& center,
    const Eigen::Matrix3d& frame,
    double radius,
    double length
);

/**
 * @brief Bounding box around a sphere.
 * @param center Centre of the sphere.
 * @param radius Sphere radius.
 * @return The box.
 */
AxisAlignedBox bounding_box_sphere(const Eigen::Vector3d& center, double radius);

/**
 * @brief Whether a rod and another rod cannot possibly be touching.
 * @param positions_one Node positions of the first rod.
 * @param radii_one Element radii of the first rod.
 * @param lengths_one Element lengths of the first rod.
 * @param positions_two Node positions of the second rod.
 * @param radii_two Element radii of the second rod.
 * @param lengths_two Element lengths of the second rod.
 * @return True when the pair can be skipped.
 */
bool prune_rod_rod(
    const Vector3DStack& positions_one,
    const Eigen::VectorXd& radii_one,
    const Eigen::VectorXd& lengths_one,
    const Vector3DStack& positions_two,
    const Eigen::VectorXd& radii_two,
    const Eigen::VectorXd& lengths_two
);

/**
 * @brief Whether a rod and a cylinder cannot possibly be touching.
 * @param positions Node positions of the rod.
 * @param radii Element radii of the rod.
 * @param lengths Element lengths of the rod.
 * @param center Centre of the cylinder.
 * @param frame Body frame of the cylinder.
 * @param radius Cylinder radius.
 * @param length Cylinder length.
 * @return True when the pair can be skipped.
 */
bool prune_rod_cylinder(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths,
    const Eigen::Vector3d& center,
    const Eigen::Matrix3d& frame,
    double radius,
    double length
);

/**
 * @brief Whether a rod and a sphere cannot possibly be touching.
 *
 * @param positions Node positions of the rod.
 * @param radii Element radii of the rod.
 * @param center Centre of the sphere.
 * @param radius Sphere radius.
 * @return True when the pair can be skipped.
 *
 * @note Unlike the rod-to-rod and rod-to-cylinder tests, the rod's box is
 *       padded by its radius alone rather than radius plus element length.
 *       That is what the reference implementation does, and it makes this
 *       test the tightest of the three.
 */
bool prune_rod_sphere(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::Vector3d& center,
    double radius
);
} // End namespace cosserat::physics
