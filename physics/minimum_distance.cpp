#include "physics/minimum_distance.hpp"

#include <algorithm>
#include <cmath>

#include "utils/assertions.hpp"

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/** @brief Relative tolerance on the parallel test between two segments. */
constexpr double parallel_tolerance = 1e-6;

/** @brief Clamps a segment parameter into its valid range. */
double clip_unit(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

/** @brief Whether a segment parameter falls outside its valid range. */
bool out_of_unit_bounds(double value)
{
    return value < 0.0 or value > 1.0;
}

} // End anonymous namespace

MinimumDistanceResult minimum_distance_segment_segment(
    const Eigen::Vector3d& first_start,
    const Eigen::Vector3d& first_edge,
    const Eigen::Vector3d& second_start,
    const Eigen::Vector3d& second_edge
)
{
    const double e1e1 = first_edge.dot(first_edge);
    const double e1e2 = first_edge.dot(second_edge);
    const double e2e2 = second_edge.dot(second_edge);

    nice_assert(e1e1 > 0.0, "First segment has zero length");
    nice_assert(e2e2 > 0.0, "Second segment has zero length");

    const double x1e1 = first_start.dot(first_edge);
    const double x1e2 = first_start.dot(second_edge);
    const double x2e1 = first_edge.dot(second_start);
    const double x2e2 = second_start.dot(second_edge);

    double s = 0.0;
    double t = 0.0;

    // Near-parallel segments make the unconstrained solution singular, so they
    // are handled by projecting one onto the other instead.
    const bool parallel =
        std::abs(1.0 - (e1e2 * e1e2) / (e1e1 * e2e2)) < parallel_tolerance;
    if (parallel)
    {
        t = clip_unit((x2e1 - x1e1) / e1e1);
        s = clip_unit((x1e2 + t * e1e2 - x2e2) / e2e2);
    }
    else
    {
        // Unconstrained minimiser of the squared distance between the two
        // infinite lines.
        s = (e1e1 * (x1e2 - x2e2) + e1e2 * (x2e1 - x1e1)) / (e1e1 * e2e2 - e1e2 * e1e2);
        t = (e1e2 * s + x2e1 - x1e1) / e1e1;

        if (out_of_unit_bounds(s) or out_of_unit_bounds(t))
        {
            // The minimum lies off at least one segment, so it must be on a
            // boundary. Try each of the four and keep the best.
            double best_t = clip_unit((x2e1 - x1e1) / e1e1);
            double best_s = 0.0;
            double best_distance =
                (first_start + first_edge * best_t - second_start).norm();

            double candidate_t = clip_unit((x2e1 + e1e2 - x1e1) / e1e1);
            double candidate_distance =
                (first_start + first_edge * candidate_t - second_start - second_edge)
                    .norm();
            if (candidate_distance < best_distance)
            {
                best_s = 1.0;
                best_t = candidate_t;
                best_distance = candidate_distance;
            }

            double candidate_s = clip_unit((x1e2 - x2e2) / e2e2);
            candidate_distance =
                (second_start + candidate_s * second_edge - first_start).norm();
            if (candidate_distance < best_distance)
            {
                best_s = candidate_s;
                best_t = 0.0;
                best_distance = candidate_distance;
            }

            candidate_s = clip_unit((x1e2 + e1e2 - x2e2) / e2e2);
            candidate_distance =
                (second_start + candidate_s * second_edge - first_start - first_edge)
                    .norm();
            if (candidate_distance < best_distance)
            {
                best_s = candidate_s;
                best_t = 1.0;
            }

            s = best_s;
            t = best_t;
        }
    }

    MinimumDistanceResult result;
    result.contact_point_one = first_start + t * first_edge;
    result.contact_point_two = second_start + s * second_edge;
    result.distance_vector = result.contact_point_two - result.contact_point_one;
    return result;
}

MinimumDistanceResult minimum_distance_segment_point(
    const Eigen::Vector3d& segment_start,
    const Eigen::Vector3d& segment_edge,
    const Eigen::Vector3d& point
)
{
    const double e1e1 = segment_edge.dot(segment_edge);
    nice_assert(e1e1 > 0.0, "Segment has zero length");

    const double x1e1 = segment_start.dot(segment_edge);
    const double x2e1 = segment_edge.dot(point);

    const double t = clip_unit((x2e1 - x1e1) / e1e1);

    MinimumDistanceResult result;
    result.contact_point_one = segment_start + t * segment_edge;
    result.contact_point_two = point;
    result.distance_vector = point - result.contact_point_one;
    return result;
}

bool boxes_not_intersecting(const AxisAlignedBox& first, const AxisAlignedBox& second)
{
    for (Eigen::Index axis = 0; axis < 3; ++axis)
    {
        if (first.upper(axis) < second.lower(axis)) return true;
        if (first.lower(axis) > second.upper(axis)) return true;
    }
    return false;
}

AxisAlignedBox bounding_box_rod(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths
)
{
    nice_assert(positions.rows() > 0, "Rod must have at least one node");
    nice_assert(radii.size() > 0, "Rod must have at least one element");
    nice_assert(lengths.size() == radii.size(), "Radii and lengths must agree");

    // One padding for every axis: the thickest element plus the longest one.
    const double padding = radii.maxCoeff() + lengths.maxCoeff();

    AxisAlignedBox box;
    box.lower = positions.colwise().minCoeff().transpose().array() - padding;
    box.upper = positions.colwise().maxCoeff().transpose().array() + padding;
    return box;
}

AxisAlignedBox bounding_box_cylinder(
    const Eigen::Vector3d& center,
    const Eigen::Matrix3d& frame,
    double radius,
    double length
)
{
    // Half-extents in the body frame, rotated into the lab. The frame maps lab
    // vectors into the body, so its transpose carries the extents back out.
    const Eigen::Vector3d local_extents(radius, radius, 0.5 * length);
    const Eigen::Vector3d world_extents =
        (frame.transpose() * local_extents).cwiseAbs();

    AxisAlignedBox box;
    box.lower = center - world_extents;
    box.upper = center + world_extents;
    return box;
}

AxisAlignedBox bounding_box_sphere(const Eigen::Vector3d& center, double radius)
{
    AxisAlignedBox box;
    box.lower = center.array() - radius;
    box.upper = center.array() + radius;
    return box;
}

bool prune_rod_rod(
    const Vector3DStack& positions_one,
    const Eigen::VectorXd& radii_one,
    const Eigen::VectorXd& lengths_one,
    const Vector3DStack& positions_two,
    const Eigen::VectorXd& radii_two,
    const Eigen::VectorXd& lengths_two
)
{
    return boxes_not_intersecting(
        bounding_box_rod(positions_two, radii_two, lengths_two),
        bounding_box_rod(positions_one, radii_one, lengths_one)
    );
}

bool prune_rod_cylinder(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths,
    const Eigen::Vector3d& center,
    const Eigen::Matrix3d& frame,
    double radius,
    double length
)
{
    return boxes_not_intersecting(
        bounding_box_cylinder(center, frame, radius, length),
        bounding_box_rod(positions, radii, lengths)
    );
}

bool prune_rod_sphere(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Eigen::Vector3d& center,
    double radius
)
{
    nice_assert(positions.rows() > 0, "Rod must have at least one node");
    nice_assert(radii.size() > 0, "Rod must have at least one element");

    // Padded by the rod's radius alone, unlike the other two tests.
    AxisAlignedBox rod_box;
    const double padding = radii.maxCoeff();
    rod_box.lower = positions.colwise().minCoeff().transpose().array() - padding;
    rod_box.upper = positions.colwise().maxCoeff().transpose().array() + padding;

    return boxes_not_intersecting(bounding_box_sphere(center, radius), rod_box);
}
} // End namespace cosserat::physics
