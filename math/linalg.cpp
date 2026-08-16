#include "math/linalg.hpp"

#include <algorithm>

#include <Eigen/Geometry>

namespace cosserat::math {
Eigen::Vector3d inverse_rotate(
    const Eigen::Matrix3d& from_frame, const Eigen::Matrix3d& to_frame
)
{
    utils::nice_assert(
        from_frame.array().isFinite().all() and to_frame.array().isFinite().all(),
        "Frames must contain only finite values"
    );

    const Eigen::Matrix3d relative = to_frame * from_frame.transpose();

    // Antisymmetric part of the relative rotation, which is 2 sin(theta) times
    // the rotation axis.
    Eigen::Vector3d axis;
    axis(0) = relative(2, 1) - relative(1, 2);
    axis(1) = relative(0, 2) - relative(2, 0);
    axis(2) = relative(1, 0) - relative(0, 1);

    // Any excursion outside [-1, 3] is numerical error in the trace.
    const double trace = std::clamp(relative.trace(), -1.0, 3.0);

    // The epsilon keeps the division defined when the frames coincide; the
    // axis is then the zero vector, so the scaled result is still zero.
    const double angle = std::acos(0.5 * trace - 0.5) + rotation_tolerance;
    const double magnitude = -0.5 * angle / std::sin(angle);

    return magnitude * axis;
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
} // End namespace cosserat::math
