#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "utils/assertions.hpp"

#include <Eigen/Dense>

namespace cosserat::math {

template<int FixedDim>
Eigen::Matrix<double, Eigen::Dynamic, FixedDim> batched_matrix_vector(
    const std::vector<Eigen::Matrix<double, FixedDim, FixedDim>>& matrices,
    const Eigen::Matrix<double, FixedDim, Eigen::Dynamic>& vectors,
    bool ignore_size_mismatch = false
)
{
    const auto num_matrices = static_cast<Eigen::Index>(matrices.size());
    const auto num_vectors = vectors.cols();
    utils::nice_assert(
        (num_matrices == num_vectors) or ignore_size_mismatch,
        "Expected number of matrices and vectors to be equal"
    );

    const auto num_entries = std::min(num_matrices, num_vectors);
    Eigen::Matrix<double, Eigen::Dynamic, FixedDim> result(num_entries, FixedDim);
    for (int idx = 0; idx < num_entries; ++idx)
    {
        result.row(idx) = (matrices[idx] * vectors.col(idx)).transpose();
    }
    return result;
}

template<typename Derived>
bool is_unit_vector(const Eigen::MatrixBase<Derived>& vec,
                    typename Derived::RealScalar tolerance)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
    using Real = typename Derived::RealScalar;
    return std::abs(vec.norm() - Real(1)) < tolerance;
}

/**
 * @brief Scaled rotation axis carrying one frame onto another.
 *
 * Forms the relative rotation @f$ R = Q_{to} Q_{from}^{T} @f$, extracts its
 * antisymmetric part, and scales by @f$ -\theta / (2\sin\theta) @f$ where
 * @f$ \theta @f$ is recovered from the trace of @f$ R @f$.
 *
 * @warning The result is @f$ -\theta \mathbf{u} @f$ rather than
 *          @f$ +\theta \mathbf{u} @f$, matching the reference implementation.
 *          The sign is absorbed by the callers that use it.
 *
 * @param from_frame Frame the rotation starts from.
 * @param to_frame Frame the rotation ends at.
 * @return The scaled rotation axis; the zero vector when the frames coincide.
 */
Eigen::Vector3d inverse_rotate(
    const Eigen::Matrix3d& from_frame, const Eigen::Matrix3d& to_frame
);


/**
 * @brief Smallest axis length and rotation angle treated as significant.
 *
 * An axis shorter than this is rejected as degenerate, and a rotation angle
 * smaller than this yields the identity exactly rather than a matrix built
 * from @c sin and @c cos of a near-zero argument.
 */
inline constexpr double rotation_tolerance = 1e-12;

/**
 * @brief Rotation matrix about an axis, in the reference implementation's
 *        convention.
 *
 * The rotation angle is @c scale multiplied by the norm of @p axis, and the
 * axis is normalised before use. The matrix itself comes from Eigen's
 * @c AngleAxis conversion rather than a hand-written Rodrigues expansion.
 *
 * @warning The returned matrix is the transpose of the textbook Rodrigues
 *          matrix, that is @f$ R(-\theta) @f$ rather than @f$ R(\theta) @f$.
 *          This matches the reference implementation, whose frame convention
 *          maps lab-frame vectors into the material frame. Passing a negated
 *          angle recovers the textbook orientation.
 *
 * @param scale Angle scale factor; the effective angle is
 *        @c scale * @c axis.norm().
 * @param axis Rotation axis; its length must exceed @ref rotation_tolerance.
 * @return The 3 by 3 rotation matrix, or the identity when the effective angle
 *         is smaller than @ref rotation_tolerance.
 */
Eigen::Matrix3d rotation_matrix(double scale, const Eigen::Vector3d& axis);
} // End namespace cosserat::math
