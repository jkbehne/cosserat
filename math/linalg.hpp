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
} // End namespace cosserat::math
