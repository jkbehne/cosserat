#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <cosserat/utils/assertions.hpp>

#include <Eigen/Dense>

namespace cosserat::math {

/**
 * @brief Multiplies a batch of square matrices by a batch of vectors.
 *
 * Entry @c i of the result is @c matrices[i] multiplied by column @c i of
 * @p vectors. Note the change of layout across the call: the vectors arrive as
 * columns of a @c FixedDim by @c N matrix, while the products are returned as
 * rows of an @c N by @c FixedDim one.
 *
 * @f[ \mathrm{result}_i = \left(M_i \, v_i\right)^{T}, \qquad
 *     \mathrm{result}_i = \left(M_i^{T} \, v_i\right)^{T}
 *     \text{ when } \mathrm{TransposeMatrices} @f]
 *
 * @p vectors is taken by @c Eigen::Ref, so blocks, maps and sub-ranges bind
 * without being copied, and a computed expression is evaluated exactly once
 * rather than once per column. Passing @c torque.rightCols(n) therefore costs
 * nothing beyond the call itself.
 *
 * @tparam AllowSizeMismatch When false, the two batches must be the same
 *         length and a mismatch fails an assertion. When true, the check is
 *         compiled out and the shorter length wins.
 * @tparam TransposeMatrices When true, each matrix is transposed before being
 *         applied. For a batch of frames this turns a material-frame
 *         projection into a lab-frame one, since a rotation's transpose is its
 *         inverse.
 * @tparam FixedDim Dimension shared by the matrices and vectors; deduced from
 *         @p matrices.
 * @param matrices One square matrix per entry.
 * @param vectors One column per entry; any Eigen expression with @c FixedDim
 *        rows and a @c double scalar.
 * @return An @c N by @c FixedDim matrix of products, where @c N is the number
 *         of entries processed.
 *
 * @warning Truncation under @p AllowSizeMismatch always keeps the *leading*
 *          entries of the longer batch. Pairing a trailing sub-range with a
 *          full batch therefore needs the sub-range extracted by the caller,
 *          not this flag.
 */
template<
    bool AllowSizeMismatch = false, bool TransposeMatrices = false, int FixedDim>
Eigen::Matrix<double, Eigen::Dynamic, FixedDim> batched_matrix_vector(
    const std::vector<Eigen::Matrix<double, FixedDim, FixedDim>>& matrices,
    // type_identity_t keeps this parameter out of deduction, so FixedDim comes
    // from the matrices alone. Without it a block argument fails to deduce.
    std::type_identity_t<
        const Eigen::Ref<const Eigen::Matrix<double, FixedDim, Eigen::Dynamic>>&>
        vectors
)
{
    const auto num_matrices = static_cast<Eigen::Index>(matrices.size());
    const auto num_vectors = vectors.cols();

    if constexpr (not AllowSizeMismatch)
    {
        utils::nice_assert(
            num_matrices == num_vectors,
            "Expected number of matrices and vectors to be equal"
        );
    }

    const Eigen::Index num_entries = std::min(num_matrices, num_vectors);
    Eigen::Matrix<double, Eigen::Dynamic, FixedDim> result(num_entries, FixedDim);
    for (Eigen::Index idx = 0; idx < num_entries; ++idx)
    {
        const auto& matrix = matrices[static_cast<std::size_t>(idx)];
        if constexpr (TransposeMatrices)
        {
            result.row(idx) = (matrix.transpose() * vectors.col(idx)).transpose();
        }
        else
        {
            result.row(idx) = (matrix * vectors.col(idx)).transpose();
        }
    }
    return result;
}

/**
 * @brief Reports whether a vector has unit length to within a tolerance.
 *
 * Compares @c |‖v‖ - 1| against @p tolerance using a strict less-than, so a
 * deviation exactly equal to the tolerance is rejected.
 *
 * Accepts any vector expression, including blocks, maps and unevaluated
 * arithmetic, without materialising it first.
 *
 * @tparam Derived Eigen vector expression type.
 * @param vec Vector to test; must be a vector rather than a general matrix.
 * @param tolerance Largest accepted deviation of the norm from one.
 * @return True when the vector is of unit length to within @p tolerance.
 *
 * @note A negative tolerance can never be satisfied, so the result is then
 *       always false.
 */
template<typename Derived>
bool is_unit_vector(const Eigen::MatrixBase<Derived>& vec,
                    typename Derived::RealScalar tolerance)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
    using Real = typename Derived::RealScalar;
    return std::abs(vec.norm() - Real(1)) < tolerance;
}

/**
 * @brief Reports whether a square matrix is orthogonal to within a tolerance.
 *
 * Forms @f$ A^{*} A @f$ and measures how far it sits from the identity,
 * comparing the largest absolute entry of the difference against @p tolerance
 * with a strict less-than. A deviation exactly equal to the tolerance is
 * therefore rejected, matching @ref is_unit_vector.
 *
 * @f[ \max_{ij} \left| \left(A^{*} A - I\right)_{ij} \right| < \mathrm{tolerance} @f]
 *
 * Both reflections and rotations pass, since orthogonality constrains the
 * determinant only to plus or minus one. Use the determinant separately if a
 * proper rotation is required.
 *
 * Accepts any square matrix expression, including blocks, maps and unevaluated
 * arithmetic. The argument is bound once through @c Eigen::Ref, so a computed
 * expression is evaluated a single time rather than once for each side of the
 * product.
 *
 * @tparam Derived Eigen expression type.
 * @param matrix Square matrix to test.
 * @param tolerance Largest accepted departure from the identity, measured
 *        entry by entry.
 * @return True when the matrix is orthogonal to within @p tolerance.
 *
 * @note For a complex scalar this uses the conjugate transpose, so it tests
 *       the unitary condition @f$ A^{H} A = I @f$ rather than the literal
 *       @f$ A^{T} A = I @f$. The two coincide for real scalars, and the
 *       unitary form is the one that characterises a norm-preserving map.
 *
 * @note A negative tolerance can never be satisfied, so the result is then
 *       always false. A zero-by-zero matrix is reported as orthogonal, being
 *       vacuously so.
 */
template<typename Derived>
bool is_orthogonal(const Eigen::MatrixBase<Derived>& matrix,
                   typename Derived::RealScalar tolerance)
{
    using ScalarType = typename Derived::Scalar;

    // Cast to int: the two dimensions carry distinct unnamed enum types, and
    // comparing them directly draws -Wenum-compare.
    static_assert(
        int(Derived::RowsAtCompileTime) == int(Eigen::Dynamic)
            or int(Derived::ColsAtCompileTime) == int(Eigen::Dynamic)
            or int(Derived::RowsAtCompileTime) == int(Derived::ColsAtCompileTime),
        "is_orthogonal requires a square matrix"
    );

    utils::nice_assert(
        matrix.rows() == matrix.cols(), "is_orthogonal requires a square matrix"
    );

    // An empty matrix is vacuously orthogonal, and reducing over it would
    // otherwise trip Eigen's own empty-matrix assertion.
    if (matrix.rows() == 0) return true;

    // Bound once so the expression is not evaluated separately for each side
    // of the product below.
    const Eigen::Ref<const typename Derived::PlainObject> bound(matrix.derived());

    auto residual = (bound.adjoint() * bound).eval();
    residual.diagonal().array() -= ScalarType(1);

    return residual.cwiseAbs().maxCoeff() < tolerance;
}

namespace detail {

/**
 * @brief Plain column-vector type holding one value per row of a matrix.
 *
 * Primary template is left undefined so that instantiating it with anything
 * other than an @c Eigen::Matrix is a compile error rather than a confusing
 * downstream failure.
 *
 * @tparam PlainType The @c Eigen::Matrix specialisation being reduced.
 * @tparam ResultScalar Scalar the reduced column holds, which need not match
 *         the scalar of @p PlainType.
 */
template<typename PlainType, typename ResultScalar>
struct RowColumn;

/**
 * @brief Specialisation computing the per-row result type for an
 *        @c Eigen::Matrix.
 *
 * The row count and maximum row count carry over while the column count
 * collapses to one. The scalar is supplied separately, because a norm reduces
 * to the real type underlying the input whereas an inner product keeps the
 * input's own scalar.
 *
 * Storage order is forced to column-major, which Eigen requires of any fixed
 * single-column matrix with more than one row and permits in every other case.
 * Alignment options carry over unchanged.
 *
 * @tparam ScalarType Scalar of the matrix being reduced.
 * @tparam Rows Compile-time row count, possibly @c Eigen::Dynamic.
 * @tparam Cols Compile-time column count, possibly @c Eigen::Dynamic.
 * @tparam Options Eigen storage options.
 * @tparam MaxRows Eigen maximum row count.
 * @tparam MaxCols Eigen maximum column count.
 * @tparam ResultScalar Scalar the reduced column holds.
 */
template<
    typename ScalarType, int Rows, int Cols, int Options, int MaxRows, int MaxCols,
    typename ResultScalar>
struct RowColumn<
    Eigen::Matrix<ScalarType, Rows, Cols, Options, MaxRows, MaxCols>, ResultScalar>
{
    /** @brief Storage options legal for a single-column result. */
    static constexpr int reduced_options = Options & ~int(Eigen::RowMajor);

    /** @brief The per-row result type. */
    using type =
        Eigen::Matrix<ResultScalar, Rows, 1, reduced_options, MaxRows, 1>;
};

/**
 * @brief Result type of a per-row norm applied to an Eigen expression.
 *
 * The scalar drops to the real type underlying the input, so a complex matrix
 * still yields real norms.
 *
 * @tparam Derived Eigen expression type.
 */
template<typename Derived>
using RowNormType = typename RowColumn<
    typename Derived::PlainObject,
    typename Eigen::NumTraits<typename Derived::Scalar>::Real>::type;

/**
 * @brief Result type of a per-row inner product applied to an Eigen
 *        expression.
 *
 * The scalar carries over unchanged, since an inner product of complex rows is
 * itself complex.
 *
 * @tparam Derived Eigen expression type.
 */
template<typename Derived>
using RowDotType = typename RowColumn<
    typename Derived::PlainObject, typename Derived::Scalar>::type;

/**
 * @brief Plain three-column type produced by a per-row cross product.
 *
 * Primary template is left undefined so that instantiating it with anything
 * other than an @c Eigen::Matrix is a compile error rather than a confusing
 * downstream failure.
 *
 * @tparam PlainType The @c Eigen::Matrix specialisation being crossed.
 */
template<typename PlainType>
struct CrossProduct;

/**
 * @brief Specialisation computing the cross-product result type.
 *
 * The row count, maximum row count and scalar carry over, and the column count
 * is pinned to three. Storage options carry over except for a single-row
 * result, which Eigen requires to be row-major once it has more than one
 * column; forcing the order there costs nothing, since a one-row matrix has
 * the same layout either way.
 *
 * @tparam ScalarType Scalar of the matrix.
 * @tparam Rows Compile-time row count, possibly @c Eigen::Dynamic.
 * @tparam Cols Compile-time column count; three or @c Eigen::Dynamic.
 * @tparam Options Eigen storage options.
 * @tparam MaxRows Eigen maximum row count.
 * @tparam MaxCols Eigen maximum column count.
 */
template<
    typename ScalarType, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct CrossProduct<Eigen::Matrix<ScalarType, Rows, Cols, Options, MaxRows, MaxCols>>
{
    /** @brief Storage options legal for a three-column result. */
    static constexpr int result_options =
        (Rows == 1) ? ((Options & ~int(Eigen::RowMajor)) | int(Eigen::RowMajor))
                    : Options;

    /** @brief The per-row cross product type. */
    using type = Eigen::Matrix<ScalarType, Rows, 3, result_options, MaxRows, 3>;
};

/**
 * @brief Result type of a per-row cross product applied to an Eigen
 *        expression.
 * @tparam Derived Eigen expression type.
 */
template<typename Derived>
using CrossProductType =
    typename CrossProduct<typename Derived::PlainObject>::type;

} // End namespace detail

/**
 * @brief Euclidean norm of each row of a matrix.
 *
 * Entry @c i of the result is the L2 norm of row @c i of the input, so an
 * @c N by @c C matrix yields an @c N by @c 1 column vector. With one row per
 * entity this turns a stack of vectors into a stack of their lengths.
 *
 * @f[ \mathrm{row\_norms}(A)_i = \sqrt{\sum_j A_{ij}^2} @f]
 *
 * Any column count is supported, fixed or dynamic, and the scalar type is
 * taken from the argument rather than fixed to @c double. Eigen expressions
 * are accepted as well as stored matrices, so @c row_norms(a - b) works
 * without materialising the difference first.
 *
 * @tparam Derived Eigen expression type.
 * @param matrix Matrix or expression whose rows are measured.
 * @return A column vector of row norms, never negative.
 *
 * @note A matrix with no columns yields a norm of zero for every row, since
 *       that is the norm of an empty sum.
 */
template<typename Derived>
detail::RowNormType<Derived> row_norms(const Eigen::MatrixBase<Derived>& matrix)
{
    return matrix.rowwise().norm();
}

/**
 * @brief Inner product of each corresponding pair of rows.
 *
 * Entry @c i of the result is the dot product of row @c i of @p left with row
 * @c i of @p right, so a pair of @c N by @c C matrices yields an @c N by @c 1
 * column vector. With one row per entity this measures a stack of vectors
 * against another stack, for instance projecting each element's force onto its
 * own tangent.
 *
 * @f[ \mathrm{batched\_dot\_product}(A, B)_i = \sum_j \overline{A_{ij}} B_{ij} @f]
 *
 * Implemented as a coefficient-wise product followed by a row-wise sum, which
 * Eigen fuses into a single traversal with no intermediate matrix. That is
 * markedly faster than calling @c dot() on each row in a loop for the narrow
 * matrices this code uses: at three columns the fused form measured about
 * twice the throughput, converging to parity by sixteen columns. Forming
 * @c (A * B.transpose()).diagonal() would instead do @c N squared work to
 * recover @c N values.
 *
 * Both arguments may be Eigen expressions, and they need not be the same kind
 * of expression, so @c batched_dot_product(a.bottomRows(n), b.topRows(n)) and
 * @c batched_dot_product(a.bottomRows(n), b) are both accepted.
 *
 * @tparam DerivedA Eigen expression type of the left argument.
 * @tparam DerivedB Eigen expression type of the right argument; must share a
 *         scalar type with @p DerivedA.
 * @param left First stack of row vectors; conjugated when the scalar is
 *        complex, matching @c Eigen::MatrixBase::dot.
 * @param right Second stack of row vectors; must have the same shape as
 *        @p left.
 * @return A column vector of per-row inner products.
 *
 * @note The compile-time row count of the result follows @p left, so pairing a
 *       dynamic left argument with a fixed-size right one yields a dynamic
 *       result. The runtime shapes are still checked against each other.
 *
 * @note A pair of matrices with no columns yields an inner product of zero for
 *       every row, since that is the value of an empty sum.
 */
template<typename DerivedA, typename DerivedB>
detail::RowDotType<DerivedA> batched_dot_product(
    const Eigen::MatrixBase<DerivedA>& left, const Eigen::MatrixBase<DerivedB>& right
)
{
    static_assert(
        std::is_same_v<typename DerivedA::Scalar, typename DerivedB::Scalar>,
        "batched_dot_product requires both arguments to share a scalar type"
    );
    // Cast to int: the two expressions carry distinct unnamed enum types, and
    // comparing them directly draws -Wenum-compare.
    static_assert(
        int(DerivedA::RowsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedB::RowsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedA::RowsAtCompileTime) == int(DerivedB::RowsAtCompileTime),
        "batched_dot_product arguments disagree on their row count"
    );
    static_assert(
        int(DerivedA::ColsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedB::ColsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedA::ColsAtCompileTime) == int(DerivedB::ColsAtCompileTime),
        "batched_dot_product arguments disagree on their column count"
    );

    utils::nice_assert(
        left.rows() == right.rows() and left.cols() == right.cols(),
        "batched_dot_product requires both arguments to have the same shape"
    );

    // conjugate() is the identity for real scalars, so this costs nothing
    // outside the complex case.
    return left.derived().conjugate().cwiseProduct(right.derived()).rowwise().sum();
}

/**
 * @brief Cross product of each corresponding pair of rows.
 *
 * Row @c i of the result is the cross product of row @c i of @p left with row
 * @c i of @p right, so a pair of @c N by @c 3 stacks yields another @c N by
 * @c 3 stack. With one row per entity this crosses a stack of vectors against
 * another stack, for instance forming the moment of each element's force about
 * its own lever arm.
 *
 * Unlike the other row-wise reductions here, this one is written as an
 * explicit loop rather than a fused column expression. Computing the three
 * output columns separately reads each input four times over three passes,
 * which measured slower than a single pass at every size tried: at 10,000 rows
 * the loop took roughly 18 microseconds against 25 for the column form, and at
 * 200,000 rows roughly 460 against 863. The loop also handles an input whose
 * column count is only known at run time, which the fixed three-column
 * expression form does not.
 *
 * Both arguments may be Eigen expressions, and they need not be the same kind
 * of expression. Each is bound once through @c Eigen::Ref, so blocks and
 * stored matrices are read in place while a computed expression is evaluated
 * a single time rather than once per coefficient access.
 *
 * @tparam DerivedA Eigen expression type of the left argument.
 * @tparam DerivedB Eigen expression type of the right argument; must share a
 *         scalar type with @p DerivedA.
 * @param left First stack of row vectors; must have exactly three columns.
 * @param right Second stack of row vectors; must have the same shape as
 *        @p left.
 * @return A three-column stack of per-row cross products.
 *
 * @note Each result row is conjugated when the scalar is complex, matching
 *       @c Eigen::MatrixBase::cross. For real scalars this is the identity.
 *
 * @note The compile-time row count of the result follows @p left, so pairing a
 *       dynamic left argument with a fixed-size right one yields a dynamic
 *       result. The runtime shapes are still checked against each other.
 */
template<typename DerivedA, typename DerivedB>
detail::CrossProductType<DerivedA> batched_cross_product(
    const Eigen::MatrixBase<DerivedA>& left, const Eigen::MatrixBase<DerivedB>& right
)
{
    using ScalarType = typename DerivedA::Scalar;

    static_assert(
        std::is_same_v<ScalarType, typename DerivedB::Scalar>,
        "batched_cross_product requires both arguments to share a scalar type"
    );
    // Cast to int: the two expressions carry distinct unnamed enum types, and
    // comparing them directly draws -Wenum-compare.
    static_assert(
        int(DerivedA::ColsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedA::ColsAtCompileTime) == 3,
        "batched_cross_product requires exactly three columns"
    );
    static_assert(
        int(DerivedB::ColsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedB::ColsAtCompileTime) == 3,
        "batched_cross_product requires exactly three columns"
    );
    static_assert(
        int(DerivedA::RowsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedB::RowsAtCompileTime) == int(Eigen::Dynamic)
            or int(DerivedA::RowsAtCompileTime) == int(DerivedB::RowsAtCompileTime),
        "batched_cross_product arguments disagree on their row count"
    );

    utils::nice_assert(
        left.rows() == right.rows(),
        "batched_cross_product requires both arguments to have the same row count"
    );
    utils::nice_assert(
        left.cols() == 3 and right.cols() == 3,
        "batched_cross_product requires exactly three columns"
    );

    // Bound once so that coefficient access below reads memory directly. A
    // stored matrix or block is read in place; a computed expression is
    // evaluated here rather than on every one of the six accesses per row.
    const Eigen::Ref<const typename DerivedA::PlainObject> a(left.derived());
    const Eigen::Ref<const typename DerivedB::PlainObject> b(right.derived());

    // Sized by resize rather than the two-argument constructor, which for some
    // small fixed shapes builds a vector from its arguments instead.
    detail::CrossProductType<DerivedA> result;
    result.resize(a.rows(), 3);

    for (Eigen::Index idx = 0; idx < a.rows(); ++idx)
    {
        result(idx, 0) = Eigen::numext::conj(
            a(idx, 1) * b(idx, 2) - a(idx, 2) * b(idx, 1));
        result(idx, 1) = Eigen::numext::conj(
            a(idx, 2) * b(idx, 0) - a(idx, 0) * b(idx, 2));
        result(idx, 2) = Eigen::numext::conj(
            a(idx, 0) * b(idx, 1) - a(idx, 1) * b(idx, 0));
    }
    return result;
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
