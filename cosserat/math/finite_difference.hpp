#pragma once

/**
 * @file finite_difference.hpp
 * @brief Row-wise finite difference and average operators for Eigen matrices.
 *
 * Two families live here, distinguished by which way they change the row count.
 *
 * The plain operators, @ref cosserat::math::row_difference and
 * @ref cosserat::math::row_average, map an @c N by @c C matrix to an
 * @c (N-1) by @c C one by combining each adjacent pair of rows. With one row
 * per node, they convert a nodal quantity into the corresponding per-element
 * quantity: the difference gives the change across each element, and the
 * average gives its midpoint value.
 *
 * @f[ \mathrm{difference}(A)_i = A_{i+1} - A_i, \qquad
 *     \mathrm{average}(A)_i = \tfrac{1}{2}\left(A_{i+1} + A_i\right) @f]
 *
 * The kernel operators, @ref cosserat::math::row_difference_kernel and
 * @ref cosserat::math::row_average_kernel, go the other way: an @c N by @c C
 * matrix becomes an @c (N+1) by @c C one, carrying a per-element quantity back
 * onto the nodes. They are the discrete operators the PyElastica governing
 * equations are written in terms of, where they appear as the difference and
 * quadrature kernels.
 *
 * The two families are adjoint to one another, which is what makes the
 * discretisation conservative. Writing @f$ D @f$ and @f$ A @f$ for the plain
 * operators and @f$ D_k @f$ and @f$ A_k @f$ for the kernels,
 *
 * @f[ \langle D u, v \rangle = -\langle u, D_k v \rangle, \qquad
 *     \langle A u, v \rangle = \langle u, A_k v \rangle. @f]
 *
 * In practice this means the difference kernel sums to zero down its rows, so
 * internal forces cancel exactly, while the average kernel preserves the total,
 * so a trapezoidal integral is conserved.
 *
 * Any column count is supported, fixed or dynamic, and the scalar type is
 * taken from the argument rather than fixed to @c double.
 *
 * All four accept Eigen expressions as well as stored matrices, so
 * @c row_difference(positions.topRows(5)) works without materialising the
 * block first. The result is a plain matrix whose storage order follows the
 * input wherever Eigen permits it.
 *
 * This header is complete on its own; there is no accompanying source file.
 */

#include <type_traits>

#include <Eigen/Core>

#include <cosserat/utils/assertions.hpp>

namespace cosserat::math {

namespace detail {

/**
 * @brief Plain matrix type produced by changing the row count by a fixed
 *        amount.
 *
 * Primary template is left undefined so that instantiating it with anything
 * other than an @c Eigen::Matrix is a compile error rather than a confusing
 * downstream failure.
 *
 * @tparam PlainType The @c Eigen::Matrix specialisation being adjusted.
 * @tparam RowDelta Rows added to the compile-time row count; negative to
 *         remove them.
 */
template<typename PlainType, int RowDelta>
struct RowCountAdjusted;

/**
 * @brief Specialisation computing the adjusted type for an @c Eigen::Matrix.
 *
 * The row count and maximum row count each shift by @p RowDelta, while the
 * scalar, the column count and the maximum column count carry over unchanged.
 * A dynamic row count stays dynamic.
 *
 * Storage options carry over too, except where the adjusted shape forbids it:
 * Eigen requires a single-row matrix with more than one column to be
 * row-major, and a single-column matrix with more than one row to be
 * column-major. Forcing the order in those two cases costs nothing, because a
 * matrix with one row or one column has the same layout either way.
 *
 * @tparam ScalarType Scalar of the matrix.
 * @tparam Rows Compile-time row count, possibly @c Eigen::Dynamic.
 * @tparam Cols Compile-time column count, possibly @c Eigen::Dynamic.
 * @tparam Options Eigen storage options.
 * @tparam MaxRows Eigen maximum row count.
 * @tparam MaxCols Eigen maximum column count.
 * @tparam RowDelta Rows added to the compile-time row count.
 */
template<
    typename ScalarType, int Rows, int Cols, int Options, int MaxRows, int MaxCols,
    int RowDelta>
struct RowCountAdjusted<
    Eigen::Matrix<ScalarType, Rows, Cols, Options, MaxRows, MaxCols>, RowDelta>
{
    /** @brief Row count after the adjustment. */
    static constexpr int adjusted_rows =
        (Rows == Eigen::Dynamic) ? Eigen::Dynamic : Rows + RowDelta;

    /** @brief Maximum row count after the adjustment. */
    static constexpr int adjusted_max_rows =
        (MaxRows == Eigen::Dynamic) ? Eigen::Dynamic : MaxRows + RowDelta;

    /** @brief Storage options legal for the adjusted shape. */
    static constexpr int adjusted_options =
        (adjusted_rows == 1 and Cols != 1)
            ? ((Options & ~int(Eigen::RowMajor)) | int(Eigen::RowMajor))
            : (Cols == 1 and adjusted_rows != 1)
                ? (Options & ~int(Eigen::RowMajor))
                : Options;

    /** @brief The adjusted matrix type. */
    using type = Eigen::Matrix<
        ScalarType, adjusted_rows, Cols, adjusted_options, adjusted_max_rows,
        MaxCols>;
};

/**
 * @brief Result type of a row-reducing operator applied to an Eigen
 *        expression.
 * @tparam Derived Eigen expression type.
 */
template<typename Derived>
using RowReducedType =
    typename RowCountAdjusted<typename Derived::PlainObject, -1>::type;

/**
 * @brief Result type of a row-expanding kernel applied to an Eigen expression.
 * @tparam Derived Eigen expression type.
 */
template<typename Derived>
using RowExpandedType =
    typename RowCountAdjusted<typename Derived::PlainObject, 1>::type;

} // End namespace detail

/**
 * @brief Difference between each adjacent pair of rows.
 *
 * Row @c i of the result is row @c i+1 of the input minus row @c i, so an
 * @c N by @c C input yields an @c (N-1) by @c C result. With one row per node
 * this gives the change across each element.
 *
 * @tparam Derived Eigen expression type; any scalar is accepted.
 * @param matrix Matrix or expression to difference; must have at least two
 *        rows.
 * @return A plain matrix of adjacent row differences.
 *
 * @note A fixed-size input with fewer than two rows is a compile error; a
 *       dynamic one is caught at run time.
 *
 * @see row_difference_kernel for the adjoint operator, which adds a row rather
 *      than removing one.
 */
template<typename Derived>
detail::RowReducedType<Derived> row_difference(const Eigen::MatrixBase<Derived>& matrix)
{
    static_assert(
        Derived::RowsAtCompileTime == Eigen::Dynamic
            or Derived::RowsAtCompileTime >= 2,
        "row_difference requires at least two rows"
    );

    utils::nice_assert(
        matrix.rows() >= 2, "row_difference requires at least two rows"
    );

    const Eigen::Index count = matrix.rows() - 1;
    return matrix.bottomRows(count) - matrix.topRows(count);
}

/**
 * @brief Midpoint of each adjacent pair of rows.
 *
 * Row @c i of the result is half the sum of rows @c i and @c i+1 of the input,
 * so an @c N by @c C input yields an @c (N-1) by @c C result. With one row per
 * node this gives the value at each element centre.
 *
 * @tparam Derived Eigen expression type; the scalar must not be an integral
 *         type, since halving would silently truncate.
 * @param matrix Matrix or expression to average; must have at least two rows.
 * @return A plain matrix of adjacent row midpoints.
 *
 * @note A fixed-size input with fewer than two rows is a compile error; a
 *       dynamic one is caught at run time.
 *
 * @see row_average_kernel for the adjoint operator, which adds a row rather
 *      than removing one.
 */
template<typename Derived>
detail::RowReducedType<Derived> row_average(const Eigen::MatrixBase<Derived>& matrix)
{
    using ScalarType = typename Derived::Scalar;

    static_assert(
        Derived::RowsAtCompileTime == Eigen::Dynamic
            or Derived::RowsAtCompileTime >= 2,
        "row_average requires at least two rows"
    );
    static_assert(
        not std::is_integral_v<ScalarType>,
        "row_average halves each sum, which would truncate for an integral scalar"
    );

    utils::nice_assert(
        matrix.rows() >= 2, "row_average requires at least two rows"
    );

    const Eigen::Index count = matrix.rows() - 1;
    return ScalarType(0.5) * (matrix.bottomRows(count) + matrix.topRows(count));
}

/**
 * @brief Difference kernel, carrying an element quantity onto the nodes.
 *
 * An @c N by @c C input yields an @c (N+1) by @c C result. The first row is
 * the first row of the input, the last row is the negation of the last row of
 * the input, and the rows between them are exactly @ref row_difference of the
 * input:
 *
 * @f[ \mathrm{result}_0 = A_0, \qquad
 *     \mathrm{result}_i = A_i - A_{i-1}, \qquad
 *     \mathrm{result}_N = -A_{N-1} @f]
 *
 * This is the difference kernel of the PyElastica governing equations. Because
 * the sum telescopes, the result always sums to zero down its rows, so a set
 * of internal element forces distributed onto the nodes cancels exactly.
 *
 * @tparam Derived Eigen expression type; any scalar is accepted.
 * @param matrix Matrix or expression to difference; must have at least one
 *        row.
 * @return A plain matrix with one more row than the input.
 *
 * @note A fixed-size input with no rows is a compile error; a dynamic one is
 *       caught at run time.
 *
 * @see row_difference, to which this operator is the negated adjoint.
 */
template<typename Derived>
detail::RowExpandedType<Derived> row_difference_kernel(
    const Eigen::MatrixBase<Derived>& matrix
)
{
    static_assert(
        Derived::RowsAtCompileTime == Eigen::Dynamic
            or Derived::RowsAtCompileTime >= 1,
        "row_difference_kernel requires at least one row"
    );

    utils::nice_assert(
        matrix.rows() >= 1, "row_difference_kernel requires at least one row"
    );

    const Eigen::Index count = matrix.rows();

    // Sized by resize rather than the two-argument constructor: for a fixed
    // two-element result that constructor builds a vector from the two
    // arguments instead of sizing the matrix.
    detail::RowExpandedType<Derived> result;
    result.resize(count + 1, matrix.cols());

    result.row(0) = matrix.row(0);
    result.middleRows(1, count - 1) =
        matrix.bottomRows(count - 1) - matrix.topRows(count - 1);
    result.row(count) = -matrix.row(count - 1);

    return result;
}

/**
 * @brief Average kernel, carrying an element quantity onto the nodes.
 *
 * An @c N by @c C input yields an @c (N+1) by @c C result. The first and last
 * rows are half the first and last rows of the input, and the rows between
 * them are exactly @ref row_average of the input:
 *
 * @f[ \mathrm{result}_0 = \tfrac{1}{2} A_0, \qquad
 *     \mathrm{result}_i = \tfrac{1}{2}\left(A_i + A_{i-1}\right), \qquad
 *     \mathrm{result}_N = \tfrac{1}{2} A_{N-1} @f]
 *
 * This is the quadrature kernel of the PyElastica governing equations, so
 * named because it is the trapezoidal rule written as a matrix. Every input
 * row contributes half its value to each of two output rows, so the result
 * sums to the same total as the input.
 *
 * @tparam Derived Eigen expression type; the scalar must not be an integral
 *         type, since halving would silently truncate.
 * @param matrix Matrix or expression to average; must have at least one row.
 * @return A plain matrix with one more row than the input.
 *
 * @note A fixed-size input with no rows is a compile error; a dynamic one is
 *       caught at run time.
 *
 * @see row_average, to which this operator is the adjoint.
 */
template<typename Derived>
detail::RowExpandedType<Derived> row_average_kernel(
    const Eigen::MatrixBase<Derived>& matrix
)
{
    using ScalarType = typename Derived::Scalar;

    static_assert(
        Derived::RowsAtCompileTime == Eigen::Dynamic
            or Derived::RowsAtCompileTime >= 1,
        "row_average_kernel requires at least one row"
    );
    static_assert(
        not std::is_integral_v<ScalarType>,
        "row_average_kernel halves each sum, which would truncate for an "
        "integral scalar"
    );

    utils::nice_assert(
        matrix.rows() >= 1, "row_average_kernel requires at least one row"
    );

    const Eigen::Index count = matrix.rows();

    // Sized by resize rather than the two-argument constructor: for a fixed
    // two-element result that constructor builds a vector from the two
    // arguments instead of sizing the matrix.
    detail::RowExpandedType<Derived> result;
    result.resize(count + 1, matrix.cols());

    const ScalarType half = ScalarType(0.5);
    result.row(0) = half * matrix.row(0);
    result.middleRows(1, count - 1) =
        half * (matrix.bottomRows(count - 1) + matrix.topRows(count - 1));
    result.row(count) = half * matrix.row(count - 1);

    return result;
}
} // End namespace cosserat::math
