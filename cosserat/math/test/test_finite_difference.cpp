#include <cosserat/math/finite_difference.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <type_traits>

namespace cosserat::math {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kTol = 1e-12;

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return ::testing::AssertionFailure()
            << "shape mismatch: " << a.rows() << "x" << a.cols() << " vs "
            << b.rows() << "x" << b.cols();
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err < tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs diff " << err << " >= " << tol;
}

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    return Near(a, b, kTol);
}

/** Reference implementation, written as an explicit loop. */
Eigen::MatrixXd loop_difference(const Eigen::MatrixXd& matrix)
{
    Eigen::MatrixXd result(matrix.rows() - 1, matrix.cols());
    for (Eigen::Index i = 0; i + 1 < matrix.rows(); ++i)
    {
        result.row(i) = matrix.row(i + 1) - matrix.row(i);
    }
    return result;
}

/** Reference implementation, written as an explicit loop. */
Eigen::MatrixXd loop_average(const Eigen::MatrixXd& matrix)
{
    Eigen::MatrixXd result(matrix.rows() - 1, matrix.cols());
    for (Eigen::Index i = 0; i + 1 < matrix.rows(); ++i)
    {
        result.row(i) = 0.5 * (matrix.row(i + 1) + matrix.row(i));
    }
    return result;
}

Eigen::MatrixXd make_matrix()
{
    Eigen::MatrixXd matrix(4, 3);
    matrix << 1.0, 2.0, 3.0,
              5.0, 8.0, 13.0,
              2.0, -4.0, 6.0,
              0.0, 0.5, -1.5;
    return matrix;
}

// ---------------------------------------------------------------------------
// row_difference: values
// ---------------------------------------------------------------------------

TEST(RowDifference, SubtractsEachRowFromTheNext)
{
    Eigen::MatrixXd matrix(3, 2);
    matrix << 1.0, 2.0,
              4.0, 8.0,
              9.0, 27.0;

    Eigen::MatrixXd expected(2, 2);
    expected << 3.0, 6.0,
                5.0, 19.0;

    EXPECT_TRUE(Near(row_difference(matrix), expected));
}

TEST(RowDifference, DropsExactlyOneRow)
{
    for (Eigen::Index rows : {2, 3, 7, 50})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 4);
        const auto result = row_difference(matrix);

        EXPECT_EQ(result.rows(), rows - 1) << "rows " << rows;
        EXPECT_EQ(result.cols(), 4) << "rows " << rows;
    }
}

TEST(RowDifference, MatchesAnExplicitLoop)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(9, 5);
    EXPECT_TRUE(Near(row_difference(matrix), loop_difference(matrix)));
}

TEST(RowDifference, ConstantRowsGiveZero)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(6, 3) * 4.25;
    EXPECT_TRUE(Near(row_difference(matrix), Eigen::MatrixXd::Zero(5, 3)));
}

// A row index ramp differences to a constant, which is the discrete analogue
// of differentiating a linear function.
TEST(RowDifference, LinearRampGivesAConstant)
{
    Eigen::MatrixXd matrix(5, 2);
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        matrix.row(i) << 2.0 * static_cast<double>(i) + 1.0,
                         -3.0 * static_cast<double>(i);
    }

    Eigen::MatrixXd expected(4, 2);
    for (Eigen::Index i = 0; i < 4; ++i) expected.row(i) << 2.0, -3.0;

    EXPECT_TRUE(Near(row_difference(matrix), expected));
}

TEST(RowDifference, TwoRowsGiveASingleRow)
{
    Eigen::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0,
              4.0, 6.0, 8.0;

    const auto result = row_difference(matrix);

    ASSERT_EQ(result.rows(), 1);
    EXPECT_TRUE(Near(result, Eigen::RowVector3d(3.0, 4.0, 5.0)));
}

TEST(RowDifference, WorksOnASingleColumn)
{
    Eigen::VectorXd vector(4);
    vector << 1.0, 3.0, 6.0, 10.0;

    Eigen::VectorXd expected(3);
    expected << 2.0, 3.0, 4.0;

    EXPECT_TRUE(Near(row_difference(vector), expected));
}

TEST(RowDifference, WorksOnManyColumns)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(3, 64);
    const auto result = row_difference(matrix);

    EXPECT_EQ(result.cols(), 64);
    EXPECT_TRUE(Near(result, loop_difference(matrix)));
}

TEST(RowDifference, DoesNotModifyItsInput)
{
    const Eigen::MatrixXd matrix = make_matrix();
    Eigen::MatrixXd copy = matrix;

    const auto result = row_difference(copy);
    (void)result;

    EXPECT_TRUE(Near(copy, matrix));
}

// ---------------------------------------------------------------------------
// row_average: values
// ---------------------------------------------------------------------------

TEST(RowAverage, HalvesTheSumOfEachAdjacentPair)
{
    Eigen::MatrixXd matrix(3, 2);
    matrix << 1.0, 2.0,
              3.0, 8.0,
              5.0, 0.0;

    Eigen::MatrixXd expected(2, 2);
    expected << 2.0, 5.0,
                4.0, 4.0;

    EXPECT_TRUE(Near(row_average(matrix), expected));
}

TEST(RowAverage, DropsExactlyOneRow)
{
    for (Eigen::Index rows : {2, 3, 7, 50})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 4);
        const auto result = row_average(matrix);

        EXPECT_EQ(result.rows(), rows - 1) << "rows " << rows;
        EXPECT_EQ(result.cols(), 4) << "rows " << rows;
    }
}

TEST(RowAverage, MatchesAnExplicitLoop)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(9, 5);
    EXPECT_TRUE(Near(row_average(matrix), loop_average(matrix)));
}

TEST(RowAverage, ConstantRowsArePreserved)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(6, 3) * 4.25;
    EXPECT_TRUE(Near(row_average(matrix), Eigen::MatrixXd::Ones(5, 3) * 4.25));
}

// Averaging a ramp gives the midpoints, offset by half a step.
TEST(RowAverage, LinearRampGivesMidpoints)
{
    Eigen::VectorXd vector(5);
    vector << 0.0, 2.0, 4.0, 6.0, 8.0;

    Eigen::VectorXd expected(4);
    expected << 1.0, 3.0, 5.0, 7.0;

    EXPECT_TRUE(Near(row_average(vector), expected));
}

TEST(RowAverage, ResultIsBoundedByItsNeighbours)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(8, 3);
    const Eigen::MatrixXd result = row_average(matrix);

    for (Eigen::Index i = 0; i < result.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < result.cols(); ++j)
        {
            const double lower = std::min(matrix(i, j), matrix(i + 1, j));
            const double upper = std::max(matrix(i, j), matrix(i + 1, j));
            EXPECT_GE(result(i, j), lower - kTol) << "at " << i << "," << j;
            EXPECT_LE(result(i, j), upper + kTol) << "at " << i << "," << j;
        }
    }
}

TEST(RowAverage, WorksOnASingleColumn)
{
    Eigen::VectorXd vector(3);
    vector << 1.0, 3.0, 11.0;

    Eigen::VectorXd expected(2);
    expected << 2.0, 7.0;

    EXPECT_TRUE(Near(row_average(vector), expected));
}

TEST(RowAverage, DoesNotModifyItsInput)
{
    const Eigen::MatrixXd matrix = make_matrix();
    Eigen::MatrixXd copy = matrix;

    const auto result = row_average(copy);
    (void)result;

    EXPECT_TRUE(Near(copy, matrix));
}

// ---------------------------------------------------------------------------
// Relationship between the two operators
// ---------------------------------------------------------------------------

// Summing the midpoint and half the difference recovers the later row, which
// ties the two operators together on the same input.
TEST(FiniteDifference, AverageAndDifferenceRecoverTheTrailingRows)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(7, 4);

    const Eigen::MatrixXd recovered =
        row_average(matrix) + 0.5 * row_difference(matrix);

    EXPECT_TRUE(Near(recovered, matrix.bottomRows(6)));
}

TEST(FiniteDifference, AverageMinusHalfDifferenceRecoversTheLeadingRows)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(7, 4);

    const Eigen::MatrixXd recovered =
        row_average(matrix) - 0.5 * row_difference(matrix);

    EXPECT_TRUE(Near(recovered, matrix.topRows(6)));
}

// Differencing repeatedly is the discrete second derivative, and reduces the
// row count by one each time.
TEST(FiniteDifference, RepeatedDifferenceReducesRowsEachTime)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(5, 2);

    const Eigen::MatrixXd first = row_difference(matrix);
    const Eigen::MatrixXd second = row_difference(first);

    EXPECT_EQ(first.rows(), 4);
    EXPECT_EQ(second.rows(), 3);
    EXPECT_TRUE(Near(second, loop_difference(first)));
}

TEST(FiniteDifference, SecondDifferenceOfALinearRampIsZero)
{
    Eigen::MatrixXd matrix(6, 2);
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        matrix.row(i) << 3.0 * static_cast<double>(i), 1.0 - static_cast<double>(i);
    }

    const Eigen::MatrixXd second = row_difference(Eigen::MatrixXd(row_difference(matrix)));

    EXPECT_TRUE(Near(second, Eigen::MatrixXd::Zero(4, 2)));
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceTypes, FixedSizeInputsGiveFixedSizeResults)
{
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::Matrix<double, 4, 3>>())),
        Eigen::Matrix<double, 3, 3>>);
    static_assert(std::is_same_v<
        decltype(row_average(std::declval<Eigen::Matrix<double, 4, 3>>())),
        Eigen::Matrix<double, 3, 3>>);
    SUCCEED();
}

// Eigen requires a fixed single-row matrix with several columns to be row
// major, so the reduced type cannot simply inherit the input's options.
TEST(FiniteDifferenceTypes, TwoRowInputGivesALegalSingleRowResult)
{
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::Matrix<double, 2, 3>>())),
        Eigen::Matrix<double, 1, 3>>);

    Eigen::Matrix<double, 2, 3> matrix;
    matrix << 1.0, 2.0, 3.0,
              4.0, 6.0, 8.0;
    const Eigen::Matrix<double, 1, 3> result = row_difference(matrix);

    EXPECT_TRUE(Near(result, Eigen::RowVector3d(3.0, 4.0, 5.0)));
    SUCCEED();
}

TEST(FiniteDifferenceTypes, FixedVectorsStayColumnVectors)
{
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::Vector3d>())),
        Eigen::Vector2d>);
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::Matrix<double, 2, 1>>())),
        Eigen::Matrix<double, 1, 1>>);
    SUCCEED();
}

TEST(FiniteDifferenceTypes, DynamicInputsGiveDynamicResults)
{
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::MatrixXd>())), Eigen::MatrixXd>);
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::VectorXd>())), Eigen::VectorXd>);
    SUCCEED();
}

TEST(FiniteDifferenceTypes, StorageOrderIsPreservedWhereLegal)
{
    using RowMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static_assert(decltype(row_difference(std::declval<RowMajorMatrix>()))::IsRowMajor);
    static_assert(not decltype(row_difference(
        std::declval<Eigen::MatrixXd>()))::IsRowMajor);
    SUCCEED();
}

// Storage order affects layout, never the computed values.
TEST(FiniteDifferenceTypes, StorageOrderDoesNotChangeTheValues)
{
    Eigen::MatrixXd column_major(4, 3);
    column_major << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        row_major = column_major;

    EXPECT_TRUE(Near(row_difference(column_major),
                     Eigen::MatrixXd(row_difference(row_major))));
    EXPECT_TRUE(Near(row_average(column_major),
                     Eigen::MatrixXd(row_average(row_major))));
}

TEST(FiniteDifferenceTypes, ScalarTypeFollowsTheInput)
{
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::Matrix<float, 4, 2>>())),
        Eigen::Matrix<float, 3, 2>>);
    static_assert(std::is_same_v<
        decltype(row_average(std::declval<Eigen::Matrix<float, 4, 2>>())),
        Eigen::Matrix<float, 3, 2>>);
    static_assert(std::is_same_v<
        decltype(row_difference(std::declval<Eigen::MatrixXf>())), Eigen::MatrixXf>);
    SUCCEED();
}

TEST(FiniteDifferenceTypes, FloatResultsAreCorrect)
{
    Eigen::Matrix<float, 3, 2> matrix;
    matrix << 1.0f, 2.0f,
              3.0f, 8.0f,
              5.0f, 0.0f;

    const Eigen::Matrix<float, 2, 2> difference = row_difference(matrix);
    const Eigen::Matrix<float, 2, 2> average = row_average(matrix);

    EXPECT_FLOAT_EQ(difference(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(difference(1, 1), -8.0f);
    EXPECT_FLOAT_EQ(average(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(average(1, 0), 4.0f);
}

// ---------------------------------------------------------------------------
// Expression inputs
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceExpressions, AcceptsBlockExpressions)
{
    const Eigen::MatrixXd matrix = make_matrix();

    EXPECT_TRUE(Near(row_difference(matrix.topRows(3)),
                     loop_difference(Eigen::MatrixXd(matrix.topRows(3)))));
    EXPECT_TRUE(Near(row_average(matrix.leftCols(2)),
                     loop_average(Eigen::MatrixXd(matrix.leftCols(2)))));
}

TEST(FiniteDifferenceExpressions, AcceptsArithmeticExpressions)
{
    const Eigen::MatrixXd matrix = make_matrix();
    const Eigen::MatrixXd doubled = matrix * 2.0;

    EXPECT_TRUE(Near(row_difference(matrix + matrix), loop_difference(doubled)));
}

TEST(FiniteDifferenceExpressions, AcceptsTransposeExpressions)
{
    Eigen::MatrixXd matrix(2, 5);
    matrix << 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10;

    // The transpose is 5x2, so differencing it walks down the original columns.
    const Eigen::MatrixXd result = row_difference(matrix.transpose());

    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_TRUE(Near(result, Eigen::MatrixXd::Ones(4, 2)));
}

// Differencing a column of the input matches differencing the whole matrix and
// then taking the same column.
TEST(FiniteDifferenceExpressions, ColumnwiseAgreesWithWholeMatrix)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(6, 3);
    const Eigen::MatrixXd whole = row_difference(matrix);

    for (Eigen::Index column = 0; column < matrix.cols(); ++column)
    {
        EXPECT_TRUE(Near(row_difference(matrix.col(column)), whole.col(column)))
            << "column " << column;
    }
}

// ---------------------------------------------------------------------------
// Failure modes
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceDeathTest, RejectsFewerThanTwoRows)
{
    const Eigen::MatrixXd single = Eigen::MatrixXd::Ones(1, 3);
    const Eigen::MatrixXd empty = Eigen::MatrixXd(0, 3);

    EXPECT_ASSERT_FAILURE(row_difference(single));
    EXPECT_ASSERT_FAILURE(row_average(single));
    EXPECT_ASSERT_FAILURE(row_difference(empty));
    EXPECT_ASSERT_FAILURE(row_average(empty));
}

TEST(FiniteDifference, AcceptsExactlyTwoRows)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(2, 3);

    EXPECT_NO_THROW({ row_difference(matrix); });
    EXPECT_NO_THROW({ row_average(matrix); });
}

// A dynamic matrix with zero columns still has rows to difference, and the
// result is simply empty in the column direction.
TEST(FiniteDifference, HandlesZeroColumns)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd(4, 0);
    const auto result = row_difference(matrix);

    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 0);
}

// ---------------------------------------------------------------------------
// row_difference_kernel: values
//
// These are the discrete operators the PyElastica governing equations are
// written in terms of, where they appear as the difference and quadrature
// kernels. Each maps an N by C matrix to an (N+1) by C one.
// ---------------------------------------------------------------------------

/** Reference implementation, written as an explicit loop. */
Eigen::MatrixXd loop_difference_kernel(const Eigen::MatrixXd& matrix)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(matrix.rows() + 1, matrix.cols());
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    {
        result.row(i) += matrix.row(i);
        result.row(i + 1) -= matrix.row(i);
    }
    return result;
}

/** Reference implementation, written as an explicit loop. */
Eigen::MatrixXd loop_average_kernel(const Eigen::MatrixXd& matrix)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(matrix.rows() + 1, matrix.cols());
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    {
        result.row(i) += 0.5 * matrix.row(i);
        result.row(i + 1) += 0.5 * matrix.row(i);
    }
    return result;
}

TEST(RowDifferenceKernel, MatchesTheExpectedLayout)
{
    Eigen::MatrixXd matrix(3, 2);
    matrix << 1.0, 2.0,
              4.0, 8.0,
              9.0, 27.0;

    Eigen::MatrixXd expected(4, 2);
    expected <<  1.0,   2.0,   // first row copied
                 3.0,   6.0,   // differences
                 5.0,  19.0,
                -9.0, -27.0;   // last row negated

    EXPECT_TRUE(Near(row_difference_kernel(matrix), expected));
}

TEST(RowDifferenceKernel, AddsExactlyOneRow)
{
    for (Eigen::Index rows : {1, 2, 3, 7, 50})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 4);
        const auto result = row_difference_kernel(matrix);

        EXPECT_EQ(result.rows(), rows + 1) << "rows " << rows;
        EXPECT_EQ(result.cols(), 4) << "rows " << rows;
    }
}

TEST(RowDifferenceKernel, MatchesAnExplicitLoop)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(9, 5);
    EXPECT_TRUE(Near(row_difference_kernel(matrix), loop_difference_kernel(matrix)));
}

TEST(RowDifferenceKernel, FirstRowIsCopiedAndLastIsNegated)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(6, 3);
    const Eigen::MatrixXd result = row_difference_kernel(matrix);

    EXPECT_TRUE(Near(result.row(0), matrix.row(0)));
    EXPECT_TRUE(Near(result.row(result.rows() - 1),
                     Eigen::MatrixXd(-matrix.row(matrix.rows() - 1))));
}

// The interior is exactly the plain operator, which is what makes the kernel
// an extension of it rather than a separate rule.
TEST(RowDifferenceKernel, InteriorRowsMatchRowDifference)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(8, 3);
    const Eigen::MatrixXd result = row_difference_kernel(matrix);

    EXPECT_TRUE(Near(result.middleRows(1, matrix.rows() - 1), row_difference(matrix)));
}

// The sum telescopes, so internal forces distributed onto nodes cancel.
TEST(RowDifferenceKernel, SumsToZeroDownItsRows)
{
    for (Eigen::Index rows : {1, 2, 5, 20})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 3);
        const Eigen::MatrixXd result = row_difference_kernel(matrix);

        EXPECT_LT(result.colwise().sum().cwiseAbs().maxCoeff(), 1e-12)
            << "rows " << rows;
    }
}

TEST(RowDifferenceKernel, ConstantInputGivesOnlyEndContributions)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(4, 2) * 3.0;
    const Eigen::MatrixXd result = row_difference_kernel(matrix);

    ASSERT_EQ(result.rows(), 5);
    EXPECT_TRUE(Near(result.row(0), Eigen::RowVector2d(3.0, 3.0)));
    EXPECT_TRUE(Near(result.middleRows(1, 3), Eigen::MatrixXd::Zero(3, 2)));
    EXPECT_TRUE(Near(result.row(4), Eigen::RowVector2d(-3.0, -3.0)));
}

// With one row there is no interior, so the result is the row and its negation.
TEST(RowDifferenceKernel, SingleRowGivesTheRowAndItsNegation)
{
    Eigen::MatrixXd matrix(1, 3);
    matrix << 1.0, -2.0, 3.5;

    const Eigen::MatrixXd result = row_difference_kernel(matrix);

    ASSERT_EQ(result.rows(), 2);
    EXPECT_TRUE(Near(result.row(0), matrix.row(0)));
    EXPECT_TRUE(Near(result.row(1), Eigen::MatrixXd(-matrix.row(0))));
}

TEST(RowDifferenceKernel, WorksOnASingleColumn)
{
    Eigen::VectorXd vector(3);
    vector << 1.0, 3.0, 6.0;

    Eigen::VectorXd expected(4);
    expected << 1.0, 2.0, 3.0, -6.0;

    EXPECT_TRUE(Near(row_difference_kernel(vector), expected));
}

TEST(RowDifferenceKernel, DoesNotModifyItsInput)
{
    const Eigen::MatrixXd matrix = make_matrix();
    Eigen::MatrixXd copy = matrix;

    const auto result = row_difference_kernel(copy);
    (void)result;

    EXPECT_TRUE(Near(copy, matrix));
}

// ---------------------------------------------------------------------------
// row_average_kernel: values
// ---------------------------------------------------------------------------

TEST(RowAverageKernel, MatchesTheExpectedLayout)
{
    Eigen::MatrixXd matrix(3, 2);
    matrix << 1.0, 2.0,
              3.0, 8.0,
              5.0, 0.0;

    Eigen::MatrixXd expected(4, 2);
    expected << 0.5, 1.0,   // half the first row
                2.0, 5.0,   // midpoints
                4.0, 4.0,
                2.5, 0.0;   // half the last row

    EXPECT_TRUE(Near(row_average_kernel(matrix), expected));
}

TEST(RowAverageKernel, AddsExactlyOneRow)
{
    for (Eigen::Index rows : {1, 2, 3, 7, 50})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 4);
        const auto result = row_average_kernel(matrix);

        EXPECT_EQ(result.rows(), rows + 1) << "rows " << rows;
        EXPECT_EQ(result.cols(), 4) << "rows " << rows;
    }
}

TEST(RowAverageKernel, MatchesAnExplicitLoop)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(9, 5);
    EXPECT_TRUE(Near(row_average_kernel(matrix), loop_average_kernel(matrix)));
}

TEST(RowAverageKernel, EndRowsAreHalfTheEndsOfTheInput)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(6, 3);
    const Eigen::MatrixXd result = row_average_kernel(matrix);

    EXPECT_TRUE(Near(result.row(0), Eigen::MatrixXd(0.5 * matrix.row(0))));
    EXPECT_TRUE(Near(result.row(result.rows() - 1),
                     Eigen::MatrixXd(0.5 * matrix.row(matrix.rows() - 1))));
}

TEST(RowAverageKernel, InteriorRowsMatchRowAverage)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(8, 3);
    const Eigen::MatrixXd result = row_average_kernel(matrix);

    EXPECT_TRUE(Near(result.middleRows(1, matrix.rows() - 1), row_average(matrix)));
}

// Each input row contributes half its value to two output rows, so the total
// is preserved. This is what makes the trapezoidal quadrature conservative.
TEST(RowAverageKernel, PreservesTheColumnTotals)
{
    for (Eigen::Index rows : {1, 2, 5, 20})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 3);
        const Eigen::MatrixXd result = row_average_kernel(matrix);

        EXPECT_TRUE(Near(Eigen::MatrixXd(result.colwise().sum()),
                         Eigen::MatrixXd(matrix.colwise().sum()), 1e-12))
            << "rows " << rows;
    }
}

TEST(RowAverageKernel, ConstantInputGivesHalvedEnds)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Ones(4, 2) * 3.0;
    const Eigen::MatrixXd result = row_average_kernel(matrix);

    ASSERT_EQ(result.rows(), 5);
    EXPECT_TRUE(Near(result.row(0), Eigen::RowVector2d(1.5, 1.5)));
    EXPECT_TRUE(Near(result.middleRows(1, 3), Eigen::MatrixXd::Ones(3, 2) * 3.0));
    EXPECT_TRUE(Near(result.row(4), Eigen::RowVector2d(1.5, 1.5)));
}

TEST(RowAverageKernel, SingleRowSplitsEvenlyAcrossBothOutputRows)
{
    Eigen::MatrixXd matrix(1, 3);
    matrix << 1.0, -2.0, 3.5;

    const Eigen::MatrixXd result = row_average_kernel(matrix);

    ASSERT_EQ(result.rows(), 2);
    EXPECT_TRUE(Near(result.row(0), Eigen::MatrixXd(0.5 * matrix.row(0))));
    EXPECT_TRUE(Near(result.row(1), Eigen::MatrixXd(0.5 * matrix.row(0))));
}

TEST(RowAverageKernel, IsNeverNegativeForNonNegativeInput)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(7, 3).cwiseAbs();
    const Eigen::MatrixXd result = row_average_kernel(matrix);

    EXPECT_TRUE((result.array() >= 0.0).all());
}

TEST(RowAverageKernel, DoesNotModifyItsInput)
{
    const Eigen::MatrixXd matrix = make_matrix();
    Eigen::MatrixXd copy = matrix;

    const auto result = row_average_kernel(copy);
    (void)result;

    EXPECT_TRUE(Near(copy, matrix));
}

// ---------------------------------------------------------------------------
// Adjoint relationships between the plain operators and the kernels
//
// The kernels are the adjoints of the plain operators, up to a sign for the
// difference. This is the property that makes the discretisation conservative,
// and it pins the end rows just as firmly as the interior ones.
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceKernels, DifferenceKernelIsTheNegatedAdjointOfRowDifference)
{
    const Eigen::Index rows = 6;
    const Eigen::MatrixXd nodal = Eigen::MatrixXd::Random(rows + 1, 3);
    const Eigen::MatrixXd element = Eigen::MatrixXd::Random(rows, 3);

    // <D u, v> == -<u, Dk v>
    const double left = (row_difference(nodal).array() * element.array()).sum();
    const double right =
        -(nodal.array() * row_difference_kernel(element).array()).sum();

    EXPECT_NEAR(left, right, 1e-12);
}

TEST(FiniteDifferenceKernels, AverageKernelIsTheAdjointOfRowAverage)
{
    const Eigen::Index rows = 6;
    const Eigen::MatrixXd nodal = Eigen::MatrixXd::Random(rows + 1, 3);
    const Eigen::MatrixXd element = Eigen::MatrixXd::Random(rows, 3);

    // <A u, v> == <u, Ak v>
    const double left = (row_average(nodal).array() * element.array()).sum();
    const double right = (nodal.array() * row_average_kernel(element).array()).sum();

    EXPECT_NEAR(left, right, 1e-12);
}

TEST(FiniteDifferenceKernels, AdjointRelationsHoldAcrossSizes)
{
    for (Eigen::Index rows : {1, 2, 3, 9})
    {
        const Eigen::MatrixXd nodal = Eigen::MatrixXd::Random(rows + 1, 2);
        const Eigen::MatrixXd element = Eigen::MatrixXd::Random(rows, 2);

        EXPECT_NEAR((row_difference(nodal).array() * element.array()).sum(),
                    -(nodal.array() * row_difference_kernel(element).array()).sum(),
                    1e-12)
            << "rows " << rows;
        EXPECT_NEAR((row_average(nodal).array() * element.array()).sum(),
                    (nodal.array() * row_average_kernel(element).array()).sum(),
                    1e-12)
            << "rows " << rows;
    }
}

// Applying the kernel then the plain operator returns to the element grid.
TEST(FiniteDifferenceKernels, KernelThenPlainOperatorRestoresTheRowCount)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(5, 3);

    const Eigen::MatrixXd there = row_difference_kernel(matrix);
    const Eigen::MatrixXd back = row_difference(there);

    EXPECT_EQ(there.rows(), 6);
    EXPECT_EQ(back.rows(), 5);
}

// ---------------------------------------------------------------------------
// Kernel result types
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceKernelTypes, FixedSizeInputsGainExactlyOneRow)
{
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::Matrix<double, 3, 4>>())),
        Eigen::Matrix<double, 4, 4>>);
    static_assert(std::is_same_v<
        decltype(row_average_kernel(std::declval<Eigen::Matrix<double, 3, 4>>())),
        Eigen::Matrix<double, 4, 4>>);
    SUCCEED();
}

TEST(FiniteDifferenceKernelTypes, FixedVectorsStayColumnVectors)
{
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::Vector3d>())),
        Eigen::Vector4d>);
    static_assert(std::is_same_v<
        decltype(row_average_kernel(std::declval<Eigen::Vector3d>())),
        Eigen::Vector4d>);
    SUCCEED();
}

// A one-by-one input expands to a two-element column, which is the case where
// Eigen's two-argument constructor would have built a vector of values
// instead of sizing the result.
TEST(FiniteDifferenceKernelTypes, SingleElementInputGivesATwoElementColumn)
{
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::Matrix<double, 1, 1>>())),
        Eigen::Matrix<double, 2, 1>>);

    Eigen::Matrix<double, 1, 1> matrix;
    matrix << 5.0;

    const Eigen::Matrix<double, 2, 1> difference = row_difference_kernel(matrix);
    const Eigen::Matrix<double, 2, 1> average = row_average_kernel(matrix);

    EXPECT_DOUBLE_EQ(difference(0), 5.0);
    EXPECT_DOUBLE_EQ(difference(1), -5.0);
    EXPECT_DOUBLE_EQ(average(0), 2.5);
    EXPECT_DOUBLE_EQ(average(1), 2.5);
}

TEST(FiniteDifferenceKernelTypes, DynamicInputsGiveDynamicResults)
{
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::MatrixXd>())),
        Eigen::MatrixXd>);
    static_assert(std::is_same_v<
        decltype(row_average_kernel(std::declval<Eigen::VectorXd>())),
        Eigen::VectorXd>);
    SUCCEED();
}

// A single-row input is row major, and two rows by several columns permits
// either order, so the input's order carries through.
TEST(FiniteDifferenceKernelTypes, StorageOrderIsPreservedWhereLegal)
{
    using RowMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static_assert(
        decltype(row_difference_kernel(std::declval<RowMajorMatrix>()))::IsRowMajor);
    static_assert(not decltype(row_difference_kernel(
        std::declval<Eigen::MatrixXd>()))::IsRowMajor);
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::RowVector3d>())),
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>);
    SUCCEED();
}

TEST(FiniteDifferenceKernelTypes, StorageOrderDoesNotChangeTheValues)
{
    Eigen::MatrixXd column_major(4, 3);
    column_major << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        row_major = column_major;

    EXPECT_TRUE(Near(row_difference_kernel(column_major),
                     Eigen::MatrixXd(row_difference_kernel(row_major))));
    EXPECT_TRUE(Near(row_average_kernel(column_major),
                     Eigen::MatrixXd(row_average_kernel(row_major))));
}

TEST(FiniteDifferenceKernelTypes, ScalarTypeFollowsTheInput)
{
    static_assert(std::is_same_v<
        decltype(row_difference_kernel(std::declval<Eigen::Matrix<float, 4, 2>>())),
        Eigen::Matrix<float, 5, 2>>);
    static_assert(std::is_same_v<
        decltype(row_average_kernel(std::declval<Eigen::MatrixXf>())),
        Eigen::MatrixXf>);

    Eigen::Matrix<float, 2, 2> matrix;
    matrix << 2.0f, 4.0f,
              6.0f, 8.0f;
    const Eigen::Matrix<float, 3, 2> average = row_average_kernel(matrix);

    EXPECT_FLOAT_EQ(average(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(average(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(average(2, 0), 3.0f);
}

// ---------------------------------------------------------------------------
// Kernel expression inputs and failure modes
// ---------------------------------------------------------------------------

TEST(FiniteDifferenceKernelExpressions, AcceptsBlockExpressions)
{
    const Eigen::MatrixXd matrix = make_matrix();

    EXPECT_TRUE(Near(row_difference_kernel(matrix.topRows(3)),
                     loop_difference_kernel(Eigen::MatrixXd(matrix.topRows(3)))));
    EXPECT_TRUE(Near(row_average_kernel(matrix.leftCols(2)),
                     loop_average_kernel(Eigen::MatrixXd(matrix.leftCols(2)))));
}

TEST(FiniteDifferenceKernelExpressions, AcceptsArithmeticExpressions)
{
    const Eigen::MatrixXd matrix = make_matrix();
    const Eigen::MatrixXd doubled = matrix * 2.0;

    EXPECT_TRUE(Near(row_difference_kernel(matrix + matrix),
                     loop_difference_kernel(doubled)));
    EXPECT_TRUE(Near(row_average_kernel(matrix + matrix),
                     loop_average_kernel(doubled)));
}

TEST(FiniteDifferenceKernelExpressions, AcceptsTransposeExpressions)
{
    Eigen::MatrixXd matrix(2, 5);
    matrix << 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10;

    const Eigen::MatrixXd result = row_difference_kernel(matrix.transpose());

    EXPECT_EQ(result.rows(), 6);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_LT(result.colwise().sum().cwiseAbs().maxCoeff(), 1e-12);
}

// One row is the minimum, unlike the plain operators which need two.
TEST(FiniteDifferenceKernels, AcceptsExactlyOneRow)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(1, 3);

    EXPECT_NO_THROW({ row_difference_kernel(matrix); });
    EXPECT_NO_THROW({ row_average_kernel(matrix); });
}

TEST(FiniteDifferenceKernelDeathTest, RejectsAnEmptyInput)
{
    const Eigen::MatrixXd empty = Eigen::MatrixXd(0, 3);

    EXPECT_ASSERT_FAILURE(row_difference_kernel(empty));
    EXPECT_ASSERT_FAILURE(row_average_kernel(empty));
}

TEST(FiniteDifferenceKernels, HandlesZeroColumns)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd(4, 0);

    const auto difference = row_difference_kernel(matrix);
    const auto average = row_average_kernel(matrix);

    EXPECT_EQ(difference.rows(), 5);
    EXPECT_EQ(difference.cols(), 0);
    EXPECT_EQ(average.rows(), 5);
    EXPECT_EQ(average.cols(), 0);
}

}  // namespace
}  // namespace cosserat::math
