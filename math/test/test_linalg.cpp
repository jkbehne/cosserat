#include "math/linalg.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <numbers>
#include <vector>

namespace cosserat::math {
namespace {

constexpr double kTol = 1e-12;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

template<int Dim>
using VecBatch = Eigen::Matrix<double, Dim, Eigen::Dynamic>;

// (a - b).norm() rather than isApprox: isApprox is relative and misbehaves
// when one operand is (near) zero.
template<typename A, typename B>
::testing::AssertionResult Near(const A& a, const B& b, double tol = kTol)
{
    const double err = (a - b).norm();
    if (err < tol)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "norm difference " << err << " >= " << tol;
}

// ---------------------------------------------------------------------------
// batched_matrix_vector
// ---------------------------------------------------------------------------

TEST(BatchedMatrixVector, ComputesEachProductIndependently)
{
    std::vector<Eigen::Matrix3d> mats(3);
    mats[0] << 1, 2, 3,
               4, 5, 6,
               7, 8, 10;
    mats[1] = Eigen::Matrix3d::Identity();
    mats[2] << 0, -1, 0,
               1,  0, 0,
               0,  0, 2;

    VecBatch<3> vecs(3, 3);
    vecs <<  1, 0,  2,
             0, 1, -1,
            -1, 2,  3;

    const auto result = batched_matrix_vector(mats, vecs);

    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(), mats[i] * vecs.col(i))) << "entry " << i;
    }
}

// Row i must be the i-th product, not a permutation or a broadcast.
TEST(BatchedMatrixVector, RowsAreOrderedAndDistinct)
{
    std::vector<Eigen::Matrix2d> mats{
        (Eigen::Matrix2d() << 1, 0, 0, 0).finished(),
        (Eigen::Matrix2d() << 0, 0, 0, 1).finished(),
    };

    VecBatch<2> vecs(2, 2);
    vecs << 5, 7,
            6, 8;

    const auto result = batched_matrix_vector(mats, vecs);

    EXPECT_TRUE(Near(result.row(0).transpose(), Eigen::Vector2d(5, 0)));
    EXPECT_TRUE(Near(result.row(1).transpose(), Eigen::Vector2d(0, 8)));
}

TEST(BatchedMatrixVector, OutputIsRowMajorTransposeOfColumnInput)
{
    std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());

    VecBatch<3> vecs(3, 2);
    vecs << 1, 4,
            2, 5,
            3, 6;

    const auto result = batched_matrix_vector(mats, vecs);

    ASSERT_EQ(result.rows(), 2);
    ASSERT_EQ(result.cols(), 3);
    EXPECT_TRUE(Near(result, vecs.transpose()));
}

TEST(BatchedMatrixVector, SingleEntry)
{
    std::vector<Eigen::Matrix3d> mats{2.0 * Eigen::Matrix3d::Identity()};

    VecBatch<3> vecs(3, 1);
    vecs << 1, -2, 0.5;

    const auto result = batched_matrix_vector(mats, vecs);

    ASSERT_EQ(result.rows(), 1);
    EXPECT_TRUE(Near(result.row(0).transpose(), Eigen::Vector3d(2, -4, 1)));
}

TEST(BatchedMatrixVector, EmptyInputYieldsEmptyResult)
{
    const std::vector<Eigen::Matrix3d> mats;
    const VecBatch<3> vecs(3, 0);

    const auto result = batched_matrix_vector(mats, vecs);

    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 3);
}

// Dim 4 is fixed-size vectorizable; exercises the std::vector alignment path.
TEST(BatchedMatrixVector, Dimension4)
{
    std::vector<Eigen::Matrix4d> mats(5);
    VecBatch<4> vecs(4, 5);
    for (int i = 0; i < 5; ++i)
    {
        mats[i] = Eigen::Matrix4d::Constant(i + 1.0);
        vecs.col(i) = Eigen::Vector4d::Constant(i + 1.0);
    }

    const auto result = batched_matrix_vector(mats, vecs);

    ASSERT_EQ(result.rows(), 5);
    ASSERT_EQ(result.cols(), 4);
    for (Eigen::Index i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(), mats[i] * vecs.col(i))) << "entry " << i;
    }
}

TEST(BatchedMatrixVector, DoesNotAliasInputVectors)
{
    std::vector<Eigen::Matrix3d> mats(2, 3.0 * Eigen::Matrix3d::Identity());

    VecBatch<3> vecs(3, 2);
    vecs << 1, 4,
            2, 5,
            3, 6;
    const VecBatch<3> vecs_before = vecs;

    const auto result = batched_matrix_vector(mats, vecs);

    EXPECT_TRUE(Near(vecs, vecs_before));
    EXPECT_TRUE(Near(result.row(0).transpose(), Eigen::Vector3d(3, 6, 9)));
}

TEST(BatchedMatrixVector, IgnoredMismatchTruncatesToFewerMatrices)
{
    std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());

    VecBatch<3> vecs(3, 5);
    vecs = VecBatch<3>::Random(3, 5);

    const auto result = batched_matrix_vector(mats, vecs, /*ignore_size_mismatch=*/true);

    ASSERT_EQ(result.rows(), 2);
    EXPECT_TRUE(Near(result, vecs.leftCols(2).transpose()));
}

TEST(BatchedMatrixVector, IgnoredMismatchTruncatesToFewerVectors)
{
    std::vector<Eigen::Matrix3d> mats(5, Eigen::Matrix3d::Identity());

    VecBatch<3> vecs(3, 2);
    vecs << 1, 4,
            2, 5,
            3, 6;

    const auto result = batched_matrix_vector(mats, vecs, /*ignore_size_mismatch=*/true);

    ASSERT_EQ(result.rows(), 2);
    EXPECT_TRUE(Near(result, vecs.transpose()));
}

// See note below: assumes nice_assert aborts and is live in this build config.
TEST(BatchedMatrixVectorDeathTest, MismatchWithoutOptOutFails)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    const std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());
    const VecBatch<3> vecs = VecBatch<3>::Zero(3, 5);

    EXPECT_DEATH({ batched_matrix_vector(mats, vecs); }, "");
}

// ---------------------------------------------------------------------------
// is_unit_vector
// ---------------------------------------------------------------------------

TEST(IsUnitVector, AcceptsAxisAlignedUnitVectors)
{
    EXPECT_TRUE(is_unit_vector(Eigen::Vector3d::UnitX(), 1e-12));
    EXPECT_TRUE(is_unit_vector(Eigen::Vector3d::UnitY(), 1e-12));
    EXPECT_TRUE(is_unit_vector(Eigen::Vector3d::UnitZ(), 1e-12));
}

TEST(IsUnitVector, AcceptsNormalizedGeneralVector)
{
    const Eigen::Vector3d v = Eigen::Vector3d(1.0, -2.0, 3.5).normalized();
    EXPECT_TRUE(is_unit_vector(v, 1e-12));
}

TEST(IsUnitVector, RejectsNonUnitVectors)
{
    EXPECT_FALSE(is_unit_vector(Eigen::Vector3d(2, 0, 0), 1e-9));
    EXPECT_FALSE(is_unit_vector(Eigen::Vector3d(0.5, 0, 0), 1e-9));
    EXPECT_FALSE(is_unit_vector(Eigen::Vector3d::Zero(), 1e-9));
}

TEST(IsUnitVector, ToleranceIsStrictAndBoundedByNormError)
{
    const double eps = 1e-6;
    const Eigen::Vector3d v(1.0 + eps, 0, 0);

    EXPECT_TRUE(is_unit_vector(v, 2.0 * eps));
    EXPECT_FALSE(is_unit_vector(v, 0.5 * eps));
}

TEST(IsUnitVector, WorksOnDynamicVectors)
{
    Eigen::VectorXd v = Eigen::VectorXd::Zero(7);
    v(3) = 1.0;
    EXPECT_TRUE(is_unit_vector(v, 1e-12));
}

TEST(IsUnitVector, WorksOnRowVectors)
{
    EXPECT_TRUE(is_unit_vector(Eigen::RowVector3d(0, 1, 0), 1e-12));
    EXPECT_FALSE(is_unit_vector(Eigen::RowVector3d(0, 3, 0), 1e-12));
}

// Expression templates must bind without materializing a Matrix.
TEST(IsUnitVector, AcceptsUnevaluatedExpressions)
{
    const Eigen::Vector3d a(2, 1, 1);
    const Eigen::Vector3d b(1, 1, 1);

    EXPECT_TRUE(is_unit_vector(a - b, 1e-12));
    EXPECT_TRUE(is_unit_vector(0.5 * Eigen::Vector3d(2, 0, 0), 1e-12));
}

TEST(IsUnitVector, AcceptsMapsAndBlocks)
{
    const double data[3] = {0.0, 0.0, 1.0};
    const Eigen::Map<const Eigen::Vector3d> mapped(data);
    EXPECT_TRUE(is_unit_vector(mapped, 1e-12));

    Eigen::Matrix<double, 3, 4> m = Eigen::Matrix<double, 3, 4>::Zero();
    m.col(2) = Eigen::Vector3d::UnitY();
    EXPECT_TRUE(is_unit_vector(m.col(2), 1e-12));
    EXPECT_FALSE(is_unit_vector(m.col(0), 1e-12));
}

TEST(IsUnitVector, WorksOnFloatScalars)
{
    EXPECT_TRUE(is_unit_vector(Eigen::Vector3f::UnitX(), 1e-6f));
    EXPECT_FALSE(is_unit_vector(Eigen::Vector3f(0.0f, 0.0f, 4.0f), 1e-6f));
}

// Documents current behavior: a negative tolerance can never be satisfied.
TEST(IsUnitVector, NegativeToleranceAlwaysFalse)
{
    EXPECT_FALSE(is_unit_vector(Eigen::Vector3d::UnitX(), -1e-12));
}

// ---------------------------------------------------------------------------
// inverse_rotate
// ---------------------------------------------------------------------------

TEST(InverseRotate, CoincidentFramesGiveZero)
{
    const Eigen::Matrix3d frame =
        Eigen::AngleAxisd(0.6, Eigen::Vector3d(1, 2, 3).normalized())
            .toRotationMatrix();

    EXPECT_TRUE(Near(inverse_rotate(frame, frame), Eigen::Vector3d::Zero(), 1e-6));
    EXPECT_TRUE(Near(
        inverse_rotate(Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity()),
        Eigen::Vector3d::Zero(), 1e-6));
}

// The reference returns the negated rotation vector, not the positive one.
TEST(InverseRotate, ReturnsNegatedRotationVector)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(0.0, 0.0, 1.0);
    const double angle = 0.7;
    const Eigen::Matrix3d target =
        Eigen::AngleAxisd(angle, axis).toRotationMatrix();

    const Eigen::Vector3d result =
        inverse_rotate(Eigen::Matrix3d::Identity(), target);

    EXPECT_TRUE(Near(result, Eigen::Vector3d(-angle * axis), 1e-10));
}

TEST(InverseRotate, MagnitudeIsTheRotationAngle)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
    for (double angle : {0.2, 0.9, 2.0, 3.0})
    {
        const Eigen::Matrix3d target =
            Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        const Eigen::Vector3d result =
            inverse_rotate(Eigen::Matrix3d::Identity(), target);

        EXPECT_NEAR(result.norm(), angle, 1e-9) << "angle " << angle;
    }
}

TEST(InverseRotate, IsRelativeNotAbsolute)
{
    const Eigen::Matrix3d shift =
        Eigen::AngleAxisd(1.3, Eigen::Vector3d(2, -1, 4).normalized())
            .toRotationMatrix();
    const Eigen::Matrix3d from =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Matrix3d to =
        Eigen::AngleAxisd(0.9, Eigen::Vector3d::UnitY()).toRotationMatrix();

    // Rotating both frames by the same amount changes the relative rotation
    // only through the frame it is expressed in, not its magnitude.
    EXPECT_NEAR(inverse_rotate(from, to).norm(),
                inverse_rotate(Eigen::Matrix3d(from * shift),
                               Eigen::Matrix3d(to * shift)).norm(),
                1e-9);
}

TEST(InverseRotate, ReversingTheFramesNegatesTheResult)
{
    const Eigen::Matrix3d from =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Matrix3d to =
        Eigen::AngleAxisd(0.9, Eigen::Vector3d::UnitY()).toRotationMatrix();

    EXPECT_TRUE(Near(inverse_rotate(from, to),
                     Eigen::Vector3d(-inverse_rotate(to, from)), 1e-10));
}

TEST(InverseRotateDeathTest, RejectsNonFiniteFrames)
{
    EXPECT_ASSERT_FAILURE(inverse_rotate(
        Eigen::Matrix3d::Constant(kNaN), Eigen::Matrix3d::Identity()));
    EXPECT_ASSERT_FAILURE(inverse_rotate(
        Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Constant(kInf)));
}

// ---------------------------------------------------------------------------
// rotation_matrix
//
// Follows the reference implementation: the angle is the scale times the axis
// norm, and the result is the transpose of the textbook Rodrigues matrix.
// ---------------------------------------------------------------------------

TEST(RotationMatrix, ZeroAngleGivesIdentity)
{
    EXPECT_TRUE(Near(rotation_matrix(0.0, Eigen::Vector3d::UnitZ()),
                     Eigen::Matrix3d::Identity()));
}

TEST(RotationMatrix, IsOrthogonalWithUnitDeterminant)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
    const Eigen::Matrix3d rotation = rotation_matrix(0.7, axis);

    EXPECT_TRUE(Near(rotation * rotation.transpose(), Eigen::Matrix3d::Identity()));
    EXPECT_NEAR(rotation.determinant(), 1.0, 1e-12);
}

TEST(RotationMatrix, LeavesItsOwnAxisFixed)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    EXPECT_TRUE(Near(rotation_matrix(1.1, axis) * axis, axis));
}

// Transposed convention: the result is R(-angle) in the textbook sense.
TEST(RotationMatrix, IsTransposeOfTextbookRodrigues)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(0.0, 0.0, 1.0);
    const double angle = 0.6;

    const Eigen::Matrix3d textbook =
        Eigen::AngleAxisd(angle, axis).toRotationMatrix();

    EXPECT_TRUE(Near(rotation_matrix(angle, axis), textbook.transpose()));
    EXPECT_TRUE(Near(rotation_matrix(-angle, axis), textbook));
}

// The angle is scale * ||axis||, so a non-unit axis rescales the rotation.
TEST(RotationMatrix, AngleScalesWithAxisNorm)
{
    const Eigen::Vector3d unit = Eigen::Vector3d::UnitZ();
    EXPECT_TRUE(Near(rotation_matrix(0.5, 2.0 * unit), rotation_matrix(1.0, unit)));
}

TEST(RotationMatrix, HalfTurnAboutZ)
{
    const Eigen::Matrix3d rotation =
        rotation_matrix(std::numbers::pi, Eigen::Vector3d::UnitZ());

    Eigen::Matrix3d expected;
    expected << -1.0, 0.0, 0.0,
                 0.0, -1.0, 0.0,
                 0.0, 0.0, 1.0;
    EXPECT_TRUE(Near(rotation, expected, 1e-14));
}

// Angles below the tolerance short-circuit to an exact identity rather than
// going through sin and cos of a near-zero argument.
TEST(RotationMatrix, SmallAngleGivesExactIdentity)
{
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    EXPECT_EQ(rotation_matrix(1e-15, Eigen::Vector3d::UnitZ()), identity);
    EXPECT_EQ(rotation_matrix(-1e-15, Eigen::Vector3d::UnitZ()), identity);
    EXPECT_EQ(rotation_matrix(0.5 * rotation_tolerance, Eigen::Vector3d::UnitZ()),
              identity);
}

// A short-but-legal axis still rotates, because the angle scales with length.
TEST(RotationMatrix, AngleJustAboveToleranceStillRotates)
{
    const Eigen::Matrix3d rotation =
        rotation_matrix(10.0 * rotation_tolerance, Eigen::Vector3d::UnitZ());

    EXPECT_NE(rotation, Eigen::Matrix3d::Identity());
    EXPECT_TRUE(Near(rotation, Eigen::Matrix3d::Identity(), 1e-9));
}

// Agrees with the reference Rodrigues expansion across a range of angles.
TEST(RotationMatrix, MatchesReferenceExpansion)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, -2.0, 3.0).normalized();
    for (double angle : {-3.5, -1.0, 0.25, 1.0, 2.7, 6.0})
    {
        const double s = std::sin(angle);
        const double c1 = 1.0 - std::cos(angle);
        const double v0 = axis(0);
        const double v1 = axis(1);
        const double v2 = axis(2);

        Eigen::Matrix3d expected;
        expected(0, 0) = 1.0 - c1 * (v1 * v1 + v2 * v2);
        expected(1, 1) = 1.0 - c1 * (v0 * v0 + v2 * v2);
        expected(2, 2) = 1.0 - c1 * (v0 * v0 + v1 * v1);
        expected(0, 1) = s * v2 + c1 * v0 * v1;
        expected(1, 0) = -s * v2 + c1 * v0 * v1;
        expected(0, 2) = -s * v1 + c1 * v0 * v2;
        expected(2, 0) = s * v1 + c1 * v0 * v2;
        expected(1, 2) = s * v0 + c1 * v1 * v2;
        expected(2, 1) = -s * v0 + c1 * v1 * v2;

        EXPECT_TRUE(Near(rotation_matrix(angle, axis), expected, 1e-14))
            << "angle " << angle;
    }
}

TEST(RotationMatrixDeathTest, RejectsAxisShorterThanTolerance)
{
    EXPECT_ASSERT_FAILURE(rotation_matrix(1.0, Eigen::Vector3d::Zero()));
    EXPECT_ASSERT_FAILURE(
        rotation_matrix(1.0, Eigen::Vector3d(0.5 * rotation_tolerance, 0.0, 0.0)));
}

TEST(RotationMatrix, AcceptsAxisLongerThanTolerance)
{
    EXPECT_NO_THROW({
        rotation_matrix(1.0, Eigen::Vector3d(10.0 * rotation_tolerance, 0.0, 0.0));
    });
}

TEST(RotationMatrixDeathTest, RejectsNonFiniteInputs)
{
    EXPECT_ASSERT_FAILURE(rotation_matrix(kNaN, Eigen::Vector3d::UnitZ()));
    EXPECT_ASSERT_FAILURE(rotation_matrix(1.0, Eigen::Vector3d(kInf, 0, 0)));
}
}  // namespace
}  // namespace cosserat::math