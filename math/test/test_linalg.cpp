#include "math/linalg.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <numbers>
#include <type_traits>
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

    const auto result = batched_matrix_vector<true>(mats, vecs);

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

    const auto result = batched_matrix_vector<true>(mats, vecs);

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
// ---------------------------------------------------------------------------
// batched_matrix_vector: template parameters
// ---------------------------------------------------------------------------

/** Three deliberately non-symmetric matrices, so transposing changes the answer. */
std::vector<Eigen::Matrix3d> asymmetric_matrices()
{
    std::vector<Eigen::Matrix3d> mats(3);
    mats[0] << 1, 2, 3,
               4, 5, 6,
               7, 8, 10;
    mats[1] << 0, -1, 0,
               1,  0, 0,
               0,  0, 2;
    mats[2] << 2, 0, 5,
               0, 3, 0,
               1, 0, 4;
    return mats;
}

VecBatch<3> sample_vectors()
{
    VecBatch<3> vecs(3, 3);
    vecs <<  1, 0,  2,
             0, 1, -1,
            -1, 2,  3;
    return vecs;
}

// The dimension comes from the matrices, so no explicit argument is needed.
TEST(BatchedMatrixVector, DeducesTheDimensionFromTheMatrices)
{
    const std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());
    VecBatch<3> vecs(3, 2);
    vecs << 1, 4,
            2, 5,
            3, 6;

    const auto deduced = batched_matrix_vector(mats, vecs);
    const auto explicitly = batched_matrix_vector<false, false>(mats, vecs);

    static_assert(std::is_same_v<
        std::decay_t<decltype(deduced)>, Eigen::Matrix<double, Eigen::Dynamic, 3>>);
    EXPECT_TRUE(Near(deduced, explicitly));
}

TEST(BatchedMatrixVector, DeducesDimensionsOtherThanThree)
{
    const std::vector<Eigen::Matrix2d> mats(2, 2.0 * Eigen::Matrix2d::Identity());
    VecBatch<2> vecs(2, 2);
    vecs << 5, 7,
            6, 8;

    const auto result = batched_matrix_vector(mats, vecs);

    static_assert(std::is_same_v<
        std::decay_t<decltype(result)>, Eigen::Matrix<double, Eigen::Dynamic, 2>>);
    EXPECT_TRUE(Near(result, Eigen::MatrixXd(2.0 * vecs.transpose())));
}

// --- TransposeMatrices ------------------------------------------------------

TEST(BatchedMatrixVector, TransposeAppliesTheTransposedMatrix)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    const VecBatch<3> vecs = sample_vectors();

    const auto result = batched_matrix_vector<false, true>(mats, vecs);

    ASSERT_EQ(result.rows(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(),
                         mats[static_cast<std::size_t>(i)].transpose() * vecs.col(i)))
            << "entry " << i;
    }
}

TEST(BatchedMatrixVector, TransposeDiffersFromTheUntransposedForm)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    const VecBatch<3> vecs = sample_vectors();

    const Eigen::MatrixXd plain = batched_matrix_vector<false, false>(mats, vecs);
    const Eigen::MatrixXd transposed = batched_matrix_vector<false, true>(mats, vecs);

    EXPECT_GT((plain - transposed).cwiseAbs().maxCoeff(), 1e-6);
}

// Transposing a symmetric matrix is a no-op, so both forms must agree.
TEST(BatchedMatrixVector, TransposeIsANoOpForSymmetricMatrices)
{
    std::vector<Eigen::Matrix3d> mats(2);
    mats[0] << 1, 2, 3,
               2, 5, 6,
               3, 6, 9;
    mats[1] = Eigen::Matrix3d::Identity();

    VecBatch<3> vecs(3, 2);
    vecs << 1, 4,
            2, 5,
            3, 6;

    EXPECT_TRUE(Near(batched_matrix_vector<false, true>(mats, vecs),
                     batched_matrix_vector<false, false>(mats, vecs)));
}

// A rotation's transpose is its inverse, so applying both recovers the input.
TEST(BatchedMatrixVector, TransposeUndoesARotationBatch)
{
    const Eigen::Index count = 4;
    std::vector<Eigen::Matrix3d> rotations;
    VecBatch<3> vecs(3, count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        rotations.push_back(
            Eigen::AngleAxisd(0.4 * static_cast<double>(i + 1),
                              Eigen::Vector3d(1, 2, 3).normalized())
                .toRotationMatrix());
        vecs.col(i) = Eigen::Vector3d(1.0 + i, 2.0 - i, 0.5 * i);
    }

    // Rotate into the material frame, then back out again.
    const Eigen::MatrixXd rotated = batched_matrix_vector<false, true>(rotations, vecs);
    const Eigen::MatrixXd recovered =
        batched_matrix_vector<false, false>(rotations, rotated.transpose());

    EXPECT_TRUE(Near(recovered, vecs.transpose(), 1e-12));
}

// --- AllowSizeMismatch ------------------------------------------------------

TEST(BatchedMatrixVector, AllowedMismatchKeepsLeadingEntries)
{
    const std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());
    VecBatch<3> vecs(3, 5);
    vecs = VecBatch<3>::Random(3, 5);

    const auto result = batched_matrix_vector<true>(mats, vecs);

    ASSERT_EQ(result.rows(), 2);
    EXPECT_TRUE(Near(result, vecs.leftCols(2).transpose()));
    // Explicitly not the trailing columns.
    EXPECT_GT((Eigen::MatrixXd(result) - vecs.rightCols(2).transpose())
                  .cwiseAbs().maxCoeff(), 1e-6);
}

TEST(BatchedMatrixVector, MatchedSizesAgreeWhicheverFlagIsUsed)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    const VecBatch<3> vecs = sample_vectors();

    EXPECT_TRUE(Near(batched_matrix_vector<false>(mats, vecs),
                     batched_matrix_vector<true>(mats, vecs)));
}

TEST(BatchedMatrixVector, BothFlagsTogether)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    VecBatch<3> vecs(3, 6);
    vecs = VecBatch<3>::Random(3, 6);

    const auto result = batched_matrix_vector<true, true>(mats, vecs);

    ASSERT_EQ(result.rows(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(),
                         mats[static_cast<std::size_t>(i)].transpose() * vecs.col(i)))
            << "entry " << i;
    }
}

// ---------------------------------------------------------------------------
// batched_matrix_vector: expression inputs
// ---------------------------------------------------------------------------

TEST(BatchedMatrixVectorExpressions, AcceptsColumnBlocks)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    VecBatch<3> wide(3, 6);
    wide = VecBatch<3>::Random(3, 6);

    const Eigen::MatrixXd from_block = batched_matrix_vector(mats, wide.leftCols(3));
    const Eigen::MatrixXd from_copy =
        batched_matrix_vector(mats, VecBatch<3>(wide.leftCols(3)));

    EXPECT_TRUE(Near(from_block, from_copy));
}

// A trailing sub-range must be extracted by the caller, since the mismatch
// flag would silently pair the leading entries instead.
TEST(BatchedMatrixVectorExpressions, TrailingBlockPairsWithLeadingMatrices)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    VecBatch<3> wide(3, 4);
    wide = VecBatch<3>::Random(3, 4);

    const Eigen::MatrixXd result = batched_matrix_vector(mats, wide.rightCols(3));

    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(),
                         mats[static_cast<std::size_t>(i)] * wide.col(i + 1)))
            << "entry " << i;
    }
}

TEST(BatchedMatrixVectorExpressions, AcceptsArithmeticExpressions)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    const VecBatch<3> vecs = sample_vectors();

    const Eigen::MatrixXd from_expression = batched_matrix_vector(mats, vecs + vecs);
    const Eigen::MatrixXd from_copy =
        batched_matrix_vector(mats, VecBatch<3>(2.0 * vecs));

    EXPECT_TRUE(Near(from_expression, from_copy));
}

// The outer product is how MuscleTorque and UniformTorque build their batches.
TEST(BatchedMatrixVectorExpressions, AcceptsOuterProductExpressions)
{
    const std::vector<Eigen::Matrix3d> mats(3, Eigen::Matrix3d::Identity());
    const Eigen::Vector3d direction(0.0, 0.0, 1.0);
    const Eigen::VectorXd magnitudes = Eigen::VectorXd::LinSpaced(3, 1.0, 3.0);

    const Eigen::MatrixXd result =
        batched_matrix_vector(mats, direction * magnitudes.transpose());

    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(result.row(i).transpose(),
                         Eigen::Vector3d(direction * magnitudes(i))))
            << "entry " << i;
    }
}

TEST(BatchedMatrixVectorExpressions, AcceptsMaps)
{
    const std::vector<Eigen::Matrix3d> mats(2, Eigen::Matrix3d::Identity());
    const double data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> mapped(data, 3, 2);

    const Eigen::MatrixXd result = batched_matrix_vector(mats, mapped);

    EXPECT_TRUE(Near(result.row(0).transpose(), Eigen::Vector3d(1.0, 2.0, 3.0)));
    EXPECT_TRUE(Near(result.row(1).transpose(), Eigen::Vector3d(4.0, 5.0, 6.0)));
}

TEST(BatchedMatrixVectorExpressions, AcceptsFixedColumnCounts)
{
    const std::vector<Eigen::Matrix3d> mats(3, Eigen::Matrix3d::Identity());
    Eigen::Matrix3d vecs;
    vecs << 1, 4, 7,
            2, 5, 8,
            3, 6, 9;

    const Eigen::MatrixXd result = batched_matrix_vector(mats, vecs);

    EXPECT_TRUE(Near(result, vecs.transpose()));
}

// A row-major source has a different layout, but the same values.
TEST(BatchedMatrixVectorExpressions, AcceptsRowMajorSources)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    const VecBatch<3> column_major = sample_vectors();
    const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> row_major =
        column_major;

    EXPECT_TRUE(Near(batched_matrix_vector(mats, row_major),
                     batched_matrix_vector(mats, column_major)));
}

TEST(BatchedMatrixVectorExpressions, DoesNotModifyItsInput)
{
    const std::vector<Eigen::Matrix3d> mats = asymmetric_matrices();
    VecBatch<3> vecs = sample_vectors();
    const VecBatch<3> before = vecs;

    const auto result = batched_matrix_vector<false, true>(mats, vecs);
    (void)result;

    EXPECT_TRUE(Near(vecs, before));
}

// ---------------------------------------------------------------------------
// row_norms
// ---------------------------------------------------------------------------

/** Reference implementation, written as an explicit loop. */
Eigen::VectorXd loop_row_norms(const Eigen::MatrixXd& matrix)
{
    Eigen::VectorXd result(matrix.rows());
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    {
        result(i) = matrix.row(i).norm();
    }
    return result;
}

TEST(RowNorms, ComputesTheEuclideanNormOfEachRow)
{
    Eigen::MatrixXd matrix(3, 3);
    matrix << 3, 4, 0,
              0, 0, 0,
              1, 2, 2;

    const Eigen::Vector3d expected(5.0, 0.0, 3.0);

    EXPECT_TRUE(Near(row_norms(matrix), expected));
}

TEST(RowNorms, ResultIsAColumnWithOneEntryPerRow)
{
    for (Eigen::Index rows : {1, 2, 5, 40})
    {
        const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, 4);
        const auto result = row_norms(matrix);

        EXPECT_EQ(result.rows(), rows) << "rows " << rows;
        EXPECT_EQ(result.cols(), 1) << "rows " << rows;
    }
}

TEST(RowNorms, MatchesAnExplicitLoop)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(9, 5);
    EXPECT_TRUE(Near(row_norms(matrix), loop_row_norms(matrix)));
}

TEST(RowNorms, IsNeverNegative)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(12, 6);
    const Eigen::VectorXd result = row_norms(matrix);

    EXPECT_TRUE((result.array() >= 0.0).all());
}

TEST(RowNorms, ZeroRowsGiveZeroNorm)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(4, 3);
    matrix.row(2).setZero();

    EXPECT_DOUBLE_EQ(row_norms(matrix)(2), 0.0);
    EXPECT_TRUE(Near(row_norms(Eigen::MatrixXd::Zero(4, 3)), Eigen::VectorXd::Zero(4)));
}

TEST(RowNorms, SingleColumnGivesAbsoluteValues)
{
    Eigen::VectorXd vector(4);
    vector << 3.0, -4.0, 0.0, -0.5;

    Eigen::VectorXd expected(4);
    expected << 3.0, 4.0, 0.0, 0.5;

    EXPECT_TRUE(Near(row_norms(vector), expected));
}

TEST(RowNorms, WorksOnManyColumns)
{
    // Each row holds 64 copies of one value, so its norm is 8 times that value.
    Eigen::MatrixXd matrix(3, 64);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        matrix.row(i).setConstant(static_cast<double>(i) + 1.0);
    }

    Eigen::Vector3d expected(8.0, 16.0, 24.0);

    EXPECT_TRUE(Near(row_norms(matrix), expected));
}

TEST(RowNorms, SingleColumnOfOneRow)
{
    const Eigen::Matrix<double, 1, 1> matrix = Eigen::Matrix<double, 1, 1>::Constant(-7.0);

    const auto result = row_norms(matrix);

    ASSERT_EQ(result.rows(), 1);
    EXPECT_DOUBLE_EQ(result(0), 7.0);
}

TEST(RowNorms, ScalesLinearlyWithTheMatrix)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(6, 3);

    EXPECT_TRUE(Near(row_norms(Eigen::MatrixXd(3.0 * matrix)),
                     Eigen::VectorXd(3.0 * row_norms(matrix))));
    EXPECT_TRUE(Near(row_norms(Eigen::MatrixXd(-matrix)), row_norms(matrix)));
}

// Rotating every row leaves its length untouched, which is the defining
// property of an orthogonal transform.
TEST(RowNorms, IsInvariantUnderRotationOfEachRow)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(7, 3);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.9, Eigen::Vector3d(1, -2, 3).normalized())
            .toRotationMatrix();

    // Rows are transformed on the right, so each row v becomes v * R^T.
    const Eigen::MatrixXd rotated = matrix * rotation.transpose();

    EXPECT_TRUE(Near(row_norms(rotated), row_norms(matrix), 1e-12));
}

TEST(RowNorms, ObeysTheTriangleInequality)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(8, 4);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(8, 4);

    const Eigen::VectorXd combined = row_norms(Eigen::MatrixXd(a + b));
    const Eigen::VectorXd bound = row_norms(a) + row_norms(b);

    EXPECT_TRUE((combined.array() <= bound.array() + kTol).all());
}

TEST(RowNorms, DoesNotModifyItsInput)
{
    Eigen::MatrixXd matrix(3, 3);
    matrix << 3, 4, 0,
              0, 0, 0,
              1, 2, 2;
    const Eigen::MatrixXd before = matrix;

    const auto result = row_norms(matrix);
    (void)result;

    EXPECT_TRUE(Near(matrix, before));
}

// A matrix with no columns has an empty sum under the square root.
TEST(RowNorms, ZeroColumnsGiveZeroNorms)
{
    const Eigen::MatrixXd matrix = Eigen::MatrixXd(4, 0);
    const auto result = row_norms(matrix);

    ASSERT_EQ(result.rows(), 4);
    EXPECT_TRUE(Near(result, Eigen::VectorXd::Zero(4)));
}

TEST(RowNorms, EmptyMatrixGivesEmptyResult)
{
    const auto result = row_norms(Eigen::MatrixXd(0, 3));

    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 1);
}

// ---------------------------------------------------------------------------
// row_norms: result types
// ---------------------------------------------------------------------------

TEST(RowNormsTypes, FixedSizeInputsGiveFixedSizeColumns)
{
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::Matrix3d>())), Eigen::Vector3d>);
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::Matrix<double, 4, 7>>())),
        Eigen::Matrix<double, 4, 1>>);
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::Vector3d>())), Eigen::Vector3d>);
    SUCCEED();
}

// A single row collapses to a 1 by 1, which is legal in either storage order.
TEST(RowNormsTypes, RowVectorGivesASingleValue)
{
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::RowVector3d>())),
        Eigen::Matrix<double, 1, 1>>);

    const Eigen::RowVector3d row(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(row_norms(row)(0), 5.0);
}

TEST(RowNormsTypes, DynamicInputsGiveDynamicColumns)
{
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::MatrixXd>())), Eigen::VectorXd>);
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::VectorXd>())), Eigen::VectorXd>);
    SUCCEED();
}

// Eigen forbids a fixed multi-row single-column matrix from being row major,
// so the result is column major whatever the input was.
TEST(RowNormsTypes, ResultIsColumnMajorEvenForRowMajorInput)
{
    using RowMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<RowMajorMatrix>())), Eigen::VectorXd>);

    RowMajorMatrix row_major(2, 3);
    row_major << 3, 4, 0,
                 1, 2, 2;
    Eigen::MatrixXd column_major = row_major;

    EXPECT_TRUE(Near(row_norms(row_major), Eigen::VectorXd(row_norms(column_major))));
}

TEST(RowNormsTypes, ScalarTypeFollowsTheInput)
{
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::MatrixXf>())), Eigen::VectorXf>);
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::Matrix<float, 4, 2>>())),
        Eigen::Matrix<float, 4, 1>>);

    Eigen::Matrix<float, 2, 2> matrix;
    matrix << 3.0f, 4.0f,
              5.0f, 12.0f;
    const Eigen::Matrix<float, 2, 1> result = row_norms(matrix);

    EXPECT_FLOAT_EQ(result(0), 5.0f);
    EXPECT_FLOAT_EQ(result(1), 13.0f);
}

// A complex matrix has real norms, so the scalar drops to the real type.
TEST(RowNormsTypes, ComplexInputGivesRealNorms)
{
    static_assert(std::is_same_v<
        decltype(row_norms(std::declval<Eigen::MatrixXcd>())), Eigen::VectorXd>);

    Eigen::MatrixXcd matrix(1, 2);
    matrix << std::complex<double>(3.0, 4.0), std::complex<double>(0.0, 0.0);

    EXPECT_NEAR(row_norms(matrix)(0), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// row_norms: expression inputs
// ---------------------------------------------------------------------------

TEST(RowNormsExpressions, AcceptsBlockExpressions)
{
    Eigen::MatrixXd matrix(3, 4);
    matrix << 3, 4, 9, 9,
              0, 0, 9, 9,
              1, 2, 9, 9;

    EXPECT_TRUE(Near(row_norms(matrix.leftCols(2)), Eigen::Vector3d(5.0, 0.0, std::sqrt(5.0))));
    EXPECT_TRUE(Near(row_norms(matrix.topRows(1)), Eigen::Matrix<double, 1, 1>(std::sqrt(187.0))));
}

TEST(RowNormsExpressions, AcceptsArithmeticExpressions)
{
    Eigen::MatrixXd a(2, 2);
    a << 4, 5,
         1, 1;
    Eigen::MatrixXd b(2, 2);
    b << 1, 1,
         1, 1;

    // Rows of (a - b) are (3, 4) and (0, 0).
    EXPECT_TRUE(Near(row_norms(a - b), Eigen::Vector2d(5.0, 0.0)));
}

TEST(RowNormsExpressions, TransposeMeasuresTheOriginalColumns)
{
    Eigen::MatrixXd matrix(2, 3);
    matrix << 3, 0, 1,
              4, 0, 1;

    // Rows of the transpose are the columns of the input.
    const Eigen::VectorXd result = row_norms(matrix.transpose());

    ASSERT_EQ(result.rows(), 3);
    EXPECT_DOUBLE_EQ(result(0), 5.0);
    EXPECT_DOUBLE_EQ(result(1), 0.0);
    EXPECT_NEAR(result(2), std::sqrt(2.0), kTol);
}

// ---------------------------------------------------------------------------
// row_norms alongside the other operators
// ---------------------------------------------------------------------------

// batched_matrix_vector applies one rotation per entry, so the norms of its
// output must match the norms of the input columns.
TEST(RowNorms, MeasuresBatchedRotationOutputUnchanged)
{
    const Eigen::Index count = 4;
    std::vector<Eigen::Matrix3d> rotations;
    VecBatch<3> vectors(3, count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        rotations.push_back(
            Eigen::AngleAxisd(0.4 * static_cast<double>(i + 1),
                              Eigen::Vector3d(1, 2, 3).normalized())
                .toRotationMatrix());
        vectors.col(i) = Eigen::Vector3d(1.0 + i, 2.0 - i, 0.5 * i);
    }

    const auto rotated = batched_matrix_vector(rotations, vectors);

    EXPECT_TRUE(Near(row_norms(rotated),
                     Eigen::VectorXd(vectors.colwise().norm().transpose()),
                     1e-12));
}

// A row is a unit vector exactly when its norm is one, so the two agree.
TEST(RowNorms, AgreesWithIsUnitVector)
{
    Eigen::MatrixXd matrix(3, 3);
    matrix.row(0) = Eigen::Vector3d(1, 2, 3).normalized().transpose();
    matrix.row(1) = Eigen::RowVector3d(2.0, 0.0, 0.0);
    matrix.row(2) = Eigen::Vector3d(0, -1, 0).transpose();

    const Eigen::VectorXd norms = row_norms(matrix);

    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    {
        const bool by_norm = std::abs(norms(i) - 1.0) < 1e-12;
        EXPECT_EQ(by_norm, is_unit_vector(matrix.row(i), 1e-12)) << "row " << i;
    }
}

// ---------------------------------------------------------------------------
// batched_dot_product
// ---------------------------------------------------------------------------

/** Reference implementation, written as an explicit loop over rows. */
Eigen::VectorXd loop_dot_product(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    Eigen::VectorXd result(a.rows());
    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        result(i) = a.row(i).dot(b.row(i));
    }
    return result;
}

TEST(BatchedDotProduct, ComputesTheDotProductOfEachRowPair)
{
    Eigen::MatrixXd a(3, 3);
    a << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    Eigen::MatrixXd b(3, 3);
    b << 1, 0, 0,
         0, 1, 0,
         1, 1, 1;

    const Eigen::Vector3d expected(1.0, 5.0, 24.0);

    EXPECT_TRUE(Near(batched_dot_product(a, b), expected));
}

TEST(BatchedDotProduct, ResultIsAColumnWithOneEntryPerRow)
{
    for (Eigen::Index rows : {1, 2, 5, 40})
    {
        const Eigen::MatrixXd a = Eigen::MatrixXd::Random(rows, 4);
        const Eigen::MatrixXd b = Eigen::MatrixXd::Random(rows, 4);
        const auto result = batched_dot_product(a, b);

        EXPECT_EQ(result.rows(), rows) << "rows " << rows;
        EXPECT_EQ(result.cols(), 1) << "rows " << rows;
    }
}

// The fused expression must agree with calling dot() on each row in turn.
TEST(BatchedDotProduct, AgreesWithPerRowDot)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(9, 5);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(9, 5);

    EXPECT_TRUE(Near(batched_dot_product(a, b), loop_dot_product(a, b)));
}

TEST(BatchedDotProduct, IsSymmetricForRealScalars)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(6, 3);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(6, 3);

    EXPECT_TRUE(Near(batched_dot_product(a, b), batched_dot_product(b, a)));
}

// Dotting a stack with itself gives the squared row norms, tying the two
// row-wise reductions together.
TEST(BatchedDotProduct, SelfProductGivesSquaredRowNorms)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(7, 3);

    const Eigen::VectorXd self = batched_dot_product(a, a);
    const Eigen::VectorXd squared = row_norms(a).array().square();

    EXPECT_TRUE(Near(self, squared, 1e-12));
    EXPECT_TRUE((self.array() >= 0.0).all());
}

TEST(BatchedDotProduct, OrthogonalRowsGiveZero)
{
    Eigen::MatrixXd a(3, 3);
    a << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    Eigen::MatrixXd b(3, 3);
    b << 0, 1, 0,
         0, 0, 1,
         1, 0, 0;

    EXPECT_TRUE(Near(batched_dot_product(a, b), Eigen::Vector3d::Zero()));
}

TEST(BatchedDotProduct, IsLinearInEachArgument)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 3);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(5, 3);
    const Eigen::MatrixXd c = Eigen::MatrixXd::Random(5, 3);

    EXPECT_TRUE(Near(batched_dot_product(Eigen::MatrixXd(3.0 * a), b),
                     Eigen::VectorXd(3.0 * batched_dot_product(a, b))));
    EXPECT_TRUE(Near(batched_dot_product(a, Eigen::MatrixXd(b + c)),
                     Eigen::VectorXd(batched_dot_product(a, b)
                                     + batched_dot_product(a, c))));
}

// Cauchy-Schwarz, checked against the row-wise norms.
TEST(BatchedDotProduct, ObeysCauchySchwarz)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(8, 4);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(8, 4);

    const Eigen::VectorXd dots = batched_dot_product(a, b);
    const Eigen::VectorXd bound =
        row_norms(a).array() * row_norms(b).array();

    EXPECT_TRUE((dots.cwiseAbs().array() <= bound.array() + kTol).all());
}

TEST(BatchedDotProduct, ZeroRowsGiveZero)
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(4, 3);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(4, 3);
    a.row(2).setZero();

    EXPECT_DOUBLE_EQ(batched_dot_product(a, b)(2), 0.0);
}

TEST(BatchedDotProduct, WorksOnASingleColumn)
{
    Eigen::VectorXd a(4);
    a << 1.0, -2.0, 3.0, 0.0;
    Eigen::VectorXd b(4);
    b << 5.0, 5.0, 5.0, 5.0;

    Eigen::VectorXd expected(4);
    expected << 5.0, -10.0, 15.0, 0.0;

    EXPECT_TRUE(Near(batched_dot_product(a, b), expected));
}

TEST(BatchedDotProduct, WorksOnManyColumns)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Ones(3, 64);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Ones(3, 64) * 2.0;

    EXPECT_TRUE(Near(batched_dot_product(a, b),
                     Eigen::VectorXd::Constant(3, 128.0)));
}

TEST(BatchedDotProduct, DoesNotModifyItsInputs)
{
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(4, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(4, 3);
    const Eigen::MatrixXd a_before = a;
    const Eigen::MatrixXd b_before = b;

    const auto result = batched_dot_product(a, b);
    (void)result;

    EXPECT_TRUE(Near(a, a_before));
    EXPECT_TRUE(Near(b, b_before));
}

// An empty sum is zero, matching row_norms on a matrix with no columns.
TEST(BatchedDotProduct, ZeroColumnsGiveZeros)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd(4, 0);
    const Eigen::MatrixXd b = Eigen::MatrixXd(4, 0);

    const auto result = batched_dot_product(a, b);

    ASSERT_EQ(result.rows(), 4);
    EXPECT_TRUE(Near(result, Eigen::VectorXd::Zero(4)));
}

TEST(BatchedDotProduct, EmptyInputGivesEmptyResult)
{
    const auto result =
        batched_dot_product(Eigen::MatrixXd(0, 3), Eigen::MatrixXd(0, 3));

    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 1);
}

TEST(BatchedDotProductDeathTest, RejectsMismatchedShapes)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(4, 3);

    EXPECT_ASSERT_FAILURE(
        batched_dot_product(a, Eigen::MatrixXd(Eigen::MatrixXd::Random(5, 3))));
    EXPECT_ASSERT_FAILURE(
        batched_dot_product(a, Eigen::MatrixXd(Eigen::MatrixXd::Random(4, 2))));
}

// ---------------------------------------------------------------------------
// batched_dot_product: result types
// ---------------------------------------------------------------------------

TEST(BatchedDotProductTypes, FixedSizeInputsGiveFixedSizeColumns)
{
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::Matrix3d>(),
                                     std::declval<Eigen::Matrix3d>())),
        Eigen::Vector3d>);
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::Matrix<double, 4, 7>>(),
                                     std::declval<Eigen::Matrix<double, 4, 7>>())),
        Eigen::Matrix<double, 4, 1>>);
    SUCCEED();
}

TEST(BatchedDotProductTypes, DynamicInputsGiveDynamicColumns)
{
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::MatrixXd>(),
                                     std::declval<Eigen::MatrixXd>())),
        Eigen::VectorXd>);
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::VectorXd>(),
                                     std::declval<Eigen::VectorXd>())),
        Eigen::VectorXd>);
    SUCCEED();
}

// Unlike row_norms, an inner product of complex rows is itself complex, so the
// scalar carries over rather than dropping to the real type.
TEST(BatchedDotProductTypes, ComplexInputKeepsAComplexScalar)
{
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::MatrixXcd>(),
                                     std::declval<Eigen::MatrixXcd>())),
        Eigen::VectorXcd>);
    SUCCEED();
}

// The left argument is conjugated, matching Eigen's own dot().
TEST(BatchedDotProductTypes, ComplexConjugatesTheLeftArgument)
{
    Eigen::MatrixXcd a(1, 2);
    a << std::complex<double>(1.0, 2.0), std::complex<double>(3.0, -1.0);
    Eigen::MatrixXcd b(1, 2);
    b << std::complex<double>(0.0, 1.0), std::complex<double>(2.0, 2.0);

    const std::complex<double> ours = batched_dot_product(a, b)(0);
    const std::complex<double> reference = a.row(0).dot(b.row(0));

    EXPECT_NEAR(ours.real(), reference.real(), kTol);
    EXPECT_NEAR(ours.imag(), reference.imag(), kTol);
    // Conjugation makes the order matter, unlike the real case.
    EXPECT_GT(std::abs(ours - batched_dot_product(b, a)(0)), 1e-9);
}

TEST(BatchedDotProductTypes, ScalarTypeFollowsTheInput)
{
    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<Eigen::MatrixXf>(),
                                     std::declval<Eigen::MatrixXf>())),
        Eigen::VectorXf>);

    Eigen::Matrix<float, 2, 2> a;
    a << 1.0f, 2.0f,
         3.0f, 4.0f;
    Eigen::Matrix<float, 2, 2> b;
    b << 5.0f, 6.0f,
         7.0f, 8.0f;

    const Eigen::Matrix<float, 2, 1> result = batched_dot_product(a, b);

    EXPECT_FLOAT_EQ(result(0), 17.0f);
    EXPECT_FLOAT_EQ(result(1), 53.0f);
}

TEST(BatchedDotProductTypes, ResultIsColumnMajorEvenForRowMajorInput)
{
    using RowMajorMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    static_assert(std::is_same_v<
        decltype(batched_dot_product(std::declval<RowMajorMatrix>(),
                                     std::declval<RowMajorMatrix>())),
        Eigen::VectorXd>);

    RowMajorMatrix a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;
    const Eigen::MatrixXd column_major = a;

    EXPECT_TRUE(Near(batched_dot_product(a, a),
                     Eigen::VectorXd(batched_dot_product(column_major, column_major))));
}

// ---------------------------------------------------------------------------
// batched_dot_product: expression inputs
// ---------------------------------------------------------------------------

TEST(BatchedDotProductExpressions, AcceptsMatchingBlocks)
{
    Eigen::MatrixXd a(4, 3);
    a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(4, 3);

    const Eigen::VectorXd from_blocks =
        batched_dot_product(a.bottomRows(3), b.bottomRows(3));
    const Eigen::VectorXd from_copies = loop_dot_product(
        Eigen::MatrixXd(a.bottomRows(3)), Eigen::MatrixXd(b.bottomRows(3)));

    EXPECT_TRUE(Near(from_blocks, from_copies));
}

// The two arguments need not be the same kind of expression.
TEST(BatchedDotProductExpressions, AcceptsABlockAgainstAWholeMatrix)
{
    Eigen::MatrixXd big(4, 3);
    big << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    const Eigen::MatrixXd small = Eigen::MatrixXd::Random(3, 3);

    const Eigen::VectorXd result = batched_dot_product(big.bottomRows(3), small);

    EXPECT_TRUE(Near(result,
                     loop_dot_product(Eigen::MatrixXd(big.bottomRows(3)), small)));
}

// Differencing a stack and projecting onto another is the shape this is for.
TEST(BatchedDotProductExpressions, AcceptsOffsetBlocksOfTheSameMatrix)
{
    Eigen::MatrixXd a(4, 3);
    a << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         1, 1, 1;

    const Eigen::VectorXd result =
        batched_dot_product(a.bottomRows(3), a.topRows(3));

    ASSERT_EQ(result.rows(), 3);
    EXPECT_TRUE(Near(result, Eigen::Vector3d(0.0, 0.0, 1.0)));
}

TEST(BatchedDotProductExpressions, AcceptsArithmeticExpressions)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(5, 3);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(5, 3);

    EXPECT_TRUE(Near(batched_dot_product(a + a, b),
                     Eigen::VectorXd(2.0 * batched_dot_product(a, b))));
    EXPECT_TRUE(Near(batched_dot_product(a, b - b), Eigen::VectorXd::Zero(5)));
}

TEST(BatchedDotProductExpressions, AcceptsColumnBlocks)
{
    Eigen::MatrixXd a(3, 4);
    a << 1, 2, 9, 9,
         3, 4, 9, 9,
         5, 6, 9, 9;
    Eigen::MatrixXd b(3, 4);
    b << 1, 1, 8, 8,
         1, 1, 8, 8,
         1, 1, 8, 8;

    EXPECT_TRUE(Near(batched_dot_product(a.leftCols(2), b.leftCols(2)),
                     Eigen::Vector3d(3.0, 7.0, 11.0)));
}

TEST(BatchedDotProductExpressions, AcceptsMaps)
{
    const double left_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double right_data[6] = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const Eigen::Map<const Eigen::Matrix<double, 3, 2>> left(left_data);
    const Eigen::Map<const Eigen::Matrix<double, 3, 2>> right(right_data);

    const Eigen::Vector3d result = batched_dot_product(left, right);

    // Column-major maps, so rows are (1,4), (2,5), (3,6) against (1,1), (0,1), (0,1).
    EXPECT_TRUE(Near(result, Eigen::Vector3d(5.0, 5.0, 6.0)));
}

TEST(BatchedDotProductExpressions, AcceptsTransposeExpressions)
{
    Eigen::MatrixXd a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;
    Eigen::MatrixXd b(2, 3);
    b << 1, 1, 1,
         2, 2, 2;

    // The transposes are 3x2, so each row pairs one column of each input.
    const Eigen::VectorXd result =
        batched_dot_product(a.transpose(), b.transpose());

    ASSERT_EQ(result.rows(), 3);
    EXPECT_TRUE(Near(result, Eigen::Vector3d(9.0, 12.0, 15.0)));
}

// ---------------------------------------------------------------------------
// batched_dot_product alongside the other operators
// ---------------------------------------------------------------------------

// Rotating both stacks by the same frames leaves every inner product alone,
// which is the defining property of an orthogonal transform.
TEST(BatchedDotProduct, IsInvariantUnderACommonRotation)
{
    const Eigen::Index count = 4;
    std::vector<Eigen::Matrix3d> rotations;
    VecBatch<3> left(3, count);
    VecBatch<3> right(3, count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        rotations.push_back(
            Eigen::AngleAxisd(0.4 * static_cast<double>(i + 1),
                              Eigen::Vector3d(1, 2, 3).normalized())
                .toRotationMatrix());
        left.col(i) = Eigen::Vector3d(1.0 + i, 2.0 - i, 0.5 * i);
        right.col(i) = Eigen::Vector3d(0.5 * i, 1.0, 3.0 - i);
    }

    const Eigen::MatrixXd rotated_left = batched_matrix_vector(rotations, left);
    const Eigen::MatrixXd rotated_right = batched_matrix_vector(rotations, right);

    EXPECT_TRUE(Near(batched_dot_product(rotated_left, rotated_right),
                     batched_dot_product(Eigen::MatrixXd(left.transpose()),
                                         Eigen::MatrixXd(right.transpose())),
                     1e-12));
}

// Projecting a stack onto its own unit directions recovers the row norms.
TEST(BatchedDotProduct, ProjectionOntoUnitRowsRecoversTheNorms)
{
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(6, 3);
    const Eigen::VectorXd norms = row_norms(a);
    const Eigen::MatrixXd unit = a.array().colwise() / norms.array();

    EXPECT_TRUE(Near(batched_dot_product(a, unit), norms, 1e-12));
}

// ---------------------------------------------------------------------------
// batched_cross_product
// ---------------------------------------------------------------------------

using Stack3 = Eigen::Matrix<double, Eigen::Dynamic, 3>;

/** Reference implementation, deferring to Eigen's own cross product. */
Stack3 loop_cross_product(const Stack3& a, const Stack3& b)
{
    Stack3 result(a.rows(), 3);
    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        result.row(i) = a.row(i).cross(b.row(i));
    }
    return result;
}

Stack3 random_stack(Eigen::Index rows)
{
    return Stack3::Random(rows, 3);
}

TEST(BatchedCrossProduct, MatchesTheRightHandRule)
{
    Stack3 a(3, 3);
    a << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    Stack3 b(3, 3);
    b << 0, 1, 0,
         0, 0, 1,
         1, 0, 0;

    // x cross y = z, y cross z = x, z cross x = y
    Stack3 expected(3, 3);
    expected << 0, 0, 1,
                1, 0, 0,
                0, 1, 0;

    EXPECT_TRUE(Near(batched_cross_product(a, b), expected));
}

TEST(BatchedCrossProduct, ComputesTheStandardFormula)
{
    Stack3 a(1, 3);
    a << 1, 2, 3;
    Stack3 b(1, 3);
    b << 4, 5, 6;

    // (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4)
    EXPECT_TRUE(Near(batched_cross_product(a, b),
                     Eigen::RowVector3d(-3.0, 6.0, -3.0)));
}

TEST(BatchedCrossProduct, ResultHasTheSameShapeAsTheInputs)
{
    for (Eigen::Index rows : {1, 2, 5, 40})
    {
        const auto result = batched_cross_product(random_stack(rows), random_stack(rows));

        EXPECT_EQ(result.rows(), rows) << "rows " << rows;
        EXPECT_EQ(result.cols(), 3) << "rows " << rows;
    }
}

// The loop must agree with calling Eigen's own cross() on each row.
TEST(BatchedCrossProduct, AgreesWithPerRowCross)
{
    const Stack3 a = random_stack(9);
    const Stack3 b = random_stack(9);

    EXPECT_TRUE(Near(batched_cross_product(a, b), loop_cross_product(a, b)));
}

TEST(BatchedCrossProduct, IsAnticommutative)
{
    const Stack3 a = random_stack(6);
    const Stack3 b = random_stack(6);

    EXPECT_TRUE(Near(batched_cross_product(a, b),
                     Stack3(-batched_cross_product(b, a))));
}

TEST(BatchedCrossProduct, SelfProductIsZero)
{
    const Stack3 a = random_stack(6);

    EXPECT_TRUE(Near(batched_cross_product(a, a), Stack3::Zero(6, 3)));
}

TEST(BatchedCrossProduct, ParallelRowsGiveZero)
{
    const Stack3 a = random_stack(5);
    const Stack3 scaled = 3.0 * a;

    EXPECT_TRUE(Near(batched_cross_product(a, scaled), Stack3::Zero(5, 3), 1e-12));
    EXPECT_TRUE(Near(batched_cross_product(a, Stack3(-a)), Stack3::Zero(5, 3)));
}

// The result is perpendicular to both inputs, checked with the dot product.
TEST(BatchedCrossProduct, IsOrthogonalToBothInputs)
{
    const Stack3 a = random_stack(8);
    const Stack3 b = random_stack(8);
    const Stack3 crossed = batched_cross_product(a, b);

    EXPECT_LT(batched_dot_product(crossed, a).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(batched_dot_product(crossed, b).cwiseAbs().maxCoeff(), 1e-12);
}

// Lagrange's identity ties the cross product to the dot product and the norms.
TEST(BatchedCrossProduct, ObeysLagrangesIdentity)
{
    const Stack3 a = random_stack(8);
    const Stack3 b = random_stack(8);

    const Eigen::VectorXd cross_squared =
        row_norms(batched_cross_product(a, b)).array().square();
    const Eigen::VectorXd dot_squared =
        batched_dot_product(a, b).array().square();
    const Eigen::VectorXd product_of_squares =
        row_norms(a).array().square() * row_norms(b).array().square();

    EXPECT_TRUE(Near(Eigen::VectorXd(cross_squared + dot_squared),
                     product_of_squares, 1e-12));
}

TEST(BatchedCrossProduct, MagnitudeIsAreaOfTheParallelogram)
{
    Stack3 a(2, 3);
    a << 3, 0, 0,
         1, 0, 0;
    Stack3 b(2, 3);
    b << 0, 4, 0,
         0, 0, 1;

    // Both pairs are perpendicular, so the magnitudes are the products.
    EXPECT_TRUE(Near(row_norms(batched_cross_product(a, b)),
                     Eigen::Vector2d(12.0, 1.0)));
}

TEST(BatchedCrossProduct, IsLinearInEachArgument)
{
    const Stack3 a = random_stack(5);
    const Stack3 b = random_stack(5);
    const Stack3 c = random_stack(5);

    EXPECT_TRUE(Near(batched_cross_product(Stack3(3.0 * a), b),
                     Stack3(3.0 * batched_cross_product(a, b)), 1e-12));
    EXPECT_TRUE(Near(batched_cross_product(a, Stack3(b + c)),
                     Stack3(batched_cross_product(a, b)
                            + batched_cross_product(a, c)), 1e-12));
}

// The Jacobi identity holds row by row, which no sign error survives.
TEST(BatchedCrossProduct, ObeysTheJacobiIdentity)
{
    const Stack3 a = random_stack(6);
    const Stack3 b = random_stack(6);
    const Stack3 c = random_stack(6);

    const Stack3 total =
        batched_cross_product(a, Stack3(batched_cross_product(b, c)))
        + batched_cross_product(b, Stack3(batched_cross_product(c, a)))
        + batched_cross_product(c, Stack3(batched_cross_product(a, b)));

    EXPECT_TRUE(Near(total, Stack3::Zero(6, 3), 1e-12));
}

// Rows are treated independently, so shuffling one pair leaves the rest alone.
TEST(BatchedCrossProduct, TreatsEachRowIndependently)
{
    Stack3 a = random_stack(5);
    Stack3 b = random_stack(5);
    const Stack3 before = batched_cross_product(a, b);

    a.row(2) = Eigen::RowVector3d(9.0, -1.0, 4.0);
    const Stack3 after = batched_cross_product(a, b);

    EXPECT_TRUE(Near(after.topRows(2), before.topRows(2)));
    EXPECT_TRUE(Near(after.bottomRows(2), before.bottomRows(2)));
    EXPECT_GT((after.row(2) - before.row(2)).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(BatchedCrossProduct, ZeroRowsGiveZero)
{
    Stack3 a = random_stack(4);
    const Stack3 b = random_stack(4);
    a.row(1).setZero();

    EXPECT_TRUE(Near(batched_cross_product(a, b).row(1),
                     Eigen::RowVector3d::Zero()));
}

TEST(BatchedCrossProduct, DoesNotModifyItsInputs)
{
    Stack3 a = random_stack(4);
    Stack3 b = random_stack(4);
    const Stack3 a_before = a;
    const Stack3 b_before = b;

    const auto result = batched_cross_product(a, b);
    (void)result;

    EXPECT_TRUE(Near(a, a_before));
    EXPECT_TRUE(Near(b, b_before));
}

TEST(BatchedCrossProduct, EmptyInputGivesEmptyResult)
{
    const auto result =
        batched_cross_product(Stack3(0, 3), Stack3(0, 3));

    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 3);
}

TEST(BatchedCrossProductDeathTest, RejectsMismatchedRowCounts)
{
    EXPECT_ASSERT_FAILURE(batched_cross_product(random_stack(4), random_stack(5)));
}

// The column count is only known at run time for a dynamic-column input.
TEST(BatchedCrossProductDeathTest, RejectsAColumnCountOtherThanThree)
{
    const Eigen::MatrixXd wide = Eigen::MatrixXd::Random(4, 4);
    const Eigen::MatrixXd narrow = Eigen::MatrixXd::Random(4, 2);

    EXPECT_ASSERT_FAILURE(batched_cross_product(wide, wide));
    EXPECT_ASSERT_FAILURE(batched_cross_product(narrow, narrow));
}

// ---------------------------------------------------------------------------
// batched_cross_product: result types
// ---------------------------------------------------------------------------

TEST(BatchedCrossProductTypes, StackInputsGiveStackResults)
{
    static_assert(std::is_same_v<
        decltype(batched_cross_product(std::declval<Stack3>(),
                                       std::declval<Stack3>())),
        Stack3>);
    static_assert(std::is_same_v<
        decltype(batched_cross_product(std::declval<Eigen::Matrix<double, 4, 3>>(),
                                       std::declval<Eigen::Matrix<double, 4, 3>>())),
        Eigen::Matrix<double, 4, 3>>);
    SUCCEED();
}

// A single row is one entity, so the result stays a row vector.
TEST(BatchedCrossProductTypes, RowVectorsGiveRowVectors)
{
    static_assert(std::is_same_v<
        decltype(batched_cross_product(std::declval<Eigen::RowVector3d>(),
                                       std::declval<Eigen::RowVector3d>())),
        Eigen::RowVector3d>);

    const Eigen::RowVector3d result = batched_cross_product(
        Eigen::RowVector3d(1, 0, 0), Eigen::RowVector3d(0, 1, 0));

    EXPECT_TRUE(Near(result, Eigen::RowVector3d(0, 0, 1)));
}

// A dynamic-column input still yields a three-column result, since that is the
// only column count a cross product can produce.
TEST(BatchedCrossProductTypes, DynamicColumnInputsGiveThreeColumnResults)
{
    static_assert(std::is_same_v<
        decltype(batched_cross_product(std::declval<Eigen::MatrixXd>(),
                                       std::declval<Eigen::MatrixXd>())),
        Stack3>);

    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(4, 3);
    const Eigen::MatrixXd b = Eigen::MatrixXd::Random(4, 3);
    const Stack3 result = batched_cross_product(a, b);

    EXPECT_EQ(result.cols(), 3);
    EXPECT_TRUE(Near(result, loop_cross_product(Stack3(a), Stack3(b))));
}

TEST(BatchedCrossProductTypes, StorageOrderIsPreservedWhereLegal)
{
    using RowMajorStack =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

    static_assert(decltype(batched_cross_product(
        std::declval<RowMajorStack>(), std::declval<RowMajorStack>()))::IsRowMajor);
    static_assert(not decltype(batched_cross_product(
        std::declval<Stack3>(), std::declval<Stack3>()))::IsRowMajor);

    RowMajorStack a(2, 3);
    a << 1, 0, 0,
         0, 1, 0;
    RowMajorStack b(2, 3);
    b << 0, 1, 0,
         0, 0, 1;
    const Stack3 column_major_a = a;
    const Stack3 column_major_b = b;

    EXPECT_TRUE(Near(Stack3(batched_cross_product(a, b)),
                     batched_cross_product(column_major_a, column_major_b)));
}

TEST(BatchedCrossProductTypes, ScalarTypeFollowsTheInput)
{
    static_assert(std::is_same_v<
        decltype(batched_cross_product(std::declval<Eigen::Matrix<float, 4, 3>>(),
                                       std::declval<Eigen::Matrix<float, 4, 3>>())),
        Eigen::Matrix<float, 4, 3>>);

    Eigen::Matrix<float, 1, 3> a;
    a << 1.0f, 2.0f, 3.0f;
    Eigen::Matrix<float, 1, 3> b;
    b << 4.0f, 5.0f, 6.0f;

    const Eigen::Matrix<float, 1, 3> result = batched_cross_product(a, b);

    EXPECT_FLOAT_EQ(result(0), -3.0f);
    EXPECT_FLOAT_EQ(result(1), 6.0f);
    EXPECT_FLOAT_EQ(result(2), -3.0f);
}

// Each result row is conjugated for complex scalars, matching Eigen's cross().
TEST(BatchedCrossProductTypes, ComplexAgreesWithEigensCross)
{
    Eigen::Matrix<std::complex<double>, 1, 3> a;
    a << std::complex<double>(1.0, 2.0), std::complex<double>(0.0, 1.0),
         std::complex<double>(3.0, 0.0);
    Eigen::Matrix<std::complex<double>, 1, 3> b;
    b << std::complex<double>(0.0, 1.0), std::complex<double>(2.0, 0.0),
         std::complex<double>(1.0, -1.0);

    const Eigen::Matrix<std::complex<double>, 1, 3> ours =
        batched_cross_product(a, b);
    const Eigen::Matrix<std::complex<double>, 1, 3> reference =
        a.row(0).cross(b.row(0));

    EXPECT_LT((ours - reference).cwiseAbs().maxCoeff(), kTol);
}

// ---------------------------------------------------------------------------
// batched_cross_product: expression inputs
// ---------------------------------------------------------------------------

TEST(BatchedCrossProductExpressions, AcceptsMatchingBlocks)
{
    const Stack3 a = random_stack(5);
    const Stack3 b = random_stack(5);

    EXPECT_TRUE(Near(batched_cross_product(a.bottomRows(3), b.bottomRows(3)),
                     loop_cross_product(Stack3(a.bottomRows(3)),
                                        Stack3(b.bottomRows(3)))));
}

// The two arguments need not be the same kind of expression.
TEST(BatchedCrossProductExpressions, AcceptsABlockAgainstAWholeMatrix)
{
    const Stack3 big = random_stack(5);
    const Stack3 small = random_stack(3);

    EXPECT_TRUE(Near(batched_cross_product(big.bottomRows(3), small),
                     loop_cross_product(Stack3(big.bottomRows(3)), small)));
}

// Crossing offset blocks of one stack is how a discrete curvature is formed.
TEST(BatchedCrossProductExpressions, AcceptsOffsetBlocksOfTheSameMatrix)
{
    Stack3 a(4, 3);
    a << 1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         1, 0, 0;

    const Stack3 result = batched_cross_product(a.bottomRows(3), a.topRows(3));

    ASSERT_EQ(result.rows(), 3);
    EXPECT_TRUE(Near(result.row(0), Eigen::RowVector3d(0, 0, -1)));
    EXPECT_TRUE(Near(result.row(1), Eigen::RowVector3d(-1, 0, 0)));
    EXPECT_TRUE(Near(result.row(2), Eigen::RowVector3d(0, -1, 0)));
}

TEST(BatchedCrossProductExpressions, AcceptsArithmeticExpressions)
{
    const Stack3 a = random_stack(5);
    const Stack3 b = random_stack(5);

    EXPECT_TRUE(Near(batched_cross_product(a + a, b),
                     Stack3(2.0 * batched_cross_product(a, b)), 1e-12));
    EXPECT_TRUE(Near(batched_cross_product(a, Stack3(b - b)),
                     Stack3::Zero(5, 3)));
}

// A computed expression is evaluated once when bound, so repeated coefficient
// access inside the loop cannot re-run it.
TEST(BatchedCrossProductExpressions, AcceptsProductExpressions)
{
    const Stack3 a = random_stack(5);
    const Stack3 b = random_stack(5);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized())
            .toRotationMatrix();

    EXPECT_TRUE(Near(batched_cross_product(a * rotation, b),
                     loop_cross_product(Stack3(a * rotation), b), 1e-12));
}

TEST(BatchedCrossProductExpressions, AcceptsColumnBlocks)
{
    Eigen::MatrixXd wide(2, 5);
    wide << 1, 0, 0, 9, 9,
            0, 1, 0, 9, 9;

    const Stack3 result =
        batched_cross_product(wide.leftCols(3), wide.leftCols(3));

    EXPECT_TRUE(Near(result, Stack3::Zero(2, 3)));
}

TEST(BatchedCrossProductExpressions, AcceptsMaps)
{
    const double left_data[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const double right_data[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const Eigen::Map<const Eigen::Matrix<double, 2, 3>> left(left_data);
    const Eigen::Map<const Eigen::Matrix<double, 2, 3>> right(right_data);

    EXPECT_TRUE(Near(batched_cross_product(left, right),
                     loop_cross_product(Stack3(left), Stack3(right))));
}

// ---------------------------------------------------------------------------
// batched_cross_product alongside the other operators
// ---------------------------------------------------------------------------

// A proper rotation commutes with the cross product, so rotating both inputs
// is the same as rotating the result.
TEST(BatchedCrossProduct, CommutesWithACommonRotation)
{
    const Eigen::Index count = 4;
    std::vector<Eigen::Matrix3d> rotations;
    VecBatch<3> left(3, count);
    VecBatch<3> right(3, count);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        rotations.push_back(
            Eigen::AngleAxisd(0.4 * static_cast<double>(i + 1),
                              Eigen::Vector3d(1, 2, 3).normalized())
                .toRotationMatrix());
        left.col(i) = Eigen::Vector3d(1.0 + i, 2.0 - i, 0.5 * i);
        right.col(i) = Eigen::Vector3d(0.5 * i, 1.0, 3.0 - i);
    }

    const Stack3 rotated_left = batched_matrix_vector(rotations, left);
    const Stack3 rotated_right = batched_matrix_vector(rotations, right);

    const Stack3 cross_then_rotate = batched_matrix_vector(
        rotations,
        VecBatch<3>(batched_cross_product(Stack3(left.transpose()),
                                          Stack3(right.transpose()))
                        .transpose()));
    const Stack3 rotate_then_cross =
        batched_cross_product(rotated_left, rotated_right);

    EXPECT_TRUE(Near(cross_then_rotate, rotate_then_cross, 1e-12));
}

// The scalar triple product is the determinant of the three stacked rows.
TEST(BatchedCrossProduct, ScalarTripleProductMatchesTheDeterminant)
{
    const Stack3 a = random_stack(5);
    const Stack3 b = random_stack(5);
    const Stack3 c = random_stack(5);

    const Eigen::VectorXd triple =
        batched_dot_product(a, Stack3(batched_cross_product(b, c)));

    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        Eigen::Matrix3d stacked;
        stacked.row(0) = a.row(i);
        stacked.row(1) = b.row(i);
        stacked.row(2) = c.row(i);

        EXPECT_NEAR(triple(i), stacked.determinant(), 1e-12) << "row " << i;
    }
}

}  // namespace
}  // namespace cosserat::math
