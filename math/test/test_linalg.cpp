#include "math/linalg.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <vector>

namespace cosserat::math {
namespace {

constexpr double kTol = 1e-12;

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
}  // namespace
}  // namespace cosserat::math