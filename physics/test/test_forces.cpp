#include "physics/forces.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <variant>
#include <vector>

namespace cosserat::physics {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kTol = 1e-12;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Minimal system satisfying the concepts. Force/torque rows are set
// independently so tests can pin the row-count conventions each rule assumes.
// ---------------------------------------------------------------------------

struct TestSystem
{
    std::uint64_t n_elements = 0;
    Eigen::VectorXd m_mass;
    Vector3DStack m_forces;
    Vector3DStack m_torques;
    Matrix3DStack m_frames;

    std::uint64_t num_elements() const { return n_elements; }
    const Eigen::VectorXd& mass() const { return m_mass; }
    Vector3DStack& external_forces() { return m_forces; }
    const Matrix3DStack& frames() const { return m_frames; }
    Vector3DStack& external_torques() { return m_torques; }
};

static_assert(ForceableSystem<TestSystem>);
static_assert(TorqueableSystem<TestSystem>);
static_assert(ForceableTorqueableSystem<TestSystem>);

// force_rows is separate from n_elements: UniformForce/GravityForce assume
// force_rows == n_elements + 1 (nodes), torques are per-element.
TestSystem make_system(std::uint64_t n_elements, Eigen::Index force_rows)
{
    TestSystem sys;
    sys.n_elements = n_elements;
    sys.m_mass = Eigen::VectorXd::Ones(force_rows);
    sys.m_forces = Vector3DStack::Zero(force_rows, 3);
    sys.m_torques = Vector3DStack::Zero(static_cast<Eigen::Index>(n_elements), 3);
    sys.m_frames.assign(n_elements, Eigen::Matrix3d::Identity());
    return sys;
}

// Distinct, valid rotations so tests can tell which frame was applied.
Matrix3DStack distinct_frames(std::size_t n)
{
    Matrix3DStack frames;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double angle = 0.3 * static_cast<double>(i + 1);
        const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
        frames.push_back(Eigen::AngleAxisd(angle, axis).toRotationMatrix());
    }
    return frames;
}

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                double tol = kTol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return ::testing::AssertionFailure() << "shape mismatch";
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err < tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs diff " << err << " >= " << tol;
}

// ---------------------------------------------------------------------------
// GravityForce
// ---------------------------------------------------------------------------

TEST(GravityForce, AppliesDownwardAlongChosenAxisOnly)
{
    auto sys = make_system(4, 5);
    GravityForce<ZTag>{}.apply_forces(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_forces.col(0), Eigen::VectorXd::Zero(5)));
    EXPECT_TRUE(Near(sys.m_forces.col(1), Eigen::VectorXd::Zero(5)));
    EXPECT_TRUE(Near(sys.m_forces.col(2),
                     Eigen::VectorXd::Constant(5, GravityForce<ZTag>::gravity)));
}

TEST(GravityForce, SignIsNegative)
{
    EXPECT_LT(GravityForce<ZTag>::gravity, 0.0);
    EXPECT_DOUBLE_EQ(GravityForce<ZTag>::gravity, -9.80665);
}

TEST(GravityForce, ScalesWithPerRowMass)
{
    auto sys = make_system(3, 4);
    sys.m_mass << 1.0, 2.0, 0.5, 4.0;

    GravityForce<YTag>{}.apply_forces(sys, 0.0);

    Eigen::VectorXd expected = GravityForce<YTag>::gravity * sys.m_mass;
    EXPECT_TRUE(Near(sys.m_forces.col(1), expected));
    EXPECT_TRUE(Near(sys.m_forces.col(0), Eigen::VectorXd::Zero(4)));
}

TEST(GravityForce, AccumulatesRatherThanOverwrites)
{
    auto sys = make_system(3, 4);
    sys.m_forces.setConstant(7.0);

    GravityForce<XTag>{}.apply_forces(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_forces.col(0),
                     Eigen::VectorXd::Constant(4, 7.0 + GravityForce<XTag>::gravity)));
    EXPECT_TRUE(Near(sys.m_forces.col(1), Eigen::VectorXd::Constant(4, 7.0)));
}

TEST(GravityForce, EachAxisTargetsItsOwnColumn)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        auto sys = make_system(2, 3);
        if (axis == 0) GravityForce<XTag>{}.apply_forces(sys, 0.0);
        if (axis == 1) GravityForce<YTag>{}.apply_forces(sys, 0.0);
        if (axis == 2) GravityForce<ZTag>{}.apply_forces(sys, 0.0);

        for (int c = 0; c < 3; ++c)
        {
            const double expected = (c == axis) ? GravityForce<XTag>::gravity : 0.0;
            EXPECT_TRUE(Near(sys.m_forces.col(c), Eigen::VectorXd::Constant(3, expected)))
                << "axis " << axis << " col " << c;
        }
    }
}

TEST(GravityForce, ApplyTorquesIsNoOp)
{
    auto sys = make_system(3, 4);
    GravityForce<ZTag>{}.apply_torques(sys, 0.0);
    EXPECT_TRUE(Near(sys.m_torques, Vector3DStack::Zero(3, 3)));
}

// ---------------------------------------------------------------------------
// EndpointForce
// ---------------------------------------------------------------------------

TEST(EndpointForce, TouchesOnlyFirstAndLastRows)
{
    auto sys = make_system(4, 5);
    const EndpointForce rule({1, 2, 3}, {4, 5, 6}, 0.0, 1.0, std::nullopt);

    rule.apply_forces(sys, 10.0);  // fully ramped

    EXPECT_TRUE(Near(sys.m_forces.row(0), Eigen::RowVector3d(1, 2, 3)));
    EXPECT_TRUE(Near(sys.m_forces.row(4), Eigen::RowVector3d(4, 5, 6)));
    EXPECT_TRUE(Near(sys.m_forces.middleRows(1, 3), Vector3DStack::Zero(3, 3)));
}

TEST(EndpointForce, ScalesWithRampFactor)
{
    const EndpointForce rule({2, 0, 0}, {0, 4, 0}, 1.0, 2.0, std::nullopt);

    auto before = make_system(3, 4);
    rule.apply_forces(before, 0.5);  // t <= onset
    EXPECT_TRUE(Near(before.m_forces, Vector3DStack::Zero(4, 3)));

    auto mid = make_system(3, 4);
    rule.apply_forces(mid, 2.0);  // halfway
    EXPECT_TRUE(Near(mid.m_forces.row(0), Eigen::RowVector3d(1, 0, 0)));
    EXPECT_TRUE(Near(mid.m_forces.row(3), Eigen::RowVector3d(0, 2, 0)));

    auto after = make_system(3, 4);
    rule.apply_forces(after, 100.0);  // saturated
    EXPECT_TRUE(Near(after.m_forces.row(0), Eigen::RowVector3d(2, 0, 0)));
    EXPECT_TRUE(Near(after.m_forces.row(3), Eigen::RowVector3d(0, 4, 0)));
}

TEST(EndpointForce, RespectsRampOffset)
{
    const EndpointForce rule({1, 0, 0}, {1, 0, 0}, 0.0, 1.0, 5.0);

    auto sys = make_system(3, 4);
    rule.apply_forces(sys, 6.0);  // past offset
    EXPECT_TRUE(Near(sys.m_forces, Vector3DStack::Zero(4, 3)));
}

TEST(EndpointForce, AccumulatesRatherThanOverwrites)
{
    auto sys = make_system(3, 4);
    sys.m_forces.setConstant(1.0);
    const EndpointForce rule({1, 0, 0}, {0, 1, 0}, 0.0, 1.0, std::nullopt);

    rule.apply_forces(sys, 10.0);

    EXPECT_TRUE(Near(sys.m_forces.row(0), Eigen::RowVector3d(2, 1, 1)));
    EXPECT_TRUE(Near(sys.m_forces.row(3), Eigen::RowVector3d(1, 2, 1)));
}

// With one row, first and last are the same row and both forces land on it.
TEST(EndpointForce, SingleRowReceivesBothForces)
{
    auto sys = make_system(1, 1);
    const EndpointForce rule({1, 0, 0}, {0, 1, 0}, 0.0, 1.0, std::nullopt);

    rule.apply_forces(sys, 10.0);

    EXPECT_TRUE(Near(sys.m_forces.row(0), Eigen::RowVector3d(1, 1, 0)));
}

TEST(EndpointForce, ExposesConstructorArguments)
{
    const EndpointForce rule({1, 2, 3}, {4, 5, 6}, 1.0, 2.0, 9.0);

    EXPECT_TRUE(Near(rule.first_link_force(), Eigen::Vector3d(1, 2, 3)));
    EXPECT_TRUE(Near(rule.last_link_force(), Eigen::Vector3d(4, 5, 6)));
    EXPECT_DOUBLE_EQ(rule.linear_ramp().onset_time(), 1.0);
    EXPECT_DOUBLE_EQ(rule.linear_ramp().ramp_time(), 2.0);
    ASSERT_TRUE(rule.linear_ramp().offset_time().has_value());
    EXPECT_DOUBLE_EQ(*rule.linear_ramp().offset_time(), 9.0);
}

TEST(EndpointForce, OneNonZeroForceIsEnough)
{
    EXPECT_NO_THROW({ EndpointForce({1, 0, 0}, {0, 0, 0}, 0.0, 1.0, std::nullopt); });
    EXPECT_NO_THROW({ EndpointForce({0, 0, 0}, {0, 1, 0}, 0.0, 1.0, std::nullopt); });
}

TEST(EndpointForceDeathTest, RejectsTwoNearZeroForces)
{
    EXPECT_ASSERT_FAILURE(
        EndpointForce({0, 0, 0}, {0, 0, 0}, 0.0, 1.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(
        EndpointForce({1e-15, 0, 0}, {0, 1e-15, 0}, 0.0, 1.0, std::nullopt));
}

TEST(EndpointForceDeathTest, PropagatesRampValidation)
{
    EXPECT_ASSERT_FAILURE(EndpointForce({1, 0, 0}, {1, 0, 0}, 0.0, 0.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(EndpointForce({1, 0, 0}, {1, 0, 0}, kNaN, 1.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(EndpointForce({1, 0, 0}, {1, 0, 0}, 0.0, 1.0, 0.5));
}

// ---------------------------------------------------------------------------
// UniformForce
// ---------------------------------------------------------------------------

// Total applied force equals the requested force only when the force rows are
// nodes (num_elements + 1). Half-weighting the ends is what makes it balance.
TEST(UniformForce, TotalEqualsRequestedForceOnNodeRows)
{
    for (std::uint64_t n : {2u, 5u, 20u})
    {
        auto sys = make_system(n, static_cast<Eigen::Index>(n) + 1);
        UniformForce({3.0, -1.5, 0.75}).apply_forces(sys, 0.0);

        EXPECT_TRUE(Near(sys.m_forces.colwise().sum().transpose(),
                         Eigen::Vector3d(3.0, -1.5, 0.75), 1e-12))
            << "n_elements = " << n;
    }
}

TEST(UniformForce, InteriorRowsShareEquallyAndEndsAreHalved)
{
    const std::uint64_t n = 5;
    auto sys = make_system(n, 6);
    UniformForce({1.0, 0.0, 0.0}).apply_forces(sys, 0.0);

    const double scale = 1.0 / static_cast<double>(n);
    for (Eigen::Index r = 1; r < 5; ++r)
    {
        EXPECT_TRUE(Near(sys.m_forces.row(r), Eigen::RowVector3d(scale, 0, 0)))
            << "row " << r;
    }
    EXPECT_TRUE(Near(sys.m_forces.row(0), Eigen::RowVector3d(0.5 * scale, 0, 0)));
    EXPECT_TRUE(Near(sys.m_forces.row(5), Eigen::RowVector3d(0.5 * scale, 0, 0)));
}

// Documents the shortfall if force rows are per-element instead of per-node.
TEST(UniformForce, ElementRowCountUnderApplies)
{
    const std::uint64_t n = 10;
    auto sys = make_system(n, static_cast<Eigen::Index>(n));
    UniformForce({1.0, 0.0, 0.0}).apply_forces(sys, 0.0);

    const double total = sys.m_forces.col(0).sum();
    EXPECT_NEAR(total, 0.9, 1e-12);  // (n - 1) / n
}

TEST(UniformForce, AccumulatesRatherThanOverwrites)
{
    auto sys = make_system(2, 3);
    sys.m_forces.setConstant(1.0);
    UniformForce({2.0, 0.0, 0.0}).apply_forces(sys, 0.0);

    EXPECT_NEAR(sys.m_forces.col(0).sum(), 3.0 + 2.0, 1e-12);
    EXPECT_NEAR(sys.m_forces.col(1).sum(), 3.0, 1e-12);
}

TEST(UniformForce, ExposesForce)
{
    EXPECT_TRUE(Near(UniformForce({1, 2, 3}).force(), Eigen::Vector3d(1, 2, 3)));
}

TEST(UniformForceDeathTest, RejectsNearZeroForce)
{
    EXPECT_ASSERT_FAILURE(UniformForce({0, 0, 0}));
    EXPECT_ASSERT_FAILURE(UniformForce({1e-15, 0, 0}));
}

// ---------------------------------------------------------------------------
// EndpointForceSinusoidal
//
// Follows PyElastica: before onset a constant -2 * magnitude * normal is
// applied, after onset the force rotates in the normal/roll plane. The jump
// at onset is inherited from that implementation, not an oversight.
// ---------------------------------------------------------------------------

const Eigen::Vector3d kNormal(0, 0, 1);
const Eigen::Vector3d kTangent(1, 0, 0);
const Eigen::Vector3d kRoll(0, 1, 0);  // normal x tangent

TEST(EndpointForceSinusoidal, RollDirectionIsNormalCrossTangent)
{
    const EndpointForceSinusoidal rule(kNormal, kTangent, 1.0, 1.0, 1.0);
    EXPECT_TRUE(Near(rule.roll_direction(), kRoll));
    EXPECT_TRUE(Near(rule.normal_direction(), kNormal));
    EXPECT_TRUE(Near(rule.tangent_direction(), kTangent));
    EXPECT_DOUBLE_EQ(rule.onset_time(), 1.0);
}

TEST(EndpointForceSinusoidal, BeforeOnsetAppliesNegativeTwiceNormal)
{
    auto sys = make_system(4, 5);
    const EndpointForceSinusoidal rule(kNormal, kTangent, 1.0, 3.0, 10.0);

    rule.apply_forces(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_forces.row(0), (-2.0 * 1.0 * kNormal).transpose()));
    EXPECT_TRUE(Near(sys.m_forces.row(4), (-2.0 * 3.0 * kNormal).transpose()));
    EXPECT_TRUE(Near(sys.m_forces.middleRows(1, 3), Vector3DStack::Zero(3, 3)));
}

// cos_arg = 0.5 * pi * (t - onset): t=onset -> roll, +1 -> normal, +2 -> -roll.
TEST(EndpointForceSinusoidal, RotatesInNormalRollPlaneAfterOnset)
{
    const double onset = 1.0;
    const EndpointForceSinusoidal rule(kNormal, kTangent, 2.0, 2.0, onset);

    auto at_onset = make_system(3, 4);
    rule.apply_forces(at_onset, onset);
    EXPECT_TRUE(Near(at_onset.m_forces.row(0), (2.0 * kRoll).transpose()));

    auto quarter = make_system(3, 4);
    rule.apply_forces(quarter, onset + 1.0);
    EXPECT_TRUE(Near(quarter.m_forces.row(0), (2.0 * kNormal).transpose()));

    auto half = make_system(3, 4);
    rule.apply_forces(half, onset + 2.0);
    EXPECT_TRUE(Near(half.m_forces.row(0), (-2.0 * kRoll).transpose()));
}

TEST(EndpointForceSinusoidal, MagnitudeIsPreservedAfterOnset)
{
    auto sys = make_system(3, 4);
    const EndpointForceSinusoidal rule(kNormal, kTangent, 5.0, 5.0, 0.0);

    rule.apply_forces(sys, 0.37);

    EXPECT_NEAR(sys.m_forces.row(0).norm(), 5.0, 1e-12);
    EXPECT_NEAR(sys.m_forces.row(3).norm(), 5.0, 1e-12);
}

TEST(EndpointForceSinusoidal, TouchesOnlyEndpoints)
{
    auto sys = make_system(5, 6);
    const EndpointForceSinusoidal rule(kNormal, kTangent, 1.0, 1.0, 0.0);

    rule.apply_forces(sys, 2.0);

    EXPECT_TRUE(Near(sys.m_forces.middleRows(1, 4), Vector3DStack::Zero(4, 3)));
}

TEST(EndpointForceSinusoidalDeathTest, RejectsNonUnitDirections)
{
    EXPECT_ASSERT_FAILURE(
        EndpointForceSinusoidal({0, 0, 2}, kTangent, 1.0, 1.0, 0.0));
    EXPECT_ASSERT_FAILURE(
        EndpointForceSinusoidal(kNormal, {3, 0, 0}, 1.0, 1.0, 0.0));
}

// A non-unit cross product means the inputs were not orthogonal.
TEST(EndpointForceSinusoidalDeathTest, RejectsNonOrthogonalDirections)
{
    const Eigen::Vector3d skewed = Eigen::Vector3d(1, 0, 1).normalized();
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, skewed, 1.0, 1.0, 0.0));
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, kNormal, 1.0, 1.0, 0.0));
}

TEST(EndpointForceSinusoidalDeathTest, RejectsBadMagnitudesAndOnset)
{
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, kTangent, 0.0, 1.0, 0.0));
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, kTangent, -1.0, 1.0, 0.0));
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, kTangent, 1.0, kInf, 0.0));
    EXPECT_ASSERT_FAILURE(EndpointForceSinusoidal(kNormal, kTangent, 1.0, 1.0, kNaN));
}

// ---------------------------------------------------------------------------
// UniformTorque
// ---------------------------------------------------------------------------

TEST(UniformTorque, DistributesEvenlyInIdentityFrames)
{
    const std::uint64_t n = 4;
    auto sys = make_system(n, 5);
    const Eigen::Vector3d torque(0.0, 0.0, 2.0);

    UniformTorque(torque).apply_torques(sys, 0.0);

    for (Eigen::Index r = 0; r < 4; ++r)
    {
        EXPECT_TRUE(Near(sys.m_torques.row(r), (torque / 4.0).transpose())) << "row " << r;
    }
    EXPECT_TRUE(Near(sys.m_torques.colwise().sum().transpose(), torque));
}

TEST(UniformTorque, RotatesIntoEachMaterialFrame)
{
    const std::uint64_t n = 4;
    auto sys = make_system(n, 5);
    sys.m_frames = distinct_frames(n);
    const Eigen::Vector3d torque(1.0, 2.0, 3.0);

    UniformTorque(torque).apply_torques(sys, 0.0);

    for (Eigen::Index r = 0; r < 4; ++r)
    {
        const Eigen::Vector3d expected = sys.m_frames[r] * torque / 4.0;
        EXPECT_TRUE(Near(sys.m_torques.row(r), expected.transpose())) << "row " << r;
    }
}

TEST(UniformTorque, AccumulatesRatherThanOverwrites)
{
    auto sys = make_system(2, 3);
    sys.m_torques.setConstant(1.0);

    UniformTorque({0, 0, 2}).apply_torques(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_torques.row(0), Eigen::RowVector3d(1, 1, 2)));
}

TEST(UniformTorque, ApplyForcesIsNoOp)
{
    auto sys = make_system(3, 4);
    UniformTorque({0, 0, 1}).apply_forces(sys, 0.0);
    EXPECT_TRUE(Near(sys.m_forces, Vector3DStack::Zero(4, 3)));
}

TEST(UniformTorque, ExposesTorque)
{
    EXPECT_TRUE(Near(UniformTorque({1, 2, 3}).torque(), Eigen::Vector3d(1, 2, 3)));
}

TEST(UniformTorqueDeathTest, RejectsNearZeroTorque)
{
    EXPECT_ASSERT_FAILURE(UniformTorque({0, 0, 0}));
    EXPECT_ASSERT_FAILURE(UniformTorque({0, 1e-15, 0}));
}

// ---------------------------------------------------------------------------
// MuscleTorque::determine_spline
// ---------------------------------------------------------------------------

TEST(DetermineSpline, EmptyCoefficientsGiveZeros)
{
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(6, 0.0, 1.0);
    const Eigen::VectorXd result = MuscleTorque::determine_spline(Eigen::VectorXd(0), s);

    ASSERT_EQ(result.size(), 6);
    EXPECT_TRUE(Near(result, Eigen::VectorXd::Zero(6)));
}

TEST(DetermineSpline, OutputSizeMatchesCoords)
{
    Eigen::VectorXd b(6);
    b << 0.0, 1.0, 2.0, 3.0, 1.0, 0.0;

    for (Eigen::Index n : {1, 5, 17})
    {
        const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(n, 0.0, 1.0);
        EXPECT_EQ(MuscleTorque::determine_spline(b, s).size(), n);
    }
}

// Clamped knot vector: the curve interpolates its first and last coefficient.
TEST(DetermineSpline, ClampsToEndpointCoefficients)
{
    Eigen::VectorXd b(6);
    b << 3.0, 1.0, -2.0, 5.0, 0.5, 7.0;
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);

    const Eigen::VectorXd result = MuscleTorque::determine_spline(b, s);

    EXPECT_NEAR(result(0), b(0), 1e-12);
    EXPECT_NEAR(result(result.size() - 1), b(b.size() - 1), 1e-12);
}

TEST(DetermineSpline, DependsOnCoefficientValues)
{
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(7, 0.0, 1.0);
    Eigen::VectorXd b1(6);
    b1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Eigen::VectorXd b2 = b1;
    b2(3) = -20.0;

    const Eigen::VectorXd r1 = MuscleTorque::determine_spline(b1, s);
    const Eigen::VectorXd r2 = MuscleTorque::determine_spline(b2, s);

    EXPECT_GT((r1 - r2).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(DetermineSpline, ConstantCoefficientsGiveConstantCurve)
{
    const Eigen::VectorXd b = Eigen::VectorXd::Constant(6, 2.5);
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(9, 0.0, 1.0);

    const Eigen::VectorXd result = MuscleTorque::determine_spline(b, s);

    EXPECT_TRUE(Near(result, Eigen::VectorXd::Constant(9, 2.5), 1e-10));
}

// degree + 1 control points is the minimum for a clamped cubic.
TEST(DetermineSpline, AcceptsExactlyFourCoefficients)
{
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(4, 0.0, 1.0);
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    EXPECT_NO_THROW({ MuscleTorque::determine_spline(b, s); });
}

TEST(DetermineSplineDeathTest, RejectsTooFewCoefficients)
{
    const Eigen::VectorXd s = Eigen::VectorXd::LinSpaced(5, 0.0, 1.0);
    EXPECT_ASSERT_FAILURE(
        MuscleTorque::determine_spline(Eigen::VectorXd::Ones(1), s));
    EXPECT_ASSERT_FAILURE(
        MuscleTorque::determine_spline(Eigen::VectorXd::Ones(3), s));
}

// ---------------------------------------------------------------------------
// MuscleTorque
// ---------------------------------------------------------------------------

MuscleTorque make_muscle(double onset = 0.0, double ramp = 1.0,
                         std::optional<double> offset = std::nullopt)
{
    Eigen::VectorXd rest_lengths = Eigen::VectorXd::Ones(5);
    Eigen::VectorXd b_coeffs(6);
    b_coeffs << 0.0, 1.0, 2.0, 2.0, 1.0, 0.0;
    return MuscleTorque({0, 0, 1}, 2.0, 3.0, 0.5, onset, ramp, offset,
                        rest_lengths, b_coeffs);
}

TEST(MuscleTorque, ExposesConstructorArguments)
{
    const MuscleTorque rule = make_muscle();

    EXPECT_TRUE(Near(rule.direction(), Eigen::Vector3d(0, 0, 1)));
    EXPECT_DOUBLE_EQ(rule.angular_frequency(), 2.0);
    EXPECT_DOUBLE_EQ(rule.wave_number(), 3.0);
    EXPECT_DOUBLE_EQ(rule.phase_shift(), 0.5);
    EXPECT_EQ(rule.normalized_coords().size(), 5);
    EXPECT_EQ(rule.spline().size(), 5);
    EXPECT_DOUBLE_EQ(rule.normalized_coords()(4), 1.0);
}

TEST(MuscleTorque, ZeroRampFactorLeavesTorquesUntouched)
{
    auto sys = make_system(5, 6);
    const MuscleTorque rule = make_muscle(/*onset=*/10.0, /*ramp=*/1.0);

    rule.apply_torques(sys, 0.0);  // before onset

    EXPECT_TRUE(Near(sys.m_torques, Vector3DStack::Zero(5, 3)));
}

TEST(MuscleTorque, PastOffsetLeavesTorquesUntouched)
{
    auto sys = make_system(5, 6);
    const MuscleTorque rule = make_muscle(0.0, 1.0, /*offset=*/2.0);

    rule.apply_torques(sys, 5.0);

    EXPECT_TRUE(Near(sys.m_torques, Vector3DStack::Zero(5, 3)));
}

TEST(MuscleTorque, ApplyForcesIsNoOp)
{
    auto sys = make_system(5, 6);
    make_muscle().apply_torques(sys, 0.5);
    auto forces_sys = make_system(5, 6);
    make_muscle().apply_forces(forces_sys, 0.5);
    EXPECT_TRUE(Near(forces_sys.m_forces, Vector3DStack::Zero(6, 3)));
}

// Recomputes the full torque expression independently, including the reversal
// of torque_mag and the frame pairing of both terms. The second term pairs
// torque columns [1, N) with frames [0, N-1): ignore_size_mismatch truncates
// from the front, so the leading frames are used, not frames[1:]. This mirrors
// the current PyElastica-derived implementation and is pinned here so that
// revisiting the pairing shows up as a failure rather than a silent change.
TEST(MuscleTorque, MatchesIndependentlyComputedTorques)
{
    const std::uint64_t n = 5;
    auto sys = make_system(n, 6);
    sys.m_frames = distinct_frames(n);

    const MuscleTorque rule = make_muscle(0.0, 1.0, std::nullopt);
    const double time = 0.75;
    rule.apply_torques(sys, time);

    const double factor = rule.linear_ramp()(time);
    const Eigen::VectorXd& s = rule.normalized_coords();
    const Eigen::VectorXd& spline = rule.spline();

    Eigen::VectorXd torque_mag(n);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n); ++i)
    {
        const double arg = rule.angular_frequency() * time + rule.phase_shift()
                           - rule.wave_number() * s(i);
        torque_mag(i) = factor * spline(i) * std::sin(arg);
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> torque(3, n);
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); ++j)
    {
        torque.col(j) = rule.direction() * torque_mag(n - 1 - j);
    }

    Vector3DStack expected = Vector3DStack::Zero(n, 3);
    for (Eigen::Index j = 1; j < static_cast<Eigen::Index>(n); ++j)
    {
        expected.row(j) += (sys.m_frames[j] * torque.col(j)).transpose();
    }
    for (Eigen::Index j = 0; j + 1 < static_cast<Eigen::Index>(n); ++j)
    {
        expected.row(j) -= (sys.m_frames[j] * torque.col(j + 1)).transpose();
    }

    EXPECT_TRUE(Near(sys.m_torques, expected, 1e-12));
}

// The mismatched pairing is observable: shifting the frames by one changes
// the result. Remove this test if the pairing is corrected to frames[1:].
TEST(MuscleTorque, SecondTermUsesLeadingFrames)
{
    const std::uint64_t n = 5;
    const MuscleTorque rule = make_muscle();

    auto sys = make_system(n, 6);
    sys.m_frames = distinct_frames(n);
    rule.apply_torques(sys, 0.75);

    auto shifted = make_system(n, 6);
    shifted.m_frames = distinct_frames(n + 1);
    shifted.m_frames.erase(shifted.m_frames.begin());
    rule.apply_torques(shifted, 0.75);

    EXPECT_GT((sys.m_torques - shifted.m_torques).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(MuscleTorque, AccumulatesRatherThanOverwrites)
{
    const std::uint64_t n = 5;
    auto fresh = make_system(n, 6);
    auto seeded = make_system(n, 6);
    seeded.m_torques.setConstant(1.0);

    const MuscleTorque rule = make_muscle();
    rule.apply_torques(fresh, 0.75);
    rule.apply_torques(seeded, 0.75);

    EXPECT_TRUE(Near(seeded.m_torques,
                     Vector3DStack(fresh.m_torques + Vector3DStack::Ones(n, 3))));
}

TEST(MuscleTorqueDeathTest, RejectsNonUnitDirection)
{
    Eigen::VectorXd rest = Eigen::VectorXd::Ones(5);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(6);
    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 2}, 1.0, 1.0, 0.0, 0.0, 1.0, std::nullopt, rest, b));
}

TEST(MuscleTorqueDeathTest, RejectsBadFrequencyWaveNumberAndPhase)
{
    Eigen::VectorXd rest = Eigen::VectorXd::Ones(5);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(6);

    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, 0.0, 1.0, 0.0, 0.0, 1.0, std::nullopt, rest, b));
    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, kInf, 1.0, 0.0, 0.0, 1.0, std::nullopt, rest, b));
    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, 1.0, -1.0, 0.0, 0.0, 1.0, std::nullopt, rest, b));
    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, 1.0, 1.0, kNaN, 0.0, 1.0, std::nullopt, rest, b));
}

TEST(MuscleTorqueDeathTest, PropagatesRestLengthValidation)
{
    Eigen::VectorXd b = Eigen::VectorXd::Ones(6);
    Eigen::VectorXd bad(3);
    bad << 1.0, -2.0, 1.0;

    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, 1.0, 1.0, 0.0, 0.0, 1.0, std::nullopt, bad, b));
    EXPECT_ASSERT_FAILURE(
        MuscleTorque({0, 0, 1}, 1.0, 1.0, 0.0, 0.0, 1.0, std::nullopt,
                     Eigen::VectorXd(0), b));
}


// ---------------------------------------------------------------------------
// ForceTorqueVariant dispatch
//
// Each rule pairs one concept-constrained method with one unconstrained no-op,
// so the `requires` probe in apply_forces/apply_torques selects the right
// concept per alternative without naming it. validate() requires BOTH methods
// to be callable, which is what makes it reject a rule the body cannot accept.
// ---------------------------------------------------------------------------

// Satisfies ForceableSystem only.
struct ForceOnlySystem
{
    std::uint64_t n_elements = 4;
    Eigen::VectorXd m_mass = Eigen::VectorXd::Ones(5);
    Vector3DStack m_forces = Vector3DStack::Zero(5, 3);

    std::uint64_t num_elements() const { return n_elements; }
    const Eigen::VectorXd& mass() const { return m_mass; }
    Vector3DStack& external_forces() { return m_forces; }
};

// Satisfies TorqueableSystem only.
struct TorqueOnlySystem
{
    std::uint64_t n_elements = 4;
    Vector3DStack m_torques = Vector3DStack::Zero(4, 3);
    Matrix3DStack m_frames = Matrix3DStack(4, Eigen::Matrix3d::Identity());

    std::uint64_t num_elements() const { return n_elements; }
    const Matrix3DStack& frames() const { return m_frames; }
    Vector3DStack& external_torques() { return m_torques; }
};

static_assert(ForceableSystem<ForceOnlySystem>);
static_assert(!TorqueableSystem<ForceOnlySystem>);
static_assert(TorqueableSystem<TorqueOnlySystem>);
static_assert(!ForceableSystem<TorqueOnlySystem>);

ForceTorqueVariant make_muscle_variant()
{
    return ForceTorqueVariant{make_muscle()};
}

// Every alternative, for loops that must cover the whole variant.
std::vector<ForceTorqueVariant> all_force_alternatives()
{
    return {
        ForceTorqueVariant{GravityForceX{}},
        ForceTorqueVariant{GravityForceY{}},
        ForceTorqueVariant{GravityForceZ{}},
        ForceTorqueVariant{EndpointForce({1, 0, 0}, {0, 1, 0}, 0.0, 1.0, std::nullopt)},
        ForceTorqueVariant{UniformForce({1, 0, 0})},
        ForceTorqueVariant{EndpointForceSinusoidal(kNormal, kTangent, 1.0, 1.0, 0.0)},
    };
}

std::vector<ForceTorqueVariant> all_torque_alternatives()
{
    return {
        ForceTorqueVariant{UniformTorque({0, 0, 1})},
        make_muscle_variant(),
    };
}

// --- variant value semantics -----------------------------------------------

TEST(ForceTorqueVariant, IsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<ForceTorqueVariant>);
    static_assert(std::is_copy_assignable_v<ForceTorqueVariant>);
    static_assert(std::is_move_assignable_v<ForceTorqueVariant>);

    ForceTorqueVariant a{GravityForceZ{}};
    const ForceTorqueVariant b{UniformForce({1, 0, 0})};

    a = b;

    EXPECT_EQ(a.index(), b.index());
    EXPECT_TRUE(std::holds_alternative<UniformForce>(a));
}

TEST(ForceTorqueVariant, WorksInStandardContainers)
{
    std::vector<ForceTorqueVariant> rules = all_force_alternatives();
    ASSERT_EQ(rules.size(), 6u);

    rules.push_back(ForceTorqueVariant{UniformTorque({0, 0, 1})});
    rules.erase(rules.begin());

    EXPECT_EQ(rules.size(), 6u);
    EXPECT_TRUE(std::holds_alternative<GravityForceY>(rules.front()));
}

TEST(ForceTorqueVariant, HoldsTheAlternativeItWasGiven)
{
    ForceTorqueVariant v{EndpointForce({1, 2, 3}, {4, 5, 6}, 0.0, 1.0, std::nullopt)};

    ASSERT_TRUE(std::holds_alternative<EndpointForce>(v));
    EXPECT_TRUE(Near(std::get<EndpointForce>(v).first_link_force(),
                     Eigen::Vector3d(1, 2, 3)));
}

// The three gravity axes are distinct variant alternatives.
TEST(ForceTorqueVariant, GravityAxesAreDistinctAlternatives)
{
    const ForceTorqueVariant x{GravityForceX{}};
    const ForceTorqueVariant y{GravityForceY{}};
    const ForceTorqueVariant z{GravityForceZ{}};

    EXPECT_NE(x.index(), y.index());
    EXPECT_NE(y.index(), z.index());
    EXPECT_TRUE(std::holds_alternative<GravityForceX>(x));
    EXPECT_TRUE(std::holds_alternative<GravityForceZ>(z));
}

// --- validate ---------------------------------------------------------------

TEST(ValidateVariant, AcceptsEveryAlternativeOnAFullSystem)
{
    auto sys = make_system(4, 5);

    for (auto& v : all_force_alternatives())  EXPECT_NO_THROW({ validate(v, sys); });
    for (auto& v : all_torque_alternatives()) EXPECT_NO_THROW({ validate(v, sys); });
}

TEST(ValidateVariant, AcceptsForceRulesOnForceOnlySystem)
{
    ForceOnlySystem sys;
    for (auto& v : all_force_alternatives()) EXPECT_NO_THROW({ validate(v, sys); });
}

TEST(ValidateVariant, AcceptsTorqueRulesOnTorqueOnlySystem)
{
    TorqueOnlySystem sys;
    for (auto& v : all_torque_alternatives()) EXPECT_NO_THROW({ validate(v, sys); });
}

TEST(ValidateVariantDeathTest, RejectsTorqueRulesOnForceOnlySystem)
{
    ForceOnlySystem sys;
    ForceTorqueVariant uniform{UniformTorque({0, 0, 1})};
    ForceTorqueVariant muscle = make_muscle_variant();

    EXPECT_ASSERT_FAILURE(validate(uniform, sys));
    EXPECT_ASSERT_FAILURE(validate(muscle, sys));
}

TEST(ValidateVariantDeathTest, RejectsForceRulesOnTorqueOnlySystem)
{
    TorqueOnlySystem sys;
    ForceTorqueVariant gravity{GravityForceZ{}};
    ForceTorqueVariant uniform{UniformForce({1, 0, 0})};

    EXPECT_ASSERT_FAILURE(validate(gravity, sys));
    EXPECT_ASSERT_FAILURE(validate(uniform, sys));
}

// --- apply_forces dispatch --------------------------------------------------

TEST(ApplyForcesVariant, MatchesDirectGravityCall)
{
    auto through_variant = make_system(4, 5);
    auto direct = make_system(4, 5);

    ForceTorqueVariant v{GravityForceZ{}};
    apply_forces(v, through_variant, 0.0);
    GravityForceZ{}.apply_forces(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_forces, direct.m_forces));
}

TEST(ApplyForcesVariant, MatchesDirectEndpointCall)
{
    auto through_variant = make_system(4, 5);
    auto direct = make_system(4, 5);
    const EndpointForce rule({2, 0, 0}, {0, 4, 0}, 1.0, 2.0, std::nullopt);

    ForceTorqueVariant v{rule};
    apply_forces(v, through_variant, 2.0);
    rule.apply_forces(direct, 2.0);

    EXPECT_TRUE(Near(through_variant.m_forces, direct.m_forces));
}

TEST(ApplyForcesVariant, DispatchesToTheHeldAxisForGravity)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        auto sys = make_system(2, 3);
        ForceTorqueVariant v = (axis == 0) ? ForceTorqueVariant{GravityForceX{}}
                             : (axis == 1) ? ForceTorqueVariant{GravityForceY{}}
                                           : ForceTorqueVariant{GravityForceZ{}};
        apply_forces(v, sys, 0.0);

        for (int c = 0; c < 3; ++c)
        {
            const double expected = (c == axis) ? GravityForceX::gravity : 0.0;
            EXPECT_TRUE(Near(sys.m_forces.col(c), Eigen::VectorXd::Constant(3, expected)))
                << "axis " << axis << " col " << c;
        }
    }
}

// Torque rules reach their unconstrained apply_forces no-op, not the assert.
TEST(ApplyForcesVariant, TorqueRulesAreNoOps)
{
    auto sys = make_system(4, 5);

    for (auto& v : all_torque_alternatives())
    {
        EXPECT_NO_THROW({ apply_forces(v, sys, 0.5); });
    }
    EXPECT_TRUE(Near(sys.m_forces, Vector3DStack::Zero(5, 3)));
}

TEST(ApplyForcesVariant, EveryForceAlternativeChangesAFullSystem)
{
    for (auto& v : all_force_alternatives())
    {
        auto sys = make_system(4, 5);
        apply_forces(v, sys, 2.0);
        EXPECT_GT(sys.m_forces.cwiseAbs().maxCoeff(), 0.0)
            << "alternative index " << v.index();
    }
}

TEST(ApplyForcesVariantDeathTest, RejectsForceRulesOnTorqueOnlySystem)
{
    TorqueOnlySystem sys;
    ForceTorqueVariant gravity{GravityForceZ{}};

    EXPECT_ASSERT_FAILURE(apply_forces(gravity, sys, 0.0));
}

// --- apply_torques dispatch -------------------------------------------------

TEST(ApplyTorquesVariant, MatchesDirectUniformTorqueCall)
{
    auto through_variant = make_system(4, 5);
    auto direct = make_system(4, 5);
    through_variant.m_frames = distinct_frames(4);
    direct.m_frames = distinct_frames(4);

    ForceTorqueVariant v{UniformTorque({1, 2, 3})};
    apply_torques(v, through_variant, 0.0);
    UniformTorque({1, 2, 3}).apply_torques(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_torques, direct.m_torques));
}

TEST(ApplyTorquesVariant, MatchesDirectMuscleTorqueCall)
{
    auto through_variant = make_system(5, 6);
    auto direct = make_system(5, 6);
    through_variant.m_frames = distinct_frames(5);
    direct.m_frames = distinct_frames(5);

    ForceTorqueVariant v = make_muscle_variant();
    apply_torques(v, through_variant, 0.75);
    make_muscle().apply_torques(direct, 0.75);

    EXPECT_TRUE(Near(through_variant.m_torques, direct.m_torques));
}

// Force rules reach their unconstrained apply_torques no-op, not the assert.
TEST(ApplyTorquesVariant, ForceRulesAreNoOps)
{
    auto sys = make_system(4, 5);

    for (auto& v : all_force_alternatives())
    {
        EXPECT_NO_THROW({ apply_torques(v, sys, 0.5); });
    }
    EXPECT_TRUE(Near(sys.m_torques, Vector3DStack::Zero(4, 3)));
}

TEST(ApplyTorquesVariantDeathTest, RejectsTorqueRulesOnForceOnlySystem)
{
    ForceOnlySystem sys;
    ForceTorqueVariant uniform{UniformTorque({0, 0, 1})};

    EXPECT_ASSERT_FAILURE(apply_torques(uniform, sys, 0.0));
}

// --- combined dispatch ------------------------------------------------------

// A rule list applied via both entry points accumulates every contribution.
TEST(VariantDispatch, ForceAndTorqueRulesAccumulateAcrossAList)
{
    auto sys = make_system(4, 5);
    sys.m_frames = distinct_frames(4);

    std::vector<ForceTorqueVariant> rules{
        ForceTorqueVariant{GravityForceZ{}},
        ForceTorqueVariant{UniformForce({1, 0, 0})},
        ForceTorqueVariant{UniformTorque({0, 0, 1})},
    };

    for (auto& rule : rules) validate(rule, sys);
    for (auto& rule : rules) apply_forces(rule, sys, 1.0);
    for (auto& rule : rules) apply_torques(rule, sys, 1.0);

    auto expected = make_system(4, 5);
    expected.m_frames = distinct_frames(4);
    GravityForceZ{}.apply_forces(expected, 1.0);
    UniformForce({1, 0, 0}).apply_forces(expected, 1.0);
    UniformTorque({0, 0, 1}).apply_torques(expected, 1.0);

    EXPECT_TRUE(Near(sys.m_forces, expected.m_forces));
    EXPECT_TRUE(Near(sys.m_torques, expected.m_torques));
}

// Dispatch follows the alternative currently held, not the one first assigned.
TEST(VariantDispatch, FollowsReassignedAlternative)
{
    auto sys = make_system(2, 3);
    ForceTorqueVariant v{GravityForceX{}};
    v = ForceTorqueVariant{GravityForceZ{}};

    apply_forces(v, sys, 0.0);

    EXPECT_TRUE(Near(sys.m_forces.col(0), Eigen::VectorXd::Zero(3)));
    EXPECT_TRUE(Near(sys.m_forces.col(2),
                     Eigen::VectorXd::Constant(3, GravityForceZ::gravity)));
}

}  // namespace
}  // namespace cosserat::physics