#include "physics/damping.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
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
// Stub systems. Velocities and forces are per-node (n_elements + 1); angular
// velocities, torques, dilatations and inertia are per-element.
// ---------------------------------------------------------------------------

struct RodStub
{
    std::uint64_t n_elements = 4;
    Eigen::VectorXd m_mass;
    Eigen::VectorXd m_rest_lengths;
    Eigen::VectorXd m_dilatations;
    Matrix3DStack m_inv_moments;
    Vector3DStack m_velocities;
    Vector3DStack m_angular_velocities;
    Vector3DStack m_forces;
    Vector3DStack m_torques;

    std::uint64_t num_elements() const { return n_elements; }
    const Eigen::VectorXd& mass() const { return m_mass; }
    const Eigen::VectorXd& rest_lengths() const { return m_rest_lengths; }
    const Eigen::VectorXd& dilatations() const { return m_dilatations; }
    const Matrix3DStack& inv_mass_second_moments() const { return m_inv_moments; }
    Vector3DStack& velocities() { return m_velocities; }
    Vector3DStack& angular_velocities() { return m_angular_velocities; }
    Vector3DStack& external_forces() { return m_forces; }
    Vector3DStack& external_torques() { return m_torques; }
};

// Rates only: no inertia, no rest lengths, no external accumulators.
struct RatesOnlySystem
{
    std::uint64_t n_elements = 4;
    Vector3DStack m_velocities = Vector3DStack::Zero(5, 3);
    Vector3DStack m_angular_velocities = Vector3DStack::Zero(4, 3);

    std::uint64_t num_elements() const { return n_elements; }
    Vector3DStack& velocities() { return m_velocities; }
    Vector3DStack& angular_velocities() { return m_angular_velocities; }
};

static_assert(DampableSystem<RodStub>);
static_assert(InertiallyDampableSystem<RodStub>);
static_assert(RayleighDampableSystem<RodStub>);
static_assert(DampableSystem<RatesOnlySystem>);
static_assert(!InertiallyDampableSystem<RatesOnlySystem>);
static_assert(!RayleighDampableSystem<RatesOnlySystem>);

RodStub make_rod(std::uint64_t n_elements = 4)
{
    const auto n = static_cast<Eigen::Index>(n_elements);
    RodStub rod;
    rod.n_elements = n_elements;
    rod.m_mass = Eigen::VectorXd::LinSpaced(n + 1, 1.0, 1.0 + n);
    rod.m_rest_lengths = Eigen::VectorXd::Constant(n, 0.25);
    rod.m_dilatations = Eigen::VectorXd::LinSpaced(n, 1.0, 1.3);
    rod.m_velocities = Vector3DStack::Zero(n + 1, 3);
    rod.m_angular_velocities = Vector3DStack::Zero(n, 3);
    rod.m_forces = Vector3DStack::Zero(n + 1, 3);
    rod.m_torques = Vector3DStack::Zero(n, 3);
    for (Eigen::Index i = 0; i < n + 1; ++i)
    {
        rod.m_velocities.row(i) << 1.0 + i, 2.0 + i, 3.0 + i;
    }
    for (Eigen::Index i = 0; i < n; ++i)
    {
        rod.m_angular_velocities.row(i) << 0.5 + i, 1.5 + i, 2.5 + i;
        Eigen::Matrix3d moment = Eigen::Matrix3d::Zero();
        moment.diagonal() << 1.0 + i, 2.0 + i, 3.0 + i;
        rod.m_inv_moments.push_back(moment);
    }
    return rod;
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
// Free helpers
// ---------------------------------------------------------------------------

TEST(InverseInertiaDiagonals, ExtractsPerElementDiagonal)
{
    const RodStub rod = make_rod(3);
    const Vector3DStack diagonals = inverse_inertia_diagonals(rod.m_inv_moments);

    ASSERT_EQ(diagonals.rows(), 3);
    ASSERT_EQ(diagonals.cols(), 3);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(Near(diagonals.row(i),
                         Eigen::RowVector3d(1.0 + i, 2.0 + i, 3.0 + i)));
    }
}

TEST(InverseInertiaDiagonals, IgnoresOffDiagonalEntries)
{
    Matrix3DStack moments;
    Eigen::Matrix3d m;
    m << 1.0, 9.0, 9.0,
         9.0, 2.0, 9.0,
         9.0, 9.0, 3.0;
    moments.push_back(m);

    const Vector3DStack diagonals = inverse_inertia_diagonals(moments);

    EXPECT_TRUE(Near(diagonals.row(0), Eigen::RowVector3d(1.0, 2.0, 3.0)));
}

TEST(InverseInertiaDiagonalsDeathTest, RejectsEmptyInput)
{
    EXPECT_ASSERT_FAILURE(inverse_inertia_diagonals(Matrix3DStack{}));
}

TEST(ElementMasses, InteriorElementsAverageTheirNodes)
{
    Eigen::VectorXd nodal(5);
    nodal << 1.0, 2.0, 3.0, 4.0, 5.0;

    const Eigen::VectorXd element = element_masses_from_nodal(nodal);

    ASSERT_EQ(element.size(), 4);
    EXPECT_DOUBLE_EQ(element(1), 2.5);  // (2 + 3) / 2
    EXPECT_DOUBLE_EQ(element(2), 3.5);  // (3 + 4) / 2
}

// End elements pick up the leftover half node, so total mass is preserved.
TEST(ElementMasses, ConserveTotalMass)
{
    Eigen::VectorXd nodal(6);
    nodal << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    const Eigen::VectorXd element = element_masses_from_nodal(nodal);

    EXPECT_DOUBLE_EQ(element.sum(), nodal.sum());
    EXPECT_DOUBLE_EQ(element(0), 0.5 * (1.0 + 2.0) + 0.5 * 1.0);
    EXPECT_DOUBLE_EQ(element(4), 0.5 * (5.0 + 6.0) + 0.5 * 6.0);
}

TEST(ElementMassesDeathTest, RejectsFewerThanTwoNodes)
{
    EXPECT_ASSERT_FAILURE(element_masses_from_nodal(Eigen::VectorXd::Ones(1)));
    EXPECT_ASSERT_FAILURE(element_masses_from_nodal(Eigen::VectorXd(0)));
}

// ---------------------------------------------------------------------------
// UniformAnalyticalDamper
// ---------------------------------------------------------------------------

TEST(UniformAnalyticalDamper, CoefficientIsExponentialDecay)
{
    const UniformAnalyticalDamper damper(0.5, 0.1);

    EXPECT_DOUBLE_EQ(damper.coefficient(), std::exp(-0.5 * 0.1));
    EXPECT_DOUBLE_EQ(damper.damping_constant(), 0.5);
    EXPECT_DOUBLE_EQ(damper.time_step(), 0.1);
}

TEST(UniformAnalyticalDamper, ScalesBothRatesByTheSameFactor)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;
    const Vector3DStack w0 = rod.m_angular_velocities;

    const UniformAnalyticalDamper damper(0.5, 0.1);
    damper.dampen_rates(rod, 0.0);

    const double c = std::exp(-0.5 * 0.1);
    EXPECT_TRUE(Near(rod.m_velocities, Vector3DStack(v0 * c)));
    EXPECT_TRUE(Near(rod.m_angular_velocities, Vector3DStack(w0 * c)));
}

TEST(UniformAnalyticalDamper, ZeroConstantLeavesRatesUnchanged)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;

    UniformAnalyticalDamper(0.0, 0.1).dampen_rates(rod, 0.0);

    EXPECT_DOUBLE_EQ(UniformAnalyticalDamper(0.0, 0.1).coefficient(), 1.0);
    EXPECT_TRUE(Near(rod.m_velocities, v0));
}

TEST(UniformAnalyticalDamper, RepeatedApplicationCompounds)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;
    const UniformAnalyticalDamper damper(0.5, 0.1);

    damper.dampen_rates(rod, 0.0);
    damper.dampen_rates(rod, 0.1);
    damper.dampen_rates(rod, 0.2);

    const double c = std::exp(-0.5 * 0.1);
    EXPECT_TRUE(Near(rod.m_velocities, Vector3DStack(v0 * c * c * c)));
}

// Unconditionally stable: rates decay toward zero and never change sign.
TEST(UniformAnalyticalDamper, LargeConstantStaysStable)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;

    const UniformAnalyticalDamper damper(1e6, 0.1);
    damper.dampen_rates(rod, 0.0);

    EXPECT_LT(rod.m_velocities.cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_TRUE((rod.m_velocities.array() >= 0.0).all());
    EXPECT_GT(v0.cwiseAbs().maxCoeff(), 0.0);
}

TEST(UniformAnalyticalDamper, WorksOnRatesOnlySystem)
{
    RatesOnlySystem sys;
    sys.m_velocities.setConstant(2.0);

    UniformAnalyticalDamper(0.5, 0.1).dampen_rates(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_velocities,
                     Vector3DStack::Constant(5, 3, 2.0 * std::exp(-0.05))));
}

TEST(UniformAnalyticalDamperDeathTest, RejectsBadConstants)
{
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(-1.0, 0.1));
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(kNaN, 0.1));
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(kInf, 0.1));
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(0.5, 0.0));
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(0.5, -0.1));
    EXPECT_ASSERT_FAILURE(UniformAnalyticalDamper(0.5, kInf));
}

// ---------------------------------------------------------------------------
// PhysicalAnalyticalDamper
// ---------------------------------------------------------------------------

TEST(PhysicalAnalyticalDamper, TranslationalCoefficientDividesByNodalMass)
{
    const RodStub rod = make_rod();

    const Eigen::VectorXd expected =
        (-0.4 * 0.1 * rod.m_mass.array().inverse()).exp();
    EXPECT_TRUE(Near(
        PhysicalAnalyticalDamper::translational_coefficients(rod.m_mass, 0.4, 0.1),
        expected));
}

TEST(PhysicalAnalyticalDamper, RotationalCoefficientUsesInverseInertia)
{
    const RodStub rod = make_rod();

    const Vector3DStack inv = inverse_inertia_diagonals(rod.m_inv_moments);
    const Vector3DStack expected = (-0.3 * 0.1 * inv.array()).exp();
    EXPECT_TRUE(Near(
        PhysicalAnalyticalDamper::rotational_coefficients(rod.m_inv_moments, 0.3, 0.1),
        expected));
}

TEST(PhysicalAnalyticalDamper, HeavierNodesDampLess)
{
    const RodStub rod = make_rod();
    const Eigen::VectorXd coefficients =
        PhysicalAnalyticalDamper::translational_coefficients(rod.m_mass, 0.4, 0.1);

    // make_rod gives strictly increasing nodal mass.
    for (Eigen::Index i = 1; i < coefficients.size(); ++i)
    {
        EXPECT_GT(coefficients(i), coefficients(i - 1)) << "node " << i;
    }
}

TEST(PhysicalAnalyticalDamper, AppliesPerNodeAndPerElementCoefficients)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;
    const Vector3DStack w0 = rod.m_angular_velocities;

    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);
    damper.dampen_rates(rod, 0.0);

    const Eigen::VectorXd translational =
        PhysicalAnalyticalDamper::translational_coefficients(rod.m_mass, 0.4, 0.1);
    const Vector3DStack rotational =
        PhysicalAnalyticalDamper::rotational_coefficients(rod.m_inv_moments, 0.3, 0.1);

    Vector3DStack expected_v = v0;
    for (Eigen::Index i = 0; i < expected_v.rows(); ++i)
    {
        expected_v.row(i) *= translational(i);
    }
    EXPECT_TRUE(Near(rod.m_velocities, expected_v));

    Vector3DStack expected_w = w0;
    for (Eigen::Index i = 0; i < expected_w.rows(); ++i)
    {
        expected_w.row(i).array() *=
            rotational.row(i).array().pow(rod.m_dilatations(i));
    }
    EXPECT_TRUE(Near(rod.m_angular_velocities, expected_w));
}

// Dilatation enters as an exponent on the rotational coefficient.
TEST(PhysicalAnalyticalDamper, DilatationChangesRotationalDamping)
{
    RodStub stretched = make_rod();
    RodStub unit = make_rod();
    unit.m_dilatations.setOnes();

    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);
    damper.dampen_rates(stretched, 0.0);
    damper.dampen_rates(unit, 0.0);

    EXPECT_GT((stretched.m_angular_velocities - unit.m_angular_velocities)
                  .cwiseAbs().maxCoeff(), 1e-6);
    // Row 0 has dilatation exactly 1.0 in both, so it must agree.
    EXPECT_TRUE(Near(stretched.m_angular_velocities.row(0),
                     unit.m_angular_velocities.row(0)));
}

TEST(PhysicalAnalyticalDamper, ZeroConstantsLeaveRatesUnchanged)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;
    const Vector3DStack w0 = rod.m_angular_velocities;

    PhysicalAnalyticalDamper(0.0, 0.0, 0.1).dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities, v0));
    EXPECT_TRUE(Near(rod.m_angular_velocities, w0));
}

TEST(PhysicalAnalyticalDamper, ExposesConstructorArguments)
{
    RodStub rod = make_rod();
    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);

    EXPECT_DOUBLE_EQ(damper.translational_damping_constant(), 0.4);
    EXPECT_DOUBLE_EQ(damper.rotational_damping_constant(), 0.3);
    EXPECT_DOUBLE_EQ(damper.time_step(), 0.1);
}

TEST(PhysicalAnalyticalDamperDeathTest, RejectsBadConstants)
{
    RodStub rod = make_rod();

    EXPECT_ASSERT_FAILURE(PhysicalAnalyticalDamper(-1.0, 0.3, 0.1));
    EXPECT_ASSERT_FAILURE(PhysicalAnalyticalDamper(0.4, -1.0, 0.1));
    EXPECT_ASSERT_FAILURE(PhysicalAnalyticalDamper(kNaN, 0.3, 0.1));
    EXPECT_ASSERT_FAILURE(PhysicalAnalyticalDamper(0.4, 0.3, 0.0));
}

// Mass is validated where it is read, which is now at application time.
TEST(PhysicalAnalyticalDamperDeathTest, RejectsNonPositiveMass)
{
    RodStub rod = make_rod();
    rod.m_mass(2) = 0.0;
    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);

    EXPECT_ASSERT_FAILURE(damper.dampen_rates(rod, 0.0));

    RodStub negative = make_rod();
    negative.m_mass(1) = -1.0;
    EXPECT_ASSERT_FAILURE(damper.dampen_rates(negative, 0.0));
}

// Nothing is cached, so one damper serves systems of differing size.
TEST(PhysicalAnalyticalDamper, SharedAcrossDifferentlySizedSystems)
{
    RodStub small = make_rod(4);
    RodStub large = make_rod(9);
    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);

    EXPECT_NO_THROW({ damper.dampen_rates(small, 0.0); });
    EXPECT_NO_THROW({ damper.dampen_rates(large, 0.0); });

    EXPECT_EQ(small.m_velocities.rows(), 5);
    EXPECT_EQ(large.m_velocities.rows(), 10);
}

// A system whose own stacks disagree is still rejected.
TEST(PhysicalAnalyticalDamperDeathTest, RejectsInconsistentSystem)
{
    RodStub rod = make_rod(4);
    rod.m_velocities = Vector3DStack::Zero(3, 3);  // no longer one row per node
    const PhysicalAnalyticalDamper damper(0.4, 0.3, 0.1);

    EXPECT_ASSERT_FAILURE(damper.dampen_rates(rod, 0.0));
}

// ---------------------------------------------------------------------------
// LegacyAnalyticalDamper
//
// The original protocol: one constant for both translation and rotation. The
// two uses have different dimensions, so the constant cannot be dimensionally
// consistent for both. Kept for validating older cases.
// ---------------------------------------------------------------------------

TEST(LegacyAnalyticalDamper, TranslationalCoefficientIsScalar)
{
    RodStub rod = make_rod();
    const LegacyAnalyticalDamper damper(0.5, 0.1);

    EXPECT_DOUBLE_EQ(damper.translational_coefficient(), std::exp(-0.5 * 0.1));
}

TEST(LegacyAnalyticalDamper, RotationalCoefficientUsesElementMassAndInertia)
{
    const RodStub rod = make_rod();

    const Eigen::VectorXd element_mass = element_masses_from_nodal(rod.m_mass);
    const Vector3DStack inv = inverse_inertia_diagonals(rod.m_inv_moments);
    const Vector3DStack expected =
        (-0.5 * 0.1 * (inv.array().colwise() * element_mass.array())).exp();

    EXPECT_TRUE(Near(
        LegacyAnalyticalDamper::rotational_coefficients(
            rod.m_mass, rod.m_inv_moments, 0.5, 0.1),
        expected));
}

TEST(LegacyAnalyticalDamper, ScalesVelocitiesUniformly)
{
    RodStub rod = make_rod();
    const Vector3DStack v0 = rod.m_velocities;

    const LegacyAnalyticalDamper damper(0.5, 0.1);
    damper.dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities,
                     Vector3DStack(v0 * std::exp(-0.5 * 0.1))));
}

TEST(LegacyAnalyticalDamper, AppliesDilatationExponentToRotation)
{
    RodStub rod = make_rod();
    const Vector3DStack w0 = rod.m_angular_velocities;

    const LegacyAnalyticalDamper damper(0.5, 0.1);
    damper.dampen_rates(rod, 0.0);

    const Vector3DStack rotational = LegacyAnalyticalDamper::rotational_coefficients(
        rod.m_mass, rod.m_inv_moments, 0.5, 0.1);
    Vector3DStack expected = w0;
    for (Eigen::Index i = 0; i < expected.rows(); ++i)
    {
        expected.row(i).array() *=
            rotational.row(i).array().pow(rod.m_dilatations(i));
    }
    EXPECT_TRUE(Near(rod.m_angular_velocities, expected));
}

// Differs from the physical protocol: same constant, different treatment.
TEST(LegacyAnalyticalDamper, DiffersFromPhysicalProtocol)
{
    RodStub legacy_rod = make_rod();
    RodStub physical_rod = make_rod();

    LegacyAnalyticalDamper(0.5, 0.1).dampen_rates(legacy_rod, 0.0);
    PhysicalAnalyticalDamper(0.5, 0.5, 0.1)
        .dampen_rates(physical_rod, 0.0);

    EXPECT_GT((legacy_rod.m_velocities - physical_rod.m_velocities)
                  .cwiseAbs().maxCoeff(), 1e-6);
}

TEST(LegacyAnalyticalDamper, ExposesConstructorArguments)
{
    RodStub rod = make_rod();
    const LegacyAnalyticalDamper damper(0.5, 0.1);

    EXPECT_DOUBLE_EQ(damper.damping_constant(), 0.5);
    EXPECT_DOUBLE_EQ(damper.time_step(), 0.1);
}

TEST(LegacyAnalyticalDamperDeathTest, RejectsBadConstants)
{
    RodStub rod = make_rod();

    EXPECT_ASSERT_FAILURE(LegacyAnalyticalDamper(-1.0, 0.1));
    EXPECT_ASSERT_FAILURE(LegacyAnalyticalDamper(kNaN, 0.1));
    EXPECT_ASSERT_FAILURE(LegacyAnalyticalDamper(0.5, 0.0));
}

// ---------------------------------------------------------------------------
// RayleighDissipation
//
// Force-based: writes into the external accumulators rather than scaling rates,
// so it is not unconditionally stable. Kept for validating older cases.
// ---------------------------------------------------------------------------

TEST(RayleighDissipation, AverageElementLengthFromRestLengths)
{
    RodStub rod = make_rod();
    rod.m_rest_lengths << 0.1, 0.2, 0.3, 0.4;

    EXPECT_DOUBLE_EQ(
        RayleighDissipation::average_element_length(rod.m_rest_lengths), 0.25);
}

TEST(RayleighDissipation, ForceOpposesVelocity)
{
    RodStub rod = make_rod();
    const RayleighDissipation damper(0.2);

    damper.dampen_rates(rod, 0.0);

    for (Eigen::Index i = 0; i < rod.m_forces.rows(); ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_LT(rod.m_forces(i, c) * rod.m_velocities(i, c), 0.0)
                << "node " << i << " component " << c;
        }
    }
}

// Endpoints get half the interior damping, matching the reference model.
TEST(RayleighDissipation, EndpointsReceiveHalfDamping)
{
    RodStub rod = make_rod();
    rod.m_velocities.setConstant(1.0);
    const RayleighDissipation damper(0.2);

    damper.dampen_rates(rod, 0.0);

    const double nu = 0.2 * RayleighDissipation::average_element_length(rod.m_rest_lengths);
    const Eigen::Index last = rod.m_forces.rows() - 1;

    EXPECT_TRUE(Near(rod.m_forces.row(0), Eigen::RowVector3d::Constant(-0.5 * nu)));
    EXPECT_TRUE(Near(rod.m_forces.row(last), Eigen::RowVector3d::Constant(-0.5 * nu)));
    EXPECT_TRUE(Near(rod.m_forces.row(1), Eigen::RowVector3d::Constant(-nu)));
    EXPECT_TRUE(Near(rod.m_forces.row(2), Eigen::RowVector3d::Constant(-nu)));
}

TEST(RayleighDissipation, TorquesAreUniformAcrossElements)
{
    RodStub rod = make_rod();
    rod.m_angular_velocities.setConstant(2.0);
    const RayleighDissipation damper(0.2);

    damper.dampen_rates(rod, 0.0);

    const double nu = 0.2 * RayleighDissipation::average_element_length(rod.m_rest_lengths);
    EXPECT_TRUE(Near(rod.m_torques, Vector3DStack::Constant(4, 3, -2.0 * nu)));
}

TEST(RayleighDissipation, AccumulatesRatherThanOverwrites)
{
    RodStub rod = make_rod();
    rod.m_forces.setConstant(1.0);
    rod.m_torques.setConstant(1.0);
    const Vector3DStack v0 = rod.m_velocities;

    RodStub fresh = make_rod();
    const RayleighDissipation damper(0.2);
    damper.dampen_rates(rod, 0.0);
    damper.dampen_rates(fresh, 0.0);

    EXPECT_TRUE(Near(rod.m_forces,
                     Vector3DStack(fresh.m_forces + Vector3DStack::Ones(5, 3))));
    EXPECT_TRUE(Near(rod.m_velocities, v0));  // rates untouched
}

TEST(RayleighDissipation, WithoutRelaxationDampingIsTimeInvariant)
{
    RodStub rod = make_rod();
    const RayleighDissipation damper(0.2);

    EXPECT_FALSE(damper.relaxation_time().has_value());
    EXPECT_DOUBLE_EQ(damper.damping_at(0.0), damper.damping_at(100.0));
    EXPECT_DOUBLE_EQ(damper.damping_at(0.0), 0.2);
}

// Relaxation restores the exponential decay the reference model describes but
// leaves unreachable (its relaxation time is hardcoded to zero).
TEST(RayleighDissipation, RelaxationDecaysDampingOverTime)
{
    RodStub rod = make_rod();
    const RayleighDissipation damper(0.2, /*relaxation_time=*/2.0);

    const double at_zero = damper.damping_at(0.0);
    const double at_two = damper.damping_at(2.0);

    EXPECT_DOUBLE_EQ(at_zero, 0.2);
    EXPECT_DOUBLE_EQ(at_two, at_zero * std::exp(-1.0));
    EXPECT_LT(damper.damping_at(100.0), at_two);
}

TEST(RayleighDissipation, ZeroConstantLeavesAccumulatorsUntouched)
{
    RodStub rod = make_rod();
    RayleighDissipation(0.0).dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_forces, Vector3DStack::Zero(5, 3)));
    EXPECT_TRUE(Near(rod.m_torques, Vector3DStack::Zero(4, 3)));
}

TEST(RayleighDissipationDeathTest, RejectsBadConstants)
{
    RodStub rod = make_rod();

    EXPECT_ASSERT_FAILURE(RayleighDissipation(-1.0));
    EXPECT_ASSERT_FAILURE(RayleighDissipation{kNaN});
    EXPECT_ASSERT_FAILURE(RayleighDissipation(0.2, 0.0));
    EXPECT_ASSERT_FAILURE(RayleighDissipation(0.2, -1.0));
    EXPECT_ASSERT_FAILURE(RayleighDissipation(0.2, kInf));
}

TEST(RayleighDissipationDeathTest, RejectsNonFiniteTime)
{
    RodStub rod = make_rod();
    const RayleighDissipation damper(0.2);

    EXPECT_ASSERT_FAILURE(damper.damping_at(kNaN));
    EXPECT_ASSERT_FAILURE(damper.damping_at(kInf));
}

// ---------------------------------------------------------------------------
// LaplaceDissipationFilter
// ---------------------------------------------------------------------------

TEST(LaplaceDissipationFilter, LeavesEndpointsUnchanged)
{
    RodStub rod = make_rod(8);
    rod.m_velocities.setRandom();
    const Vector3DStack v0 = rod.m_velocities;

    LaplaceDissipationFilter filter(3);
    filter.dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0), v0.row(0)));
    EXPECT_TRUE(Near(rod.m_velocities.row(v0.rows() - 1), v0.row(v0.rows() - 1)));
}

// The discrete Laplacian annihilates constants, so smooth modes survive.
TEST(LaplaceDissipationFilter, PreservesConstantField)
{
    RodStub rod = make_rod(8);
    rod.m_velocities.setConstant(3.0);

    LaplaceDissipationFilter(3).dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities, Vector3DStack::Constant(9, 3, 3.0)));
}

TEST(LaplaceDissipationFilter, PreservesLinearField)
{
    RodStub rod = make_rod(8);
    for (Eigen::Index i = 0; i < rod.m_velocities.rows(); ++i)
    {
        rod.m_velocities.row(i).setConstant(2.0 * static_cast<double>(i) + 1.0);
    }
    const Vector3DStack v0 = rod.m_velocities;

    LaplaceDissipationFilter(4).dampen_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities, v0, 1e-10));
}

// High-frequency content is what the filter is meant to remove.
TEST(LaplaceDissipationFilter, AttenuatesAlternatingMode)
{
    RodStub rod = make_rod(10);
    for (Eigen::Index i = 0; i < rod.m_velocities.rows(); ++i)
    {
        rod.m_velocities.row(i).setConstant((i % 2 == 0) ? 1.0 : -1.0);
    }
    const double before = rod.m_velocities.middleRows(1, 9).norm();

    LaplaceDissipationFilter(1).dampen_rates(rod, 0.0);

    const double after = rod.m_velocities.middleRows(1, 9).norm();
    EXPECT_LT(after, before);
}

// Higher order means weaker filtering, so the result stays closer to the input.
TEST(LaplaceDissipationFilter, HigherOrderFiltersLess)
{
    RodStub aggressive = make_rod(10);
    RodStub gentle = make_rod(10);
    for (Eigen::Index i = 0; i < aggressive.m_velocities.rows(); ++i)
    {
        const double value = (i % 2 == 0) ? 1.0 : -1.0;
        aggressive.m_velocities.row(i).setConstant(value);
        gentle.m_velocities.row(i).setConstant(value);
    }
    const Vector3DStack original = aggressive.m_velocities;

    LaplaceDissipationFilter(1).dampen_rates(aggressive, 0.0);
    LaplaceDissipationFilter(8).dampen_rates(gentle, 0.0);

    EXPECT_LT((gentle.m_velocities - original).norm(),
              (aggressive.m_velocities - original).norm());
}

TEST(LaplaceDissipationFilter, FiltersBothVelocityAndOmega)
{
    RodStub rod = make_rod(8);
    rod.m_velocities.setRandom();
    rod.m_angular_velocities.setRandom();
    const Vector3DStack v0 = rod.m_velocities;
    const Vector3DStack w0 = rod.m_angular_velocities;

    LaplaceDissipationFilter(2).dampen_rates(rod, 0.0);

    EXPECT_GT((rod.m_velocities - v0).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_GT((rod.m_angular_velocities - w0).cwiseAbs().maxCoeff(), 0.0);
}

// Fewer than three rows means no interior to filter.
TEST(LaplaceDissipationFilter, ShortSystemIsUnchanged)
{
    RatesOnlySystem sys;
    sys.m_velocities = Vector3DStack::Constant(2, 3, 5.0);
    sys.m_angular_velocities = Vector3DStack::Constant(1, 3, 7.0);
    const Vector3DStack v0 = sys.m_velocities;
    const Vector3DStack w0 = sys.m_angular_velocities;

    LaplaceDissipationFilter(3).dampen_rates(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_velocities, v0));
    EXPECT_TRUE(Near(sys.m_angular_velocities, w0));
}

TEST(LaplaceDissipationFilter, IsRepeatable)
{
    RodStub first = make_rod(8);
    RodStub second = make_rod(8);
    LaplaceDissipationFilter filter(3);

    filter.dampen_rates(first, 0.0);
    const Vector3DStack after_first = first.m_velocities;
    filter.dampen_rates(second, 0.0);

    EXPECT_TRUE(Near(second.m_velocities, after_first));
}

TEST(LaplaceDissipationFilter, ExposesFilterOrder)
{
    RodStub rod = make_rod();
    LaplaceDissipationFilter filter(5);

    EXPECT_EQ(filter.filter_order(), 5u);
    // Buffers stay empty until the first call reveals the system size.
    EXPECT_EQ(filter.velocity_filter_term().rows(), 0);

    filter.dampen_rates(rod, 0.0);
    EXPECT_EQ(filter.velocity_filter_term().rows(), 5);
    EXPECT_EQ(filter.omega_filter_term().rows(), 4);
}

TEST(LaplaceDissipationFilterDeathTest, RejectsZeroFilterOrder)
{
    RodStub rod = make_rod();
    EXPECT_ASSERT_FAILURE(LaplaceDissipationFilter{0u});
}

// Scratch buffers resize to whichever system arrives, so one filter serves
// systems of differing size.
TEST(LaplaceDissipationFilter, ResizesBuffersForDifferentlySizedSystems)
{
    RodStub small = make_rod(4);
    RodStub large = make_rod(9);
    LaplaceDissipationFilter filter(3);

    filter.dampen_rates(small, 0.0);
    EXPECT_EQ(filter.velocity_filter_term().rows(), 5);

    filter.dampen_rates(large, 0.0);
    EXPECT_EQ(filter.velocity_filter_term().rows(), 10);
    EXPECT_EQ(filter.omega_filter_term().rows(), 9);
}

// ---------------------------------------------------------------------------
// DamperVariant dispatch
// ---------------------------------------------------------------------------

DamperVariant make_uniform()  { return DamperVariant{UniformAnalyticalDamper(0.5, 0.1)}; }
DamperVariant make_physical()
{
    return DamperVariant{PhysicalAnalyticalDamper(0.4, 0.3, 0.1)};
}
DamperVariant make_legacy()
{
    return DamperVariant{LegacyAnalyticalDamper(0.5, 0.1)};
}
DamperVariant make_rayleigh()
{
    return DamperVariant{RayleighDissipation(0.2)};
}
DamperVariant make_laplace()
{
    return DamperVariant{LaplaceDissipationFilter(3)};
}

std::vector<DamperVariant> all_dampers()
{
    return {make_uniform(), make_physical(), make_legacy(),
            make_rayleigh(), make_laplace()};
}

TEST(DamperVariant, IsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<DamperVariant>);
    static_assert(std::is_copy_assignable_v<DamperVariant>);
    static_assert(std::is_move_assignable_v<DamperVariant>);

    RodStub rod = make_rod();
    DamperVariant a = make_uniform();
    const DamperVariant b = make_rayleigh();

    a = b;

    EXPECT_TRUE(std::holds_alternative<RayleighDissipation>(a));
}

TEST(DamperVariant, WorksInStandardContainers)
{
    RodStub rod = make_rod();
    std::vector<DamperVariant> dampers = all_dampers();

    ASSERT_EQ(dampers.size(), 5u);
    dampers.erase(dampers.begin());
    EXPECT_EQ(dampers.size(), 4u);
    EXPECT_TRUE(std::holds_alternative<PhysicalAnalyticalDamper>(dampers.front()));
}

TEST(ValidateDamper, AcceptsEveryAlternativeOnAFullRod)
{
    RodStub rod = make_rod();
    for (auto& damper : all_dampers())
    {
        EXPECT_NO_THROW({ validate(damper, rod); });
    }
}

TEST(ValidateDamper, AcceptsRateOnlyDampersOnRatesOnlySystem)
{
    RatesOnlySystem sys;
    DamperVariant uniform = make_uniform();
    DamperVariant laplace{LaplaceDissipationFilter(3)};

    EXPECT_NO_THROW({ validate(uniform, sys); });
    EXPECT_NO_THROW({ validate(laplace, sys); });
}

TEST(ValidateDamperDeathTest, RejectsInertialDampersOnRatesOnlySystem)
{
    RodStub rod = make_rod();
    RatesOnlySystem sys;

    DamperVariant physical = make_physical();
    DamperVariant legacy = make_legacy();
    DamperVariant rayleigh = make_rayleigh();

    EXPECT_ASSERT_FAILURE(validate(physical, sys));
    EXPECT_ASSERT_FAILURE(validate(legacy, sys));
    EXPECT_ASSERT_FAILURE(validate(rayleigh, sys));
}

TEST(DampenRatesVariant, MatchesDirectUniformCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();

    DamperVariant damper = make_uniform();
    dampen_rates(damper, through_variant, 0.0);
    UniformAnalyticalDamper(0.5, 0.1).dampen_rates(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_velocities, direct.m_velocities));
    EXPECT_TRUE(Near(through_variant.m_angular_velocities,
                     direct.m_angular_velocities));
}

TEST(DampenRatesVariant, MatchesDirectPhysicalCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();
    RodStub reference = make_rod();

    DamperVariant damper = make_physical();
    dampen_rates(damper, through_variant, 0.0);
    PhysicalAnalyticalDamper(0.4, 0.3, 0.1).dampen_rates(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_velocities, direct.m_velocities));
    EXPECT_TRUE(Near(through_variant.m_angular_velocities,
                     direct.m_angular_velocities));
}

TEST(DampenRatesVariant, MatchesDirectRayleighCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();
    RodStub reference = make_rod();

    DamperVariant damper = make_rayleigh();
    dampen_rates(damper, through_variant, 1.5);
    RayleighDissipation(0.2).dampen_rates(direct, 1.5);

    EXPECT_TRUE(Near(through_variant.m_forces, direct.m_forces));
    EXPECT_TRUE(Near(through_variant.m_torques, direct.m_torques));
}

TEST(DampenRatesVariantDeathTest, RejectsInertialDamperOnRatesOnlySystem)
{
    RodStub rod = make_rod();
    RatesOnlySystem sys;
    DamperVariant physical = make_physical();

    EXPECT_ASSERT_FAILURE(dampen_rates(physical, sys, 0.0));
}

// A damper list applied in sequence composes each contribution.
TEST(DampenRatesVariant, ComposesAcrossAList)
{
    RodStub rod = make_rod();
    RodStub expected = make_rod();
    RodStub reference = make_rod();

    std::vector<DamperVariant> dampers{make_uniform(), make_rayleigh()};

    for (auto& damper : dampers) validate(damper, rod);
    for (auto& damper : dampers) dampen_rates(damper, rod, 0.5);

    UniformAnalyticalDamper(0.5, 0.1).dampen_rates(expected, 0.5);
    RayleighDissipation(0.2).dampen_rates(expected, 0.5);

    EXPECT_TRUE(Near(rod.m_velocities, expected.m_velocities));
    EXPECT_TRUE(Near(rod.m_forces, expected.m_forces));
}

}  // namespace
}  // namespace cosserat::physics
