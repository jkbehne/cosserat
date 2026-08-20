#include "physics/rigid_body.hpp"

#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <type_traits>

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

Sphere make_sphere() { return Sphere(Eigen::Vector3d(1.0, 2.0, 3.0), 0.25, 1200.0); }

Cylinder make_cylinder()
{
    return Cylinder(Eigen::Vector3d(0.5, -1.0, 2.0), Eigen::Vector3d(0.0, 0.0, 1.0),
                    Eigen::Vector3d(1.0, 0.0, 0.0), 2.0, 0.1, 900.0);
}

// ---------------------------------------------------------------------------
// Domain shape
//
// A rigid body reports one node and one element so the rules written against a
// rod's domains apply to it unchanged.
// ---------------------------------------------------------------------------

TEST(RigidBody, ReportsASingleNodeAndElement)
{
    const Sphere sphere = make_sphere();

    EXPECT_EQ(sphere.num_nodes(), 1);
    EXPECT_EQ(sphere.num_elements(), 1);
}

TEST(RigidBody, EveryStackHoldsExactlyOneEntry)
{
    const Cylinder cylinder = make_cylinder();

    EXPECT_EQ(cylinder.positions().rows(), 1);
    EXPECT_EQ(cylinder.velocities().rows(), 1);
    EXPECT_EQ(cylinder.accelerations().rows(), 1);
    EXPECT_EQ(cylinder.internal_forces().rows(), 1);
    EXPECT_EQ(cylinder.external_forces().rows(), 1);
    EXPECT_EQ(cylinder.masses().size(), 1);
    EXPECT_EQ(cylinder.frames().size(), 1u);
    EXPECT_EQ(cylinder.mass_2nd_moments().size(), 1u);
    EXPECT_EQ(cylinder.inv_mass_2nd_moments().size(), 1u);
    EXPECT_EQ(cylinder.angular_velocities().rows(), 1);
    EXPECT_EQ(cylinder.angular_accelerations().rows(), 1);
    EXPECT_EQ(cylinder.internal_torques().rows(), 1);
    EXPECT_EQ(cylinder.external_torques().rows(), 1);
    EXPECT_EQ(cylinder.radii().size(), 1);
    EXPECT_EQ(cylinder.densities().size(), 1);
    EXPECT_EQ(cylinder.volumes().size(), 1);
    EXPECT_EQ(cylinder.lengths().size(), 1);
    EXPECT_EQ(cylinder.rest_lengths().size(), 1);
    EXPECT_EQ(cylinder.dilatations().size(), 1);
}

TEST(RigidBody, StartsAtRestWithNoLoads)
{
    const Sphere sphere = make_sphere();

    EXPECT_TRUE(Near(sphere.velocities(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(sphere.angular_velocities(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(sphere.accelerations(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(sphere.angular_accelerations(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(sphere.external_forces(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(sphere.external_torques(), Vector3DStack::Zero(1, 3)));
}

// A rigid body never deforms, so these two are fixed for its whole life.
TEST(RigidBody, RestLengthEqualsLengthAndDilatationIsOne)
{
    const Cylinder cylinder = make_cylinder();

    EXPECT_TRUE(Near(cylinder.rest_lengths(), cylinder.lengths()));
    EXPECT_DOUBLE_EQ(cylinder.dilatations()(0), 1.0);
}

// ---------------------------------------------------------------------------
// Sphere construction
// ---------------------------------------------------------------------------

TEST(Sphere, MassFollowsFromTheSphereVolume)
{
    const double radius = 0.25;
    const double density = 1200.0;
    const Sphere sphere(Eigen::Vector3d::Zero(), radius, density);

    const double expected_volume =
        (4.0 / 3.0) * std::numbers::pi * std::pow(radius, 3);

    EXPECT_NEAR(sphere.volume(), expected_volume, 1e-15);
    EXPECT_NEAR(sphere.total_mass(), expected_volume * density, 1e-12);
    EXPECT_NEAR(sphere.masses()(0), expected_volume * density, 1e-12);
    EXPECT_NEAR(sphere.radius(), radius, kTol);
    EXPECT_NEAR(sphere.density(), density, kTol);
}

// The reference implementation treats the diameter as the characteristic
// length of a sphere.
TEST(Sphere, LengthIsTheDiameter)
{
    const Sphere sphere = make_sphere();

    EXPECT_NEAR(sphere.length(), 2.0 * sphere.radius(), kTol);
}

// Exact for a uniform solid sphere, and isotropic.
TEST(Sphere, InertiaIsTwoFifthsMassRadiusSquared)
{
    const Sphere sphere = make_sphere();
    const double expected =
        0.4 * sphere.total_mass() * sphere.radius() * sphere.radius();

    const Eigen::Matrix3d& inertia = sphere.mass_2nd_moments()[0];
    EXPECT_TRUE(Near(inertia, expected * Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(Sphere, InverseInertiaInvertsIt)
{
    const Sphere sphere = make_sphere();

    EXPECT_TRUE(Near(sphere.mass_2nd_moments()[0] * sphere.inv_mass_2nd_moments()[0],
                     Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(Sphere, FrameIsTheIdentity)
{
    const Sphere sphere = make_sphere();

    EXPECT_TRUE(Near(sphere.frames()[0], Eigen::Matrix3d::Identity()));
    EXPECT_TRUE(math::is_orthogonal(sphere.frames()[0], 1e-12));
    EXPECT_TRUE(Near(sphere.tangent(), Eigen::Vector3d(0.0, 0.0, 1.0)));
}

TEST(Sphere, PositionIsTheCentre)
{
    const Eigen::Vector3d center(1.0, 2.0, 3.0);
    const Sphere sphere(center, 0.25, 1200.0);

    EXPECT_TRUE(Near(sphere.positions().row(0), center.transpose()));
    EXPECT_TRUE(Near(sphere.position_center_of_mass(), center));
}

TEST(SphereDeathTest, RejectsNonPositiveGeometry)
{
    const Eigen::Vector3d center = Eigen::Vector3d::Zero();

    EXPECT_ASSERT_FAILURE(Sphere(center, 0.0, 1000.0));
    EXPECT_ASSERT_FAILURE(Sphere(center, -1.0, 1000.0));
    EXPECT_ASSERT_FAILURE(Sphere(center, 0.25, 0.0));
    EXPECT_ASSERT_FAILURE(Sphere(center, 0.25, -1.0));
}

TEST(SphereDeathTest, RejectsNonFiniteInputs)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EXPECT_ASSERT_FAILURE(Sphere(Eigen::Vector3d(nan, 0.0, 0.0), 0.25, 1000.0));
    EXPECT_ASSERT_FAILURE(Sphere(Eigen::Vector3d::Zero(), inf, 1000.0));
}

// ---------------------------------------------------------------------------
// Cylinder construction
// ---------------------------------------------------------------------------

TEST(Cylinder, MassFollowsFromTheCylinderVolume)
{
    const double radius = 0.1;
    const double length = 2.0;
    const double density = 900.0;
    const Cylinder cylinder(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
                            Eigen::Vector3d::UnitX(), length, radius, density);

    const double expected_volume = std::numbers::pi * radius * radius * length;

    EXPECT_NEAR(cylinder.volume(), expected_volume, 1e-15);
    EXPECT_NEAR(cylinder.total_mass(), expected_volume * density, 1e-12);
    EXPECT_NEAR(cylinder.radius(), radius, kTol);
    EXPECT_NEAR(cylinder.length(), length, kTol);
}

// The body sits at its midpoint, half a length from the supplied end face.
TEST(Cylinder, PositionIsTheMidpoint)
{
    const Eigen::Vector3d start(0.5, -1.0, 2.0);
    const Eigen::Vector3d direction(0.0, 0.0, 1.0);
    const Cylinder cylinder(start, direction, Eigen::Vector3d::UnitX(),
                            2.0, 0.1, 900.0);

    EXPECT_TRUE(Near(cylinder.position_center_of_mass(),
                     Eigen::Vector3d(start + 1.0 * direction)));
}

TEST(Cylinder, FrameRowsAreNormalBinormalAndTangent)
{
    const Eigen::Vector3d direction(0.0, 0.0, 1.0);
    const Eigen::Vector3d normal(1.0, 0.0, 0.0);
    const Cylinder cylinder(Eigen::Vector3d::Zero(), direction, normal,
                            2.0, 0.1, 900.0);

    const Eigen::Matrix3d& frame = cylinder.frames()[0];
    EXPECT_TRUE(Near(frame.row(0).transpose(), normal));
    EXPECT_TRUE(Near(frame.row(1).transpose(), Eigen::Vector3d(direction.cross(normal))));
    EXPECT_TRUE(Near(frame.row(2).transpose(), direction));
    EXPECT_TRUE(math::is_orthogonal(frame, 1e-12));
    EXPECT_TRUE(Near(cylinder.tangent(), direction));
}

TEST(Cylinder, FrameIsOrthogonalForATiltedAxis)
{
    const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    const Eigen::Vector3d seed(0.0, 0.0, 1.0);
    const Eigen::Vector3d normal =
        (seed - seed.dot(direction) * direction).normalized();

    const Cylinder cylinder(Eigen::Vector3d::Zero(), direction, normal,
                            2.0, 0.1, 900.0);

    EXPECT_TRUE(math::is_orthogonal(cylinder.frames()[0], 1e-12));
    EXPECT_TRUE(Near(cylinder.tangent(), direction));
}

// The transverse entries come from the cross-section's second moment of area,
// which reproduces the reference implementation. A uniform solid cylinder
// would instead have m(3r^2 + L^2)/12 transversely; the missing mL^2/12 term
// is pinned here so the deviation is visible rather than assumed.
TEST(Cylinder, InertiaMatchesTheReferenceNotTheSolidCylinder)
{
    const double radius = 0.1;
    const double length = 2.0;
    const double density = 900.0;
    const Cylinder cylinder(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
                            Eigen::Vector3d::UnitX(), length, radius, density);

    const double area = std::numbers::pi * radius * radius;
    const double transverse = area * area / (4.0 * std::numbers::pi) * density * length;
    const double axial = 2.0 * transverse;

    const Eigen::Matrix3d& inertia = cylinder.mass_2nd_moments()[0];
    EXPECT_NEAR(inertia(0, 0), transverse, 1e-12);
    EXPECT_NEAR(inertia(1, 1), transverse, 1e-12);
    EXPECT_NEAR(inertia(2, 2), axial, 1e-12);

    // The axial entry is the correct m r^2 / 2 for a solid cylinder...
    const double mass = cylinder.total_mass();
    EXPECT_NEAR(inertia(2, 2), 0.5 * mass * radius * radius, 1e-12);
    // ...while the transverse entry is m r^2 / 4, short of the true
    // m (3 r^2 + L^2) / 12 by exactly m L^2 / 12.
    EXPECT_NEAR(inertia(0, 0), 0.25 * mass * radius * radius, 1e-12);
    const double solid_cylinder =
        mass * (3.0 * radius * radius + length * length) / 12.0;
    EXPECT_NEAR(solid_cylinder - inertia(0, 0), mass * length * length / 12.0, 1e-10);
}

TEST(Cylinder, InverseInertiaInvertsIt)
{
    const Cylinder cylinder = make_cylinder();

    EXPECT_TRUE(Near(
        cylinder.mass_2nd_moments()[0] * cylinder.inv_mass_2nd_moments()[0],
        Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(CylinderDeathTest, RejectsNonPositiveGeometry)
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const Eigen::Vector3d dir = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d nrm = Eigen::Vector3d::UnitX();

    EXPECT_ASSERT_FAILURE(Cylinder(zero, dir, nrm, 0.0, 0.1, 900.0));
    EXPECT_ASSERT_FAILURE(Cylinder(zero, dir, nrm, 2.0, 0.0, 900.0));
    EXPECT_ASSERT_FAILURE(Cylinder(zero, dir, nrm, 2.0, 0.1, 0.0));
}

// An addition over the reference, which checks only that the vectors have
// three entries and would otherwise build a non-rotation frame.
TEST(CylinderDeathTest, RejectsNonUnitOrNonOrthogonalDirections)
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    EXPECT_ASSERT_FAILURE(Cylinder(zero, Eigen::Vector3d(0.0, 0.0, 2.0),
                                   Eigen::Vector3d::UnitX(), 2.0, 0.1, 900.0));
    EXPECT_ASSERT_FAILURE(Cylinder(zero, Eigen::Vector3d::UnitZ(),
                                   Eigen::Vector3d(3.0, 0.0, 0.0), 2.0, 0.1, 900.0));
    EXPECT_ASSERT_FAILURE(Cylinder(zero, Eigen::Vector3d::UnitZ(),
                                   Eigen::Vector3d::UnitZ(), 2.0, 0.1, 900.0));
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

TEST(RigidBodyDynamics, HasNoInternalLoads)
{
    Cylinder cylinder = make_cylinder();

    cylinder.compute_internal_forces_and_torques(0.0);

    EXPECT_TRUE(Near(cylinder.internal_forces(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(cylinder.internal_torques(), Vector3DStack::Zero(1, 3)));
}

TEST(RigidBodyDynamics, AccelerationIsExternalForceOverMass)
{
    Sphere sphere = make_sphere();
    sphere.mutable_external_forces().row(0) << 10.0, -5.0, 2.0;

    sphere.update_accelerations(0.0, 1e-4);

    const Eigen::RowVector3d expected =
        Eigen::RowVector3d(10.0, -5.0, 2.0) / sphere.total_mass();
    EXPECT_TRUE(Near(sphere.accelerations().row(0), expected, 1e-12));
}

// Internal forces are always zero, so they contribute nothing here. This
// distinguishes the rigid body from the rod, which adds them.
TEST(RigidBodyDynamics, InternalForcesDoNotEnterTheAcceleration)
{
    Sphere sphere = make_sphere();
    sphere.mutable_external_forces().row(0) << 3.0, 0.0, 0.0;

    sphere.update_accelerations(0.0, 1e-4);

    EXPECT_NEAR(sphere.accelerations()(0, 0), 3.0 / sphere.total_mass(), 1e-12);
    EXPECT_NEAR(sphere.accelerations()(0, 1), 0.0, kTol);
}

TEST(RigidBodyDynamics, AngularAccelerationIncludesLagrangianTransport)
{
    Cylinder cylinder = make_cylinder();
    const Eigen::Vector3d omega(1.5, -0.7, 2.2);
    cylinder.mutable_angular_velocities().row(0) = omega.transpose();
    cylinder.mutable_external_torques().row(0) << 0.4, 0.9, -0.2;

    cylinder.update_accelerations(0.0, 1e-4);

    const Eigen::Matrix3d& inertia = cylinder.mass_2nd_moments()[0];
    const Eigen::Vector3d j_omega = inertia * omega;
    const Eigen::Vector3d transport = j_omega.cross(omega);
    const Eigen::Vector3d expected =
        cylinder.inv_mass_2nd_moments()[0]
        * (transport + Eigen::Vector3d(0.4, 0.9, -0.2));

    EXPECT_TRUE(Near(cylinder.angular_accelerations().row(0), expected.transpose(),
                     1e-10));
}

// With an isotropic inertia the transport term vanishes, because J w is
// parallel to w.
TEST(RigidBodyDynamics, TransportTermVanishesForAnIsotropicBody)
{
    Sphere sphere = make_sphere();
    sphere.mutable_angular_velocities().row(0) << 1.5, -0.7, 2.2;

    sphere.update_accelerations(0.0, 1e-4);

    EXPECT_LT(sphere.angular_accelerations().cwiseAbs().maxCoeff(), 1e-12);
}

// Spinning about a principal axis produces no transport torque even when the
// inertia is anisotropic.
TEST(RigidBodyDynamics, SpinAboutAPrincipalAxisIsSteady)
{
    Cylinder cylinder = make_cylinder();
    cylinder.mutable_angular_velocities().row(0) << 0.0, 0.0, 5.0;

    cylinder.update_accelerations(0.0, 1e-4);

    EXPECT_LT(cylinder.angular_accelerations().cwiseAbs().maxCoeff(), 1e-12);
}

TEST(RigidBodyDynamics, NoLoadsAndNoSpinGiveNoAcceleration)
{
    Cylinder cylinder = make_cylinder();

    cylinder.update_accelerations(0.0, 1e-4);

    EXPECT_LT(cylinder.accelerations().cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(cylinder.angular_accelerations().cwiseAbs().maxCoeff(), 1e-12);
}

TEST(RigidBodyDynamics, ZeroOutClearsOnlyTheExternalAccumulators)
{
    Cylinder cylinder = make_cylinder();
    cylinder.mutable_external_forces().setConstant(4.0);
    cylinder.mutable_external_torques().setConstant(6.0);
    cylinder.mutable_velocities().row(0) << 1.0, 2.0, 3.0;

    cylinder.zero_out_external_forces_and_torques(0.0 /* time */);

    EXPECT_TRUE(Near(cylinder.external_forces(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(cylinder.external_torques(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(cylinder.velocities().row(0), Eigen::RowVector3d(1.0, 2.0, 3.0)));
}

// ---------------------------------------------------------------------------
// Energies
// ---------------------------------------------------------------------------

TEST(RigidBodyEnergy, TranslationalEnergyIsHalfMassVelocitySquared)
{
    Sphere sphere = make_sphere();
    const Eigen::Vector3d velocity(0.3, 0.4, -0.5);
    sphere.mutable_velocities().row(0) = velocity.transpose();

    EXPECT_NEAR(sphere.translational_energy(),
                0.5 * sphere.total_mass() * velocity.squaredNorm(), 1e-12);
}

TEST(RigidBodyEnergy, RotationalEnergyUsesTheInertia)
{
    Cylinder cylinder = make_cylinder();
    const Eigen::Vector3d omega(1.5, -0.7, 2.2);
    cylinder.mutable_angular_velocities().row(0) = omega.transpose();

    const double expected =
        0.5 * omega.dot(cylinder.mass_2nd_moments()[0] * omega);

    EXPECT_NEAR(cylinder.rotational_energy(), expected, 1e-12);
}

TEST(RigidBodyEnergy, EnergiesAreZeroAtRest)
{
    const Sphere sphere = make_sphere();

    EXPECT_NEAR(sphere.translational_energy(), 0.0, kTol);
    EXPECT_NEAR(sphere.rotational_energy(), 0.0, kTol);
}

TEST(RigidBodyEnergy, EnergiesAreNeverNegative)
{
    Cylinder cylinder = make_cylinder();
    cylinder.mutable_velocities().row(0) << -1.0, -2.0, -3.0;
    cylinder.mutable_angular_velocities().row(0) << -1.0, 2.0, -3.0;

    EXPECT_GT(cylinder.translational_energy(), 0.0);
    EXPECT_GT(cylinder.rotational_energy(), 0.0);
}

// ---------------------------------------------------------------------------
// Reference parity
//
// Expected values come from a NumPy transcription of PyElastica's rigidbody
// module applied to the same two bodies.
// ---------------------------------------------------------------------------

TEST(ReferenceParity, SphereMatchesPyElastica)
{
    const Sphere sphere = make_sphere();

    EXPECT_NEAR(sphere.total_mass(), 78.5398163397, 1e-9);
    EXPECT_NEAR(sphere.volume(), 0.0654498469498, 1e-12);
    EXPECT_NEAR(sphere.length(), 0.5, 1e-12);
    EXPECT_NEAR(sphere.mass_2nd_moments()[0](0, 0), 1.96349540849, 1e-10);
    EXPECT_TRUE(Near(sphere.frames()[0], Eigen::Matrix3d::Identity()));
}

TEST(ReferenceParity, CylinderMatchesPyElastica)
{
    const Cylinder cylinder = make_cylinder();

    EXPECT_NEAR(cylinder.total_mass(), 56.5486677646, 1e-9);
    EXPECT_NEAR(cylinder.volume(), 0.0628318530718, 1e-12);
    EXPECT_TRUE(Near(cylinder.position_center_of_mass(),
                     Eigen::Vector3d(0.5, -1.0, 3.0), 1e-12));

    Eigen::Matrix3d expected_inertia = Eigen::Matrix3d::Zero();
    expected_inertia(0, 0) = 0.141371669412;
    expected_inertia(1, 1) = 0.141371669412;
    expected_inertia(2, 2) = 0.282743338823;
    EXPECT_TRUE(Near(cylinder.mass_2nd_moments()[0], expected_inertia, 1e-10));
}

TEST(ReferenceParity, CylinderDynamicsMatchPyElastica)
{
    Cylinder cylinder = make_cylinder();
    cylinder.mutable_angular_velocities().row(0) << 1.5, -0.7, 2.2;
    cylinder.mutable_velocities().row(0) << 0.3, 0.4, -0.5;
    cylinder.mutable_external_forces().row(0) << 10.0, -5.0, 2.0;
    cylinder.mutable_external_torques().row(0) << 0.4, 0.9, -0.2;

    cylinder.update_accelerations(0.0, 1e-4);

    EXPECT_TRUE(Near(cylinder.accelerations().row(0),
                     Eigen::RowVector3d(0.176838825658, -0.0884194128288,
                                        0.0353677651315),
                     1e-10));
    EXPECT_TRUE(Near(cylinder.angular_accelerations().row(0),
                     Eigen::RowVector3d(4.36942121052, 9.66619772368,
                                        -0.707355302631),
                     1e-9));
    EXPECT_NEAR(cylinder.translational_energy(), 14.1371669412, 1e-9);
    EXPECT_NEAR(cylinder.rotational_energy(), 0.877918067046, 1e-10);
}

// ---------------------------------------------------------------------------
// Interface agreement with CosseratRod
//
// The point of giving a rigid body one node and one element is that the same
// rules can drive either. These checks assert the two types expose the same
// accessor set with the same return types, so a template written against one
// compiles against the other.
// ---------------------------------------------------------------------------

#define BOTH_PROVIDE(expr)                                                     \
    static_assert(requires(CosseratRod sys) { expr; },                         \
                  "CosseratRod is missing " #expr);                            \
    static_assert(requires(RigidBody sys) { expr; },                           \
                  "RigidBody is missing " #expr)

TEST(InterfaceAgreement, BothBodiesExposeTheSameReadAccessors)
{
    BOTH_PROVIDE({ sys.num_nodes() } -> std::convertible_to<std::int64_t>);
    BOTH_PROVIDE({ sys.num_elements() } -> std::convertible_to<std::int64_t>);
    BOTH_PROVIDE({ sys.positions() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.velocities() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.accelerations() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.internal_forces() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.external_forces() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.masses() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.frames() } -> std::same_as<const Matrix3DStack&>);
    BOTH_PROVIDE({ sys.mass_2nd_moments() } -> std::same_as<const Matrix3DStack&>);
    BOTH_PROVIDE({ sys.inv_mass_2nd_moments() } -> std::same_as<const Matrix3DStack&>);
    BOTH_PROVIDE({ sys.angular_velocities() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.angular_accelerations() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.internal_torques() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.external_torques() } -> std::same_as<const Vector3DStack&>);
    BOTH_PROVIDE({ sys.radii() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.densities() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.volumes() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.lengths() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.rest_lengths() } -> std::same_as<const Eigen::VectorXd&>);
    BOTH_PROVIDE({ sys.dilatations() } -> std::same_as<const Eigen::VectorXd&>);
    SUCCEED();
}

TEST(InterfaceAgreement, BothBodiesExposeTheSameMutableAccessors)
{
    BOTH_PROVIDE({ sys.mutable_positions() } -> std::same_as<Vector3DStack&>);
    BOTH_PROVIDE({ sys.mutable_velocities() } -> std::same_as<Vector3DStack&>);
    BOTH_PROVIDE({ sys.mutable_frames() } -> std::same_as<Matrix3DStack&>);
    BOTH_PROVIDE({ sys.mutable_angular_velocities() } -> std::same_as<Vector3DStack&>);
    BOTH_PROVIDE({ sys.mutable_external_forces() } -> std::same_as<Vector3DStack&>);
    BOTH_PROVIDE({ sys.mutable_external_torques() } -> std::same_as<Vector3DStack&>);
    SUCCEED();
}

TEST(InterfaceAgreement, BothBodiesExposeTheSameStepperEntryPoints)
{
    BOTH_PROVIDE(sys.compute_internal_forces_and_torques(0.0));
    BOTH_PROVIDE(sys.update_accelerations(0.0, 1e-4));
    BOTH_PROVIDE(sys.zero_out_external_forces_and_torques(0.0 /* time */));
    SUCCEED();
}

// A template written once drives either body type.
TEST(InterfaceAgreement, ATemplatedStepDrivesEitherBody)
{
    const auto step = [](auto& body, const Eigen::Vector3d& force) {
        body.zero_out_external_forces_and_torques(0.0 /* time */);
        body.mutable_external_forces().row(0) += force.transpose();
        body.compute_internal_forces_and_torques(0.0);
        body.update_accelerations(0.0, 1e-4);
        return body.accelerations().row(0).eval();
    };

    Sphere sphere = make_sphere();
    Cylinder cylinder = make_cylinder();
    CosseratRod rod = straight_cosserat_rod(
        4, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 1.0, 0.05, 1000.0, 1e6, false);

    const Eigen::Vector3d force(0.0, 0.0, 1.0);
    const auto sphere_acceleration = step(sphere, force);
    const auto cylinder_acceleration = step(cylinder, force);
    const auto rod_acceleration = step(rod, force);

    EXPECT_NEAR(sphere_acceleration(2), 1.0 / sphere.total_mass(), 1e-12);
    EXPECT_NEAR(cylinder_acceleration(2), 1.0 / cylinder.total_mass(), 1e-12);
    EXPECT_GT(std::abs(rod_acceleration(2)), 0.0);
}

// Slicing a shape to its base is well defined: the derived classes add no
// members, so the base holds the whole state.
TEST(InterfaceAgreement, ShapesSliceCleanlyToTheBase)
{
    static_assert(std::is_base_of_v<RigidBody, Sphere>);
    static_assert(std::is_base_of_v<RigidBody, Cylinder>);
    static_assert(sizeof(Sphere) == sizeof(RigidBody));
    static_assert(sizeof(Cylinder) == sizeof(RigidBody));
    static_assert(!std::has_virtual_destructor_v<RigidBody>);

    const Sphere sphere = make_sphere();
    const RigidBody sliced = sphere;

    EXPECT_NEAR(sliced.total_mass(), sphere.total_mass(), kTol);
    EXPECT_TRUE(Near(sliced.frames()[0], sphere.frames()[0]));
}

// ---------------------------------------------------------------------------
// General constructor
// ---------------------------------------------------------------------------

TEST(RigidBodyConstruction, AcceptsAnArbitraryShapesMassProperties)
{
    const Eigen::Vector3d inertia(1.0, 2.0, 3.0);
    const RigidBody body(Eigen::Vector3d(1.0, 0.0, 0.0),
                         Eigen::Matrix3d::Identity(), 0.5, 1.0, 100.0, 0.25,
                         inertia);

    EXPECT_NEAR(body.total_mass(), 25.0, kTol);
    EXPECT_TRUE(Near(body.mass_2nd_moments()[0],
                     Eigen::Matrix3d(inertia.asDiagonal())));
    EXPECT_TRUE(Near(body.inv_mass_2nd_moments()[0],
                     Eigen::Matrix3d(inertia.cwiseInverse().asDiagonal())));
}

TEST(RigidBodyConstructionDeathTest, RejectsANonRotationFrame)
{
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    frame(0, 0) = 2.0;

    EXPECT_ASSERT_FAILURE(RigidBody(Eigen::Vector3d::Zero(), frame, 0.5, 1.0,
                                    100.0, 0.25, Eigen::Vector3d::Ones()));
}

TEST(RigidBodyConstructionDeathTest, RejectsASingularInertia)
{
    EXPECT_ASSERT_FAILURE(RigidBody(Eigen::Vector3d::Zero(),
                                    Eigen::Matrix3d::Identity(), 0.5, 1.0, 100.0,
                                    0.25, Eigen::Vector3d(1.0, 0.0, 1.0)));
    EXPECT_ASSERT_FAILURE(RigidBody(Eigen::Vector3d::Zero(),
                                    Eigen::Matrix3d::Identity(), 0.5, 1.0, 100.0,
                                    0.25, Eigen::Vector3d(1.0, -1.0, 1.0)));
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

class RigidBodyWriteTest : public ::testing::Test
{
protected:
    std::filesystem::path m_directory;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_directory = std::filesystem::temp_directory_path()
            / (std::string("rigid_body_") + info->name());
        std::filesystem::remove_all(m_directory);
        std::filesystem::create_directories(m_directory);
    }

    void TearDown() override { std::filesystem::remove_all(m_directory); }

    bool pair_exists(const std::string& name) const
    {
        return std::filesystem::is_regular_file(m_directory / (name + ".bin"))
            and std::filesystem::is_regular_file(m_directory / (name + ".md.json"));
    }
};

TEST_F(RigidBodyWriteTest, WriteEmitsPoseOnly)
{
    const Sphere sphere = make_sphere();

    sphere.write(m_directory);

    EXPECT_TRUE(pair_exists("positions"));
    EXPECT_TRUE(pair_exists("frames"));
    EXPECT_FALSE(pair_exists("velocities"));
}

TEST_F(RigidBodyWriteTest, WriteDebugEmitsEveryStack)
{
    const Cylinder cylinder = make_cylinder();

    cylinder.write_debug(m_directory);

    for (const char* name :
         {"positions", "velocities", "accelerations", "internal_forces",
          "external_forces", "masses", "frames", "mass_2nd_moments",
          "inverse_mass_2nd_moments", "angular_velocities",
          "angular_accelerations", "internal_torques", "external_torques",
          "radii", "densities", "volumes", "lengths", "rest_lengths",
          "dilatations"})
    {
        EXPECT_TRUE(pair_exists(name)) << "missing " << name;
    }
}

TEST_F(RigidBodyWriteTest, WrittenSizesReflectTheSingleEntry)
{
    const Sphere sphere = make_sphere();

    sphere.write_debug(m_directory);

    const auto bytes = [&](const std::string& name) {
        return std::filesystem::file_size(m_directory / (name + ".bin"));
    };

    EXPECT_EQ(bytes("positions"), 3u * sizeof(double));
    EXPECT_EQ(bytes("masses"), 1u * sizeof(double));
    EXPECT_EQ(bytes("frames"), 9u * sizeof(double));
    EXPECT_EQ(bytes("dilatations"), 1u * sizeof(double));
}

TEST_F(RigidBodyWriteTest, WriteCreatesMissingDirectories)
{
    const Sphere sphere = make_sphere();
    const std::filesystem::path nested = m_directory / "run" / "sphere" / "step_0";

    sphere.write(nested);

    EXPECT_TRUE(std::filesystem::is_regular_file(nested / "positions.bin"));
}

}  // namespace
}  // namespace cosserat::physics
