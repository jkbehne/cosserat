#include "physics/contacts.hpp"

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <numbers>
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

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                double tol);

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    return Near(a, b, kTol);
}

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

CosseratRod make_rod(const Eigen::Vector3d& start, const Eigen::Vector3d& direction,
                     double length = 1.0, double radius = 0.05,
                     std::int64_t elements = 6)
{
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    if (std::abs(normal.dot(direction)) > 0.9) normal = Eigen::Vector3d(0.0, 1.0, 0.0);
    normal = (normal - normal.dot(direction) * direction).normalized();

    return straight_cosserat_rod(elements, start, direction, normal, length, radius,
                                 1000.0, 1.0e6, false);
}

Sphere make_sphere(const Eigen::Vector3d& center, double radius = 0.1)
{
    return Sphere(center, radius, 1000.0);
}

/** Builds a cylinder whose CENTRE is at the given point. */
Cylinder make_cylinder_centred(const Eigen::Vector3d& center, double length = 0.4,
                               double radius = 0.06)
{
    // The constructor takes the starting end face, so step back half a length.
    const Eigen::Vector3d start = center - 0.5 * length * Eigen::Vector3d::UnitZ();
    return Cylinder(start, Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitX(),
                    length, radius, 1000.0);
}

double max_force(const CosseratRod& rod)
{
    return rod.external_forces().cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

TEST(ContactConcepts, BodiesSatisfyTheirContactRoles)
{
    static_assert(ContactableRod<CosseratRod>);
    static_assert(FrictionalRod<CosseratRod>);
    static_assert(ContactableCylinder<Cylinder>);
    static_assert(ContactableSphere<Sphere>);
    static_assert(ContactableSurface<Plane>);
    SUCCEED();
}

// A rigid body has no per-element radii or tangents, so it cannot stand in for
// a rod. That is what keeps the variant dispatch from silently accepting the
// wrong pairing.
TEST(ContactConcepts, ARigidBodyIsNotARod)
{
    static_assert(!ContactableRod<RigidBody>);
    static_assert(!ContactableRod<Sphere>);
    static_assert(!ContactableRod<Cylinder>);
    static_assert(!ContactableSurface<CosseratRod>);
    SUCCEED();
}

// Sphere and Cylinder add no members to RigidBody, so they are structurally
// identical and every rigid type satisfies both roles. The concepts therefore
// cannot tell a sphere from a cylinder: passing one where the other is
// expected compiles and runs, using the sphere's diameter as a length. What
// separates them is which variant alternative is held and which body the call
// site passes, not the type system.
TEST(ContactConcepts, TheRigidShapesAreIndistinguishableToTheConcepts)
{
    static_assert(ContactableSphere<Sphere>);
    static_assert(ContactableSphere<Cylinder>);
    static_assert(ContactableCylinder<Cylinder>);
    static_assert(ContactableCylinder<Sphere>);
    static_assert(ContactableCylinder<RigidBody>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(ContactConstruction, StoresItsCoefficients)
{
    const RodRodContact rod_rod(1.0e4, 10.0);
    EXPECT_DOUBLE_EQ(rod_rod.k(), 1.0e4);
    EXPECT_DOUBLE_EQ(rod_rod.nu(), 10.0);

    const RodPlaneContactWithAnisotropicFriction friction(
        1.0e4, 10.0, 1e-4, Eigen::Vector3d(0.1, 0.2, 0.3),
        Eigen::Vector3d(0.4, 0.5, 0.6));
    EXPECT_TRUE(Near(friction.static_mu(), Eigen::Vector3d(0.1, 0.2, 0.3)));
    EXPECT_TRUE(Near(friction.kinetic_mu(), Eigen::Vector3d(0.4, 0.5, 0.6)));
}

TEST(ContactConstructionDeathTest, RejectsBadCoefficients)
{
    EXPECT_ASSERT_FAILURE(RodRodContact(0.0, 10.0));
    EXPECT_ASSERT_FAILURE(RodRodContact(-1.0, 10.0));
    EXPECT_ASSERT_FAILURE(RodRodContact(1.0e4, -1.0));
    EXPECT_ASSERT_FAILURE(RodSelfContact(0.0, 10.0));
    EXPECT_ASSERT_FAILURE(RodCylinderContact(1e4, 10.0, -1.0, 0.0));
    EXPECT_ASSERT_FAILURE(RodCylinderContact(1e4, 10.0, 0.0, -1.0));
    EXPECT_ASSERT_FAILURE(RodSphereContact(0.0, 10.0, 0.0, 0.0));
    EXPECT_ASSERT_FAILURE(RodPlaneContact(-1.0, 10.0));
    EXPECT_ASSERT_FAILURE(CylinderPlaneContact(1e4, -1.0));
    EXPECT_ASSERT_FAILURE(RodPlaneContactWithAnisotropicFriction(
        1e4, 10.0, 0.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    EXPECT_ASSERT_FAILURE(RodPlaneContactWithAnisotropicFriction(
        1e4, 10.0, 1e-4, Eigen::Vector3d(-1, 0, 0), Eigen::Vector3d::Zero()));
}

TEST(ContactConstructionDeathTest, RejectsANonUnitPlaneNormal)
{
    EXPECT_ASSERT_FAILURE(
        Plane(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 2.0)));
    EXPECT_ASSERT_FAILURE(
        Plane(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
}

// ---------------------------------------------------------------------------
// Rod to rod
// ---------------------------------------------------------------------------

TEST(RodRodContactTest, SeparatedRodsFeelNothing)
{
    CosseratRod first = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    CosseratRod second =
        make_rod(Eigen::Vector3d(0.0, 50.0, 0.0), Eigen::Vector3d::UnitX());
    const RodRodContact contact(1.0e4, 10.0);

    contact.apply_contact(first, second, 0.0);

    EXPECT_LT(max_force(first), kTol);
    EXPECT_LT(max_force(second), kTol);
}

// Two rods laid across one another, closer than the sum of their radii.
TEST(RodRodContactTest, CrossingRodsPushApart)
{
    CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                 Eigen::Vector3d::UnitX());
    CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                  Eigen::Vector3d::UnitY());
    const RodRodContact contact(1.0e4, 0.0);

    contact.apply_contact(first, second, 0.0);

    EXPECT_GT(max_force(first), 0.0);
    EXPECT_GT(max_force(second), 0.0);
    // The lower rod is pushed down and the upper one up.
    EXPECT_LT(first.external_forces().col(2).sum(), 0.0);
    EXPECT_GT(second.external_forces().col(2).sum(), 0.0);
}

// Contact is an internal interaction, so the total force on the pair vanishes.
TEST(RodRodContactTest, TotalForceOnThePairIsZero)
{
    CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                 Eigen::Vector3d::UnitX());
    CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                  Eigen::Vector3d::UnitY());
    const RodRodContact contact(1.0e4, 0.0);

    contact.apply_contact(first, second, 0.0);

    const Eigen::RowVector3d total =
        first.external_forces().colwise().sum() + second.external_forces().colwise().sum();
    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-9);
}

TEST(RodRodContactTest, DeeperOverlapPushesHarder)
{
    double previous = 0.0;
    // Stopping short of zero: at exactly zero the two axes intersect and the
    // contact normal is undefined, which is a separately tested case.
    for (double height : {0.09, 0.06, 0.03, 0.01})
    {
        CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                     Eigen::Vector3d::UnitX());
        CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, height),
                                      Eigen::Vector3d::UnitY());
        RodRodContact(1.0e4, 0.0).apply_contact(first, second, 0.0);

        const double magnitude = max_force(first);
        EXPECT_GT(magnitude, previous) << "height " << height;
        previous = magnitude;
    }
}

// Rod-to-rod contact writes forces only; no rod ever gains a torque from it.
TEST(RodRodContactTest, AppliesNoTorque)
{
    CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                 Eigen::Vector3d::UnitX());
    CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                  Eigen::Vector3d::UnitY());

    RodRodContact(1.0e4, 10.0).apply_contact(first, second, 0.0);

    EXPECT_LT(first.external_torques().cwiseAbs().maxCoeff(), kTol);
    EXPECT_LT(second.external_torques().cwiseAbs().maxCoeff(), kTol);
}

// Damping opposes approach, so two rods driving together feel more than two at
// rest in the same position.
TEST(RodRodContactTest, DampingRespondsToApproachVelocity)
{
    const auto run = [](double approach_speed) {
        CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                     Eigen::Vector3d::UnitX());
        CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                      Eigen::Vector3d::UnitY());
        second.mutable_velocities().col(2).setConstant(-approach_speed);
        RodRodContact(1.0e4, 100.0).apply_contact(first, second, 0.0);
        return max_force(first);
    };

    EXPECT_GT(run(1.0), run(0.0));
}

// ---------------------------------------------------------------------------
// Rod self contact
// ---------------------------------------------------------------------------

/** A rod bent into a hairpin, so its two arms lie within a radius. */
CosseratRod make_hairpin(double arm_separation)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                               1.0, 0.05, 20);
    Vector3DStack& positions = rod.mutable_positions();
    const Eigen::Index count = positions.rows();
    for (Eigen::Index i = 0; i < count; ++i)
    {
        const double s = static_cast<double>(i) / static_cast<double>(count - 1);
        const double angle = std::numbers::pi * s;
        positions.row(i) << 0.3 * std::cos(angle),
                            arm_separation * std::sin(angle), 0.0;
    }
    rod.compute_internal_forces_and_torques(0.0);
    rod.zero_out_external_forces_and_torques(0.0 /* time */);
    return rod;
}

TEST(RodSelfContactTest, AStraightRodDoesNotTouchItself)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                               1.0, 0.05, 20);
    const RodSelfContact contact(1.0e4, 10.0);

    contact.apply_contact(rod, 0.0);

    EXPECT_LT(max_force(rod), kTol);
}

TEST(RodSelfContactTest, AFoldedRodPushesItselfApart)
{
    CosseratRod rod = make_hairpin(0.02);
    const RodSelfContact contact(1.0e4, 0.0);

    contact.apply_contact(rod, 0.0);

    EXPECT_GT(max_force(rod), 0.0);
}

TEST(RodSelfContactTest, TotalSelfForceIsZero)
{
    CosseratRod rod = make_hairpin(0.02);

    RodSelfContact(1.0e4, 0.0).apply_contact(rod, 0.0);

    EXPECT_LT(rod.external_forces().colwise().sum().cwiseAbs().maxCoeff(), 1e-9);
}

// The skip window excludes neighbouring elements, which are always touching by
// construction. Without it a straight rod would generate contact everywhere.
TEST(RodSelfContactTest, NeighbouringElementsAreExcluded)
{
    for (double radius : {0.02, 0.05, 0.1})
    {
        CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                                   1.0, radius, 20);
        RodSelfContact(1.0e4, 10.0).apply_contact(rod, 0.0);

        EXPECT_LT(max_force(rod), kTol) << "radius " << radius;
    }
}

TEST(RodSelfContactTest, AppliesNoTorque)
{
    CosseratRod rod = make_hairpin(0.02);

    RodSelfContact(1.0e4, 10.0).apply_contact(rod, 0.0);

    EXPECT_LT(rod.external_torques().cwiseAbs().maxCoeff(), kTol);
}

TEST(RodSelfContactTest, ATighterFoldPushesHarder)
{
    CosseratRod loose = make_hairpin(0.05);
    CosseratRod tight = make_hairpin(0.01);

    RodSelfContact(1.0e4, 0.0).apply_contact(loose, 0.0);
    RodSelfContact(1.0e4, 0.0).apply_contact(tight, 0.0);

    EXPECT_GT(max_force(tight), max_force(loose));
}

// ---------------------------------------------------------------------------
// Rod to cylinder
// ---------------------------------------------------------------------------

TEST(RodCylinderContactTest, SeparatedBodiesFeelNothing)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d(0.0, 50.0, 0.0));

    RodCylinderContact(1.0e4, 10.0, 0.0, 0.0).apply_contact(rod, cylinder, 0.0);

    EXPECT_LT(max_force(rod), kTol);
    EXPECT_LT(cylinder.external_forces().cwiseAbs().maxCoeff(), kTol);
}

TEST(RodCylinderContactTest, OverlappingBodiesPushApart)
{
    // A rod along x passing beside a cylinder standing on z at the origin,
    // offset so the two axes do not intersect exactly.
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.04, 0.0),
                               Eigen::Vector3d::UnitX());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());

    RodCylinderContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, cylinder, 0.0);

    EXPECT_GT(max_force(rod), 0.0);
    EXPECT_GT(cylinder.external_forces().cwiseAbs().maxCoeff(), 0.0);
}

// Unlike a sphere, a cylinder does take a torque from contact.
TEST(RodCylinderContactTest, TorquesTheCylinderWhenContactIsOffCentre)
{
    // Offset in y so the axes do not intersect, and in z so the contact point
    // sits away from the cylinder's midpoint and has a moment arm.
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.04, 0.15),
                               Eigen::Vector3d::UnitX());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());

    RodCylinderContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, cylinder, 0.0);

    EXPECT_GT(cylinder.external_torques().cwiseAbs().maxCoeff(), 0.0);
}

// Contact level with the cylinder's midpoint has no moment arm along the axis,
// so it produces force but no torque.
TEST(RodCylinderContactTest, ContactAtTheMidpointProducesNoTorque)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.04, 0.0),
                               Eigen::Vector3d::UnitX());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());

    RodCylinderContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, cylinder, 0.0);

    EXPECT_GT(cylinder.external_forces().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_LT(cylinder.external_torques().cwiseAbs().maxCoeff(), kTol);
}

TEST(RodCylinderContactTest, FrictionOpposesSlidingAcrossTheContact)
{
    const auto run = [](double friction_coefficient) {
        CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.04, 0.0),
                                   Eigen::Vector3d::UnitX());
        Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());
        // Slide the rod along its own axis across the cylinder.
        rod.mutable_velocities().col(0).setConstant(1.0);
        RodCylinderContact(1.0e4, 0.0, 100.0, friction_coefficient)
            .apply_contact(rod, cylinder, 0.0);
        return rod.external_forces().col(0).cwiseAbs().maxCoeff();
    };

    EXPECT_LT(run(0.0), run(0.5));
}

// ---------------------------------------------------------------------------
// Rod to sphere
// ---------------------------------------------------------------------------

TEST(RodSphereContactTest, SeparatedBodiesFeelNothing)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d(0.0, 50.0, 0.0));

    RodSphereContact(1.0e4, 10.0, 0.0, 0.0).apply_contact(rod, sphere, 0.0);

    EXPECT_LT(max_force(rod), kTol);
    EXPECT_LT(sphere.external_forces().cwiseAbs().maxCoeff(), kTol);
}

TEST(RodSphereContactTest, AnEmbeddedSpherePushesTheRodAside)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                               Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d(0.0, 0.08, 0.0));

    RodSphereContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, sphere, 0.0);

    EXPECT_GT(max_force(rod), 0.0);
    // The rod is pushed away from the sphere and the sphere the other way.
    EXPECT_LT(rod.external_forces().col(1).sum(), 0.0);
    EXPECT_GT(sphere.external_forces()(0, 1), 0.0);
}

TEST(RodSphereContactTest, TotalForceOnThePairIsZero)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                               Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d(0.0, 0.08, 0.0));

    RodSphereContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, sphere, 0.0);

    const Eigen::RowVector3d total = rod.external_forces().colwise().sum()
        + sphere.external_forces().colwise().sum();
    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-9);
}

// The sphere gains force but never torque, even for an off-centre contact.
// That is faithful to the reference rather than physically complete.
TEST(RodSphereContactTest, NeverTorquesTheSphere)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                               Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d(0.2, 0.08, 0.0));

    RodSphereContact(1.0e4, 10.0, 10.0, 0.5).apply_contact(rod, sphere, 0.0);

    ASSERT_GT(sphere.external_forces().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_LT(sphere.external_torques().cwiseAbs().maxCoeff(), kTol);
}

// ---------------------------------------------------------------------------
// Rod to plane
// ---------------------------------------------------------------------------

TEST(RodPlaneContactTest, ARodWellAboveThePlaneFeelsNothing)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, 5.0),
                               Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    RodPlaneContact(1.0e4, 10.0).apply_contact(rod, ground, 0.0);

    EXPECT_LT(max_force(rod), kTol);
}

TEST(RodPlaneContactTest, ASunkenRodIsPushedUp)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                               Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    RodPlaneContact(1.0e4, 0.0).apply_contact(rod, ground, 0.0);

    EXPECT_GT(rod.external_forces().col(2).sum(), 0.0);
}

// The plane cancels whatever is pressing the rod into it, so a rod loaded
// downward and resting on the surface nets out near zero.
TEST(RodPlaneContactTest, CancelsAnImposedDownwardLoad)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, 0.05),
                               Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    rod.mutable_external_forces().col(2).setConstant(-1.0);
    const double imposed = rod.external_forces().col(2).sum();

    RodPlaneContact(1.0e4, 0.0).apply_contact(rod, ground, 0.0);

    EXPECT_GT(rod.external_forces().col(2).sum(), imposed);
}

TEST(RodPlaneContactTest, DeeperPenetrationPushesHarder)
{
    double previous = 0.0;
    for (double height : {0.04, 0.0, -0.04, -0.08})
    {
        CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, height),
                                   Eigen::Vector3d::UnitX());
        Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
        RodPlaneContact(1.0e4, 0.0).apply_contact(rod, ground, 0.0);

        const double push = rod.external_forces().col(2).sum();
        EXPECT_GT(push, previous) << "height " << height;
        previous = push;
    }
}

TEST(RodPlaneContactTest, WorksForATiltedPlane)
{
    const Eigen::Vector3d normal = Eigen::Vector3d(0.3, 0.0, 1.0).normalized();
    Plane slope(Eigen::Vector3d::Zero(), normal);
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                               Eigen::Vector3d::UnitY());

    RodPlaneContact(1.0e4, 0.0).apply_contact(rod, slope, 0.0);

    // The push is along the plane's normal.
    const Eigen::Vector3d total = rod.external_forces().colwise().sum().transpose();
    ASSERT_GT(total.norm(), 0.0);
    EXPECT_NEAR(total.normalized().dot(normal), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Rod to plane with friction
// ---------------------------------------------------------------------------

// Friction scales against the plane's response, which is only the part of an
// existing load that the surface has to cancel. A weightless rod resting on
// the ground therefore feels no friction however fast it slides, so every test
// here presses the rod down first.
TEST(RodPlaneFrictionTest, FrictionOpposesSlidingAlongTheRod)
{
    const auto run = [](double kinetic) {
        CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                                   Eigen::Vector3d::UnitX());
        Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
        rod.mutable_external_forces().col(2).setConstant(-1.0);
        rod.mutable_velocities().col(0).setConstant(1.0);
        RodPlaneContactWithAnisotropicFriction(
            1.0e4, 0.0, 1e-4, Eigen::Vector3d::Zero(),
            Eigen::Vector3d(kinetic, kinetic, kinetic))
            .apply_contact(rod, ground, 0.0);
        return rod.external_forces().col(0).sum();
    };

    // Sliding forward, friction acts backward and grows with the coefficient.
    EXPECT_LT(run(0.5), run(0.0));
}

// Forward and backward coefficients differ, so reversing direction changes the
// magnitude of the resistance.
TEST(RodPlaneFrictionTest, FrictionIsAnisotropicAlongTheAxis)
{
    const auto run = [](double direction_sign) {
        CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                                   Eigen::Vector3d::UnitX());
        Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
        rod.mutable_external_forces().col(2).setConstant(-1.0);
        rod.mutable_velocities().col(0).setConstant(direction_sign);
        RodPlaneContactWithAnisotropicFriction(
            1.0e4, 0.0, 1e-4, Eigen::Vector3d::Zero(),
            Eigen::Vector3d(0.2, 0.8, 0.4))
            .apply_contact(rod, ground, 0.0);
        return std::abs(rod.external_forces().col(0).sum());
    };

    // Backward has the larger coefficient, so it resists more.
    EXPECT_GT(run(-1.0), run(1.0));
}

// The only contact in the file that torques a rod.
TEST(RodPlaneFrictionTest, RollingFrictionTorquesTheRod)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                               Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    rod.mutable_external_forces().col(2).setConstant(-1.0);
    rod.mutable_velocities().col(1).setConstant(1.0);  // slide sideways

    RodPlaneContactWithAnisotropicFriction(
        1.0e4, 0.0, 1e-4, Eigen::Vector3d(0.1, 0.1, 0.1),
        Eigen::Vector3d(0.4, 0.4, 0.4))
        .apply_contact(rod, ground, 0.0);

    EXPECT_GT(rod.external_torques().cwiseAbs().maxCoeff(), 0.0);
}

TEST(RodPlaneFrictionTest, ARodAboveThePlaneFeelsNoFriction)
{
    CosseratRod rod = make_rod(Eigen::Vector3d(0.0, 0.0, 5.0),
                               Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    rod.mutable_velocities().col(0).setConstant(1.0);

    RodPlaneContactWithAnisotropicFriction(
        1.0e4, 10.0, 1e-4, Eigen::Vector3d(0.5, 0.5, 0.5),
        Eigen::Vector3d(0.5, 0.5, 0.5))
        .apply_contact(rod, ground, 0.0);

    EXPECT_LT(max_force(rod), kTol);
    EXPECT_LT(rod.external_torques().cwiseAbs().maxCoeff(), kTol);
}

// With no friction coefficients at all, the result must match plain contact.
TEST(RodPlaneFrictionTest, ZeroCoefficientsReduceToPlainContact)
{
    CosseratRod with_friction = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                                         Eigen::Vector3d::UnitX());
    CosseratRod plain = make_rod(Eigen::Vector3d(0.0, 0.0, -0.02),
                                 Eigen::Vector3d::UnitX());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    with_friction.mutable_external_forces().col(2).setConstant(-1.0);
    plain.mutable_external_forces().col(2).setConstant(-1.0);

    RodPlaneContactWithAnisotropicFriction(
        1.0e4, 5.0, 1e-4, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())
        .apply_contact(with_friction, ground, 0.0);
    RodPlaneContact(1.0e4, 5.0).apply_contact(plain, ground, 0.0);

    EXPECT_TRUE(Near(with_friction.external_forces(), plain.external_forces(), 1e-9));
    EXPECT_LT(with_friction.external_torques().cwiseAbs().maxCoeff(), 1e-12);
}

// ---------------------------------------------------------------------------
// Cylinder to plane
// ---------------------------------------------------------------------------

TEST(CylinderPlaneContactTest, ACylinderAboveThePlaneFeelsNothing)
{
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d(0.0, 0.0, 5.0));
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    CylinderPlaneContact(1.0e4, 10.0).apply_contact(cylinder, ground, 0.0);

    EXPECT_LT(cylinder.external_forces().cwiseAbs().maxCoeff(), kTol);
}

TEST(CylinderPlaneContactTest, ASunkenCylinderIsPushedUp)
{
    // Centre below half a length, so it penetrates.
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d(0.0, 0.0, -0.25), 0.4);
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    CylinderPlaneContact(1.0e4, 0.0).apply_contact(cylinder, ground, 0.0);

    EXPECT_GT(cylinder.external_forces()(0, 2), 0.0);
}

TEST(CylinderPlaneContactTest, CancelsAnImposedDownwardLoad)
{
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d(0.0, 0.0, 0.2), 0.4);
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    cylinder.mutable_external_forces().row(0) << 0.0, 0.0, -3.0;

    CylinderPlaneContact(1.0e4, 0.0).apply_contact(cylinder, ground, 0.0);

    EXPECT_GT(cylinder.external_forces()(0, 2), -3.0);
}

// ---------------------------------------------------------------------------
// NoContact
// ---------------------------------------------------------------------------

TEST(NoContactTest, DoesNothingToAnything)
{
    CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                 Eigen::Vector3d::UnitX());
    CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.0),
                                  Eigen::Vector3d::UnitY());
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    const NoContact contact;

    contact.apply_contact(first, second, 0.0);
    contact.apply_contact(first, sphere, 0.0);

    EXPECT_LT(max_force(first), kTol);
    EXPECT_LT(max_force(second), kTol);
    EXPECT_LT(sphere.external_forces().cwiseAbs().maxCoeff(), kTol);
}

// ---------------------------------------------------------------------------
// Variant dispatch, two systems
// ---------------------------------------------------------------------------

TEST(ContactVariantTest, ValidateAcceptsTheRightPairings)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    CosseratRod other = make_rod(Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    ContactVariant rod_rod = RodRodContact(1e4, 10.0);
    ContactVariant rod_sphere = RodSphereContact(1e4, 10.0, 0.0, 0.0);
    ContactVariant rod_cylinder = RodCylinderContact(1e4, 10.0, 0.0, 0.0);
    ContactVariant rod_plane = RodPlaneContact(1e4, 10.0);
    ContactVariant cylinder_plane = CylinderPlaneContact(1e4, 10.0);

    EXPECT_NO_THROW({ validate(rod_rod, rod, other); });
    EXPECT_NO_THROW({ validate(rod_sphere, rod, sphere); });
    EXPECT_NO_THROW({ validate(rod_cylinder, rod, cylinder); });
    EXPECT_NO_THROW({ validate(rod_plane, rod, ground); });
    EXPECT_NO_THROW({ validate(cylinder_plane, cylinder, ground); });
}

TEST(ContactVariantDeathTest, ValidateRejectsTheWrongPairings)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    Cylinder cylinder = make_cylinder_centred(Eigen::Vector3d::Zero());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());

    ContactVariant rod_rod = RodRodContact(1e4, 10.0);
    ContactVariant rod_sphere = RodSphereContact(1e4, 10.0, 0.0, 0.0);
    ContactVariant cylinder_plane = CylinderPlaneContact(1e4, 10.0);

    // A sphere is not a rod.
    EXPECT_ASSERT_FAILURE(validate(rod_rod, rod, sphere));
    EXPECT_ASSERT_FAILURE(validate(rod_rod, sphere, sphere));
    // A plane is not a sphere.
    EXPECT_ASSERT_FAILURE(validate(rod_sphere, rod, ground));
    // Order matters: cylinder first, plane second.
    EXPECT_ASSERT_FAILURE(validate(cylinder_plane, ground, cylinder));
}

TEST(ContactVariantTest, ApplyThroughTheVariantMatchesADirectCall)
{
    CosseratRod direct_one = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                      Eigen::Vector3d::UnitX());
    CosseratRod direct_two = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                      Eigen::Vector3d::UnitY());
    CosseratRod variant_one = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                       Eigen::Vector3d::UnitX());
    CosseratRod variant_two = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                       Eigen::Vector3d::UnitY());

    RodRodContact(1e4, 10.0).apply_contact(direct_one, direct_two, 0.0);
    ContactVariant held = RodRodContact(1e4, 10.0);
    apply_contact(held, variant_one, variant_two, 0.0);

    EXPECT_TRUE(Near(variant_one.external_forces(), direct_one.external_forces()));
    EXPECT_TRUE(Near(variant_two.external_forces(), direct_two.external_forces()));
}

TEST(ContactVariantDeathTest, ApplyRejectsTheWrongPairing)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    ContactVariant rod_rod = RodRodContact(1e4, 10.0);

    EXPECT_ASSERT_FAILURE(apply_contact(rod_rod, rod, sphere, 0.0));
}

// NoContact accepts anything, so it is always a safe alternative to hold.
TEST(ContactVariantTest, NoContactAcceptsEveryPairing)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    Plane ground(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    ContactVariant nothing = NoContact{};

    EXPECT_NO_THROW({ validate(nothing, rod, sphere); });
    EXPECT_NO_THROW({ validate(nothing, sphere, ground); });
    EXPECT_NO_THROW({ validate(nothing, rod); });
    EXPECT_NO_THROW({ apply_contact(nothing, rod, sphere, 0.0); });
}

// ---------------------------------------------------------------------------
// Variant dispatch, one system
// ---------------------------------------------------------------------------

TEST(ContactVariantTest, SelfContactDispatchesThroughTheSingleSystemForm)
{
    CosseratRod direct = make_hairpin(0.02);
    CosseratRod through = make_hairpin(0.02);

    RodSelfContact(1e4, 10.0).apply_contact(direct, 0.0);
    ContactVariant held = RodSelfContact(1e4, 10.0);
    apply_contact(held, through, 0.0);

    EXPECT_TRUE(Near(through.external_forces(), direct.external_forces()));
    EXPECT_GT(direct.external_forces().cwiseAbs().maxCoeff(), 0.0);
}

TEST(ContactVariantTest, ValidateAcceptsSelfContactOnARod)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    ContactVariant self = RodSelfContact(1e4, 10.0);

    EXPECT_NO_THROW({ validate(self, rod); });
}

// A pair contact has no single-system form, and self contact has no pair form.
TEST(ContactVariantDeathTest, TheTwoArityFormsAreDistinct)
{
    CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX());
    CosseratRod other = make_rod(Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitX());

    ContactVariant pair_only = RodRodContact(1e4, 10.0);
    ContactVariant self_only = RodSelfContact(1e4, 10.0);

    EXPECT_ASSERT_FAILURE(validate(pair_only, rod));
    EXPECT_ASSERT_FAILURE(apply_contact(pair_only, rod, 0.0));
    EXPECT_ASSERT_FAILURE(validate(self_only, rod, other));
    EXPECT_ASSERT_FAILURE(apply_contact(self_only, rod, other, 0.0));
}

TEST(ContactVariantDeathTest, SelfContactRejectsARigidBody)
{
    Sphere sphere = make_sphere(Eigen::Vector3d::Zero());
    ContactVariant self = RodSelfContact(1e4, 10.0);

    EXPECT_ASSERT_FAILURE(validate(self, sphere));
}

// ---------------------------------------------------------------------------
// A list of contacts driven from one loop
// ---------------------------------------------------------------------------

TEST(ContactVariantTest, AListOfContactsAppliesInOnePass)
{
    CosseratRod first = make_rod(Eigen::Vector3d(-0.5, 0.0, 0.0),
                                 Eigen::Vector3d::UnitX());
    CosseratRod second = make_rod(Eigen::Vector3d(0.0, -0.5, 0.05),
                                  Eigen::Vector3d::UnitY());
    Sphere sphere = make_sphere(Eigen::Vector3d(0.3, 0.0, 0.08));

    ContactVariant rod_rod = RodRodContact(1e4, 10.0);
    ContactVariant rod_sphere = RodSphereContact(1e4, 10.0, 0.0, 0.0);
    ContactVariant self = RodSelfContact(1e4, 10.0);

    validate(rod_rod, first, second);
    validate(rod_sphere, first, sphere);
    validate(self, first);

    apply_contact(rod_rod, first, second, 0.0);
    apply_contact(rod_sphere, first, sphere, 0.0);
    apply_contact(self, first, 0.0);

    EXPECT_GT(max_force(first), 0.0);
    EXPECT_GT(max_force(second), 0.0);
}

// ---------------------------------------------------------------------------
// Reference parity
//
// Expected values come from a NumPy transcription of PyElastica's
// _contact_functions kernels on the same configuration.
// ---------------------------------------------------------------------------

TEST(ReferenceParity, RodRodForcesMatchPyElastica)
{
    // Two straight overlapping rods, at rest and unloaded, so the response is
    // purely the contact spring. Expected values come from a NumPy
    // transcription of the reference kernel on this exact configuration.
    const Eigen::Index elements_one = 6;
    const Eigen::Index elements_two = 5;

    Vector3DStack positions_one(elements_one + 1, 3);
    for (Eigen::Index i = 0; i <= elements_one; ++i)
    {
        positions_one.row(i) << 0.10 * static_cast<double>(i), 0.0, 0.0;
    }
    Vector3DStack positions_two(elements_two + 1, 3);
    for (Eigen::Index i = 0; i <= elements_two; ++i)
    {
        positions_two.row(i) << 0.12 * static_cast<double>(i) - 0.1, 0.05, 0.02;
    }

    const auto tangents = [](const Vector3DStack& p) {
        Vector3DStack t(p.rows() - 1, 3);
        for (Eigen::Index i = 0; i + 1 < p.rows(); ++i)
        {
            t.row(i) = (p.row(i + 1) - p.row(i)).normalized();
        }
        return t;
    };
    const auto lengths = [](const Vector3DStack& p) {
        Eigen::VectorXd l(p.rows() - 1);
        for (Eigen::Index i = 0; i + 1 < p.rows(); ++i)
        {
            l(i) = (p.row(i + 1) - p.row(i)).norm();
        }
        return l;
    };

    Vector3DStack forces_one = Vector3DStack::Zero(elements_one + 1, 3);
    Vector3DStack forces_two = Vector3DStack::Zero(elements_two + 1, 3);

    detail::contact_forces_rod_rod(
        detail::element_start_positions(positions_one),
        Eigen::VectorXd::Constant(elements_one, 0.04), lengths(positions_one),
        tangents(positions_one), Vector3DStack::Zero(elements_one + 1, 3),
        Vector3DStack::Zero(elements_one + 1, 3), forces_one,
        detail::element_start_positions(positions_two),
        Eigen::VectorXd::Constant(elements_two, 0.05), lengths(positions_two),
        tangents(positions_two), Vector3DStack::Zero(elements_two + 1, 3),
        Vector3DStack::Zero(elements_two + 1, 3), forces_two, 1.0e3, 0.0);

    Vector3DStack expected_one(elements_one + 1, 3);
    expected_one << -4.55521048666, -28.0692812281, -11.2277124912,
              -12.5997993124, -92.6092799185, -37.0437119674,
              -5.77315972805e-15, -72.9414349245, -29.1765739698,
              10.3221940691, -78.5746393044, -31.4298557218,
              12.499804766, -73.052845522, -29.2211382088,
              5.66698903601, -53.3241917997, -21.3296767199,
              0, -11.1876340599, -4.47505362396;

    Vector3DStack expected_two(elements_two + 1, 3);
    expected_two << 0, 11.1876340599, 4.47505362396,
              -3.48937833909, 58.845985582, 23.5383942328,
              -3.48937833909, 87.1156589669, 34.8462635868,
              -2.17761069693, 101.283131557, 40.5132526228,
              -2.17761069693, 117.763994412, 47.1055977646,
              0, 33.5629021797, 13.4251608719;

    EXPECT_TRUE(Near(forces_one, expected_one, 1e-9));
    EXPECT_TRUE(Near(forces_two, expected_two, 1e-9));

    // Newton's third law across the pair. Note that the net on either rod
    // alone is not zero; only the pair balances.
    const Eigen::RowVector3d total =
        forces_one.colwise().sum() + forces_two.colwise().sum();
    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-10);
    EXPECT_GT(forces_one.colwise().sum().cwiseAbs().maxCoeff(), 1.0);
}

TEST(ReferenceParity, SelfContactSkipWindowMatchesTheFormula)
{
    // skip = 1 + ceil(0.8 * pi * r / l), so a fatter rod excludes more of its
    // own neighbourhood.
    for (double radius : {0.01, 0.05, 0.2})
    {
        const double length = 0.05;
        const auto expected = static_cast<std::int64_t>(
            1 + std::ceil(0.8 * std::numbers::pi * radius / length));

        CosseratRod rod = make_rod(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                                   1.0, radius, 20);
        RodSelfContact(1.0e4, 0.0).apply_contact(rod, 0.0);

        // A straight rod never self-contacts whatever the skip, which is the
        // observable consequence of the window being at least that wide.
        EXPECT_LT(max_force(rod), kTol) << "radius " << radius;
        EXPECT_GE(expected, 1);
    }
}

}  // namespace
}  // namespace cosserat::physics
