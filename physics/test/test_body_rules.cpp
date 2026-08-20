#include "physics/bodies.hpp"

#include "physics/constraints.hpp"
#include "physics/damping.hpp"
#include "physics/forces.hpp"
#include "physics/joints.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <variant>

namespace cosserat::physics {
namespace {

// ---------------------------------------------------------------------------
// Body and rule compatibility
//
// The rule families are templates constrained by concepts, and the bodies are
// separate classes that have to satisfy them. Nothing forces the two to agree,
// so a rename on either side silently stops every rule from applying: the
// variant dispatch probes whether the call is well formed and falls through to
// an assertion when it is not, which looks like a runtime rejection rather
// than a compile error.
//
// These static assertions turn that failure mode back into a build failure.
// ---------------------------------------------------------------------------

#define BODY_SATISFIES(concept_name)                                           \
    static_assert(concept_name<CosseratRod>,                                   \
                  "CosseratRod no longer satisfies " #concept_name);           \
    static_assert(concept_name<RigidBody>,                                     \
                  "RigidBody no longer satisfies " #concept_name);             \
    static_assert(concept_name<Sphere>,                                        \
                  "Sphere no longer satisfies " #concept_name);                \
    static_assert(concept_name<Cylinder>,                                      \
                  "Cylinder no longer satisfies " #concept_name)

TEST(BodyRuleCompatibility, EveryBodyTypeSatisfiesEveryRuleConcept)
{
    BODY_SATISFIES(ForceableSystem);
    BODY_SATISFIES(TorqueableSystem);
    BODY_SATISFIES(ForceableTorqueableSystem);
    BODY_SATISFIES(DampableSystem);
    BODY_SATISFIES(InertiallyDampableSystem);
    BODY_SATISFIES(RayleighDampableSystem);
    BODY_SATISFIES(PositionConstrainableSystem);
    BODY_SATISFIES(DirectorConstrainableSystem);
    BODY_SATISFIES(ConstrainableSystem);
    BODY_SATISFIES(ForceJointableSystem);
    BODY_SATISFIES(TorqueJointableSystem);
    BODY_SATISFIES(JointableSystem);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// The rules actually run
//
// Satisfying a concept is necessary but not sufficient: validate() probes the
// call rather than the concept, so these exercise the dispatch that the
// simulation graph will use.
// ---------------------------------------------------------------------------

CosseratRod make_rod(double start_z)
{
    return straight_cosserat_rod(
        6, Eigen::Vector3d(0.0, 0.0, start_z), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitY(), 0.2, 0.007, 1750.0, 3.0e7, false);
}

TEST(BodyRuleCompatibility, ForcesApplyToBothBodyKinds)
{
    CosseratRod rod = make_rod(0.0);
    Sphere sphere(Eigen::Vector3d::Zero(), 0.02, 1000.0);
    ForceTorqueVariant gravity = GravityForceZ{};

    EXPECT_NO_THROW({ validate(gravity, rod); });
    EXPECT_NO_THROW({ validate(gravity, sphere); });

    apply_forces(gravity, rod, 0.0);
    apply_forces(gravity, sphere, 0.0);

    // Weight is negative along z and proportional to the node mass.
    EXPECT_LT(rod.external_forces()(0, 2), 0.0);
    EXPECT_NEAR(rod.external_forces()(0, 2), -9.80665 * rod.masses()(0), 1e-12);
    EXPECT_NEAR(sphere.external_forces()(0, 2),
                -9.80665 * sphere.total_mass(), 1e-12);
}

TEST(BodyRuleCompatibility, ConstraintsApplyToBothBodyKinds)
{
    CosseratRod rod = make_rod(0.0);
    ConstraintVariant clamp = OneEndFixedBoundaryCondition(
        rod.positions().row(0).transpose(), rod.frames()[0]);

    EXPECT_NO_THROW({ validate(clamp, rod); });

    rod.mutable_velocities().setConstant(5.0);
    rod.mutable_angular_velocities().setConstant(5.0);
    constrain_rates(clamp, rod, 0.0);

    EXPECT_LT(rod.velocities().row(0).norm(), 1e-12);
    EXPECT_LT(rod.angular_velocities().row(0).norm(), 1e-12);
    // Only the constrained end is pinned.
    EXPECT_GT(rod.velocities().row(1).norm(), 1.0);
}

TEST(BodyRuleCompatibility, DampersApplyToBothBodyKinds)
{
    CosseratRod rod = make_rod(0.0);
    Sphere sphere(Eigen::Vector3d::Zero(), 0.02, 1000.0);
    DamperVariant damper = UniformAnalyticalDamper(0.4, 1.0e-4);

    EXPECT_NO_THROW({ validate(damper, rod); });
    EXPECT_NO_THROW({ validate(damper, sphere); });

    rod.mutable_velocities().setConstant(2.0);
    sphere.mutable_angular_velocities().setConstant(3.0);
    dampen_rates(damper, rod, 0.0);
    dampen_rates(damper, sphere, 0.0);

    // Rates are scaled down, never up or past zero.
    EXPECT_LT(rod.velocities()(1, 0), 2.0);
    EXPECT_GT(rod.velocities()(1, 0), 0.0);
    EXPECT_LT(sphere.angular_velocities()(0, 0), 3.0);
    EXPECT_GT(sphere.angular_velocities()(0, 0), 0.0);
}

TEST(BodyRuleCompatibility, JointsApplyBetweenTwoBodies)
{
    CosseratRod rod_one = make_rod(0.0);
    CosseratRod rod_two = make_rod(0.21);  // a 0.01 gap past rod_one's tip
    JointVariant joint =
        FixedJoint(1.0e5, 0.0, 1.0e1, 0.0, Eigen::Matrix3d::Identity());

    EXPECT_NO_THROW({ validate(joint, rod_one, rod_two); });

    apply_forces(joint, rod_one, -1, rod_two, 0, 0.0);

    const Eigen::Vector3d force_one =
        rod_one.external_forces().row(rod_one.num_nodes() - 1).transpose();
    const Eigen::Vector3d force_two = rod_two.external_forces().row(0).transpose();

    // A linear spring over a 0.01 gap, equal and opposite.
    EXPECT_NEAR(force_one.norm(), 1.0e5 * 0.01, 1e-6);
    EXPECT_LT((force_one + force_two).norm(), 1e-9);
}

// ---------------------------------------------------------------------------
// Dispatch through the body variant
//
// This is the shape the simulation graph uses: a rule held in a variant is
// applied to a body held in a variant, with neither side knowing the other's
// alternative.
// ---------------------------------------------------------------------------

TEST(BodyRuleCompatibility, RulesApplyThroughTheBodyVariant)
{
    std::vector<BodyVariant> bodies;
    bodies.emplace_back(make_rod(0.0));
    bodies.emplace_back(Sphere(Eigen::Vector3d::Zero(), 0.02, 1000.0));
    bodies.emplace_back(Cylinder(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
                                 Eigen::Vector3d::UnitX(), 0.2, 0.01, 1000.0));

    ForceTorqueVariant gravity = GravityForceZ{};

    for (BodyVariant& body : bodies)
    {
        std::visit([&](auto& concrete) {
            EXPECT_NO_THROW({ validate(gravity, concrete); });
            apply_forces(gravity, concrete, 0.0);
            EXPECT_LT(concrete.external_forces()(0, 2), 0.0);
        }, body);
    }
}

TEST(BodyRuleCompatibility, ZeroingClearsLoadsAppliedThroughTheVariant)
{
    BodyVariant body = make_rod(0.0);
    ForceTorqueVariant gravity = GravityForceZ{};

    std::visit([&](auto& concrete) {
        apply_forces(gravity, concrete, 0.0);
        EXPECT_GT(concrete.external_forces().cwiseAbs().maxCoeff(), 0.0);
        concrete.zero_out_external_forces_and_torques(0.0 /* time */);
        EXPECT_LT(concrete.external_forces().cwiseAbs().maxCoeff(), 1e-15);
    }, body);
}

// A body kept as a const reference can still be read, which is what the
// diagnostics rely on.
TEST(BodyRuleCompatibility, ConstBodiesRemainReadable)
{
    const CosseratRod rod = make_rod(0.0);

    EXPECT_EQ(rod.positions().rows(), rod.num_nodes());
    EXPECT_EQ(rod.frames().size(), static_cast<std::size_t>(rod.num_elements()));
    EXPECT_GT(rod.masses().sum(), 0.0);
}

}  // namespace
}  // namespace cosserat::physics
