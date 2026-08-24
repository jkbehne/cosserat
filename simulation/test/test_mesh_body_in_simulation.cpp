/**
 * @file test_mesh_body_in_simulation.cpp
 * @brief Tests that a mesh body can be held and driven by a simulation.
 *
 * The pieces are all tested individually elsewhere. What is checked here is
 * that they compose: that a @ref MeshBody survives being put in a
 * @ref BodyVariant, that a @ref RodMeshContact survives being put in a
 * @ref ContactVariant, and that a graph holding both steps without either one
 * being quietly mishandled.
 *
 * The one genuinely new hazard is a mesh body being mistaken for a primitive.
 * It derives from the same rigid body as a sphere and a cylinder, so it
 * exposes their whole interface, and only an explicit exclusion keeps it out
 * of their contact rules. That exclusion is tested here rather than left to be
 * discovered as wrong physics.
 */

#include "physics/bodies.hpp"
#include "physics/contacts.hpp"
#include "physics/mesh_body.hpp"
#include "physics/rod_mesh_contact.hpp"
#include "physics/rods.hpp"

#include "simulation/simulation_graph.hpp"
#include "simulation/solver.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cstdint>
#include <memory>
#include <vector>
#include <variant>

namespace cosserat::physics {
namespace {

constexpr double kDensity = 1750.0;

/** A box mesh body sitting at the origin. */
MeshBody make_obstacle(
    const Eigen::Vector3d& center = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& half_extent = Eigen::Vector3d(0.3, 0.3, 0.3)
)
{
    return MeshBody(
        math::make_box_mesh(center, half_extent), kDensity, 0.2, true);
}

/** A straight rod, by default laid alongside the obstacle's +x face. */
CosseratRod make_rod(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction = Eigen::Vector3d::UnitY()
)
{
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    if (std::abs(normal.dot(direction)) > 0.9) normal = Eigen::Vector3d(0.0, 1.0, 0.0);
    normal = (normal - normal.dot(direction) * direction).normalized();
    return straight_cosserat_rod(
        8, start, direction, normal, 0.8, 0.05, 1000.0, 1.0e6, false, 1e-12);
}

/**
 * @brief A constraint pinning a rigid body exactly where it currently is.
 *
 * A rigid body has one node and one element, so both index lists hold only
 * zero.
 */
FixedConstraint pin_in_place(const RigidBody& body)
{
    return FixedConstraint(
        std::vector<std::int64_t>{0}, body.positions(),
        std::vector<std::int64_t>{0}, body.frames());
}

// ---------------------------------------------------------------------------
// The variants admit the new types
// ---------------------------------------------------------------------------

TEST(MeshBodyVariant, ABodyVariantCanHoldAMeshBody)
{
    BodyVariant body = make_obstacle();

    EXPECT_TRUE(std::holds_alternative<MeshBody>(body));
    // Not confused with the rigid body it derives from.
    EXPECT_FALSE(std::holds_alternative<RigidBody>(body));
}

TEST(MeshBodyVariant, AContactVariantCanHoldARodMeshContact)
{
    ContactVariant contact = RodMeshContact(1.0e4, 10.0, 0.0, 0.0);

    EXPECT_TRUE(std::holds_alternative<RodMeshContact>(contact));
}

TEST(MeshBodyVariant, TheWrapperStepsAMeshBodyLikeAnyOtherBody)
{
    auto held = std::make_shared<BodyVariant>(make_obstacle());
    BodyVariantWrapper wrapper(held);

    EXPECT_NO_THROW({
        wrapper.compute_internal_forces_and_torques(0.0);
        wrapper.update_accelerations(0.0, 1e-4);
        wrapper.update_dynamics(0.0, 1e-4);
        wrapper.update_kinematics(0.0, 0.5e-4);
        wrapper.zero_out_external_forces_and_torques(0.0);
    });
}

// A mesh body has real mass and inertia, so it responds to a load exactly as
// any rigid body does. Nothing about it is inherently static.
TEST(MeshBodyVariant, AMeshBodyAcceleratesUnderALoad)
{
    auto held = std::make_shared<BodyVariant>(make_obstacle());
    BodyVariantWrapper wrapper(held);

    MeshBody& body = std::get<MeshBody>(*held);
    body.mutable_external_forces().row(0) << 0.0, 0.0, -10.0;
    wrapper.update_accelerations(0.0, 1e-4);

    EXPECT_LT(body.accelerations()(0, 2), 0.0);
    EXPECT_NEAR(body.accelerations()(0, 2), -10.0 / body.masses()(0), 1e-9);
}

// ---------------------------------------------------------------------------
// A mesh body must not be mistaken for a primitive
// ---------------------------------------------------------------------------

// It derives from the same rigid body as a sphere, so it exposes radius(),
// length() and everything else those rules read. Only the explicit exclusion
// keeps it out of them; without it a mesh would be silently treated as a
// sphere of its nominal bounding radius.
TEST(MeshBodyVariant, IsNotAcceptedByThePrimitiveContactRoles)
{
    static_assert(ContactableMeshBody<MeshBody>);
    static_assert(CarriesDistanceField<MeshBody>);
    static_assert(!ContactableSphere<MeshBody>);
    static_assert(!ContactableCylinder<MeshBody>);

    // The primitives themselves are unaffected.
    static_assert(ContactableSphere<Sphere>);
    static_assert(ContactableCylinder<Cylinder>);
    static_assert(!CarriesDistanceField<Sphere>);
    SUCCEED();
}

TEST(MeshBodyVariantDeathTest, ValidateRejectsAMeshBodyAsASphereOrCylinder)
{
    CosseratRod rod = make_rod({0.35, -0.4, 0.0});
    MeshBody obstacle = make_obstacle();

    ContactVariant rod_sphere = RodSphereContact(1.0e4, 10.0, 0.0, 0.0);
    ContactVariant rod_cylinder = RodCylinderContact(1.0e4, 10.0, 0.0, 0.0);

    EXPECT_DEATH({ validate(rod_sphere, rod, obstacle); }, "");
    EXPECT_DEATH({ validate(rod_cylinder, rod, obstacle); }, "");
}

TEST(MeshBodyVariant, ValidateAcceptsTheRightPairing)
{
    CosseratRod rod = make_rod({0.35, -0.4, 0.0});
    MeshBody obstacle = make_obstacle();
    ContactVariant contact = RodMeshContact(1.0e4, 10.0, 0.0, 0.0);

    EXPECT_NO_THROW({ validate(contact, rod, obstacle); });
}

TEST(MeshBodyVariantDeathTest, RodMeshContactRejectsANonMeshSecondBody)
{
    CosseratRod rod = make_rod({0.35, -0.4, 0.0});
    Sphere ball(Eigen::Vector3d::Zero(), 0.1, kDensity);
    ContactVariant contact = RodMeshContact(1.0e4, 10.0, 0.0, 0.0);

    EXPECT_DEATH({ validate(contact, rod, ball); }, "");
}

// Order matters, as it does for every other pair contact.
TEST(MeshBodyVariantDeathTest, RodMeshContactRejectsTheReversedOrder)
{
    CosseratRod rod = make_rod({0.35, -0.4, 0.0});
    MeshBody obstacle = make_obstacle();
    ContactVariant contact = RodMeshContact(1.0e4, 10.0, 0.0, 0.0);

    EXPECT_DEATH({ validate(contact, obstacle, rod); }, "");
}

// ---------------------------------------------------------------------------
// Through the simulation graph
// ---------------------------------------------------------------------------

TEST(MeshBodyInGraph, AGraphAcceptsAMeshBodyAndAMeshContact)
{
    auto rod = std::make_shared<BodyVariant>(make_rod({0.35, -0.4, 0.0}));
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());

    simulation::SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(rod));
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));

    EXPECT_NO_THROW({
        graph.add_contact("rod", "obstacle", RodMeshContact(1.0e4, 10.0, 0.0, 0.0));
        graph.finalize();
    });
    EXPECT_EQ(graph.final_systems().size(), 2u);
}

TEST(MeshBodyInGraph, SynchronizeAppliesTheMeshContact)
{
    // Overlapping the obstacle's +x face, so contact is active immediately.
    auto rod = std::make_shared<BodyVariant>(make_rod({0.305, -0.4, 0.0}));
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());

    simulation::SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(rod));
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));
    graph.add_contact("rod", "obstacle", RodMeshContact(1.0e4, 0.0, 0.0, 0.0));
    graph.finalize();

    graph.synchronize(0.0);

    const CosseratRod& pushed = std::get<CosseratRod>(*rod);
    const MeshBody& pushed_back = std::get<MeshBody>(*obstacle);
    EXPECT_GT(pushed.external_forces().cwiseAbs().maxCoeff(), 0.0);
    // Pushed out along +x, and the obstacle pushed the other way.
    EXPECT_GT(pushed.external_forces().col(0).sum(), 0.0);
    EXPECT_LT(pushed_back.external_forces()(0, 0), 0.0);
}

TEST(MeshBodyInGraph, ARodClearOfTheObstacleIsUndisturbed)
{
    auto rod = std::make_shared<BodyVariant>(make_rod({5.0, -0.4, 0.0}));
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());

    simulation::SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(rod));
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));
    graph.add_contact("rod", "obstacle", RodMeshContact(1.0e4, 10.0, 0.0, 0.0));
    graph.finalize();

    graph.synchronize(0.0);

    EXPECT_LT(std::get<CosseratRod>(*rod).external_forces().cwiseAbs().maxCoeff(),
              1e-12);
}

// The whole point of the exercise: a rod driven into a mesh obstacle is turned
// away by it rather than passing through.
TEST(MeshBodyInGraph, ARodDrivenIntoTheObstacleIsStopped)
{
    // Starting clear of the box and moving toward it along -x. The box face
    // is at x = 0.3 and the rod's radius is 0.05, so contact begins once the
    // axis reaches 0.35, and the run is long enough to get there.
    auto rod = std::make_shared<BodyVariant>(make_rod({0.42, -0.4, 0.0}));
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());

    simulation::SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(rod));
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));
    // Pinning the obstacle makes it an immovable wall, which is the case this
    // is currently good for.
    graph.add_constraint("obstacle", pin_in_place(std::get<MeshBody>(*obstacle)));
    graph.add_contact("rod", "obstacle", RodMeshContact(1.0e5, 10.0, 0.0, 0.0));
    graph.finalize();

    std::get<CosseratRod>(*rod).mutable_velocities().col(0).setConstant(-1.0);

    simulation::Solver<simulation::SimulationGraph> solver(1e-5);
    solver.full_solve(graph, 0.0, 0.1);

    const CosseratRod& stopped = std::get<CosseratRod>(*rod);
    EXPECT_TRUE(stopped.positions().array().isFinite().all());

    // Unopposed it would have travelled 0.1 and ended at 0.32, inside the box.
    // Instead its nearest node never crosses the surface at x = 0.35.
    EXPECT_GT(stopped.positions().col(0).minCoeff(), 0.35)
        << "the rod passed into the obstacle";

    // And it was turned around rather than merely slowed: every node is now
    // moving back out along +x.
    EXPECT_GT(stopped.velocities().col(0).minCoeff(), 0.0)
        << "the rod was not repelled";
}

TEST(MeshBodyInGraph, TheObstacleStaysPutWhenConstrained)
{
    auto rod = std::make_shared<BodyVariant>(make_rod({0.305, -0.4, 0.0}));
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());
    const Eigen::Vector3d start =
        std::get<MeshBody>(*obstacle).positions().row(0).transpose();

    simulation::SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(rod));
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));
    graph.add_constraint("obstacle", pin_in_place(std::get<MeshBody>(*obstacle)));
    graph.add_contact("rod", "obstacle", RodMeshContact(1.0e4, 10.0, 0.0, 0.0));
    graph.finalize();

    simulation::Solver<simulation::SimulationGraph> solver(1e-5);
    solver.full_solve(graph, 0.0, 0.01);

    EXPECT_LT((std::get<MeshBody>(*obstacle).positions().row(0).transpose() - start)
                  .norm(), 1e-12);
}

// Rules written for rigid bodies apply to a mesh body unchanged, because it is
// one. Nothing about it is contact specific.
TEST(MeshBodyInGraph, OrdinaryRigidBodyRulesApplyToIt)
{
    auto obstacle = std::make_shared<BodyVariant>(make_obstacle());

    simulation::SimulationGraph graph;
    graph.add_body("obstacle", BodyVariantWrapper(obstacle));

    EXPECT_NO_THROW({
        graph.add_forcing_to("obstacle", GravityForceZ{});
        graph.dampen("obstacle", UniformAnalyticalDamper(0.4, 1e-4));
        graph.finalize();
    });

    graph.synchronize(0.0);
    const MeshBody& body = std::get<MeshBody>(*obstacle);
    EXPECT_NEAR(body.external_forces()(0, 2), -9.80665 * body.masses()(0), 1e-9);
}

}  // namespace
}  // namespace cosserat::physics
