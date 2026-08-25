/**
 * @file test_quantities.cpp
 * @brief Tests for @ref quantities.hpp.
 *
 * Each quantity is checked against a state whose answer can be worked out by
 * hand: a rod translating rigidly at a known speed, a rod stretched by a known
 * factor, a body held still by a constraint. The whole-collection forms are
 * then checked to agree with summing or maximising the per-body ones, so the
 * two cannot drift apart.
 */

#include "physics/quantities.hpp"

#include "physics/bodies.hpp"
#include "physics/constraints.hpp"
#include "physics/forces.hpp"
#include "physics/mesh_body.hpp"
#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

#include "simulation/simulation_graph.hpp"
#include "simulation/solver.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace cosserat::physics {
namespace {

constexpr double kTol = 1e-12;
constexpr double kDt = 1e-4;

/** A straight rod along +z, at rest. */
CosseratRod make_rod(std::int64_t elements = 8)
{
    return straight_cosserat_rod(
        elements, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitY(), 1.0, 0.05, 1000.0, 1.0e6, false, 1e-12);
}

/**
 * @brief A graph holding bodies, keeping them reachable after finalizing.
 *
 * The graph shares its bodies rather than owning them outright, so the handles
 * kept here refer to the same bodies the quantities will walk.
 */
struct Scene
{
    std::vector<std::shared_ptr<BodyVariant>> held;
    simulation::SimulationGraph graph;

    /**
     * @brief Adds a body under a name.
     * @param name Name to register it under.
     * @param body The body to add.
     * @return A handle to it, for setting state afterwards.
     */
    std::shared_ptr<BodyVariant> add(const std::string& name, BodyVariant body)
    {
        auto shared = std::make_shared<BodyVariant>(std::move(body));
        held.push_back(shared);
        graph.add_body(name, BodyVariantWrapper(shared));
        return shared;
    }
};

// ---------------------------------------------------------------------------
// The concept
// ---------------------------------------------------------------------------

TEST(BodyCollectionConcept, AGraphIsACollectionAndOtherThingsAreNot)
{
    static_assert(BodyCollection<simulation::SimulationGraph>);

    struct NoSuchMethod {};
    static_assert(!BodyCollection<NoSuchMethod>);

    // Has the name but returns something that cannot be walked.
    struct WrongReturn { int final_systems() {return 0;} };
    static_assert(!BodyCollection<WrongReturn>);

    // Walkable, but the elements hold no body.
    struct NoBodies { std::vector<int>& final_systems(); };
    static_assert(!BodyCollection<NoBodies>);

    SUCCEED();
}

// ---------------------------------------------------------------------------
// Kinetic energy of one body
// ---------------------------------------------------------------------------

TEST(KineticEnergyTest, ARodAtRestHasNone)
{
    const CosseratRod rod = make_rod();

    EXPECT_NEAR(kinetic_energy(rod), 0.0, kTol);
    EXPECT_NEAR(translational_kinetic_energy(rod), 0.0, kTol);
    EXPECT_NEAR(rotational_kinetic_energy(rod), 0.0, kTol);
}

// Every node at one speed, so the answer is half the whole mass times the
// speed squared.
TEST(KineticEnergyTest, RigidTranslationMatchesTheClosedForm)
{
    CosseratRod rod = make_rod();
    const double speed = 3.0;
    rod.mutable_velocities().col(0).setConstant(speed);

    const double expected = 0.5 * rod.masses().sum() * speed * speed;
    EXPECT_NEAR(translational_kinetic_energy(rod), expected, 1e-9);
    // Nothing is turning, so that is the whole of it.
    EXPECT_NEAR(kinetic_energy(rod), expected, 1e-9);
}

TEST(KineticEnergyTest, ItGrowsWithTheSquareOfSpeed)
{
    CosseratRod slow = make_rod();
    CosseratRod fast = make_rod();
    slow.mutable_velocities().col(0).setConstant(1.0);
    fast.mutable_velocities().col(0).setConstant(2.0);

    EXPECT_NEAR(kinetic_energy(fast), 4.0 * kinetic_energy(slow), 1e-9);
}

TEST(KineticEnergyTest, RotationIsCountedSeparatelyAndAddedIn)
{
    CosseratRod rod = make_rod();
    rod.mutable_angular_velocities().col(2).setConstant(5.0);

    const double rotational = rotational_kinetic_energy(rod);
    EXPECT_GT(rotational, 0.0);
    // Nothing translating, so the total is the rotational part alone.
    EXPECT_NEAR(translational_kinetic_energy(rod), 0.0, kTol);
    EXPECT_NEAR(kinetic_energy(rod), rotational, kTol);

    // Adding translation adds to it rather than replacing it.
    rod.mutable_velocities().col(0).setConstant(1.0);
    EXPECT_NEAR(kinetic_energy(rod),
                rotational + translational_kinetic_energy(rod), 1e-12);
}

TEST(KineticEnergyTest, IsNeverNegative)
{
    CosseratRod rod = make_rod();
    rod.mutable_velocities().setRandom();
    rod.mutable_angular_velocities().setRandom();

    EXPECT_GE(kinetic_energy(rod), 0.0);
    EXPECT_GE(translational_kinetic_energy(rod), 0.0);
    EXPECT_GE(rotational_kinetic_energy(rod), 0.0);
}

TEST(KineticEnergyTest, ARigidBodyIsMeasuredTheSameWay)
{
    Sphere ball(Eigen::Vector3d::Zero(), 0.1, 1000.0);
    EXPECT_NEAR(kinetic_energy(ball), 0.0, kTol);

    ball.mutable_velocities().row(0) << 1.0, 0.0, 0.0;
    EXPECT_NEAR(kinetic_energy(ball), 0.5 * ball.masses()(0), 1e-12);
}

// ---------------------------------------------------------------------------
// Speed of one body
// ---------------------------------------------------------------------------

TEST(MaxSpeedTest, IsZeroAtRest)
{
    EXPECT_NEAR(max_speed(make_rod()), 0.0, kTol);
}

TEST(MaxSpeedTest, FindsTheFastestNodeNotTheAverage)
{
    CosseratRod rod = make_rod();
    rod.mutable_velocities().col(0).setConstant(1.0);
    rod.mutable_velocities().row(3) << 0.0, 4.0, 3.0;  // speed five

    EXPECT_NEAR(max_speed(rod), 5.0, 1e-12);
}

// It reads a speed, so it is blind to a body turning on the spot. That is the
// trade against kinetic energy.
TEST(MaxSpeedTest, DoesNotSeePureRotation)
{
    CosseratRod rod = make_rod();
    rod.mutable_angular_velocities().col(2).setConstant(10.0);

    EXPECT_NEAR(max_speed(rod), 0.0, kTol);
    EXPECT_GT(kinetic_energy(rod), 0.0);
}

// ---------------------------------------------------------------------------
// Axial strain of one rod
// ---------------------------------------------------------------------------

TEST(AxialStrainTest, AnUnstretchedRodHasNone)
{
    EXPECT_NEAR(max_axial_elastic_strain(make_rod()), 0.0, 1e-12);
}

// Stretching by a known factor gives a known strain, because without shear the
// axial strain is the dilatation less one.
TEST(AxialStrainTest, MatchesTheDilatationLessOne)
{
    CosseratRod rod = make_rod();
    const double stretch = 1.05;
    rod.mutable_positions().col(2) *= stretch;
    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_NEAR(max_axial_elastic_strain(rod), stretch - 1.0, 1e-9);
    EXPECT_NEAR(max_axial_elastic_strain(rod),
                (rod.dilatations().array() - 1.0).abs().maxCoeff(), 1e-9);
}

TEST(AxialStrainTest, CompressionCountsAsMuchAsStretching)
{
    CosseratRod squashed = make_rod();
    squashed.mutable_positions().col(2) *= 0.95;
    squashed.compute_internal_forces_and_torques(0.0);

    EXPECT_NEAR(max_axial_elastic_strain(squashed), 0.05, 1e-9);
}

// What is measured is the elastic part, meaning the strain less whatever the
// rod considers its rest state, since that is the part the constitutive law
// turns into stress. A rod from the straight builder rests unstrained, so the
// two coincide throughout these tests. Worth pinning, because every other
// expectation here relies on it.
TEST(AxialStrainTest, AStraightRodRestsUnstrainedSoElasticIsTheWholeStrain)
{
    CosseratRod rod = make_rod();
    ASSERT_LT(rod.rest_sigmas().cwiseAbs().maxCoeff(), 1e-15)
        << "the builder gave the rod a rest strain, so elastic and total "
           "strain no longer agree and the expectations below are wrong";

    rod.mutable_positions().col(2) *= 1.05;
    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_NEAR(max_axial_elastic_strain(rod),
                rod.sigmas().col(2).cwiseAbs().maxCoeff(), 1e-15);
}

/**
 * @brief A stand in carrying whatever strains a test wants to give it.
 *
 * Every rod the builders produce rests unstrained, so a rod cannot show
 * whether the rest strain is really being subtracted: the subtraction is a no
 * op for all of them. The quantity is written against a concept rather than
 * against @ref CosseratRod, so a two field stub satisfies it and can carry a
 * rest strain that a built rod cannot.
 */
struct StrainStub
{
public: // Members
    /** @brief The current shear and stretch strain. */
    Vector3DStack strains;

    /** @brief The strain the body considers its rest state. */
    Vector3DStack rest_strains;

public: // Methods
    /** @brief The current strains. */
    const Vector3DStack& sigmas() const {return strains;}

    /** @brief The rest strains. */
    const Vector3DStack& rest_sigmas() const {return rest_strains;}
};

// The elastic strain is what is left after the rest state is taken off, since
// that is the part the constitutive law turns into stress. A body sitting
// exactly at its rest strain is under no stress however stretched it looks.
TEST(AxialStrainTest, TheRestStrainIsSubtracted)
{
    static_assert(detail::HasStrains<StrainStub>);

    StrainStub stub;
    stub.strains = Vector3DStack::Zero(3, 3);
    stub.rest_strains = Vector3DStack::Zero(3, 3);
    stub.strains.col(2) << 0.05, 0.02, 0.01;

    // With no rest strain, the elastic strain is the whole of it.
    EXPECT_NEAR(max_axial_elastic_strain(stub), 0.05, 1e-15);

    // Resting at that same strain leaves nothing elastic at all.
    stub.rest_strains = stub.strains;
    EXPECT_NEAR(max_axial_elastic_strain(stub), 0.0, 1e-15);

    // And a rest strain the body is not at leaves the difference.
    stub.rest_strains.col(2) << 0.04, 0.02, 0.01;
    EXPECT_NEAR(max_axial_elastic_strain(stub), 0.01, 1e-15);
}

// A body held in compression relative to its rest state is as strained as one
// held in tension, so the magnitude is what counts.
TEST(AxialStrainTest, TheSignOfTheDifferenceDoesNotMatter)
{
    StrainStub stub;
    stub.strains = Vector3DStack::Zero(2, 3);
    stub.rest_strains = Vector3DStack::Zero(2, 3);
    stub.rest_strains.col(2) << 0.03, 0.0;

    EXPECT_NEAR(max_axial_elastic_strain(stub), 0.03, 1e-15);
}

// ---------------------------------------------------------------------------
// Over a whole collection
// ---------------------------------------------------------------------------

TEST(CollectionQuantitiesTest, EnergyAddsUpAcrossBodies)
{
    Scene scene;
    auto first = scene.add("a", make_rod());
    auto second = scene.add("b", make_rod());
    scene.graph.finalize();

    std::get<CosseratRod>(*first).mutable_velocities().col(0).setConstant(2.0);
    const double alone = total_kinetic_energy(scene.graph);

    std::get<CosseratRod>(*second).mutable_velocities().col(0).setConstant(2.0);

    EXPECT_NEAR(total_kinetic_energy(scene.graph), 2.0 * alone, 1e-9);
    // And it agrees with summing the bodies by hand.
    EXPECT_NEAR(
        total_kinetic_energy(scene.graph),
        kinetic_energy(std::get<CosseratRod>(*first))
            + kinetic_energy(std::get<CosseratRod>(*second)),
        1e-12);
}

TEST(CollectionQuantitiesTest, SpeedIsTheMaximumNotTheSum)
{
    Scene scene;
    auto first = scene.add("a", make_rod());
    auto second = scene.add("b", make_rod());
    scene.graph.finalize();

    std::get<CosseratRod>(*first).mutable_velocities().col(0).setConstant(1.0);
    std::get<CosseratRod>(*second).mutable_velocities().col(0).setConstant(7.0);

    EXPECT_NEAR(max_speed(scene.graph), 7.0, 1e-12);
}

TEST(CollectionQuantitiesTest, StrainIsTheMaximumOverTheRods)
{
    Scene scene;
    auto first = scene.add("a", make_rod());
    auto second = scene.add("b", make_rod());
    scene.graph.finalize();

    CosseratRod& stretched = std::get<CosseratRod>(*second);
    stretched.mutable_positions().col(2) *= 1.05;
    stretched.compute_internal_forces_and_torques(0.0);

    EXPECT_NEAR(max_axial_elastic_strain(scene.graph), 0.05, 1e-9);
}

TEST(CollectionQuantitiesTest, AnUnmovingCollectionMeasuresZero)
{
    Scene scene;
    scene.add("rod", make_rod());
    scene.add("ball", Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0));
    scene.graph.finalize();

    EXPECT_NEAR(total_kinetic_energy(scene.graph), 0.0, kTol);
    EXPECT_NEAR(max_speed(scene.graph), 0.0, kTol);
    EXPECT_NEAR(max_axial_elastic_strain(scene.graph), 0.0, kTol);
}

// A rigid body cannot be stretched, so it contributes nothing to the strain
// and a collection of nothing else measures zero however fast it is moving.
TEST(CollectionQuantitiesTest, RigidBodiesAreInvisibleToTheStrain)
{
    Scene scene;
    auto held = scene.add("ball", Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0));
    scene.graph.finalize();

    std::get<Sphere>(*held).mutable_velocities().row(0) << 100.0, 0.0, 0.0;

    EXPECT_NEAR(max_axial_elastic_strain(scene.graph), 0.0, kTol);
    // Though it is plainly moving, by the other two measures.
    EXPECT_GT(total_kinetic_energy(scene.graph), 0.0);
    EXPECT_NEAR(max_speed(scene.graph), 100.0, 1e-9);
}

TEST(CollectionQuantitiesTest, MeshBodiesAreCountedLikeAnyRigidBody)
{
    Scene scene;
    auto held = scene.add(
        "block",
        MeshBody(math::make_box_mesh(Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d(0.3, 0.4, 0.5)),
                 1750.0, 0.1, true));
    scene.graph.finalize();

    MeshBody& block = std::get<MeshBody>(*held);
    block.mutable_velocities().row(0) << 2.0, 0.0, 0.0;

    EXPECT_NEAR(total_kinetic_energy(scene.graph),
                0.5 * block.masses()(0) * 4.0, 1e-9);
    EXPECT_NEAR(max_speed(scene.graph), 2.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Bodies held in place
// ---------------------------------------------------------------------------

// Nothing filters constrained bodies out, and nothing needs to. Their rates
// are pinned to zero before any of this runs, so they drop out on their own.
TEST(CollectionQuantitiesTest, AConstrainedBodyContributesNothing)
{
    Scene scene;
    auto pinned = scene.add("ball", Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0));
    auto free_body = scene.add("rod", make_rod());

    const Sphere& ball = std::get<Sphere>(*pinned);
    scene.graph.add_constraint(
        "ball",
        FixedConstraint(std::vector<std::int64_t>{0}, ball.positions(),
                        std::vector<std::int64_t>{0}, ball.frames()));
    // Pulled on for the whole run, and held anyway.
    scene.graph.add_forcing_to("ball", GravityForceZ{});
    scene.graph.finalize();

    simulation::Solver<simulation::SimulationGraph> solver(kDt);
    solver.full_solve(scene.graph, 0.0, 0.05);

    EXPECT_NEAR(kinetic_energy(std::get<Sphere>(*pinned)), 0.0, kTol);
    // The unconstrained body is all that is left in the totals.
    EXPECT_NEAR(total_kinetic_energy(scene.graph),
                kinetic_energy(std::get<CosseratRod>(*free_body)), 1e-12);
}

// ---------------------------------------------------------------------------
// Behaviour over a run
// ---------------------------------------------------------------------------

// Energy is what these are for: it falls as motion dies away, which the other
// measures should agree about.
TEST(CollectionQuantitiesTest, EnergyFallsAsADampedSystemSettles)
{
    Scene scene;
    auto held = scene.add("rod", make_rod());
    scene.graph.dampen("rod", UniformAnalyticalDamper(20.0, kDt));
    scene.graph.finalize();

    std::get<CosseratRod>(*held).mutable_velocities().col(0).setConstant(1.0);
    const double before = total_kinetic_energy(scene.graph);
    const double speed_before = max_speed(scene.graph);

    simulation::Solver<simulation::SimulationGraph> solver(kDt);
    solver.full_solve(scene.graph, 0.0, 0.2);

    EXPECT_LT(total_kinetic_energy(scene.graph), before);
    EXPECT_LT(max_speed(scene.graph), speed_before);
}

}  // namespace
}  // namespace cosserat::physics
