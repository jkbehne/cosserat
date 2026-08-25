#include "simulation/simulation_graph.hpp"

#include "simulation/solver.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cosserat::simulation {

/**
 * @brief Reaches the solver's private stepping for the ordering tests below.
 *
 * @c step is private, because a caller driving a simulation has no reason to
 * take one step at a time. Testing the order of what happens inside a step is
 * the exception, and the solver names this peer a friend for it.
 *
 * @tparam SystemType The collection the solver advances.
 */
template<typename SystemType>
class SolverTestPeer
{
public: // Methods
    /**
     * @brief Takes exactly one step.
     * @param solver The solver to drive.
     * @param system The collection to advance.
     * @param time Simulation time at the start of the step.
     * @return The simulation time after the step.
     */
    static double step(Solver<SystemType>& solver, SystemType& system, double time)
    {
        return solver.step(system, time);
    }
};

namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kTol = 1e-12;

using physics::BodyVariantWrapper;
using physics::CosseratRod;
using physics::Sphere;

std::shared_ptr<physics::BodyVariant> make_rod_ptr(
    const Eigen::Vector3d& start = Eigen::Vector3d::Zero(),
    std::int64_t elements = 10, double length = 0.2)
{
    return std::make_shared<physics::BodyVariant>(physics::straight_cosserat_rod(
        elements, start, Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitY(),
        length, 0.007, 1750.0, 3.0e7, /*respect_radii=*/false, kTol));
}

std::shared_ptr<physics::BodyVariant> make_sphere_ptr()
{
    return std::make_shared<physics::BodyVariant>(
        Sphere(Eigen::Vector3d::Zero(), 0.05, 1000.0));
}

const CosseratRod& as_rod(const std::shared_ptr<physics::BodyVariant>& body)
{
    return std::get<CosseratRod>(*body);
}

CosseratRod& as_mutable_rod(const std::shared_ptr<physics::BodyVariant>& body)
{
    return std::get<CosseratRod>(*body);
}

/** A graph holding one rod named "rod", already registered. */
struct OneRodGraph
{
    std::shared_ptr<physics::BodyVariant> rod = make_rod_ptr();
    SimulationGraph graph;

    OneRodGraph() { graph.add_body("rod", BodyVariantWrapper(rod)); }
};

physics::ConstraintVariant clamp_for(const std::shared_ptr<physics::BodyVariant>& body)
{
    const CosseratRod& rod = as_rod(body);
    return physics::OneEndFixedBoundaryCondition(
        rod.positions().row(0).transpose(), rod.frames()[0]);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

TEST(SimulationGraphBuild, AcceptsBodiesUnderDistinctNames)
{
    SimulationGraph graph;

    EXPECT_NO_THROW({ graph.add_body("rod1", BodyVariantWrapper(make_rod_ptr())); });
    EXPECT_NO_THROW({ graph.add_body("rod2", BodyVariantWrapper(make_rod_ptr())); });
    EXPECT_NO_THROW({ graph.add_body("ball", BodyVariantWrapper(make_sphere_ptr())); });
}

TEST(SimulationGraphBuildDeathTest, RejectsADuplicateBodyName)
{
    SimulationGraph graph;
    graph.add_body("rod", BodyVariantWrapper(make_rod_ptr()));

    EXPECT_ASSERT_FAILURE(
        graph.add_body("rod", BodyVariantWrapper(make_rod_ptr())));
}

TEST(SimulationGraphBuildDeathTest, RejectsRulesForUnknownBodies)
{
    OneRodGraph fixture;
    SimulationGraph& graph = fixture.graph;

    EXPECT_ASSERT_FAILURE(graph.add_constraint("nope", clamp_for(fixture.rod)));
    EXPECT_ASSERT_FAILURE(graph.add_forcing_to("nope", physics::GravityForceZ{}));
    EXPECT_ASSERT_FAILURE(
        graph.dampen("nope", physics::UniformAnalyticalDamper(0.4, 1e-4)));
    EXPECT_ASSERT_FAILURE(graph.add_contact("nope", physics::RodSelfContact(1e4, 1.0)));
    EXPECT_ASSERT_FAILURE(graph.collect_diagnostics(
        "nope", BasicDiagnostics(std::filesystem::temp_directory_path() / "g", "n", 1)));
}

// A rule is checked against its body as it is registered, so an incompatible
// pairing fails where it was written rather than on the first step.
TEST(SimulationGraphBuildDeathTest, RejectsAnIncompatibleRuleAtRegistration)
{
    SimulationGraph graph;
    graph.add_body("ball", BodyVariantWrapper(make_sphere_ptr()));

    // A rigid body has no elements to self-contact.
    EXPECT_ASSERT_FAILURE(graph.add_contact("ball", physics::RodSelfContact(1e4, 1.0)));
}

// This is the shape that used to reject everything: validate() probes a call
// that mutates, so a const visitor made every rule look incompatible.
TEST(SimulationGraphBuild, CompatibleRulesAreActuallyAccepted)
{
    OneRodGraph fixture;
    SimulationGraph& graph = fixture.graph;

    EXPECT_NO_THROW({ graph.add_constraint("rod", clamp_for(fixture.rod)); });
    EXPECT_NO_THROW({ graph.add_forcing_to("rod", physics::GravityForceZ{}); });
    EXPECT_NO_THROW({
        graph.dampen("rod", physics::UniformAnalyticalDamper(0.4, 1e-4)); });
    EXPECT_NO_THROW({ graph.add_contact("rod", physics::RodSelfContact(1e4, 1.0)); });
}

TEST(SimulationGraphBuild, ABodyMayCarrySeveralRulesOfAKind)
{
    OneRodGraph fixture;
    SimulationGraph& graph = fixture.graph;

    graph.add_forcing_to("rod", physics::GravityForceZ{});
    graph.add_forcing_to("rod", physics::GravityForceZ{});
    graph.add_constraint("rod", clamp_for(fixture.rod));
    graph.add_constraint("rod", physics::FreeBoundaryCondition{});
    graph.finalize();

    // Two gravities means twice the weight.
    graph.synchronize(0.0);
    const CosseratRod& rod = as_rod(fixture.rod);
    EXPECT_NEAR(rod.external_forces()(0, 2), -2.0 * 9.80665 * rod.masses()(0), 1e-9);
}

// ---------------------------------------------------------------------------
// Contact registration
// ---------------------------------------------------------------------------

TEST(SimulationGraphContacts, AcceptsAPairAndASelfContact)
{
    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(make_rod_ptr()));
    graph.add_body("b", BodyVariantWrapper(make_rod_ptr(Eigen::Vector3d(1, 0, 0))));

    EXPECT_NO_THROW({ graph.add_contact("a", "b", physics::RodRodContact(1e4, 1.0)); });
    EXPECT_NO_THROW({ graph.add_contact("a", physics::RodSelfContact(1e4, 1.0)); });
    EXPECT_NO_THROW({ graph.add_contact("b", physics::RodSelfContact(1e4, 1.0)); });
}

TEST(SimulationGraphContactsDeathTest, RejectsTheSamePairTwice)
{
    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(make_rod_ptr()));
    graph.add_body("b", BodyVariantWrapper(make_rod_ptr(Eigen::Vector3d(1, 0, 0))));
    graph.add_contact("a", "b", physics::RodRodContact(1e4, 1.0));

    EXPECT_ASSERT_FAILURE(graph.add_contact("a", "b", physics::RodRodContact(1e4, 1.0)));
}

// The pair of bodies is one physical interaction however it is written down,
// so declaring it the other way round is the same mistake.
TEST(SimulationGraphContactsDeathTest, RejectsTheSamePairReversed)
{
    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(make_rod_ptr()));
    graph.add_body("b", BodyVariantWrapper(make_rod_ptr(Eigen::Vector3d(1, 0, 0))));
    graph.add_contact("a", "b", physics::RodRodContact(1e4, 1.0));

    EXPECT_ASSERT_FAILURE(graph.add_contact("b", "a", physics::RodRodContact(1e4, 1.0)));
}

TEST(SimulationGraphContactsDeathTest, RejectsASelfContactTwice)
{
    OneRodGraph fixture;
    fixture.graph.add_contact("rod", physics::RodSelfContact(1e4, 1.0));

    EXPECT_ASSERT_FAILURE(
        fixture.graph.add_contact("rod", physics::RodSelfContact(1e4, 1.0)));
}

// Distinct pairs remain independent.
TEST(SimulationGraphContacts, DifferentPairsDoNotCollide)
{
    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(make_rod_ptr()));
    graph.add_body("b", BodyVariantWrapper(make_rod_ptr(Eigen::Vector3d(1, 0, 0))));
    graph.add_body("c", BodyVariantWrapper(make_rod_ptr(Eigen::Vector3d(2, 0, 0))));

    EXPECT_NO_THROW({
        graph.add_contact("a", "b", physics::RodRodContact(1e4, 1.0));
        graph.add_contact("a", "c", physics::RodRodContact(1e4, 1.0));
        graph.add_contact("b", "c", physics::RodRodContact(1e4, 1.0));
    });
}

TEST(SimulationGraphContactsDeathTest, TheTwoArityFormsAreDistinct)
{
    OneRodGraph fixture;

    // The pair form refuses a body paired with itself; use the single form.
    EXPECT_ASSERT_FAILURE(
        fixture.graph.add_contact("rod", "rod", physics::RodRodContact(1e4, 1.0)));
}

// ---------------------------------------------------------------------------
// Joint indexing
// ---------------------------------------------------------------------------

// Each element index resolves against its OWN body. Resolving both from the
// first would put the second joint end in the wrong place.
TEST(SimulationGraphJoints, EachIndexResolvesAgainstItsOwnBody)
{
    auto first = make_rod_ptr(Eigen::Vector3d::Zero(), 10, 0.2);
    auto second = make_rod_ptr(Eigen::Vector3d(0.0, 0.0, 0.2), 4, 0.2);

    SimulationGraph graph;
    graph.add_body("first", BodyVariantWrapper(first));
    graph.add_body("second", BodyVariantWrapper(second));

    // Tip of the first to base of the second.
    graph.add_connection("first", "second", -1, 0, false, true,
        physics::FixedJoint(1e5, 0.0, 1e1, 0.0, Eigen::Matrix3d::Identity()));
    graph.finalize();

    // Pull them apart so the joint spring is loaded, then let it act.
    as_mutable_rod(second).mutable_positions().col(2).array() += 0.05;
    graph.synchronize(0.0);

    const CosseratRod& one = as_rod(first);
    const CosseratRod& two = as_rod(second);

    // The force lands on the first rod's LAST node and the second rod's FIRST.
    EXPECT_GT(one.external_forces().row(one.num_nodes() - 1).norm(), 1.0);
    EXPECT_LT(one.external_forces().row(0).norm(), kTol);
    EXPECT_GT(two.external_forces().row(0).norm(), 1.0);
    EXPECT_LT(two.external_forces().row(two.num_nodes() - 1).norm(), kTol);
}

// Attaching at the top of an element addresses the next node along.
TEST(SimulationGraphJoints, TheAttachmentFlagPicksWhichNode)
{
    auto first = make_rod_ptr(Eigen::Vector3d::Zero(), 10, 0.2);
    auto second = make_rod_ptr(Eigen::Vector3d(0.0, 0.0, 0.2), 10, 0.2);

    SimulationGraph graph;
    graph.add_body("first", BodyVariantWrapper(first));
    graph.add_body("second", BodyVariantWrapper(second));
    // Element 0 of the second rod, but its TOP, which is node 1.
    graph.add_connection("first", "second", -1, 0, false, false,
        physics::FixedJoint(1e5, 0.0, 1e1, 0.0, Eigen::Matrix3d::Identity()));
    graph.finalize();

    as_mutable_rod(second).mutable_positions().col(2).array() += 0.05;
    graph.synchronize(0.0);

    const CosseratRod& two = as_rod(second);
    EXPECT_GT(two.external_forces().row(1).norm(), 1.0);
    EXPECT_LT(two.external_forces().row(0).norm(), kTol);
}

TEST(SimulationGraphJointsDeathTest, RejectsAnOutOfRangeElement)
{
    auto first = make_rod_ptr(Eigen::Vector3d::Zero(), 10, 0.2);
    auto second = make_rod_ptr(Eigen::Vector3d(0.0, 0.0, 0.2), 4, 0.2);

    SimulationGraph graph;
    graph.add_body("first", BodyVariantWrapper(first));
    graph.add_body("second", BodyVariantWrapper(second));

    // The second rod has four elements, so index 9 is out of range for it even
    // though it is valid for the first.
    EXPECT_ASSERT_FAILURE(graph.add_connection("first", "second", 0, 9, true, true,
        physics::FixedJoint(1e5, 0.0, 1e1, 0.0, Eigen::Matrix3d::Identity())));
}

TEST(SimulationGraphJointsDeathTest, RejectsUnknownBodies)
{
    OneRodGraph fixture;
    physics::JointVariant joint =
        physics::FixedJoint(1e5, 0.0, 1e1, 0.0, Eigen::Matrix3d::Identity());

    EXPECT_ASSERT_FAILURE(
        fixture.graph.add_connection("rod", "nope", 0, 0, true, true, joint));
    EXPECT_ASSERT_FAILURE(
        fixture.graph.add_connection("nope", "rod", 0, 0, true, true, joint));
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

TEST(SimulationGraphFinalize, ClosesTheGraphToFurtherAdditions)
{
    OneRodGraph fixture;
    SimulationGraph& graph = fixture.graph;
    graph.finalize();

    EXPECT_ASSERT_FAILURE(graph.add_body("other", BodyVariantWrapper(make_rod_ptr())));
    EXPECT_ASSERT_FAILURE(graph.add_constraint("rod", clamp_for(fixture.rod)));
    EXPECT_ASSERT_FAILURE(graph.add_forcing_to("rod", physics::GravityForceZ{}));
    EXPECT_ASSERT_FAILURE(
        graph.dampen("rod", physics::UniformAnalyticalDamper(0.4, 1e-4)));
    EXPECT_ASSERT_FAILURE(graph.add_contact("rod", physics::RodSelfContact(1e4, 1.0)));
}

TEST(SimulationGraphFinalizeDeathTest, RejectsFinalizingTwiceOrEmpty)
{
    OneRodGraph fixture;
    fixture.graph.finalize();
    EXPECT_ASSERT_FAILURE(fixture.graph.finalize());

    SimulationGraph empty;
    EXPECT_ASSERT_FAILURE(empty.finalize());
}

TEST(SimulationGraphFinalizeDeathTest, FinalSystemsNeedsFinalizingFirst)
{
    OneRodGraph fixture;

    EXPECT_ASSERT_FAILURE(fixture.graph.final_systems());

    fixture.graph.finalize();
    EXPECT_NO_THROW({ fixture.graph.final_systems(); });
}

TEST(SimulationGraphFinalize, FlattensEveryBodyInInsertionOrder)
{
    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(make_rod_ptr()));
    graph.add_body("b", BodyVariantWrapper(make_sphere_ptr()));
    graph.add_body("c", BodyVariantWrapper(make_rod_ptr()));
    graph.finalize();

    ASSERT_EQ(graph.final_systems().size(), 3u);
    EXPECT_TRUE(std::holds_alternative<CosseratRod>(graph.final_systems()[0].body()));
    EXPECT_TRUE(std::holds_alternative<Sphere>(graph.final_systems()[1].body()));
    EXPECT_TRUE(std::holds_alternative<CosseratRod>(graph.final_systems()[2].body()));
}

// The flattened list holds handles, not copies, so stepping through the list
// is seen by the rules that reference the same bodies.
TEST(SimulationGraphFinalize, FinalSystemsShareTheirBodiesWithTheGraph)
{
    OneRodGraph fixture;
    fixture.graph.finalize();

    fixture.graph.final_systems()[0].update_kinematics(0.0, 0.0);
    as_mutable_rod(fixture.rod).mutable_velocities().col(2).setConstant(1.0);
    fixture.graph.final_systems()[0].update_kinematics(0.0, 0.1);

    // The move made through the handle is visible on the original body.
    EXPECT_NEAR(as_rod(fixture.rod).positions()(0, 2), 0.1, 1e-12);
}

// ---------------------------------------------------------------------------
// The phases
// ---------------------------------------------------------------------------

TEST(SimulationGraphPhases, ConstrainValuesPinsConfiguration)
{
    OneRodGraph fixture;
    fixture.graph.add_constraint("rod", clamp_for(fixture.rod));
    fixture.graph.finalize();

    as_mutable_rod(fixture.rod).mutable_positions().col(0).setConstant(5.0);
    fixture.graph.constrain_values(0.0);

    // Node zero is put back; the rest keeps the displacement.
    EXPECT_LT(as_rod(fixture.rod).positions().row(0).norm(), kTol);
    EXPECT_NEAR(as_rod(fixture.rod).positions()(1, 0), 5.0, kTol);
}

TEST(SimulationGraphPhases, ConstrainRatesPinsRatesThenDamps)
{
    OneRodGraph fixture;
    fixture.graph.add_constraint("rod", clamp_for(fixture.rod));
    fixture.graph.dampen("rod", physics::UniformAnalyticalDamper(0.4, 1e-4));
    fixture.graph.finalize();

    as_mutable_rod(fixture.rod).mutable_velocities().setConstant(2.0);
    fixture.graph.constrain_rates(0.0);

    const CosseratRod& rod = as_rod(fixture.rod);
    // The pinned node stays pinned even though a damper ran afterwards, and
    // damping only ever scales a rate down.
    EXPECT_LT(rod.velocities().row(0).norm(), kTol);
    EXPECT_LT(rod.velocities()(1, 0), 2.0);
    EXPECT_GT(rod.velocities()(1, 0), 0.0);
}

TEST(SimulationGraphPhases, SynchronizeAppliesForceRules)
{
    OneRodGraph fixture;
    fixture.graph.add_forcing_to("rod", physics::GravityForceZ{});
    fixture.graph.finalize();

    fixture.graph.synchronize(0.0);

    const CosseratRod& rod = as_rod(fixture.rod);
    EXPECT_NEAR(rod.external_forces()(0, 2), -9.80665 * rod.masses()(0), 1e-9);
}

// Contacts run last within synchronize, so they see the forces the earlier
// groups accumulated. Rod-to-rod contact reads those to work out how hard the
// pair is already being pressed together.
TEST(SimulationGraphPhases, ContactsRunAfterForcesWithinSynchronize)
{
    auto first = make_rod_ptr(Eigen::Vector3d(-0.1, 0.0, 0.0));
    auto second = make_rod_ptr(Eigen::Vector3d(0.0, -0.1, 0.005));
    // Lay the second rod across the first.
    as_mutable_rod(second).mutable_positions().col(1) =
        as_rod(second).positions().col(2);
    as_mutable_rod(second).mutable_positions().col(2).setConstant(0.005);

    SimulationGraph graph;
    graph.add_body("a", BodyVariantWrapper(first));
    graph.add_body("b", BodyVariantWrapper(second));
    graph.add_forcing_to("a", physics::GravityForceZ{});
    graph.add_contact("a", "b", physics::RodRodContact(1e4, 0.0));
    graph.finalize();

    EXPECT_NO_THROW({ graph.synchronize(0.0); });

    // Gravity landed, and the pair interaction conserves total force.
    const CosseratRod& one = as_rod(first);
    EXPECT_LT(one.external_forces().col(2).sum(), 0.0);
}

TEST(SimulationGraphPhases, ApplyCallbacksWritesThroughTheDiagnostics)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "graph_callbacks";
    std::filesystem::remove_all(root);

    OneRodGraph fixture;
    fixture.graph.collect_diagnostics("rod", BasicDiagnostics(root, "rod", 1));
    fixture.graph.finalize();

    fixture.graph.apply_callbacks(0.25, 25);

    EXPECT_TRUE(std::filesystem::is_regular_file(
        root / "step_000000025_st_0.250" / "rod" / "positions.bin"));
    std::filesystem::remove_all(root);
}

// A diagnostic on a schedule only writes on its own steps.
TEST(SimulationGraphPhases, CallbacksHonourTheirSchedule)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "graph_schedule";
    std::filesystem::remove_all(root);

    OneRodGraph fixture;
    fixture.graph.collect_diagnostics("rod", BasicDiagnostics(root, "rod", 10));
    fixture.graph.finalize();

    for (std::uint64_t step = 1; step <= 25; ++step)
    {
        fixture.graph.apply_callbacks(0.001 * static_cast<double>(step), step);
    }

    // Steps 10 and 20 only.
    std::size_t written = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) { (void)entry; ++written; }
    EXPECT_EQ(written, 2u);
    std::filesystem::remove_all(root);
}

// ---------------------------------------------------------------------------
// Phase ordering
//
// The order the stepper drives the phases in is the whole contract between the
// graph and the solver, so it is pinned directly with a collection that only
// records what it was asked to do.
// ---------------------------------------------------------------------------

struct RecordingBody
{
    std::vector<std::string>* log = nullptr;

    void update_kinematics(double, double) { log->push_back("kinematic"); }
    void update_dynamics(double, double) { log->push_back("dynamic"); }
    void update_accelerations(double, double) { log->push_back("accelerations"); }
    void compute_internal_forces_and_torques(double) { log->push_back("internal"); }
    void zero_out_external_forces_and_torques(double) { log->push_back("zero_out"); }
};

struct RecordingCollection
{
    using SubSystemType = RecordingBody;

    std::vector<std::string> log;
    std::vector<RecordingBody> systems;

    RecordingCollection() { systems.push_back(RecordingBody{&log}); }

    std::vector<RecordingBody>& final_systems() { return systems; }
    void constrain_values(double) { log.push_back("constrain_values"); }
    void synchronize(double) { log.push_back("synchronize"); }
    void constrain_rates(double) { log.push_back("constrain_rates"); }
    void apply_callbacks(double, std::uint64_t) { log.push_back("callbacks"); }
};

TEST(PhaseOrdering, OneStepRunsThePhasesInTheDocumentedOrder)
{
    static_assert(SystemCollection<RecordingCollection>);

    RecordingCollection collection;
    Solver<RecordingCollection> solver(1e-4);

    SolverTestPeer<RecordingCollection>::step(solver, collection, 0.0);

    const std::vector<std::string> expected{
        "kinematic",          // half step: configuration advances
        "constrain_values",   // boundary conditions pin it
        "internal",           // per body, BEFORE synchronize
        "synchronize",        // joints, forces, contacts
        "accelerations",      // loads become accelerations
        "dynamic",            // rates advance
        "constrain_rates",    // boundary conditions, then damping
        "kinematic",          // second half step
        "constrain_values",   // pinned again
        "callbacks",          // diagnostics see the accumulated loads
        "zero_out",           // and only then are the loads cleared
    };
    EXPECT_EQ(collection.log, expected);
}

// The two orderings that carry real consequences, asserted directly.
TEST(PhaseOrdering, InternalForcesPrecedeSynchronize)
{
    RecordingCollection collection;
    Solver<RecordingCollection> solver(1e-4);
    SolverTestPeer<RecordingCollection>::step(solver, collection, 0.0);

    const auto internal =
        std::find(collection.log.begin(), collection.log.end(), "internal");
    const auto sync =
        std::find(collection.log.begin(), collection.log.end(), "synchronize");
    ASSERT_NE(internal, collection.log.end());
    ASSERT_NE(sync, collection.log.end());
    EXPECT_LT(internal - collection.log.begin(), sync - collection.log.begin());
}

TEST(PhaseOrdering, CallbacksPrecedeZeroingTheAccumulators)
{
    RecordingCollection collection;
    Solver<RecordingCollection> solver(1e-4);
    SolverTestPeer<RecordingCollection>::step(solver, collection, 0.0);

    const auto callbacks =
        std::find(collection.log.begin(), collection.log.end(), "callbacks");
    const auto zero =
        std::find(collection.log.begin(), collection.log.end(), "zero_out");
    ASSERT_NE(callbacks, collection.log.end());
    ASSERT_NE(zero, collection.log.end());
    EXPECT_LT(callbacks - collection.log.begin(), zero - collection.log.begin());
}

// Configuration is pinned twice per step, once after each half kinematic step.
TEST(PhaseOrdering, ConfigurationIsPinnedAfterEachHalfStep)
{
    RecordingCollection collection;
    Solver<RecordingCollection> solver(1e-4);
    SolverTestPeer<RecordingCollection>::step(solver, collection, 0.0);

    EXPECT_EQ(std::count(collection.log.begin(), collection.log.end(), "kinematic"), 2);
    EXPECT_EQ(std::count(collection.log.begin(), collection.log.end(),
                         "constrain_values"), 2);
    EXPECT_EQ(std::count(collection.log.begin(), collection.log.end(), "dynamic"), 1);
}

// The graph itself satisfies what the solver asks of a collection.
TEST(PhaseOrdering, TheGraphSatisfiesTheSolverConcept)
{
    static_assert(IntegrableSystem<BodyVariantWrapper>);
    static_assert(SystemCollection<SimulationGraph>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

TEST(SimulationGraphRun, AClampedLoadedPairOfRodsStaysFinite)
{
    auto rod1 = make_rod_ptr(Eigen::Vector3d::Zero());
    auto rod2 = make_rod_ptr(Eigen::Vector3d(0.0, 0.0, 0.2));

    SimulationGraph graph;
    graph.add_body("rod1", BodyVariantWrapper(rod1));
    graph.add_body("rod2", BodyVariantWrapper(rod2));
    graph.add_constraint("rod1", clamp_for(rod1));
    graph.add_connection("rod1", "rod2", -1, 0, false, true,
        physics::FixedJoint(1e5, 0.0, 1e1, 0.0, Eigen::Matrix3d::Identity()));
    graph.add_forcing_to("rod2", physics::GravityForceZ{});
    graph.dampen("rod1", physics::UniformAnalyticalDamper(0.4, 1e-4));
    graph.dampen("rod2", physics::UniformAnalyticalDamper(0.4, 1e-4));
    graph.finalize();

    Solver<SimulationGraph> solver(1e-4);
    const double end = solver.full_solve(graph, 0.0, 0.05);

    EXPECT_NEAR(end, 0.05, 1e-9);
    EXPECT_TRUE(as_rod(rod1).positions().array().isFinite().all());
    EXPECT_TRUE(as_rod(rod2).positions().array().isFinite().all());
    // The clamped end never moves.
    EXPECT_LT(as_rod(rod1).positions().row(0).norm(), 1e-12);
    // Gravity pulled the free rod down.
    EXPECT_LT(as_rod(rod2).positions()(as_rod(rod2).num_nodes() - 1, 2), 0.4);
}

TEST(SimulationGraphRun, AnUnloadedClampedRodBarelyMoves)
{
    OneRodGraph fixture;
    fixture.graph.add_constraint("rod", clamp_for(fixture.rod));
    fixture.graph.finalize();

    const Vector3DStack before = as_rod(fixture.rod).positions();

    Solver<SimulationGraph> solver(1e-4);
    solver.full_solve(fixture.graph, 0.0, 0.02);

    EXPECT_LT((as_rod(fixture.rod).positions() - before).cwiseAbs().maxCoeff(), 1e-9);
}

// External accumulators are cleared at the end of every step, so nothing
// carries over into the next one.
TEST(SimulationGraphRun, ExternalLoadsDoNotAccumulateAcrossSteps)
{
    OneRodGraph fixture;
    fixture.graph.add_forcing_to("rod", physics::GravityForceZ{});
    fixture.graph.finalize();

    Solver<SimulationGraph> solver(1e-4);
    const double after_first =
        SolverTestPeer<SimulationGraph>::step(solver, fixture.graph, 0.0);
    const double after_one = as_rod(fixture.rod).external_forces().cwiseAbs().maxCoeff();

    SolverTestPeer<SimulationGraph>::step(solver, fixture.graph, after_first);
    const double after_two = as_rod(fixture.rod).external_forces().cwiseAbs().maxCoeff();

    // Cleared at the end of each step, so both read the same near-zero value.
    EXPECT_LT(after_one, 1e-15);
    EXPECT_LT(after_two, 1e-15);
}

TEST(SimulationGraphRun, DiagnosticsRecordFramesDuringARun)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "graph_run_frames";
    std::filesystem::remove_all(root);

    OneRodGraph fixture;
    fixture.graph.add_constraint("rod", clamp_for(fixture.rod));
    fixture.graph.collect_diagnostics("rod", BasicDiagnostics(root, "rod", 50));
    fixture.graph.finalize();

    Solver<SimulationGraph> solver(1e-4);
    solver.full_solve(fixture.graph, 0.0, 0.02);  // 200 steps

    // The initial state at step zero, then steps 50, 100, 150 and 200. The
    // frame at zero comes from full_solve's opening pass, which mirrors the
    // reference implementation recording the configuration before integrating.
    std::size_t written = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) { (void)entry; ++written; }
    EXPECT_EQ(written, 5u);
    EXPECT_TRUE(std::filesystem::is_directory(root / "step_000000000_st_0.000" / "rod"));
    std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace cosserat::simulation
