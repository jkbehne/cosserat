/**
 * @file test_stop_criteria.cpp
 * @brief Tests for @ref stop_criteria.hpp.
 *
 * Most of these drive the criterion by hand with a measure the test controls,
 * because that is the only way to say precisely what the measure did and when.
 * A criterion is a small state machine over one number, and the behaviour
 * worth pinning is the timing: when it starts counting, what resets it, and
 * what it reports about why it stopped.
 *
 * The last group runs a real settling scene, to check the pieces compose and
 * that the criterion behaves monotonically in its parameters.
 */

#include <cosserat/simulation/stop_criteria.hpp>

#include <cosserat/physics/bodies.hpp>
#include <cosserat/physics/damping.hpp>
#include <cosserat/physics/quantities.hpp>
#include <cosserat/physics/rigid_body.hpp>
#include <cosserat/physics/rods.hpp>

#include <cosserat/simulation/simulation_graph.hpp>
#include <cosserat/simulation/solver.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace cosserat::simulation {
namespace {

constexpr double kDt = 1e-3;

/**
 * @brief A collection that does nothing, for driving a criterion by hand.
 *
 * A criterion only ever reads what its measure tells it, so the collection it
 * is handed need not be a real one.
 */
struct DummySystem {};

/** A criterion over a measure the test sets directly. */
struct ScriptedMeasure
{
    double value = 0.0;
    double operator()(DummySystem&) const {return value;}
};

/**
 * @brief Builds a criterion reading a value the caller can change.
 *
 * @param source The measure to read; must outlive the criterion.
 * @param threshold Below this counts as quiet.
 * @param required How long quiet must last.
 * @param max_time When to give up.
 * @return The criterion.
 */
auto scripted(
    const ScriptedMeasure& source,
    double threshold,
    double required,
    double max_time
)
{
    auto measure = [&source](DummySystem& system) {return source(system);};
    return SettledWhenQuiet<decltype(measure)>(
        measure, threshold, required, max_time);
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

TEST(StopCriteriaTest, TheBuildersProduceUsableStopCriteria)
{
    static_assert(StopCriterion<
        decltype(settled_when_slow<SimulationGraph>(1.0, 1.0, 1.0)),
        SimulationGraph>);
    static_assert(StopCriterion<
        decltype(settled_when_still<SimulationGraph>(1.0, 1.0, 1.0)),
        SimulationGraph>);
    static_assert(StopCriterion<decltype(stop_at_time(1.0)), SimulationGraph>);
    SUCCEED();
}

TEST(StopCriteriaTest, ItRemembersWhatItWasBuiltWith)
{
    const auto criterion = settled_when_slow<SimulationGraph>(1e-2, 0.25, 10.0);

    EXPECT_DOUBLE_EQ(1e-2, criterion.threshold());
    EXPECT_DOUBLE_EQ(0.25, criterion.required_time_below());
    EXPECT_DOUBLE_EQ(10.0, criterion.max_time());
    EXPECT_FALSE(criterion.settled());
}

// ---------------------------------------------------------------------------
// The timing, driven by hand
// ---------------------------------------------------------------------------

// Quiet is a duration, so none has elapsed at the first question however small
// the measure reads. Otherwise a scene that happens to start at rest, which is
// most of them, would stop before it began.
TEST(StopCriteriaTest, ItDoesNotStopOnTheFirstQuietSample)
{
    ScriptedMeasure measure{0.0};
    auto criterion = scripted(measure, 1.0, 0.5, 100.0);
    DummySystem system;

    EXPECT_FALSE(criterion(system, 0.0));
    EXPECT_FALSE(criterion.settled());
}

TEST(StopCriteriaTest, ItStopsOnceQuietHasLastedLongEnough)
{
    ScriptedMeasure measure{0.0};
    auto criterion = scripted(measure, 1.0, 0.5, 100.0);
    DummySystem system;

    criterion(system, 0.0);
    EXPECT_FALSE(criterion(system, 0.4));
    EXPECT_TRUE(criterion(system, 0.5));
    EXPECT_TRUE(criterion.settled());
}

// A single sample above the threshold restarts the clock, so a lull in the
// middle of a swing does not accumulate toward a stop.
TEST(StopCriteriaTest, ALoudSampleRestartsTheClock)
{
    ScriptedMeasure measure{0.0};
    auto criterion = scripted(measure, 1.0, 0.5, 100.0);
    DummySystem system;

    criterion(system, 0.0);
    criterion(system, 0.4);   // quiet, nearly there

    measure.value = 5.0;
    EXPECT_FALSE(criterion(system, 0.45));

    measure.value = 0.0;
    // The clock now runs from 0.45, so what would have been enough is not.
    EXPECT_FALSE(criterion(system, 0.5));
    EXPECT_FALSE(criterion(system, 0.9));
    // Comfortably past 0.45 + 0.5 rather than exactly on it: the comparison is
    // an ordinary one on doubles, and a difference that should land exactly on
    // the required time can come out a hair under it. See the note on the
    // class. A real run simply takes one more step.
    EXPECT_TRUE(criterion(system, 0.96));
}

TEST(StopCriteriaTest, TimeBelowTracksTheQuietSoFar)
{
    ScriptedMeasure measure{0.0};
    auto criterion = scripted(measure, 1.0, 10.0, 100.0);
    DummySystem system;

    EXPECT_DOUBLE_EQ(0.0, criterion.time_below(0.0));
    criterion(system, 0.0);
    criterion(system, 3.0);
    EXPECT_DOUBLE_EQ(3.0, criterion.time_below(3.0));

    measure.value = 5.0;
    criterion(system, 4.0);
    EXPECT_DOUBLE_EQ(0.0, criterion.time_below(4.0));
}

// Exactly at the threshold counts as quiet: the comparison is strict, so a
// measure resting exactly on it does not keep the run alive forever.
TEST(StopCriteriaTest, ExactlyAtTheThresholdCountsAsQuiet)
{
    ScriptedMeasure measure{1.0};
    auto criterion = scripted(measure, 1.0, 0.5, 100.0);
    DummySystem system;

    criterion(system, 0.0);
    EXPECT_TRUE(criterion(system, 0.5));
    EXPECT_TRUE(criterion.settled());
}

TEST(StopCriteriaTest, AMeasureAboveTheThresholdNeverSettles)
{
    ScriptedMeasure measure{100.0};
    auto criterion = scripted(measure, 1.0, 0.5, 5.0);
    DummySystem system;

    for (double time = 0.0; time < 4.9; time += 0.1)
    {
        ASSERT_FALSE(criterion(system, time)) << "stopped at " << time;
    }
    EXPECT_FALSE(criterion.settled());
}

// ---------------------------------------------------------------------------
// The time limit
// ---------------------------------------------------------------------------

TEST(StopCriteriaTest, TheTimeLimitStopsARunThatNeverQuietens)
{
    ScriptedMeasure measure{100.0};
    auto criterion = scripted(measure, 1.0, 0.5, 5.0);
    DummySystem system;

    EXPECT_FALSE(criterion(system, 4.9));
    EXPECT_TRUE(criterion(system, 5.0));
    // Stopped, but not settled: the state left behind is still moving.
    EXPECT_FALSE(criterion.settled());
}

// The two endings are what the caller has to tell apart, since the solver
// reports only a time.
TEST(StopCriteriaTest, SettledDistinguishesTheTwoEndings)
{
    DummySystem system;

    ScriptedMeasure quiet{0.0};
    auto settles = scripted(quiet, 1.0, 0.5, 100.0);
    settles(system, 0.0);
    settles(system, 1.0);
    EXPECT_TRUE(settles.settled());

    ScriptedMeasure loud{100.0};
    auto times_out = scripted(loud, 1.0, 0.5, 1.0);
    times_out(system, 0.0);
    times_out(system, 1.0);
    EXPECT_FALSE(times_out.settled());
}

// Reached before anything is measured, so there is nothing to report.
TEST(StopCriteriaTest, TheTimeLimitIsCheckedBeforeTheMeasure)
{
    ScriptedMeasure measure{0.0};
    auto criterion = scripted(measure, 1.0, 0.5, 1.0);
    DummySystem system;

    EXPECT_TRUE(criterion(system, 1.0));
    EXPECT_TRUE(std::isnan(criterion.last_measured()));
}

TEST(StopCriteriaTest, LastMeasuredReportsWhatTheMeasureSaid)
{
    ScriptedMeasure measure{7.5};
    auto criterion = scripted(measure, 1.0, 0.5, 100.0);
    DummySystem system;

    EXPECT_TRUE(std::isnan(criterion.last_measured()));
    criterion(system, 0.0);
    EXPECT_DOUBLE_EQ(7.5, criterion.last_measured());
}

// ---------------------------------------------------------------------------
// Rejected settings
// ---------------------------------------------------------------------------

TEST(StopCriteriaDeathTest, RejectsNonsensicalSettings)
{
    ScriptedMeasure measure{0.0};

    EXPECT_DEATH({ scripted(measure, -1.0, 0.5, 10.0); }, "");
    EXPECT_DEATH({ scripted(measure, 1.0, 0.0, 10.0); }, "");
    EXPECT_DEATH({ scripted(measure, 1.0, -1.0, 10.0); }, "");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DEATH({ scripted(measure, nan, 0.5, 10.0); }, "");
    EXPECT_DEATH({ scripted(measure, 1.0, nan, 10.0); }, "");
    EXPECT_DEATH({ scripted(measure, 1.0, 0.5, nan); }, "");
}

TEST(StopCriteriaDeathTest, StopAtTimeRejectsANonFiniteEnd)
{
    EXPECT_DEATH(
        { stop_at_time(std::numeric_limits<double>::quiet_NaN()); }, "");
}

// ---------------------------------------------------------------------------
// Against a real settling scene
// ---------------------------------------------------------------------------

/** A rod given an initial speed and damped, so it comes to rest on its own. */
struct SettlingScene
{
    std::shared_ptr<physics::BodyVariant> rod;
    SimulationGraph graph;

    /**
     * @brief Builds the scene.
     * @param speed Speed to start the rod at.
     * @param damping Damping constant; larger settles sooner.
     */
    explicit SettlingScene(double speed = 1.0, double damping = 5.0)
    {
        rod = std::make_shared<physics::BodyVariant>(
            physics::straight_cosserat_rod(
                8, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
                Eigen::Vector3d::UnitY(), 1.0, 0.05, 1000.0, 1.0e6, false, 1e-12));
        graph.add_body("rod", physics::BodyVariantWrapper(rod));
        graph.dampen("rod", physics::UniformAnalyticalDamper(damping, kDt));
        graph.finalize();
        std::get<physics::CosseratRod>(*rod)
            .mutable_velocities().col(0).setConstant(speed);
    }
};

TEST(StopCriteriaSceneTest, ADampedSceneSettlesBeforeTheLimit)
{
    SettlingScene scene;
    auto criterion = settled_when_slow<SimulationGraph>(1e-3, 0.05, 20.0);
    Solver<SimulationGraph> solver(kDt);

    const double reached = solver.full_solve(scene.graph, criterion, 0.0);

    EXPECT_TRUE(criterion.settled());
    EXPECT_LT(reached, 20.0);
    EXPECT_LE(criterion.last_measured(), 1e-3);
}

// A threshold below anything the scene reaches can only end at the limit.
TEST(StopCriteriaSceneTest, AnUnreachableThresholdRunsToTheLimit)
{
    SettlingScene scene;
    auto criterion = settled_when_slow<SimulationGraph>(0.0, 0.05, 0.5);
    Solver<SimulationGraph> solver(kDt);

    const double reached = solver.full_solve(scene.graph, criterion, 0.0);

    EXPECT_FALSE(criterion.settled());
    EXPECT_NEAR(0.5, reached, kDt);
}

// Both parameters push the stop later, which is the sanity check that the
// criterion is doing what its names say.
TEST(StopCriteriaSceneTest, StricterSettingsStopLater)
{
    const auto stop_time = [](double threshold, double required) {
        SettlingScene scene;
        auto criterion =
            settled_when_slow<SimulationGraph>(threshold, required, 30.0);
        Solver<SimulationGraph> solver(kDt);
        const double reached = solver.full_solve(scene.graph, criterion, 0.0);
        EXPECT_TRUE(criterion.settled()) << threshold << " " << required;
        return reached;
    };

    // A tighter threshold means waiting for the scene to get quieter.
    EXPECT_GT(stop_time(1e-4, 0.05), stop_time(1e-2, 0.05));
    // A longer required time means waiting longer once it has.
    EXPECT_GT(stop_time(1e-3, 0.50), stop_time(1e-3, 0.05));
}

TEST(StopCriteriaSceneTest, TheEnergyFormWorksToo)
{
    SettlingScene scene;
    auto criterion = settled_when_still<SimulationGraph>(1e-9, 0.05, 20.0);
    Solver<SimulationGraph> solver(kDt);

    solver.full_solve(scene.graph, criterion, 0.0);

    EXPECT_TRUE(criterion.settled());
    EXPECT_LE(criterion.last_measured(), 1e-9);
}

// Nothing is filtered: a scene already at rest simply has nothing to wait for
// beyond the required time itself.
TEST(StopCriteriaSceneTest, ASceneStartingAtRestStopsAfterTheRequiredTime)
{
    SettlingScene scene(0.0, 5.0);
    auto criterion = settled_when_slow<SimulationGraph>(1e-6, 0.1, 20.0);
    Solver<SimulationGraph> solver(kDt);

    const double reached = solver.full_solve(scene.graph, criterion, 0.0);

    EXPECT_TRUE(criterion.settled());
    // The required time, and no more than a step over it.
    EXPECT_NEAR(0.1, reached, 2.0 * kDt);
}

// The measure sees the collection the solver is advancing, not a copy of it.
TEST(StopCriteriaSceneTest, TheMeasureReadsTheLiveState)
{
    SettlingScene scene;
    std::vector<double> seen;
    auto recorder = [&seen](SimulationGraph& graph)
    {
        seen.push_back(physics::max_material_point_speed(graph));
        return seen.back();
    };
    SettledWhenQuiet<decltype(recorder)> criterion(recorder, 1e-3, 0.05, 5.0);
    Solver<SimulationGraph> solver(kDt);

    solver.full_solve(scene.graph, criterion, 0.0);

    ASSERT_GT(seen.size(), 2u);
    // It started fast and ended slow, so it was watching a real run.
    EXPECT_GT(seen.front(), seen.back());
    EXPECT_LE(seen.back(), 1e-3);
}

// ---------------------------------------------------------------------------
// Spin, which a nodal speed misses
// ---------------------------------------------------------------------------

// A rigid body has one node, and that node sits on the axis, so a body turning
// in place moves no node at all. The material speed counts the spin, which is
// why it is the default rather than the nodal speed.
TEST(StopCriteriaSceneTest, ASpinningRigidBodyIsNotMistakenForStill)
{
    auto ball = std::make_shared<physics::BodyVariant>(
        physics::Sphere(Eigen::Vector3d::Zero(), 0.5, 1000.0));
    SimulationGraph graph;
    graph.add_body("ball", physics::BodyVariantWrapper(ball));
    graph.finalize();

    std::get<physics::Sphere>(*ball)
        .mutable_angular_velocities().row(0) << 0.0, 0.0, 10.0;

    // Nothing translating, so a nodal speed reads nothing at all.
    EXPECT_NEAR(physics::max_speed(graph), 0.0, 1e-15);
    // The surface is moving at omega times the radius.
    EXPECT_NEAR(physics::max_material_point_speed(graph), 10.0 * 0.5, 1e-12);
}

}  // namespace
}  // namespace cosserat::simulation
