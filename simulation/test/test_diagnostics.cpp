#include "simulation/diagnostics.hpp"

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace cosserat::simulation {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

/** A system offering write but not write_debug, to exercise the concepts. */
struct WriteOnlySystem
{
    void write(const std::filesystem::path& out_dir) const
    {
        std::ofstream stream(out_dir / "marker.txt");
        stream << "written\n";
    }
};

/** A system offering neither entry point. */
struct OpaqueSystem
{
    int value = 0;
};

static_assert(Writeable<WriteOnlySystem>);
static_assert(!DebugWriteable<WriteOnlySystem>);
static_assert(!Writeable<OpaqueSystem>);
static_assert(!DebugWriteable<OpaqueSystem>);
static_assert(Writeable<physics::CosseratRod>);
static_assert(DebugWriteable<physics::CosseratRod>);
static_assert(Writeable<physics::RigidBody>);
static_assert(DebugWriteable<physics::RigidBody>);

physics::CosseratRod make_rod()
{
    return physics::straight_cosserat_rod(
        4, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 1.0, 0.05, 1000.0, 1.0e6, false);
}

physics::Sphere make_sphere()
{
    return physics::Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0);
}

/** Directory names directly beneath a path, sorted the way a shell would. */
std::vector<std::string> sorted_entries(const std::filesystem::path& directory)
{
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

/**
 * Every test gets its own scratch root, removed afterwards. The root is not
 * created here: several tests need to observe the manager creating it.
 */
class DiagnosticsTest : public ::testing::Test
{
protected:
    std::filesystem::path m_root;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_root = std::filesystem::temp_directory_path()
            / (std::string("diagnostics_") + info->test_suite_name() + "_"
               + info->name());
        std::filesystem::remove_all(m_root);
    }

    void TearDown() override { std::filesystem::remove_all(m_root); }
};

// ---------------------------------------------------------------------------
// BasePathManager: layout
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, ConstructionCreatesTheRoot)
{
    ASSERT_FALSE(std::filesystem::exists(m_root));

    const BasePathManager manager(m_root, "rod_one");

    EXPECT_TRUE(std::filesystem::is_directory(m_root));
    EXPECT_EQ(manager.base_path(), m_root);
    EXPECT_EQ(manager.body_name(), "rod_one");
}

TEST_F(DiagnosticsTest, ConstructionAcceptsAnExistingRoot)
{
    std::filesystem::create_directories(m_root);

    EXPECT_NO_THROW({ BasePathManager(m_root, "rod_one"); });
    EXPECT_TRUE(std::filesystem::is_directory(m_root));
}

// The body name is the last component, so two bodies at one step land in
// sibling directories rather than on top of each other.
TEST_F(DiagnosticsTest, DirectoryIsStepThenBody)
{
    const BasePathManager manager(m_root, "rod_one");

    const std::filesystem::path directory = manager.get_next_dir(0.42, 42);

    EXPECT_EQ(directory.filename(), "rod_one");
    EXPECT_EQ(directory.parent_path().filename(), "step_000000042_st_0.420");
    EXPECT_EQ(directory.parent_path().parent_path(), m_root);
    EXPECT_TRUE(std::filesystem::is_directory(directory));
}

TEST_F(DiagnosticsTest, StepNumbersAreZeroPadded)
{
    const BasePathManager manager(m_root, "body");

    EXPECT_EQ(manager.get_next_dir(0.0, 0).parent_path().filename(),
              "step_000000000_st_0.000");
    EXPECT_EQ(manager.get_next_dir(0.0, 7).parent_path().filename(),
              "step_000000007_st_0.000");
    EXPECT_EQ(manager.get_next_dir(0.0, 999999999).parent_path().filename(),
              "step_999999999_st_0.000");
}

// The reason for the padding: sorting by name must give step order. Without
// it, step_10 sorts before step_2 and any glob-driven post-processing reads
// the frames out of order.
TEST_F(DiagnosticsTest, StepDirectoriesSortIntoStepOrder)
{
    const BasePathManager manager(m_root, "body");
    for (std::uint64_t step = 0; step < 12; ++step)
    {
        manager.get_next_dir(0.01 * static_cast<double>(step), step);
    }

    const std::vector<std::string> names = sorted_entries(m_root);

    ASSERT_EQ(names.size(), 12u);
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        const std::string expected =
            "step_" + std::string(9 - std::to_string(i).size(), '0')
            + std::to_string(i);
        EXPECT_EQ(names[i].substr(0, expected.size()), expected)
            << "entry " << i << " was " << names[i];
    }
}

TEST_F(DiagnosticsTest, TimeIsRenderedToThreeDecimals)
{
    const BasePathManager manager(m_root, "body");

    EXPECT_EQ(manager.get_next_dir(0.5, 1).parent_path().filename(),
              "step_000000001_st_0.500");
    EXPECT_EQ(manager.get_next_dir(12.3456, 2).parent_path().filename(),
              "step_000000002_st_12.346");
    EXPECT_EQ(manager.get_next_dir(-1.25, 3).parent_path().filename(),
              "step_000000003_st_-1.250");
}

TEST_F(DiagnosticsTest, RepeatedCallsForOneStepAreIdempotent)
{
    const BasePathManager manager(m_root, "body");

    const std::filesystem::path first = manager.get_next_dir(0.1, 5);
    const std::filesystem::path second = manager.get_next_dir(0.1, 5);

    EXPECT_EQ(first, second);
    EXPECT_EQ(sorted_entries(m_root).size(), 1u);
}

// Two managers rooted at the same place but naming different bodies write to
// sibling directories under a shared step directory.
TEST_F(DiagnosticsTest, DifferentBodiesShareAStepDirectory)
{
    const BasePathManager rod_manager(m_root, "rod_one");
    const BasePathManager sphere_manager(m_root, "sphere_one");

    const std::filesystem::path rod_dir = rod_manager.get_next_dir(0.2, 20);
    const std::filesystem::path sphere_dir = sphere_manager.get_next_dir(0.2, 20);

    EXPECT_NE(rod_dir, sphere_dir);
    EXPECT_EQ(rod_dir.parent_path(), sphere_dir.parent_path());
    EXPECT_EQ(sorted_entries(m_root).size(), 1u);
    EXPECT_EQ(sorted_entries(rod_dir.parent_path()),
              (std::vector<std::string>{"rod_one", "sphere_one"}));
}

TEST_F(DiagnosticsTest, ManagerIsUsableThroughAConstReference)
{
    const BasePathManager manager(m_root, "body");
    const BasePathManager& reference = manager;

    EXPECT_EQ(reference.base_path(), m_root);
    EXPECT_EQ(reference.body_name(), "body");
    EXPECT_TRUE(std::filesystem::is_directory(reference.get_next_dir(0.0, 0)));
}

// ---------------------------------------------------------------------------
// BasePathManager: rejected inputs
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, RejectsAnEmptyBasePath)
{
    EXPECT_ASSERT_FAILURE(BasePathManager("", "body"));
}

// The previous behaviour accepted this and only failed on the first write.
TEST_F(DiagnosticsTest, RejectsABasePathThatIsAFile)
{
    std::filesystem::create_directories(m_root.parent_path());
    std::ofstream(m_root) << "not a directory\n";
    ASSERT_TRUE(std::filesystem::is_regular_file(m_root));

    EXPECT_ASSERT_FAILURE(BasePathManager(m_root, "body"));
}

TEST_F(DiagnosticsTest, RejectsABadBodyName)
{
    EXPECT_ASSERT_FAILURE(BasePathManager(m_root, ""));
    EXPECT_ASSERT_FAILURE(BasePathManager(m_root, "rod/one"));
    EXPECT_ASSERT_FAILURE(BasePathManager(m_root, "rod\\one"));
}

// An unstable run must not quietly produce a directory named after a NaN.
TEST_F(DiagnosticsTest, RejectsNonFiniteTime)
{
    const BasePathManager manager(m_root, "body");

    EXPECT_ASSERT_FAILURE(manager.get_next_dir(kNaN, 0));
    EXPECT_ASSERT_FAILURE(manager.get_next_dir(kInf, 0));
    EXPECT_ASSERT_FAILURE(manager.get_next_dir(-kInf, 0));
}

// The padding is a convenience, not a limit. A step wider than the reserved
// width is still written; only the name-order sorting of those later
// directories degrades, which is not the diagnostics' business to police.
TEST_F(DiagnosticsTest, AcceptsAStepBeyondThePaddedWidth)
{
    const BasePathManager manager(m_root, "body");

    const std::uint64_t wide = 1234567890;  // ten digits
    EXPECT_NO_THROW({ manager.get_next_dir(0.0, wide); });
    EXPECT_EQ(manager.get_next_dir(0.0, wide).parent_path().filename(),
              "step_1234567890_st_0.000");

    // Nothing near the old cap is special any more.
    EXPECT_NO_THROW({ manager.get_next_dir(0.0, 999999999); });
    EXPECT_NO_THROW({ manager.get_next_dir(0.0, 1000000000); });
}

// ---------------------------------------------------------------------------
// BasicDiagnostics
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, BasicDiagnosticsWritesConfigurationOnly)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 1);

    diagnostics.make_callback(rod, 0.25, 25);

    const std::filesystem::path directory =
        m_root / "step_000000025_st_0.250" / "rod_one";
    EXPECT_TRUE(std::filesystem::is_regular_file(directory / "positions.bin"));
    EXPECT_TRUE(std::filesystem::is_regular_file(directory / "positions.md.json"));
    EXPECT_TRUE(std::filesystem::is_regular_file(directory / "frames.bin"));
    EXPECT_FALSE(std::filesystem::exists(directory / "velocities.bin"));
}

TEST_F(DiagnosticsTest, BasicDiagnosticsExposesItsManager)
{
    const BasicDiagnostics diagnostics(m_root, "rod_one", 1);

    EXPECT_EQ(diagnostics.manager().base_path(), m_root);
    EXPECT_EQ(diagnostics.manager().body_name(), "rod_one");
}

TEST_F(DiagnosticsTest, BasicDiagnosticsWritesEachStepSeparately)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 1);

    for (std::uint64_t step = 0; step < 5; ++step)
    {
        diagnostics.make_callback(rod, 0.1 * static_cast<double>(step), step);
    }

    EXPECT_EQ(sorted_entries(m_root).size(), 5u);
}

TEST_F(DiagnosticsTest, BasicDiagnosticsAcceptsAConstSystem)
{
    const physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 1);

    // Writing is read-only, so a const body binds.
    diagnostics.make_callback(rod, 0.0, 0);

    EXPECT_TRUE(std::filesystem::is_regular_file(
        m_root / "step_000000000_st_0.000" / "rod_one" / "positions.bin"));
}

TEST_F(DiagnosticsTest, BasicDiagnosticsWorksOnARigidBody)
{
    physics::Sphere sphere = make_sphere();
    BasicDiagnostics diagnostics(m_root, "sphere_one", 1);

    diagnostics.make_callback(sphere, 0.0, 0);

    const std::filesystem::path directory =
        m_root / "step_000000000_st_0.000" / "sphere_one";
    EXPECT_EQ(std::filesystem::file_size(directory / "positions.bin"),
              3u * sizeof(double));
}

TEST_F(DiagnosticsTest, BasicDiagnosticsWorksOnAMinimalWriteableSystem)
{
    WriteOnlySystem system;
    BasicDiagnostics diagnostics(m_root, "widget", 1);

    diagnostics.make_callback(system, 0.0, 0);

    EXPECT_TRUE(std::filesystem::is_regular_file(
        m_root / "step_000000000_st_0.000" / "widget" / "marker.txt"));
}

// ---------------------------------------------------------------------------
// DebugDiagnostics
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, DebugDiagnosticsWritesEveryStack)
{
    physics::CosseratRod rod = make_rod();
    DebugDiagnostics diagnostics(m_root, "rod_one", 1);

    diagnostics.make_callback(rod, 0.25, 25);

    const std::filesystem::path directory =
        m_root / "step_000000025_st_0.250" / "rod_one";
    for (const char* name :
         {"positions", "velocities", "accelerations", "internal_forces",
          "external_forces", "masses", "frames", "kappas", "dilatations"})
    {
        EXPECT_TRUE(std::filesystem::is_regular_file(
            directory / (std::string(name) + ".bin")))
            << "missing " << name;
    }
}

TEST_F(DiagnosticsTest, DebugDiagnosticsWritesMoreThanBasic)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics basic(m_root / "basic", "rod_one", 1);
    DebugDiagnostics debug(m_root / "debug", "rod_one", 1);

    basic.make_callback(rod, 0.0, 0);
    debug.make_callback(rod, 0.0, 0);

    const auto count = [](const std::filesystem::path& directory) {
        return sorted_entries(directory).size();
    };
    const std::string step = "step_000000000_st_0.000";

    EXPECT_GT(count(m_root / "debug" / step / "rod_one"),
              count(m_root / "basic" / step / "rod_one"));
}

// ---------------------------------------------------------------------------
// The collision this layout exists to prevent
// ---------------------------------------------------------------------------

// Before the body name became part of the path, writing two bodies at the same
// step left only the second one's files behind.
TEST_F(DiagnosticsTest, TwoBodiesAtOneStepDoNotOverwriteEachOther)
{
    physics::CosseratRod rod = make_rod();
    physics::Sphere sphere = make_sphere();

    BasicDiagnostics rod_diagnostics(m_root, "rod_one", 1);
    BasicDiagnostics sphere_diagnostics(m_root, "sphere_one", 1);

    sphere_diagnostics.make_callback(sphere, 0.0, 100);
    rod_diagnostics.make_callback(rod, 0.0, 100);

    const std::filesystem::path step = m_root / "step_000000100_st_0.000";
    ASSERT_TRUE(std::filesystem::is_directory(step / "sphere_one"));
    ASSERT_TRUE(std::filesystem::is_directory(step / "rod_one"));

    // The rod has five nodes and the sphere one, so the sizes distinguish them.
    EXPECT_EQ(std::filesystem::file_size(step / "rod_one" / "positions.bin"),
              5u * 3u * sizeof(double));
    EXPECT_EQ(std::filesystem::file_size(step / "sphere_one" / "positions.bin"),
              1u * 3u * sizeof(double));
}

TEST_F(DiagnosticsTest, TheSameBodyNameUnderDifferentRootsStaysSeparate)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics first(m_root / "run_a", "rod_one", 1);
    BasicDiagnostics second(m_root / "run_b", "rod_one", 1);

    first.make_callback(rod, 0.0, 0);
    second.make_callback(rod, 0.0, 0);

    const std::string tail = "step_000000000_st_0.000/rod_one/positions.bin";
    EXPECT_TRUE(std::filesystem::is_regular_file(m_root / "run_a" / tail));
    EXPECT_TRUE(std::filesystem::is_regular_file(m_root / "run_b" / tail));
}

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, VariantIsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<DiagnosticVariant>);
    static_assert(std::is_copy_assignable_v<DiagnosticVariant>);

    DiagnosticVariant variant = BasicDiagnostics(m_root, "body", 1);
    const DiagnosticVariant other = DebugDiagnostics(m_root, "body", 1);

    variant = other;

    EXPECT_TRUE(std::holds_alternative<DebugDiagnostics>(variant));
}

TEST_F(DiagnosticsTest, ValidateAcceptsBothAlternativesOnAFullBody)
{
    physics::CosseratRod rod = make_rod();

    DiagnosticVariant basic = BasicDiagnostics(m_root, "rod_one", 1);
    DiagnosticVariant debug = DebugDiagnostics(m_root, "rod_one", 1);

    EXPECT_NO_THROW({ validate(basic, rod); });
    EXPECT_NO_THROW({ validate(debug, rod); });
}

TEST_F(DiagnosticsTest, ValidateAcceptsBasicOnAWriteOnlySystem)
{
    WriteOnlySystem system;
    DiagnosticVariant basic = BasicDiagnostics(m_root, "widget", 1);

    EXPECT_NO_THROW({ validate(basic, system); });
}

// A system with write but no write_debug cannot serve the debug diagnostic.
TEST_F(DiagnosticsTest, ValidateRejectsDebugOnAWriteOnlySystem)
{
    WriteOnlySystem system;
    DiagnosticVariant debug = DebugDiagnostics(m_root, "widget", 1);

    EXPECT_ASSERT_FAILURE(validate(debug, system));
}

TEST_F(DiagnosticsTest, ValidateRejectsBothOnAnOpaqueSystem)
{
    OpaqueSystem system;
    DiagnosticVariant basic = BasicDiagnostics(m_root, "widget", 1);
    DiagnosticVariant debug = DebugDiagnostics(m_root, "widget", 1);

    EXPECT_ASSERT_FAILURE(validate(basic, system));
    EXPECT_ASSERT_FAILURE(validate(debug, system));
}

TEST_F(DiagnosticsTest, MakeCallbackThroughTheVariantMatchesADirectCall)
{
    physics::CosseratRod rod = make_rod();

    BasicDiagnostics direct(m_root / "direct", "rod_one", 1);
    DiagnosticVariant through = BasicDiagnostics(m_root / "through", "rod_one", 1);

    direct.make_callback(rod, 0.5, 5);
    make_callback(through, rod, 0.5, 5);

    const std::string tail = "step_000000005_st_0.500/rod_one/positions.bin";
    ASSERT_TRUE(std::filesystem::is_regular_file(m_root / "direct" / tail));
    ASSERT_TRUE(std::filesystem::is_regular_file(m_root / "through" / tail));
    EXPECT_EQ(std::filesystem::file_size(m_root / "direct" / tail),
              std::filesystem::file_size(m_root / "through" / tail));
}

TEST_F(DiagnosticsTest, MakeCallbackRejectsAnIncompatibleSystem)
{
    WriteOnlySystem system;
    DiagnosticVariant debug = DebugDiagnostics(m_root, "widget", 1);

    EXPECT_ASSERT_FAILURE(make_callback(debug, system, 0.0, 0));
}

// A list of diagnostics, each naming its own body, driven from one loop.
TEST_F(DiagnosticsTest, AListOfDiagnosticsDrivesSeveralBodies)
{
    physics::CosseratRod rod = make_rod();
    physics::Sphere sphere = make_sphere();

    std::vector<DiagnosticVariant> diagnostics;
    diagnostics.emplace_back(BasicDiagnostics(m_root, "rod_one", 1));
    diagnostics.emplace_back(DebugDiagnostics(m_root, "sphere_one", 1));

    validate(diagnostics[0], rod);
    validate(diagnostics[1], sphere);

    for (std::uint64_t step = 0; step < 3; ++step)
    {
        const double time = 0.1 * static_cast<double>(step);
        make_callback(diagnostics[0], rod, time, step);
        make_callback(diagnostics[1], sphere, time, step);
    }

    EXPECT_EQ(sorted_entries(m_root).size(), 3u);
    for (std::uint64_t step = 0; step < 3; ++step)
    {
        const std::string name =
            "step_00000000" + std::to_string(step) + "_st_0."
            + std::to_string(step) + "00";
        EXPECT_EQ(sorted_entries(m_root / name),
                  (std::vector<std::string>{"rod_one", "sphere_one"}))
            << "step " << step;
    }
}

// ---------------------------------------------------------------------------
// StepSchedule
// ---------------------------------------------------------------------------

TEST(StepScheduleTest, IntervalOfOneWritesEveryStep)
{
    const StepSchedule schedule(1);

    EXPECT_EQ(schedule.steps_to_skip(), 1u);
    for (std::uint64_t step = 0; step < 20; ++step)
    {
        EXPECT_TRUE(schedule.should_write(step)) << "step " << step;
    }
}

TEST(StepScheduleTest, IntervalFiresOnMultiplesOnly)
{
    const StepSchedule schedule(5);

    for (std::uint64_t step = 0; step < 21; ++step)
    {
        EXPECT_EQ(schedule.should_write(step), step % 5 == 0) << "step " << step;
    }
}

// Step zero is a multiple of everything, so the initial state is always
// recorded whatever the interval.
TEST(StepScheduleTest, StepZeroIsAlwaysWritten)
{
    for (std::uint64_t interval : {1u, 2u, 7u, 1000u})
    {
        EXPECT_TRUE(StepSchedule(interval).should_write(0))
            << "interval " << interval;
    }
}

TEST(StepScheduleTest, LargeIntervalsAndStepsBehave)
{
    const StepSchedule schedule(1000);

    EXPECT_TRUE(schedule.should_write(0));
    EXPECT_FALSE(schedule.should_write(999));
    EXPECT_TRUE(schedule.should_write(1000));
    EXPECT_TRUE(schedule.should_write(1000000000));
    EXPECT_FALSE(schedule.should_write(1000000001));
}

// An interval of zero would never fire, and would divide by zero deciding so.
TEST(StepScheduleDeathTest, RejectsAnIntervalOfZero)
{
    EXPECT_ASSERT_FAILURE(StepSchedule(0));
}

// ---------------------------------------------------------------------------
// Diagnostics honour their schedule
// ---------------------------------------------------------------------------

TEST_F(DiagnosticsTest, DiagnosticsExposeTheirSchedule)
{
    const BasicDiagnostics basic(m_root / "basic", "rod_one", 7);
    const DebugDiagnostics debug(m_root / "debug", "rod_one", 250);

    EXPECT_EQ(basic.schedule().steps_to_skip(), 7u);
    EXPECT_EQ(debug.schedule().steps_to_skip(), 250u);
}

// The return value says whether the step fell on the schedule.
TEST_F(DiagnosticsTest, MakeCallbackReportsWhetherItWrote)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 4);

    EXPECT_TRUE(diagnostics.make_callback(rod, 0.0, 0));
    EXPECT_FALSE(diagnostics.make_callback(rod, 0.1, 1));
    EXPECT_FALSE(diagnostics.make_callback(rod, 0.3, 3));
    EXPECT_TRUE(diagnostics.make_callback(rod, 0.4, 4));
}

// A skipped step must leave no trace at all, not an empty directory.
TEST_F(DiagnosticsTest, SkippedStepsCreateNoDirectory)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 3);

    for (std::uint64_t step = 0; step < 10; ++step)
    {
        diagnostics.make_callback(rod, 0.01 * static_cast<double>(step), step);
    }

    // Steps 0, 3, 6 and 9 only.
    const std::vector<std::string> names = sorted_entries(m_root);
    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[0].substr(0, 14), "step_000000000");
    EXPECT_EQ(names[1].substr(0, 14), "step_000000003");
    EXPECT_EQ(names[2].substr(0, 14), "step_000000006");
    EXPECT_EQ(names[3].substr(0, 14), "step_000000009");
}

TEST_F(DiagnosticsTest, AnIntervalOfOneWritesEveryStep)
{
    physics::CosseratRod rod = make_rod();
    BasicDiagnostics diagnostics(m_root, "rod_one", 1);

    for (std::uint64_t step = 0; step < 6; ++step)
    {
        EXPECT_TRUE(diagnostics.make_callback(rod, 0.0, step)) << "step " << step;
    }

    EXPECT_EQ(sorted_entries(m_root).size(), 6u);
}

TEST_F(DiagnosticsTest, DebugDiagnosticsHonoursItsSchedule)
{
    physics::CosseratRod rod = make_rod();
    DebugDiagnostics diagnostics(m_root, "rod_one", 5);

    for (std::uint64_t step = 0; step < 12; ++step)
    {
        diagnostics.make_callback(rod, 0.0, step);
    }

    // Steps 0, 5 and 10.
    EXPECT_EQ(sorted_entries(m_root).size(), 3u);
}

// A long run with a coarse interval writes only what the interval allows,
// which is the reason the parameter exists.
TEST_F(DiagnosticsTest, ACoarseIntervalKeepsOutputSmall)
{
    physics::Sphere sphere = make_sphere();
    BasicDiagnostics diagnostics(m_root, "sphere_one", 100);

    std::uint64_t written = 0;
    for (std::uint64_t step = 0; step < 1000; ++step)
    {
        if (diagnostics.make_callback(sphere, 1e-4 * static_cast<double>(step), step))
        {
            ++written;
        }
    }

    EXPECT_EQ(written, 10u);
    EXPECT_EQ(sorted_entries(m_root).size(), 10u);
}

// Two bodies on different schedules share the step directories they land on.
TEST_F(DiagnosticsTest, BodiesMayRunOnDifferentSchedules)
{
    physics::CosseratRod rod = make_rod();
    physics::Sphere sphere = make_sphere();

    BasicDiagnostics rod_diagnostics(m_root, "rod_one", 2);
    BasicDiagnostics sphere_diagnostics(m_root, "sphere_one", 4);

    for (std::uint64_t step = 0; step < 8; ++step)
    {
        rod_diagnostics.make_callback(rod, 0.0, step);
        sphere_diagnostics.make_callback(sphere, 0.0, step);
    }

    // The rod writes on 0, 2, 4, 6; the sphere on 0 and 4.
    EXPECT_EQ(sorted_entries(m_root).size(), 4u);
    EXPECT_EQ(sorted_entries(m_root / "step_000000000_st_0.000"),
              (std::vector<std::string>{"rod_one", "sphere_one"}));
    EXPECT_EQ(sorted_entries(m_root / "step_000000002_st_0.000"),
              (std::vector<std::string>{"rod_one"}));
    EXPECT_EQ(sorted_entries(m_root / "step_000000004_st_0.000"),
              (std::vector<std::string>{"rod_one", "sphere_one"}));
}

TEST_F(DiagnosticsTest, ScheduleIsHonouredThroughTheVariant)
{
    physics::CosseratRod rod = make_rod();
    DiagnosticVariant diagnostic = BasicDiagnostics(m_root, "rod_one", 3);

    EXPECT_TRUE(make_callback(diagnostic, rod, 0.0, 0));
    EXPECT_FALSE(make_callback(diagnostic, rod, 0.1, 1));
    EXPECT_FALSE(make_callback(diagnostic, rod, 0.2, 2));
    EXPECT_TRUE(make_callback(diagnostic, rod, 0.3, 3));

    EXPECT_EQ(sorted_entries(m_root).size(), 2u);
}

// Skipping is decided before the system is touched, so a body that cannot
// serve the diagnostic is still rejected on a skipped step.
TEST_F(DiagnosticsTest, IncompatibleSystemsAreRejectedRegardlessOfSchedule)
{
    WriteOnlySystem system;
    DiagnosticVariant debug = DebugDiagnostics(m_root, "widget", 100);

    EXPECT_ASSERT_FAILURE(make_callback(debug, system, 0.0, 1));
    EXPECT_ASSERT_FAILURE(make_callback(debug, system, 0.0, 100));
}

}  // namespace
}  // namespace cosserat::simulation
