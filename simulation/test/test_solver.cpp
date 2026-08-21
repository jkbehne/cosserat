// Tests for cosserat::simulation::Solver
//
// Build note: link against gtest_main.
//
// IMPORTANT: these tests deliberately live in namespace `cosserat::simulation`
// and NOT in an anonymous namespace. `SolverTestPeer` is forward-declared in
// solver.hpp as `cosserat::simulation::SolverTestPeer`; defining it inside an
// anonymous namespace would produce a *different* type
// (cosserat::simulation::<unnamed>::SolverTestPeer) and the friend declaration
// in Solver would not apply to it.

#include "simulation/solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace cosserat::simulation {

// ---------------------------------------------------------------------------
// Assertion harness
// ---------------------------------------------------------------------------
// utils::nice_assert calls std::abort and is always compiled in, so failures are
// verified with death tests. Per Google's convention, every suite containing one
// is named *DeathTest so the test runner schedules it before threaded tests.
//
// The regex is permissive because nice_assert's diagnostic text is not part of
// its contract. If you decide to guarantee that the message reaches stderr,
// tighten this to match and the tests below gain message coverage for free.
#define EXPECT_ASSERT_ABORT(stmt) EXPECT_DEATH(stmt, ".*")

// ---------------------------------------------------------------------------
// Test peer: grants access to Solver's private methods and state.
// ---------------------------------------------------------------------------
template <typename SystemType>
class SolverTestPeer
{
public:
    using SubSystemType = typename SystemType::SubSystemType;

    static void kinematic_step(Solver<SystemType>& solver, SubSystemType& sub,
                               double time)
    {
        solver.kinematic_step(sub, time);
    }

    static void dynamics_step(Solver<SystemType>& solver, SubSystemType& sub,
                              double time)
    {
        solver.dynamics_step(sub, time);
    }

    static std::uint64_t current_step(const Solver<SystemType>& solver)
    {
        return solver.current_step;
    }

    static double initial_time(const Solver<SystemType>& solver)
    {
        return solver.initial_time;
    }

    static double dt(const Solver<SystemType>& solver) { return solver.dt; }
};

// ---------------------------------------------------------------------------
// Call recording
// ---------------------------------------------------------------------------

inline constexpr int kSystemId = -1;

struct Call
{
    std::string method;
    int id = kSystemId;  // kSystemId for the collection itself, else subsystem index
    double time = 0.0;
    double scale = 0.0;
    std::uint64_t step = 0;
    bool has_scale = false;
    bool has_step = false;
};

class CallLog
{
public:
    void add(Call call) { calls_.push_back(std::move(call)); }

    void clear() { calls_.clear(); }

    const std::vector<Call>& calls() const { return calls_; }

    // Human-readable ordering, e.g. {"0.update_kinematics", "S.constrain_values"}.
    // final_systems() is noise for ordering assertions, so it is excluded by
    // default; use count("final_systems") to assert on it directly.
    std::vector<std::string> sequence(bool include_final_systems = false) const
    {
        std::vector<std::string> out;
        for (const Call& call : calls_)
        {
            if (!include_final_systems && call.method == "final_systems") continue;
            const std::string prefix =
                (call.id == kSystemId) ? std::string("S") : std::to_string(call.id);
            out.push_back(prefix + "." + call.method);
        }
        return out;
    }

    std::size_t count(std::string_view method) const
    {
        std::size_t total = 0;
        for (const Call& call : calls_)
        {
            if (call.method == method) ++total;
        }
        return total;
    }

    std::vector<Call> filter(std::string_view method) const
    {
        std::vector<Call> out;
        for (const Call& call : calls_)
        {
            if (call.method == method) out.push_back(call);
        }
        return out;
    }

    // Convenience for "the Nth call to `method`".
    const Call& nth(std::string_view method, std::size_t index) const
    {
        static const Call kEmpty{};
        std::size_t seen = 0;
        for (const Call& call : calls_)
        {
            if (call.method != method) continue;
            if (seen == index) return call;
            ++seen;
        }
        ADD_FAILURE() << "No call #" << index << " to " << method;
        return kEmpty;
    }

private:
    std::vector<Call> calls_;
};

// ---------------------------------------------------------------------------
// Mocks that satisfy the concepts
// ---------------------------------------------------------------------------

class MockSubSystem
{
public:
    MockSubSystem() = default;
    MockSubSystem(CallLog* log, int id) : log_(log), id_(id) {}

    void update_kinematics(double time, double scale)
    {
        record("update_kinematics", time, scale);
    }

    void update_dynamics(double time, double scale)
    {
        record("update_dynamics", time, scale);
    }

    void update_accelerations(double time, double scale)
    {
        record("update_accelerations", time, scale);
    }

    void compute_internal_forces_and_torques(double time)
    {
        record("compute_internal_forces_and_torques", time);
    }

    void zero_out_external_forces_and_torques(double time)
    {
        record("zero_out_external_forces_and_torques", time);
    }

    int id() const { return id_; }

private:
    void record(std::string method, double time)
    {
        if (log_ == nullptr) return;
        Call call;
        call.method = std::move(method);
        call.id = id_;
        call.time = time;
        log_->add(std::move(call));
    }

    void record(std::string method, double time, double scale)
    {
        if (log_ == nullptr) return;
        Call call;
        call.method = std::move(method);
        call.id = id_;
        call.time = time;
        call.scale = scale;
        call.has_scale = true;
        log_->add(std::move(call));
    }

    CallLog* log_ = nullptr;
    int id_ = 0;
};

class MockSystem
{
public:
    using SubSystemType = MockSubSystem;

    MockSystem(CallLog* log, std::size_t subsystem_count) : log_(log)
    {
        subsystems_.reserve(subsystem_count);
        for (std::size_t idx = 0; idx < subsystem_count; ++idx)
        {
            subsystems_.emplace_back(log, static_cast<int>(idx));
        }
    }

    std::vector<SubSystemType>& final_systems()
    {
        record("final_systems", 0.0);
        return subsystems_;
    }

    void constrain_values(double time) { record("constrain_values", time); }

    void synchronize(double time) { record("synchronize", time); }

    void constrain_rates(double time) { record("constrain_rates", time); }

    void apply_callbacks(double time, std::uint64_t step)
    {
        Call call;
        call.method = "apply_callbacks";
        call.id = kSystemId;
        call.time = time;
        call.step = step;
        call.has_step = true;
        log_->add(std::move(call));
    }

private:
    void record(std::string method, double time)
    {
        if (log_ == nullptr) return;
        Call call;
        call.method = std::move(method);
        call.id = kSystemId;
        call.time = time;
        log_->add(std::move(call));
    }

    CallLog* log_ = nullptr;
    std::vector<SubSystemType> subsystems_;
};

// ---------------------------------------------------------------------------
// Types used only for negative (and surprising-positive) concept checks
// ---------------------------------------------------------------------------
namespace concept_fixtures {

// A minimal type that satisfies IntegrableSystem, used as the SubSystemType of
// the collection fixtures below.
struct GoodSub
{
    void update_kinematics(double, double) {}
    void update_dynamics(double, double) {}
    void update_accelerations(double, double) {}
    void compute_internal_forces_and_torques(double) {}
    void zero_out_external_forces_and_torques(double) {}
};

struct MissingUpdateDynamics
{
    void update_kinematics(double, double) {}
    void update_accelerations(double, double) {}
    void compute_internal_forces_and_torques(double) {}
    void zero_out_external_forces_and_torques(double) {}
};

// Retains the old (misspelled) name from an earlier revision of the header.
struct OldZeroedOutName
{
    void update_kinematics(double, double) {}
    void update_dynamics(double, double) {}
    void update_accelerations(double, double) {}
    void compute_internal_forces_and_torques(double) {}
    void zeroed_out_external_forces_and_torques(double) {}
};

// Retains the singular spelling the body wrapper used before it was renamed to
// match the concept.
struct SingularUpdateAcceleration
{
    void update_kinematics(double, double) {}
    void update_dynamics(double, double) {}
    void update_acceleration(double, double) {}
    void compute_internal_forces_and_torques(double) {}
    void zero_out_external_forces_and_torques(double) {}
};

struct PrivateMethods
{
private:
    void update_kinematics(double, double) {}
    void update_dynamics(double, double) {}
    void update_accelerations(double, double) {}
    void compute_internal_forces_and_torques(double) {}
    void zero_out_external_forces_and_torques(double) {}
};

// Takes int rather than double everywhere. Still satisfies the concept via
// implicit conversion -- see IntegrableSystemConcept.AcceptsTruncatingSignatures.
struct TruncatingSub
{
    void update_kinematics(int, int) {}
    void update_dynamics(int, int) {}
    void update_accelerations(int, int) {}
    void compute_internal_forces_and_torques(int) {}
    void zero_out_external_forces_and_torques(int) {}
};

// Returns a status code rather than void. Satisfies the concept because the
// return-type constraints were removed.
struct NonVoidReturns
{
    int update_kinematics(double, double) { return 0; }
    int update_dynamics(double, double) { return 0; }
    int update_accelerations(double, double) { return 0; }
    int compute_internal_forces_and_torques(double) { return 0; }
    int zero_out_external_forces_and_torques(double) { return 0; }
};

// --- Collection fixtures ---------------------------------------------------

struct GoodCollection
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

struct NoNestedType
{
    std::vector<GoodSub>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<GoodSub> subs;
};

struct NonIntegrableSub
{
    using SubSystemType = MissingUpdateDynamics;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

// The original bug: returning the vector by value means the solver mutates a
// temporary copy and every integration result is discarded.
struct ByValueFinalSystems
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType> final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

struct ConstRefFinalSystems
{
    using SubSystemType = GoodSub;
    const std::vector<SubSystemType>& final_systems() const { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

// Perfectly safe and mutable, but rejected because the concept hardcodes
// std::vector. Documents the storage coupling.
struct SpanFinalSystems
{
    using SubSystemType = GoodSub;
    std::span<SubSystemType> final_systems() { return std::span{subs}; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

struct MissingConstrainRates
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

// Retains the old (misspelled) name from the first revision of the header.
struct ContrainRatesTypo
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void contrain_rates(double) {}
    void apply_callbacks(double, std::uint64_t) {}
    std::vector<SubSystemType> subs;
};

struct WrongCallbackArity
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(double) {}
    std::vector<SubSystemType> subs;
};

// Narrows the step counter to 32 bits and truncates the time. Satisfies the
// concept anyway -- see SystemCollectionConcept.AcceptsNarrowingCallback.
struct NarrowingCallback
{
    using SubSystemType = GoodSub;
    std::vector<SubSystemType>& final_systems() { return subs; }
    void constrain_values(double) {}
    void synchronize(double) {}
    void constrain_rates(double) {}
    void apply_callbacks(int, unsigned int) {}
    std::vector<SubSystemType> subs;
};

}  // namespace concept_fixtures

// ---------------------------------------------------------------------------
// Concept tests
//
// Concept-ids are constant expressions, but evaluating them inside EXPECT_*
// turns a mismatch into a reported test failure rather than a build break.
// ---------------------------------------------------------------------------

TEST(IntegrableSystemConcept, AcceptsMockSubSystem)
{
    EXPECT_TRUE(IntegrableSystem<MockSubSystem>);
    EXPECT_TRUE(IntegrableSystem<concept_fixtures::GoodSub>);
}

TEST(IntegrableSystemConcept, RejectsMissingMethod)
{
    EXPECT_FALSE(IntegrableSystem<concept_fixtures::MissingUpdateDynamics>);
}

TEST(IntegrableSystemConcept, RejectsPreviousZeroedOutSpelling)
{
    EXPECT_FALSE(IntegrableSystem<concept_fixtures::OldZeroedOutName>);
}

// The body wrapper once spelled this in the singular, which made it fail this
// concept and prevented Solver from instantiating at all.
TEST(IntegrableSystemConcept, RejectsSingularUpdateAcceleration)
{
    EXPECT_FALSE(IntegrableSystem<concept_fixtures::SingularUpdateAcceleration>);
}

TEST(IntegrableSystemConcept, RejectsInaccessibleMethods)
{
    EXPECT_FALSE(IntegrableSystem<concept_fixtures::PrivateMethods>);
}

TEST(IntegrableSystemConcept, RejectsUnrelatedTypes)
{
    EXPECT_FALSE(IntegrableSystem<int>);
    EXPECT_FALSE(IntegrableSystem<std::vector<double>>);
}

// Characterization, not endorsement: implicit conversion means an
// implementation taking `int` passes the concept while silently truncating both
// the simulation time and the 0.5*dt scale (to 0 for any dt < 2).
TEST(IntegrableSystemConcept, AcceptsTruncatingSignatures)
{
    EXPECT_TRUE(IntegrableSystem<concept_fixtures::TruncatingSub>);
}

// Characterization: return types are no longer constrained, so a method
// returning a status code satisfies the concept and the solver discards it.
TEST(IntegrableSystemConcept, AcceptsNonVoidReturns)
{
    EXPECT_TRUE(IntegrableSystem<concept_fixtures::NonVoidReturns>);
}

TEST(SystemCollectionConcept, AcceptsMockSystem)
{
    EXPECT_TRUE(SystemCollection<MockSystem>);
    EXPECT_TRUE(SystemCollection<concept_fixtures::GoodCollection>);
}

TEST(SystemCollectionConcept, RejectsMissingNestedType)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::NoNestedType>);
}

TEST(SystemCollectionConcept, RejectsNonIntegrableSubSystem)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::NonIntegrableSub>);
}

TEST(SystemCollectionConcept, RejectsFinalSystemsByValue)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::ByValueFinalSystems>);
}

TEST(SystemCollectionConcept, RejectsFinalSystemsByConstReference)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::ConstRefFinalSystems>);
}

// Documents the coupling to std::vector: a span over the same mutable storage
// is rejected even though it would work correctly.
TEST(SystemCollectionConcept, RejectsSpanFinalSystems)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::SpanFinalSystems>);
}

TEST(SystemCollectionConcept, RejectsMissingConstrainRates)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::MissingConstrainRates>);
    EXPECT_FALSE(SystemCollection<concept_fixtures::ContrainRatesTypo>);
}

TEST(SystemCollectionConcept, RejectsWrongCallbackArity)
{
    EXPECT_FALSE(SystemCollection<concept_fixtures::WrongCallbackArity>);
}

// Characterization: the concept does not pin the callback parameter types, so a
// collection that takes (int, unsigned int) is accepted and will truncate the
// time and wrap the step counter at 2^32.
TEST(SystemCollectionConcept, AcceptsNarrowingCallback)
{
    EXPECT_TRUE(SystemCollection<concept_fixtures::NarrowingCallback>);
}

// ---------------------------------------------------------------------------
// Solver fixture
// ---------------------------------------------------------------------------

class SolverTest : public ::testing::Test
{
protected:
    static constexpr double kDt = 0.25;  // exact in binary floating point
    static constexpr std::size_t kSubSystemCount = 2;

    CallLog log;
    MockSystem system{&log, kSubSystemCount};
    Solver<MockSystem> solver{kDt};
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(SolverConstruction, StoresDtAndZeroesCounters)
{
    Solver<MockSystem> solver{0.5};
    EXPECT_EQ(0u, SolverTestPeer<MockSystem>::current_step(solver));
    EXPECT_EQ(0.0, SolverTestPeer<MockSystem>::initial_time(solver));
    EXPECT_EQ(0.5, SolverTestPeer<MockSystem>::dt(solver));
}

#if GTEST_HAS_DEATH_TEST

TEST(SolverConstructionDeathTest, RejectsNonPositiveDt)
{
    EXPECT_ASSERT_ABORT(Solver<MockSystem>{0.0});
    EXPECT_ASSERT_ABORT(Solver<MockSystem>{-1.0});
}

TEST(SolverConstructionDeathTest, RejectsNaNDt)
{
    // NaN > 0.0 is false, so this aborts -- but note it aborts because every
    // comparison against NaN is false, not because the constructor recognises
    // NaN as invalid. Same outcome, different reason.
    EXPECT_ASSERT_ABORT(
        Solver<MockSystem>{std::numeric_limits<double>::quiet_NaN()});
}

#endif  // GTEST_HAS_DEATH_TEST

// ---------------------------------------------------------------------------
// Private methods, reached through the friend peer
// ---------------------------------------------------------------------------

TEST(SolverPrivateMethods, KinematicStepHalvesDt)
{
    CallLog log;
    MockSubSystem sub{&log, 0};
    Solver<MockSystem> solver{0.4};

    SolverTestPeer<MockSystem>::kinematic_step(solver, sub, 3.0);

    ASSERT_EQ(1u, log.calls().size());
    const Call& call = log.calls().front();
    EXPECT_EQ("update_kinematics", call.method);
    EXPECT_DOUBLE_EQ(3.0, call.time);
    ASSERT_TRUE(call.has_scale);
    EXPECT_DOUBLE_EQ(0.2, call.scale);
}

TEST(SolverPrivateMethods, DynamicsStepOrdersAccelerationsBeforeDynamics)
{
    CallLog log;
    MockSubSystem sub{&log, 0};
    Solver<MockSystem> solver{0.4};

    SolverTestPeer<MockSystem>::dynamics_step(solver, sub, 3.0);

    const std::vector<std::string> expected = {"0.update_accelerations",
                                               "0.update_dynamics"};
    EXPECT_EQ(expected, log.sequence());

    for (const Call& call : log.calls())
    {
        EXPECT_DOUBLE_EQ(3.0, call.time);
        ASSERT_TRUE(call.has_scale);
        EXPECT_DOUBLE_EQ(0.4, call.scale) << "in " << call.method;
    }
}

// ---------------------------------------------------------------------------
// step(): ordering
// ---------------------------------------------------------------------------

TEST_F(SolverTest, StepCallsMethodsInStaggeredOrder)
{
    solver.step(system, 0.0);

    const std::vector<std::string> expected = {
        "0.update_kinematics",
        "1.update_kinematics",
        "S.constrain_values",
        "0.compute_internal_forces_and_torques",
        "1.compute_internal_forces_and_torques",
        "S.synchronize",
        "0.update_accelerations",
        "0.update_dynamics",
        "1.update_accelerations",
        "1.update_dynamics",
        "S.constrain_rates",
        "0.update_kinematics",
        "1.update_kinematics",
        "S.constrain_values",
        "S.apply_callbacks",
        "0.zero_out_external_forces_and_torques",
        "1.zero_out_external_forces_and_torques",
    };
    EXPECT_EQ(expected, log.sequence());
}

TEST_F(SolverTest, StepVisitsEverySubSystemInEveryPhase)
{
    MockSystem wide_system{&log, 5};
    solver.step(wide_system, 0.0);

    EXPECT_EQ(10u, log.count("update_kinematics"));  // two half-steps each
    EXPECT_EQ(5u, log.count("compute_internal_forces_and_torques"));
    EXPECT_EQ(5u, log.count("update_accelerations"));
    EXPECT_EQ(5u, log.count("update_dynamics"));
    EXPECT_EQ(5u, log.count("zero_out_external_forces_and_torques"));
}

// Documents the redundant re-fetching noted in review: cheap for a member
// vector, not cheap if an implementation ever builds the container on demand.
TEST_F(SolverTest, StepRefetchesFinalSystemsOncePerPhase)
{
    solver.step(system, 0.0);
    EXPECT_EQ(5u, log.count("final_systems"));
}

TEST_F(SolverTest, StepOnEmptyCollectionStillRunsSystemLevelHooks)
{
    MockSystem empty_system{&log, 0};

    const double result = solver.step(empty_system, 1.0);

    const std::vector<std::string> expected = {
        "S.constrain_values", "S.synchronize", "S.constrain_rates",
        "S.constrain_values", "S.apply_callbacks",
    };
    EXPECT_EQ(expected, log.sequence());
    EXPECT_EQ(1.0 + kDt, result);
}

// ---------------------------------------------------------------------------
// step(): stage times and scales
// ---------------------------------------------------------------------------

TEST_F(SolverTest, StepPassesCorrectStageTimesAndScales)
{
    constexpr double kStart = 2.0;
    const double mid = kStart + 0.5 * kDt;
    const double end = mid + 0.5 * kDt;

    solver.step(system, kStart);

    // First kinematic half-step: un-advanced time, half dt.
    for (std::size_t idx = 0; idx < kSubSystemCount; ++idx)
    {
        const Call& call = log.nth("update_kinematics", idx);
        EXPECT_DOUBLE_EQ(kStart, call.time);
        EXPECT_DOUBLE_EQ(0.5 * kDt, call.scale);
    }

    // Mid-step phases.
    EXPECT_DOUBLE_EQ(mid, log.nth("constrain_values", 0).time);
    EXPECT_DOUBLE_EQ(mid, log.nth("compute_internal_forces_and_torques", 0).time);
    EXPECT_DOUBLE_EQ(mid, log.nth("synchronize", 0).time);
    EXPECT_DOUBLE_EQ(mid, log.nth("constrain_rates", 0).time);

    // Dynamics uses the full dt, not a half.
    for (std::size_t idx = 0; idx < kSubSystemCount; ++idx)
    {
        EXPECT_DOUBLE_EQ(mid, log.nth("update_accelerations", idx).time);
        EXPECT_DOUBLE_EQ(kDt, log.nth("update_accelerations", idx).scale);
        EXPECT_DOUBLE_EQ(mid, log.nth("update_dynamics", idx).time);
        EXPECT_DOUBLE_EQ(kDt, log.nth("update_dynamics", idx).scale);
    }

    // Second kinematic half-step: mid time, half dt.
    for (std::size_t idx = 0; idx < kSubSystemCount; ++idx)
    {
        const Call& call = log.nth("update_kinematics", kSubSystemCount + idx);
        EXPECT_DOUBLE_EQ(mid, call.time);
        EXPECT_DOUBLE_EQ(0.5 * kDt, call.scale);
    }

    // End-of-step phases.
    EXPECT_DOUBLE_EQ(end, log.nth("constrain_values", 1).time);
    EXPECT_DOUBLE_EQ(end, log.nth("apply_callbacks", 0).time);
    EXPECT_DOUBLE_EQ(end, log.nth("zero_out_external_forces_and_torques", 0).time);
}

// The value handed to callbacks is the twice-accumulated `sim_time`; the value
// returned to the caller is recomputed as initial_time + n*dt. They agree
// exactly for kDt = 0.25 but are not guaranteed to for arbitrary dt.
TEST_F(SolverTest, CallbackTimeAgreesWithReturnedTime)
{
    const double result = solver.step(system, 2.0);
    EXPECT_NEAR(result, log.nth("apply_callbacks", 0).time, 1e-12);
}

// ---------------------------------------------------------------------------
// step(): counter and returned time
// ---------------------------------------------------------------------------

TEST_F(SolverTest, FirstCallbackReceivesStepOne)
{
    solver.step(system, 0.0);
    EXPECT_EQ(1u, log.nth("apply_callbacks", 0).step);
}

TEST_F(SolverTest, StepCounterIncrementsMonotonically)
{
    double time = 0.0;
    for (std::uint64_t idx = 0; idx < 4; ++idx)
    {
        time = solver.step(system, time);
        EXPECT_EQ(idx + 1, SolverTestPeer<MockSystem>::current_step(solver));
    }

    const std::vector<Call> callbacks = log.filter("apply_callbacks");
    ASSERT_EQ(4u, callbacks.size());
    for (std::size_t idx = 0; idx < callbacks.size(); ++idx)
    {
        EXPECT_EQ(idx + 1, callbacks[idx].step);
    }
}

TEST_F(SolverTest, ReturnedTimeDoesNotDriftOverManySteps)
{
    constexpr std::uint64_t kSteps = 1000;
    double time = 0.0;
    for (std::uint64_t idx = 0; idx < kSteps; ++idx)
    {
        time = solver.step(system, time);
    }
    // Exact equality on purpose: the return is recomputed from the origin, so
    // this must not degrade into an accumulation.
    EXPECT_EQ(static_cast<double>(kSteps) * kDt, time);
}

TEST_F(SolverTest, InitialTimeIsLatchedOnFirstStepOnly)
{
    solver.step(system, 5.0);
    EXPECT_EQ(5.0, SolverTestPeer<MockSystem>::initial_time(solver));

    // A caller passing an inconsistent time is silently ignored for the return
    // value -- but the value IS used as the first stage time, so the sub-step
    // and the returned clock disagree. See notes.
    log.clear();
    const double result = solver.step(system, 999.0);

    EXPECT_EQ(5.0, SolverTestPeer<MockSystem>::initial_time(solver));
    EXPECT_EQ(5.0 + 2.0 * kDt, result);
    EXPECT_DOUBLE_EQ(999.0, log.nth("update_kinematics", 0).time);
}

// ---------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------

TEST_F(SolverTest, ResetClearsCounterAndOrigin)
{
    solver.step(system, 5.0);
    solver.step(system, 5.0 + kDt);
    ASSERT_EQ(2u, SolverTestPeer<MockSystem>::current_step(solver));

    solver.reset();

    EXPECT_EQ(0u, SolverTestPeer<MockSystem>::current_step(solver));
    EXPECT_EQ(0.0, SolverTestPeer<MockSystem>::initial_time(solver));
}

TEST_F(SolverTest, ResetAllowsRelatchingANewOrigin)
{
    solver.step(system, 0.0);
    solver.reset();

    const double result = solver.step(system, 100.0);

    EXPECT_EQ(100.0, SolverTestPeer<MockSystem>::initial_time(solver));
    EXPECT_EQ(100.0 + kDt, result);
}

// ---------------------------------------------------------------------------
// full_solve(): the initial pass
//
// Before integrating anything, full_solve pins the configuration and the rates
// and fires the callbacks at step zero. This mirrors the reference
// implementation, whose finalize() does exactly constrain_values(0),
// constrain_rates(0), apply_callbacks(0, 0) -- so the initial state is recorded
// rather than the first frame on disk being one interval into the run.
// ---------------------------------------------------------------------------

TEST_F(SolverTest, FullSolveConstrainsAndRecordsBeforeIntegrating)
{
    solver.full_solve(system, 0.0, kDt);  // exactly one step

    const std::vector<std::string> sequence = log.sequence();
    ASSERT_GE(sequence.size(), 3u);
    // The run opens with the initial pass, before any subsystem is touched.
    EXPECT_EQ("S.constrain_values", sequence[0]);
    EXPECT_EQ("S.constrain_rates", sequence[1]);
    EXPECT_EQ("S.apply_callbacks", sequence[2]);
}

TEST_F(SolverTest, TheInitialCallbackReceivesStepZeroAndTheStartTime)
{
    solver.full_solve(system, 5.0, 5.0 + kDt);

    const Call& initial = log.nth("apply_callbacks", 0);
    EXPECT_EQ(0u, initial.step);
    EXPECT_DOUBLE_EQ(5.0, initial.time);
}

// The initial pass records the state as it stands; it must not advance it.
TEST_F(SolverTest, TheInitialPassTouchesNoSubSystem)
{
    MockSystem lone_system{&log, 1};
    Solver<MockSystem> tiny{kDt};

    // Stop before the loop by inspecting the log up to the first callback.
    tiny.full_solve(lone_system, 0.0, kDt);

    const std::vector<std::string> sequence = log.sequence();
    const auto first_callback =
        std::find(sequence.begin(), sequence.end(), "S.apply_callbacks");
    ASSERT_NE(first_callback, sequence.end());

    for (auto it = sequence.begin(); it != first_callback; ++it)
    {
        EXPECT_EQ('S', it->front()) << "subsystem work before the initial frame: " << *it;
    }
}

// synchronize is not part of the initial pass, so no loads are accumulated
// before the first frame is recorded.
TEST_F(SolverTest, TheInitialPassDoesNotSynchronize)
{
    solver.full_solve(system, 0.0, kDt);

    const std::vector<std::string> sequence = log.sequence();
    const auto first_callback =
        std::find(sequence.begin(), sequence.end(), "S.apply_callbacks");
    const auto first_sync =
        std::find(sequence.begin(), sequence.end(), "S.synchronize");
    ASSERT_NE(first_callback, sequence.end());
    ASSERT_NE(first_sync, sequence.end());
    EXPECT_LT(first_callback - sequence.begin(), first_sync - sequence.begin());
}

// step() alone is unchanged: the initial pass belongs to full_solve, so a
// caller driving the solver step by step still gets no frame at step zero.
TEST_F(SolverTest, StepAloneDoesNotRunTheInitialPass)
{
    solver.step(system, 0.0);

    EXPECT_EQ(1u, log.count("apply_callbacks"));
    EXPECT_EQ(1u, log.nth("apply_callbacks", 0).step);
    EXPECT_EQ("0.update_kinematics", log.sequence().front());
}

TEST_F(SolverTest, TheInitialPassRunsOncePerFullSolve)
{
    solver.full_solve(system, 0.0, 1.0);

    const std::vector<Call> callbacks = log.filter("apply_callbacks");
    ASSERT_EQ(5u, callbacks.size());
    // Exactly one frame at step zero, then one per step.
    EXPECT_EQ(0u, callbacks.front().step);
    for (std::size_t idx = 1; idx < callbacks.size(); ++idx)
    {
        EXPECT_EQ(idx, callbacks[idx].step);
    }
}

// The initial pass does not disturb the step counter or the returned clock.
TEST_F(SolverTest, TheInitialPassLeavesTheCounterAtZero)
{
    solver.full_solve(system, 0.0, kDt);

    // One step ran after the initial pass.
    EXPECT_EQ(1u, SolverTestPeer<MockSystem>::current_step(solver));
    EXPECT_EQ(2u, log.count("apply_callbacks"));
}

// ---------------------------------------------------------------------------
// full_solve()
// ---------------------------------------------------------------------------

TEST_F(SolverTest, FullSolveRunsExpectedNumberOfSteps)
{
    const double result = solver.full_solve(system, 0.0, 1.0);

    // Four steps, plus the frame recorded before the first one.
    EXPECT_EQ(5u, log.count("apply_callbacks"));
    EXPECT_EQ(1.0, result);
}

TEST_F(SolverTest, FullSolveHonoursNonZeroStart)
{
    const double result = solver.full_solve(system, 5.0, 6.0);

    EXPECT_EQ(5u, log.count("apply_callbacks"));
    EXPECT_EQ(6.0, result);

    // The origin is no longer observable after the run (reset() clears it), so
    // verify it through the stage times the subsystems actually received.
    EXPECT_DOUBLE_EQ(5.0, log.nth("update_kinematics", 0).time);
    EXPECT_DOUBLE_EQ(5.0, log.nth("apply_callbacks", 0).time);
    EXPECT_DOUBLE_EQ(6.0, log.nth("apply_callbacks", 4).time);
}

TEST_F(SolverTest, FullSolveAcceptsExactlyOneStep)
{
    const double result = solver.full_solve(system, 0.0, kDt);

    // The initial frame plus the one step.
    EXPECT_EQ(2u, log.count("apply_callbacks"));
    EXPECT_EQ(kDt, result);
}

// Characterization: the interval is silently rounded to the nearest whole
// number of steps, and std::round breaks ties away from zero, so a half-step
// remainder overshoots `end`.
TEST(SolverFullSolve, SilentlyOvershootsNonMultipleInterval)
{
    CallLog log;
    MockSystem system{&log, 1};
    Solver<MockSystem> solver{0.5};

    const double result = solver.full_solve(system, 0.0, 1.25);

    // Three steps, plus the initial frame.
    EXPECT_EQ(4u, log.count("apply_callbacks"));
    EXPECT_EQ(1.5, result) << "integrated past the requested end time";
}

#if GTEST_HAS_DEATH_TEST

using SolverDeathTest = SolverTest;

TEST_F(SolverDeathTest, FullSolveRejectsIntervalShorterThanDt)
{
    EXPECT_ASSERT_ABORT(solver.full_solve(system, 0.0, 0.5 * kDt));
}

TEST_F(SolverDeathTest, FullSolveRejectsReversedInterval)
{
    EXPECT_ASSERT_ABORT(solver.full_solve(system, 1.0, 0.0));
}

TEST_F(SolverDeathTest, FullSolveRejectsNaNInterval)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_ASSERT_ABORT(solver.full_solve(system, 0.0, nan));
    EXPECT_ASSERT_ABORT(solver.full_solve(system, nan, 1.0));
}

// The interval check runs before the initial pass, so a rejected run records
// nothing at all rather than leaving a stray frame at step zero.
TEST_F(SolverDeathTest, ARejectedRunRecordsNothing)
{
    EXPECT_ASSERT_ABORT(solver.full_solve(system, 0.0, 0.5 * kDt));
}

#endif  // GTEST_HAS_DEATH_TEST

// ---------------------------------------------------------------------------
// full_solve(): state lifecycle
// ---------------------------------------------------------------------------

TEST_F(SolverTest, FullSolveRestartsTheClockOnSecondCall)
{
    const double first = solver.full_solve(system, 0.0, 1.0);
    ASSERT_EQ(1.0, first);

    const double second = solver.full_solve(system, 10.0, 11.0);

    EXPECT_EQ(11.0, second);
}

TEST_F(SolverTest, SecondFullSolveUsesCorrectStageTimes)
{
    solver.full_solve(system, 0.0, 1.0);
    log.clear();

    solver.full_solve(system, 10.0, 11.0);

    // First sub-step of the run starts at `start`...
    EXPECT_DOUBLE_EQ(10.0, log.nth("update_kinematics", 0).time);
    // ...and the first sub-step of the following step continues from there.
    EXPECT_DOUBLE_EQ(10.0 + kDt,
                     log.nth("update_kinematics", 2 * kSubSystemCount).time);

    const std::vector<Call> callbacks = log.filter("apply_callbacks");
    ASSERT_EQ(5u, callbacks.size());
    // The initial frame sits at the start time, then one per step after it.
    for (std::size_t idx = 0; idx < callbacks.size(); ++idx)
    {
        EXPECT_DOUBLE_EQ(10.0 + static_cast<double>(idx) * kDt, callbacks[idx].time);
    }
}

// Characterization, not endorsement: because reset() zeroes the counter at the
// end of every run, callbacks in a segmented simulation see 0..N repeatedly
// rather than a monotonically increasing global step index. A callback that
// writes output keyed on the step number will overwrite the first segment's
// results with the second's.
TEST_F(SolverTest, FullSolveRestartsCallbackStepIndexingEachRun)
{
    solver.full_solve(system, 0.0, 1.0);
    solver.full_solve(system, 1.0, 2.0);

    const std::vector<Call> callbacks = log.filter("apply_callbacks");
    ASSERT_EQ(10u, callbacks.size());

    const std::vector<std::uint64_t> expected = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    std::vector<std::uint64_t> actual;
    for (const Call& call : callbacks) actual.push_back(call.step);
    EXPECT_EQ(expected, actual);
}

// Each run re-records its own starting state, so a segmented simulation has a
// frame at the seam from both sides.
TEST_F(SolverTest, EachRunRecordsItsOwnStartingState)
{
    solver.full_solve(system, 0.0, 1.0);
    solver.full_solve(system, 1.0, 2.0);

    const std::vector<Call> callbacks = log.filter("apply_callbacks");
    ASSERT_EQ(10u, callbacks.size());

    // Last frame of the first run and first frame of the second are the same
    // instant, recorded twice.
    EXPECT_DOUBLE_EQ(1.0, callbacks[4].time);
    EXPECT_DOUBLE_EQ(1.0, callbacks[5].time);
    EXPECT_EQ(4u, callbacks[4].step);
    EXPECT_EQ(0u, callbacks[5].step);
}
}  // namespace cosserat::simulation
