#include "math/functions.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace cosserat::math {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// LinearRamp: accessors and value semantics
// ---------------------------------------------------------------------------

TEST(LinearRamp, AccessorsReturnConstructorArguments)
{
    const LinearRamp ramp(1.5, 2.5, 7.0);

    EXPECT_DOUBLE_EQ(ramp.onset_time(), 1.5);
    EXPECT_DOUBLE_EQ(ramp.ramp_time(), 2.5);
    ASSERT_TRUE(ramp.offset_time().has_value());
    EXPECT_DOUBLE_EQ(*ramp.offset_time(), 7.0);
}

TEST(LinearRamp, OffsetAccessorIsEmptyWhenUnset)
{
    const LinearRamp ramp(0.0, 1.0, std::nullopt);

    EXPECT_FALSE(ramp.offset_time().has_value());
    EXPECT_EQ(ramp.offset_time(), std::nullopt);
}

TEST(LinearRamp, IsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<LinearRamp>);
    static_assert(std::is_copy_assignable_v<LinearRamp>);
    static_assert(std::is_move_constructible_v<LinearRamp>);
    static_assert(std::is_move_assignable_v<LinearRamp>);

    LinearRamp a(0.0, 1.0, std::nullopt);
    const LinearRamp b(5.0, 2.0, 10.0);

    a = b;

    EXPECT_DOUBLE_EQ(a.onset_time(), 5.0);
    EXPECT_DOUBLE_EQ(a.ramp_time(), 2.0);
    ASSERT_TRUE(a.offset_time().has_value());
    EXPECT_DOUBLE_EQ(*a.offset_time(), 10.0);
    EXPECT_DOUBLE_EQ(a(6.0), b(6.0));
}

TEST(LinearRamp, WorksInStandardContainers)
{
    std::vector<LinearRamp> ramps;
    ramps.emplace_back(3.0, 1.0, std::nullopt);
    ramps.emplace_back(1.0, 1.0, std::nullopt);
    ramps.emplace_back(2.0, 1.0, std::nullopt);

    std::sort(ramps.begin(), ramps.end(), [](const LinearRamp& l, const LinearRamp& r) {
        return l.onset_time() < r.onset_time();
    });
    EXPECT_DOUBLE_EQ(ramps.front().onset_time(), 1.0);

    ramps.erase(ramps.begin());
    ASSERT_EQ(ramps.size(), 2u);
    EXPECT_DOUBLE_EQ(ramps.front().onset_time(), 2.0);
}

TEST(LinearRamp, CopyEvaluatesIdenticallyToOriginal)
{
    const LinearRamp original(1.0, 2.0, 5.0);
    const LinearRamp copy = original;

    for (double t : {0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 5.5, 100.0})
    {
        EXPECT_DOUBLE_EQ(copy(t), original(t)) << "at t = " << t;
    }
}

// ---------------------------------------------------------------------------
// LinearRamp: valid evaluation
// ---------------------------------------------------------------------------

TEST(LinearRamp, ZeroBeforeAndAtOnset)
{
    const LinearRamp ramp(1.0, 2.0, std::nullopt);

    EXPECT_EQ(ramp(-100.0), 0.0);
    EXPECT_EQ(ramp(0.0), 0.0);
    EXPECT_EQ(ramp(1.0), 0.0);  // closed at onset
    EXPECT_EQ(ramp(std::nextafter(1.0, -kInf)), 0.0);
}

TEST(LinearRamp, RisesLinearlyAcrossRamp)
{
    const LinearRamp ramp(1.0, 2.0, std::nullopt);

    EXPECT_DOUBLE_EQ(ramp(1.5), 0.25);
    EXPECT_DOUBLE_EQ(ramp(2.0), 0.50);
    EXPECT_DOUBLE_EQ(ramp(2.5), 0.75);
}

TEST(LinearRamp, SaturatesAtOneFromRampEnd)
{
    const LinearRamp ramp(1.0, 2.0, std::nullopt);

    EXPECT_DOUBLE_EQ(ramp(3.0), 1.0);  // exactly onset + ramp
    EXPECT_EQ(ramp(3.5), 1.0);
    EXPECT_EQ(ramp(1e6), 1.0);
}

TEST(LinearRamp, ShortRampStillClampsToOne)
{
    const LinearRamp ramp(0.0, 1e-9, std::nullopt);

    EXPECT_EQ(ramp(1.0), 1.0);
    EXPECT_DOUBLE_EQ(ramp(5e-10), 0.5);
}

TEST(LinearRamp, NegativeOnsetIsAccepted)
{
    const LinearRamp ramp(-4.0, 2.0, std::nullopt);

    EXPECT_EQ(ramp(-5.0), 0.0);
    EXPECT_DOUBLE_EQ(ramp(-3.0), 0.5);
    EXPECT_EQ(ramp(0.0), 1.0);
}

TEST(LinearRamp, NonDecreasingUpToRampEnd)
{
    const LinearRamp ramp(0.5, 3.0, std::nullopt);

    double previous = -1.0;
    for (int i = 0; i <= 200; ++i)
    {
        const double value = ramp(i * 0.025);
        EXPECT_GE(value, previous) << "at step " << i;
        previous = value;
    }
}

// ---------------------------------------------------------------------------
// LinearRamp: offset behavior
// ---------------------------------------------------------------------------

TEST(LinearRamp, ActiveUpToAndIncludingOffset)
{
    const LinearRamp ramp(1.0, 2.0, 5.0);

    EXPECT_EQ(ramp(4.9), 1.0);
    EXPECT_EQ(ramp(5.0), 1.0);  // closed at offset
}

// Documents the intentional step at offset: 1.0 -> 0.0 with no ramp down.
TEST(LinearRamp, DropsToZeroImmediatelyPastOffset)
{
    const LinearRamp ramp(1.0, 2.0, 5.0);

    EXPECT_EQ(ramp(std::nextafter(5.0, kInf)), 0.0);
    EXPECT_EQ(ramp(5.1), 0.0);
    EXPECT_EQ(ramp(1e6), 0.0);
}

TEST(LinearRamp, OffsetExactlyAtRampEndPeaksThenStops)
{
    const LinearRamp ramp(1.0, 2.0, 3.0);  // offset == onset + ramp, allowed

    EXPECT_DOUBLE_EQ(ramp(2.0), 0.5);
    EXPECT_DOUBLE_EQ(ramp(3.0), 1.0);
    EXPECT_EQ(ramp(std::nextafter(3.0, kInf)), 0.0);
}

TEST(LinearRamp, NoOffsetNeverDeactivates)
{
    const LinearRamp ramp(0.0, 1.0, std::nullopt);

    EXPECT_EQ(ramp(1e12), 1.0);
}

// ---------------------------------------------------------------------------
// LinearRamp: construction and argument validation
// ---------------------------------------------------------------------------

TEST(LinearRampDeathTest, RejectsNonFiniteOnset)
{
    EXPECT_ASSERT_FAILURE(LinearRamp(kNaN, 1.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(LinearRamp(kInf, 1.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(LinearRamp(-kInf, 1.0, std::nullopt));
}

TEST(LinearRampDeathTest, RejectsNonPositiveOrNonFiniteRamp)
{
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, 0.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, -1.0, std::nullopt));
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, kNaN, std::nullopt));
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, kInf, std::nullopt));
}

TEST(LinearRampDeathTest, RejectsNonFiniteOffset)
{
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, 1.0, kNaN));
    EXPECT_ASSERT_FAILURE(LinearRamp(0.0, 1.0, kInf));
}

TEST(LinearRampDeathTest, RejectsOffsetBeforeRampCompletes)
{
    EXPECT_ASSERT_FAILURE(LinearRamp(1.0, 2.0, 2.999));
    EXPECT_ASSERT_FAILURE(LinearRamp(1.0, 2.0, 0.0));
    EXPECT_ASSERT_FAILURE(LinearRamp(1.0, 2.0, std::nextafter(3.0, -kInf)));
}

// The nullopt path must skip the offset checks entirely.
TEST(LinearRamp, NoOffsetSkipsOffsetValidation)
{
    EXPECT_NO_THROW({ LinearRamp(1e300, 1e300, std::nullopt); });
}

TEST(LinearRampDeathTest, RejectsNonFiniteTimeArgument)
{
    const LinearRamp ramp(1.0, 2.0, std::nullopt);

    EXPECT_ASSERT_FAILURE(ramp(kNaN));
    EXPECT_ASSERT_FAILURE(ramp(kInf));
    EXPECT_ASSERT_FAILURE(ramp(-kInf));
}

// ---------------------------------------------------------------------------
// to_normalized_coords
// ---------------------------------------------------------------------------

TEST(ToNormalizedCoords, UniformLengthsGiveEvenSpacing)
{
    Eigen::VectorXd lengths(4);
    lengths << 2.0, 2.0, 2.0, 2.0;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    ASSERT_EQ(coords.size(), 4);
    EXPECT_DOUBLE_EQ(coords(0), 0.25);
    EXPECT_DOUBLE_EQ(coords(1), 0.50);
    EXPECT_DOUBLE_EQ(coords(2), 0.75);
    EXPECT_DOUBLE_EQ(coords(3), 1.00);
}

TEST(ToNormalizedCoords, NonUniformLengthsAccumulate)
{
    Eigen::VectorXd lengths(3);
    lengths << 1.0, 3.0, 4.0;  // total 8

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_DOUBLE_EQ(coords(0), 0.125);
    EXPECT_DOUBLE_EQ(coords(1), 0.500);
    EXPECT_DOUBLE_EQ(coords(2), 1.000);
}

TEST(ToNormalizedCoords, SingleElement)
{
    Eigen::VectorXd lengths(1);
    lengths << 7.3;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    ASSERT_EQ(coords.size(), 1);
    EXPECT_DOUBLE_EQ(coords(0), 1.0);
}

// Returns element endpoints (N entries for N elements), not nodes (N+1).
TEST(ToNormalizedCoords, ReturnsElementEndpointsNotNodes)
{
    Eigen::VectorXd lengths(5);
    lengths << 1.0, 1.0, 1.0, 1.0, 1.0;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_EQ(coords.size(), lengths.size());
    EXPECT_GT(coords(0), 0.0);  // no leading zero
}

TEST(ToNormalizedCoords, StrictlyIncreasing)
{
    Eigen::VectorXd lengths(6);
    lengths << 0.5, 2.0, 0.1, 9.0, 0.25, 3.0;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    for (Eigen::Index i = 1; i < coords.size(); ++i)
    {
        EXPECT_GT(coords(i), coords(i - 1)) << "at index " << i;
    }
}

TEST(ToNormalizedCoords, FinalEntryIsOne)
{
    Eigen::VectorXd lengths(4);
    lengths << 1.0, 2.5, 0.75, 3.25;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_DOUBLE_EQ(coords(coords.size() - 1), 1.0);
}

// (1.0 / total) * total is not always exactly 1.0; total == 49 is one case.
// Use DOUBLE_EQ, not EQ, on the final entry.
TEST(ToNormalizedCoords, FinalEntryMayNotBeBitExactOne)
{
    Eigen::VectorXd lengths(3);
    lengths << 10.0, 14.0, 25.0;  // total 49

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_DOUBLE_EQ(coords(2), 1.0);
    EXPECT_LE(coords(2), 1.0);
}

TEST(ToNormalizedCoords, InvariantUnderUniformScaling)
{
    Eigen::VectorXd lengths(4);
    lengths << 1.0, 2.5, 0.75, 3.25;

    const Eigen::VectorXd base = to_normalized_coords(lengths);
    const Eigen::VectorXd scaled = to_normalized_coords(Eigen::VectorXd(lengths * 1000.0));

    EXPECT_LT((base - scaled).norm(), 1e-14);
}

TEST(ToNormalizedCoords, DoesNotModifyInput)
{
    Eigen::VectorXd lengths(3);
    lengths << 1.0, 2.0, 3.0;
    const Eigen::VectorXd before = lengths;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_EQ((lengths - before).norm(), 0.0);
}

TEST(ToNormalizedCoords, HandlesWidelyVaryingMagnitudes)
{
    Eigen::VectorXd lengths(3);
    lengths << 1e-8, 1.0, 1e8;

    const Eigen::VectorXd coords = to_normalized_coords(lengths);

    EXPECT_GT(coords(0), 0.0);
    EXPECT_DOUBLE_EQ(coords(2), 1.0);
}

TEST(ToNormalizedCoordsDeathTest, RejectsEmptyInput)
{
    EXPECT_ASSERT_FAILURE(to_normalized_coords(Eigen::VectorXd(0)));
}

TEST(ToNormalizedCoordsDeathTest, RejectsNonPositiveEntries)
{
    Eigen::VectorXd with_zero(3);
    with_zero << 1.0, 0.0, 2.0;
    EXPECT_ASSERT_FAILURE(to_normalized_coords(with_zero));

    Eigen::VectorXd with_negative(3);
    with_negative << 1.0, -2.0, 2.0;
    EXPECT_ASSERT_FAILURE(to_normalized_coords(with_negative));
}

TEST(ToNormalizedCoordsDeathTest, RejectsNonFiniteEntries)
{
    Eigen::VectorXd with_nan(3);
    with_nan << 1.0, kNaN, 2.0;
    EXPECT_ASSERT_FAILURE(to_normalized_coords(with_nan));

    Eigen::VectorXd with_inf(3);
    with_inf << 1.0, kInf, 2.0;
    EXPECT_ASSERT_FAILURE(to_normalized_coords(with_inf));
}

}  // namespace
}  // namespace cosserat::math
