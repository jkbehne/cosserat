#include "math/indexing.hpp"

#include <gtest/gtest.h>

namespace cosserat::math {
namespace {
// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif
// ---------------------------------------------------------------------------
// resolve_index
// ---------------------------------------------------------------------------

TEST(ResolveIndex, PassesThroughNonNegativeIndices)
{
    EXPECT_EQ(resolve_index(0, 5), 0);
    EXPECT_EQ(resolve_index(3, 5), 3);
    EXPECT_EQ(resolve_index(4, 5), 4);
}

TEST(ResolveIndex, CountsBackFromTheEndForNegativeIndices)
{
    EXPECT_EQ(resolve_index(-1, 5), 4);
    EXPECT_EQ(resolve_index(-2, 5), 3);
    EXPECT_EQ(resolve_index(-5, 5), 0);
}

TEST(ResolveIndexDeathTest, RejectsOutOfRangeIndices)
{
    EXPECT_ASSERT_FAILURE(resolve_index(5, 5));
    EXPECT_ASSERT_FAILURE(resolve_index(-6, 5));
    EXPECT_ASSERT_FAILURE(resolve_index(0, 0));
}
} // End anonymous namespace
} // End namespace cosserat::math
