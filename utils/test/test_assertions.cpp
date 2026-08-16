#include "utils/assertions.hpp"

#include <gtest/gtest.h>

#include <csignal>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace cosserat;

namespace {

/// Deliberately has no operator<<, so utils::streamable must reject it.
struct NotStreamable
{
    int value{0};
};

/// A user defined type that opts in to streaming via a free operator<<.
struct CustomStreamable
{
    int value{0};
};

std::ostream& operator<<(std::ostream& stream, const CustomStreamable& value)
{
    return stream << "CustomStreamable(" << value.value << ")";
}

/// A stand in for the wrapper types that logging libraries return from
/// operator<<: convertible to std::ostream&, but not the same type.
struct OstreamProxy
{
    std::ostream& stream;

    operator std::ostream&() const
    {
        return stream;
    }
};

/// Streams via an operator<< whose return type is only convertible to
/// std::ostream&. This is what separates the convertible_to spelling of the
/// concept from the stricter same_as spelling.
struct ProxyReturnStreamable
{
    int value{0};
};

OstreamProxy operator<<(std::ostream& stream, const ProxyReturnStreamable& value)
{
    stream << "ProxyReturnStreamable(" << value.value << ")";
    return OstreamProxy{stream};
}

/// Models "is utils::nice_assert callable with these argument types" as an
/// ordinary bool, so the constraints can be probed at runtime rather than
/// with static_assert.
template<typename ConditionType, typename MessageType>
concept nice_assert_invocable = requires(const ConditionType condition, const MessageType& message)
{
    utils::nice_assert(condition, message);
};

/// Redirects std::cerr for the lifetime of the object and hands back whatever
/// was written to it.
class CerrCapture
{
public:
    CerrCapture() : original_buffer{std::cerr.rdbuf(buffer.rdbuf())} {}

    ~CerrCapture()
    {
        std::cerr.rdbuf(original_buffer);
    }

    CerrCapture(const CerrCapture&) = delete;
    CerrCapture& operator=(const CerrCapture&) = delete;
    CerrCapture(CerrCapture&&) = delete;
    CerrCapture& operator=(CerrCapture&&) = delete;

    std::string contents() const
    {
        return buffer.str();
    }

private:
    std::ostringstream buffer;
    std::streambuf* original_buffer;
};

} // namespace

TEST(StreamableConcept, AcceptsStdString)
{
    EXPECT_TRUE(utils::streamable<std::string>);
}

TEST(StreamableConcept, AcceptsStdStringView)
{
    EXPECT_TRUE(utils::streamable<std::string_view>);
}

TEST(StreamableConcept, AcceptsFundamentalTypes)
{
    EXPECT_TRUE(utils::streamable<bool>);
    EXPECT_TRUE(utils::streamable<char>);
    EXPECT_TRUE(utils::streamable<int>);
    EXPECT_TRUE(utils::streamable<std::int64_t>);
    EXPECT_TRUE(utils::streamable<double>);
    EXPECT_TRUE(utils::streamable<const char*>);
}

TEST(StreamableConcept, AcceptsTypeWithUserDefinedStreamOperator)
{
    EXPECT_TRUE(utils::streamable<CustomStreamable>);
}

TEST(StreamableConcept, AcceptsOperatorReturningConvertibleType)
{
    // OstreamProxy converts to std::ostream&, so the convertible_to form of
    // the concept accepts this. A same_as form would reject it.
    EXPECT_TRUE(utils::streamable<ProxyReturnStreamable>);
}

TEST(StreamableConcept, RejectsTypeWithoutStreamOperator)
{
    EXPECT_FALSE(utils::streamable<NotStreamable>);
}

TEST(StreamableConcept, RejectsContainerWithoutStreamOperator)
{
    EXPECT_FALSE(utils::streamable<std::vector<int>>);
}

TEST(StreamableConcept, AcceptsAnyPointerBecauseOfTheVoidPointerOverload)
{
    // Worth pinning down because it is easy to misread the concept as a
    // guarantee of a meaningful message. Every object pointer converts to
    // const void*, and basic_ostream has a member overload for that, so a
    // pointer to a type with no operator<< of its own still satisfies
    // streamable. It just prints an address.
    EXPECT_TRUE(utils::streamable<NotStreamable*>);
    EXPECT_FALSE(utils::streamable<NotStreamable>);
}

TEST(NiceAssertConstraints, AcceptsBoolConditionAndStreamableMessage)
{
    EXPECT_TRUE((nice_assert_invocable<bool, const char*>));
    EXPECT_TRUE((nice_assert_invocable<bool, std::string>));
    EXPECT_TRUE((nice_assert_invocable<bool, CustomStreamable>));
}

TEST(NiceAssertConstraints, RejectsPointerCondition)
{
    // The whole point of constraining the condition to exactly bool: a raw
    // pointer would otherwise convert silently.
    EXPECT_FALSE((nice_assert_invocable<int*, const char*>));
}

TEST(NiceAssertConstraints, RejectsIntegerCondition)
{
    EXPECT_FALSE((nice_assert_invocable<int, const char*>));
}

TEST(NiceAssertConstraints, RejectsNonStreamableMessage)
{
    EXPECT_FALSE((nice_assert_invocable<bool, NotStreamable>));
}

TEST(NiceAssert, DoesNotAbortWhenConditionIsTrue)
{
    // If nice_assert aborted here the whole test binary would die, so
    // reaching the end of this test is itself the assertion.
    utils::nice_assert(true, "this message must never be printed");
    utils::nice_assert(true, std::string{"string message"});
    utils::nice_assert(true, CustomStreamable{42});
    SUCCEED();
}

TEST(NiceAssert, WritesNothingToStderrWhenConditionIsTrue)
{
    std::string captured;
    {
        const CerrCapture capture;
        utils::nice_assert(true, "silent");
        captured = capture.contents();
    }
    EXPECT_TRUE(captured.empty()) << "unexpected output: " << captured;
}

TEST(NiceAssertDeathTest, AbortsWhenConditionIsFalse)
{
    EXPECT_DEATH(utils::nice_assert(false, "boom"), "Assertion Failed: boom");
}

TEST(NiceAssertDeathTest, TerminatesViaAbortSignal)
{
    EXPECT_EXIT(
        utils::nice_assert(false, "signal check"),
        testing::KilledBySignal(SIGABRT),
        "Assertion Failed"
    );
}

TEST(NiceAssertDeathTest, PrintsStdStringMessage)
{
    const std::string message{"message carried by std::string"};
    EXPECT_DEATH(utils::nice_assert(false, message), "Assertion Failed: message carried by std::string");
}

TEST(NiceAssertDeathTest, PrintsUserDefinedStreamableMessage)
{
    EXPECT_DEATH(
        utils::nice_assert(false, CustomStreamable{7}),
        "Assertion Failed: CustomStreamable\\(7\\)"
    );
}

TEST(NiceAssertDeathTest, PrintsMessageWhoseOperatorReturnsProxy)
{
    EXPECT_DEATH(
        utils::nice_assert(false, ProxyReturnStreamable{9}),
        "Assertion Failed: ProxyReturnStreamable\\(9\\)"
    );
}

TEST(NiceAssertDeathTest, PrintsIntegerMessage)
{
    EXPECT_DEATH(utils::nice_assert(false, 1234), "Assertion Failed: 1234");
}

TEST(NiceAssertDeathTest, ReportsEveryDiagnosticField)
{
    EXPECT_DEATH(
        utils::nice_assert(false, "full report"),
        "Assertion Failed: full report\nFile: [^\n]+\nLine: [0-9]+\nFunction: [^\n]+\n"
    );
}

TEST(NiceAssertDeathTest, ReportsCallSiteFileName)
{
    EXPECT_DEATH(
        utils::nice_assert(false, "file check"),
        "File: [^\n]*test_assertions\\.cpp"
    );
}

TEST(NiceAssertDeathTest, ReportsEnclosingFunctionName)
{
    EXPECT_DEATH(
        utils::nice_assert(false, "function check"),
        "Function: [^\n]*ReportsEnclosingFunctionName"
    );
}

TEST(NiceAssertDeathTest, ReportsCallerLineNotHeaderLine)
{
    // This is the behaviour that makes the default argument spelling of
    // std::source_location::current() worth having: the location must name
    // this call site, not a line inside assertions.hpp. Both statements sit
    // on a single line each so the recorded line number is unambiguous.
    const std::string expected_line = std::to_string(__LINE__ + 2);
    const std::string pattern = "Line: " + expected_line + "[^0-9]";
    EXPECT_DEATH(utils::nice_assert(false, "line check"), pattern);
}
