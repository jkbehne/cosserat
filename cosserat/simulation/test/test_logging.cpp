#include <cosserat/simulation/logging.hpp>

#include <gtest/gtest.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

/**
 * spdlog's default logger is process-wide state, so each test gets its own
 * scratch directory and the default logger is put back afterwards. Without the
 * restore, a test that installs a file logger would leave every later test
 * writing into a deleted directory.
 */
class LoggingTest : public ::testing::Test
{
protected:
    std::filesystem::path m_root;
    std::shared_ptr<spdlog::logger> m_original;
    spdlog::level::level_enum m_original_level = spdlog::level::info;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_root = std::filesystem::temp_directory_path()
            / (std::string("logging_") + info->name());
        std::filesystem::remove_all(m_root);

        m_original = spdlog::default_logger();
        m_original_level = m_original->level();
    }

    void TearDown() override
    {
        spdlog::set_default_logger(m_original);
        spdlog::set_level(m_original_level);
        std::filesystem::remove_all(m_root);
    }

    /** Every regular file directly beneath the scratch directory. */
    std::vector<std::filesystem::path> log_files() const
    {
        std::vector<std::filesystem::path> files;
        if (!std::filesystem::is_directory(m_root)) return files;
        for (const auto& entry : std::filesystem::directory_iterator(m_root))
        {
            if (entry.is_regular_file()) files.push_back(entry.path());
        }
        return files;
    }

    /** Contents of the one log file, failing if there is not exactly one. */
    std::string only_log_contents() const
    {
        const std::vector<std::filesystem::path> files = log_files();
        if (files.size() != 1)
        {
            ADD_FAILURE() << "expected exactly one log file, found " << files.size();
            return {};
        }
        std::ifstream stream(files.front());
        return std::string(std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>());
    }
};

// ---------------------------------------------------------------------------
// Level parsing
// ---------------------------------------------------------------------------

TEST_F(LoggingTest, AcceptsEveryDocumentedLevel)
{
    const std::vector<std::pair<std::string, spdlog::level::level_enum>> levels{
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"warning", spdlog::level::warn},
        {"err", spdlog::level::err},
        {"error", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off},
    };

    for (const auto& [name, expected] : levels)
    {
        EXPECT_NO_THROW({ setup_logging(name); }) << name;
        EXPECT_EQ(spdlog::default_logger()->level(), expected) << name;
    }
}

// ---------------------------------------------------------------------------
// File output
// ---------------------------------------------------------------------------

TEST_F(LoggingTest, CreatesTheDirectoryAndAFileNamedAfterTheLog)
{
    ASSERT_FALSE(std::filesystem::exists(m_root));

    setup_logging("info", m_root.string(), "myrun");

    ASSERT_TRUE(std::filesystem::is_directory(m_root));
    const std::vector<std::filesystem::path> files = log_files();
    ASSERT_EQ(files.size(), 1u);

    const std::string name = files.front().filename().string();
    EXPECT_EQ(name.rfind("myrun_", 0), 0u) << name;
    EXPECT_EQ(files.front().extension(), ".log");
    // stem is myrun_YYYYMMDD_HHMMSS
    EXPECT_EQ(files.front().stem().string().size(), std::string("myrun_").size() + 15u);
}

TEST_F(LoggingTest, CreatesNestedDirectories)
{
    const std::filesystem::path nested = m_root / "a" / "b" / "c";

    setup_logging("info", nested.string(), "deep");

    EXPECT_TRUE(std::filesystem::is_directory(nested));
    bool found = false;
    for (const auto& entry : std::filesystem::directory_iterator(nested))
    {
        if (entry.is_regular_file()) found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(LoggingTest, TheTwoArgumentFormNamesTheFileUnnamed)
{
    setup_logging("info", m_root.string());

    const std::vector<std::filesystem::path> files = log_files();
    ASSERT_EQ(files.size(), 1u);
    EXPECT_EQ(files.front().filename().string().rfind("unnamed_", 0), 0u)
        << files.front().filename().string();
}

TEST_F(LoggingTest, MessagesReachTheFile)
{
    setup_logging("info", m_root.string(), "run");
    spdlog::info("hello from the test");
    spdlog::default_logger()->flush();

    EXPECT_NE(only_log_contents().find("hello from the test"), std::string::npos);
}

// The level applies to the file sink too, so anything below it is dropped
// rather than written and filtered later.
TEST_F(LoggingTest, MessagesBelowTheLevelDoNotReachTheFile)
{
    setup_logging("warn", m_root.string(), "run");
    spdlog::info("this should not appear");
    spdlog::warn("but this should");
    spdlog::default_logger()->flush();

    const std::string contents = only_log_contents();
    EXPECT_EQ(contents.find("this should not appear"), std::string::npos);
    EXPECT_NE(contents.find("but this should"), std::string::npos);
}

TEST_F(LoggingTest, AnOffLevelWritesNothing)
{
    setup_logging("off", m_root.string(), "run");
    spdlog::critical("not even this");
    spdlog::default_logger()->flush();

    EXPECT_TRUE(only_log_contents().empty());
}

// The one-argument form only changes the level; it must not create a file.
TEST_F(LoggingTest, TheOneArgumentFormWritesNoFile)
{
    setup_logging("debug");

    EXPECT_FALSE(std::filesystem::exists(m_root));
    EXPECT_EQ(spdlog::default_logger()->level(), spdlog::level::debug);
}

// ---------------------------------------------------------------------------
// Flushing
//
// spdlog buffers file output and flushes only when the sink is destroyed.
// Assertions in this library abort, and std::abort runs no destructors, so a
// logger that did not flush would leave an empty file exactly when a failure
// needed explaining.
// ---------------------------------------------------------------------------

TEST_F(LoggingTest, WarningsAreFlushedWithoutBeingAskedTo)
{
    setup_logging("info", m_root.string(), "run");
    spdlog::info("buffered context");
    spdlog::warn("something went wrong");
    // Deliberately no explicit flush.

    const std::string contents = only_log_contents();
    EXPECT_NE(contents.find("something went wrong"), std::string::npos);
    // A flush writes everything buffered, so the earlier context survives too.
    EXPECT_NE(contents.find("buffered context"), std::string::npos);
}

TEST_F(LoggingTest, ErrorsAndCriticalsAreAlsoFlushed)
{
    setup_logging("info", m_root.string(), "run");
    spdlog::error("an error");
    EXPECT_NE(only_log_contents().find("an error"), std::string::npos);

    spdlog::critical("a critical");
    EXPECT_NE(only_log_contents().find("a critical"), std::string::npos);
}

// Below warn the output stays buffered, which is the trade being made: the
// per-step debug chatter does not cost a write each.
TEST_F(LoggingTest, MessagesBelowWarnStayBufferedUntilFlushed)
{
    setup_logging("info", m_root.string(), "run");
    spdlog::info("still in the buffer");

    EXPECT_TRUE(only_log_contents().empty());

    spdlog::default_logger()->flush();
    EXPECT_NE(only_log_contents().find("still in the buffer"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Repeated setup
// ---------------------------------------------------------------------------

TEST_F(LoggingTest, CallingTwiceReplacesTheDefaultLoggerWithoutThrowing)
{
    EXPECT_NO_THROW({
        setup_logging("info", m_root.string(), "first");
        setup_logging("debug", m_root.string(), "second");
    });
    EXPECT_EQ(spdlog::default_logger()->level(), spdlog::level::debug);
}

// The timestamp resolves to the second, so two calls inside one second name
// the same file. The sink appends, so the first call's output survives instead
// of being truncated away.
TEST_F(LoggingTest, ASecondSetupInTheSameSecondAppendsRatherThanTruncates)
{
    setup_logging("info", m_root.string(), "run");
    spdlog::warn("from the first logger");   // warn, so it is flushed

    setup_logging("info", m_root.string(), "run");
    spdlog::warn("from the second logger");

    const std::vector<std::filesystem::path> files = log_files();
    ASSERT_EQ(files.size(), 1u) << "the same second should reuse one filename";

    const std::string contents = only_log_contents();
    EXPECT_NE(contents.find("from the first logger"), std::string::npos)
        << "the first logger's output was truncated away";
    EXPECT_NE(contents.find("from the second logger"), std::string::npos);
}

TEST_F(LoggingTest, DifferentNamesGiveDifferentFiles)
{
    setup_logging("info", m_root.string(), "alpha");
    setup_logging("info", m_root.string(), "beta");

    EXPECT_EQ(log_files().size(), 2u);
}

// ---------------------------------------------------------------------------
// Rejected input
// ---------------------------------------------------------------------------

TEST_F(LoggingTest, RejectsAnUnrecognisedLevel)
{
    EXPECT_ASSERT_FAILURE(setup_logging("nonsense"));
    EXPECT_ASSERT_FAILURE(setup_logging(""));
    EXPECT_ASSERT_FAILURE(setup_logging("nonsense", m_root.string(), "run"));
}

// The comparison is case sensitive, so these are not levels.
TEST_F(LoggingTest, RejectsLevelsInTheWrongCase)
{
    EXPECT_ASSERT_FAILURE(setup_logging("OFF"));
    EXPECT_ASSERT_FAILURE(setup_logging("Info"));
    EXPECT_ASSERT_FAILURE(setup_logging("WARN"));
}

TEST_F(LoggingTest, RejectsAnEmptyDirectoryOrName)
{
    EXPECT_ASSERT_FAILURE(setup_logging("info", ""));
    EXPECT_ASSERT_FAILURE(setup_logging("info", "", "run"));
    EXPECT_ASSERT_FAILURE(setup_logging("info", m_root.string(), ""));
}

// A path that already exists as a regular file cannot become a log directory.
// This arrives as an assertion rather than a raw filesystem_error.
TEST_F(LoggingTest, RejectsADirectoryPathThatIsAFile)
{
    std::filesystem::create_directories(m_root);
    const std::filesystem::path blocker = m_root / "not_a_directory";
    std::ofstream(blocker) << "in the way\n";
    ASSERT_TRUE(std::filesystem::is_regular_file(blocker));

    EXPECT_ASSERT_FAILURE(setup_logging("info", blocker.string(), "run"));
}

}  // namespace
}  // namespace cosserat::simulation
