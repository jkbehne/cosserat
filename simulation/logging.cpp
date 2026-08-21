#include "simulation/logging.hpp"

#include <filesystem>
#include <memory>
#include <system_error>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "utils/assertions.hpp"
#include "utils/file_utils.hpp"

namespace cosserat::simulation {

using utils::nice_assert;
namespace fs = std::filesystem;

namespace {

/**
 * @brief Parses a level string, rejecting anything spdlog does not recognise.
 *
 * spdlog reports an unrecognised name as @c off rather than failing, which
 * would turn a typo into a silent run. The only way to tell the two apart is
 * that a genuine request for @c off spells it exactly, so that is what is
 * checked.
 *
 * @param log_level Level as a string.
 * @return The parsed level.
 */
spdlog::level::level_enum get_log_level(const std::string& log_level)
{
    const auto log_level_enum = spdlog::level::from_str(log_level);
    nice_assert(
        (log_level_enum != spdlog::level::off) or (log_level == "off"),
        "log_level string " + log_level + " was not recognized"
    );
    return log_level_enum;
}

/**
 * @brief Creates a directory and everything above it, failing loudly.
 *
 * Uses the error-code overload rather than the throwing one so that a failure
 * arrives through the same assertion channel as the rest of the library
 * instead of as a bare @c std::filesystem_error.
 *
 * @param directory Directory to create.
 */
void create_log_directory(const fs::path& directory)
{
    std::error_code error;
    fs::create_directories(directory, error);
    nice_assert(
        not error, "Failed to create the log directory: " + error.message()
    );
    // create_directories reports no error when the path already exists as a
    // regular file, so the result has to be checked rather than assumed.
    nice_assert(
        fs::is_directory(directory),
        "Log path exists but is not a directory: " + directory.string()
    );
}
} // End anonymous namespace

void setup_logging(const std::string& log_level) {spdlog::set_level(get_log_level(log_level));}

void setup_logging(const std::string& log_level, const std::string& log_directory)
{
    setup_logging(log_level, log_directory, "unnamed");
}

void setup_logging(
    const std::string& log_level, const std::string& log_directory, const std::string& log_name
)
{
    nice_assert(not log_directory.empty(), "Call one argument setup_logging for no file logging");
    nice_assert(not log_name.empty(), "Call two argument setup_logging for no log name");

    // Parsed before anything is created, so a mistyped level does not leave an
    // empty log directory behind.
    const auto ll = get_log_level(log_level);

    fs::path log_path(log_directory);
    create_log_directory(log_path);
    log_path /= (log_name + "_" + utils::timestamp_string() + ".log");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(ll);

    // Appending rather than truncating. The timestamp resolves only to the
    // second, so two calls inside the same second name the same file, and
    // truncating would erase whatever the first call had already written.
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        log_path.string(), false /* append to an existing file */
    );
    file_sink->set_level(ll);

    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
    auto multi_logger = std::make_shared<spdlog::logger>(
        "global_logger", sinks.begin(), sinks.end()
    );
    multi_logger->set_level(ll);
    // spdlog buffers file output and flushes only when the sink is destroyed.
    // Assertions in this library abort, and std::abort neither runs
    // destructors nor flushes stdio, so without this the log file is empty
    // exactly when a failure needs explaining. Flushing from warn upward keeps
    // the record of anything going wrong while leaving the per-step debug
    // chatter buffered; pass spdlog::level::trace instead to flush everything,
    // at the cost of a write per record.
    multi_logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(multi_logger);
}
} // End namespace cosserat::simulation
