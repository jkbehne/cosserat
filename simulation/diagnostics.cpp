#include "simulation/diagnostics.hpp"

#include <cmath>
#include <format>
#include <string>
#include <system_error>
#include <utility>

namespace cosserat::simulation {
using utils::nice_assert;

namespace {

/**
 * @brief Creates a directory and everything above it, failing loudly.
 *
 * Uses the error-code overload rather than the throwing one so that a failure
 * arrives through the same assertion channel as the rest of the library
 * instead of as a bare @c std::filesystem_error.
 *
 * @param directory Directory to create.
 */
void create_output_directory(const std::filesystem::path& directory)
{
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    nice_assert(
        not error, "Failed to create the output directory: " + error.message()
    );
    // create_directories reports no error when the path already exists as a
    // regular file, so the result has to be checked rather than assumed.
    nice_assert(
        std::filesystem::is_directory(directory),
        "Output path exists but is not a directory: " + directory.string()
    );
}

/** @brief Fails unless a name is usable as a single directory component. */
void assert_path_component(const std::string& name, const char* message)
{
    nice_assert(not name.empty(), message);
    nice_assert(
        name.find('/') == std::string::npos
            and name.find('\\') == std::string::npos,
        message
    );
}

} // End anonymous namespace

BasePathManager::BasePathManager(
    std::filesystem::path base_path, std::string body_name
) : m_base_path(std::move(base_path)), m_body_name(std::move(body_name))
{
    nice_assert(not m_base_path.empty(), "base_path must not be empty");
    assert_path_component(
        m_body_name, "body_name must be non-empty and free of path separators"
    );

    create_output_directory(m_base_path);
}

std::filesystem::path BasePathManager::get_next_dir(
    double time, std::uint64_t step
) const
{
    // A NaN or infinite time would otherwise become a directory literally
    // named step_000000001_st_nan, created without complaint.
    nice_assert(std::isfinite(time), "Expected time to be a finite value");

    // A step wider than step_digits is written correctly; only the name-order
    // sorting of those later directories degrades, which is not worth
    // capping how long a simulation may run.
    const std::string step_folder =
        std::format("step_{:0{}d}_st_{:.3f}", step, step_digits, time);

    // One directory per body beneath the step, so that two bodies written at
    // the same step cannot overwrite one another.
    const std::filesystem::path full_path = m_base_path / step_folder / m_body_name;
    create_output_directory(full_path);
    return full_path;
}

const std::filesystem::path& BasePathManager::base_path() const {return m_base_path;}

const std::string& BasePathManager::body_name() const {return m_body_name;}

StepSchedule::StepSchedule(std::uint64_t steps_to_skip)
    : m_steps_to_skip(steps_to_skip)
{
    // Zero would never fire and would divide by zero below.
    nice_assert(m_steps_to_skip > 0, "steps_to_skip must be at least one");
}

bool StepSchedule::should_write(std::uint64_t step) const
{
    return step % m_steps_to_skip == 0;
}

std::uint64_t StepSchedule::steps_to_skip() const {return m_steps_to_skip;}

BasicDiagnostics::BasicDiagnostics(
    std::filesystem::path base_path,
    std::string body_name,
    std::uint64_t steps_to_skip
) : m_manager(std::move(base_path), std::move(body_name)),
    m_schedule(steps_to_skip) {}

const BasePathManager& BasicDiagnostics::manager() const {return m_manager;}

const StepSchedule& BasicDiagnostics::schedule() const {return m_schedule;}

DebugDiagnostics::DebugDiagnostics(
    std::filesystem::path base_path,
    std::string body_name,
    std::uint64_t steps_to_skip
) : m_manager(std::move(base_path), std::move(body_name)),
    m_schedule(steps_to_skip) {}

const BasePathManager& DebugDiagnostics::manager() const {return m_manager;}

const StepSchedule& DebugDiagnostics::schedule() const {return m_schedule;}
} // End namespace cosserat::simulation
