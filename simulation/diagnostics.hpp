#pragma once

/**
 * @file diagnostics.hpp
 * @brief Periodic writing of simulation state to disk.
 *
 * A diagnostic is a rule that a time stepper hands a body to at whatever
 * interval the caller chooses. It works out where that body's state belongs
 * for the current step and asks the body to write itself there.
 *
 * The layout is
 *
 * @verbatim
 * base_path/step_000000042_st_0.420/body_name/
 * @endverbatim
 *
 * with one directory per step and one directory per body beneath it. The body
 * level is what keeps two bodies written at the same step from overwriting one
 * another, so the body's name is fixed when the manager is built rather than
 * passed per call.
 *
 * Step numbers are zero padded so that listing the step directories in name
 * order gives them in step order. Without the padding @c step_10 sorts before
 * @c step_2, which quietly reorders any glob-driven post-processing. The
 * padding is a convenience, not a limit: a run longer than the reserved width
 * simply produces wider names, and only the ordering of those later
 * directories degrades.
 *
 * Writing every step is rarely wanted, so each diagnostic carries a
 * @ref StepSchedule saying how often it fires. A schedule of one writes every
 * step; a schedule of @c n writes whenever the step index is a multiple of
 * @c n, which always includes step zero.
 *
 * Two diagnostics are provided, differing only in how much they write:
 * @ref BasicDiagnostics asks for the configuration needed to reconstruct a
 * body's shape, and @ref DebugDiagnostics asks for everything. Each is
 * constrained to systems that offer the corresponding entry point, so pairing
 * a diagnostic with a body that cannot serve it is caught at the call.
 *
 * This mirrors PyElastica's callback modules in intent rather than in detail.
 *
 * @note Nothing here removes existing output. Re-running with a different
 *       timestep produces different step directory names, so stale
 *       directories from an earlier run are left interleaved with the new
 *       ones. Rooting @p base_path at a fresh, timestamped directory per run
 *       is the safe way to avoid that; deleting on construction would be far
 *       too easy to point at the wrong path.
 */

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

#include "utils/assertions.hpp"

namespace cosserat::simulation {

/**
 * @brief A system that can write its full state to a directory.
 * @tparam T System type.
 */
template<typename T>
concept DebugWriteable = requires(T obj, const std::filesystem::path& out_dir)
{
    obj.write_debug(out_dir);
};

/**
 * @brief A system that can write its configuration to a directory.
 * @tparam T System type.
 */
template<typename T>
concept Writeable = requires(T obj, const std::filesystem::path& out_dir)
{
    obj.write(out_dir);
};

/**
 * @brief Works out and creates the output directory for one body at one step.
 *
 * Holds the root to write beneath and the name of the body it writes for.
 * Binding the body name here rather than passing it per call means a manager
 * can only ever write one body's data, which is what makes collisions between
 * bodies impossible rather than merely unlikely.
 */
class BasePathManager
{
public: // Static constexpr members
    /**
     * @brief Digits used for the zero-padded step number.
     *
     * Nine digits keeps the step directories in sorted order up to just under
     * a billion steps. A longer run is still written correctly; its step
     * numbers simply outgrow the padding, so those later directories no longer
     * sort into step order by name. That is a presentation detail and is not
     * enforced, since how long a simulation may run is not this class's
     * business.
     */
    static constexpr int step_digits = 9;

private: // Members
    std::filesystem::path m_base_path;
    std::string m_body_name;

public: // Methods
    /**
     * @brief Builds a manager rooted at a directory, for one named body.
     *
     * The root is created if it does not already exist.
     *
     * @param base_path Directory to write beneath; must be non-empty, and must
     *        not already exist as something other than a directory.
     * @param body_name Name of the body this manager writes for; must be
     *        non-empty and free of path separators, since it becomes a single
     *        directory component.
     */
    BasePathManager(std::filesystem::path base_path, std::string body_name);

    /**
     * @brief Returns the directory this body's state belongs in, creating it.
     *
     * The result is @c base_path/step_<step>_st_<time>/body_name, with the
     * step zero padded to @ref step_digits.
     *
     * @param time Current simulation time; must be finite, so that an
     *        unstable run cannot produce a directory named after a NaN.
     * @param step Current step index.
     * @return The created directory.
     *
     * @note Const because it does not change the manager, though it does
     *       create directories on disk as a side effect.
     */
    std::filesystem::path get_next_dir(double time, std::uint64_t step) const;

    /** @brief Root directory everything is written beneath. */
    const std::filesystem::path& base_path() const;

    /** @brief Name of the body this manager writes for. */
    const std::string& body_name() const;
};

/**
 * @brief How often a diagnostic fires.
 *
 * A schedule of @c n writes on every step whose index is a multiple of @c n,
 * so a schedule of one writes every step and step zero is always written.
 * Kept as its own type so both diagnostics share one definition of the rule
 * rather than each carrying a copy.
 */
class StepSchedule
{
private: // Members
    std::uint64_t m_steps_to_skip;

public: // Methods
    /**
     * @brief Builds a schedule firing once every @p steps_to_skip steps.
     * @param steps_to_skip Interval between writes; must be at least one,
     *        since an interval of zero would never fire.
     */
    explicit StepSchedule(std::uint64_t steps_to_skip);

    /**
     * @brief Whether a step should be written.
     * @param step Current step index.
     * @return True when @p step is a multiple of the interval.
     */
    bool should_write(std::uint64_t step) const;

    /** @brief Interval between writes. */
    std::uint64_t steps_to_skip() const;
};

/**
 * @brief Writes the configuration needed to reconstruct a body's shape.
 *
 * Suitable for the output a run actually keeps: enough to replay or plot the
 * motion, without the derived quantities.
 *
 * @see DebugDiagnostics for everything a body holds.
 */
class BasicDiagnostics
{
private: // Members
    BasePathManager m_manager;
    StepSchedule m_schedule;

public: // Methods
    /**
     * @brief Builds a diagnostic rooted at a directory, for one named body.
     * @param base_path Directory to write beneath.
     * @param body_name Name of the body this diagnostic writes for.
     * @param steps_to_skip Interval between writes; one writes every step.
     */
    BasicDiagnostics(
        std::filesystem::path base_path,
        std::string body_name,
        std::uint64_t steps_to_skip
    );

    /**
     * @brief Writes the body's configuration for the current step.
     *
     * @tparam System Any @ref Writeable system.
     * @param system Body to write; read but not modified.
     * @param time Current simulation time.
     * @param step Current step index.
     * @return True if the step fell on the schedule and was written, false if
     *         it was skipped. No directory is created for a skipped step.
     *
     * @note Non-const so that a diagnostic which buffers state between calls
     *       can be added later without changing every call site.
     */
    template<Writeable System>
    bool make_callback(System& system, double time, std::uint64_t step)
    {
        if (not m_schedule.should_write(step)) return false;
        const auto path = m_manager.get_next_dir(time, step);
        system.write(path);
        return true;
    }

    /** @brief The path manager this diagnostic writes through. */
    const BasePathManager& manager() const;

    /** @brief The schedule deciding which steps are written. */
    const StepSchedule& schedule() const;
};

/**
 * @brief Writes every stack a body holds.
 *
 * Suitable for diagnosing a run rather than keeping its results: the output is
 * much larger than @ref BasicDiagnostics and includes derived quantities that
 * can be recomputed.
 *
 * @see BasicDiagnostics for configuration only.
 */
class DebugDiagnostics
{
private: // Members
    BasePathManager m_manager;
    StepSchedule m_schedule;

public: // Methods
    /**
     * @brief Builds a diagnostic rooted at a directory, for one named body.
     * @param base_path Directory to write beneath.
     * @param body_name Name of the body this diagnostic writes for.
     * @param steps_to_skip Interval between writes; one writes every step.
     */
    DebugDiagnostics(
        std::filesystem::path base_path,
        std::string body_name,
        std::uint64_t steps_to_skip
    );

    /**
     * @brief Writes the body's full state for the current step.
     *
     * @tparam System Any @ref DebugWriteable system.
     * @param system Body to write; read but not modified.
     * @param time Current simulation time.
     * @param step Current step index.
     * @return True if the step fell on the schedule and was written, false if
     *         it was skipped. No directory is created for a skipped step.
     *
     * @note Non-const for the same reason as
     *       @ref BasicDiagnostics::make_callback.
     */
    template<DebugWriteable System>
    bool make_callback(System& system, double time, std::uint64_t step)
    {
        if (not m_schedule.should_write(step)) return false;
        const auto path = m_manager.get_next_dir(time, step);
        system.write_debug(path);
        return true;
    }

    /** @brief The path manager this diagnostic writes through. */
    const BasePathManager& manager() const;

    /** @brief The schedule deciding which steps are written. */
    const StepSchedule& schedule() const;
};

/** @brief Any one of the diagnostics, held by value. */
using DiagnosticVariant = std::variant<BasicDiagnostics, DebugDiagnostics>;

/**
 * @brief Fails if the held diagnostic cannot write the given system.
 *
 * Each diagnostic constrains its entry point to the concept it needs, so
 * probing whether the call is well formed decides compatibility without
 * naming the concept here.
 *
 * @tparam BodyType The system type to check against.
 * @param diag_var Diagnostic to check.
 * @param system System to check against; not modified.
 */
template<typename BodyType>
void validate(DiagnosticVariant& diag_var, BodyType& system)
{
    std::visit([&](auto& diag)
    {
        const double time = 0.0;
        const std::uint64_t step = 0;
        if constexpr (not requires {diag.make_callback(system, time, step);})
        {
            utils::nice_assert(false, "Diagnostic is incompatible with this system");
        }
    }, diag_var);
}

/**
 * @brief Applies the held diagnostic to the given system.
 *
 * @tparam BodyType The system type to write.
 * @param diag_var Diagnostic to apply.
 * @param system Body to write; read but not modified.
 * @param time Current simulation time.
 * @param step Current step index.
 * @return True if the step fell on the diagnostic's schedule and was written.
 */
template<typename BodyType>
bool make_callback(
    DiagnosticVariant& diag_var, BodyType& system, double time, std::uint64_t step
)
{
    return std::visit([&](auto& diag) -> bool
    {
        if constexpr (requires {diag.make_callback(system, time, step);})
        {
            return diag.make_callback(system, time, step);
        }
        else
        {
            utils::nice_assert(
                false, "Diagnostic is incompatible with this system"
            );
            return false;
        }
    }, diag_var);
}
} // End namespace cosserat::simulation
