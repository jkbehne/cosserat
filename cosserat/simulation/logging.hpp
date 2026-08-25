#pragma once

/**
 * @file logging.hpp
 * @brief Simple utilities for setting up logging via spdlog
 *
 * Intended as a place to put utilities relating to parsing command line arguments to setup
 * logging.
 *
 * All three overloads set the level on spdlog's default logger, so a library
 * that logs through the free @c spdlog::info and friends picks the
 * configuration up without being told about it. The overloads differ only in
 * where the output goes: the one-argument form leaves the existing sinks
 * alone, and the others install a logger that mirrors to the console and to a
 * timestamped file.
 *
 * @section logging_levels Recognised levels
 *
 * @c trace , @c debug , @c info , @c warn or @c warning , @c err or @c error ,
 * @c critical , and @c off . The comparison is case sensitive, so @c OFF and
 * @c Info are not levels and are rejected along with anything else
 * unrecognised.
 */

#include <string>

namespace cosserat::simulation {

/**
 * @brief Sets the log level, leaving the current sinks in place.
 *
 * Nothing is written to file. Use one of the other overloads for that.
 *
 * @param log_level Level as a string; see @ref logging_levels. Anything
 *        unrecognised fails an assertion rather than quietly disabling
 *        logging, since a mistyped level would otherwise look exactly like a
 *        working run that happened to be silent.
 */
void setup_logging(const std::string& log_level);

/**
 * @brief Sets the log level and mirrors output to a file named @c unnamed.
 *
 * Equivalent to calling the three-argument overload with a log name of
 * @c unnamed.
 *
 * @param log_level Level as a string; see @ref logging_levels.
 * @param log_directory Directory to write the log file into; must not be
 *        empty. Created if it does not exist.
 */
void setup_logging(const std::string& log_level, const std::string& log_directory);

/**
 * @brief Sets the log level and mirrors output to the console and a file.
 *
 * The file is created at @c log_directory/log_name_timestamp.log, with the
 * directory created if it does not already exist. Both sinks are installed on
 * a logger named @c global_logger which becomes spdlog's default, so calling
 * this replaces whatever the default logger was before.
 *
 * @param log_level Level as a string; see @ref logging_levels. Anything
 *        unrecognised fails an assertion.
 * @param log_directory Directory to write the log file into; must not be
 *        empty, and must not already exist as something other than a
 *        directory.
 * @param log_name Stem of the log file; must not be empty. Pass the
 *        two-argument overload to get the default of @c unnamed.
 *
 * @note The timestamp resolves to the second, so two calls inside the same
 *       second land on the same filename. The file sink appends rather than
 *       truncates precisely so that the second call adds to the first one's
 *       output instead of erasing it.
 *
 * @note The logger flushes from @c warn upward. spdlog otherwise buffers file
 *       output until the sink is destroyed, and an aborting assertion runs no
 *       destructors, so a failing run would leave an empty log. Messages below
 *       @c warn may still be lost on a crash; flushing on every level is a one
 *       line change if that matters more than the extra writes.
 *
 * @note Neither the file path nor the logger is returned, so a caller cannot
 *       report where the log went. If that becomes useful, returning the
 *       resolved path is the smallest change that would provide it.
 */
void setup_logging(
    const std::string& log_level, const std::string& log_directory, const std::string& log_name
);
} // End namespace cosserat::simulation
