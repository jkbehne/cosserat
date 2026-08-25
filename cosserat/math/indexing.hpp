#pragma once

#include <Eigen/Core>

namespace cosserat {
/**
 * @brief Resolves a possibly negative index against a stack length.
 *
 * Negative indices count back from the end, so -1 addresses the last entry.
 * This mirrors the Python indexing the reference implementation relies on for
 * expressions such as constraining nodes @c (0, -1).
 *
 * @param index Index to resolve; may be negative.
 * @param count Number of entries in the stack being indexed.
 * @return The equivalent non-negative index.
 */
Eigen::Index resolve_index(std::int64_t index, Eigen::Index count);
} // End namespace cosserat
