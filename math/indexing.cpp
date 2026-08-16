#include "math/indexing.hpp"

#include "utils/assertions.hpp"

namespace cosserat {
Eigen::Index resolve_index(std::int64_t index, Eigen::Index count)
{
    utils::nice_assert(count > 0, "Cannot index into an empty stack");

    const Eigen::Index resolved = (index < 0)
        ? count + static_cast<Eigen::Index>(index)
        : static_cast<Eigen::Index>(index);

    utils::nice_assert(
        resolved >= 0 and resolved < count, "Constraint index is out of range"
    );
    return resolved;
}
} // End namespace cosserat
