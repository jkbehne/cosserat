#include "math/functions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "utils/assertions.hpp"

namespace cosserat::math {
// Constructor for LinearRamp
LinearRamp::LinearRamp(
    double onset_time_,
    double ramp_time_,
    std::optional<double> offset_time_
) : m_onset_time(onset_time_),
    m_ramp_time(ramp_time_),
    m_offset_time(offset_time_)
{
    utils::nice_assert(
        std::isfinite(m_onset_time), "onset_time must be finite"
    );
    utils::nice_assert(
        std::isfinite(m_ramp_time) and m_ramp_time > 0.0,
        "Expected ramp time to be finite and greater than 0"
    );
    if (m_offset_time)
    {
        utils::nice_assert(
            std::isfinite(*m_offset_time),
            "Expected offset_time to be finite. Use no value for infinite offset time."
        );
        utils::nice_assert(
            *m_offset_time >= m_onset_time + m_ramp_time,
            "Expected offset_time to be >= onset_time + ramp_time"
        );
    }
}

// Methods for LinearRamp
double LinearRamp::operator()(double time) const
{
    utils::nice_assert(std::isfinite(time), "Expected time to be a finite value");
    if (time <= m_onset_time) return 0.0;
    if (m_offset_time and time > *m_offset_time) return 0.0;
    return std::min(1.0, (time - m_onset_time) / m_ramp_time);
}

double LinearRamp::onset_time() const {return m_onset_time;}
double LinearRamp::ramp_time() const {return m_ramp_time;}
std::optional<double> LinearRamp::offset_time() const {return m_offset_time;}

// Utility functions
Eigen::VectorXd to_normalized_coords(const Eigen::VectorXd& lengths)
{
    const auto num_elements = lengths.size();
    utils::nice_assert(num_elements > 0, "Must have at least one length");
    utils::nice_assert(
        (lengths.array() > 0.0).all(),
        "Expected all entries of lengths to be greater than zero"
    );
    utils::nice_assert(
        lengths.array().isFinite().all(), "All lengths should be finite numbers"
    );
    Eigen::VectorXd result(num_elements);
    std::partial_sum(
        lengths.data(), lengths.data() + num_elements, result.data()
    );
    return (1.0 / result(num_elements - 1)) * result;
}
} // End namespace cosserat::math
