#pragma once

#include <optional>

#include <Eigen/Core>

namespace cosserat::math {

struct LinearRamp
{
private: // Members
    double m_onset_time;
    double m_ramp_time;
    std::optional<double> m_offset_time;

public: // Constructor
    LinearRamp(
        double onset_time_,
        double ramp_time_,
        std::optional<double> offset_time_
    );

public: // Methods
    double operator()(double time) const;
    double onset_time() const;
    double ramp_time() const;
    std::optional<double> offset_time() const;
};

Eigen::VectorXd to_normalized_coords(const Eigen::VectorXd& lengths);

} // End namespace cosserat::math
