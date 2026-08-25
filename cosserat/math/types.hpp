#pragma once

#include <vector>

#include <Eigen/Dense>

namespace cosserat {

using Vector3DStack = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using Vector3DStackT = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix3DStack = std::vector<Eigen::Matrix3d>;

} // End namespace cosserat
