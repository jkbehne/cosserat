#include "physics/plane.hpp"

#include <Eigen/Dense>

#include "math/linalg.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

using utils::nice_assert;

// Plane public methods
Plane::Plane(const Eigen::Vector3d& origin, const Eigen::Vector3d& normal)
    : m_origin(origin), m_normal(normal)
{
    nice_assert(origin.array().isFinite().all(), "Plane origin must be finite");
    nice_assert(math::is_unit_vector(normal, tolerance), "Plane normal must be a unit vector");
}

double Plane::signed_distance(const Eigen::Vector3d& point) const
{
    return m_normal.dot(point - m_origin);
}

// Plane accessors
const Eigen::Vector3d& Plane::origin() const {return m_origin;}
const Eigen::Vector3d& Plane::normal() const {return m_normal;}
} // End namespace cosserat::physics
