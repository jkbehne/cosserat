#include "math/signed_distance_field.hpp"

#include <algorithm>

#include "utils/assertions.hpp"

namespace cosserat::math {

// AnalyticSphereField public methods
AnalyticSphereField::AnalyticSphereField(
    const Eigen::Vector3d& center, double radius, double margin
) : m_center(center), m_radius(radius), m_margin(margin)
{
    utils::nice_assert(
        std::isfinite(radius) and radius > 0.0, "sphere radius must be finite and positive"
    );
    utils::nice_assert(
        std::isfinite(margin) and margin > 0.0, "domain margin must be finite and positive"
    );
}

SignedDistance AnalyticSphereField::signed_distance(const Eigen::Vector3d& point) const
{
    const Eigen::Vector3d offset = point - m_center;
    const double length = offset.norm();

    SignedDistance result;
    result.distance = length - m_radius;
    // The centre is the medial axis: every direction is equally near, so
    // there is no gradient to report.
    result.gradient = length > 0.0 ? Eigen::Vector3d(offset / length) : Eigen::Vector3d::Zero();
    return result;
}

Eigen::AlignedBox3d AnalyticSphereField::domain() const
{
    const Eigen::Vector3d extent = Eigen::Vector3d::Constant(m_radius + m_margin);
    return Eigen::AlignedBox3d(m_center - extent, m_center + extent);
}

// AnalyticSphereField accessors
const Eigen::Vector3d& AnalyticSphereField::center() const {return m_center;}
double AnalyticSphereField::radius() const {return m_radius;}
double AnalyticSphereField::margin() const {return m_margin;}

// AnalyticBoxField public methods
AnalyticBoxField::AnalyticBoxField(
    const Eigen::Vector3d& center, const Eigen::Vector3d& half_extent, double margin
) : m_center(center), m_half_extent(half_extent), m_margin(margin)
{
    utils::nice_assert(
        half_extent.array().isFinite().all() and (half_extent.array() > 0.0).all(),
        "box half extents must be finite and positive"
    );
    utils::nice_assert(
        std::isfinite(margin) and margin > 0.0, "domain margin must be finite and positive"
    );
}

SignedDistance AnalyticBoxField::signed_distance(const Eigen::Vector3d& point) const
{
    const Eigen::Vector3d offset = (point - m_center).cwiseAbs() - m_half_extent;
    const Eigen::Vector3d outside = offset.cwiseMax(0.0);
    const double outside_length = outside.norm();
    // Negative inside: the least negative component is the nearest face.
    const double inside_depth = std::min(offset.maxCoeff(), 0.0);

    SignedDistance result;
    result.distance = outside_length + inside_depth;

    const Eigen::Vector3d sign = (point - m_center).cwiseSign();
    if (outside_length > 0.0)
    {
        result.gradient = sign.cwiseProduct(outside) / outside_length;
    }
    else
    {
        // Inside: the gradient points out through the nearest face.
        Eigen::Index axis = 0;
        offset.maxCoeff(&axis);
        result.gradient = Eigen::Vector3d::Zero();
        result.gradient(axis) = sign(axis) != 0.0 ? sign(axis) : 1.0;
    }
    return result;
}

Eigen::AlignedBox3d AnalyticBoxField::domain() const
{
    const Eigen::Vector3d extent = m_half_extent + Eigen::Vector3d::Constant(m_margin);
    return Eigen::AlignedBox3d(m_center - extent, m_center + extent);
}

// AnalyticBoxField accessors
const Eigen::Vector3d& AnalyticBoxField::center() const {return m_center;}
const Eigen::Vector3d& AnalyticBoxField::half_extent() const {return m_half_extent;}
double AnalyticBoxField::margin() const {return m_margin;}
} // End namespace cosserat::math
