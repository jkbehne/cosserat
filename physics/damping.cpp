/**
 * @file damping.cpp
 * @brief Non-template implementations for the damping rules.
 */

#include "physics/damping.hpp"

#include <cmath>

namespace cosserat::physics {

Vector3DStack inverse_inertia_diagonals(const Matrix3DStack& inv_moments)
{
    const auto num_elements = static_cast<Eigen::Index>(inv_moments.size());
    utils::nice_assert(num_elements > 0, "Need at least one inertia entry");

    Vector3DStack diagonals(num_elements, 3);
    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        diagonals.row(idx) = inv_moments[idx].diagonal().transpose();
    }
    return diagonals;
}

Eigen::VectorXd element_masses_from_nodal(const Eigen::VectorXd& nodal_mass)
{
    const Eigen::Index num_nodes = nodal_mass.size();
    utils::nice_assert(num_nodes > 1, "Need at least two nodes to form an element");

    // Interior elements average their two nodes. The terminal half-nodes are
    // then added back at the ends, which is what preserves the total mass.
    const Eigen::Index num_elements = num_nodes - 1;
    Eigen::VectorXd element_mass =
        0.5 * (nodal_mass.tail(num_elements) + nodal_mass.head(num_elements));
    element_mass(0) += 0.5 * nodal_mass(0);
    element_mass(num_elements - 1) += 0.5 * nodal_mass(num_nodes - 1);
    return element_mass;
}

// ---------------------------------------------------------------------------
// UniformAnalyticalDamper
// ---------------------------------------------------------------------------

UniformAnalyticalDamper::UniformAnalyticalDamper(
    double uniform_damping_constant, double time_step
) : m_damping_constant(uniform_damping_constant), m_time_step(time_step)
{
    utils::nice_assert(
        std::isfinite(m_damping_constant) and m_damping_constant >= 0.0,
        "uniform_damping_constant must be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_time_step) and m_time_step > 0.0,
        "time_step must be finite and greater than zero"
    );
    // Independent of any system, so it is safe to cache here.
    m_coefficient = std::exp(-m_damping_constant * m_time_step);
}

double UniformAnalyticalDamper::damping_constant() const {return m_damping_constant;}
double UniformAnalyticalDamper::time_step() const {return m_time_step;}
double UniformAnalyticalDamper::coefficient() const {return m_coefficient;}

// ---------------------------------------------------------------------------
// PhysicalAnalyticalDamper
// ---------------------------------------------------------------------------

PhysicalAnalyticalDamper::PhysicalAnalyticalDamper(
    double translational_damping_constant,
    double rotational_damping_constant,
    double time_step
) : m_translational_damping_constant(translational_damping_constant),
    m_rotational_damping_constant(rotational_damping_constant),
    m_time_step(time_step)
{
    utils::nice_assert(
        std::isfinite(m_translational_damping_constant)
        and m_translational_damping_constant >= 0.0,
        "translational_damping_constant must be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_rotational_damping_constant)
        and m_rotational_damping_constant >= 0.0,
        "rotational_damping_constant must be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_time_step) and m_time_step > 0.0,
        "time_step must be finite and greater than zero"
    );
}

Eigen::VectorXd PhysicalAnalyticalDamper::translational_coefficients(
    const Eigen::VectorXd& nodal_mass, double damping_constant, double time_step
)
{
    utils::nice_assert(nodal_mass.size() > 0, "Need at least one nodal mass");
    utils::nice_assert(
        (nodal_mass.array() > 0.0).all(), "All nodal masses must be positive"
    );
    return (-damping_constant * time_step * nodal_mass.array().inverse()).exp();
}

Vector3DStack PhysicalAnalyticalDamper::rotational_coefficients(
    const Matrix3DStack& inv_moments, double damping_constant, double time_step
)
{
    const Vector3DStack inv_inertia = inverse_inertia_diagonals(inv_moments);
    return (-damping_constant * time_step * inv_inertia.array()).exp();
}

double PhysicalAnalyticalDamper::translational_damping_constant() const
{
    return m_translational_damping_constant;
}
double PhysicalAnalyticalDamper::rotational_damping_constant() const
{
    return m_rotational_damping_constant;
}
double PhysicalAnalyticalDamper::time_step() const {return m_time_step;}

// ---------------------------------------------------------------------------
// LegacyAnalyticalDamper
// ---------------------------------------------------------------------------

LegacyAnalyticalDamper::LegacyAnalyticalDamper(
    double damping_constant, double time_step
) : m_damping_constant(damping_constant), m_time_step(time_step)
{
    utils::nice_assert(
        std::isfinite(m_damping_constant) and m_damping_constant >= 0.0,
        "damping_constant must be finite and non-negative"
    );
    utils::nice_assert(
        std::isfinite(m_time_step) and m_time_step > 0.0,
        "time_step must be finite and greater than zero"
    );
    // Scalar and system-independent, unlike the rotational term.
    m_translational_coefficient = std::exp(-m_damping_constant * m_time_step);
}

Vector3DStack LegacyAnalyticalDamper::rotational_coefficients(
    const Eigen::VectorXd& nodal_mass,
    const Matrix3DStack& inv_moments,
    double damping_constant,
    double time_step
)
{
    const Eigen::VectorXd element_mass = element_masses_from_nodal(nodal_mass);
    const Vector3DStack inv_inertia = inverse_inertia_diagonals(inv_moments);
    utils::nice_assert(
        element_mass.size() == inv_inertia.rows(),
        "Element mass count must match the number of inertia entries"
    );

    return (-damping_constant * time_step
            * (inv_inertia.array().colwise() * element_mass.array()))
        .exp();
}

double LegacyAnalyticalDamper::damping_constant() const {return m_damping_constant;}
double LegacyAnalyticalDamper::time_step() const {return m_time_step;}
double LegacyAnalyticalDamper::translational_coefficient() const
{
    return m_translational_coefficient;
}

// ---------------------------------------------------------------------------
// RayleighDissipation
// ---------------------------------------------------------------------------

RayleighDissipation::RayleighDissipation(
    double damping_constant, std::optional<double> relaxation_time
) : m_damping_constant(damping_constant), m_relaxation_time(relaxation_time)
{
    utils::nice_assert(
        std::isfinite(m_damping_constant) and m_damping_constant >= 0.0,
        "damping_constant must be finite and non-negative"
    );
    if (m_relaxation_time)
    {
        utils::nice_assert(
            std::isfinite(*m_relaxation_time) and *m_relaxation_time > 0.0,
            "relaxation_time must be finite and greater than zero. "
            "Use no value for no relaxation."
        );
    }
}

double RayleighDissipation::damping_at(double time) const
{
    utils::nice_assert(std::isfinite(time), "Expected time to be a finite value");

    if (not m_relaxation_time) return m_damping_constant;
    return m_damping_constant * std::exp(-time / *m_relaxation_time);
}

double RayleighDissipation::average_element_length(
    const Eigen::VectorXd& rest_lengths
)
{
    const Eigen::Index num_elements = rest_lengths.size();
    utils::nice_assert(num_elements > 0, "Need at least one rest length");
    utils::nice_assert(
        (rest_lengths.array() > 0.0).all(), "All rest lengths must be positive"
    );
    return rest_lengths.sum() / static_cast<double>(num_elements);
}

double RayleighDissipation::damping_constant() const {return m_damping_constant;}
std::optional<double> RayleighDissipation::relaxation_time() const
{
    return m_relaxation_time;
}

// ---------------------------------------------------------------------------
// LaplaceDissipationFilter
// ---------------------------------------------------------------------------

LaplaceDissipationFilter::LaplaceDissipationFilter(unsigned int filter_order)
    : m_filter_order(filter_order)
{
    utils::nice_assert(filter_order > 0, "Filter order must be a positive integer");
    // Buffers stay empty until the first call, when the system size is known.
}

void LaplaceDissipationFilter::filter_rate(
    Vector3DStack& rates, Vector3DStack& filter_term, unsigned int filter_order
)
{
    const Eigen::Index num_rows = rates.rows();
    utils::nice_assert(
        filter_term.rows() == num_rows, "Filter term must match the rate rows"
    );

    filter_term = rates;
    if (num_rows < 3)
    {
        // No interior rows to filter, and the endpoints are held fixed.
        filter_term.setZero();
        return;
    }

    const Eigen::Index interior = num_rows - 2;
    // Each interior row reads the neighbours that the same pass overwrites, so
    // the update must be evaluated into a temporary before being assigned back.
    Vector3DStack updated(interior, 3);
    for (unsigned int pass = 0; pass < filter_order; ++pass)
    {
        updated = 0.25
            * (-filter_term.bottomRows(interior) - filter_term.topRows(interior)
               + 2.0 * filter_term.middleRows(1, interior));
        filter_term.middleRows(1, interior) = updated;
        filter_term.row(0).setZero();
        filter_term.row(num_rows - 1).setZero();
    }
    rates -= filter_term;
}

unsigned int LaplaceDissipationFilter::filter_order() const {return m_filter_order;}
const Vector3DStack& LaplaceDissipationFilter::velocity_filter_term() const
{
    return m_velocity_filter_term;
}
const Vector3DStack& LaplaceDissipationFilter::omega_filter_term() const
{
    return m_omega_filter_term;
}
} // End namespace cosserat::physics
