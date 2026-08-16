#pragma once

/**
 * @file damping.hpp
 * @brief Damping rules for Cosserat rod systems.
 *
 * Each rule is an independent value type that knows nothing about any
 * particular body. A rule is constructed from its own parameters alone and
 * reads everything it needs from the system handed to @c dampen_rates, so a
 * single rule may be applied to any number of compatible systems.
 *
 * Two families are provided:
 *
 * - Analytical dampers (@ref UniformAnalyticalDamper,
 *   @ref PhysicalAnalyticalDamper, @ref LegacyAnalyticalDamper) integrate the
 *   damping term exactly and scale the rates in place. They impose no extra
 *   restriction on the simulation timestep.
 * - @ref RayleighDissipation writes damping into the external force and torque
 *   accumulators instead, leaving the time stepper to integrate it. Large
 *   damping constants may therefore require a smaller timestep.
 *
 * @ref LaplaceDissipationFilter is a low-pass filter rather than a damper in
 * the physical sense: it removes high-frequency content from the rates while
 * leaving smooth modes essentially untouched.
 */

#include <cmath>
#include <concepts>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "math/types.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

/**
 * @brief A system whose translational and rotational rates can be damped.
 *
 * Velocities are stored per node and angular velocities per element, so the
 * two stacks differ in length by one on an open rod.
 */
template<typename T>
concept DampableSystem = requires(T sys)
{
    {sys.num_elements()} -> std::convertible_to<std::uint64_t>;
    {sys.velocities()} -> std::same_as<Vector3DStack&>;
    {sys.angular_velocities()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A dampable system that also exposes mass, inertia and dilatation.
 *
 * Required by the analytical protocols that divide the damping constant
 * through by an inertia, and that raise the rotational coefficient to the
 * element dilatation.
 */
template<typename T>
concept InertiallyDampableSystem = DampableSystem<T> and requires(T sys)
{
    {sys.mass()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.inv_mass_second_moments()} -> std::same_as<const Matrix3DStack&>;
    {sys.dilatations()} -> std::same_as<const Eigen::VectorXd&>;
};

/**
 * @brief A dampable system exposing rest lengths and external accumulators.
 *
 * Required by @ref RayleighDissipation, which adds damping forces and torques
 * rather than scaling the rates directly.
 */
template<typename T>
concept RayleighDampableSystem = DampableSystem<T> and requires(T sys)
{
    {sys.rest_lengths()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.external_forces()} -> std::same_as<Vector3DStack&>;
    {sys.external_torques()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief Extracts the diagonal of each element's inverse second mass moment.
 *
 * @param inv_moments One inverse second mass moment of inertia per element.
 * @return An @c N by 3 stack whose row @c i is the diagonal of
 *         @c inv_moments[i]. Off-diagonal entries are ignored.
 */
Vector3DStack inverse_inertia_diagonals(const Matrix3DStack& inv_moments);

/**
 * @brief Converts nodal masses to element masses on an open rod.
 *
 * Interior elements take the average of their two nodes; the first and last
 * elements additionally pick up the leftover half of the terminal node, so the
 * total mass is preserved exactly.
 *
 * @param nodal_mass Mass at each node. At least two entries are required.
 * @return One mass per element, of length @c nodal_mass.size() - 1.
 */
Eigen::VectorXd element_masses_from_nodal(const Eigen::VectorXd& nodal_mass);

// ---------------------------------------------------------------------------
// Analytical dampers
// ---------------------------------------------------------------------------

/**
 * @brief Damps both rates by a single shared exponential coefficient.
 *
 * The coefficient is @f$ e^{-\gamma \Delta t} @f$ for damping constant
 * @f$ \gamma @f$ of dimension 1/T, and is applied identically to translational
 * and rotational velocities.
 *
 * Because the damping term is treated analytically, this rule is
 * unconditionally stable: raising the damping constant never destabilises the
 * simulation. A practical way to tune it is to start high enough to obtain a
 * stable run, then reduce until the expected dynamics reappear.
 */
struct UniformAnalyticalDamper
{
private: // Members
    double m_damping_constant;
    double m_time_step;
    double m_coefficient;

public: // Methods
    /**
     * @brief Builds the damper and precomputes its coefficient.
     * @param uniform_damping_constant Damping constant, finite and non-negative.
     * @param time_step Simulation timestep, finite and positive.
     */
    UniformAnalyticalDamper(double uniform_damping_constant, double time_step);

    /**
     * @brief Scales both rate stacks by the shared coefficient.
     * @tparam SystemType Any @ref DampableSystem.
     * @param system System whose rates are damped in place.
     */
    template<DampableSystem SystemType>
    void dampen_rates(SystemType& system, double) const
    {
        system.velocities() *= m_coefficient;
        system.angular_velocities() *= m_coefficient;
    }

    /** @brief The damping constant supplied at construction. */
    double damping_constant() const;
    /** @brief The timestep supplied at construction. */
    double time_step() const;
    /** @brief The precomputed exponential coefficient. */
    double coefficient() const;
};

/**
 * @brief Damps translation and rotation with separately dimensioned constants.
 *
 * The translational coefficient divides through by nodal mass and the
 * rotational coefficient by the element inertia:
 *
 * @f[ c^{t}_{i} = e^{-\gamma_t \Delta t / m_i}, \qquad
 *     c^{r}_{ij} = e^{-\gamma_r \Delta t \, (J^{-1})_{ij}} @f]
 *
 * The rotational coefficient is then raised to the element dilatation before
 * being applied. This keeps both constants dimensionally consistent, which is
 * the reason to prefer this rule over @ref LegacyAnalyticalDamper.
 *
 * Coefficients are recomputed from the system on every call, so one damper may
 * be shared across systems of differing size.
 */
struct PhysicalAnalyticalDamper
{
private: // Members
    double m_translational_damping_constant;
    double m_rotational_damping_constant;
    double m_time_step;

public: // Methods
    /**
     * @brief Builds the damper from its constants alone.
     * @param translational_damping_constant Finite and non-negative.
     * @param rotational_damping_constant Finite and non-negative.
     * @param time_step Simulation timestep, finite and positive.
     */
    PhysicalAnalyticalDamper(
        double translational_damping_constant,
        double rotational_damping_constant,
        double time_step
    );

    /**
     * @brief Per-node translational coefficients for the given masses.
     * @param nodal_mass Mass at each node; every entry must be positive.
     * @param damping_constant Translational damping constant.
     * @param time_step Simulation timestep.
     * @return One coefficient per node.
     */
    static Eigen::VectorXd translational_coefficients(
        const Eigen::VectorXd& nodal_mass, double damping_constant, double time_step
    );

    /**
     * @brief Per-element, per-axis rotational coefficients.
     * @param inv_moments Inverse second mass moment of inertia per element.
     * @param damping_constant Rotational damping constant.
     * @param time_step Simulation timestep.
     * @return An @c N by 3 stack of coefficients.
     */
    static Vector3DStack rotational_coefficients(
        const Matrix3DStack& inv_moments, double damping_constant, double time_step
    );

    /**
     * @brief Scales the rates by the mass- and inertia-weighted coefficients.
     * @tparam SystemType Any @ref InertiallyDampableSystem.
     * @param system System whose rates are damped in place.
     */
    template<InertiallyDampableSystem SystemType>
    void dampen_rates(SystemType& system, double) const
    {
        Vector3DStack& velocities = system.velocities();
        Vector3DStack& omegas = system.angular_velocities();
        const Eigen::VectorXd& nodal_mass = system.mass();
        const Eigen::VectorXd& dilatations = system.dilatations();

        utils::nice_assert(
            velocities.rows() == nodal_mass.size(),
            "Expected one nodal mass per velocity row"
        );
        utils::nice_assert(
            omegas.rows() == dilatations.size(),
            "Expected one dilatation per angular velocity row"
        );

        const Eigen::VectorXd translational = translational_coefficients(
            nodal_mass, m_translational_damping_constant, m_time_step
        );
        const Vector3DStack rotational = rotational_coefficients(
            system.inv_mass_second_moments(),
            m_rotational_damping_constant,
            m_time_step
        );
        utils::nice_assert(
            omegas.rows() == rotational.rows(),
            "Expected one inertia entry per angular velocity row"
        );

        velocities.array().colwise() *= translational.array();
        for (Eigen::Index idx = 0; idx < omegas.rows(); ++idx)
        {
            omegas.row(idx).array() *=
                rotational.row(idx).array().pow(dilatations(idx));
        }
    }

    /** @brief The translational damping constant supplied at construction. */
    double translational_damping_constant() const;
    /** @brief The rotational damping constant supplied at construction. */
    double rotational_damping_constant() const;
    /** @brief The timestep supplied at construction. */
    double time_step() const;
};

/**
 * @brief The original protocol: one constant for both translation and rotation.
 *
 * Translation is scaled by @f$ e^{-\gamma \Delta t} @f$ while rotation uses
 * @f$ e^{-\gamma \Delta t \, m^{e}_i (J^{-1})_{ij}} @f$, where @f$ m^e @f$ is
 * the element mass. The same @f$ \gamma @f$ therefore carries different
 * dimensions in the two expressions and cannot be consistent for both.
 *
 * @deprecated Retained only for reproducing results from older cases. Prefer
 *             @ref PhysicalAnalyticalDamper.
 * @see PhysicalAnalyticalDamper
 */
struct LegacyAnalyticalDamper
{
private: // Members
    double m_damping_constant;
    double m_time_step;
    double m_translational_coefficient;

public: // Methods
    /**
     * @brief Builds the damper and precomputes the scalar translational term.
     * @param damping_constant Finite and non-negative.
     * @param time_step Simulation timestep, finite and positive.
     */
    LegacyAnalyticalDamper(double damping_constant, double time_step);

    /**
     * @brief Per-element, per-axis rotational coefficients.
     *
     * Weighted by element mass as well as inverse inertia, which is what makes
     * this protocol dimensionally inconsistent with its translational term.
     *
     * @param nodal_mass Mass at each node; converted to element masses.
     * @param inv_moments Inverse second mass moment of inertia per element.
     * @param damping_constant Damping constant.
     * @param time_step Simulation timestep.
     * @return An @c N by 3 stack of coefficients.
     */
    static Vector3DStack rotational_coefficients(
        const Eigen::VectorXd& nodal_mass,
        const Matrix3DStack& inv_moments,
        double damping_constant,
        double time_step
    );

    /**
     * @brief Scales velocities uniformly and rotation by element weighting.
     * @tparam SystemType Any @ref InertiallyDampableSystem.
     * @param system System whose rates are damped in place.
     */
    template<InertiallyDampableSystem SystemType>
    void dampen_rates(SystemType& system, double) const
    {
        Vector3DStack& omegas = system.angular_velocities();
        const Eigen::VectorXd& dilatations = system.dilatations();

        utils::nice_assert(
            omegas.rows() == dilatations.size(),
            "Expected one dilatation per angular velocity row"
        );

        const Vector3DStack rotational = rotational_coefficients(
            system.mass(),
            system.inv_mass_second_moments(),
            m_damping_constant,
            m_time_step
        );
        utils::nice_assert(
            omegas.rows() == rotational.rows(),
            "Expected one inertia entry per angular velocity row"
        );

        system.velocities() *= m_translational_coefficient;
        for (Eigen::Index idx = 0; idx < omegas.rows(); ++idx)
        {
            omegas.row(idx).array() *=
                rotational.row(idx).array().pow(dilatations(idx));
        }
    }

    /** @brief The damping constant supplied at construction. */
    double damping_constant() const;
    /** @brief The timestep supplied at construction. */
    double time_step() const;
    /** @brief The precomputed scalar translational coefficient. */
    double translational_coefficient() const;
};

// ---------------------------------------------------------------------------
// Rayleigh dissipation
// ---------------------------------------------------------------------------

/**
 * @brief Force-based damping added to the external accumulators.
 *
 * Adds @f$ \mathbf{F} = -\nu \mathbf{v} @f$ and
 * @f$ \boldsymbol{\tau} = -\nu \boldsymbol{\omega} @f$, where the endpoints
 * receive half the interior weighting. The damping constant is specified per
 * unit length and is rescaled internally by the average element length.
 *
 * Because the damping is integrated by the time stepper rather than treated
 * analytically, large constants may require a smaller timestep.
 *
 * @deprecated Retained for validating older cases. Prefer
 *             @ref UniformAnalyticalDamper or @ref PhysicalAnalyticalDamper.
 */
struct RayleighDissipation
{
public: // Static constexpr members
    /** @brief Weighting applied to the first and last nodes. */
    static constexpr double endpoint_factor = 0.5;

private: // Members
    double m_damping_constant;
    std::optional<double> m_relaxation_time;

public: // Methods
    /**
     * @brief Builds the damper from its constants alone.
     * @param damping_constant Finite and non-negative, per unit length.
     * @param relaxation_time Optional decay timescale. With no value the
     *        damping is constant in time; with a value the constant decays as
     *        @f$ e^{-t/\tau} @f$. Must be finite and positive when present.
     */
    explicit RayleighDissipation(
        double damping_constant,
        std::optional<double> relaxation_time = std::nullopt
    );

    /**
     * @brief The damping constant at a given time, before length rescaling.
     * @param time Simulation time; must be finite.
     * @return The constant itself, or its exponentially decayed value when a
     *         relaxation time was supplied.
     */
    double damping_at(double time) const;

    /**
     * @brief Mean element length, used to rescale the per-unit-length constant.
     * @param rest_lengths One rest length per element; at least one entry.
     */
    static double average_element_length(const Eigen::VectorXd& rest_lengths);

    /**
     * @brief Adds damping forces and torques to the external accumulators.
     *
     * Rates are read but not modified.
     *
     * @tparam SystemType Any @ref RayleighDampableSystem.
     * @param system System whose accumulators are updated in place.
     * @param time Simulation time, used only if a relaxation time was supplied.
     */
    template<RayleighDampableSystem SystemType>
    void dampen_rates(SystemType& system, double time) const
    {
        const Eigen::VectorXd& rest_lengths = system.rest_lengths();
        utils::nice_assert(
            rest_lengths.size() == static_cast<Eigen::Index>(system.num_elements()),
            "Expected one rest length per element"
        );
        const double nu = damping_at(time) * average_element_length(rest_lengths);

        Vector3DStack& forces = system.external_forces();
        const Vector3DStack& velocities = system.velocities();
        utils::nice_assert(forces.rows() > 0, "Need at least one node");
        utils::nice_assert(
            forces.rows() == velocities.rows(),
            "External force rows must match velocity rows"
        );

        forces -= nu * velocities;
        forces.row(0) += (1.0 - endpoint_factor) * nu * velocities.row(0);
        const Eigen::Index last = forces.rows() - 1;
        forces.row(last) += (1.0 - endpoint_factor) * nu * velocities.row(last);

        Vector3DStack& torques = system.external_torques();
        const Vector3DStack& omegas = system.angular_velocities();
        utils::nice_assert(
            torques.rows() == omegas.rows(),
            "External torque rows must match angular velocity rows"
        );
        torques -= nu * omegas;
    }

    /** @brief The damping constant supplied at construction. */
    double damping_constant() const;
    /** @brief The relaxation time supplied at construction, if any. */
    std::optional<double> relaxation_time() const;
};

// ---------------------------------------------------------------------------
// Laplace dissipation filter
// ---------------------------------------------------------------------------

/**
 * @brief Low-pass filter built from repeated 1D Laplacian application.
 *
 * Filters high-frequency (noise) modes out of the rates while leaving smooth
 * low-frequency modes essentially untouched. Constant and linear fields are
 * annihilated by the discrete Laplacian and therefore pass through unchanged;
 * the endpoints are always held fixed.
 *
 * The filter order is the number of times the operator is applied. Small
 * values (1, 2) filter aggressively and can remove genuine physics; large
 * values (9, 10) barely filter at all. Values from 3 to 7 are usually a
 * reasonable choice.
 *
 * @note Unlike the other rules, @c dampen_rates is non-const: the filter owns
 *       scratch buffers that it sizes on first use and reuses thereafter. A
 *       single instance must not be shared across threads.
 *
 * @see Jeanmart, H., and Winckelmans, G. (2007), Physics of Fluids 19(5),
 *      055110, for the numerics behind the filtering.
 */
struct LaplaceDissipationFilter
{
private: // Members
    unsigned int m_filter_order;
    Vector3DStack m_velocity_filter_term;
    Vector3DStack m_omega_filter_term;

public: // Methods
    /**
     * @brief Builds the filter from its order alone.
     * @param filter_order Number of Laplacian applications; must be positive.
     *        Higher values imply weaker filtering.
     */
    explicit LaplaceDissipationFilter(unsigned int filter_order);

    /**
     * @brief Filters both rate stacks in place.
     *
     * Scratch buffers are resized to match the system on first use, and reused
     * on subsequent calls with a system of the same size.
     *
     * @tparam SystemType Any @ref DampableSystem.
     * @param system System whose rates are filtered in place.
     */
    template<DampableSystem SystemType>
    void dampen_rates(SystemType& system, double)
    {
        Vector3DStack& velocities = system.velocities();
        Vector3DStack& omegas = system.angular_velocities();

        if (m_velocity_filter_term.rows() != velocities.rows())
        {
            m_velocity_filter_term.resize(velocities.rows(), 3);
        }
        if (m_omega_filter_term.rows() != omegas.rows())
        {
            m_omega_filter_term.resize(omegas.rows(), 3);
        }

        filter_rate(velocities, m_velocity_filter_term, m_filter_order);
        filter_rate(omegas, m_omega_filter_term, m_filter_order);
    }

    /**
     * @brief Subtracts repeated-Laplacian content from a stack of rates.
     *
     * The first and last rows are never modified.
     *
     * @param rates Rates to filter in place.
     * @param filter_term Scratch space with the same number of rows as
     *        @p rates; its contents are overwritten.
     * @param filter_order Number of Laplacian applications.
     */
    static void filter_rate(
        Vector3DStack& rates, Vector3DStack& filter_term, unsigned int filter_order
    );

    /** @brief The filter order supplied at construction. */
    unsigned int filter_order() const;
    /** @brief Scratch buffer for velocities; empty until the first call. */
    const Vector3DStack& velocity_filter_term() const;
    /** @brief Scratch buffer for angular velocities; empty until first call. */
    const Vector3DStack& omega_filter_term() const;
};

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

/** @brief Any one of the damping rules, held by value. */
using DamperVariant = std::variant<
    UniformAnalyticalDamper,
    PhysicalAnalyticalDamper,
    LegacyAnalyticalDamper,
    RayleighDissipation,
    LaplaceDissipationFilter
>;

/**
 * @brief Fails if the held rule cannot be applied to the given system.
 *
 * Each rule constrains @c dampen_rates to the concept it needs, so probing
 * whether the call is well formed is enough to decide compatibility without
 * naming the concept here.
 *
 * @tparam BodyType The system type to check against.
 * @param damper_var Rule to check. Taken by mutable reference because some
 *        rules have a non-const @c dampen_rates.
 * @param system System to check against; not modified.
 */
template<typename BodyType>
void validate(DamperVariant& damper_var, BodyType& system)
{
    std::visit([&](auto& damper)
    {
        const double time = 0.0;
        if constexpr (not requires {damper.dampen_rates(system, time);})
        {
            utils::nice_assert(false, "Damper is incompatible with this system");
        }
    }, damper_var);
}

/**
 * @brief Applies the held rule to the given system.
 *
 * @tparam BodyType The system type to damp.
 * @param damper_var Rule to apply.
 * @param system System to damp in place.
 * @param time Current simulation time.
 */
template<typename BodyType>
void dampen_rates(DamperVariant& damper_var, BodyType& system, double time)
{
    std::visit([&](auto& damper)
    {
        if constexpr (requires {damper.dampen_rates(system, time);})
        {
            damper.dampen_rates(system, time);
        }
        else utils::nice_assert(false, "Damper is incompatible with this system");
    }, damper_var);
}
} // End namespace cosserat::physics
