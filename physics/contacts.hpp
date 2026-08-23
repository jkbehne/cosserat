#pragma once

/**
 * @file contacts.hpp
 * @brief Contact rules between rods, rigid bodies and surfaces.
 *
 * Contact here is a penalty model, not a constraint solve. Bodies are allowed
 * to overlap, and the overlap is what generates the force: a spring
 * proportional to the penetration depth, plus a damper proportional to the
 * rate at which the two are driving into one another. There is no impulse
 * solver, no complementarity condition, and no guarantee of non-penetration.
 * A contact rule is therefore just another external load, accumulated into the
 * same stacks a force rule writes to, and it needs nothing from the time
 * stepper beyond being called once per step.
 *
 * @f[ \gamma = r_1 + r_2 - d, \qquad
 *     \mathbf{F} = \tfrac{1}{2} H(\gamma)
 *     \left(k\,\gamma + \nu\,(\mathbf{v}_{rel} \cdot \hat{\mathbf{n}})\right)
 *     \hat{\mathbf{n}} @f]
 *
 * Geometry is handled in @ref minimum_distance.hpp: every shape is a capsule,
 * and contact is the distance between two segments against the sum of their
 * radii.
 *
 * @section contacts_pairs What is and is not modelled
 *
 * Following the reference implementation, contact exists for rod-to-rod, a rod
 * with itself, rod-to-cylinder, rod-to-sphere, rod-to-plane with and without
 * friction, and cylinder-to-plane. Every pair but the last involves a rod.
 * There is deliberately no sphere-to-sphere, sphere-to-cylinder,
 * cylinder-to-cylinder or sphere-to-plane contact, because the reference has
 * none.
 *
 * @warning Rod-to-rod and self contact are **frictionless and apply no
 *          torque**. They write only to the force accumulators, so two rods in
 *          contact slide past one another freely and cannot transmit spin.
 *          Friction exists only against a plane, and contact torque only for a
 *          cylinder against a rod or a plane. This is faithful to the
 *          reference rather than a simplification made here.
 *
 * This is a port of PyElastica's @c contact_forces and @c _contact_functions
 * modules.
 */

#include <concepts>
#include <cstdint>
#include <variant>

#include <Eigen/Core>

#include "math/minimum_distance.hpp"
#include "math/types.hpp"

#include "physics/plane.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

/** @brief Penetration below which a pair is treated as separated. */
inline constexpr double contact_separation_tolerance = -1e-5;

/**
 * @brief Closest approach below which the contact normal is taken as undefined.
 *
 * When two capsule axes intersect exactly the separation is zero and there is
 * no direction to push along. The reference implementation divides by that
 * zero and produces a NaN, which then spreads silently through the whole
 * simulation. Two rods crossing at a right angle is an entirely ordinary
 * initial condition, so such a pair is skipped here instead. This is the one
 * deliberate departure from the reference in this file, and it differs only
 * where the reference yields NaN.
 */
inline constexpr double contact_normal_tolerance = 1e-12;

/** @brief Distance from a surface within which an element counts as touching. */
inline constexpr double surface_tolerance = 1e-4;

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

/**
 * @brief A rod that can take part in contact.
 * @tparam T System type.
 */
template<typename T>
concept ContactableRod = requires(T sys)
{
    {sys.num_elements()} -> std::convertible_to<std::int64_t>;
    {sys.positions()} -> std::same_as<const Vector3DStack&>;
    {sys.velocities()} -> std::same_as<const Vector3DStack&>;
    {sys.tangents()} -> std::same_as<const Vector3DStack&>;
    {sys.radii()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.lengths()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.internal_forces()} -> std::same_as<const Vector3DStack&>;
    {sys.mutable_external_forces()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A rod that can take part in contact with friction against a surface.
 *
 * Friction needs the material frame and the angular state, which plain contact
 * does not.
 *
 * @tparam T System type.
 */
template<typename T>
concept FrictionalRod = ContactableRod<T> and requires(T sys)
{
    {sys.masses()} -> std::same_as<const Eigen::VectorXd&>;
    {sys.frames()} -> std::same_as<const Matrix3DStack&>;
    {sys.angular_velocities()} -> std::same_as<const Vector3DStack&>;
    {sys.internal_torques()} -> std::same_as<const Vector3DStack&>;
    {sys.mutable_external_torques()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A rigid cylinder that can take part in contact.
 * @tparam T System type.
 */
template<typename T>
concept ContactableCylinder = requires(T sys)
{
    {sys.positions()} -> std::same_as<const Vector3DStack&>;
    {sys.velocities()} -> std::same_as<const Vector3DStack&>;
    {sys.frames()} -> std::same_as<const Matrix3DStack&>;
    {sys.radius()} -> std::convertible_to<double>;
    {sys.length()} -> std::convertible_to<double>;
    {sys.mutable_external_forces()} -> std::same_as<Vector3DStack&>;
    {sys.mutable_external_torques()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A rigid sphere that can take part in contact.
 *
 * Structurally identical to @ref ContactableCylinder minus the torque
 * accumulator, since sphere contact applies none.
 *
 * @tparam T System type.
 */
template<typename T>
concept ContactableSphere = requires(T sys)
{
    {sys.positions()} -> std::same_as<const Vector3DStack&>;
    {sys.velocities()} -> std::same_as<const Vector3DStack&>;
    {sys.radius()} -> std::convertible_to<double>;
    {sys.mutable_external_forces()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A surface a body can rest against.
 * @tparam T Surface type.
 */
template<typename T>
concept ContactableSurface = requires(T surface)
{
    {surface.origin()} -> std::same_as<const Eigen::Vector3d&>;
    {surface.normal()} -> std::same_as<const Eigen::Vector3d&>;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

namespace detail {

/**
 * @brief What a plane contact leaves behind for a friction pass to use.
 */
struct PlaneContactResult
{
public: // Members
    /** @brief Magnitude of the plane's response on each element. */
    Eigen::VectorXd response_magnitude;

    /** @brief True for each element far enough from the plane to be free. */
    Eigen::Array<bool, Eigen::Dynamic, 1> out_of_contact;
};

/**
 * @brief Averages a nodal quantity onto the elements, doubling at the ends.
 *
 * Interior elements take half of each adjacent node; the two end elements pick
 * up an extra half of the terminal node, so that the total is conserved.
 *
 * @param nodal Per-node quantity.
 * @return Per-element quantity, one row shorter.
 */
Vector3DStack node_to_element_mass_or_force(const Vector3DStack& nodal);

/**
 * @brief Mass-weighted average of a nodal quantity onto the elements.
 * @param masses Node masses.
 * @param nodal Per-node quantity.
 * @return Per-element quantity, one row shorter.
 */
Vector3DStack node_to_element_velocity(
    const Eigen::VectorXd& masses, const Vector3DStack& nodal
);

/**
 * @brief Adds an element quantity onto the nodes, half to each end.
 * @param element Per-element quantity.
 * @param nodal Per-node accumulator, one row longer; updated in place.
 */
void elements_to_nodes_inplace(const Vector3DStack& element, Vector3DStack& nodal);

/**
 * @brief How much of each element's friction is static rather than kinetic.
 *
 * One where the element is slower than the threshold, tapering to zero as it
 * exceeds it.
 *
 * @param slip_velocity Per-element slip velocity.
 * @param threshold Speed at which an element counts as slipping.
 * @return One value per element, between zero and one.
 */
Eigen::VectorXd find_slipping_elements(
    const Vector3DStack& slip_velocity, double threshold
);

/**
 * @brief Contact forces between the elements of two rods.
 *
 * Every element of one rod is tested against every element of the other, with
 * a cheap rejection on the distance between element starts before the full
 * closest-approach test.
 *
 * The response carries a term beyond the spring and damper: the net force
 * already acting on each element is projected onto the contact normal, and its
 * compressive part is cancelled so that the two rods are not driven further
 * into each other. The reference implementation marks that term with a note
 * saying it should be removed and the example retuned, so it is reproduced
 * here for parity rather than endorsed.
 *
 * @param element_start_one Start node position of each element of the first rod.
 * @param radii_one Element radii of the first rod.
 * @param lengths_one Element lengths of the first rod.
 * @param tangents_one Element tangents of the first rod.
 * @param velocities_one Node velocities of the first rod.
 * @param internal_forces_one Node internal forces of the first rod.
 * @param external_forces_one Node external force accumulator of the first rod.
 * @param element_start_two Start node position of each element of the second rod.
 * @param radii_two Element radii of the second rod.
 * @param lengths_two Element lengths of the second rod.
 * @param tangents_two Element tangents of the second rod.
 * @param velocities_two Node velocities of the second rod.
 * @param internal_forces_two Node internal forces of the second rod.
 * @param external_forces_two Node external force accumulator of the second rod.
 * @param contact_k Contact spring constant.
 * @param contact_nu Contact damping constant.
 */
void contact_forces_rod_rod(
    const Vector3DStack& element_start_one,
    const Eigen::VectorXd& radii_one,
    const Eigen::VectorXd& lengths_one,
    const Vector3DStack& tangents_one,
    const Vector3DStack& velocities_one,
    const Vector3DStack& internal_forces_one,
    Vector3DStack& external_forces_one,
    const Vector3DStack& element_start_two,
    const Eigen::VectorXd& radii_two,
    const Eigen::VectorXd& lengths_two,
    const Vector3DStack& tangents_two,
    const Vector3DStack& velocities_two,
    const Vector3DStack& internal_forces_two,
    Vector3DStack& external_forces_two,
    double contact_k,
    double contact_nu
);

/**
 * @brief Contact forces between the elements of one rod and itself.
 *
 * Only pairs separated by more than a few elements are tested. The gap is
 * @f$ 1 + \lceil 0.8 \pi r / l \rceil @f$ elements wide, roughly the arc a rod
 * of that radius needs in order to bend back onto itself; without it the
 * always-touching neighbours would generate spurious contact.
 *
 * Unlike @ref contact_forces_rod_rod there is no equilibrium-force term, only
 * the spring and damper.
 *
 * @param element_start Start node position of each element.
 * @param radii Element radii.
 * @param lengths Element lengths.
 * @param tangents Element tangents.
 * @param velocities Node velocities.
 * @param external_forces Node external force accumulator.
 * @param contact_k Contact spring constant.
 * @param contact_nu Contact damping constant.
 */
void contact_forces_self_rod(
    const Vector3DStack& element_start,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths,
    const Vector3DStack& tangents,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces,
    double contact_k,
    double contact_nu
);

/**
 * @brief Contact forces and torques between a rod and a rigid cylinder.
 *
 * Includes a friction term in the slip direction, capped by a Coulomb limit,
 * and applies the resulting moment about the cylinder's centre to its torque
 * accumulator.
 *
 * @param element_center Midpoint of each rod element.
 * @param element_edge Edge vector of each rod element.
 * @param cylinder_center Centre of the cylinder.
 * @param cylinder_tip Centre of the cylinder's starting end face.
 * @param cylinder_edge Edge vector along the cylinder's axis.
 * @param radii_sum Rod element radius plus cylinder radius, per element.
 * @param lengths_sum Rod element length plus cylinder length, per element.
 * @param internal_forces_rod Node internal forces of the rod; unused, accepted
 *        for signature parity with the reference.
 * @param external_forces_rod Node external force accumulator of the rod.
 * @param external_forces_cylinder External force accumulator of the cylinder.
 * @param external_torques_cylinder External torque accumulator of the cylinder.
 * @param cylinder_frame Body frame of the cylinder.
 * @param velocities_rod Node velocities of the rod.
 * @param velocity_cylinder Velocity of the cylinder.
 * @param contact_k Contact spring constant.
 * @param contact_nu Contact damping constant.
 * @param velocity_damping_coefficient Damping in the slip direction.
 * @param friction_coefficient Coulomb friction coefficient.
 */
void contact_forces_rod_cylinder(
    const Vector3DStack& element_center,
    const Vector3DStack& element_edge,
    const Eigen::Vector3d& cylinder_center,
    const Eigen::Vector3d& cylinder_tip,
    const Eigen::Vector3d& cylinder_edge,
    const Eigen::VectorXd& radii_sum,
    const Eigen::VectorXd& lengths_sum,
    const Vector3DStack& internal_forces_rod,
    Vector3DStack& external_forces_rod,
    Vector3DStack& external_forces_cylinder,
    Vector3DStack& external_torques_cylinder,
    const Eigen::Matrix3d& cylinder_frame,
    const Vector3DStack& velocities_rod,
    const Eigen::Vector3d& velocity_cylinder,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
);

/**
 * @brief Contact forces between a rod and a rigid sphere.
 *
 * Applies force to both bodies and torque to neither, matching the reference.
 *
 * @param element_center Midpoint of each rod element.
 * @param element_edge Edge vector of each rod element.
 * @param sphere_center Centre of the sphere.
 * @param radii_sum Rod element radius plus sphere radius, per element.
 * @param lengths_sum Rod element length plus sphere diameter, per element.
 * @param external_forces_rod Node external force accumulator of the rod.
 * @param external_forces_sphere External force accumulator of the sphere.
 * @param velocities_rod Node velocities of the rod.
 * @param velocity_sphere Velocity of the sphere.
 * @param contact_k Contact spring constant.
 * @param contact_nu Contact damping constant.
 * @param velocity_damping_coefficient Damping in the slip direction.
 * @param friction_coefficient Coulomb friction coefficient.
 */
void contact_forces_rod_sphere(
    const Vector3DStack& element_center,
    const Vector3DStack& element_edge,
    const Eigen::Vector3d& sphere_center,
    const Eigen::VectorXd& radii_sum,
    const Eigen::VectorXd& lengths_sum,
    Vector3DStack& external_forces_rod,
    Vector3DStack& external_forces_sphere,
    const Vector3DStack& velocities_rod,
    const Eigen::Vector3d& velocity_sphere,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
);

/**
 * @brief Contact forces between a rod and a plane.
 *
 * The response has three parts: cancelling whatever net force is pressing the
 * element into the surface, an elastic push proportional to penetration, and a
 * damper on the normal velocity.
 *
 * @param plane_origin A point on the plane.
 * @param plane_normal Unit normal to the plane.
 * @param k Contact spring constant.
 * @param nu Contact damping constant.
 * @param radii Element radii of the rod.
 * @param masses Node masses of the rod.
 * @param positions Node positions of the rod.
 * @param velocities Node velocities of the rod.
 * @param internal_forces Node internal forces of the rod.
 * @param external_forces Node external force accumulator of the rod.
 * @return The response magnitude and contact mask, for a friction pass.
 */
PlaneContactResult contact_forces_rod_plane(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double k,
    double nu,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& masses,
    const Vector3DStack& positions,
    const Vector3DStack& velocities,
    const Vector3DStack& internal_forces,
    Vector3DStack& external_forces
);

/**
 * @brief Contact forces between a rod and a plane, with anisotropic friction.
 *
 * Runs @ref contact_forces_rod_plane first, then adds kinetic and static
 * friction along the rod's axis and in the rolling direction. Friction along
 * the axis differs forward and backward, and the rolling terms also produce a
 * torque about the contact point.
 *
 * @param plane_origin A point on the plane.
 * @param plane_normal Unit normal to the plane.
 * @param slip_velocity_tol Speed at which an element counts as slipping.
 * @param k Contact spring constant.
 * @param nu Contact damping constant.
 * @param kinetic_mu_forward Kinetic friction coefficient, forward.
 * @param kinetic_mu_backward Kinetic friction coefficient, backward.
 * @param kinetic_mu_sideways Kinetic friction coefficient, sideways.
 * @param static_mu_forward Static friction coefficient, forward.
 * @param static_mu_backward Static friction coefficient, backward.
 * @param static_mu_sideways Static friction coefficient, sideways.
 * @param radii Element radii of the rod.
 * @param masses Node masses of the rod.
 * @param tangents Element tangents of the rod.
 * @param positions Node positions of the rod.
 * @param frames Element frames of the rod.
 * @param velocities Node velocities of the rod.
 * @param angular_velocities Element angular velocities of the rod.
 * @param internal_forces Node internal forces of the rod.
 * @param external_forces Node external force accumulator of the rod.
 * @param internal_torques Element internal torques of the rod.
 * @param external_torques Element external torque accumulator of the rod.
 */
void contact_forces_rod_plane_with_anisotropic_friction(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double slip_velocity_tol,
    double k,
    double nu,
    double kinetic_mu_forward,
    double kinetic_mu_backward,
    double kinetic_mu_sideways,
    double static_mu_forward,
    double static_mu_backward,
    double static_mu_sideways,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& masses,
    const Vector3DStack& tangents,
    const Vector3DStack& positions,
    const Matrix3DStack& frames,
    const Vector3DStack& velocities,
    const Vector3DStack& angular_velocities,
    const Vector3DStack& internal_forces,
    Vector3DStack& external_forces,
    const Vector3DStack& internal_torques,
    Vector3DStack& external_torques
);

/**
 * @brief Contact forces between a rigid cylinder and a plane.
 *
 * The cylinder is treated as a point with a half-length clearance rather than
 * as an oriented capsule, matching the reference.
 *
 * @param plane_origin A point on the plane.
 * @param plane_normal Unit normal to the plane.
 * @param k Contact spring constant.
 * @param nu Contact damping constant.
 * @param length Cylinder length.
 * @param positions Position of the cylinder.
 * @param velocities Velocity of the cylinder.
 * @param external_forces External force accumulator of the cylinder.
 * @return The response magnitude and contact mask.
 */
PlaneContactResult contact_forces_cylinder_plane(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double k,
    double nu,
    double length,
    const Vector3DStack& positions,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces
);

/**
 * @brief Start node position of each element of a rod.
 * @param positions Node positions.
 * @return All node positions but the last.
 */
Vector3DStack element_start_positions(const Vector3DStack& positions);

/**
 * @brief Midpoint of each element of a rod.
 * @param positions Node positions.
 * @return The average of each adjacent node pair.
 */
Vector3DStack element_center_positions(const Vector3DStack& positions);

} // End namespace detail

// ---------------------------------------------------------------------------
// Contact rules
// ---------------------------------------------------------------------------

/**
 * @brief A contact that never does anything.
 *
 * Accepts any pair of systems, so it is the safe alternative to hold when a
 * contact is registered but should not act.
 */
class NoContact
{
public: // Methods
    /**
     * @brief Does nothing.
     * @tparam SystemOne First system type.
     * @tparam SystemTwo Second system type.
     * @param system_one First system.
     * @param system_two Second system.
     * @param time Current simulation time.
     */
    template<typename SystemOne, typename SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)system_one;
        (void)system_two;
        (void)time;
    }

    /**
     * @brief Does nothing.
     *
     * The single-system arity, so that a registered self contact can also be
     * disabled by holding this instead.
     *
     * @tparam SystemType System type.
     * @param system The system.
     * @param time Current simulation time.
     */
    template<typename SystemType>
    void apply_contact(SystemType& system, double time) const
    {
        (void)system;
        (void)time;
    }
};

/**
 * @brief Contact between the elements of two distinct rods.
 *
 * @warning Frictionless and torque-free; see the file-level warning.
 */
class RodRodContact
{
private: // Members
    double m_k;
    double m_nu;

public: // Methods
    /**
     * @brief Builds a rod-to-rod contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     */
    RodRodContact(double k, double nu);

    /**
     * @brief Applies contact forces between two rods.
     * @tparam SystemOne Any @ref ContactableRod.
     * @tparam SystemTwo Any @ref ContactableRod.
     * @param system_one First rod.
     * @param system_two Second rod.
     * @param time Current simulation time; unused.
     */
    template<ContactableRod SystemOne, ContactableRod SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        if (math::prune_rod_rod(
                system_one.positions(), system_one.radii(), system_one.lengths(),
                system_two.positions(), system_two.radii(), system_two.lengths()))
        {
            return;
        }

        detail::contact_forces_rod_rod(
            detail::element_start_positions(system_one.positions()),
            system_one.radii(), system_one.lengths(), system_one.tangents(),
            system_one.velocities(), system_one.internal_forces(),
            system_one.mutable_external_forces(),
            detail::element_start_positions(system_two.positions()),
            system_two.radii(), system_two.lengths(), system_two.tangents(),
            system_two.velocities(), system_two.internal_forces(),
            system_two.mutable_external_forces(),
            m_k, m_nu
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

/**
 * @brief Contact between a rod and itself.
 *
 * The only single-system contact, since a rod is the only body with enough
 * elements to touch itself.
 *
 * @warning Frictionless and torque-free; see the file-level warning.
 */
class RodSelfContact
{
private: // Members
    double m_k;
    double m_nu;

public: // Methods
    /**
     * @brief Builds a self contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     */
    RodSelfContact(double k, double nu);

    /**
     * @brief Applies self contact forces to a rod.
     * @tparam SystemType Any @ref ContactableRod.
     * @param system The rod.
     * @param time Current simulation time; unused.
     */
    template<ContactableRod SystemType>
    void apply_contact(SystemType& system, double time) const
    {
        (void)time;
        detail::contact_forces_self_rod(
            detail::element_start_positions(system.positions()),
            system.radii(), system.lengths(), system.tangents(),
            system.velocities(), system.mutable_external_forces(),
            m_k, m_nu
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

/**
 * @brief Contact between a rod and a rigid cylinder, with friction.
 *
 * The rod is always the first system and the cylinder the second.
 */
class RodCylinderContact
{
private: // Members
    double m_k;
    double m_nu;
    double m_velocity_damping_coefficient;
    double m_friction_coefficient;

public: // Methods
    /**
     * @brief Builds a rod-to-cylinder contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     * @param velocity_damping_coefficient Damping in the slip direction. A
     *        large value approximates stiction.
     * @param friction_coefficient Coulomb friction coefficient.
     */
    RodCylinderContact(
        double k,
        double nu,
        double velocity_damping_coefficient,
        double friction_coefficient
    );

    /**
     * @brief Applies contact between a rod and a cylinder.
     * @tparam SystemOne Any @ref ContactableRod.
     * @tparam SystemTwo Any @ref ContactableCylinder.
     * @param system_one The rod.
     * @param system_two The cylinder.
     * @param time Current simulation time; unused.
     */
    template<ContactableRod SystemOne, ContactableCylinder SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        const Eigen::Vector3d center = system_two.positions().row(0).transpose();
        const Eigen::Matrix3d& frame = system_two.frames()[0];

        if (math::prune_rod_cylinder(
                system_one.positions(), system_one.radii(), system_one.lengths(),
                center, frame, system_two.radius(), system_two.length()))
        {
            return;
        }

        // The cylinder's axis in the lab frame, and the end face the segment
        // is measured from.
        const Eigen::Vector3d axis = frame.row(2).transpose();
        const Eigen::Vector3d tip = center - 0.5 * system_two.length() * axis;
        const Eigen::Vector3d edge = system_two.length() * axis;

        const Eigen::VectorXd radii_sum =
            system_one.radii().array() + system_two.radius();
        const Eigen::VectorXd lengths_sum =
            system_one.lengths().array() + system_two.length();

        Vector3DStack element_edge = system_one.tangents();
        element_edge.array().colwise() *= system_one.lengths().array();

        detail::contact_forces_rod_cylinder(
            detail::element_center_positions(system_one.positions()),
            element_edge, center, tip, edge, radii_sum, lengths_sum,
            system_one.internal_forces(), system_one.mutable_external_forces(),
            system_two.mutable_external_forces(),
            system_two.mutable_external_torques(), frame,
            system_one.velocities(),
            Eigen::Vector3d(system_two.velocities().row(0).transpose()),
            m_k, m_nu, m_velocity_damping_coefficient, m_friction_coefficient
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

/**
 * @brief Contact between a rod and a rigid sphere, with friction.
 *
 * The rod is always the first system and the sphere the second.
 *
 * @warning Applies force to both bodies but torque to neither, even though a
 *          contact off the sphere's centre has a moment arm. The reference
 *          does the same.
 */
class RodSphereContact
{
private: // Members
    double m_k;
    double m_nu;
    double m_velocity_damping_coefficient;
    double m_friction_coefficient;

public: // Methods
    /**
     * @brief Builds a rod-to-sphere contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     * @param velocity_damping_coefficient Damping in the slip direction.
     * @param friction_coefficient Coulomb friction coefficient.
     */
    RodSphereContact(
        double k,
        double nu,
        double velocity_damping_coefficient,
        double friction_coefficient
    );

    /**
     * @brief Applies contact between a rod and a sphere.
     * @tparam SystemOne Any @ref ContactableRod.
     * @tparam SystemTwo Any @ref ContactableSphere.
     * @param system_one The rod.
     * @param system_two The sphere.
     * @param time Current simulation time; unused.
     */
    template<ContactableRod SystemOne, ContactableSphere SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        const Eigen::Vector3d center = system_two.positions().row(0).transpose();

        if (math::prune_rod_sphere(
                system_one.positions(), system_one.radii(), center,
                system_two.radius()))
        {
            return;
        }

        const Eigen::VectorXd radii_sum =
            system_one.radii().array() + system_two.radius();
        const Eigen::VectorXd lengths_sum =
            system_one.lengths().array() + 2.0 * system_two.radius();

        Vector3DStack element_edge = system_one.tangents();
        element_edge.array().colwise() *= system_one.lengths().array();

        detail::contact_forces_rod_sphere(
            detail::element_center_positions(system_one.positions()),
            element_edge, center, radii_sum, lengths_sum,
            system_one.mutable_external_forces(),
            system_two.mutable_external_forces(), system_one.velocities(),
            Eigen::Vector3d(system_two.velocities().row(0).transpose()),
            m_k, m_nu, m_velocity_damping_coefficient, m_friction_coefficient
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

/**
 * @brief Contact between a rod and a plane, without friction.
 *
 * The rod is always the first system and the plane the second.
 */
class RodPlaneContact
{
private: // Members
    double m_k;
    double m_nu;

public: // Methods
    /**
     * @brief Builds a rod-to-plane contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     */
    RodPlaneContact(double k, double nu);

    /**
     * @brief Applies contact between a rod and a plane.
     * @tparam SystemOne Any @ref FrictionalRod, which supplies the node masses
     *         this needs beyond plain contact.
     * @tparam SystemTwo Any @ref ContactableSurface.
     * @param system_one The rod.
     * @param system_two The plane.
     * @param time Current simulation time; unused.
     */
    template<FrictionalRod SystemOne, ContactableSurface SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        detail::contact_forces_rod_plane(
            system_two.origin(), system_two.normal(), m_k, m_nu,
            system_one.radii(), system_one.masses(), system_one.positions(),
            system_one.velocities(), system_one.internal_forces(),
            system_one.mutable_external_forces()
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

/**
 * @brief Contact between a rod and a plane, with anisotropic friction.
 *
 * The only contact in this file that models friction, and the only one that
 * applies torque to a rod.
 */
class RodPlaneContactWithAnisotropicFriction
{
private: // Members
    double m_k;
    double m_nu;
    double m_slip_velocity_tol;
    Eigen::Vector3d m_static_mu;
    Eigen::Vector3d m_kinetic_mu;

public: // Methods
    /**
     * @brief Builds a rod-to-plane contact with friction.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     * @param slip_velocity_tol Speed at which an element counts as slipping;
     *        must be finite and positive.
     * @param static_mu Static friction coefficients, ordered forward,
     *        backward, sideways; each must be finite and non-negative.
     * @param kinetic_mu Kinetic friction coefficients, in the same order.
     */
    RodPlaneContactWithAnisotropicFriction(
        double k,
        double nu,
        double slip_velocity_tol,
        const Eigen::Vector3d& static_mu,
        const Eigen::Vector3d& kinetic_mu
    );

    /**
     * @brief Applies contact with friction between a rod and a plane.
     * @tparam SystemOne Any @ref FrictionalRod.
     * @tparam SystemTwo Any @ref ContactableSurface.
     * @param system_one The rod.
     * @param system_two The plane.
     * @param time Current simulation time; unused.
     */
    template<FrictionalRod SystemOne, ContactableSurface SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        detail::contact_forces_rod_plane_with_anisotropic_friction(
            system_two.origin(), system_two.normal(), m_slip_velocity_tol, m_k, m_nu,
            m_kinetic_mu(0), m_kinetic_mu(1), m_kinetic_mu(2),
            m_static_mu(0), m_static_mu(1), m_static_mu(2),
            system_one.radii(), system_one.masses(), system_one.tangents(),
            system_one.positions(), system_one.frames(), system_one.velocities(),
            system_one.angular_velocities(), system_one.internal_forces(),
            system_one.mutable_external_forces(), system_one.internal_torques(),
            system_one.mutable_external_torques()
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;

    /** @brief Static friction coefficients, forward, backward and sideways. */
    const Eigen::Vector3d& static_mu() const;

    /** @brief Kinetic friction coefficients, forward, backward and sideways. */
    const Eigen::Vector3d& kinetic_mu() const;
};

/**
 * @brief Contact between a rigid cylinder and a plane.
 *
 * The cylinder is always the first system and the plane the second.
 */
class CylinderPlaneContact
{
private: // Members
    double m_k;
    double m_nu;

public: // Methods
    /**
     * @brief Builds a cylinder-to-plane contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     */
    CylinderPlaneContact(double k, double nu);

    /**
     * @brief Applies contact between a cylinder and a plane.
     * @tparam SystemOne Any @ref ContactableCylinder.
     * @tparam SystemTwo Any @ref ContactableSurface.
     * @param system_one The cylinder.
     * @param system_two The plane.
     * @param time Current simulation time; unused.
     */
    template<ContactableCylinder SystemOne, ContactableSurface SystemTwo>
    void apply_contact(SystemOne& system_one, SystemTwo& system_two, double time) const
    {
        (void)time;
        detail::contact_forces_cylinder_plane(
            system_two.origin(), system_two.normal(), m_k, m_nu,
            system_one.length(), system_one.positions(), system_one.velocities(),
            system_one.mutable_external_forces()
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;
};

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

/** @brief Any one of the contact rules, held by value. */
using ContactVariant = std::variant<
    NoContact,
    RodRodContact,
    RodSelfContact,
    RodCylinderContact,
    RodSphereContact,
    RodPlaneContact,
    RodPlaneContactWithAnisotropicFriction,
    CylinderPlaneContact>;

/**
 * @brief Fails if the held contact cannot act between two given systems.
 *
 * Each rule constrains its entry point to the concepts it needs, so probing
 * whether the call is well formed decides compatibility without naming the
 * concepts here. Order matters: a rule expecting a rod and a cylinder will
 * reject the same pair given the other way round.
 *
 * @tparam SystemOne First system type.
 * @tparam SystemTwo Second system type.
 * @param contact_var Contact to check.
 * @param system_one First system.
 * @param system_two Second system.
 */
template<typename SystemOne, typename SystemTwo>
void validate(
    ContactVariant& contact_var, SystemOne& system_one, SystemTwo& system_two
)
{
    std::visit([&](const auto& contact)
    {
        const double time = 0.0;
        if constexpr (not requires {
            contact.apply_contact(system_one, system_two, time); })
        {
            utils::nice_assert(
                false, "Contact is incompatible with this pair of systems"
            );
        }
    }, contact_var);
}

/**
 * @brief Fails if the held contact cannot act on a single given system.
 *
 * The single-system form exists for self contact, where the two endpoints are
 * one body.
 *
 * @tparam SystemType System type.
 * @param contact_var Contact to check.
 * @param system The system.
 */
template<typename SystemType>
void validate(ContactVariant& contact_var, SystemType& system)
{
    std::visit([&](const auto& contact)
    {
        const double time = 0.0;
        if constexpr (not requires { contact.apply_contact(system, time); })
        {
            utils::nice_assert(
                false, "Contact is incompatible with this system"
            );
        }
    }, contact_var);
}

/**
 * @brief Applies the held contact between two systems.
 *
 * @tparam SystemOne First system type.
 * @tparam SystemTwo Second system type.
 * @param contact_var Contact to apply.
 * @param system_one First system.
 * @param system_two Second system.
 * @param time Current simulation time.
 */
template<typename SystemOne, typename SystemTwo>
void apply_contact(
    ContactVariant& contact_var,
    SystemOne& system_one,
    SystemTwo& system_two,
    double time
)
{
    std::visit([&](const auto& contact)
    {
        if constexpr (requires {
            contact.apply_contact(system_one, system_two, time); })
        {
            contact.apply_contact(system_one, system_two, time);
        }
        else
        {
            utils::nice_assert(
                false, "Contact is incompatible with this pair of systems"
            );
        }
    }, contact_var);
}

/**
 * @brief Applies the held contact to a single system.
 *
 * @tparam SystemType System type.
 * @param contact_var Contact to apply.
 * @param system The system.
 * @param time Current simulation time.
 */
template<typename SystemType>
void apply_contact(ContactVariant& contact_var, SystemType& system, double time)
{
    std::visit([&](const auto& contact)
    {
        if constexpr (requires { contact.apply_contact(system, time); })
        {
            contact.apply_contact(system, time);
        }
        else
        {
            utils::nice_assert(false, "Contact is incompatible with this system");
        }
    }, contact_var);
}
} // End namespace cosserat::physics
