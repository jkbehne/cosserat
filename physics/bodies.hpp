#pragma once

/**
 * @file bodies.hpp
 * @brief A body of any kind, and a handle that steps it.
 *
 * Every body type this library simulates presents the same stepper interface,
 * so a time stepper should not need to know which one it is holding.
 * @ref BodyVariant erases that difference, and @ref BodyVariantWrapper turns
 * the variant into an object with the stepper's own vocabulary, dispatching
 * each call across the alternatives.
 *
 * The body is held by shared pointer because more than one thing refers to it:
 * the collection that owns the simulation, the rules attached to it, and any
 * joint or contact naming it as an endpoint. Copying a wrapper shares the body
 * rather than duplicating it, so two handles to one body stay in step.
 *
 * Unlike the force, damping, constraint and joint variants, dispatch here is
 * an unconditional @c std::visit rather than a probe for whether the call is
 * well formed. That is deliberate: a rule may reasonably be inapplicable to a
 * given body, but a body that cannot be stepped is not a body, so the
 * requirement belongs in the type system and a violation should be a compile
 * error rather than a runtime rejection.
 */

#include <memory>
#include <variant>

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

namespace cosserat::physics {

/**
 * @brief Any body the simulation can hold, stored by value.
 *
 * @note @ref Sphere and @ref Cylinder derive from @ref RigidBody and add no
 *       members, so they are distinguishable here only by which alternative is
 *       engaged. Every visitor therefore needs three arms that behave
 *       identically for the three rigid shapes, and @c std::get<RigidBody>
 *       will not find a @ref Sphere. Narrowing this to
 *       @c std::variant<CosseratRod, RigidBody> would lose nothing but the
 *       shape tag.
 */
using BodyVariant = std::variant<CosseratRod, RigidBody, Sphere, Cylinder>;

/** @brief Shared ownership of a body. */
using BodyVariantPtr = std::shared_ptr<BodyVariant>;

/**
 * @brief A handle that steps whichever body it points at.
 *
 * Each method forwards to the corresponding entry point on the engaged
 * alternative. The handle is copyable and every copy refers to the same body.
 *
 * @warning Moving from a wrapper leaves it holding nothing. Calling any method
 *          afterwards fails an assertion rather than dereferencing null.
 */
class BodyVariantWrapper
{
private: // Members
    BodyVariantPtr m_body;

private: // Methods
    /**
     * @brief The held body, failing if the handle is empty.
     * @return Reference to the held variant.
     */
    BodyVariant& variant();

public: // Methods
    /**
     * @brief Wraps a body.
     * @param body Body to step; must not be null.
     */
    explicit BodyVariantWrapper(BodyVariantPtr body);

    /**
     * @brief Advances the body's positions and frames by its current rates.
     * @param time Current simulation time.
     * @param scale Integration prefactor for this substage.
     */
    void update_kinematics(double time, double scale);

    /**
     * @brief Advances the body's rates by its current accelerations.
     * @param time Current simulation time.
     * @param scale Integration prefactor for this substage.
     */
    void update_dynamics(double time, double scale);

    /**
     * @brief Recomputes the body's accelerations from its accumulated loads.
     *
     * @param time Current simulation time.
     * @param dt Timestep. Note that this is the whole step rather than a
     *        substage prefactor, unlike @ref update_dynamics.
     */
    void update_accelerations(double time, double dt);

    /**
     * @brief Recomputes the body's internal forces and torques.
     *
     * A no-op for a rigid body, which has no internal loads.
     *
     * @param time Current simulation time.
     */
    void compute_internal_forces_and_torques(double time);

    /**
     * @brief Clears the body's external force and torque accumulators.
     *
     * @param time Current simulation time; accepted so the whole stepper
     *        interface takes a time, though no body needs it here.
     */
    void zero_out_external_forces_and_torques(double time);

    /**
     * @brief The held body, for applying rules or reading state.
     *
     * Rules, diagnostics and joints all reach the body through here, since the
     * stepping methods above deliberately expose nothing else.
     *
     * @return Reference to the held variant.
     */
    BodyVariant& body();

    /** @copydoc body() */
    const BodyVariant& body() const;

    /**
     * @brief Whether this handle points at the same body as another.
     *
     * Compares identity, not contents. Useful for a rule that must know
     * whether its two endpoints are one body, such as self-contact.
     *
     * @param other Handle to compare against.
     * @return True when both refer to the same body.
     */
    bool refers_to_same_body_as(const BodyVariantWrapper& other) const;
};
} // End namespace cosserat::physics
