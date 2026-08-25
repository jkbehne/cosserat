#pragma once

/**
 * @file rod_mesh_contact.hpp
 * @brief Contact between a rod and a rigid body described by a distance field.
 *
 * This is a deliberate departure from the reference implementation, which has
 * no mesh contact at all: every contact pair it offers needs a rod or an
 * infinite plane. Nothing here is a port, so nothing here has a reference to
 * be checked against. It is verified instead against analytic fields, where
 * the exact answer is known in closed form.
 *
 * The penalty law is the same one the other contact rules use, so a rod
 * touching a mesh behaves like a rod touching a cylinder:
 *
 * @f[ \gamma = r_e - d(\mathbf{x}), \qquad
 *     \mathbf{F} = \tfrac{1}{2} H(\gamma)
 *     \left(k\,\gamma + \nu\,(\mathbf{v}_{rel} \cdot \hat{\mathbf{n}})\right)
 *     \hat{\mathbf{n}} @f]
 *
 * @section rmc_no_tessellation The rod is never tessellated
 *
 * This is not mesh against mesh. The representation is asymmetric: the mesh
 * becomes a field, and the rod stays a chain of capsules. Because a signed
 * distance function measures distance to the surface from anywhere, the
 * clearance of a capsule is the clearance of its axis segment minus its
 * radius, and no rod geometry ever has to be built.
 *
 * Tessellating the rod would be worse in three separate ways. The rod deforms
 * every step, so its acceleration structure would need rebuilding a million
 * times over a run, which is exactly the cost a field avoids by being built
 * once. Triangle against triangle is far more expensive than a point query.
 * And it would discard the per-element radius, which for a volume preserving
 * rod varies along the rod and changes as it stretches.
 *
 * @section rmc_marching Why the segment is marched rather than sampled
 *
 * Testing an element at a fixed handful of points can miss a feature that
 * passes between the samples. A true signed distance function is 1-Lipschitz,
 * changing no faster than the distance travelled, so from a point at distance
 * @f$ d @f$ no surface can be reached within @f$ d - r @f$ of it. Advancing by
 * that amount therefore cannot skip a contact. This is sphere tracing, from
 * Hart, *Sphere tracing: a geometric method for the antialiased ray tracing of
 * implicit surfaces*, The Visual Computer 12(10), 1996.
 *
 * A discretised field is only approximately 1-Lipschitz, since interpolation
 * can overshoot slightly near features, so the step carries the safety factor
 * @ref march_safety_factor. Once inside a contact region the march switches to
 * fixed fine steps to find the deepest point rather than merely the first.
 *
 * @section rmc_frames Where the query happens
 *
 * The field belongs to the body and is expressed in the body's own frame, so
 * it never needs rebuilding as the body moves. Each rod point is carried into
 * that frame, queried, and the resulting normal carried back out. Distance is
 * invariant under rigid motion, so this is exact rather than an approximation.
 */

#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cosserat/math/signed_distance_field.hpp>
#include <cosserat/math/types.hpp>

namespace cosserat::physics {

/** @brief Penetration below which a pair is treated as separated. */
inline constexpr double mesh_separation_tolerance = -1e-5;

/**
 * @brief Gradient magnitude below which the contact normal is unusable.
 *
 * A distance field has unit gradient wherever it is differentiable. It fails
 * to on the medial axis, where the nearest surface point is not unique: the
 * centre of a sphere, the spine of a tube. There is no meaningful direction to
 * push along there, so such a sample is skipped rather than normalised.
 */
inline constexpr double mesh_gradient_tolerance = 1e-6;

/**
 * @brief Fraction of the safe distance actually stepped while marching.
 *
 * One would be exact for a true distance function. Interpolated fields can
 * overshoot slightly near sharp features, so the step is shortened to keep the
 * no-skipping guarantee in practice.
 */
inline constexpr double march_safety_factor = 0.9;

/**
 * @brief Step, as a fraction of element length, taken inside a contact region.
 *
 * Sphere tracing finds the first point of contact. Scanning on from there at a
 * fixed rate is what finds the deepest one, which is the point the force
 * should act at.
 */
inline constexpr double march_fine_step = 0.05;

/**
 * @brief Smallest step the march may take, as a fraction of element length.
 *
 * Guards against stalling where the safe distance collapses toward zero.
 */
inline constexpr double march_minimum_step = 1e-3;

/**
 * @brief A rigid body whose geometry is given by a distance field.
 *
 * Structurally a rigid body plus a field. The accessors mirror
 * @ref ContactableCylinder so that the two kinds of obstacle can be driven the
 * same way.
 *
 * @tparam T Candidate body type.
 */
template<typename T>
concept ContactableMeshBody = requires(T body)
{
    {body.positions()} -> std::same_as<const Vector3DStack&>;
    {body.velocities()} -> std::same_as<const Vector3DStack&>;
    {body.angular_velocities()} -> std::same_as<const Vector3DStack&>;
    {body.frames()} -> std::same_as<const Matrix3DStack&>;
    {body.field_query()} -> std::same_as<math::FieldQuery>;
    {body.field_domain()} -> std::same_as<Eigen::AlignedBox3d>;
    {body.mutable_external_forces()} -> std::same_as<Vector3DStack&>;
    {body.mutable_external_torques()} -> std::same_as<Vector3DStack&>;
};

namespace detail {

/**
 * @brief One element's closest approach to a field, found by marching.
 */
struct MeshContactSample
{
public: // Members
    /** @brief Whether any point of the element is in contact. */
    bool in_contact = false;

    /** @brief Parameter along the element, between zero and one. */
    double parameter = 0.0;

    /** @brief Deepest penetration found, as radius minus distance. */
    double penetration = 0.0;

    /** @brief Point on the element axis at the deepest penetration. */
    Eigen::Vector3d point = Eigen::Vector3d::Zero();

    /** @brief Outward unit normal there, from the field's gradient. */
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();

    /** @brief Number of field queries the march consumed. */
    std::int64_t queries = 0;
};

/**
 * @brief Marches one element against a field and returns the deepest contact.
 *
 * Steps conservatively while clear of the surface and finely once within a
 * contact region, so that a feature cannot be skipped and the point reported
 * is the deepest rather than the first. See @ref rmc_marching.
 *
 * @param query The field, expressed in the same frame as the endpoints.
 * @param start Element start, in field coordinates.
 * @param finish Element end, in field coordinates.
 * @param radius Element radius.
 * @return The deepest contact found, or a sample with @c in_contact false.
 */
MeshContactSample march_element_against_field(
    const math::FieldQuery& query,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& finish,
    double radius
);

/**
 * @brief Contact forces and torques between a rod and a distance field.
 *
 * The rod's nodes are carried into the body frame, each element is marched
 * against the field, and the resulting force is split onto the element's two
 * nodes with the same end weighting the other contact kernels use. The body
 * receives the equal and opposite force at the contact point together with the
 * moment it produces about the body's centre.
 *
 * @param positions Node positions of the rod, in the lab frame.
 * @param radii Element radii of the rod.
 * @param velocities Node velocities of the rod, in the lab frame.
 * @param external_forces_rod Node external force accumulator of the rod.
 * @param query The body's field, in body coordinates.
 * @param domain The field's domain, in body coordinates.
 * @param body_center Position of the body, in the lab frame.
 * @param body_frame Body frame, mapping lab vectors into body coordinates.
 * @param body_velocity Linear velocity of the body, in the lab frame.
 * @param body_angular_velocity Angular velocity of the body, in body
 *        coordinates as the rigid body stores it.
 * @param external_forces_body External force accumulator of the body.
 * @param external_torques_body External torque accumulator of the body, in
 *        body coordinates.
 * @param contact_k Contact spring constant.
 * @param contact_nu Contact damping constant.
 * @param velocity_damping_coefficient Damping in the slip direction.
 * @param friction_coefficient Coulomb friction coefficient.
 * @return Total number of field queries performed, for profiling.
 */
std::int64_t contact_forces_rod_field(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces_rod,
    const math::FieldQuery& query,
    const Eigen::AlignedBox3d& domain,
    const Eigen::Vector3d& body_center,
    const Eigen::Matrix3d& body_frame,
    const Eigen::Vector3d& body_velocity,
    const Eigen::Vector3d& body_angular_velocity,
    Vector3DStack& external_forces_body,
    Vector3DStack& external_torques_body,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
);

} // End namespace detail

/**
 * @brief Contact between a rod and a mesh body, with friction.
 *
 * The rod is always the first system and the mesh body the second.
 */
class RodMeshContact
{
private: // Members
    double m_k;
    double m_nu;
    double m_velocity_damping_coefficient;
    double m_friction_coefficient;

public: // Methods
    /**
     * @brief Builds a rod-to-mesh contact.
     * @param k Contact spring constant; must be finite and positive.
     * @param nu Contact damping constant; must be finite and non-negative.
     * @param velocity_damping_coefficient Damping in the slip direction. A
     *        large value approximates stiction.
     * @param friction_coefficient Coulomb friction coefficient.
     */
    RodMeshContact(
        double k,
        double nu,
        double velocity_damping_coefficient,
        double friction_coefficient
    );

    /**
     * @brief Applies contact between a rod and a mesh body.
     * @tparam SystemOne Any rod exposing positions, radii, velocities and a
     *         force accumulator.
     * @tparam SystemTwo Any @ref ContactableMeshBody.
     * @param system_one The rod.
     * @param system_two The mesh body.
     * @param time Current simulation time; unused.
     */
    template<typename SystemOne, ContactableMeshBody SystemTwo>
    requires requires(SystemOne rod)
    {
        {rod.positions()} -> std::same_as<const Vector3DStack&>;
        {rod.velocities()} -> std::same_as<const Vector3DStack&>;
        {rod.radii()} -> std::same_as<const Eigen::VectorXd&>;
        {rod.mutable_external_forces()} -> std::same_as<Vector3DStack&>;
    }
    void apply_contact(
        SystemOne& system_one, SystemTwo& system_two, [[maybe_unused]] double time
    ) const
    {
        detail::contact_forces_rod_field(
            system_one.positions(),
            system_one.radii(),
            system_one.velocities(),
            system_one.mutable_external_forces(),
            system_two.field_query(),
            system_two.field_domain(),
            Eigen::Vector3d(system_two.positions().row(0).transpose()),
            system_two.frames()[0],
            Eigen::Vector3d(system_two.velocities().row(0).transpose()),
            Eigen::Vector3d(system_two.angular_velocities().row(0).transpose()),
            system_two.mutable_external_forces(),
            system_two.mutable_external_torques(),
            m_k, m_nu, m_velocity_damping_coefficient, m_friction_coefficient
        );
    }

    /** @brief Contact spring constant. */
    double k() const;

    /** @brief Contact damping constant. */
    double nu() const;

    /** @brief Damping in the slip direction. */
    double velocity_damping_coefficient() const;

    /** @brief Coulomb friction coefficient. */
    double friction_coefficient() const;
};
} // End namespace cosserat::physics
