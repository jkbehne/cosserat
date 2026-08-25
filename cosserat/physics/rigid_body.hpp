#pragma once

/**
 * @file rigid_body.hpp
 * @brief Rigid bodies that share the rod's state layout and load interface.
 *
 * A rigid body carries a single node and a single element, so every stack it
 * owns has exactly one row. That is deliberate: it lets the same force,
 * torque, damping, constraint and joint rules that act on a
 * @ref cosserat::physics::CosseratRod act on a rigid body without knowing
 * which it is holding. The reference implementation does the same thing for
 * the same reason.
 *
 * Unlike a rod, a rigid body has no internal forces or torques. Its shape is
 * fixed, so there is no strain to generate them, and
 * @ref RigidBody::compute_internal_forces_and_torques is a no-op kept only so
 * the two body types present the same interface to a time stepper.
 *
 * @ref RigidBody holds the state and the dynamics. @ref Sphere and
 * @ref Cylinder derive from it and add nothing but a constructor that works
 * out the mass and inertia for that shape.
 *
 * @note The derived shapes add no data members and no virtual functions, so
 *       there is no virtual destructor and none is needed. They are value
 *       types: copy or slice one to a @ref RigidBody freely, but never own one
 *       through a base pointer.
 *
 * This is a port of PyElastica's @c rigidbody module, merging its base, sphere
 * and cylinder into one translation unit.
 */

#include <cstdint>
#include <filesystem>
#include <numbers>
#include <string>

#include <Eigen/Core>

#include <cosserat/math/linalg.hpp>
#include <cosserat/math/types.hpp>

#include <cosserat/utils/assertions.hpp>
#include <cosserat/utils/file_utils.hpp>

namespace cosserat::physics {

/**
 * @brief A rigid body with a single node and a single element.
 *
 * State is stored in the same stacks a rod uses, each holding one row, so that
 * the load rules can treat the two interchangeably. The frame maps lab-frame
 * vectors into the body frame, exactly as a rod's element frames do.
 *
 * The body's shape enters only through its mass, volume and mass second moment
 * of inertia, all fixed at construction. @ref Sphere and @ref Cylinder work
 * those out for the two common cases; the constructor here takes them directly
 * for any other shape.
 */
class RigidBody
{
public: // Static constexpr members
    /** @brief Basic tolerance that can be applied to most situations */
    static constexpr double tolerance = 1e-12;

private: // Members
    // Every stack holds exactly one row: one node and one element.
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_accelerations;
    Vector3DStack m_internal_forces;
    Vector3DStack m_external_forces;
    Eigen::VectorXd m_masses;

    Matrix3DStack m_frames;
    Matrix3DStack m_mass_2nd_moments;
    Matrix3DStack m_inv_mass_2nd_moments;
    Vector3DStack m_angular_velocities;
    Vector3DStack m_angular_accelerations;
    Vector3DStack m_internal_torques;
    Vector3DStack m_external_torques;
    Eigen::VectorXd m_radii;
    Eigen::VectorXd m_densities;
    Eigen::VectorXd m_volumes;
    Eigen::VectorXd m_lengths;
    Eigen::VectorXd m_rest_lengths;
    Eigen::VectorXd m_dilatations;

public: // Methods
    /**
     * @brief Builds a rigid body from its shape's mass properties.
     *
     * Mass is the product of @p volume and @p density. The inertia matrix is
     * diagonal in the body frame, which covers every shape whose principal
     * axes align with its frame; the two provided shapes both qualify.
     *
     * @param position Centre of mass, in lab coordinates.
     * @param frame Body frame; must be orthogonal. Its rows are the body's
     *        normal, binormal and tangent directions.
     * @param radius Characteristic radius; finite and positive.
     * @param length Characteristic length along the tangent; finite and
     *        positive.
     * @param density Uniform density; finite and positive.
     * @param volume Enclosed volume; finite and positive.
     * @param inertia_diagonal Diagonal of the mass second moment of inertia in
     *        the body frame; every entry finite and positive.
     */
    RigidBody(
        Eigen::Vector3d position,
        Eigen::Matrix3d frame,
        double radius,
        double length,
        double density,
        double volume,
        Eigen::Vector3d inertia_diagonal
    );

    /**
     * @brief Does nothing; a rigid body has no internal loads.
     *
     * Present so a rigid body and a rod can be driven by the same stepper. The
     * internal force and torque stacks stay at zero for the body's whole life.
     *
     * @param time Current simulation time; unused.
     */
    void compute_internal_forces_and_torques(double time);

    /**
     * @brief Converts accumulated external loads into accelerations.
     *
     * @f[ \mathbf{a} = \frac{\mathbf{F}_{ext}}{m}, \qquad
     *     \pmb{\alpha} = \mathbf{J}^{-1}\left(
     *         \left(\mathbf{J}\pmb{\omega}\right) \times \pmb{\omega}
     *         + \pmb{\tau}_{ext}\right) @f]
     *
     * The cross term is the Lagrangian transport that keeps the angular
     * momentum equation correct in the rotating body frame. A rod carries the
     * same term inside its internal torques; a rigid body has none, so it
     * appears here instead.
     *
     * @param time Current simulation time; unused.
     * @param dt Timestep; unused.
     *
     * @note Unlike a rod, there is no dilatation factor: a rigid body cannot
     *       stretch, so its dilatation is identically one.
     */
    void update_accelerations(double time, double dt);

    /** @brief Clears the external force and torque accumulators. */
    void zero_out_external_forces_and_torques(double);

    /**
     * @brief Position of the centre of mass.
     * @return The single node's position.
     */
    Eigen::Vector3d position_center_of_mass() const;

    /**
     * @brief Translational kinetic energy.
     * @return @f$ \tfrac{1}{2} m \, \mathbf{v} \cdot \mathbf{v} @f$.
     */
    double translational_energy() const;

    /**
     * @brief Rotational kinetic energy.
     * @return @f$ \tfrac{1}{2} \pmb{\omega} \cdot \mathbf{J} \pmb{\omega} @f$,
     *         evaluated in the body frame.
     */
    double rotational_energy() const;

    /**
     * @brief Writes every stored stack as a binary and metadata pair.
     * @param write_path Directory to write into; created if missing.
     */
    void write_debug(const std::filesystem::path& write_path) const;

    /**
     * @brief Writes the configuration needed to reconstruct the body's pose.
     * @param write_path Directory to write into; created if missing.
     */
    void write(const std::filesystem::path& write_path) const;

private: // Assert methods
    /** @brief Fails unless the single frame is a rotation matrix. */
    void assert_frame_validity() const;

    /** @brief Fails unless every stack holds exactly one row. */
    void assert_is_proper() const;

public: // Constant accessors
    /**
     * @brief Number of nodes, always one.
     *
     * A rigid body has no discretisation. Both this and @ref num_elements
     * report one so that the rules written against a rod's domains apply
     * unchanged.
     */
    std::int64_t num_nodes() const;

    /** @brief Number of elements, always one. */
    std::int64_t num_elements() const;

    /** @brief Node position, one row. */
    const Vector3DStack& positions() const;
    /** @brief Node velocity, one row. */
    const Vector3DStack& velocities() const;
    /** @brief Node acceleration, one row. */
    const Vector3DStack& accelerations() const;
    /** @brief Node internal force; identically zero. */
    const Vector3DStack& internal_forces() const;
    /** @brief Node external force accumulator, one row. */
    const Vector3DStack& external_forces() const;
    /** @brief Node mass, one entry. */
    const Eigen::VectorXd& masses() const;
    /** @brief Body frame, one matrix. */
    const Matrix3DStack& frames() const;
    /** @brief Mass second moment of inertia, one matrix. */
    const Matrix3DStack& mass_2nd_moments() const;
    /** @brief Inverse mass second moment of inertia, one matrix. */
    const Matrix3DStack& inv_mass_2nd_moments() const;
    /** @brief Angular velocity in the body frame, one row. */
    const Vector3DStack& angular_velocities() const;
    /** @brief Angular acceleration in the body frame, one row. */
    const Vector3DStack& angular_accelerations() const;
    /** @brief Element internal torque; identically zero. */
    const Vector3DStack& internal_torques() const;
    /** @brief Element external torque accumulator, one row. */
    const Vector3DStack& external_torques() const;
    /** @brief Characteristic radius, one entry. */
    const Eigen::VectorXd& radii() const;
    /** @brief Density, one entry. */
    const Eigen::VectorXd& densities() const;
    /** @brief Volume, one entry. */
    const Eigen::VectorXd& volumes() const;
    /** @brief Characteristic length, one entry. */
    const Eigen::VectorXd& lengths() const;
    /**
     * @brief Rest length, one entry.
     *
     * A rigid body never deforms, so this is always equal to @ref lengths.
     * Provided because the load rules that work in terms of a rest
     * configuration expect it.
     */
    const Eigen::VectorXd& rest_lengths() const;
    /**
     * @brief Dilatation, one entry, always one.
     *
     * A rigid body cannot stretch. Provided so that rules written for a rod's
     * dilatation apply unchanged.
     */
    const Eigen::VectorXd& dilatations() const;

    /** @brief Unit tangent, taken as the third row of the frame. */
    Eigen::Vector3d tangent() const;
    /** @brief Total mass. */
    double total_mass() const;
    /** @brief Characteristic radius as a scalar. */
    double radius() const;
    /** @brief Characteristic length as a scalar. */
    double length() const;
    /** @brief Volume as a scalar. */
    double volume() const;
    /** @brief Density as a scalar. */
    double density() const;

public: // Mutable accessors
    /** @brief Node position, for integrators and boundary conditions. */
    Vector3DStack& mutable_positions();
    /** @brief Node velocity, for integrators and boundary conditions. */
    Vector3DStack& mutable_velocities();
    /** @brief Body frame, for integrators and boundary conditions. */
    Matrix3DStack& mutable_frames();
    /** @brief Angular velocity, for integrators and constraints. */
    Vector3DStack& mutable_angular_velocities();
    /** @brief External force accumulator, for force rules and joints. */
    Vector3DStack& mutable_external_forces();
    /** @brief External torque accumulator, for torque rules and joints. */
    Vector3DStack& mutable_external_torques();
};

/**
 * @brief A uniform solid sphere.
 *
 * Adds only a constructor. Mass follows from the sphere's volume and the
 * inertia is isotropic, @f$ \tfrac{2}{5} m r^2 @f$ about every axis, which is
 * exact for a uniform solid sphere.
 *
 * The body frame is the identity: normal along x, binormal along y and tangent
 * along z. A sphere has no preferred orientation, so any frame would do, and
 * this is the one the reference implementation picks.
 */
class Sphere : public RigidBody
{
public: // Methods
    /**
     * @brief Builds a sphere about a centre.
     * @param center Centre of the sphere, which is also its centre of mass.
     * @param base_radius Sphere radius; finite and positive.
     * @param density Uniform density; finite and positive.
     */
    Sphere(const Eigen::Vector3d& center, double base_radius, double density);
};

/**
 * @brief A uniform solid cylinder.
 *
 * Adds only a constructor. The body is positioned at its midpoint, half a
 * length along @p direction from @p start, and its frame has @p normal,
 * @p direction crossed with @p normal, and @p direction as its three rows.
 *
 * @warning The transverse inertia follows the reference implementation, which
 *          uses the cross-section's second moment of area scaled by density
 *          and length, giving @f$ m r^2 / 4 @f$. A uniform solid cylinder
 *          rotating about a transverse axis through its centre actually has
 *          @f$ m (3r^2 + L^2) / 12 @f$, so the @f$ m L^2 / 12 @f$ term is
 *          missing and a long cylinder is far too easy to tumble. The axial
 *          entry, @f$ m r^2 / 2 @f$, is correct. This is reproduced
 *          deliberately for parity; see the tests for the exact numbers.
 */
class Cylinder : public RigidBody
{
public: // Methods
    /**
     * @brief Builds a cylinder from one end face.
     * @param start Centre of the starting end face.
     * @param direction Unit vector along the cylinder's axis.
     * @param normal Unit vector orthogonal to @p direction, fixing the roll of
     *        the body frame.
     * @param base_length Cylinder length; finite and positive.
     * @param base_radius Cylinder radius; finite and positive.
     * @param density Uniform density; finite and positive.
     * @param tolerance Tolerance for the unit-vector and orthogonality checks
     *        on @p direction and @p normal.
     *
     * @note The reference implementation checks only that these vectors have
     *       three entries. Requiring them to be orthonormal is an addition,
     *       without which the body frame silently fails to be a rotation.
     */
    Cylinder(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& direction,
        const Eigen::Vector3d& normal,
        double base_length,
        double base_radius,
        double density,
        double tolerance = 1e-12
    );
};
} // End namespace cosserat::physics
