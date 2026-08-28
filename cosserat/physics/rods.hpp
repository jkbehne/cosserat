#pragma once

/**
 * @file rods.hpp
 * @brief Cosserat rod state and the internal dynamics computed from it.
 *
 * A rod is discretised into three interleaved domains, and every stored
 * quantity belongs to exactly one of them:
 *
 * - @b Nodes carry position, velocity, acceleration, mass and the force
 *   accumulators. There are @c num_nodes of them.
 * - @b Elements span the gap between adjacent nodes and carry the frame,
 *   inertia, shear stiffness, angular rates, tangent, shear strain and the
 *   torque accumulators. There are @c num_nodes - 1 of them.
 * - @b Voronoi regions span the gap between adjacent elements and carry the
 *   bending stiffness, curvature and internal couple. There are
 *   @c num_nodes - 2 of them.
 *
 * Every stack stores one row per entity, so a per-element vector quantity is
 * an @c (num_elements, 3) matrix rather than the @c (3, num_elements) layout
 * the reference implementation uses. Frames are stored as a vector of 3 by 3
 * rotation matrices mapping lab-frame vectors into the material frame.
 *
 * The governing equations follow Gazzola et al., RSoS (2018), and this is a
 * direct port of PyElastica's @c cosserat_rod module.
 *
 * @note Energy and centre-of-mass diagnostics are deliberately absent. They
 *       are pure functions of the state exposed by the const accessors and are
 *       not used anywhere in the dynamics, so they belong in a separate header.
 */

#include <cstdint>
#include <filesystem>
#include <numbers>
#include <string>

#include <Eigen/Core>

#include <cosserat/math/finite_difference.hpp>
#include <cosserat/math/indexing.hpp>
#include <cosserat/math/linalg.hpp>
#include <cosserat/math/types.hpp>

#include <cosserat/utils/assertions.hpp>
#include <cosserat/utils/file_utils.hpp>

namespace cosserat::physics {

/**
 * @brief A discretised Cosserat rod and its internal dynamics.
 *
 * The class owns the full state and recomputes every derived quantity from
 * position, frame and velocity whenever
 * @ref compute_internal_forces_and_torques is called. External loads are
 * accumulated through the mutable accessors and cleared by
 * @ref zero_out_external_forces_and_torques.
 */
class CosseratRod
{
public: // Static constexpr members
    /** @brief Basic tolerance that can be applied to most situations */
    static constexpr double tolerance = 1e-12;

    /**
     * @brief Timoshenko shear correction factor for a circular cross-section.
     *
     * Value pulled for Poisson ratio of 0.5.
     * @see https://iopscience.iop.org/article/10.1088/0022-3727/8/16/003
     */
    static constexpr double alpha_c = 27.0 / 28.0;

public: // Types
    /**
     * @brief Cross-sectional area and second moments of area, per element.
     *
     * Both are purely geometric, free of density and length. The mass second
     * moment of inertia is @ref I0 scaled by density times rest length, and is
     * stored on the rod rather than returned here, because the stiffness
     * matrices need the unscaled form.
     */
    struct Mass2ndMomentResult
    {
    public: // Members
        /** @brief Cross-sectional area of each element. */
        Eigen::VectorXd A0;

        /**
         * @brief Second moments of area of each element.
         *
         * The first two columns are the bending moments and the third is the
         * polar moment, which is their sum for a circular section.
         */
        Vector3DStack I0;
    };

private: // Static methods
    /**
     * @brief Fails unless a vector has the expected length and exceeds a bound.
     * @param vector Vector to check.
     * @param exp_length Required number of entries.
     * @param lower_bound Value every entry must strictly exceed.
     * @param name Name reported in the failure message.
     */
    static void assert_size_and_lower_bound(
        const Eigen::VectorXd& vector,
        std::int64_t exp_length,
        double lower_bound,
        const std::string& name
    );

    /**
     * @brief Builds the per-element shear and stretch stiffness matrices.
     *
     * Diagonal entries are @f$ \alpha_c G A_0 @f$ for the two shear directions
     * and @f$ E A_0 @f$ for stretch along the tangent.
     *
     * @param youngs_modulus Young's modulus.
     * @param shear_modulus Shear modulus.
     * @param A0 Cross-sectional area of each element.
     * @return One shear matrix per element.
     */
    static Matrix3DStack build_shearing_matrices(
        double youngs_modulus, double shear_modulus, const Eigen::VectorXd& A0
    );

    /**
     * @brief Builds the per-element bend and twist stiffness matrices.
     *
     * Diagonal entries are @f$ E I_1 @f$ and @f$ E I_2 @f$ for bending and
     * @f$ G I_3 @f$ for twist.
     *
     * @param youngs_modulus Young's modulus.
     * @param shear_modulus Shear modulus.
     * @param I0 Second moments of area, unscaled by density or length.
     * @return One bending matrix per element, before Voronoi averaging.
     */
    static Matrix3DStack build_bending_matrices(
        double youngs_modulus, double shear_modulus, const Vector3DStack& I0
    );

private: // Members
    std::int64_t m_num_nodes;
    std::int64_t m_num_elements;
    std::int64_t m_num_voronoi;

    // All should have num_nodes entries
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_accelerations;
    Vector3DStack m_internal_forces;
    Vector3DStack m_external_forces;
    Eigen::VectorXd m_masses;

    // All should have num_elements entries
    Matrix3DStack m_frames;
    Matrix3DStack m_mass_2nd_moments;
    Matrix3DStack m_inv_mass_2nd_moments;
    Matrix3DStack m_shearing_matrices;
    Vector3DStack m_angular_velocities;
    Vector3DStack m_angular_accelerations;
    Vector3DStack m_internal_torques;
    Vector3DStack m_external_torques;
    Vector3DStack m_tangents;
    Vector3DStack m_sigmas;
    Vector3DStack m_rest_sigmas;
    Vector3DStack m_internal_stresses;
    Eigen::VectorXd m_radii;
    Eigen::VectorXd m_densities;
    Eigen::VectorXd m_volumes;
    Eigen::VectorXd m_lengths;
    Eigen::VectorXd m_rest_lengths;
    Eigen::VectorXd m_dilatations;
    Eigen::VectorXd m_dilatation_rates;

    // All should have num_voronoi entries
    Matrix3DStack m_bending_matrices;
    Vector3DStack m_kappas;
    Vector3DStack m_rest_kappas;
    Vector3DStack m_internal_couples;
    Eigen::VectorXd m_voronoi_dilatations;
    Eigen::VectorXd m_voronoi_rest_lengths;

    bool m_respect_radii;

public: // Methods
    /**
     * @brief Builds a rod from a complete, already-computed state.
     *
     * Nothing is derived; every stack is taken as given and only checked for
     * size and basic validity. Intended for restoring a rod from disk or for
     * tests that need an exact configuration.
     *
     * @warning The rod is left in radius-preserving mode, so the first compute
     *          pass recomputes volumes from radii and discards the volumes
     *          supplied here. Supply radii consistent with the volumes you
     *          want, or use one of the deriving constructors.
     *
     * @param num_nodes Number of nodes; element and Voronoi counts follow.
     * @param positions Node positions.
     * @param velocities Node velocities.
     * @param accelerations Node accelerations.
     * @param internal_forces Node internal forces.
     * @param external_forces Node external force accumulator.
     * @param masses Node masses.
     * @param frames Element frames.
     * @param mass_2nd_moments Element mass second moments of inertia.
     * @param inv_mass_2nd_moments Element inverse mass second moments.
     * @param bending_matrices Voronoi bend and twist stiffness matrices.
     * @param shearing_matrices Element shear and stretch stiffness matrices.
     * @param angular_velocities Element angular velocities, material frame.
     * @param angular_accelerations Element angular accelerations.
     * @param internal_torques Element internal torques.
     * @param external_torques Element external torque accumulator.
     * @param tangents Element unit tangents.
     * @param sigmas Element shear and stretch strains.
     * @param rest_sigmas Element rest shear and stretch strains.
     * @param internal_stresses Element internal stresses.
     * @param radii Element radii.
     * @param densities Element densities.
     * @param volumes Element volumes.
     * @param lengths Element current lengths.
     * @param rest_lengths Element rest lengths.
     * @param dilatations Element dilatations.
     * @param dilatation_rates Element dilatation rates.
     * @param kappas Voronoi curvatures.
     * @param rest_kappas Voronoi rest curvatures.
     * @param internal_couples Voronoi internal couples.
     * @param voronoi_dilatations Voronoi dilatations.
     * @param voronoi_rest_lengths Voronoi rest lengths.
     */
    CosseratRod(
        std::int64_t num_nodes,
        Vector3DStack positions,
        Vector3DStack velocities,
        Vector3DStack accelerations,
        Vector3DStack internal_forces,
        Vector3DStack external_forces,
        Eigen::VectorXd masses,
        Matrix3DStack frames,
        Matrix3DStack mass_2nd_moments,
        Matrix3DStack inv_mass_2nd_moments,
        Matrix3DStack bending_matrices,
        Matrix3DStack shearing_matrices,
        Vector3DStack angular_velocities,
        Vector3DStack angular_accelerations,
        Vector3DStack internal_torques,
        Vector3DStack external_torques,
        Vector3DStack tangents,
        Vector3DStack sigmas,
        Vector3DStack rest_sigmas,
        Vector3DStack internal_stresses,
        Eigen::VectorXd radii,
        Eigen::VectorXd densities,
        Eigen::VectorXd volumes,
        Eigen::VectorXd lengths,
        Eigen::VectorXd rest_lengths,
        Eigen::VectorXd dilatations,
        Eigen::VectorXd dilatation_rates,
        Vector3DStack kappas,
        Vector3DStack rest_kappas,
        Vector3DStack internal_couples,
        Eigen::VectorXd voronoi_dilatations,
        Eigen::VectorXd voronoi_rest_lengths
    );

    /**
     * @brief Builds a rod from geometry and user-supplied stiffness matrices.
     *
     * Inertia, mass, strains and the zero dynamics are all derived. Bending
     * matrices are averaged onto the Voronoi domain internally, so pass one
     * per element.
     *
     * @param positions Node positions; at least three nodes.
     * @param frames Element frames; each must be orthogonal.
     * @param bending_matrices One bend and twist matrix per element.
     * @param shearing_matrices One shear and stretch matrix per element.
     * @param rest_sigmas Rest shear strains, or an empty stack for zero.
     * @param densities Element densities.
     * @param volumes_or_radii Element volumes or radii, selected by
     *        @p respect_radii.
     * @param rest_lengths Element rest lengths.
     * @param rest_kappas Rest curvatures, or an empty stack for zero.
     * @param respect_radii See @ref respect_radii for what this controls.
     */
    CosseratRod(
        Vector3DStack positions,
        Matrix3DStack frames,
        Matrix3DStack bending_matrices,
        Matrix3DStack shearing_matrices,
        Vector3DStack rest_sigmas,
        Eigen::VectorXd densities,
        Eigen::VectorXd volumes_or_radii,
        Eigen::VectorXd rest_lengths,
        Vector3DStack rest_kappas,
        bool respect_radii
    );

    /**
     * @brief Builds a rod from geometry and an isotropic material.
     *
     * Stiffness matrices are derived from the moduli and the cross-section.
     *
     * @param positions Node positions; at least three nodes.
     * @param frames Element frames; each must be orthogonal.
     * @param rest_sigmas Rest shear strains, or an empty stack for zero.
     * @param densities Element densities.
     * @param volumes_or_radii Element volumes or radii, selected by
     *        @p respect_radii.
     * @param rest_lengths Element rest lengths.
     * @param rest_kappas Rest curvatures, or an empty stack for zero.
     * @param respect_radii See @ref respect_radii for what this controls.
     * @param youngs_modulus Young's modulus.
     * @param shear_modulus Shear modulus.
     */
    CosseratRod(
        Vector3DStack positions,
        Matrix3DStack frames,
        Vector3DStack rest_sigmas,
        Eigen::VectorXd densities,
        Eigen::VectorXd volumes_or_radii,
        Eigen::VectorXd rest_lengths,
        Vector3DStack rest_kappas,
        bool respect_radii,
        double youngs_modulus,
        double shear_modulus
    );

    /**
     * @brief Builds a rod assuming a Poisson ratio of one half.
     *
     * Delegates with @f$ G = E / 3 @f$, matching the reference implementation's
     * default when no shear modulus is supplied.
     *
     * @param positions Node positions; at least three nodes.
     * @param frames Element frames; each must be orthogonal.
     * @param rest_sigmas Rest shear strains, or an empty stack for zero.
     * @param densities Element densities.
     * @param volumes_or_radii Element volumes or radii, selected by
     *        @p respect_radii.
     * @param rest_lengths Element rest lengths.
     * @param rest_kappas Rest curvatures, or an empty stack for zero.
     * @param respect_radii See @ref respect_radii for what this controls.
     * @param youngs_modulus Young's modulus.
     */
    CosseratRod(
        Vector3DStack positions,
        Matrix3DStack frames,
        Vector3DStack rest_sigmas,
        Eigen::VectorXd densities,
        Eigen::VectorXd volumes_or_radii,
        Eigen::VectorXd rest_lengths,
        Vector3DStack rest_kappas,
        bool respect_radii,
        double youngs_modulus
    );

    /**
     * @brief Recomputes internal forces and torques from the current state.
     *
     * Every derived quantity is refreshed along the way: geometry, dilatations,
     * strains, stresses and dilatation rates. Call this before
     * @ref update_accelerations on each step.
     *
     * @param time Current simulation time; unused, present for interface
     *        symmetry with the force and damping rules.
     */
    void compute_internal_forces_and_torques(double time);

    /**
     * @brief Converts accumulated forces and torques into accelerations.
     *
     * @f[ \mathbf{a} = \frac{\mathbf{F}_{int} + \mathbf{F}_{ext}}{m}, \qquad
     *     \pmb{\alpha} = e \, \mathbf{J}^{-1}
     *     \left(\pmb{\tau}_{int} + \pmb{\tau}_{ext}\right) @f]
     *
     * @param time Current simulation time; unused.
     * @param dt Timestep; unused.
     *
     * @note Reads the internal forces and torques cached by
     *       @ref compute_internal_forces_and_torques, so call that first.
     */
    void update_accelerations(double time, double dt);

    /** @brief Clears the external force and torque accumulators. */
    void zero_out_external_forces_and_torques(double);

    /**
     * @brief Writes every stored stack as a binary and metadata pair.
     * @param write_path Directory to write into; created if missing.
     */
    void write_debug(const std::filesystem::path& write_path) const;

    /**
     * @brief Writes the configuration needed to reconstruct the rod's shape.
     * @param write_path Directory to write into; created if missing.
     */
    void write(const std::filesystem::path& write_path) const;

private: // Build methods
    /**
     * @brief Populates geometry and the strains derived from it.
     *
     * Runs the compute chain once, so lengths, tangents, whichever of volume
     * or radius is derived, dilatations, shear strains and curvatures are all
     * valid on return.
     */
    void build_geometry(
        Vector3DStack positions,
        Eigen::VectorXd volumes_or_radii,
        Eigen::VectorXd densities,
        Matrix3DStack frames,
        Eigen::VectorXd rest_lengths,
        bool respect_radii
    );

    /**
     * @brief Fills the mass second moments and returns the geometric moments.
     *
     * The stored inertia is the second moment of area scaled by density and
     * rest length; the returned result carries the unscaled area and second
     * moments, which is what the stiffness matrices need.
     */
    Mass2ndMomentResult build_mass_2nd_moments();

    /** @brief Stores shear matrices and averages bending onto the Voronoi domain. */
    void build_shearing_bending_matrices(
        Matrix3DStack shearing_matrices, Matrix3DStack bending_matrices
    );

    /** @brief Derives both stiffness families from the moduli and geometry. */
    void build_shearing_bending_matrices(
        const Mass2ndMomentResult& mass_result,
        double youngs_modulus,
        double shear_modulus
    );

    /** @brief Lumps element mass onto the nodes, halving at each end. */
    void build_mass();

    /** @brief Averages adjacent rest lengths onto the Voronoi domain. */
    void build_voronoi_rest_lengths();

    /** @brief Stores rest strains, defaulting an empty stack to zero. */
    void build_rest_state(Vector3DStack rest_sigmas, Vector3DStack rest_kappas);

    /** @brief Zeroes every rate and accumulator. */
    void build_zero_dynamics();

private: // Assert methods
    /** @brief Fails unless every frame is a correctly sized rotation matrix. */
    void assert_frame_validity() const;

    /** @brief Fails unless radii, volumes and densities are sized and positive. */
    void assert_volume_radii_density_validity() const;

    /** @brief Fails unless rest lengths are sized and positive. */
    void assert_rest_lengths_validity() const;

    /** @brief Fails unless every stack matches its domain's entry count. */
    void assert_is_proper() const;

private: // Compute methods
    //
    // These form a chain, each calling the one above it, so the only entry
    // points that should be called directly are compute_internal_forces and
    // compute_internal_torques. Invoking a lower link on its own leaves the
    // quantities produced by the links above it stale.
    //
    //   compute_internal_forces
    //     -> compute_internal_shear_stretch_stresses_from_model
    //          -> compute_shear_stretch_strains
    //               -> compute_dilatations
    //                    -> compute_geometry
    //
    //   compute_internal_torques
    //     -> compute_internal_bending_twist_stresses_from_model
    //          -> compute_bending_twist_strains
    //     -> compute_dilatation_rates
    //
    // compute_internal_torques reads lengths and dilatations produced by the
    // forces pass, which is why compute_internal_forces_and_torques runs the
    // two in that order.
    //

    /** @brief Updates lengths, tangents, and whichever of volume or radius is derived. */
    void compute_geometry();

    /** @brief Updates element and Voronoi dilatations. */
    void compute_dilatations();

    /** @brief Updates the shear and stretch strain from frames and tangents. */
    void compute_shear_stretch_strains();

    /** @brief Updates curvature from the rotation between adjacent frames. */
    void compute_bending_twist_strains();

    /** @brief Updates node internal forces from the shear and stretch stress. */
    void compute_internal_forces();

    /** @brief Applies the shear stiffness to the shear and stretch strain. */
    void compute_internal_shear_stretch_stresses_from_model();

    /** @brief Updates element internal torques from all five couple terms. */
    void compute_internal_torques();

    /** @brief Applies the bending stiffness to the curvature. */
    void compute_internal_bending_twist_stresses_from_model();

    /** @brief Updates dilatation rates from node positions and velocities. */
    void compute_dilatation_rates();

public: // Constant accessors
    /** @brief Number of nodes. */
    std::int64_t num_nodes() const;
    /** @brief Number of elements, one fewer than the nodes. */
    std::int64_t num_elements() const;
    /** @brief Number of Voronoi regions, two fewer than the nodes. */
    std::int64_t num_voronoi() const;
    /** @brief Node positions. */
    const Vector3DStack& positions() const;
    /** @brief Node velocities. */
    const Vector3DStack& velocities() const;
    /** @brief Node accelerations. */
    const Vector3DStack& accelerations() const;
    /** @brief Node internal forces. */
    const Vector3DStack& internal_forces() const;
    /** @brief Node external force accumulator. */
    const Vector3DStack& external_forces() const;
    /** @brief Node masses. */
    const Eigen::VectorXd& masses() const;
    /** @brief Element frames, mapping lab vectors into the material frame. */
    const Matrix3DStack& frames() const;
    /** @brief Element mass second moments of inertia. */
    const Matrix3DStack& mass_2nd_moments() const;
    /** @brief Element inverse mass second moments of inertia. */
    const Matrix3DStack& inv_mass_2nd_moments() const;
    /** @brief Voronoi bend and twist stiffness matrices. */
    const Matrix3DStack& bending_matrices() const;
    /** @brief Element shear and stretch stiffness matrices. */
    const Matrix3DStack& shearing_matrices() const;
    /** @brief Element angular velocities in the material frame. */
    const Vector3DStack& angular_velocities() const;
    /** @brief Element angular accelerations in the material frame. */
    const Vector3DStack& angular_accelerations() const;
    /** @brief Element internal torques. */
    const Vector3DStack& internal_torques() const;
    /** @brief Element external torque accumulator. */
    const Vector3DStack& external_torques() const;
    /** @brief Element unit tangents. */
    const Vector3DStack& tangents() const;
    /** @brief Element shear and stretch strains. */
    const Vector3DStack& sigmas() const;
    /** @brief Element rest shear and stretch strains. */
    const Vector3DStack& rest_sigmas() const;
    /** @brief Element internal stresses. */
    const Vector3DStack& internal_stresses() const;
    /** @brief Element radii. */
    const Eigen::VectorXd& radii() const;
    /** @brief Element densities. */
    const Eigen::VectorXd& densities() const;
    /** @brief Element volumes. */
    const Eigen::VectorXd& volumes() const;
    /** @brief Element current lengths. */
    const Eigen::VectorXd& lengths() const;
    /** @brief Element rest lengths. */
    const Eigen::VectorXd& rest_lengths() const;
    /** @brief Element dilatations, current length over rest length. */
    const Eigen::VectorXd& dilatations() const;
    /** @brief Element dilatation rates. */
    const Eigen::VectorXd& dilatation_rates() const;
    /** @brief Voronoi curvatures. */
    const Vector3DStack& kappas() const;
    /** @brief Voronoi rest curvatures. */
    const Vector3DStack& rest_kappas() const;
    /** @brief Voronoi internal couples. */
    const Vector3DStack& internal_couples() const;
    /** @brief Voronoi dilatations. */
    const Eigen::VectorXd& voronoi_dilatations() const;
    /** @brief Voronoi rest lengths. */
    const Eigen::VectorXd& voronoi_rest_lengths() const;

    /**
     * @brief Whether radius or volume is held fixed as the rod deforms.
     *
     * When true the radius is held and volume is recomputed from the current
     * length, so the rod is compressible. When false the volume is held and
     * the radius shrinks as the rod stretches, which is the incompressible
     * behaviour of the reference implementation.
     */
    bool respect_radii() const;

public: // Mutable accessors
    /** @brief Node positions, for integrators and boundary conditions. */
    Vector3DStack& mutable_positions();
    /** @brief Node velocities, for integrators and boundary conditions. */
    Vector3DStack& mutable_velocities();
    /** @brief Element frames, for integrators and boundary conditions. */
    Matrix3DStack& mutable_frames();
    /** @brief Element angular velocities, for integrators and constraints. */
    Vector3DStack& mutable_angular_velocities();
    /** @brief Node external force accumulator, for force rules and joints. */
    Vector3DStack& mutable_external_forces();
    /** @brief Element external torque accumulator, for torque rules and joints. */
    Vector3DStack& mutable_external_torques();
    /** @brief Internal (voronoi) element curvaturesat rest, for growing rods */
    Vector3DStack& mutable_rest_kappas();
    /** @brief Rest lengths of the elements, for growing rods */
    Eigen::VectorXd& mutable_rest_lengths();
    /** @brief Radii of the elements, for growing rods */
    Eigen::VectorXd& mutable_radii();
    /** @brief Densities of the elements, for growing rods */
    Eigen::VectorXd& mutable_densities();
    /** @brief Shearing matrices, for growing rods */
    Matrix3DStack& mutable_shearing_matrices();
    /** @brief Rest sigmas, for growing rods */
    Vector3DStack& mutable_rest_sigmas();
};

/**
 * @brief Builds a straight rod of uniform radius and density.
 *
 * Nodes are spaced evenly from @p start along @p direction, and every element
 * receives the same frame, built from @p normal and @p direction so that its
 * third row is the tangent.
 *
 * @param num_elements Number of elements; at least three.
 * @param start Position of the first node.
 * @param direction Unit vector along the rod.
 * @param normal Unit vector orthogonal to @p direction, fixing the roll of the
 *        material frame.
 * @param base_length Total rod length.
 * @param base_radius Uniform element radius.
 * @param density Uniform density.
 * @param youngs_modulus Young's modulus; the shear modulus is taken as a third
 *        of it, matching a Poisson ratio of one half.
 * @param respect_radii When true the radius is held fixed and volume follows
 *        the current length. Pass false to hold volume instead, which is what
 *        the reference implementation does.
 * @param tolerance Tolerance for the unit-vector and orthogonality checks on
 *        @p direction and @p normal.
 * @return The constructed rod.
 */
CosseratRod straight_cosserat_rod(
    std::int64_t num_elements,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction,
    Eigen::Vector3d normal,
    double base_length,
    double base_radius,
    double density,
    double youngs_modulus,
    bool respect_radii,
    double tolerance
);

/**
 * @brief Builds a straight rod of uniform radius and density.
 *
 * Nodes are spaced evenly from @p start along @p direction, and every element
 * receives the same frame, built from @p normal and @p direction so that its
 * third row is the tangent.
 *
 * @param num_elements Number of elements; at least three.
 * @param start Position of the first node.
 * @param direction Unit vector along the rod.
 * @param normal Unit vector orthogonal to @p direction, fixing the roll of the
 *        material frame.
 * @param base_length Total rod length.
 * @param base_radius Uniform element radius.
 * @param density Uniform density.
 * @param youngs_modulus Young's modulus
 * @param shear_modulus Shear modulus
 * @param respect_radii When true the radius is held fixed and volume follows
 *        the current length. Pass false to hold volume instead, which is what
 *        the reference implementation does.
 * @param tolerance Tolerance for the unit-vector and orthogonality checks on
 *        @p direction and @p normal.
 * @return The constructed rod.
 */
CosseratRod straight_cosserat_rod(
    std::int64_t num_elements,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction,
    Eigen::Vector3d normal,
    double base_length,
    double base_radius,
    double density,
    double youngs_modulus,
    double shear_modulus,
    bool respect_radii,
    double tolerance
);
} // End namespace cosserat::physics
