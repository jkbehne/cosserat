#pragma once

/**
 * @file mass_properties.hpp
 * @brief Mass, centre of mass and inertia of a solid bounded by a mesh.
 *
 * A triangle mesh describes only a surface, but a closed surface of uniform
 * density determines a solid, and everything the dynamics needs about that
 * solid follows from integrals over its interior. The divergence theorem turns
 * those volume integrals into surface integrals, so all of them can be
 * accumulated in a single pass over the triangles.
 *
 * @f[ \int_V f \, dV = \oint_{\partial V} \mathbf{F} \cdot \hat{\mathbf{n}} \, dA @f]
 *
 * Ten integrals are needed: the volume itself, the three first moments giving
 * the centre of mass, and the six second moments giving the inertia tensor.
 * Taking them together lets the per-triangle work be shared, which is the point
 * of the algorithm.
 *
 * @section mass_sources Where this comes from
 *
 * The algorithm is Mirtich, *Fast and Accurate Computation of Polyhedral Mass
 * Properties*, Journal of Graphics Tools 1(2), 1996, pp. 31-50, arranged as in
 * Eberly, *Polyhedral Mass Properties (Revisited)*, 2009, whose formulation is
 * specialised to triangles and is what the subexpressions below follow.
 *
 * Mirtich projects each face onto a coordinate plane chosen to avoid poor
 * conditioning when a face lies nearly parallel to it. The triangle-only form
 * used here does not need that choice, because a triangle's contribution is
 * written directly in terms of its edge cross product, which degenerates only
 * for a triangle of zero area.
 *
 * @section mass_requirements What the mesh must be
 *
 * The surface has to be closed and wound outward, exactly as a signed distance
 * field requires. An open surface does not bound a solid at all, and one wound
 * inward yields a negative volume and hence a negative mass. Both are rejected
 * rather than propagated; see @ref cosserat::math::validate_closed_surface.
 */

#include <Eigen/Core>
#include <Eigen/Dense>

#include "math/triangle_mesh_field.hpp"

namespace cosserat::physics {

/**
 * @brief What a closed mesh implies about the solid it bounds.
 *
 * The inertia tensor is reported about the centre of mass rather than about
 * the mesh's coordinate origin, since that is what rigid body dynamics wants
 * and the shift is part of the same computation.
 */
struct MassProperties
{
public: // Members
    /** @brief Volume enclosed by the surface; always positive. */
    double volume = 0.0;

    /** @brief Mass, being the volume times the density. */
    double mass = 0.0;

    /** @brief Centre of mass, in the frame the mesh was given in. */
    Eigen::Vector3d center_of_mass = Eigen::Vector3d::Zero();

    /**
     * @brief Inertia tensor about the centre of mass.
     *
     * Symmetric and positive definite for any non degenerate solid. Its axes
     * are those of the mesh's own frame, so it is diagonal only when the mesh
     * happens to be aligned with its principal axes; see @ref principal_axes.
     */
    Eigen::Matrix3d inertia_about_center = Eigen::Matrix3d::Zero();
};

/**
 * @brief The frame in which an inertia tensor is diagonal.
 */
struct PrincipalAxes
{
public: // Members
    /**
     * @brief The three principal moments, ascending.
     *
     * These are the diagonal an aligned body would carry.
     */
    Eigen::Vector3d moments = Eigen::Vector3d::Zero();

    /**
     * @brief The principal directions, as columns, in the original frame.
     *
     * A rotation: orthonormal and right handed. Reading the columns as the
     * body's axes expressed in the original frame, its transpose is the matrix
     * carrying a vector from that frame into the body, which is the convention
     * the rigid bodies here store.
     */
    Eigen::Matrix3d axes = Eigen::Matrix3d::Identity();
};

/**
 * @brief Computes the mass properties of the solid a mesh bounds.
 *
 * Accumulates all ten integrals in one pass over the triangles, then shifts
 * the inertia from the coordinate origin to the centre of mass by the parallel
 * axis theorem.
 *
 * @param mesh A closed, outward wound triangle mesh.
 * @param density Uniform density of the solid; must be finite and positive.
 * @return The volume, mass, centre of mass and inertia about that centre.
 *
 * @throws Fails an assertion if the density is not positive, if the mesh has
 *         no triangles, or if the computed volume is not positive, which is
 *         what an inward wound or unclosed surface produces.
 *
 * @note Does not itself run @ref cosserat::math::validate_closed_surface,
 *       which is the more informative check. The volume test here is a last
 *       line of defence rather than a substitute for validating on load.
 */
MassProperties compute_mass_properties(const math::TriangleMesh& mesh, double density);

/**
 * @brief Diagonalises an inertia tensor into moments and principal directions.
 *
 * A rigid body whose inertia is stored as three numbers is implicitly
 * expressed in its principal frame, so a mesh that is not aligned with its own
 * principal axes has to be rotated into them before its inertia can be stored
 * that way.
 *
 * @param inertia A symmetric inertia tensor.
 * @return The principal moments, ascending, with the matching directions as
 *         the columns of a right handed rotation.
 *
 * @throws Fails an assertion if the tensor is not symmetric, or if the
 *         decomposition does not converge.
 *
 * @note The eigenvectors of a tensor with repeated moments are not unique: a
 *       sphere or a cube has no distinguished axes, and any orthonormal frame
 *       diagonalises it. The result is still correct, just arbitrary among the
 *       valid choices, so do not expect a particular orientation from a
 *       symmetric shape.
 */
PrincipalAxes principal_axes(const Eigen::Matrix3d& inertia);
} // End namespace cosserat::physics
