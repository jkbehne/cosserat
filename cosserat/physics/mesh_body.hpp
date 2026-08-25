#pragma once

/**
 * @file mesh_body.hpp
 * @brief A rigid body whose shape is an arbitrary closed triangle mesh.
 *
 * The rigid bodies the reference implementation offers are a sphere and a
 * cylinder, both described by a couple of numbers. A @ref MeshBody is
 * described by a surface instead, which is what lets a rod collide with
 * something that is not an analytic primitive. Everything the dynamics needs
 * is derived from that surface: its mass, its centre of mass and its inertia
 * come from integrating over the solid it bounds, and its geometry for contact
 * comes from a signed distance field over the same triangles.
 *
 * @section mb_canonical The body's own frame
 *
 * A rigid body here stores its inertia as three numbers, which is only
 * meaningful in the frame where the inertia tensor is diagonal. A mesh drawn
 * in some arbitrary modelling frame will not be aligned that way, so the mesh
 * is carried into its principal frame on construction:
 *
 * @f[ \mathbf{p}_{body} = \mathbf{R}^{T}(\mathbf{p}_{input} - \mathbf{c}) @f]
 *
 * with @f$ \mathbf{c} @f$ the centre of mass and @f$ \mathbf{R} @f$ the
 * principal directions as columns. The body's position is then the centre of
 * mass in the input frame, and its stored frame is @f$ \mathbf{R}^{T} @f$,
 * matching the convention that a frame carries lab vectors into the body.
 *
 * Two consequences worth knowing. The origin of the body frame is the centre
 * of mass, not wherever the modeller put the origin, so a mesh drawn far from
 * its own centroid still rotates about the right point. And the distance field
 * is built from the carried mesh, so it too is in body coordinates and never
 * needs rebuilding as the body moves.
 *
 * @section mb_symmetric A caveat for symmetric shapes
 *
 * Principal directions are eigenvectors, and an eigenvector's sign is
 * arbitrary, so even a shape with three distinct moments can be handed a frame
 * differing by a half turn from the one a reader expects. Two identical bodies
 * built at different positions may end up with different frames and
 * correspondingly rotated body frame meshes; what is invariant is the pair,
 * which always reconstructs the same world geometry.
 *
 * The freedom is larger still for a shape with repeated moments: a cube and a
 * sphere are diagonal in every frame. The orientation such a body
 * ends up with is therefore arbitrary among equally valid choices, and may not
 * be the one the mesh was drawn in. It is still correct, but do not rely on a
 * symmetric mesh keeping its modelling orientation.
 */

#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cosserat/math/signed_distance_field.hpp>
#include <cosserat/math/triangle_mesh_field.hpp>

#include <cosserat/physics/mass_properties.hpp>
#include <cosserat/physics/rigid_body.hpp>

namespace cosserat::physics {

/**
 * @brief A rigid body bounded by a closed triangle mesh.
 *
 * Adds a signed distance field to @ref RigidBody, which is what
 * @ref ContactableMeshBody needs beyond the rigid body interface. Everything
 * else, the state, the accumulators and the dynamics, is inherited unchanged,
 * so a mesh body integrates exactly as a sphere or a cylinder does.
 */
class MeshBody : public RigidBody
{
private: // Types
    /**
     * @brief Everything derived from the mesh, computed once.
     *
     * A rigid body's members are all set by its constructor's initialiser
     * list, so a derived class either recomputes each argument in place or
     * gathers them first and delegates. Gathering keeps the single pass over
     * the triangles that the mass property algorithm is built around.
     */
    struct Setup
    {
    public: // Members
        /** @brief Mass, centre and inertia of the solid. */
        MassProperties properties;

        /** @brief The frame in which that inertia is diagonal. */
        PrincipalAxes axes;

        /** @brief The mesh carried into that frame. */
        math::TriangleMesh body_frame_mesh;

        /** @brief Bounding sphere of the carried mesh, as a nominal radius. */
        double bounding_radius = 0.0;
    };

private: // Members
    math::TriangleMeshField m_field;
    math::TriangleMesh m_body_frame_mesh;
    double m_volume;
    Eigen::Vector3d m_principal_moments;

public: // Methods
    /**
     * @brief Builds a body from a mesh and a density.
     *
     * The mesh is validated, its mass properties computed, and it is then
     * carried into its principal frame, where the distance field is built.
     *
     * @param mesh A closed, outward wound triangle mesh, in any frame.
     * @param density Uniform density of the solid; must be finite and
     *        positive.
     * @param field_margin How far beyond the mesh the field's domain reaches.
     *        Wants to be at least the largest rod radius that will touch it,
     *        since the domain doubles as the broad phase bound.
     * @param validate Whether to reject a mesh that is not a closed, outward
     *        wound surface. Leaving it on is strongly preferred: an inward
     *        wound mesh yields a negative volume, and an open one has no
     *        meaningful inside at all.
     *
     * @throws Fails an assertion if the mesh is unusable, if the density is
     *         not positive, or if the solid is degenerate enough that one of
     *         its principal moments vanishes, which a flat or zero volume mesh
     *         produces.
     */
    MeshBody(
        const math::TriangleMesh& mesh,
        double density,
        double field_margin,
        bool validate
    );

    /**
     * @brief The body's distance field, as a callable.
     *
     * Expressed in body coordinates, so a query has to be carried into the
     * body frame first. @ref RodMeshContact does that.
     *
     * @return A query into the field, valid while this body is.
     */
    math::FieldQuery field_query() const;

    /**
     * @brief The region the field is defined over, in body coordinates.
     * @return The domain box.
     */
    Eigen::AlignedBox3d field_domain() const;

    /**
     * @brief The field itself.
     * @return The distance field over the body frame mesh.
     */
    const math::TriangleMeshField& field() const;

    /**
     * @brief The mesh as carried into the body frame.
     *
     * Its centroid is the origin and its axes are the principal ones, so this
     * is the mesh a renderer should draw when it has already applied the
     * body's position and frame.
     *
     * @return The carried mesh.
     */
    const math::TriangleMesh& body_frame_mesh() const;

    /**
     * @brief Writes the body frame mesh, which never changes.
     *
     * A rigid body's shape is constant in its own frame, so the triangles only
     * ever need writing once for a whole run. What varies is the body's pose,
     * and @ref RigidBody::write already records that at every step. Together
     * the two are enough to place every vertex at every frame:
     *
     * @f[ \mathbf{v}_{world} = \mathbf{c}(t) + \mathbf{Q}(t)^{T}\mathbf{v}_{body} @f]
     *
     * Two files are produced. @c mesh_vertices holds the vertices as an
     * @c (n, 3) matrix in body coordinates, and @c mesh_triangles holds the
     * vertex indices as an @c (m, 3) matrix.
     *
     * @param write_path Directory to write into; must already exist.
     *
     * @note The indices are stored as doubles, because the binary format only
     *       carries that one scalar type. Every index below 2^53 is exact, so
     *       nothing is lost, but a reader has to cast them back.
     */
    void write_mesh(const std::filesystem::path& write_path) const;

    /** @brief Volume of the solid the mesh bounds. */
    double volume() const;

    /**
     * @brief The three principal moments of inertia, ascending.
     * @return The diagonal the body stores its inertia as.
     */
    const Eigen::Vector3d& principal_moments() const;

private: // Methods
    /**
     * @brief Derives everything the constructor needs, in one pass.
     *
     * @param mesh The mesh, in its original frame.
     * @param density Uniform density of the solid.
     * @return The gathered results.
     */
    static Setup prepare(const math::TriangleMesh& mesh, double density);

    /**
     * @brief Builds the body from already gathered results.
     *
     * @param setup What @ref prepare produced.
     * @param density Uniform density, which the base also records.
     * @param field_margin How far beyond the mesh the field reaches.
     * @param validate Whether the field should re-check the carried mesh.
     */
    MeshBody(Setup setup, double density, double field_margin, bool validate);
};
} // End namespace cosserat::physics
