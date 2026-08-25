#pragma once

/**
 * @file triangle_mesh_field.hpp
 * @brief A signed distance field read directly from a triangle mesh.
 *
 * Satisfies @ref SignedDistanceField by querying the triangles themselves
 * through a bounding volume hierarchy, so distances are exact and no
 * precomputation, grid or memory budget is involved. That makes it both the
 * simplest backend to reach for and the natural reference to check any
 * approximate one against.
 *
 * @section tmf_dependency Where the dependency lives
 *
 * The hierarchy comes from TriangleMeshDistance, vendored under
 * @c external/TriangleMeshDistance. That header appears only in this file's
 * translation unit, never in this header, so nothing downstream needs its
 * include path: a target using a mesh field needs @c math::math and no more.
 * The cost is one out-of-line call per query, which is nothing beside the
 * traversal it wraps.
 *
 * @section tmf_watertight What the mesh has to be
 *
 * Signs come from the angle weighted pseudonormal of Bærentzen and Aanæs, and
 * that construction assumes a closed surface with consistently outward
 * winding. Handed a triangle soup, an open shell or a mesh with flipped
 * faces, the library still returns the correct *distance* but the sign is
 * meaningless, which in contact means a rod pushed in an arbitrary direction
 * rather than out.
 *
 * @ref TriangleMeshField checks what can be checked cheaply at construction:
 * that every edge is shared by exactly two triangles, and that neighbours
 * agree on orientation. See @ref validate_closed_surface. Neither test is a
 * full manifoldness proof, but between them they reject the mistakes that
 * actually happen.
 *
 * @section tmf_cost What a query costs
 *
 * Cost grows with triangle count, roughly as its square root, because a
 * nearest point query has to keep a running best and cannot prune as sharply
 * as a ray cast.
 *
 * A discretised field is flat in that respect, costing the same whatever the
 * mesh, so it overtakes this backend somewhere past a hundred thousand
 * triangles. Below that the margin is a factor of two or three, paid for with
 * discretisation error, a build step and tens to hundreds of megabytes. See
 * @ref sdf_discrete_backend.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cosserat/math/signed_distance_field.hpp>

namespace cosserat::math {

/**
 * @brief A triangle mesh, as vertices and the triples indexing them.
 *
 * Deliberately plain: any loader can fill one of these without knowing
 * anything about the field built from it.
 */
struct TriangleMesh
{
public: // Members
    /** @brief Vertex positions. */
    std::vector<Eigen::Vector3d> vertices;

    /** @brief Triangles, as indices into @ref vertices, wound outward. */
    std::vector<Eigen::Vector3i> triangles;
};

/**
 * @brief What a surface validity check found.
 */
struct SurfaceValidity
{
public: // Members
    /** @brief Whether the mesh looks like a closed, consistently wound surface. */
    bool is_closed_surface = false;

    /** @brief Edges used by exactly one triangle, so the surface has a hole. */
    std::int64_t boundary_edges = 0;

    /** @brief Edges used by three or more triangles, so it is not a surface. */
    std::int64_t non_manifold_edges = 0;

    /** @brief Edges whose two triangles disagree about which way is out. */
    std::int64_t inconsistent_edges = 0;

    /**
     * @brief Whether the surface faces outward rather than being inside out.
     *
     * Every edge test passes on a mesh wound uniformly inward, so this is
     * checked separately, by the sign of @ref signed_volume.
     */
    bool is_wound_outward = false;

    /**
     * @brief Volume the surface encloses, from the divergence theorem.
     *
     * Negative when the mesh is inside out, which is the one way a mesh can be
     * perfectly consistent and still give every distance the wrong sign.
     */
    double signed_volume = 0.0;

    /** @brief A description of what went wrong, empty when nothing did. */
    std::string message;
};

/**
 * @brief Checks that a mesh is a closed surface with consistent winding.
 *
 * Walks the half edges. In a closed, consistently wound surface every directed
 * edge appears exactly once and its reverse appears exactly once in the
 * neighbouring triangle. An edge seen only once is a hole; an edge seen three
 * or more times is not a surface; the same direction seen twice means two
 * triangles disagree about which side is outside.
 *
 * This does not establish manifoldness at vertices, where two cones can meet
 * at a point with every edge still well behaved. It does catch open shells,
 * duplicated faces and flipped normals, which is what tends to go wrong.
 *
 * @param mesh Mesh to inspect.
 * @return What was found, including a message when the mesh is unusable.
 */
SurfaceValidity validate_closed_surface(const TriangleMesh& mesh);

/**
 * @brief An exact signed distance field over a triangle mesh.
 *
 * Positive outside the surface, negative within, with a gradient pointing away
 * from it, as @ref SignedDistanceField requires.
 */
class TriangleMeshField
{
private: // Types
    struct Impl;

private: // Members
    // Shared rather than unique so the field is copyable, which lets a body
    // hold one by value without the hierarchy being rebuilt.
    std::shared_ptr<const Impl> m_impl;

public: // Methods
    /**
     * @brief Builds the field, and the hierarchy behind it, from a mesh.
     *
     * @param mesh The mesh; must be a closed surface with outward winding.
     * @param margin How far beyond the mesh's own extent the domain reaches.
     *        Unlike a grid this field is defined everywhere, so the domain is
     *        only a hint for broad phase culling; it wants to be at least the
     *        largest radius that will be tested against it.
     * @param validate Whether to run @ref validate_closed_surface and reject a
     *        mesh that fails. Turning it off is for meshes already known good,
     *        and means a bad one produces silently meaningless signs.
     *
     * @throws Fails an assertion if the mesh is empty, indexes a vertex that
     *         does not exist, or is rejected by validation.
     */
    explicit TriangleMeshField(
        const TriangleMesh& mesh, double margin = 1.0, bool validate = true
    );

    /**
     * @brief Distance and gradient at a point.
     *
     * @param point Query position, in the same frame as the mesh.
     * @return Signed distance and the outward gradient there.
     *
     * @note The gradient degenerates on the medial axis, where the nearest
     *       point on the surface is not unique. Callers should check its
     *       length rather than normalising blindly.
     */
    SignedDistance signed_distance(const Eigen::Vector3d& point) const;

    /**
     * @brief The box the mesh occupies, grown by the construction margin.
     * @return The domain, for broad phase culling.
     */
    Eigen::AlignedBox3d domain() const;

    /** @brief Number of triangles in the mesh. */
    std::int64_t num_triangles() const;

    /** @brief Number of vertices in the mesh. */
    std::int64_t num_vertices() const;
};

// A mesh field is usable anywhere a field is asked for.
static_assert(SignedDistanceField<TriangleMeshField>);

/**
 * @brief A closed, outward wound box, for tests and simple obstacles.
 *
 * Twelve triangles. Its exact field is @ref AnalyticBoxField, so the two can
 * be compared to check the mesh backend against a closed form.
 *
 * @param center Centre of the box.
 * @param half_extent Half its size on each axis; all positive.
 * @return The mesh.
 */
TriangleMesh make_box_mesh(
    const Eigen::Vector3d& center, const Eigen::Vector3d& half_extent
);

/**
 * @brief A closed, outward wound sphere approximation, for tests.
 *
 * Built by recursively subdividing an icosahedron and projecting onto the
 * sphere, so the triangles stay near uniform and there are no degenerate
 * slivers of the sort a latitude and longitude sphere produces at its poles.
 *
 * @param center Centre of the sphere.
 * @param radius Radius; must be positive.
 * @param subdivisions Number of subdivision passes. Each multiplies the
 *        triangle count by four, from 20 at zero.
 * @return The mesh.
 */
TriangleMesh make_sphere_mesh(
    const Eigen::Vector3d& center, double radius, int subdivisions = 3
);
} // End namespace cosserat::math
