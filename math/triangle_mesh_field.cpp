#include "math/triangle_mesh_field.hpp"

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include <tmd/TriangleMeshDistance.h>

#include "utils/assertions.hpp"

namespace cosserat::math {
using utils::nice_assert;

namespace {

/** @brief Packs a directed edge into one key, for half edge bookkeeping. */
std::uint64_t directed_edge_key(std::int32_t from, std::int32_t to)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32)
        | static_cast<std::uint32_t>(to);
}

} // End anonymous namespace

SurfaceValidity validate_closed_surface(const TriangleMesh& mesh)
{
    SurfaceValidity result;

    if (mesh.triangles.empty())
    {
        result.message = "the mesh has no triangles";
        return result;
    }

    // Count how many times each undirected edge is used, and how many times
    // each directed edge is. A closed, consistently wound surface uses every
    // undirected edge twice and every directed edge once.
    std::unordered_map<std::uint64_t, std::int32_t> undirected;
    std::unordered_map<std::uint64_t, std::int32_t> directed;
    undirected.reserve(mesh.triangles.size() * 3);
    directed.reserve(mesh.triangles.size() * 3);

    for (const Eigen::Vector3i& triangle : mesh.triangles)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const std::int32_t from = triangle(corner);
            const std::int32_t to = triangle((corner + 1) % 3);
            ++directed[directed_edge_key(from, to)];
            ++undirected[directed_edge_key(std::min(from, to), std::max(from, to))];
        }
    }

    for (const auto& [key, count] : undirected)
    {
        (void)key;
        if (count == 1) ++result.boundary_edges;
        else if (count > 2) ++result.non_manifold_edges;
    }

    // A directed edge seen twice means two triangles traverse it the same way,
    // so one of them is wound backwards relative to its neighbour.
    for (const auto& [key, count] : directed)
    {
        (void)key;
        if (count > 1) ++result.inconsistent_edges;
    }

    // Every edge test above is blind to a mesh wound uniformly inward: it is
    // perfectly consistent, just inside out, and its signs would all be
    // flipped. The divergence theorem separates the two, since the enclosed
    // volume comes out negative when the surface faces the wrong way.
    double six_volumes = 0.0;
    for (const Eigen::Vector3i& triangle : mesh.triangles)
    {
        const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(triangle(0))];
        const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(triangle(1))];
        const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(triangle(2))];
        six_volumes += a.dot(b.cross(c));
    }
    result.signed_volume = six_volumes / 6.0;
    result.is_wound_outward = result.signed_volume > 0.0;

    result.is_closed_surface = result.boundary_edges == 0
        and result.non_manifold_edges == 0 and result.inconsistent_edges == 0
        and result.is_wound_outward;

    if (not result.is_closed_surface)
    {
        result.message = "the mesh is not a closed, consistently wound surface: "
            + std::to_string(result.boundary_edges) + " boundary edge(s), "
            + std::to_string(result.non_manifold_edges) + " edge(s) shared by more "
              "than two triangles, "
            + std::to_string(result.inconsistent_edges)
            + " edge(s) whose triangles disagree on winding, signed volume "
            + std::to_string(result.signed_volume)
            + ". Signed distance needs a closed surface wound outward; without one "
              "the sign is meaningless or inverted.";
    }
    return result;
}

/**
 * @brief Holds the hierarchy, keeping the dependency out of the header.
 */
struct TriangleMeshField::Impl
{
public: // Members
    /** @brief The bounding volume hierarchy over the triangles. */
    tmd::TriangleMeshDistance distance;

    /** @brief The mesh's own extent, grown by the construction margin. */
    Eigen::AlignedBox3d domain;

    /** @brief Number of triangles, kept for reporting. */
    std::int64_t num_triangles = 0;

    /** @brief Number of vertices, kept for reporting. */
    std::int64_t num_vertices = 0;
};

TriangleMeshField::TriangleMeshField(
    const TriangleMesh& mesh, double margin, bool validate
)
{
    nice_assert(not mesh.vertices.empty(), "a mesh field needs at least one vertex");
    nice_assert(not mesh.triangles.empty(), "a mesh field needs at least one triangle");
    nice_assert(
        std::isfinite(margin) and margin > 0.0, "the domain margin must be finite and positive"
    );

    const auto vertex_count = static_cast<std::int32_t>(mesh.vertices.size());
    for (const Eigen::Vector3i& triangle : mesh.triangles)
    {
        nice_assert(
            (triangle.array() >= 0).all() and (triangle.array() < vertex_count).all(),
            "a triangle indexes a vertex that does not exist"
        );
    }
    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        nice_assert(vertex.array().isFinite().all(), "mesh vertices must be finite");
    }

    if (validate)
    {
        const SurfaceValidity validity = validate_closed_surface(mesh);
        nice_assert(validity.is_closed_surface, validity.message);
    }

    // Flattened, because the library takes contiguous coordinate and index
    // arrays rather than Eigen types.
    std::vector<double> coordinates;
    coordinates.reserve(mesh.vertices.size() * 3);
    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        coordinates.push_back(vertex.x());
        coordinates.push_back(vertex.y());
        coordinates.push_back(vertex.z());
    }

    std::vector<int> indices;
    indices.reserve(mesh.triangles.size() * 3);
    for (const Eigen::Vector3i& triangle : mesh.triangles)
    {
        indices.push_back(triangle.x());
        indices.push_back(triangle.y());
        indices.push_back(triangle.z());
    }

    auto impl = std::make_shared<Impl>();
    impl->distance = tmd::TriangleMeshDistance(
        coordinates.data(), static_cast<int>(mesh.vertices.size()),
        indices.data(), static_cast<int>(mesh.triangles.size())
    );
    impl->num_triangles = static_cast<std::int64_t>(mesh.triangles.size());
    impl->num_vertices = static_cast<std::int64_t>(mesh.vertices.size());

    Eigen::AlignedBox3d extent;
    for (const Eigen::Vector3d& vertex : mesh.vertices) extent.extend(vertex);
    const Eigen::Vector3d pad = Eigen::Vector3d::Constant(margin);
    impl->domain = Eigen::AlignedBox3d(extent.min() - pad, extent.max() + pad);

    m_impl = std::move(impl);
}

SignedDistance TriangleMeshField::signed_distance(const Eigen::Vector3d& point) const
{
    const tmd::Result result =
        m_impl->distance.signed_distance({point.x(), point.y(), point.z()});

    SignedDistance out;
    out.distance = result.distance;

    // The library returns the nearest point rather than a gradient. Away from
    // the surface the two are related exactly: the gradient is the unit vector
    // from that point toward the query, flipped when the query is inside so
    // the result still points outward.
    const Eigen::Vector3d nearest(
        result.nearest_point[0], result.nearest_point[1], result.nearest_point[2]
    );
    const Eigen::Vector3d offset = point - nearest;
    const double length = offset.norm();
    if (length > 0.0)
    {
        out.gradient = (result.distance < 0.0 ? -1.0 : 1.0) * (offset / length);
    }
    else
    {
        // Exactly on the surface, where the offset carries no direction. The
        // face normal is the right answer there, but it is not reported, so
        // this degenerates like the medial axis and callers skip it.
        out.gradient = Eigen::Vector3d::Zero();
    }
    return out;
}

Eigen::AlignedBox3d TriangleMeshField::domain() const {return m_impl->domain;}

std::int64_t TriangleMeshField::num_triangles() const {return m_impl->num_triangles;}

std::int64_t TriangleMeshField::num_vertices() const {return m_impl->num_vertices;}

TriangleMesh make_box_mesh(
    const Eigen::Vector3d& center, const Eigen::Vector3d& half_extent
)
{
    nice_assert(
        half_extent.array().isFinite().all() and (half_extent.array() > 0.0).all(),
        "box half extents must be finite and positive"
    );

    TriangleMesh mesh;
    mesh.vertices.reserve(8);
    for (int corner = 0; corner < 8; ++corner)
    {
        // Bit per axis, so the eight corners come out in a predictable order.
        const Eigen::Vector3d sign(
            (corner & 1) ? 1.0 : -1.0,
            (corner & 2) ? 1.0 : -1.0,
            (corner & 4) ? 1.0 : -1.0
        );
        mesh.vertices.push_back(center + sign.cwiseProduct(half_extent));
    }

    // Two triangles per face, every one wound counter clockwise seen from
    // outside so the surface is consistently oriented.
    const std::array<std::array<int, 4>, 6> faces{{
        {{0, 2, 6, 4}},  // -x
        {{1, 5, 7, 3}},  // +x
        {{0, 4, 5, 1}},  // -y
        {{2, 3, 7, 6}},  // +y
        {{0, 1, 3, 2}},  // -z
        {{4, 6, 7, 5}},  // +z
    }};
    // Reversed relative to the quad order above, which winds each face
    // outward rather than into the box.
    mesh.triangles.reserve(12);
    for (const auto& face : faces)
    {
        mesh.triangles.emplace_back(face[0], face[2], face[1]);
        mesh.triangles.emplace_back(face[0], face[3], face[2]);
    }
    return mesh;
}

TriangleMesh make_sphere_mesh(
    const Eigen::Vector3d& center, double radius, int subdivisions
)
{
    nice_assert(
        std::isfinite(radius) and radius > 0.0, "the sphere radius must be finite and positive"
    );
    nice_assert(subdivisions >= 0, "the subdivision count cannot be negative");

    // Icosahedron, whose vertices are the corners of three golden rectangles.
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Eigen::Vector3d> points{
        {-1, golden, 0}, {1, golden, 0}, {-1, -golden, 0}, {1, -golden, 0},
        {0, -1, golden}, {0, 1, golden}, {0, -1, -golden}, {0, 1, -golden},
        {golden, 0, -1}, {golden, 0, 1}, {-golden, 0, -1}, {-golden, 0, 1},
    };
    for (Eigen::Vector3d& point : points) point.normalize();

    std::vector<Eigen::Vector3i> faces{
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    };

    for (int pass = 0; pass < subdivisions; ++pass)
    {
        // Midpoints are cached per edge so neighbouring triangles share the
        // vertex they create, which is what keeps the surface closed.
        std::map<std::pair<int, int>, int> midpoints;
        const auto midpoint = [&](int first, int second)
        {
            const auto key = std::minmax(first, second);
            const auto found = midpoints.find({key.first, key.second});
            if (found != midpoints.end()) return found->second;

            Eigen::Vector3d created = 0.5 * (points[static_cast<std::size_t>(first)]
                                             + points[static_cast<std::size_t>(second)]);
            created.normalize();
            points.push_back(created);
            const int index = static_cast<int>(points.size()) - 1;
            midpoints[{key.first, key.second}] = index;
            return index;
        };

        std::vector<Eigen::Vector3i> refined;
        refined.reserve(faces.size() * 4);
        for (const Eigen::Vector3i& face : faces)
        {
            const int a = midpoint(face(0), face(1));
            const int b = midpoint(face(1), face(2));
            const int c = midpoint(face(2), face(0));
            refined.emplace_back(face(0), a, c);
            refined.emplace_back(face(1), b, a);
            refined.emplace_back(face(2), c, b);
            refined.emplace_back(a, b, c);
        }
        faces.swap(refined);
    }

    TriangleMesh mesh;
    mesh.vertices.reserve(points.size());
    for (const Eigen::Vector3d& point : points)
    {
        mesh.vertices.push_back(center + radius * point);
    }
    mesh.triangles = std::move(faces);
    return mesh;
}
} // End namespace cosserat::math
