#include <cosserat/physics/mesh_body.hpp>

#include <cmath>
#include <utility>

#include <cosserat/utils/assertions.hpp>
#include <cosserat/utils/file_utils.hpp>

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/**
 * @brief Carries a mesh into the frame a rigid body wants to hold it in.
 *
 * Shifts the centroid to the origin and rotates the principal directions onto
 * the axes, which is the frame in which the inertia tensor is the diagonal a
 * rigid body stores.
 *
 * @param mesh Mesh in its original frame.
 * @param center Centre of mass, in that frame.
 * @param axes Principal directions as columns, in that frame.
 * @return The carried mesh, with the same triangles and moved vertices.
 */
math::TriangleMesh carry_into_body_frame(
    const math::TriangleMesh& mesh, const Eigen::Vector3d& center, const Eigen::Matrix3d& axes
)
{
    math::TriangleMesh carried;
    carried.triangles = mesh.triangles;
    carried.vertices.reserve(mesh.vertices.size());

    // Columns of axes are the body's directions in the input frame, so the
    // transpose reads an input vector's components along them.
    const Eigen::Matrix3d into_body = axes.transpose();
    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        carried.vertices.push_back(into_body * (vertex - center));
    }
    return carried;
}

/**
 * @brief A nominal radius for a mesh, being its bounding sphere about the
 *        centroid.
 *
 * A rigid body carries a radius and a length because the primitives it was
 * written for have them. Neither is meaningful for an arbitrary mesh, but both
 * must be positive, so the tightest sphere containing the body frame mesh
 * stands in. Contact never consults these; the field describes the shape.
 *
 * @param mesh Mesh already carried into the body frame.
 * @return The largest distance from the origin to any vertex.
 */
double bounding_radius(const math::TriangleMesh& mesh)
{
    double radius = 0.0;
    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        radius = std::max(radius, vertex.norm());
    }
    return radius;
}
} // End anonymous namespace

MeshBody::Setup MeshBody::prepare(const math::TriangleMesh& mesh, double density)
{
    Setup setup;
    setup.properties = compute_mass_properties(mesh, density);
    setup.axes = principal_axes(setup.properties.inertia_about_center);
    setup.body_frame_mesh = carry_into_body_frame(
        mesh, setup.properties.center_of_mass, setup.axes.axes
    );
    setup.bounding_radius = bounding_radius(setup.body_frame_mesh);
    nice_assert(
        setup.bounding_radius > 0.0,
        "every vertex of the mesh coincides with its centroid, so it bounds no solid"
    );
    return setup;
}

MeshBody::MeshBody(Setup setup, double density, double field_margin, bool validate)
    : RigidBody(
          setup.properties.center_of_mass,
          // Columns of axes are the body's directions in the input frame; the
          // stored frame carries an input vector into the body, so transpose.
          setup.axes.axes.transpose(),
          setup.bounding_radius,
          // A mesh has no length, but the base requires a positive one, so the
          // bounding sphere's diameter stands in. Nothing reads it.
          2.0 * setup.bounding_radius,
          density,
          setup.properties.volume,
          setup.axes.moments
      ),
      m_field(setup.body_frame_mesh, field_margin, validate),
      m_body_frame_mesh(std::move(setup.body_frame_mesh)),
      m_volume(setup.properties.volume),
      m_principal_moments(setup.axes.moments) {}

MeshBody::MeshBody(
    const math::TriangleMesh& mesh,
    double density,
    double field_margin,
    bool validate
) : MeshBody(prepare(mesh, density), density, field_margin, validate)
{
    nice_assert(
        std::isfinite(field_margin) and field_margin > 0.0,
        "the field margin must be finite and positive"
    );
}

math::FieldQuery MeshBody::field_query() const {return math::as_query(m_field);}

Eigen::AlignedBox3d MeshBody::field_domain() const {return m_field.domain();}

const math::TriangleMeshField& MeshBody::field() const {return m_field;}

const math::TriangleMesh& MeshBody::body_frame_mesh() const
{
    return m_body_frame_mesh;
}

void MeshBody::write_mesh(const std::filesystem::path& write_path) const
{
    const auto vertex_count =
        static_cast<Eigen::Index>(m_body_frame_mesh.vertices.size());
    const auto triangle_count =
        static_cast<Eigen::Index>(m_body_frame_mesh.triangles.size());
    nice_assert(vertex_count > 0, "a mesh body must have vertices to write");
    nice_assert(triangle_count > 0, "a mesh body must have triangles to write");

    Eigen::MatrixXd vertices(vertex_count, 3);
    for (Eigen::Index row = 0; row < vertex_count; ++row)
    {
        vertices.row(row) =
            m_body_frame_mesh.vertices[static_cast<std::size_t>(row)].transpose();
    }

    // Indices widened to double, which is the only scalar the format carries.
    // Exact for every index a mesh could plausibly reach.
    Eigen::MatrixXd triangles(triangle_count, 3);
    for (Eigen::Index row = 0; row < triangle_count; ++row)
    {
        triangles.row(row) =
            m_body_frame_mesh.triangles[static_cast<std::size_t>(row)]
                .cast<double>().transpose();
    }

    utils::write_matrix(write_path / "mesh_vertices", vertices);
    utils::write_matrix(write_path / "mesh_triangles", triangles);
}

double MeshBody::volume() const {return m_volume;}

const Eigen::Vector3d& MeshBody::principal_moments() const
{
    return m_principal_moments;
}
} // End namespace cosserat::physics
