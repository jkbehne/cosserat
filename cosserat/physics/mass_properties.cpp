#include <cosserat/physics/mass_properties.hpp>

#include <cmath>

#include <cosserat/utils/assertions.hpp>

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/** @brief Tolerance on the symmetry check applied to an inertia tensor. */
constexpr double symmetry_tolerance = 1e-9;

/**
 * @brief The power sums one coordinate axis contributes to a triangle.
 *
 * Integrating a polynomial over a triangle reduces to symmetric functions of
 * the three vertex coordinates along one axis. Naming them makes the
 * accumulation below readable, and computing them together lets the higher
 * powers reuse the lower ones rather than recomputing them.
 */
struct AxisPowerSums
{
public: // Members
    /** @brief Sum of the three coordinates. */
    double first = 0.0;

    /** @brief The symmetric sum of degree two. */
    double second = 0.0;

    /** @brief The symmetric sum of degree three. */
    double third = 0.0;

    /** @brief Per-vertex weights used by the mixed second moments. */
    Eigen::Vector3d vertex_weights = Eigen::Vector3d::Zero();
};

/**
 * @brief Builds the power sums for one axis of one triangle.
 *
 * @param w0 Coordinate of the first vertex along this axis.
 * @param w1 Coordinate of the second vertex.
 * @param w2 Coordinate of the third vertex.
 * @return The sums, following Eberly's arrangement of Mirtich's reduction.
 */
AxisPowerSums axis_power_sums(double w0, double w1, double w2)
{
    const double leading_pair = w0 + w1;
    const double squared_first = w0 * w0;
    const double partial_second = squared_first + w1 * leading_pair;

    AxisPowerSums sums;
    sums.first = leading_pair + w2;
    sums.second = partial_second + w2 * sums.first;
    sums.third = w0 * squared_first + w1 * partial_second + w2 * sums.second;
    sums.vertex_weights = Eigen::Vector3d(
        sums.second + w0 * (sums.first + w0),
        sums.second + w1 * (sums.first + w1),
        sums.second + w2 * (sums.first + w2)
    );
    return sums;
}

} // End anonymous namespace

MassProperties compute_mass_properties(const math::TriangleMesh& mesh, double density)
{
    nice_assert(
        std::isfinite(density) and density > 0.0, "density must be finite and positive"
    );
    nice_assert(not mesh.triangles.empty(), "a solid needs at least one triangle");
    nice_assert(not mesh.vertices.empty(), "a solid needs at least one vertex");

    const auto vertex_count = static_cast<std::int32_t>(mesh.vertices.size());

    // The ten integrals over the solid, accumulated together so each triangle
    // is walked once: the volume, the three first moments, the three second
    // moments about each axis, and the three mixed second moments.
    double integrated_volume = 0.0;
    Eigen::Vector3d first_moments = Eigen::Vector3d::Zero();
    Eigen::Vector3d second_moments = Eigen::Vector3d::Zero();
    Eigen::Vector3d mixed_moments = Eigen::Vector3d::Zero();

    for (const Eigen::Vector3i& triangle : mesh.triangles)
    {
        nice_assert(
            (triangle.array() >= 0).all() and (triangle.array() < vertex_count).all(),
            "a triangle indexes a vertex that does not exist"
        );
        const Eigen::Vector3d& v0 = mesh.vertices[static_cast<std::size_t>(triangle(0))];
        const Eigen::Vector3d& v1 = mesh.vertices[static_cast<std::size_t>(triangle(1))];
        const Eigen::Vector3d& v2 = mesh.vertices[static_cast<std::size_t>(triangle(2))];

        // Area weighted normal. Its length is twice the triangle's area, and
        // it carries the orientation, which is what makes the surface integral
        // signed and therefore able to represent a solid.
        const Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0);

        const AxisPowerSums along_x = axis_power_sums(v0.x(), v1.x(), v2.x());
        const AxisPowerSums along_y = axis_power_sums(v0.y(), v1.y(), v2.y());
        const AxisPowerSums along_z = axis_power_sums(v0.z(), v1.z(), v2.z());

        integrated_volume += normal.x() * along_x.first;

        first_moments += Eigen::Vector3d(
            normal.x() * along_x.second,
            normal.y() * along_y.second,
            normal.z() * along_z.second
        );
        second_moments += Eigen::Vector3d(
            normal.x() * along_x.third,
            normal.y() * along_y.third,
            normal.z() * along_z.third
        );

        // Each mixed moment pairs one axis's per-vertex weights with the
        // vertex coordinates along the next axis round.
        const Eigen::Vector3d y_coordinates(v0.y(), v1.y(), v2.y());
        const Eigen::Vector3d z_coordinates(v0.z(), v1.z(), v2.z());
        const Eigen::Vector3d x_coordinates(v0.x(), v1.x(), v2.x());
        mixed_moments += Eigen::Vector3d(
            normal.x() * y_coordinates.dot(along_x.vertex_weights),
            normal.y() * z_coordinates.dot(along_y.vertex_weights),
            normal.z() * x_coordinates.dot(along_z.vertex_weights)
        );
    }

    // Normalisations from the reduction: each degree of the integrand brings its own factor.
    integrated_volume /= 6.0;
    first_moments /= 24.0;
    second_moments /= 60.0;
    mixed_moments /= 120.0;

    nice_assert(
        integrated_volume > 0.0,
        "the mesh encloses no positive volume, so it is either open or wound "
        "inward; a solid needs a closed surface facing outward"
    );

    MassProperties properties;
    properties.volume = integrated_volume;
    properties.mass = density * integrated_volume;
    properties.center_of_mass = first_moments / integrated_volume;

    // Inertia about the mesh's coordinate origin. Each diagonal entry is the
    // second moment about the other two axes; each off diagonal entry is minus
    // the corresponding product of inertia.
    Eigen::Matrix3d inertia;
    inertia(0, 0) = density * (second_moments.y() + second_moments.z());
    inertia(1, 1) = density * (second_moments.z() + second_moments.x());
    inertia(2, 2) = density * (second_moments.x() + second_moments.y());
    inertia(0, 1) = inertia(1, 0) = -density * mixed_moments.x();
    inertia(1, 2) = inertia(2, 1) = -density * mixed_moments.y();
    inertia(0, 2) = inertia(2, 0) = -density * mixed_moments.z();

    // Parallel axis theorem, carrying the tensor from the origin to the centre
    // of mass. Subtracting because the origin form already includes the
    // contribution of the whole mass sitting at that offset.
    const Eigen::Vector3d& center = properties.center_of_mass;
    const double mass = properties.mass;
    inertia(0, 0) -= mass * (center.y() * center.y() + center.z() * center.z());
    inertia(1, 1) -= mass * (center.z() * center.z() + center.x() * center.x());
    inertia(2, 2) -= mass * (center.x() * center.x() + center.y() * center.y());
    inertia(0, 1) = inertia(1, 0) = inertia(0, 1) + mass * center.x() * center.y();
    inertia(1, 2) = inertia(2, 1) = inertia(1, 2) + mass * center.y() * center.z();
    inertia(0, 2) = inertia(2, 0) = inertia(0, 2) + mass * center.z() * center.x();

    properties.inertia_about_center = inertia;
    return properties;
}

PrincipalAxes principal_axes(const Eigen::Matrix3d& inertia)
{
    nice_assert(
        (inertia - inertia.transpose()).cwiseAbs().maxCoeff()
            <= symmetry_tolerance * std::max(1.0, inertia.cwiseAbs().maxCoeff()),
        "an inertia tensor must be symmetric"
    );

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(inertia);
    nice_assert(solver.info() == Eigen::Success, "the inertia tensor could not be diagonalised");

    PrincipalAxes result;
    result.moments = solver.eigenvalues();
    result.axes = solver.eigenvectors();

    // The solver returns an orthonormal basis but says nothing about its
    // handedness, and a left handed one is a reflection rather than a
    // rotation. Flipping a single column repairs it without disturbing which
    // moment belongs to which direction.
    if (result.axes.determinant() < 0.0) result.axes.col(0) = -result.axes.col(0);

    return result;
}
} // End namespace cosserat::physics
