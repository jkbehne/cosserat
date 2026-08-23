/**
 * @file test_triangle_mesh_field.cpp
 * @brief Tests for @ref triangle_mesh_field.hpp.
 *
 * A mesh field has an unusually good oracle available: a box made of twelve
 * triangles is exactly the box @ref AnalyticBoxField describes, so the two must
 * agree to machine precision. Anything less means the backend is wrong, not
 * merely approximate. The sphere is different in kind, since a triangulated
 * sphere is genuinely not a sphere, so the comparison there is that the
 * disagreement shrinks as the mesh refines.
 */

#include "math/triangle_mesh_field.hpp"

#include "math/signed_distance_field.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <vector>

namespace cosserat::math {
namespace {

/** Points scattered through a box around the origin, for bulk comparisons. */
std::vector<Eigen::Vector3d> scatter(int count, double reach, unsigned seed = 1)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform(-reach, reach);
    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        points.emplace_back(uniform(generator), uniform(generator), uniform(generator));
    }
    return points;
}

// ---------------------------------------------------------------------------
// Mesh construction
// ---------------------------------------------------------------------------

TEST(MeshBuilders, TheBoxHasTwelveTrianglesAndEightVertices)
{
    const TriangleMesh mesh =
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());

    EXPECT_EQ(mesh.vertices.size(), 8u);
    EXPECT_EQ(mesh.triangles.size(), 12u);
}

TEST(MeshBuilders, TheSphereQuadruplesItsTrianglesPerSubdivision)
{
    for (int subdivisions = 0; subdivisions <= 3; ++subdivisions)
    {
        const TriangleMesh mesh =
            make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, subdivisions);
        const std::size_t expected =
            20u * static_cast<std::size_t>(std::pow(4, subdivisions));
        EXPECT_EQ(mesh.triangles.size(), expected) << "subdivisions " << subdivisions;
    }
}

TEST(MeshBuilders, EverySphereVertexLiesOnTheSphere)
{
    const Eigen::Vector3d center(1.0, -2.0, 0.5);
    const TriangleMesh mesh = make_sphere_mesh(center, 0.75, 2);

    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        EXPECT_NEAR((vertex - center).norm(), 0.75, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

TEST(SurfaceValidation, AcceptsTheBuiltMeshes)
{
    EXPECT_TRUE(validate_closed_surface(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()))
                    .is_closed_surface);
    EXPECT_TRUE(validate_closed_surface(
        make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, 2)).is_closed_surface);
}

TEST(SurfaceValidation, RejectsAnEmptyMesh)
{
    const SurfaceValidity validity = validate_closed_surface(TriangleMesh{});

    EXPECT_FALSE(validity.is_closed_surface);
    EXPECT_FALSE(validity.message.empty());
}

// A hole leaves edges used by only one triangle.
TEST(SurfaceValidation, RejectsAMeshWithAHole)
{
    TriangleMesh mesh = make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    mesh.triangles.pop_back();

    const SurfaceValidity validity = validate_closed_surface(mesh);

    EXPECT_FALSE(validity.is_closed_surface);
    EXPECT_EQ(validity.boundary_edges, 3);
}

// A flipped face traverses its shared edges the same way as its neighbours.
TEST(SurfaceValidation, RejectsAFaceWoundAgainstItsNeighbours)
{
    TriangleMesh mesh = make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    std::swap(mesh.triangles[0](0), mesh.triangles[0](1));

    const SurfaceValidity validity = validate_closed_surface(mesh);

    EXPECT_FALSE(validity.is_closed_surface);
    EXPECT_EQ(validity.inconsistent_edges, 3);
}

// The case every edge test is blind to: reverse the whole mesh and it stays
// perfectly consistent, just inside out, with every distance sign inverted.
// Only the enclosed volume distinguishes it.
TEST(SurfaceValidation, RejectsAMeshThatIsEntirelyInsideOut)
{
    TriangleMesh mesh = make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    for (Eigen::Vector3i& triangle : mesh.triangles)
    {
        std::swap(triangle(1), triangle(2));
    }

    const SurfaceValidity validity = validate_closed_surface(mesh);

    EXPECT_FALSE(validity.is_closed_surface);
    // Every edge test is satisfied.
    EXPECT_EQ(validity.boundary_edges, 0);
    EXPECT_EQ(validity.non_manifold_edges, 0);
    EXPECT_EQ(validity.inconsistent_edges, 0);
    // Only the winding is wrong.
    EXPECT_FALSE(validity.is_wound_outward);
    EXPECT_LT(validity.signed_volume, 0.0);
}

TEST(SurfaceValidation, ReportsTheEnclosedVolume)
{
    const SurfaceValidity box = validate_closed_surface(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 2.0, 3.0)));

    EXPECT_TRUE(box.is_wound_outward);
    // A box of these half extents holds 2 * 4 * 6.
    EXPECT_NEAR(box.signed_volume, 48.0, 1e-12);
}

// ---------------------------------------------------------------------------
// The field, against closed forms
// ---------------------------------------------------------------------------

TEST(TriangleMeshFieldTest, SatisfiesTheFieldConcept)
{
    static_assert(SignedDistanceField<TriangleMeshField>);
    SUCCEED();
}

// Twelve triangles arranged as a box are exactly a box, so this has to agree
// with the closed form to machine precision, not approximately.
TEST(TriangleMeshFieldTest, TheBoxMatchesTheAnalyticFieldExactly)
{
    const Eigen::Vector3d half_extent(1.0, 0.6, 1.4);
    const TriangleMeshField field(
        make_box_mesh(Eigen::Vector3d::Zero(), half_extent), 1.0);
    const AnalyticBoxField exact(Eigen::Vector3d::Zero(), half_extent, 1.0);

    double worst = 0.0;
    for (const Eigen::Vector3d& point : scatter(4000, 2.5))
    {
        worst = std::max(worst, std::abs(field.signed_distance(point).distance
                                         - exact.signed_distance(point).distance));
    }
    EXPECT_LT(worst, 1e-12);
}

TEST(TriangleMeshFieldTest, TheBoxGradientMatchesTheAnalyticFieldOutside)
{
    const TriangleMeshField field(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()), 1.0);
    const AnalyticBoxField exact(
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 1.0);

    for (const Eigen::Vector3d& point : {
             Eigen::Vector3d(2.0, 0.0, 0.0), Eigen::Vector3d(0.0, -3.0, 0.0),
             Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d(1.5, 1.5, 0.0)})
    {
        const SignedDistance from_mesh = field.signed_distance(point);
        const SignedDistance from_exact = exact.signed_distance(point);
        EXPECT_NEAR(from_mesh.gradient.norm(), 1.0, 1e-12);
        EXPECT_LT((from_mesh.gradient - from_exact.gradient).norm(), 1e-9)
            << "at " << point.transpose();
    }
}

TEST(TriangleMeshFieldTest, DistanceIsPositiveOutsideAndNegativeInside)
{
    const TriangleMeshField field(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()), 1.0);

    EXPECT_GT(field.signed_distance({2.0, 0.0, 0.0}).distance, 0.0);
    EXPECT_LT(field.signed_distance({0.0, 0.0, 0.0}).distance, 0.0);
    EXPECT_NEAR(field.signed_distance({2.0, 0.0, 0.0}).distance, 1.0, 1e-12);
    EXPECT_NEAR(field.signed_distance({0.0, 0.0, 0.0}).distance, -1.0, 1e-12);
}

TEST(TriangleMeshFieldTest, TheGradientPointsAwayFromTheSurfaceOnBothSides)
{
    const TriangleMeshField field(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()), 1.0);

    // Outside the +x face, out is +x.
    EXPECT_TRUE(field.signed_distance({2.0, 0.0, 0.0}).gradient
                    .isApprox(Eigen::Vector3d::UnitX(), 1e-9));
    // Just inside it, out is still +x.
    EXPECT_TRUE(field.signed_distance({0.9, 0.0, 0.0}).gradient
                    .isApprox(Eigen::Vector3d::UnitX(), 1e-9));
}

// A triangulated sphere is not a sphere, so the two disagree by the faceting.
// What must hold is that refining the mesh closes the gap.
TEST(TriangleMeshFieldTest, TheSphereConvergesToTheAnalyticFieldAsItRefines)
{
    const AnalyticSphereField exact(Eigen::Vector3d::Zero(), 1.0, 1.0);
    const std::vector<Eigen::Vector3d> points = scatter(2000, 1.8);

    double previous = std::numeric_limits<double>::infinity();
    for (int subdivisions : {1, 2, 3, 4})
    {
        const TriangleMeshField field(
            make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, subdivisions), 1.0);

        double worst = 0.0;
        for (const Eigen::Vector3d& point : points)
        {
            worst = std::max(worst, std::abs(field.signed_distance(point).distance
                                             - exact.signed_distance(point).distance));
        }
        EXPECT_LT(worst, previous) << "subdivisions " << subdivisions;
        previous = worst;
    }
    // By the finest level the faceting is well under a percent of the radius.
    EXPECT_LT(previous, 5e-3);
}

TEST(TriangleMeshFieldTest, ReportsItsMeshSize)
{
    const TriangleMeshField field(
        make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, 2), 1.0);

    EXPECT_EQ(field.num_triangles(), 320);
    EXPECT_EQ(field.num_vertices(), 162);
}

TEST(TriangleMeshFieldTest, TheDomainCoversTheMeshPlusTheMargin)
{
    const TriangleMeshField field(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()), 0.25);

    const Eigen::AlignedBox3d domain = field.domain();
    EXPECT_TRUE(domain.min().isApprox(Eigen::Vector3d::Constant(-1.25), 1e-12));
    EXPECT_TRUE(domain.max().isApprox(Eigen::Vector3d::Constant(1.25), 1e-12));
}

// Moving the mesh moves the field with it, since the field is just the mesh.
TEST(TriangleMeshFieldTest, TranslatingTheMeshTranslatesTheField)
{
    const Eigen::Vector3d shift(3.0, -1.0, 2.0);
    const TriangleMeshField at_origin(
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()), 1.0);
    const TriangleMeshField moved(
        make_box_mesh(shift, Eigen::Vector3d::Ones()), 1.0);

    for (const Eigen::Vector3d& point : scatter(500, 2.0))
    {
        EXPECT_NEAR(at_origin.signed_distance(point).distance,
                    moved.signed_distance(point + shift).distance, 1e-12);
    }
}

TEST(TriangleMeshFieldTest, CopiesShareTheHierarchyAndAgree)
{
    const TriangleMeshField original(
        make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, 2), 1.0);
    const TriangleMeshField copy = original;

    for (const Eigen::Vector3d& point : scatter(200, 1.8))
    {
        EXPECT_DOUBLE_EQ(original.signed_distance(point).distance,
                         copy.signed_distance(point).distance);
    }
}

TEST(TriangleMeshFieldDeathTest, RejectsAnUnusableMesh)
{
    // Empty.
    EXPECT_DEATH({ TriangleMeshField(TriangleMesh{}, 1.0); }, "");

    // Indexing a vertex that does not exist.
    TriangleMesh dangling;
    dangling.vertices = {Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
                         Eigen::Vector3d::UnitY()};
    dangling.triangles = {Eigen::Vector3i(0, 1, 7)};
    EXPECT_DEATH({ TriangleMeshField(dangling, 1.0); }, "");

    // Open, so the sign would be meaningless.
    TriangleMesh open = make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    open.triangles.pop_back();
    EXPECT_DEATH({ TriangleMeshField(open, 1.0); }, "");

    // Inside out, so every sign would be inverted.
    TriangleMesh inverted =
        make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    for (Eigen::Vector3i& triangle : inverted.triangles) std::swap(triangle(1), triangle(2));
    EXPECT_DEATH({ TriangleMeshField(inverted, 1.0); }, "");
}

// Validation can be waived for a mesh already known good. An open mesh then
// builds, and returns a correct distance with a meaningless sign.
TEST(TriangleMeshFieldTest, ValidationCanBeWaived)
{
    TriangleMesh open = make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    open.triangles.pop_back();

    EXPECT_NO_THROW({ TriangleMeshField(open, 1.0, /*validate=*/false); });
}

}  // namespace
}  // namespace cosserat::math
