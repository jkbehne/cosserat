/**
 * @file test_mesh_body.cpp
 * @brief Tests for @ref mesh_body.hpp.
 *
 * The demanding case throughout is a box drawn rotated and far from the
 * origin. Everything the body has to get right shows up there at once: it must
 * recover the centroid as its position, the principal directions as its frame,
 * the unrotated moments as its inertia, and a field that still describes the
 * box where the box actually is.
 */

#include <cosserat/physics/mesh_body.hpp>

#include <cosserat/physics/rod_mesh_contact.hpp>
#include <cosserat/physics/rods.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cosserat::physics {
namespace {

constexpr double kDensity = 1750.0;
constexpr double kMargin = 0.2;

/** A box mesh, optionally rotated and moved away from the origin. */
math::TriangleMesh placed_box(
    const Eigen::Vector3d& half_extent,
    const Eigen::Matrix3d& rotation = Eigen::Matrix3d::Identity(),
    const Eigen::Vector3d& offset = Eigen::Vector3d::Zero()
)
{
    math::TriangleMesh mesh =
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent);
    for (Eigen::Vector3d& vertex : mesh.vertices)
    {
        vertex = rotation * vertex + offset;
    }
    return mesh;
}

/** A rotation used repeatedly, chosen to be aligned with nothing. */
Eigen::Matrix3d awkward_rotation()
{
    return Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
        .toRotationMatrix();
}

// ---------------------------------------------------------------------------
// What the body derives from its mesh
// ---------------------------------------------------------------------------

TEST(MeshBodyTest, TakesItsPositionFromTheCentreOfMass)
{
    const Eigen::Vector3d offset(2.0, -3.0, 1.0);
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation(), offset),
        kDensity, kMargin, true);

    EXPECT_LT((body.positions().row(0).transpose() - offset).norm(), 1e-9);
}

TEST(MeshBodyTest, TakesItsMassFromTheEnclosedVolume)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MeshBody body(placed_box(half_extent), kDensity, kMargin, true);

    const double expected =
        8.0 * half_extent.x() * half_extent.y() * half_extent.z();
    EXPECT_NEAR(body.volume(), expected, 1e-12);
    EXPECT_NEAR(body.masses()(0), kDensity * expected, 1e-9);
}

// The moments must not depend on how the mesh happened to be drawn.
TEST(MeshBodyTest, RecoversTheSameMomentsHoweverTheMeshIsPlaced)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);

    const MeshBody square(placed_box(half_extent), kDensity, kMargin, true);
    const MeshBody awkward(
        placed_box(half_extent, awkward_rotation(), Eigen::Vector3d(2.0, -3.0, 1.0)),
        kDensity, kMargin, true);

    EXPECT_LT((square.principal_moments() - awkward.principal_moments())
                  .cwiseAbs().maxCoeff(), 1e-8);
}

TEST(MeshBodyTest, TheMomentsMatchTheClosedFormForABox)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MeshBody body(
        placed_box(half_extent, awkward_rotation()), kDensity, kMargin, true);

    const double mass = body.masses()(0);
    Eigen::Vector3d expected(
        mass / 3.0 * (half_extent.y() * half_extent.y()
                      + half_extent.z() * half_extent.z()),
        mass / 3.0 * (half_extent.x() * half_extent.x()
                      + half_extent.z() * half_extent.z()),
        mass / 3.0 * (half_extent.x() * half_extent.x()
                      + half_extent.y() * half_extent.y())
    );
    std::sort(expected.data(), expected.data() + 3);

    EXPECT_LT((body.principal_moments() - expected).cwiseAbs().maxCoeff(), 1e-8);
}

TEST(MeshBodyTest, TheStoredFrameIsARotation)
{
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation()),
        kDensity, kMargin, true);

    const Eigen::Matrix3d frame = body.frames()[0];
    EXPECT_LT((frame * frame.transpose() - Eigen::Matrix3d::Identity())
                  .cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_NEAR(frame.determinant(), 1.0, 1e-12);
}

// The inertia the base stores must be the diagonal of the principal moments,
// which is only meaningful because the mesh was carried into that frame.
TEST(MeshBodyTest, TheStoredInertiaIsTheDiagonalOfTheMoments)
{
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation()),
        kDensity, kMargin, true);

    const Eigen::Matrix3d expected = body.principal_moments().asDiagonal();
    EXPECT_LT((body.mass_2nd_moments()[0] - expected).cwiseAbs().maxCoeff(), 1e-9);
}

// ---------------------------------------------------------------------------
// The carried mesh
// ---------------------------------------------------------------------------

TEST(MeshBodyTest, TheBodyFrameMeshIsCentredOnTheOrigin)
{
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation(),
                   Eigen::Vector3d(2.0, -3.0, 1.0)),
        kDensity, kMargin, true);

    const MassProperties carried =
        compute_mass_properties(body.body_frame_mesh(), kDensity);
    EXPECT_LT(carried.center_of_mass.norm(), 1e-12);
}

// Carried into the principal frame, the products of inertia must vanish.
TEST(MeshBodyTest, TheBodyFrameMeshIsDiagonalInInertia)
{
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation()),
        kDensity, kMargin, true);

    const Eigen::Matrix3d inertia =
        compute_mass_properties(body.body_frame_mesh(), kDensity).inertia_about_center;
    const double scale = inertia.diagonal().cwiseAbs().maxCoeff();

    EXPECT_LT(std::abs(inertia(0, 1)) / scale, 1e-9);
    EXPECT_LT(std::abs(inertia(0, 2)) / scale, 1e-9);
    EXPECT_LT(std::abs(inertia(1, 2)) / scale, 1e-9);
}

TEST(MeshBodyTest, TheCarriedMeshKeepsItsTrianglesAndVolume)
{
    const math::TriangleMesh mesh =
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation(),
                   Eigen::Vector3d(2.0, -3.0, 1.0));
    const MeshBody body(mesh, kDensity, kMargin, true);

    EXPECT_EQ(body.body_frame_mesh().triangles.size(), mesh.triangles.size());
    EXPECT_EQ(body.body_frame_mesh().vertices.size(), mesh.vertices.size());
    EXPECT_NEAR(compute_mass_properties(body.body_frame_mesh(), kDensity).volume,
                body.volume(), 1e-12);
}

// ---------------------------------------------------------------------------
// The field
// ---------------------------------------------------------------------------

TEST(MeshBodyTest, TheFieldDescribesTheBoxInBodyCoordinates)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MeshBody body(placed_box(half_extent), kDensity, kMargin, true);
    const math::FieldQuery query = body.field_query();

    // The body frame axes are the principal ones, ordered by moment rather
    // than by extent, so probe along each and check the distances are the
    // three half extents in some order.
    Eigen::Vector3d found;
    for (int axis = 0; axis < 3; ++axis)
    {
        Eigen::Vector3d probe = Eigen::Vector3d::Zero();
        probe(axis) = 10.0;
        // Distance from far away is (probe - half extent along that axis).
        found(axis) = 10.0 - query(probe).distance;
    }
    Eigen::Vector3d expected = half_extent;
    std::sort(found.data(), found.data() + 3);
    std::sort(expected.data(), expected.data() + 3);
    EXPECT_LT((found - expected).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(MeshBodyTest, TheFieldIsNegativeAtTheBodyOrigin)
{
    const MeshBody body(
        placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), awkward_rotation(),
                   Eigen::Vector3d(2.0, -3.0, 1.0)),
        kDensity, kMargin, true);

    // The origin of the body frame is the centre of mass, which for a box is
    // inside it.
    EXPECT_LT(body.field_query()(Eigen::Vector3d::Zero()).distance, 0.0);
}

TEST(MeshBodyTest, TheDomainCoversTheBodyPlusTheMargin)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MeshBody body(placed_box(half_extent), kDensity, kMargin, true);

    const Eigen::AlignedBox3d domain = body.field_domain();
    // Whatever the axis order, the box plus its margin must fit.
    EXPECT_GT(domain.min().minCoeff(), -(half_extent.maxCoeff() + kMargin) - 1e-9);
    EXPECT_LT(domain.max().maxCoeff(), half_extent.maxCoeff() + kMargin + 1e-9);
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_GT(domain.max()(axis) - domain.min()(axis), 2.0 * kMargin);
    }
}

// ---------------------------------------------------------------------------
// Driving contact through a real body
// ---------------------------------------------------------------------------

TEST(MeshBodyTest, SatisfiesTheContactableMeshBodyConcept)
{
    static_assert(ContactableMeshBody<MeshBody>);
    SUCCEED();
}

// The whole point: a rod pressed against a box drawn rotated and offset must
// be pushed straight out along that box's real face normal.
TEST(MeshBodyTest, ARodIsPushedOutAlongTheRealFaceNormal)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const Eigen::Matrix3d rotation = awkward_rotation();
    const Eigen::Vector3d offset(2.0, -3.0, 1.0);
    MeshBody body(placed_box(half_extent, rotation, offset), kDensity, kMargin, true);

    // Laid against the +x face of the box, in the box's own orientation.
    const Eigen::Vector3d start =
        offset + rotation * Eigen::Vector3d(half_extent.x() + 0.005, -0.4, 0.0);
    CosseratRod rod = straight_cosserat_rod(
        8, start, rotation * Eigen::Vector3d::UnitY(),
        rotation * Eigen::Vector3d::UnitZ(), 0.8, 0.05, 1000.0, 1.0e6, false, 1e-12);

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    const Eigen::Vector3d total =
        rod.external_forces().colwise().sum().transpose();
    ASSERT_GT(total.norm(), 0.0);
    const Eigen::Vector3d outward = rotation * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(total.normalized().dot(outward), 1.0, 1e-4);
}

TEST(MeshBodyTest, ContactConservesForceAcrossThePair)
{
    const Eigen::Matrix3d rotation = awkward_rotation();
    const Eigen::Vector3d offset(2.0, -3.0, 1.0);
    MeshBody body(placed_box(Eigen::Vector3d(0.3, 0.5, 0.7), rotation, offset),
                  kDensity, kMargin, true);

    const Eigen::Vector3d start = offset + rotation * Eigen::Vector3d(0.305, -0.4, 0.0);
    CosseratRod rod = straight_cosserat_rod(
        8, start, rotation * Eigen::Vector3d::UnitY(),
        rotation * Eigen::Vector3d::UnitZ(), 0.8, 0.05, 1000.0, 1.0e6, false, 1e-12);

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    const Eigen::RowVector3d total = rod.external_forces().colwise().sum()
        + body.external_forces().colwise().sum();
    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-9);
}

TEST(MeshBodyTest, ARodWellAwayFeelsNothing)
{
    MeshBody body(placed_box(Eigen::Vector3d(0.3, 0.5, 0.7)), kDensity, kMargin, true);
    CosseratRod rod = straight_cosserat_rod(
        8, Eigen::Vector3d(20.0, -0.4, 0.0), Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ(), 0.8, 0.05, 1000.0, 1.0e6, false, 1e-12);

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    EXPECT_LT(rod.external_forces().cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT(body.external_forces().cwiseAbs().maxCoeff(), 1e-12);
}

// ---------------------------------------------------------------------------
// Rejected input
// ---------------------------------------------------------------------------

TEST(MeshBodyDeathTest, RejectsAnUnusableMeshOrDensity)
{
    const math::TriangleMesh good = placed_box(Eigen::Vector3d(0.3, 0.5, 0.7));

    EXPECT_DEATH({ MeshBody(good, 0.0, kMargin, true); }, "");
    EXPECT_DEATH({ MeshBody(good, kDensity, 0.0, true); }, "");
    EXPECT_DEATH({ MeshBody(math::TriangleMesh{}, kDensity, kMargin, true); }, "");

    math::TriangleMesh inverted = good;
    for (Eigen::Vector3i& triangle : inverted.triangles)
    {
        std::swap(triangle(1), triangle(2));
    }
    EXPECT_DEATH({ MeshBody(inverted, kDensity, kMargin, true); }, "");
}

TEST(MeshBodyTest, AcceptsASphereAsWellAsABox)
{
    const MeshBody body(
        math::make_sphere_mesh(Eigen::Vector3d(1.0, 2.0, 3.0), 0.4, 3),
        kDensity, kMargin, true);

    EXPECT_LT((body.positions().row(0).transpose() - Eigen::Vector3d(1.0, 2.0, 3.0))
                  .norm(), 1e-3);
    // Isotropic, so the three moments should nearly coincide.
    const Eigen::Vector3d moments = body.principal_moments();
    EXPECT_NEAR(moments(2) / moments(0), 1.0, 1e-2);
}

}  // namespace
}  // namespace cosserat::physics
