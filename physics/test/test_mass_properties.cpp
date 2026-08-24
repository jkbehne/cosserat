/**
 * @file test_mass_properties.cpp
 * @brief Tests for @ref mass_properties.hpp.
 *
 * A box and a sphere both have mass properties in closed form, which makes
 * them the right shapes to check against. They test different things: a mesh
 * box *is* a box, so it must agree exactly, while a triangulated sphere is
 * only an approximation, so what must hold there is that refining the mesh
 * closes the gap.
 */

#include "physics/mass_properties.hpp"

#include "math/triangle_mesh_field.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cosserat::physics {
namespace {

constexpr double kTol = 1e-12;
constexpr double kDensity = 1750.0;

/** The inertia a solid box has about its own centre, ascending. */
Eigen::Vector3d box_moments(const Eigen::Vector3d& half_extent, double mass)
{
    Eigen::Vector3d moments(
        mass / 3.0 * (half_extent.y() * half_extent.y()
                      + half_extent.z() * half_extent.z()),
        mass / 3.0 * (half_extent.x() * half_extent.x()
                      + half_extent.z() * half_extent.z()),
        mass / 3.0 * (half_extent.x() * half_extent.x()
                      + half_extent.y() * half_extent.y())
    );
    std::sort(moments.data(), moments.data() + 3);
    return moments;
}

// ---------------------------------------------------------------------------
// A box, where every answer is exact
// ---------------------------------------------------------------------------

TEST(MassPropertiesTest, TheBoxVolumeIsExact)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MassProperties properties = compute_mass_properties(
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent), kDensity);

    const double expected =
        8.0 * half_extent.x() * half_extent.y() * half_extent.z();
    EXPECT_NEAR(properties.volume, expected, kTol);
    EXPECT_NEAR(properties.mass, kDensity * expected, 1e-9);
}

TEST(MassPropertiesTest, TheBoxCentreIsWhereItWasDrawn)
{
    const Eigen::Vector3d center(2.0, -3.0, 1.0);
    const MassProperties properties = compute_mass_properties(
        math::make_box_mesh(center, Eigen::Vector3d(0.3, 0.5, 0.7)), kDensity);

    EXPECT_LT((properties.center_of_mass - center).norm(), 1e-12);
}

TEST(MassPropertiesTest, TheBoxInertiaMatchesTheClosedForm)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const MassProperties properties = compute_mass_properties(
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent), kDensity);

    const double mass = properties.mass;
    EXPECT_NEAR(properties.inertia_about_center(0, 0),
                mass / 3.0 * (half_extent.y() * half_extent.y()
                              + half_extent.z() * half_extent.z()), 1e-9);
    EXPECT_NEAR(properties.inertia_about_center(1, 1),
                mass / 3.0 * (half_extent.x() * half_extent.x()
                              + half_extent.z() * half_extent.z()), 1e-9);
    EXPECT_NEAR(properties.inertia_about_center(2, 2),
                mass / 3.0 * (half_extent.x() * half_extent.x()
                              + half_extent.y() * half_extent.y()), 1e-9);
}

// A box drawn on the axes is already in its principal frame, so the products
// of inertia must vanish rather than merely be small.
TEST(MassPropertiesTest, AnAxisAlignedBoxHasNoProductsOfInertia)
{
    const MassProperties properties = compute_mass_properties(
        math::make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.3, 0.5, 0.7)),
        kDensity);

    EXPECT_NEAR(properties.inertia_about_center(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(properties.inertia_about_center(0, 2), 0.0, 1e-9);
    EXPECT_NEAR(properties.inertia_about_center(1, 2), 0.0, 1e-9);
}

// The inertia is reported about the centre of mass, so moving the mesh must
// move the centre and leave the tensor alone.
TEST(MassPropertiesTest, TranslationMovesTheCentreAndNotTheInertia)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const Eigen::Vector3d shift(2.0, -3.0, 1.0);

    const MassProperties at_origin = compute_mass_properties(
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent), kDensity);
    const MassProperties moved = compute_mass_properties(
        math::make_box_mesh(shift, half_extent), kDensity);

    EXPECT_LT((moved.center_of_mass - shift).norm(), 1e-12);
    EXPECT_LT((moved.inertia_about_center - at_origin.inertia_about_center)
                  .cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_NEAR(moved.volume, at_origin.volume, kTol);
}

TEST(MassPropertiesTest, MassScalesWithDensityAndInertiaWithIt)
{
    const auto mesh =
        math::make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.3, 0.5, 0.7));
    const MassProperties light = compute_mass_properties(mesh, 1.0);
    const MassProperties heavy = compute_mass_properties(mesh, 7.0);

    EXPECT_NEAR(heavy.volume, light.volume, kTol);
    EXPECT_NEAR(heavy.mass, 7.0 * light.mass, 1e-9);
    EXPECT_LT((heavy.inertia_about_center - 7.0 * light.inertia_about_center)
                  .cwiseAbs().maxCoeff(), 1e-9);
}

// The tensor is a physical quantity, so rotating the body must rotate it the
// same way: R I R^T.
TEST(MassPropertiesTest, RotatingTheMeshRotatesTheInertiaTensor)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
            .toRotationMatrix();

    const MassProperties square = compute_mass_properties(
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent), kDensity);

    math::TriangleMesh turned =
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent);
    for (Eigen::Vector3d& vertex : turned.vertices) vertex = rotation * vertex;
    const MassProperties rotated = compute_mass_properties(turned, kDensity);

    const Eigen::Matrix3d expected =
        rotation * square.inertia_about_center * rotation.transpose();
    EXPECT_LT((rotated.inertia_about_center - expected).cwiseAbs().maxCoeff(), 1e-8);
    EXPECT_NEAR(rotated.volume, square.volume, 1e-12);
}

// ---------------------------------------------------------------------------
// A sphere, where the mesh only approximates the solid
// ---------------------------------------------------------------------------

TEST(MassPropertiesTest, TheSphereConvergesToItsClosedForm)
{
    const double radius = 0.4;
    const double exact_volume = 4.0 / 3.0 * std::numbers::pi * std::pow(radius, 3);
    const double exact_mass = kDensity * exact_volume;
    const double exact_moment = 0.4 * exact_mass * radius * radius;

    double previous_volume_error = std::numeric_limits<double>::infinity();
    double previous_moment_error = std::numeric_limits<double>::infinity();
    for (int subdivisions : {1, 2, 3, 4})
    {
        const MassProperties properties = compute_mass_properties(
            math::make_sphere_mesh(Eigen::Vector3d::Zero(), radius, subdivisions),
            kDensity);

        const double volume_error = std::abs(properties.volume - exact_volume);
        const double moment_error =
            std::abs(properties.inertia_about_center(0, 0) - exact_moment);

        EXPECT_LT(volume_error, previous_volume_error) << subdivisions;
        EXPECT_LT(moment_error, previous_moment_error) << subdivisions;
        previous_volume_error = volume_error;
        previous_moment_error = moment_error;
    }
    // A faceted sphere always falls short of the true one, so the error never
    // reaches zero; by the finest level it is under half a percent.
    EXPECT_LT(previous_volume_error / exact_volume, 5e-3);
    EXPECT_LT(previous_moment_error / exact_moment, 5e-3);
}

// A sphere is isotropic, so all three moments coincide and every product of
// inertia vanishes, up to the faceting.
TEST(MassPropertiesTest, TheSphereIsNearlyIsotropic)
{
    const MassProperties properties = compute_mass_properties(
        math::make_sphere_mesh(Eigen::Vector3d::Zero(), 0.4, 4), kDensity);

    const Eigen::Matrix3d& inertia = properties.inertia_about_center;
    const double average = inertia.trace() / 3.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(inertia(axis, axis) / average, 1.0, 1e-3);
    }
    EXPECT_LT(std::abs(inertia(0, 1)) / average, 1e-3);
    EXPECT_LT(std::abs(inertia(0, 2)) / average, 1e-3);
    EXPECT_LT(std::abs(inertia(1, 2)) / average, 1e-3);
}

// ---------------------------------------------------------------------------
// Rejected input
// ---------------------------------------------------------------------------

TEST(MassPropertiesDeathTest, RejectsANonPositiveDensity)
{
    const auto mesh =
        math::make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());

    EXPECT_DEATH({ compute_mass_properties(mesh, 0.0); }, "");
    EXPECT_DEATH({ compute_mass_properties(mesh, -1.0); }, "");
}

TEST(MassPropertiesDeathTest, RejectsAnEmptyMesh)
{
    EXPECT_DEATH({ compute_mass_properties(math::TriangleMesh{}, 1.0); }, "");
}

// An inward wound mesh integrates to a negative volume, which would mean a
// negative mass, so it is rejected rather than silently propagated.
TEST(MassPropertiesDeathTest, RejectsAMeshThatIsInsideOut)
{
    math::TriangleMesh inverted =
        math::make_box_mesh(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
    for (Eigen::Vector3i& triangle : inverted.triangles)
    {
        std::swap(triangle(1), triangle(2));
    }

    EXPECT_DEATH({ compute_mass_properties(inverted, 1.0); }, "");
}

// ---------------------------------------------------------------------------
// Principal axes
// ---------------------------------------------------------------------------

TEST(PrincipalAxesTest, AnAlreadyDiagonalTensorKeepsItsMoments)
{
    const Eigen::Vector3d diagonal(2.0, 5.0, 9.0);
    const PrincipalAxes axes =
        principal_axes(Eigen::Matrix3d(diagonal.asDiagonal()));

    EXPECT_LT((axes.moments - diagonal).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(PrincipalAxesTest, TheMomentsComeBackAscending)
{
    const Eigen::Vector3d diagonal(9.0, 2.0, 5.0);
    const PrincipalAxes axes =
        principal_axes(Eigen::Matrix3d(diagonal.asDiagonal()));

    EXPECT_LE(axes.moments(0), axes.moments(1));
    EXPECT_LE(axes.moments(1), axes.moments(2));
}

TEST(PrincipalAxesTest, TheAxesAreAlwaysARightHandedRotation)
{
    for (double angle : {0.0, 0.4, 1.9, 3.0})
    {
        const Eigen::Matrix3d rotation =
            Eigen::AngleAxisd(angle, Eigen::Vector3d(1.0, -2.0, 0.5).normalized())
                .toRotationMatrix();
        const Eigen::Matrix3d tensor =
            rotation * Eigen::Vector3d(2.0, 5.0, 9.0).asDiagonal() * rotation.transpose();

        const PrincipalAxes axes = principal_axes(tensor);

        EXPECT_LT((axes.axes * axes.axes.transpose() - Eigen::Matrix3d::Identity())
                      .cwiseAbs().maxCoeff(), 1e-12);
        EXPECT_NEAR(axes.axes.determinant(), 1.0, 1e-12)
            << "a left handed basis is a reflection, not a rotation";
    }
}

// Diagonalising and rebuilding must return the tensor unchanged.
TEST(PrincipalAxesTest, TheDecompositionReconstructsTheTensor)
{
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
            .toRotationMatrix();
    const Eigen::Matrix3d tensor =
        rotation * Eigen::Vector3d(2.0, 5.0, 9.0).asDiagonal() * rotation.transpose();

    const PrincipalAxes axes = principal_axes(tensor);
    const Eigen::Matrix3d rebuilt =
        axes.axes * axes.moments.asDiagonal() * axes.axes.transpose();

    EXPECT_LT((rebuilt - tensor).cwiseAbs().maxCoeff(), 1e-12);
}

// The point of the whole exercise: a box drawn in some arbitrary orientation
// has the same principal moments as the same box drawn on the axes.
TEST(PrincipalAxesTest, ARotatedBoxRecoversItsUnrotatedMoments)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
            .toRotationMatrix();

    math::TriangleMesh turned =
        math::make_box_mesh(Eigen::Vector3d::Zero(), half_extent);
    for (Eigen::Vector3d& vertex : turned.vertices) vertex = rotation * vertex;

    const MassProperties properties = compute_mass_properties(turned, kDensity);
    const PrincipalAxes axes = principal_axes(properties.inertia_about_center);

    EXPECT_LT((axes.moments - box_moments(half_extent, properties.mass))
                  .cwiseAbs().maxCoeff(), 1e-8);
}

TEST(PrincipalAxesDeathTest, RejectsAnAsymmetricTensor)
{
    Eigen::Matrix3d lopsided = Eigen::Matrix3d::Identity();
    lopsided(0, 1) = 1.0;

    EXPECT_DEATH({ principal_axes(lopsided); }, "");
}

}  // namespace
}  // namespace cosserat::physics
