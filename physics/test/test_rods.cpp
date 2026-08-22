#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <vector>

namespace cosserat::physics {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kTol = 1e-12;

// Reference rod parameters, shared by most tests.
constexpr std::int64_t kElements = 10;
constexpr double kLength = 1.0;
constexpr double kRadius = 0.05;
constexpr double kDensity = 1000.0;
constexpr double kYoungs = 1.0e6;

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return ::testing::AssertionFailure()
            << "shape mismatch: " << a.rows() << "x" << a.cols() << " vs "
            << b.rows() << "x" << b.cols();
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err < tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs diff " << err << " >= " << tol;
}

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    return Near(a, b, kTol);
}

CosseratRod make_rod(bool respect_radii)
{
    return straight_cosserat_rod(
        kElements, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 0.0), kLength, kRadius, kDensity, kYoungs,
        respect_radii, kTol);
}

CosseratRod make_rod() { return make_rod(false); }

/**
 * Applies a fixed, reproducible deformation: a quadratic bend in x, a linear
 * drift in y, a stretch in z, a progressive twist of the frames, and non-zero
 * translational and angular rates. Mirrors the configuration used to generate
 * the reference values in the parity tests below.
 */
void deform(CosseratRod& rod)
{
    Vector3DStack& positions = rod.mutable_positions();
    for (Eigen::Index i = 0; i < positions.rows(); ++i)
    {
        const double t = static_cast<double>(i);
        positions(i, 0) += 0.03 * t * t;
        positions(i, 1) += 0.02 * t;
        positions(i, 2) *= 1.05;
    }

    Matrix3DStack& frames = rod.mutable_frames();
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const double angle = 0.15 * static_cast<double>(i + 1);
        frames[i] =
            (Eigen::AngleAxisd(angle, axis).toRotationMatrix() * frames[i]).eval();
    }

    Vector3DStack& velocities = rod.mutable_velocities();
    for (Eigen::Index i = 0; i < velocities.rows(); ++i)
    {
        velocities.row(i) << 0.1 * i, -0.05 * i, 0.2 + 0.01 * i;
    }

    Vector3DStack& omegas = rod.mutable_angular_velocities();
    for (Eigen::Index i = 0; i < omegas.rows(); ++i)
    {
        omegas.row(i) << 0.3 - 0.1 * i, 0.2, 0.05 * i;
    }
}

// ---------------------------------------------------------------------------
// Construction: domain sizes
// ---------------------------------------------------------------------------

TEST(StraightRod, DomainCountsFollowTheNodeCount)
{
    const CosseratRod rod = make_rod();

    EXPECT_EQ(rod.num_nodes(), kElements + 1);
    EXPECT_EQ(rod.num_elements(), kElements);
    EXPECT_EQ(rod.num_voronoi(), kElements - 1);
}

TEST(StraightRod, EveryStackIsSizedToItsDomain)
{
    const CosseratRod rod = make_rod();
    const Eigen::Index nodes = rod.num_nodes();
    const Eigen::Index elements = rod.num_elements();
    const Eigen::Index voronoi = rod.num_voronoi();

    EXPECT_EQ(rod.positions().rows(), nodes);
    EXPECT_EQ(rod.velocities().rows(), nodes);
    EXPECT_EQ(rod.accelerations().rows(), nodes);
    EXPECT_EQ(rod.internal_forces().rows(), nodes);
    EXPECT_EQ(rod.external_forces().rows(), nodes);
    EXPECT_EQ(rod.masses().size(), nodes);

    EXPECT_EQ(static_cast<Eigen::Index>(rod.frames().size()), elements);
    EXPECT_EQ(static_cast<Eigen::Index>(rod.mass_2nd_moments().size()), elements);
    EXPECT_EQ(static_cast<Eigen::Index>(rod.inv_mass_2nd_moments().size()), elements);
    EXPECT_EQ(static_cast<Eigen::Index>(rod.shearing_matrices().size()), elements);
    EXPECT_EQ(rod.angular_velocities().rows(), elements);
    EXPECT_EQ(rod.angular_accelerations().rows(), elements);
    EXPECT_EQ(rod.internal_torques().rows(), elements);
    EXPECT_EQ(rod.external_torques().rows(), elements);
    EXPECT_EQ(rod.tangents().rows(), elements);
    EXPECT_EQ(rod.sigmas().rows(), elements);
    EXPECT_EQ(rod.rest_sigmas().rows(), elements);
    EXPECT_EQ(rod.internal_stresses().rows(), elements);
    EXPECT_EQ(rod.radii().size(), elements);
    EXPECT_EQ(rod.densities().size(), elements);
    EXPECT_EQ(rod.volumes().size(), elements);
    EXPECT_EQ(rod.lengths().size(), elements);
    EXPECT_EQ(rod.rest_lengths().size(), elements);
    EXPECT_EQ(rod.dilatations().size(), elements);
    EXPECT_EQ(rod.dilatation_rates().size(), elements);

    EXPECT_EQ(static_cast<Eigen::Index>(rod.bending_matrices().size()), voronoi);
    EXPECT_EQ(rod.kappas().rows(), voronoi);
    EXPECT_EQ(rod.rest_kappas().rows(), voronoi);
    EXPECT_EQ(rod.internal_couples().rows(), voronoi);
    EXPECT_EQ(rod.voronoi_dilatations().size(), voronoi);
    EXPECT_EQ(rod.voronoi_rest_lengths().size(), voronoi);
}

// ---------------------------------------------------------------------------
// Construction: geometry against closed-form values
// ---------------------------------------------------------------------------

TEST(StraightRod, NodesAreEvenlySpacedAlongTheDirection)
{
    const CosseratRod rod = make_rod();
    const double spacing = kLength / static_cast<double>(kElements);

    for (Eigen::Index i = 0; i < rod.num_nodes(); ++i)
    {
        EXPECT_NEAR(rod.positions()(i, 0), 0.0, kTol) << "node " << i;
        EXPECT_NEAR(rod.positions()(i, 1), 0.0, kTol) << "node " << i;
        EXPECT_NEAR(rod.positions()(i, 2), spacing * static_cast<double>(i), 1e-12)
            << "node " << i;
    }
}

TEST(StraightRod, RestLengthsDivideTheTotalLength)
{
    const CosseratRod rod = make_rod();
    const double expected = kLength / static_cast<double>(kElements);

    EXPECT_TRUE(Near(rod.rest_lengths(),
                     Eigen::VectorXd::Constant(kElements, expected)));
    EXPECT_TRUE(Near(rod.lengths(),
                     Eigen::VectorXd::Constant(kElements, expected), 1e-12));
}

// Half the sum of two adjacent rest lengths, so a uniform rod has Voronoi rest
// lengths equal to its element rest lengths.
TEST(StraightRod, VoronoiRestLengthsAverageAdjacentElements)
{
    const CosseratRod rod = make_rod();
    const double expected = kLength / static_cast<double>(kElements);

    EXPECT_TRUE(Near(rod.voronoi_rest_lengths(),
                     Eigen::VectorXd::Constant(kElements - 1, expected)));
}

TEST(StraightRod, TangentsPointAlongTheDirection)
{
    const CosseratRod rod = make_rod();

    for (Eigen::Index i = 0; i < rod.num_elements(); ++i)
    {
        EXPECT_TRUE(Near(rod.tangents().row(i), Eigen::RowVector3d(0.0, 0.0, 1.0)))
            << "element " << i;
    }
}

TEST(StraightRod, FramesAreOrthogonalAndAlignedWithTheTangent)
{
    const CosseratRod rod = make_rod();

    for (std::size_t i = 0; i < rod.frames().size(); ++i)
    {
        EXPECT_TRUE(math::is_orthogonal(rod.frames()[i], 1e-12)) << "element " << i;
        // For an undeformed straight rod the frame's third row is the tangent,
        // though this is not asserted by the class in general.
        EXPECT_TRUE(Near(rod.frames()[i].row(2).transpose(),
                         Eigen::Vector3d(0.0, 0.0, 1.0)))
            << "element " << i;
    }
}

TEST(StraightRod, VolumeMatchesTheCylinderFormula)
{
    const CosseratRod rod = make_rod();
    const double rest_length = kLength / static_cast<double>(kElements);
    const double expected = std::numbers::pi * kRadius * kRadius * rest_length;

    EXPECT_TRUE(Near(rod.volumes(),
                     Eigen::VectorXd::Constant(kElements, expected), 1e-15));
    EXPECT_TRUE(Near(rod.radii(),
                     Eigen::VectorXd::Constant(kElements, kRadius), 1e-15));
}

// Ends carry half an element's mass, interior nodes a full one, and the total
// is the mass of the whole rod.
TEST(StraightRod, MassIsLumpedOntoNodesAndConserved)
{
    const CosseratRod rod = make_rod();
    const double rest_length = kLength / static_cast<double>(kElements);
    const double element_mass =
        kDensity * std::numbers::pi * kRadius * kRadius * rest_length;

    EXPECT_NEAR(rod.masses()(0), 0.5 * element_mass, 1e-15);
    EXPECT_NEAR(rod.masses()(rod.num_nodes() - 1), 0.5 * element_mass, 1e-15);
    for (Eigen::Index i = 1; i < rod.num_nodes() - 1; ++i)
    {
        EXPECT_NEAR(rod.masses()(i), element_mass, 1e-15) << "node " << i;
    }
    EXPECT_NEAR(rod.masses().sum(),
                kDensity * std::numbers::pi * kRadius * kRadius * kLength, 1e-14);
}

// ---------------------------------------------------------------------------
// Construction: inertia and stiffness against closed-form values
// ---------------------------------------------------------------------------

TEST(StraightRod, MassSecondMomentIsScaledByDensityAndRestLength)
{
    const CosseratRod rod = make_rod();
    const double rest_length = kLength / static_cast<double>(kElements);
    const double area_moment = std::numbers::pi * std::pow(kRadius, 4) / 4.0;
    const double scale = kDensity * rest_length;

    for (std::size_t i = 0; i < rod.mass_2nd_moments().size(); ++i)
    {
        const Eigen::Matrix3d& moment = rod.mass_2nd_moments()[i];
        EXPECT_NEAR(moment(0, 0), area_moment * scale, 1e-15) << "element " << i;
        EXPECT_NEAR(moment(1, 1), area_moment * scale, 1e-15) << "element " << i;
        EXPECT_NEAR(moment(2, 2), 2.0 * area_moment * scale, 1e-15) << "element " << i;
        EXPECT_NEAR(moment(0, 1), 0.0, kTol) << "element " << i;
    }
}

TEST(StraightRod, InverseMassSecondMomentInvertsIt)
{
    const CosseratRod rod = make_rod();

    for (std::size_t i = 0; i < rod.mass_2nd_moments().size(); ++i)
    {
        const Eigen::Matrix3d product =
            rod.mass_2nd_moments()[i] * rod.inv_mass_2nd_moments()[i];
        EXPECT_TRUE(Near(product, Eigen::Matrix3d::Identity(), 1e-12))
            << "element " << i;
    }
}

// The bending stiffness uses the second moment of area, free of density and
// length. Scaling it by the mass factor would inflate it by density times rest
// length.
TEST(StraightRod, BendingMatrixUsesTheAreaMomentNotTheMassMoment)
{
    const CosseratRod rod = make_rod();
    const double area_moment = std::numbers::pi * std::pow(kRadius, 4) / 4.0;
    const double shear_modulus = kYoungs / 3.0;

    for (std::size_t i = 0; i < rod.bending_matrices().size(); ++i)
    {
        const Eigen::Matrix3d& bend = rod.bending_matrices()[i];
        EXPECT_NEAR(bend(0, 0), kYoungs * area_moment, 1e-9) << "voronoi " << i;
        EXPECT_NEAR(bend(1, 1), kYoungs * area_moment, 1e-9) << "voronoi " << i;
        EXPECT_NEAR(bend(2, 2), shear_modulus * 2.0 * area_moment, 1e-9)
            << "voronoi " << i;
    }
}

TEST(StraightRod, ShearMatrixUsesTheCrossSectionalArea)
{
    const CosseratRod rod = make_rod();
    const double area = std::numbers::pi * kRadius * kRadius;
    const double shear_modulus = kYoungs / 3.0;

    for (std::size_t i = 0; i < rod.shearing_matrices().size(); ++i)
    {
        const Eigen::Matrix3d& shear = rod.shearing_matrices()[i];
        EXPECT_NEAR(shear(0, 0), CosseratRod::alpha_c * shear_modulus * area, 1e-9)
            << "element " << i;
        EXPECT_NEAR(shear(1, 1), CosseratRod::alpha_c * shear_modulus * area, 1e-9)
            << "element " << i;
        EXPECT_NEAR(shear(2, 2), kYoungs * area, 1e-9) << "element " << i;
    }
}

// ---------------------------------------------------------------------------
// The rest configuration
// ---------------------------------------------------------------------------

// A rod built at its rest configuration must report unit dilatation on both
// domains. A Voronoi rest length that is not the average of its neighbours
// shows up here first.
TEST(StraightRod, RestConfigurationHasUnitDilatations)
{
    const CosseratRod rod = make_rod();

    EXPECT_TRUE(Near(rod.dilatations(), Eigen::VectorXd::Ones(kElements), 1e-14));
    EXPECT_TRUE(Near(rod.voronoi_dilatations(),
                     Eigen::VectorXd::Ones(kElements - 1), 1e-14));
}

TEST(StraightRod, RestConfigurationHasZeroStrains)
{
    const CosseratRod rod = make_rod();

    EXPECT_TRUE(Near(rod.sigmas(), Vector3DStack::Zero(kElements, 3), 1e-14));
    EXPECT_TRUE(Near(rod.kappas(), Vector3DStack::Zero(kElements - 1, 3), 1e-11));
    EXPECT_TRUE(Near(rod.rest_sigmas(), Vector3DStack::Zero(kElements, 3)));
    EXPECT_TRUE(Near(rod.rest_kappas(), Vector3DStack::Zero(kElements - 1, 3)));
}

TEST(StraightRod, RestConfigurationHasNoInternalLoads)
{
    CosseratRod rod = make_rod();

    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_LT(rod.internal_forces().cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(rod.internal_torques().cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(rod.internal_stresses().cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(rod.internal_couples().cwiseAbs().maxCoeff(), 1e-9);
}

TEST(StraightRod, RestConfigurationStartsAtRest)
{
    const CosseratRod rod = make_rod();

    EXPECT_TRUE(Near(rod.velocities(), Vector3DStack::Zero(kElements + 1, 3)));
    EXPECT_TRUE(Near(rod.angular_velocities(), Vector3DStack::Zero(kElements, 3)));
    EXPECT_TRUE(Near(rod.external_forces(), Vector3DStack::Zero(kElements + 1, 3)));
    EXPECT_TRUE(Near(rod.external_torques(), Vector3DStack::Zero(kElements, 3)));
    EXPECT_TRUE(Near(rod.dilatation_rates(), Eigen::VectorXd::Zero(kElements)));
}

// ---------------------------------------------------------------------------
// Volume versus radius preservation
// ---------------------------------------------------------------------------

// With respect_radii false the rod is incompressible: stretching it shrinks
// the radius and leaves the volume alone. This is the reference behaviour.
TEST(RodGeometry, VolumePreservingRodShrinksItsRadiusUnderStretch)
{
    CosseratRod rod = make_rod(/*respect_radii=*/false);
    const Eigen::VectorXd volumes_before = rod.volumes();

    rod.mutable_positions().col(2) *= 2.0;
    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_TRUE(Near(rod.volumes(), volumes_before, 1e-15));
    EXPECT_TRUE(Near(rod.dilatations(),
                     Eigen::VectorXd::Constant(kElements, 2.0), 1e-12));
    // radius scales as one over the square root of the stretch
    EXPECT_TRUE(Near(rod.radii(),
                     Eigen::VectorXd::Constant(kElements, kRadius / std::sqrt(2.0)),
                     1e-14));
}

// With respect_radii true the radius is held and the volume grows instead.
TEST(RodGeometry, RadiusPreservingRodGrowsItsVolumeUnderStretch)
{
    CosseratRod rod = make_rod(/*respect_radii=*/true);
    const Eigen::VectorXd volumes_before = rod.volumes();

    rod.mutable_positions().col(2) *= 2.0;
    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_TRUE(Near(rod.radii(), Eigen::VectorXd::Constant(kElements, kRadius)));
    EXPECT_TRUE(Near(rod.volumes(), Eigen::VectorXd(2.0 * volumes_before), 1e-15));
}

// Both conventions agree at the rest configuration, so the flag only matters
// once the rod deforms.
TEST(RodGeometry, BothConventionsAgreeAtRest)
{
    const CosseratRod volume_rod = make_rod(false);
    const CosseratRod radius_rod = make_rod(true);

    EXPECT_TRUE(Near(volume_rod.volumes(), radius_rod.volumes(), 1e-15));
    EXPECT_TRUE(Near(volume_rod.radii(), radius_rod.radii(), 1e-15));
    EXPECT_TRUE(Near(volume_rod.masses(), radius_rod.masses(), 1e-15));
    EXPECT_EQ(volume_rod.respect_radii(), false);
    EXPECT_EQ(radius_rod.respect_radii(), true);
}

// ---------------------------------------------------------------------------
// Deformation responses
// ---------------------------------------------------------------------------

TEST(RodMechanics, StretchingProducesRestoringForce)
{
    CosseratRod rod = make_rod();
    rod.mutable_positions().col(2) *= 1.1;

    rod.compute_internal_forces_and_torques(0.0);

    // Stretch is along the tangent, so it appears in the third stress column.
    EXPECT_GT(rod.internal_stresses().col(2).maxCoeff(), 0.0);
    EXPECT_TRUE(Near(rod.dilatations(),
                     Eigen::VectorXd::Constant(kElements, 1.1), 1e-12));
    EXPECT_GT(rod.internal_forces().cwiseAbs().maxCoeff(), 0.0);
}

// Internal forces are formed by a difference kernel, so they always sum to
// zero: the rod cannot accelerate its own centre of mass.
TEST(RodMechanics, InternalForcesSumToZero)
{
    CosseratRod rod = make_rod();
    deform(rod);

    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_LT(rod.internal_forces().colwise().sum().cwiseAbs().maxCoeff(), 1e-9);
}

TEST(RodMechanics, TwistingProducesInternalCouple)
{
    CosseratRod rod = make_rod();
    Matrix3DStack& frames = rod.mutable_frames();
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const double angle = 0.05 * static_cast<double>(i);
        frames[i] =
            (Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix()
             * frames[i]).eval();
    }

    rod.compute_internal_forces_and_torques(0.0);

    // Twist about the tangent shows up in the third curvature column.
    EXPECT_GT(rod.kappas().col(2).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_GT(rod.internal_couples().cwiseAbs().maxCoeff(), 0.0);
}

TEST(RodMechanics, DilatationRateFollowsAStretchingVelocity)
{
    CosseratRod rod = make_rod();
    // Move every node outward in proportion to its position, a uniform stretch.
    rod.mutable_velocities() = rod.positions() * 0.5;

    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_TRUE((rod.dilatation_rates().array() > 0.0).all());
}

TEST(RodMechanics, StationaryRodHasZeroDilatationRate)
{
    CosseratRod rod = make_rod();
    rod.mutable_velocities().setConstant(1.0);  // rigid translation

    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_LT(rod.dilatation_rates().cwiseAbs().maxCoeff(), 1e-12);
}

TEST(RodMechanics, ComputeIsIdempotentForFixedState)
{
    CosseratRod rod = make_rod();
    deform(rod);

    rod.compute_internal_forces_and_torques(0.0);
    const Vector3DStack forces = rod.internal_forces();
    const Vector3DStack torques = rod.internal_torques();

    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_TRUE(Near(rod.internal_forces(), forces));
    EXPECT_TRUE(Near(rod.internal_torques(), torques));
}

// ---------------------------------------------------------------------------
// update_accelerations
// ---------------------------------------------------------------------------

TEST(RodAccelerations, TranslationalAccelerationIsForceOverMass)
{
    CosseratRod rod = make_rod();
    rod.compute_internal_forces_and_torques(0.0);
    rod.mutable_external_forces().col(2).setConstant(3.0);

    rod.update_accelerations(0.0, 1e-4);

    ASSERT_EQ(rod.accelerations().rows(), rod.num_nodes());
    for (Eigen::Index i = 0; i < rod.num_nodes(); ++i)
    {
        const Eigen::RowVector3d expected =
            (rod.internal_forces().row(i) + rod.external_forces().row(i))
            / rod.masses()(i);
        EXPECT_TRUE(Near(rod.accelerations().row(i), expected, 1e-9)) << "node " << i;
    }
}

// The angular result must land in its own stack; writing it over the
// translational accelerations would leave this one sized to the elements.
TEST(RodAccelerations, AngularAccelerationIsWrittenToItsOwnStack)
{
    CosseratRod rod = make_rod();
    rod.compute_internal_forces_and_torques(0.0);
    rod.mutable_external_torques().col(0).setConstant(2.0);

    rod.update_accelerations(0.0, 1e-4);

    ASSERT_EQ(rod.accelerations().rows(), rod.num_nodes());
    ASSERT_EQ(rod.angular_accelerations().rows(), rod.num_elements());
    EXPECT_GT(rod.angular_accelerations().cwiseAbs().maxCoeff(), 0.0);

    for (Eigen::Index i = 0; i < rod.num_elements(); ++i)
    {
        const Eigen::Vector3d torque =
            (rod.internal_torques().row(i) + rod.external_torques().row(i)).transpose();
        const Eigen::Vector3d expected =
            rod.inv_mass_2nd_moments()[static_cast<std::size_t>(i)] * torque
            * rod.dilatations()(i);
        EXPECT_TRUE(Near(rod.angular_accelerations().row(i), expected.transpose(), 1e-9))
            << "element " << i;
    }
}

TEST(RodAccelerations, NoLoadsGiveNoAcceleration)
{
    CosseratRod rod = make_rod();
    rod.compute_internal_forces_and_torques(0.0);

    rod.update_accelerations(0.0, 1e-4);

    EXPECT_LT(rod.accelerations().cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(rod.angular_accelerations().cwiseAbs().maxCoeff(), 1e-9);
}

TEST(RodAccelerations, ZeroOutClearsOnlyTheExternalAccumulators)
{
    CosseratRod rod = make_rod();
    deform(rod);
    rod.compute_internal_forces_and_torques(0.0);
    rod.mutable_external_forces().setConstant(5.0);
    rod.mutable_external_torques().setConstant(7.0);

    const Vector3DStack internal_forces = rod.internal_forces();

    rod.zero_out_external_forces_and_torques(0.0 /* time */);

    EXPECT_TRUE(Near(rod.external_forces(), Vector3DStack::Zero(kElements + 1, 3)));
    EXPECT_TRUE(Near(rod.external_torques(), Vector3DStack::Zero(kElements, 3)));
    EXPECT_TRUE(Near(rod.internal_forces(), internal_forces));
}

// ---------------------------------------------------------------------------
// Reference parity
//
// Expected values come from a NumPy transcription of PyElastica's
// cosserat_rod formulas applied to the same four-element rod under the same
// deformation. They pin the whole pipeline, not just its parts.
// ---------------------------------------------------------------------------

CosseratRod make_reference_rod()
{
    CosseratRod rod = straight_cosserat_rod(
        4, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 0.0), 1.0, 0.05, 1000.0, 1.0e6,
        /*respect_radii=*/false, kTol);
    deform(rod);
    rod.compute_internal_forces_and_torques(0.0);
    return rod;
}

TEST(ReferenceParity, StrainsMatchPyElastica)
{
    const CosseratRod rod = make_reference_rod();

    Vector3DStack expected_sigmas(4, 3);
    expected_sigmas << 0.195690194084, 0.0570459336432, 0.04007264621,
                       0.502534197679, 0.102212792755, -0.0123199277294,
                       0.784295901175, 0.214830244072, -0.10131879644,
                       1.02574454121, 0.39141123905, -0.219522339771;

    Vector3DStack expected_kappas(3, 3);
    expected_kappas << -0.160356745147, -0.320713490295, -0.481070235442,
                       -0.160356745147, -0.320713490295, -0.481070235442,
                       -0.160356745147, -0.320713490295, -0.481070235442;

    EXPECT_TRUE(Near(rod.sigmas(), expected_sigmas, 1e-10));
    EXPECT_TRUE(Near(rod.kappas(), expected_kappas, 1e-10));
}

TEST(ReferenceParity, InternalForcesMatchPyElastica)
{
    const CosseratRod rod = make_reference_rod();

    Vector3DStack expected(5, 3);
    expected << 454.750431082, 92.9768027568, 329.344869725,
                707.25751551, -139.875897168, -237.796751758,
                621.689481134, -188.278359128, -343.034133099,
                515.674878282, -234.081857024, -352.25345821,
                -2299.37230601, 469.259310562, 603.739473341;

    EXPECT_TRUE(Near(rod.internal_forces(), expected, 1e-7));
}

// Covers all five couple terms at once: the two bend and twist contributions,
// the shear and stretch couple, Lagrangian transport, and unsteady dilatation.
TEST(ReferenceParity, InternalTorquesMatchPyElastica)
{
    const CosseratRod rod = make_reference_rod();

    Vector3DStack expected(4, 3);
    expected << -31.7344283322, 105.455721825, -1.22787810643,
                -59.4052683327, 292.651817701, 0.22561999344,
                -135.70826691, 496.037390241, 0.250759192195,
                -268.014851443, 704.072444317, 0.751605254884;

    EXPECT_TRUE(Near(rod.internal_torques(), expected, 1e-7));
}

TEST(ReferenceParity, AccelerationsMatchPyElastica)
{
    CosseratRod rod = make_reference_rod();
    rod.mutable_external_forces().col(2).setConstant(1.5);
    rod.mutable_external_torques().col(0).setConstant(0.7);

    rod.update_accelerations(0.0, 1e-4);

    Vector3DStack expected_acc(5, 3);
    expected_acc << 463.204985472, 94.7053936104, 336.995817045,
                    360.203294823, -71.2382094517, -120.344947452,
                    316.62385277, -95.8893809036, -173.941905655,
                    262.631058902, -119.216910827, -178.637269378,
                    -2342.12139846, 477.983608754, 616.491865194;

    Vector3DStack expected_alpha(4, 3);
    expected_alpha << -26802.8960453, 91076.8749852, -530.22870102,
                      -53237.1962971, 265392.914608, 102.302367577,
                      -133335.738396, 489892.309663, 123.826350846,
                      -293421.320194, 772833.47711, 412.504215464;

    EXPECT_TRUE(Near(rod.accelerations(), expected_acc, 1e-6));
    EXPECT_TRUE(Near(rod.angular_accelerations(), expected_alpha, 1e-3));
}

TEST(ReferenceParity, DerivedGeometryMatchesPyElastica)
{
    const CosseratRod rod = make_reference_rod();

    Eigen::VectorXd expected_lengths(4);
    expected_lengths << 0.26496462028, 0.278219787219, 0.302995462012,
                        0.336758444586;
    Eigen::VectorXd expected_dilatations(4);
    expected_dilatations << 1.05985848112, 1.11287914887, 1.21198184805,
                            1.34703377834;

    EXPECT_TRUE(Near(rod.lengths(), expected_lengths, 1e-10));
    EXPECT_TRUE(Near(rod.dilatations(), expected_dilatations, 1e-10));
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

class RodWriteTest : public ::testing::Test
{
protected:
    std::filesystem::path m_directory;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_directory =
            std::filesystem::temp_directory_path() / (std::string("rods_") + info->name());
        std::filesystem::remove_all(m_directory);
        std::filesystem::create_directories(m_directory);
    }

    void TearDown() override { std::filesystem::remove_all(m_directory); }

    bool pair_exists(const std::string& name) const
    {
        return std::filesystem::is_regular_file(m_directory / (name + ".bin"))
            and std::filesystem::is_regular_file(m_directory / (name + ".md.json"));
    }
};

TEST_F(RodWriteTest, WriteEmitsConfigurationOnly)
{
    const CosseratRod rod = make_rod();

    rod.write(m_directory);

    EXPECT_TRUE(pair_exists("positions"));
    EXPECT_TRUE(pair_exists("frames"));
    EXPECT_TRUE(pair_exists("radii"));
    EXPECT_FALSE(pair_exists("velocities"));
}

TEST_F(RodWriteTest, WriteDebugEmitsEveryStack)
{
    const CosseratRod rod = make_rod();

    rod.write_debug(m_directory);

    for (const char* name :
         {"positions", "velocities", "accelerations", "internal_forces",
          "external_forces", "masses", "frames", "mass_2nd_moments",
          "inverse_mass_2nd_moments", "bending_matrices", "shearing_matrices",
          "angular_velocities", "angular_accelerations", "internal_torques",
          "external_torques", "tangents", "sigmas", "rest_sigmas", "radii",
          "densities", "volumes", "lengths", "rest_lengths", "dilatations",
          "dilatation_rates", "internal_stresses", "kappas", "rest_kappas",
          "internal_couples", "voronoi_dilatations", "voronoi_rest_lengths"})
    {
        EXPECT_TRUE(pair_exists(name)) << "missing " << name;
    }
}

TEST_F(RodWriteTest, WrittenSizesMatchTheDomains)
{
    const CosseratRod rod = make_rod();

    rod.write_debug(m_directory);

    const auto bytes = [&](const std::string& name) {
        return std::filesystem::file_size(m_directory / (name + ".bin"));
    };
    const std::size_t nodes = static_cast<std::size_t>(rod.num_nodes());
    const std::size_t elements = static_cast<std::size_t>(rod.num_elements());
    const std::size_t voronoi = static_cast<std::size_t>(rod.num_voronoi());

    EXPECT_EQ(bytes("positions"), nodes * 3 * sizeof(double));
    EXPECT_EQ(bytes("masses"), nodes * sizeof(double));
    EXPECT_EQ(bytes("frames"), elements * 9 * sizeof(double));
    EXPECT_EQ(bytes("tangents"), elements * 3 * sizeof(double));
    EXPECT_EQ(bytes("bending_matrices"), voronoi * 9 * sizeof(double));
    EXPECT_EQ(bytes("kappas"), voronoi * 3 * sizeof(double));
}

TEST_F(RodWriteTest, WriteCreatesMissingDirectories)
{
    const CosseratRod rod = make_rod();
    const std::filesystem::path nested = m_directory / "run" / "rod" / "step_0";

    rod.write(nested);

    EXPECT_TRUE(std::filesystem::is_regular_file(nested / "positions.bin"));
}

// ---------------------------------------------------------------------------
// Construction failures
// ---------------------------------------------------------------------------

TEST(StraightRodDeathTest, RejectsTooFewElements)
{
    EXPECT_ASSERT_FAILURE(straight_cosserat_rod(
        2, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 1.0, 0.05, 1000.0, 1e6, false, kTol));
}

TEST(StraightRodDeathTest, RejectsNonPositiveGeometry)
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const Eigen::Vector3d dir = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d nrm = Eigen::Vector3d::UnitX();

    EXPECT_ASSERT_FAILURE(
        straight_cosserat_rod(10, zero, dir, nrm, 0.0, 0.05, 1000.0, 1e6, false, kTol));
    EXPECT_ASSERT_FAILURE(
        straight_cosserat_rod(10, zero, dir, nrm, 1.0, 0.0, 1000.0, 1e6, false, kTol));
    EXPECT_ASSERT_FAILURE(
        straight_cosserat_rod(10, zero, dir, nrm, 1.0, 0.05, 0.0, 1e6, false, kTol));
}

TEST(StraightRodDeathTest, RejectsNonUnitOrNonOrthogonalDirections)
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    EXPECT_ASSERT_FAILURE(straight_cosserat_rod(
        10, zero, Eigen::Vector3d(0.0, 0.0, 2.0), Eigen::Vector3d::UnitX(),
        1.0, 0.05, 1000.0, 1e6, false, kTol));
    EXPECT_ASSERT_FAILURE(straight_cosserat_rod(
        10, zero, Eigen::Vector3d::UnitZ(), Eigen::Vector3d(3.0, 0.0, 0.0),
        1.0, 0.05, 1000.0, 1e6, false, kTol));
    // normal parallel to direction
    EXPECT_ASSERT_FAILURE(straight_cosserat_rod(
        10, zero, Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ(),
        1.0, 0.05, 1000.0, 1e6, false, kTol));
}

TEST(RodConstructionDeathTest, RejectsFewerThanThreeNodes)
{
    Vector3DStack positions(2, 3);
    positions << 0, 0, 0,
                 0, 0, 1;
    Matrix3DStack frames(1, Eigen::Matrix3d::Identity());

    EXPECT_ASSERT_FAILURE(CosseratRod(
        positions, frames, Vector3DStack{}, Eigen::VectorXd::Constant(1, 1000.0),
        Eigen::VectorXd::Constant(1, 0.05), Eigen::VectorXd::Constant(1, 1.0),
        Vector3DStack{}, true, 1e6));
}

TEST(RodConstructionDeathTest, RejectsNonOrthogonalFrames)
{
    Vector3DStack positions(4, 3);
    positions << 0, 0, 0,
                 0, 0, 1,
                 0, 0, 2,
                 0, 0, 3;
    Matrix3DStack frames(3, Eigen::Matrix3d::Identity());
    frames[1](0, 0) = 2.0;

    EXPECT_ASSERT_FAILURE(CosseratRod(
        positions, frames, Vector3DStack{}, Eigen::VectorXd::Constant(3, 1000.0),
        Eigen::VectorXd::Constant(3, 0.05), Eigen::VectorXd::Constant(3, 1.0),
        Vector3DStack{}, true, 1e6));
}

TEST(RodConstructionDeathTest, RejectsMismatchedElementCounts)
{
    Vector3DStack positions(4, 3);
    positions << 0, 0, 0,
                 0, 0, 1,
                 0, 0, 2,
                 0, 0, 3;
    Matrix3DStack frames(3, Eigen::Matrix3d::Identity());

    // Two densities for three elements.
    EXPECT_ASSERT_FAILURE(CosseratRod(
        positions, frames, Vector3DStack{}, Eigen::VectorXd::Constant(2, 1000.0),
        Eigen::VectorXd::Constant(3, 0.05), Eigen::VectorXd::Constant(3, 1.0),
        Vector3DStack{}, true, 1e6));
}

TEST(RodConstructionDeathTest, RejectsNonPositiveRestLengths)
{
    Vector3DStack positions(4, 3);
    positions << 0, 0, 0,
                 0, 0, 1,
                 0, 0, 2,
                 0, 0, 3;
    Matrix3DStack frames(3, Eigen::Matrix3d::Identity());
    Eigen::VectorXd rest_lengths(3);
    rest_lengths << 1.0, 0.0, 1.0;

    EXPECT_ASSERT_FAILURE(CosseratRod(
        positions, frames, Vector3DStack{}, Eigen::VectorXd::Constant(3, 1000.0),
        Eigen::VectorXd::Constant(3, 0.05), rest_lengths,
        Vector3DStack{}, true, 1e6));
}

// A rod with two coincident nodes has a zero-length element and no tangent.
TEST(RodComputeDeathTest, RejectsDegenerateElementLength)
{
    CosseratRod rod = make_rod();
    rod.mutable_positions().row(3) = rod.positions().row(2);

    EXPECT_ASSERT_FAILURE(rod.compute_internal_forces_and_torques(0.0));
}

// ---------------------------------------------------------------------------
// Custom stiffness constructor
// ---------------------------------------------------------------------------

TEST(RodConstruction, AcceptsUserSuppliedStiffnessMatrices)
{
    Vector3DStack positions(5, 3);
    for (Eigen::Index i = 0; i < 5; ++i) positions.row(i) << 0.0, 0.0, 0.25 * i;
    Matrix3DStack frames(4, Eigen::Matrix3d::Identity());

    Matrix3DStack shearing(4, 2.0 * Eigen::Matrix3d::Identity());
    Matrix3DStack bending(4, 3.0 * Eigen::Matrix3d::Identity());

    const CosseratRod rod(
        positions, frames, bending, shearing, Vector3DStack{},
        Eigen::VectorXd::Constant(4, 1000.0), Eigen::VectorXd::Constant(4, 0.05),
        Eigen::VectorXd::Constant(4, 0.25), Vector3DStack{}, true);

    EXPECT_EQ(static_cast<Eigen::Index>(rod.shearing_matrices().size()), 4);
    EXPECT_TRUE(Near(rod.shearing_matrices()[0], 2.0 * Eigen::Matrix3d::Identity()));
    // Bending matrices are averaged onto the Voronoi domain, uniformly here.
    EXPECT_EQ(static_cast<Eigen::Index>(rod.bending_matrices().size()), 3);
    EXPECT_TRUE(Near(rod.bending_matrices()[0], 3.0 * Eigen::Matrix3d::Identity()));
}

// The Voronoi average is weighted by the adjacent rest lengths.
TEST(RodConstruction, BendingMatricesAreRestLengthWeightedOnTheVoronoiDomain)
{
    Vector3DStack positions(4, 3);
    positions << 0, 0, 0,
                 0, 0, 1,
                 0, 0, 3,
                 0, 0, 6;
    Matrix3DStack frames(3, Eigen::Matrix3d::Identity());
    Matrix3DStack shearing(3, Eigen::Matrix3d::Identity());

    Matrix3DStack bending;
    bending.push_back(1.0 * Eigen::Matrix3d::Identity());
    bending.push_back(2.0 * Eigen::Matrix3d::Identity());
    bending.push_back(4.0 * Eigen::Matrix3d::Identity());

    Eigen::VectorXd rest_lengths(3);
    rest_lengths << 1.0, 2.0, 3.0;

    const CosseratRod rod(
        positions, frames, bending, shearing, Vector3DStack{},
        Eigen::VectorXd::Constant(3, 1000.0), Eigen::VectorXd::Constant(3, 0.05),
        rest_lengths, Vector3DStack{}, true);

    // (l0*B0 + l1*B1) / (l0 + l1) = (1*1 + 2*2)/3
    EXPECT_TRUE(Near(rod.bending_matrices()[0],
                     (5.0 / 3.0) * Eigen::Matrix3d::Identity(), 1e-12));
    // (2*2 + 3*4)/5
    EXPECT_TRUE(Near(rod.bending_matrices()[1],
                     (16.0 / 5.0) * Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(RodConstruction, RestStrainsAreHonouredWhenSupplied)
{
    Vector3DStack positions(5, 3);
    for (Eigen::Index i = 0; i < 5; ++i) positions.row(i) << 0.0, 0.0, 0.25 * i;
    Matrix3DStack frames(4, Eigen::Matrix3d::Identity());

    Vector3DStack rest_sigmas = Vector3DStack::Constant(4, 3, 0.1);
    Vector3DStack rest_kappas = Vector3DStack::Constant(3, 3, 0.2);

    const CosseratRod rod(
        positions, frames, rest_sigmas, Eigen::VectorXd::Constant(4, 1000.0),
        Eigen::VectorXd::Constant(4, 0.05), Eigen::VectorXd::Constant(4, 0.25),
        rest_kappas, true, 1e6);

    EXPECT_TRUE(Near(rod.rest_sigmas(), rest_sigmas));
    EXPECT_TRUE(Near(rod.rest_kappas(), rest_kappas));
}

// A rod at rest with a non-zero rest strain still carries no internal load,
// because the stresses are driven by the departure from rest.
TEST(RodConstruction, RestStrainMatchingTheStateGivesNoInternalStress)
{
    CosseratRod plain = make_rod();
    plain.compute_internal_forces_and_torques(0.0);
    const Vector3DStack sigmas = plain.sigmas();
    const Vector3DStack kappas = plain.kappas();

    const double rest_length = kLength / static_cast<double>(kElements);
    Vector3DStack positions(kElements + 1, 3);
    for (Eigen::Index i = 0; i < kElements + 1; ++i)
    {
        positions.row(i) << 0.0, 0.0, rest_length * i;
    }
    Matrix3DStack frames(kElements, Eigen::Matrix3d::Identity());

    CosseratRod rod(
        positions, frames, sigmas, Eigen::VectorXd::Constant(kElements, kDensity),
        Eigen::VectorXd::Constant(kElements, kRadius),
        Eigen::VectorXd::Constant(kElements, rest_length), kappas, true, kYoungs);
    rod.compute_internal_forces_and_torques(0.0);

    EXPECT_LT(rod.internal_stresses().cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(rod.internal_couples().cwiseAbs().maxCoeff(), 1e-9);
}

}  // namespace
}  // namespace cosserat::physics
