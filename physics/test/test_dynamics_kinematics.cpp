#include "physics/dynamics_kinematics.hpp"

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace cosserat::physics::dynamics {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

constexpr double kTol = 1e-12;

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

CosseratRod make_rod(std::int64_t elements = 4)
{
    return straight_cosserat_rod(
        elements, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 1.0, 0.05, 1000.0, 1.0e6, false, kTol);
}

Sphere make_sphere() { return Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0); }

/** Fills a stack with a reproducible, non-degenerate pattern. */
void fill(Vector3DStack& stack, double a, double b, double c)
{
    for (Eigen::Index i = 0; i < stack.rows(); ++i)
    {
        const double t = static_cast<double>(i);
        stack.row(i) << a + 0.1 * t, b - 0.05 * t, c + 0.01 * t;
    }
}

// ---------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------

TEST(DynamicsKinematicsConcepts, EveryBodyTypeSupportsBothHalfSteps)
{
    static_assert(DynamicSystem<CosseratRod>);
    static_assert(KinematicSystem<CosseratRod>);
    static_assert(DynamicSystem<RigidBody>);
    static_assert(KinematicSystem<RigidBody>);
    static_assert(DynamicSystem<Sphere>);
    static_assert(KinematicSystem<Sphere>);
    static_assert(DynamicSystem<Cylinder>);
    static_assert(KinematicSystem<Cylinder>);
    SUCCEED();
}

/** Only the rates half; deliberately missing the kinematic accessors. */
struct RatesOnlySystem
{
    Vector3DStack v, w, a, alpha;
    Vector3DStack& mutable_velocities() { return v; }
    Vector3DStack& mutable_angular_velocities() { return w; }
    const Vector3DStack& accelerations() const { return a; }
    const Vector3DStack& angular_accelerations() const { return alpha; }
};

TEST(DynamicsKinematicsConcepts, ASystemMaySupportOnlyOneHalf)
{
    static_assert(DynamicSystem<RatesOnlySystem>);
    static_assert(!KinematicSystem<RatesOnlySystem>);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// update_dynamics
// ---------------------------------------------------------------------------

TEST(UpdateDynamics, AdvancesRatesByScaledAccelerations)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_velocities(), 1.0, 2.0, 3.0);
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);

    const Vector3DStack velocities_before = rod.velocities();
    const Vector3DStack omegas_before = rod.angular_velocities();
    const Vector3DStack accelerations = rod.accelerations();
    const Vector3DStack angular_accelerations = rod.angular_accelerations();

    const double scale = 0.037;
    update_dynamics(rod, 0.0, scale);

    EXPECT_TRUE(Near(rod.velocities(),
                     Vector3DStack(velocities_before + scale * accelerations)));
    EXPECT_TRUE(Near(rod.angular_velocities(),
                     Vector3DStack(omegas_before + scale * angular_accelerations)));
}

// The accelerations themselves must not be disturbed.
TEST(UpdateDynamics, LeavesAccelerationsAlone)
{
    CosseratRod rod = make_rod();
    rod.mutable_external_forces().setConstant(3.0);
    rod.mutable_external_torques().setConstant(2.0);
    rod.compute_internal_forces_and_torques(0.0);
    rod.update_accelerations(0.0, 1e-4);

    const Vector3DStack accelerations = rod.accelerations();
    const Vector3DStack angular_accelerations = rod.angular_accelerations();

    update_dynamics(rod, 0.0, 1e-4);

    EXPECT_TRUE(Near(rod.accelerations(), accelerations));
    EXPECT_TRUE(Near(rod.angular_accelerations(), angular_accelerations));
}

TEST(UpdateDynamics, ZeroScaleIsANoOp)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_velocities(), 1.0, 2.0, 3.0);
    rod.mutable_external_forces().setConstant(5.0);
    rod.compute_internal_forces_and_torques(0.0);
    rod.update_accelerations(0.0, 1e-4);

    const Vector3DStack velocities_before = rod.velocities();
    const Vector3DStack omegas_before = rod.angular_velocities();

    update_dynamics(rod, 0.0, 0.0);

    EXPECT_TRUE(Near(rod.velocities(), velocities_before));
    EXPECT_TRUE(Near(rod.angular_velocities(), omegas_before));
}

// Two half-steps of h are one step of 2h, since the accelerations are fixed.
TEST(UpdateDynamics, StepsCompose)
{
    CosseratRod split = make_rod();
    CosseratRod whole = make_rod();
    for (CosseratRod* rod : {&split, &whole})
    {
        rod->mutable_external_forces().setConstant(4.0);
        rod->mutable_external_torques().setConstant(1.0);
        rod->compute_internal_forces_and_torques(0.0);
        rod->update_accelerations(0.0, 1e-4);
    }

    update_dynamics(split, 0.0, 0.5e-4);
    update_dynamics(split, 0.0, 0.5e-4);
    update_dynamics(whole, 0.0, 1.0e-4);

    EXPECT_TRUE(Near(split.velocities(), whole.velocities(), 1e-15));
    EXPECT_TRUE(Near(split.angular_velocities(), whole.angular_velocities(), 1e-15));
}

TEST(UpdateDynamics, WorksOnARigidBody)
{
    Sphere sphere = make_sphere();
    sphere.mutable_external_forces().row(0) << 10.0, -5.0, 2.0;
    sphere.update_accelerations(0.0, 1e-4);

    const Eigen::RowVector3d accelerations = sphere.accelerations().row(0);

    update_dynamics(sphere, 0.0, 0.5);

    EXPECT_TRUE(Near(sphere.velocities().row(0),
                     Eigen::RowVector3d(0.5 * accelerations)));
}

TEST(UpdateDynamicsDeathTest, RejectsMismatchedSizes)
{
    RatesOnlySystem system;
    system.v = Vector3DStack::Zero(4, 3);
    system.a = Vector3DStack::Zero(3, 3);  // one too few
    system.w = Vector3DStack::Zero(4, 3);
    system.alpha = Vector3DStack::Zero(4, 3);

    EXPECT_ASSERT_FAILURE(update_dynamics(system, 0.0, 1e-4));
}

TEST(UpdateDynamicsDeathTest, RejectsMismatchedAngularSizes)
{
    RatesOnlySystem system;
    system.v = Vector3DStack::Zero(4, 3);
    system.a = Vector3DStack::Zero(4, 3);
    system.w = Vector3DStack::Zero(4, 3);
    system.alpha = Vector3DStack::Zero(2, 3);  // mismatched

    EXPECT_ASSERT_FAILURE(update_dynamics(system, 0.0, 1e-4));
}

// ---------------------------------------------------------------------------
// update_kinematics: positions
// ---------------------------------------------------------------------------

TEST(UpdateKinematics, AdvancesPositionsByScaledVelocities)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_velocities(), 1.0, 2.0, 3.0);

    const Vector3DStack positions_before = rod.positions();
    const Vector3DStack velocities = rod.velocities();

    const double scale = 0.037;
    update_kinematics(rod, 0.0, scale);

    EXPECT_TRUE(Near(rod.positions(),
                     Vector3DStack(positions_before + scale * velocities)));
    // The velocities themselves are untouched.
    EXPECT_TRUE(Near(rod.velocities(), velocities));
}

TEST(UpdateKinematics, ZeroScaleLeavesEverythingAlone)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_velocities(), 1.0, 2.0, 3.0);
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);

    const Vector3DStack positions_before = rod.positions();
    const Matrix3DStack frames_before = rod.frames();

    update_kinematics(rod, 0.0, 0.0);

    EXPECT_TRUE(Near(rod.positions(), positions_before));
    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.frames()[i], frames_before[i])) << "frame " << i;
    }
}

// ---------------------------------------------------------------------------
// update_kinematics: frames
// ---------------------------------------------------------------------------

TEST(UpdateKinematics, RotatesFramesOnTheLeft)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);

    const Matrix3DStack frames_before = rod.frames();
    const Vector3DStack omegas = rod.angular_velocities();
    const double scale = 0.037;

    update_kinematics(rod, 0.0, scale);

    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        const Eigen::Vector3d omega =
            omegas.row(static_cast<Eigen::Index>(i)).transpose();
        const Eigen::Matrix3d expected =
            math::rotation_matrix(scale, omega) * frames_before[i];
        EXPECT_TRUE(Near(rod.frames()[i], expected)) << "frame " << i;
    }
}

// Left multiplication, not right: the two differ for a general frame.
TEST(UpdateKinematics, TheMultiplicationOrderIsNotReversible)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);
    Matrix3DStack& frames = rod.mutable_frames();
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        frames[i] = (Eigen::AngleAxisd(0.4 * static_cast<double>(i + 1),
                                       Eigen::Vector3d(1, 2, 3).normalized())
                         .toRotationMatrix() * frames[i]).eval();
    }

    const Matrix3DStack frames_before = rod.frames();
    const Eigen::Vector3d omega = rod.angular_velocities().row(0).transpose();
    const double scale = 0.5;

    update_kinematics(rod, 0.0, scale);

    const Eigen::Matrix3d rotation = math::rotation_matrix(scale, omega);
    EXPECT_TRUE(Near(rod.frames()[0], rotation * frames_before[0]));
    EXPECT_GT((rod.frames()[0] - frames_before[0] * rotation).cwiseAbs().maxCoeff(),
              1e-6);
}

TEST(UpdateKinematics, FramesStayOrthogonal)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);

    for (int step = 0; step < 1000; ++step)
    {
        update_kinematics(rod, 0.0, 1e-4);
    }

    for (std::size_t i = 0; i < rod.frames().size(); ++i)
    {
        EXPECT_TRUE(math::is_orthogonal(rod.frames()[i], 1e-12)) << "frame " << i;
    }
}

// A constant angular velocity for a total angle t turns a frame by exactly
// that much, whether applied in one step or many.
TEST(UpdateKinematics, RepeatedSmallRotationsComposeIntoOneLarge)
{
    CosseratRod stepped = make_rod();
    CosseratRod direct = make_rod();
    const Eigen::RowVector3d omega(0.0, 0.0, 1.0);
    stepped.mutable_angular_velocities().rowwise() = omega;
    direct.mutable_angular_velocities().rowwise() = omega;

    for (int step = 0; step < 100; ++step) update_kinematics(stepped, 0.0, 0.01);
    update_kinematics(direct, 0.0, 1.0);

    for (std::size_t i = 0; i < direct.frames().size(); ++i)
    {
        EXPECT_TRUE(Near(stepped.frames()[i], direct.frames()[i], 1e-12))
            << "frame " << i;
    }
}

// ---------------------------------------------------------------------------
// Negligible rotations
//
// A frame with nothing to do is left strictly alone rather than multiplied by
// an identity, so it cannot pick up rounding error. These are also the states
// that used to fail outright, since rotation_matrix rejects an axis too short
// to define a direction.
// ---------------------------------------------------------------------------

TEST(UpdateKinematics, AFreshlyBuiltRodIsAtRestAndStillAdvances)
{
    CosseratRod rod = make_rod();
    ASSERT_EQ(rod.angular_velocities().cwiseAbs().maxCoeff(), 0.0);

    EXPECT_NO_THROW({ update_kinematics(rod, 0.0, 1e-4); });
}

TEST(UpdateKinematics, ZeroAngularVelocityLeavesTheFrameBitwiseUnchanged)
{
    CosseratRod rod = make_rod();
    Matrix3DStack& frames = rod.mutable_frames();
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        frames[i] = (Eigen::AngleAxisd(0.37 * static_cast<double>(i + 1),
                                       Eigen::Vector3d(1, 2, 3).normalized())
                         .toRotationMatrix() * frames[i]).eval();
    }
    const Matrix3DStack frames_before = rod.frames();
    ASSERT_EQ(rod.angular_velocities().cwiseAbs().maxCoeff(), 0.0);

    update_kinematics(rod, 0.0, 1e-4);

    // Bitwise, not merely close: nothing was multiplied.
    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        EXPECT_TRUE((rod.frames()[i].array() == frames_before[i].array()).all())
            << "frame " << i << " was modified";
    }
}

// A clamped element has its angular velocity zeroed every step, so this is the
// steady state of any constrained simulation rather than an edge case.
TEST(UpdateKinematics, AnElementWithZeroOmegaAmongSpinningOnesIsSkipped)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);
    rod.mutable_angular_velocities().row(2).setZero();

    const Matrix3DStack frames_before = rod.frames();

    EXPECT_NO_THROW({ update_kinematics(rod, 0.0, 1e-4); });

    // The still element is untouched; its neighbours have turned.
    EXPECT_TRUE((rod.frames()[2].array() == frames_before[2].array()).all());
    EXPECT_GT((rod.frames()[1] - frames_before[1]).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_GT((rod.frames()[3] - frames_before[3]).cwiseAbs().maxCoeff(), 0.0);
}

// Below the tolerance on the axis length itself.
TEST(UpdateKinematics, AnUnresolvablySmallAngularVelocityIsSkipped)
{
    CosseratRod rod = make_rod();
    rod.mutable_angular_velocities().setConstant(1e-13);
    const Matrix3DStack frames_before = rod.frames();

    EXPECT_NO_THROW({ update_kinematics(rod, 0.0, 1e-4); });

    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        EXPECT_TRUE((rod.frames()[i].array() == frames_before[i].array()).all())
            << "frame " << i;
    }
}

// Resolvable axis, but the scale makes the rotation angle negligible.
TEST(UpdateKinematics, ANegligibleRotationAngleIsSkipped)
{
    CosseratRod rod = make_rod();
    rod.mutable_angular_velocities().setConstant(1.0);
    const Matrix3DStack frames_before = rod.frames();

    // |omega| is order one, so an angle of ~1e-15 needs a tiny scale.
    update_kinematics(rod, 0.0, 1e-15);

    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        EXPECT_TRUE((rod.frames()[i].array() == frames_before[i].array()).all())
            << "frame " << i;
    }
}

// Just above the tolerance the rotation is applied, so the guard is a
// threshold rather than a blanket skip of small angles.
TEST(UpdateKinematics, ARotationJustAboveTheToleranceIsApplied)
{
    CosseratRod rod = make_rod();
    rod.mutable_angular_velocities().setConstant(1.0);
    const Matrix3DStack frames_before = rod.frames();

    // |omega| = sqrt(3), so scale 1e-11 gives an angle well above 1e-12.
    update_kinematics(rod, 0.0, 1e-11);

    bool any_changed = false;
    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        if (!(rod.frames()[i].array() == frames_before[i].array()).all())
        {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed);
}

// Position integration is independent of the frames, so a body that is not
// spinning still translates.
TEST(UpdateKinematics, PositionsAdvanceEvenWhenNoFrameTurns)
{
    CosseratRod rod = make_rod();
    fill(rod.mutable_velocities(), 1.0, 2.0, 3.0);
    ASSERT_EQ(rod.angular_velocities().cwiseAbs().maxCoeff(), 0.0);

    const Vector3DStack positions_before = rod.positions();

    update_kinematics(rod, 0.0, 0.1);

    EXPECT_GT((rod.positions() - positions_before).cwiseAbs().maxCoeff(), 0.0);
}

TEST(UpdateKinematics, WorksOnARigidBody)
{
    Sphere sphere = make_sphere();
    sphere.mutable_velocities().row(0) << 1.0, 2.0, 3.0;
    sphere.mutable_angular_velocities().row(0) << 0.0, 0.0, 1.0;

    const Eigen::Matrix3d frame_before = sphere.frames()[0];

    update_kinematics(sphere, 0.0, 0.5);

    EXPECT_TRUE(Near(sphere.positions().row(0), Eigen::RowVector3d(0.5, 1.0, 1.5)));
    EXPECT_TRUE(Near(sphere.frames()[0],
                     math::rotation_matrix(0.5, Eigen::Vector3d(0, 0, 1))
                         * frame_before));
}

TEST(UpdateKinematicsDeathTest, RejectsMismatchedFrameCount)
{
    CosseratRod rod = make_rod();
    rod.mutable_frames().pop_back();
    fill(rod.mutable_angular_velocities(), 0.3, 0.2, 0.4);

    EXPECT_ASSERT_FAILURE(update_kinematics(rod, 0.0, 1e-4));
}

// ---------------------------------------------------------------------------
// Reference parity
//
// Expected values come from a NumPy transcription of PyElastica's
// overload_operator_kinematic_numba applied to the same rod and prefactor:
//   position += prefac * velocity
//   director  = _get_rotation_matrix(prefac, omega) @ director
// ---------------------------------------------------------------------------

TEST(ReferenceParity, KinematicStepMatchesPyElastica)
{
    CosseratRod rod = make_rod(4);

    Vector3DStack& velocities = rod.mutable_velocities();
    for (Eigen::Index i = 0; i < velocities.rows(); ++i)
    {
        const double t = static_cast<double>(i);
        velocities.row(i) << 0.1 * t, -0.05 * t, 0.2 + 0.01 * t;
    }
    Vector3DStack& omegas = rod.mutable_angular_velocities();
    for (Eigen::Index i = 0; i < omegas.rows(); ++i)
    {
        const double t = static_cast<double>(i);
        omegas.row(i) << 0.3 - 0.1 * t, 0.2 + 0.05 * t, 0.05 * t + 0.4;
    }
    Matrix3DStack& frames = rod.mutable_frames();
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        frames[i] = (Eigen::AngleAxisd(0.2 * static_cast<double>(i + 1),
                                       Eigen::Vector3d(1, 2, 3).normalized())
                         .toRotationMatrix() * frames[i]).eval();
    }

    update_kinematics(rod, 0.0, 0.037);

    Vector3DStack expected_positions(5, 3);
    expected_positions << 0, 0, 0.0074,
                          0.0037, -0.00185, 0.25777,
                          0.0074, -0.0037, 0.50814,
                          0.0111, -0.00555, 0.75851,
                          0.0148, -0.0074, 1.00888;
    EXPECT_TRUE(Near(rod.positions(), expected_positions, 1e-12));

    std::vector<Eigen::Matrix3d> expected_frames(4);
    expected_frames[0] <<
        0.984507966299132, -0.142243033717764, 0.102523088386541,
        0.146488096996509, 0.98858945348654, -0.0351017078434314,
        -0.0963602705054782, 0.0495763231185121, 0.994111103677129;
    expected_frames[1] <<
        0.933685522153881, -0.286422368961475, 0.21492689984907,
        0.306626051284375, 0.949489639848504, -0.0667074845434244,
        -0.18496434898106, 0.128185999153012, 0.974349290155823;
    expected_frames[2] <<
        0.849388125618429, -0.413958971195171, 0.327380179952066,
        0.461101824174867, 0.883846713572583, -0.0787406797630937,
        -0.256758485353496, 0.217836996568489, 0.94160614012707;
    expected_frames[3] <<
        0.734870183337349, -0.519907913187333, 0.435501521750412,
        0.603704611096005, 0.79406395686918, -0.0707331247908843,
        -0.309041350280433, 0.314893941203151, 0.897404730102529;

    for (std::size_t i = 0; i < expected_frames.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.frames()[i], expected_frames[i], 1e-12))
            << "frame " << i;
    }
}

// ---------------------------------------------------------------------------
// The two half-steps together
// ---------------------------------------------------------------------------

// One position-Verlet step: half kinematic, dynamic, half kinematic. Under a
// constant acceleration this is exact for the position, which pins the
// interleaving.
TEST(HalfSteps, AVerletStepReproducesConstantAccelerationExactly)
{
    Sphere sphere = make_sphere();
    sphere.mutable_external_forces().row(0) << 0.0, 0.0, -9.80665 * sphere.total_mass();
    sphere.update_accelerations(0.0, 1e-4);

    const Eigen::Vector3d acceleration = sphere.accelerations().row(0).transpose();
    const Eigen::Vector3d position_before = sphere.positions().row(0).transpose();
    const double dt = 0.01;

    update_kinematics(sphere, 0.0, 0.5 * dt);
    update_dynamics(sphere, 0.0, dt);
    update_kinematics(sphere, 0.0, 0.5 * dt);

    // x = x0 + v0 t + a t^2 / 2, with v0 = 0.
    const Eigen::Vector3d expected =
        position_before + 0.5 * acceleration * dt * dt;

    EXPECT_TRUE(Near(sphere.positions().row(0), expected.transpose(), 1e-14));
    EXPECT_TRUE(Near(sphere.velocities().row(0),
                     Eigen::RowVector3d(dt * acceleration), 1e-14));
}

// Running the loop on a rod at rest with no loads must leave it exactly where
// it started, which is the smoke test for a stepper wired up correctly.
TEST(HalfSteps, ARodAtRestWithNoLoadsDoesNotMove)
{
    CosseratRod rod = make_rod();
    const Vector3DStack positions_before = rod.positions();
    const Matrix3DStack frames_before = rod.frames();

    for (int step = 0; step < 50; ++step)
    {
        update_kinematics(rod, 0.0, 0.5e-4);
        rod.compute_internal_forces_and_torques(0.0);
        rod.update_accelerations(0.0, 1e-4);
        update_dynamics(rod, 0.0, 1e-4);
        update_kinematics(rod, 0.0, 0.5e-4);
        rod.zero_out_external_forces_and_torques(0.0 /* time */);
    }

    EXPECT_TRUE(Near(rod.positions(), positions_before, 1e-9));
    for (std::size_t i = 0; i < frames_before.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.frames()[i], frames_before[i], 1e-9)) << "frame " << i;
    }
}

}  // namespace
}  // namespace cosserat::physics::dynamics
