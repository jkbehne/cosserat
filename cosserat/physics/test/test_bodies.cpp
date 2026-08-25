#include <cosserat/physics/bodies.hpp>

#include <cosserat/physics/dynamics_kinematics.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <utility>
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

CosseratRod make_rod()
{
    return straight_cosserat_rod(
        4, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 1.0, 0.05, 1000.0, 1.0e6, false, kTol);
}

Sphere make_sphere() { return Sphere(Eigen::Vector3d::Zero(), 0.1, 1000.0); }

Cylinder make_cylinder()
{
    return Cylinder(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(),
                    Eigen::Vector3d::UnitX(), 0.2, 0.01, 1000.0);
}

BodyVariantPtr wrap(BodyVariant body)
{
    return std::make_shared<BodyVariant>(std::move(body));
}

/** Every stack a body exposes for reading, as one matrix, for comparison. */
Eigen::MatrixXd state_of(const BodyVariant& body)
{
    return std::visit([](const auto& concrete) {
        const Eigen::Index rows =
            concrete.positions().rows() + concrete.velocities().rows()
            + concrete.angular_velocities().rows();
        Eigen::MatrixXd all(rows, 3);
        all << concrete.positions(), concrete.velocities(),
               concrete.angular_velocities();
        return all;
    }, body);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(BodyVariantWrapperTest, WrapsEachBodyKind)
{
    EXPECT_NO_THROW({ BodyVariantWrapper(wrap(make_rod())); });
    EXPECT_NO_THROW({ BodyVariantWrapper(wrap(make_sphere())); });
    EXPECT_NO_THROW({ BodyVariantWrapper(wrap(make_cylinder())); });
    EXPECT_NO_THROW({ BodyVariantWrapper(wrap(RigidBody(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 0.5, 1.0, 100.0,
        0.25, Eigen::Vector3d::Ones()))); });
}

TEST(BodyVariantWrapperDeathTest, RejectsANullBody)
{
    EXPECT_ASSERT_FAILURE(BodyVariantWrapper(BodyVariantPtr{}));
    EXPECT_ASSERT_FAILURE(BodyVariantWrapper(nullptr));
}

TEST(BodyVariantWrapperTest, HoldsTheAlternativeItWasGiven)
{
    const BodyVariantWrapper rod(wrap(make_rod()));
    const BodyVariantWrapper sphere(wrap(make_sphere()));
    const BodyVariantWrapper cylinder(wrap(make_cylinder()));

    EXPECT_TRUE(std::holds_alternative<CosseratRod>(rod.body()));
    EXPECT_TRUE(std::holds_alternative<Sphere>(sphere.body()));
    EXPECT_TRUE(std::holds_alternative<Cylinder>(cylinder.body()));
}

// ---------------------------------------------------------------------------
// Forwarding
//
// Each method must reach the engaged alternative and do exactly what calling
// it directly would.
// ---------------------------------------------------------------------------

TEST(BodyVariantWrapperTest, UpdateKinematicsMatchesADirectCall)
{
    CosseratRod direct = make_rod();
    BodyVariantWrapper wrapped(wrap(make_rod()));

    for (auto* velocities : {&direct.mutable_velocities(),
                             &std::get<CosseratRod>(wrapped.body()).mutable_velocities()})
    {
        velocities->setConstant(1.5);
    }
    for (auto* omegas : {&direct.mutable_angular_velocities(),
                         &std::get<CosseratRod>(wrapped.body()).mutable_angular_velocities()})
    {
        omegas->setConstant(0.4);
    }

    dynamics::update_kinematics(direct, 0.0, 0.037);
    wrapped.update_kinematics(0.0, 0.037);

    const CosseratRod& through = std::get<CosseratRod>(wrapped.body());
    EXPECT_TRUE(Near(through.positions(), direct.positions()));
    for (std::size_t i = 0; i < direct.frames().size(); ++i)
    {
        EXPECT_TRUE(Near(through.frames()[i], direct.frames()[i])) << "frame " << i;
    }
}

TEST(BodyVariantWrapperTest, UpdateDynamicsMatchesADirectCall)
{
    CosseratRod direct = make_rod();
    BodyVariantWrapper wrapped(wrap(make_rod()));

    direct.mutable_external_forces().setConstant(4.0);
    std::get<CosseratRod>(wrapped.body()).mutable_external_forces().setConstant(4.0);
    direct.compute_internal_forces_and_torques(0.0);
    direct.update_accelerations(0.0, 1e-4);
    wrapped.compute_internal_forces_and_torques(0.0);
    wrapped.update_accelerations(0.0, 1e-4);

    dynamics::update_dynamics(direct, 0.0, 1e-4);
    wrapped.update_dynamics(0.0, 1e-4);

    const CosseratRod& through = std::get<CosseratRod>(wrapped.body());
    EXPECT_TRUE(Near(through.velocities(), direct.velocities()));
    EXPECT_TRUE(Near(through.angular_velocities(), direct.angular_velocities()));
}

TEST(BodyVariantWrapperTest, UpdateAccelerationReachesTheBody)
{
    BodyVariantWrapper wrapped(wrap(make_sphere()));
    Sphere& sphere = std::get<Sphere>(wrapped.body());
    sphere.mutable_external_forces().row(0) << 10.0, -5.0, 2.0;

    wrapped.update_accelerations(0.0, 1e-4);

    EXPECT_TRUE(Near(sphere.accelerations().row(0),
                     Eigen::RowVector3d(10.0, -5.0, 2.0) / sphere.total_mass(),
                     1e-12));
}

TEST(BodyVariantWrapperTest, ComputeInternalForcesReachesARod)
{
    BodyVariantWrapper wrapped(wrap(make_rod()));
    CosseratRod& rod = std::get<CosseratRod>(wrapped.body());
    rod.mutable_positions().col(2) *= 1.1;  // stretch it

    wrapped.compute_internal_forces_and_torques(0.0);

    EXPECT_GT(rod.internal_forces().cwiseAbs().maxCoeff(), 0.0);
}

// A rigid body has no internal loads, so this must be a harmless no-op.
TEST(BodyVariantWrapperTest, ComputeInternalForcesIsANoOpForARigidBody)
{
    BodyVariantWrapper wrapped(wrap(make_cylinder()));

    EXPECT_NO_THROW({ wrapped.compute_internal_forces_and_torques(0.0); });

    const Cylinder& cylinder = std::get<Cylinder>(wrapped.body());
    EXPECT_TRUE(Near(cylinder.internal_forces(), Vector3DStack::Zero(1, 3)));
    EXPECT_TRUE(Near(cylinder.internal_torques(), Vector3DStack::Zero(1, 3)));
}

// The wrapper takes a time here even though no body does, so that every
// stepper entry point has the same shape.
TEST(BodyVariantWrapperTest, ZeroOutClearsTheAccumulators)
{
    BodyVariantWrapper wrapped(wrap(make_rod()));
    CosseratRod& rod = std::get<CosseratRod>(wrapped.body());
    rod.mutable_external_forces().setConstant(7.0);
    rod.mutable_external_torques().setConstant(9.0);
    rod.mutable_velocities().setConstant(2.0);

    wrapped.zero_out_external_forces_and_torques(0.5);

    EXPECT_LT(rod.external_forces().cwiseAbs().maxCoeff(), 1e-15);
    EXPECT_LT(rod.external_torques().cwiseAbs().maxCoeff(), 1e-15);
    // Only the accumulators are cleared.
    EXPECT_GT(rod.velocities().cwiseAbs().maxCoeff(), 1.0);
}

TEST(BodyVariantWrapperTest, EveryMethodWorksOnEveryBodyKind)
{
    std::vector<BodyVariantWrapper> bodies{
        BodyVariantWrapper(wrap(make_rod())),
        BodyVariantWrapper(wrap(make_sphere())),
        BodyVariantWrapper(wrap(make_cylinder())),
    };

    for (BodyVariantWrapper& body : bodies)
    {
        EXPECT_NO_THROW({
            body.compute_internal_forces_and_torques(0.0);
            body.update_accelerations(0.0, 1e-4);
            body.update_dynamics(0.0, 1e-4);
            body.update_kinematics(0.0, 0.5e-4);
            body.zero_out_external_forces_and_torques(0.0);
        });
    }
}

// ---------------------------------------------------------------------------
// Shared ownership
// ---------------------------------------------------------------------------

TEST(BodyVariantWrapperTest, CopiesShareTheSameBody)
{
    BodyVariantWrapper first(wrap(make_rod()));
    BodyVariantWrapper second = first;

    std::get<CosseratRod>(first.body()).mutable_velocities().setConstant(3.0);

    EXPECT_TRUE(Near(std::get<CosseratRod>(second.body()).velocities(),
                     std::get<CosseratRod>(first.body()).velocities()));
    EXPECT_TRUE(first.refers_to_same_body_as(second));
}

// Stepping through one handle is visible through the other.
TEST(BodyVariantWrapperTest, SteppingThroughOneHandleIsSeenByTheOther)
{
    const BodyVariantPtr shared = wrap(make_rod());
    BodyVariantWrapper stepper(shared);
    const BodyVariantWrapper observer(shared);

    std::get<CosseratRod>(stepper.body()).mutable_velocities().setConstant(1.0);
    const Vector3DStack before = std::get<CosseratRod>(observer.body()).positions();

    stepper.update_kinematics(0.0, 0.1);

    const Vector3DStack after = std::get<CosseratRod>(observer.body()).positions();
    EXPECT_GT((after - before).cwiseAbs().maxCoeff(), 0.0);
}

TEST(BodyVariantWrapperTest, SeparatelyWrappedBodiesAreIndependent)
{
    BodyVariantWrapper first(wrap(make_rod()));
    BodyVariantWrapper second(wrap(make_rod()));

    EXPECT_FALSE(first.refers_to_same_body_as(second));

    const CosseratRod& untouched = std::get<CosseratRod>(second.body());
    const Vector3DStack positions_before = untouched.positions();

    std::get<CosseratRod>(first.body()).mutable_velocities().setConstant(5.0);
    first.update_kinematics(0.0, 0.1);

    // The second body saw none of it.
    EXPECT_LT(untouched.velocities().cwiseAbs().maxCoeff(), 1e-15);
    EXPECT_TRUE(Near(untouched.positions(), positions_before));
}

// Self-contact and a self-joint both need to know that two endpoints are one
// body, which is identity rather than equality of state.
TEST(BodyVariantWrapperTest, IdentityIsNotStateEquality)
{
    const BodyVariantPtr shared = wrap(make_rod());
    const BodyVariantWrapper one(shared);
    const BodyVariantWrapper same(shared);
    const BodyVariantWrapper identical_but_separate(wrap(make_rod()));

    EXPECT_TRUE(one.refers_to_same_body_as(same));
    // Same contents, different body.
    EXPECT_FALSE(one.refers_to_same_body_as(identical_but_separate));
    EXPECT_TRUE(Near(state_of(one.body()), state_of(identical_but_separate.body())));
}

TEST(BodyVariantWrapperTest, TheWrapperOutlivesTheOriginalPointer)
{
    BodyVariantWrapper wrapped(wrap(make_rod()));
    {
        BodyVariantPtr temporary = wrap(make_sphere());
        wrapped = BodyVariantWrapper(temporary);
    }  // temporary is gone, but the wrapper still owns a share

    EXPECT_NO_THROW({ wrapped.update_kinematics(0.0, 1e-4); });
    EXPECT_TRUE(std::holds_alternative<Sphere>(wrapped.body()));
}

// ---------------------------------------------------------------------------
// The moved-from handle
// ---------------------------------------------------------------------------

// Moving leaves the source empty. Calling through it fails an assertion rather
// than dereferencing null.
TEST(BodyVariantWrapperDeathTest, AMovedFromHandleFailsCleanly)
{
    BodyVariantWrapper source(wrap(make_rod()));
    BodyVariantWrapper destination = std::move(source);

    EXPECT_NO_THROW({ destination.update_kinematics(0.0, 1e-4); });

    // NOLINTBEGIN(bugprone-use-after-move)
    EXPECT_ASSERT_FAILURE(source.update_kinematics(0.0, 1e-4));
    EXPECT_ASSERT_FAILURE(source.update_dynamics(0.0, 1e-4));
    EXPECT_ASSERT_FAILURE(source.update_accelerations(0.0, 1e-4));
    EXPECT_ASSERT_FAILURE(source.compute_internal_forces_and_torques(0.0));
    EXPECT_ASSERT_FAILURE(source.zero_out_external_forces_and_torques(0.0));
    EXPECT_ASSERT_FAILURE(source.body());
    // NOLINTEND(bugprone-use-after-move)
}

// ---------------------------------------------------------------------------
// Driving a mixed collection
// ---------------------------------------------------------------------------

// The point of the wrapper: one loop steps rods and rigid bodies alike.
TEST(BodyVariantWrapperTest, OneLoopStepsAMixedCollection)
{
    std::vector<BodyVariantWrapper> bodies{
        BodyVariantWrapper(wrap(make_rod())),
        BodyVariantWrapper(wrap(make_sphere())),
        BodyVariantWrapper(wrap(make_cylinder())),
    };

    std::vector<double> initial_heights;
    for (const BodyVariantWrapper& body : bodies)
    {
        initial_heights.push_back(std::visit([](const auto& concrete) {
            return concrete.positions().col(2).maxCoeff();
        }, body.body()));
    }

    // Give everything a downward pull.
    for (BodyVariantWrapper& body : bodies)
    {
        std::visit([](auto& concrete) {
            concrete.mutable_external_forces().col(2).array() -= 1.0;
        }, body.body());
    }

    const double dt = 1e-3;
    for (int step = 0; step < 20; ++step)
    {
        for (BodyVariantWrapper& body : bodies) body.update_kinematics(0.0, 0.5 * dt);
        for (BodyVariantWrapper& body : bodies)
        {
            body.compute_internal_forces_and_torques(0.0);
            body.update_accelerations(0.0, dt);
            body.update_dynamics(0.0, dt);
        }
        for (BodyVariantWrapper& body : bodies) body.update_kinematics(0.0, 0.5 * dt);
        // Loads are cleared at the end of the step, so re-apply next time.
        for (BodyVariantWrapper& body : bodies)
        {
            std::visit([](auto& concrete) {
                concrete.mutable_external_forces().col(2).array() -= 1.0;
            }, body.body());
        }
    }

    // Everything should have accelerated downward and fallen.
    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        std::visit([&](const auto& concrete) {
            EXPECT_LT(concrete.velocities().col(2).maxCoeff(), 0.0)
                << "body " << i << " is not moving downward";
            EXPECT_LT(concrete.positions().col(2).maxCoeff(),
                      initial_heights[i])
                << "body " << i << " did not fall";
        }, bodies[i].body());
    }
}

TEST(BodyVariantWrapperTest, AnUnloadedCollectionDoesNotMove)
{
    std::vector<BodyVariantWrapper> bodies{
        BodyVariantWrapper(wrap(make_rod())),
        BodyVariantWrapper(wrap(make_sphere())),
    };
    std::vector<Eigen::MatrixXd> before;
    for (const BodyVariantWrapper& body : bodies) before.push_back(state_of(body.body()));

    const double dt = 1e-4;
    for (int step = 0; step < 25; ++step)
    {
        for (BodyVariantWrapper& body : bodies)
        {
            body.update_kinematics(0.0, 0.5 * dt);
            body.compute_internal_forces_and_torques(0.0);
            body.update_accelerations(0.0, dt);
            body.update_dynamics(0.0, dt);
            body.update_kinematics(0.0, 0.5 * dt);
            body.zero_out_external_forces_and_torques(0.0);
        }
    }

    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        EXPECT_TRUE(Near(state_of(bodies[i].body()), before[i], 1e-9)) << "body " << i;
    }
}

}  // namespace
}  // namespace cosserat::physics
