#include "physics/joints.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>
#include <variant>
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
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Stub systems. Positions, velocities and forces are per-node; frames, angular
// velocities and torques are per-element.
// ---------------------------------------------------------------------------

struct RodStub
{
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_forces;
    Vector3DStack m_angular_velocities;
    Vector3DStack m_torques;
    Matrix3DStack m_frames;

    Vector3DStack& positions() { return m_positions; }
    Vector3DStack& velocities() { return m_velocities; }
    Vector3DStack& external_forces() { return m_forces; }
    const Matrix3DStack& frames() const { return m_frames; }
    Vector3DStack& angular_velocities() { return m_angular_velocities; }
    Vector3DStack& external_torques() { return m_torques; }
};

// Translation only: no frames, no angular velocities, no torques.
struct TranslationOnlySystem
{
    Vector3DStack m_positions = Vector3DStack::Zero(4, 3);
    Vector3DStack m_velocities = Vector3DStack::Zero(4, 3);
    Vector3DStack m_forces = Vector3DStack::Zero(4, 3);

    Vector3DStack& positions() { return m_positions; }
    Vector3DStack& velocities() { return m_velocities; }
    Vector3DStack& external_forces() { return m_forces; }
};

static_assert(ForceJointableSystem<RodStub>);
static_assert(TorqueJointableSystem<RodStub>);
static_assert(JointableSystem<RodStub>);
static_assert(ForceJointableSystem<TranslationOnlySystem>);
static_assert(!TorqueJointableSystem<TranslationOnlySystem>);
static_assert(!JointableSystem<TranslationOnlySystem>);

RodStub make_rod(double offset, Eigen::Index num_elements)
{
    const Eigen::Index nodes = num_elements + 1;
    RodStub rod;
    rod.m_positions = Vector3DStack::Zero(nodes, 3);
    rod.m_velocities = Vector3DStack::Zero(nodes, 3);
    rod.m_forces = Vector3DStack::Zero(nodes, 3);
    rod.m_angular_velocities = Vector3DStack::Zero(num_elements, 3);
    rod.m_torques = Vector3DStack::Zero(num_elements, 3);

    for (Eigen::Index i = 0; i < nodes; ++i)
    {
        rod.m_positions.row(i) << offset + i, offset + 2 * i, offset + 3 * i;
        rod.m_velocities.row(i)
            << 0.1 * (i + 1) + offset, 0.2 * (i + 1), 0.3 * (i + 1);
    }
    for (Eigen::Index i = 0; i < num_elements; ++i)
    {
        rod.m_angular_velocities.row(i)
            << 0.4 + 0.1 * i + offset, 0.5 - 0.2 * i, 0.6 + 0.3 * i;
        const Eigen::Vector3d axis =
            Eigen::Vector3d(1.0 + offset, 2.0, 3.0 - offset).normalized();
        rod.m_frames.push_back(
            Eigen::AngleAxisd(0.35 * (i + 1) + offset, axis).toRotationMatrix());
    }
    return rod;
}

RodStub make_rod(double offset) { return make_rod(offset, 3); }

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return ::testing::AssertionFailure() << "shape mismatch";
    }
    const double err = (a - b).cwiseAbs().maxCoeff();
    if (err < tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs diff " << err << " >= " << tol;
}

::testing::AssertionResult Near(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    return Near(a, b, kTol);
}

// A proper rotation whose third row is the requested tangent. Overwriting one
// row of the identity would not do: the result is singular, and multiplying a
// torque by it silently yields zero.
Eigen::Matrix3d frame_with_tangent(const Eigen::Vector3d& tangent)
{
    const Eigen::Vector3d d3 = tangent.normalized();
    const Eigen::Vector3d seed = (std::abs(d3(0)) < 0.9)
        ? Eigen::Vector3d::UnitX()
        : Eigen::Vector3d::UnitY();
    const Eigen::Vector3d d1 = (seed - seed.dot(d3) * d3).normalized();
    const Eigen::Vector3d d2 = d3.cross(d1);

    Eigen::Matrix3d frame;
    frame.row(0) = d1.transpose();
    frame.row(1) = d2.transpose();
    frame.row(2) = d3.transpose();
    return frame;
}

// ---------------------------------------------------------------------------
// FreeJoint
// ---------------------------------------------------------------------------

TEST(FreeJoint, AppliesEqualAndOppositeForces)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    const FreeJoint joint(3.0, 0.5);

    joint.apply_forces(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_forces.row(3), -two.m_forces.row(0)));
    EXPECT_GT(one.m_forces.row(3).norm(), 0.0);
}

TEST(FreeJoint, ForceIsStiffnessTimesSeparationPlusDampingTimesRelativeVelocity)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    const Eigen::RowVector3d separation =
        two.m_positions.row(0) - one.m_positions.row(3);
    const Eigen::RowVector3d relative_velocity =
        two.m_velocities.row(0) - one.m_velocities.row(3);

    const FreeJoint joint(3.0, 0.5);
    joint.apply_forces(one, -1, two, 0, 0.0);

    const Eigen::RowVector3d expected =
        3.0 * separation + 0.5 * relative_velocity;
    EXPECT_TRUE(Near(one.m_forces.row(3), expected));
    EXPECT_TRUE(Near(two.m_forces.row(0), -expected));
}

// The spring pulls the two connection points toward each other.
TEST(FreeJoint, SpringPullsSeparatedNodesTogether)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.0);
    one.m_velocities.setZero();
    two.m_velocities.setZero();
    two.m_positions.row(0) = one.m_positions.row(0) + Eigen::RowVector3d(1.0, 0.0, 0.0);

    FreeJoint(3.0, 0.0).apply_forces(one, 0, two, 0, 0.0);

    EXPECT_GT(one.m_forces(0, 0), 0.0);  // pulled toward system two
    EXPECT_LT(two.m_forces(0, 0), 0.0);  // pulled toward system one
}

TEST(FreeJoint, CoincidentNodesAtRestGiveNoForce)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.0);

    FreeJoint(3.0, 0.5).apply_forces(one, 2, two, 2, 0.0);

    EXPECT_TRUE(Near(one.m_forces, Vector3DStack::Zero(4, 3)));
    EXPECT_TRUE(Near(two.m_forces, Vector3DStack::Zero(4, 3)));
}

TEST(FreeJoint, ZeroCoefficientsGiveNoForce)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    FreeJoint(0.0, 0.0).apply_forces(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_forces, Vector3DStack::Zero(4, 3)));
}

TEST(FreeJoint, TouchesOnlyTheConnectedNodes)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    FreeJoint(3.0, 0.5).apply_forces(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_forces.topRows(3), Vector3DStack::Zero(3, 3)));
    EXPECT_TRUE(Near(two.m_forces.bottomRows(3), Vector3DStack::Zero(3, 3)));
}

TEST(FreeJoint, AccumulatesRatherThanOverwrites)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    one.m_forces.setConstant(1.0);
    RodStub fresh_one = make_rod(0.0);
    RodStub fresh_two = make_rod(0.7);

    const FreeJoint joint(3.0, 0.5);
    joint.apply_forces(one, -1, two, 0, 0.0);
    joint.apply_forces(fresh_one, -1, fresh_two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_forces,
                     Vector3DStack(fresh_one.m_forces + Vector3DStack::Ones(4, 3))));
}

TEST(FreeJoint, LeavesRotationFree)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    FreeJoint(3.0, 0.5).apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3)));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3)));
}

TEST(FreeJoint, NegativeIndicesAddressTheEnd)
{
    RodStub explicit_one = make_rod(0.0);
    RodStub explicit_two = make_rod(0.7);
    RodStub negative_one = make_rod(0.0);
    RodStub negative_two = make_rod(0.7);

    const FreeJoint joint(3.0, 0.5);
    joint.apply_forces(explicit_one, 3, explicit_two, 3, 0.0);
    joint.apply_forces(negative_one, -1, negative_two, -1, 0.0);

    EXPECT_TRUE(Near(explicit_one.m_forces, negative_one.m_forces));
}

TEST(FreeJoint, ConnectsSystemsOfDifferentSizes)
{
    RodStub small = make_rod(0.0, 2);
    RodStub large = make_rod(0.7, 8);

    EXPECT_NO_THROW({ FreeJoint(3.0, 0.5).apply_forces(small, -1, large, 0, 0.0); });
    EXPECT_GT(small.m_forces.row(2).norm(), 0.0);
}

TEST(FreeJoint, WorksOnTranslationOnlySystems)
{
    TranslationOnlySystem one;
    TranslationOnlySystem two;
    two.m_positions.row(0) << 1.0, 0.0, 0.0;

    EXPECT_NO_THROW({ FreeJoint(3.0, 0.0).apply_forces(one, 0, two, 0, 0.0); });
    EXPECT_TRUE(Near(one.m_forces.row(0), Eigen::RowVector3d(3.0, 0.0, 0.0)));
}

TEST(FreeJoint, ExposesConstructorArguments)
{
    const FreeJoint joint(3.0, 0.5);

    EXPECT_DOUBLE_EQ(joint.stiffness(), 3.0);
    EXPECT_DOUBLE_EQ(joint.damping(), 0.5);
}

TEST(FreeJoint, AliasesNameTheSameType)
{
    static_assert(std::is_same_v<BallJoint, FreeJoint>);
    static_assert(std::is_same_v<SphericalJoint, FreeJoint>);
}

TEST(FreeJointDeathTest, RejectsBadCoefficients)
{
    EXPECT_ASSERT_FAILURE(FreeJoint(-1.0, 0.5));
    EXPECT_ASSERT_FAILURE(FreeJoint(3.0, -0.5));
    EXPECT_ASSERT_FAILURE(FreeJoint(kNaN, 0.5));
    EXPECT_ASSERT_FAILURE(FreeJoint(3.0, kInf));
}

TEST(FreeJointDeathTest, RejectsOutOfRangeIndices)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    const FreeJoint joint(3.0, 0.5);

    EXPECT_ASSERT_FAILURE(joint.apply_forces(one, 99, two, 0, 0.0));
    EXPECT_ASSERT_FAILURE(joint.apply_forces(one, 0, two, -99, 0.0));
}

// ---------------------------------------------------------------------------
// HingeJoint
// ---------------------------------------------------------------------------

TEST(HingeJoint, TestFrameHelperBuildsAProperRotation)
{
    const Eigen::Vector3d tangent =
        Eigen::Vector3d(std::sqrt(0.5), std::sqrt(0.5), 0.0);
    const Eigen::Matrix3d frame = frame_with_tangent(tangent);

    EXPECT_TRUE(Near(frame * frame.transpose(), Eigen::Matrix3d::Identity()));
    EXPECT_NEAR(frame.determinant(), 1.0, 1e-12);
    EXPECT_TRUE(Near(frame.row(2).transpose(), tangent));
}

TEST(HingeJoint, ForcesMatchFreeJoint)
{
    RodStub hinge_one = make_rod(0.0);
    RodStub hinge_two = make_rod(0.7);
    RodStub free_one = make_rod(0.0);
    RodStub free_two = make_rod(0.7);

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_forces(hinge_one, -1, hinge_two, 0, 0.0);
    FreeJoint(3.0, 0.5).apply_forces(free_one, -1, free_two, 0, 0.0);

    EXPECT_TRUE(Near(hinge_one.m_forces, free_one.m_forces));
    EXPECT_TRUE(Near(hinge_two.m_forces, free_two.m_forces));
}

TEST(HingeJoint, NormalDirectionIsNormalised)
{
    const HingeJoint joint(3.0, 0.5, 2.5, Eigen::Vector3d(0.0, 0.0, 5.0));

    EXPECT_TRUE(Near(joint.normal_direction(), Eigen::Vector3d::UnitZ()));
    EXPECT_NEAR(joint.normal_direction().norm(), 1.0, 1e-15);
}

TEST(HingeJoint, ScalingTheNormalDoesNotChangeTheTorque)
{
    RodStub unit_one = make_rod(0.0);
    RodStub unit_two = make_rod(0.7);
    RodStub scaled_one = make_rod(0.0);
    RodStub scaled_two = make_rod(0.7);

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d(1.0, 0.0, 0.0))
        .apply_torques(unit_one, -1, unit_two, 0, 0.0);
    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d(17.0, 0.0, 0.0))
        .apply_torques(scaled_one, -1, scaled_two, 0, 0.0);

    EXPECT_TRUE(Near(unit_one.m_torques, scaled_one.m_torques));
}

// A tangent already lying in the hinge plane has nothing to correct.
TEST(HingeJoint, TangentInPlaneGivesNoTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    // Make element 0 of system two have its tangent along +y, and constrain
    // rotation to the yz plane, whose normal is +x.
    two.m_frames[0] = frame_with_tangent(Eigen::Vector3d(0.0, 1.0, 0.0));

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3)));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3)));
}

// A tangent parallel to the plane normal has no in-plane component to rotate
// toward, so the cross product vanishes and there is still no torque.
TEST(HingeJoint, TangentAlongNormalGivesNoTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    two.m_frames[0] = frame_with_tangent(Eigen::Vector3d(1.0, 0.0, 0.0));

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3)));
}

TEST(HingeJoint, TiltedTangentProducesTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    two.m_frames[0] =
        frame_with_tangent(Eigen::Vector3d(std::sqrt(0.5), std::sqrt(0.5), 0.0));

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    EXPECT_GT(two.m_torques.row(0).norm(), 0.0);
}

TEST(HingeJoint, TorquesUseEachSystemsOwnFrame)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    const Eigen::Matrix3d frame_one = one.m_frames[2];
    const Eigen::Matrix3d frame_two = two.m_frames[0];

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    const Eigen::Vector3d tangent = frame_two.row(2).transpose();
    const Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d out_of_plane = -tangent.dot(normal) * normal;
    const Eigen::Vector3d torque = 2.5 * tangent.cross(out_of_plane);

    EXPECT_TRUE(Near(one.m_torques.row(2), (-(frame_one * torque)).transpose()));
    EXPECT_TRUE(Near(two.m_torques.row(0), (frame_two * torque).transpose()));
}

TEST(HingeJoint, ZeroRotationalStiffnessGivesNoTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    HingeJoint(3.0, 0.5, 0.0, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3)));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3)));
}

TEST(HingeJoint, TouchesOnlyTheConnectedElements)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())
        .apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques.topRows(2), Vector3DStack::Zero(2, 3)));
    EXPECT_TRUE(Near(two.m_torques.bottomRows(2), Vector3DStack::Zero(2, 3)));
}

TEST(HingeJoint, ExposesConstructorArguments)
{
    const HingeJoint joint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitZ());

    EXPECT_DOUBLE_EQ(joint.stiffness(), 3.0);
    EXPECT_DOUBLE_EQ(joint.damping(), 0.5);
    EXPECT_DOUBLE_EQ(joint.rotational_stiffness(), 2.5);
    EXPECT_TRUE(Near(joint.normal_direction(), Eigen::Vector3d::UnitZ()));
}

TEST(HingeJointDeathTest, RejectsBadCoefficients)
{
    EXPECT_ASSERT_FAILURE(HingeJoint(-1.0, 0.5, 2.5, Eigen::Vector3d::UnitX()));
    EXPECT_ASSERT_FAILURE(HingeJoint(3.0, 0.5, -2.5, Eigen::Vector3d::UnitX()));
    EXPECT_ASSERT_FAILURE(HingeJoint(3.0, 0.5, kNaN, Eigen::Vector3d::UnitX()));
}

TEST(HingeJointDeathTest, RejectsDegenerateNormalDirection)
{
    EXPECT_ASSERT_FAILURE(HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::Zero()));
    EXPECT_ASSERT_FAILURE(HingeJoint(
        3.0, 0.5, 2.5,
        Eigen::Vector3d(0.5 * joint_direction_tolerance, 0.0, 0.0)));
    EXPECT_ASSERT_FAILURE(HingeJoint(
        3.0, 0.5, 2.5, Eigen::Vector3d(kNaN, 0.0, 0.0)));
}

// ---------------------------------------------------------------------------
// FixedJoint
// ---------------------------------------------------------------------------

FixedJoint make_fixed(double rotational_stiffness, double rotational_damping)
{
    return FixedJoint(3.0, 0.5, rotational_stiffness, rotational_damping,
                      Eigen::Matrix3d::Identity());
}

TEST(FixedJoint, ForcesMatchFreeJoint)
{
    RodStub fixed_one = make_rod(0.0);
    RodStub fixed_two = make_rod(0.7);
    RodStub free_one = make_rod(0.0);
    RodStub free_two = make_rod(0.7);

    make_fixed(2.5, 0.9).apply_forces(fixed_one, -1, fixed_two, 0, 0.0);
    FreeJoint(3.0, 0.5).apply_forces(free_one, -1, free_two, 0, 0.0);

    EXPECT_TRUE(Near(fixed_one.m_forces, free_one.m_forces));
    EXPECT_TRUE(Near(fixed_two.m_forces, free_two.m_forces));
}

// Identity rest rotation with already-aligned frames and matched rates means
// there is no deviation to correct.
TEST(FixedJoint, AlignedFramesAtRestGiveNoTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.0);
    one.m_angular_velocities.setZero();
    two.m_angular_velocities.setZero();

    make_fixed(2.5, 0.9).apply_torques(one, 1, two, 1, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3), 1e-9));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3), 1e-9));
}

TEST(FixedJoint, MisalignedFramesProduceTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.0);
    one.m_angular_velocities.setZero();
    two.m_angular_velocities.setZero();
    two.m_frames[1] =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()).toRotationMatrix()
        * two.m_frames[1];

    make_fixed(2.5, 0.0).apply_torques(one, 1, two, 1, 0.0);

    EXPECT_GT(one.m_torques.row(1).norm(), 0.0);
    EXPECT_GT(two.m_torques.row(1).norm(), 0.0);
}

// A rest rotation matching the current misalignment cancels the spring term.
TEST(FixedJoint, MatchingRestRotationCancelsTheSpringTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    one.m_angular_velocities.setZero();
    two.m_angular_velocities.setZero();

    const Eigen::Matrix3d rest =
        one.m_frames[2] * two.m_frames[0].transpose();
    const FixedJoint joint(3.0, 0.5, 2.5, 0.0, rest);

    joint.apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3), 1e-9));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3), 1e-9));
}

TEST(FixedJoint, DampingOpposesRelativeAngularVelocity)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.0);
    one.m_angular_velocities.setZero();
    two.m_angular_velocities.setZero();
    two.m_angular_velocities.row(1) << 0.0, 0.0, 1.0;

    make_fixed(0.0, 0.9).apply_torques(one, 1, two, 1, 0.0);

    // With no spring term the torque comes purely from the rate difference.
    const Eigen::Matrix3d director = two.m_frames[1];
    const Eigen::Vector3d deviation =
        director.transpose() * Eigen::Vector3d(0.0, 0.0, 1.0);
    const Eigen::Vector3d expected = -0.9 * deviation;

    EXPECT_TRUE(Near(two.m_torques.row(1), (director * expected).transpose(), 1e-9));
}

TEST(FixedJoint, ZeroCoefficientsGiveNoTorque)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    make_fixed(0.0, 0.0).apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3)));
    EXPECT_TRUE(Near(two.m_torques, Vector3DStack::Zero(3, 3)));
}

TEST(FixedJoint, TorquesAreOppositeInTheInertialFrame)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    const Eigen::Matrix3d frame_one = one.m_frames[2];
    const Eigen::Matrix3d frame_two = two.m_frames[0];

    make_fixed(2.5, 0.9).apply_torques(one, -1, two, 0, 0.0);

    // Each system stores torque in its own material frame; moving both back to
    // the inertial frame should reveal an equal and opposite pair.
    const Eigen::Vector3d inertial_one =
        frame_one.transpose() * one.m_torques.row(2).transpose();
    const Eigen::Vector3d inertial_two =
        frame_two.transpose() * two.m_torques.row(0).transpose();

    EXPECT_TRUE(Near(inertial_one, Eigen::Vector3d(-inertial_two), 1e-10));
}

TEST(FixedJoint, TouchesOnlyTheConnectedElements)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    make_fixed(2.5, 0.9).apply_torques(one, -1, two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques.topRows(2), Vector3DStack::Zero(2, 3)));
    EXPECT_TRUE(Near(two.m_torques.bottomRows(2), Vector3DStack::Zero(2, 3)));
}

TEST(FixedJoint, AccumulatesRatherThanOverwrites)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    one.m_torques.setConstant(1.0);
    RodStub fresh_one = make_rod(0.0);
    RodStub fresh_two = make_rod(0.7);

    const FixedJoint joint = make_fixed(2.5, 0.9);
    joint.apply_torques(one, -1, two, 0, 0.0);
    joint.apply_torques(fresh_one, -1, fresh_two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_torques,
                     Vector3DStack(fresh_one.m_torques + Vector3DStack::Ones(3, 3))));
}

TEST(FixedJoint, ExposesConstructorArguments)
{
    const Eigen::Matrix3d rest =
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const FixedJoint joint(3.0, 0.5, 2.5, 0.9, rest);

    EXPECT_DOUBLE_EQ(joint.stiffness(), 3.0);
    EXPECT_DOUBLE_EQ(joint.damping(), 0.5);
    EXPECT_DOUBLE_EQ(joint.rotational_stiffness(), 2.5);
    EXPECT_DOUBLE_EQ(joint.rotational_damping(), 0.9);
    EXPECT_TRUE(Near(joint.rest_rotation_matrix(), rest));
}

TEST(FixedJointDeathTest, RejectsBadCoefficients)
{
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    EXPECT_ASSERT_FAILURE(FixedJoint(-1.0, 0.5, 2.5, 0.9, identity));
    EXPECT_ASSERT_FAILURE(FixedJoint(3.0, 0.5, -2.5, 0.9, identity));
    EXPECT_ASSERT_FAILURE(FixedJoint(3.0, 0.5, 2.5, -0.9, identity));
    EXPECT_ASSERT_FAILURE(FixedJoint(3.0, 0.5, 2.5, kNaN, identity));
}

// Stricter than the reference, which only checks the matrix shape.
TEST(FixedJointDeathTest, RejectsNonRotationRestMatrix)
{
    EXPECT_ASSERT_FAILURE(FixedJoint(
        3.0, 0.5, 2.5, 0.9, Eigen::Matrix3d::Constant(kNaN)));
    EXPECT_ASSERT_FAILURE(FixedJoint(
        3.0, 0.5, 2.5, 0.9, Eigen::Matrix3d::Zero()));
    EXPECT_ASSERT_FAILURE(FixedJoint(
        3.0, 0.5, 2.5, 0.9, Eigen::Matrix3d(2.0 * Eigen::Matrix3d::Identity())));

    // Orthogonal but a reflection, so the determinant is -1.
    Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
    reflection(2, 2) = -1.0;
    EXPECT_ASSERT_FAILURE(FixedJoint(3.0, 0.5, 2.5, 0.9, reflection));
}

TEST(FixedJoint, AcceptsAnyProperRotationAsRest)
{
    const Eigen::Matrix3d rest =
        Eigen::AngleAxisd(1.2, Eigen::Vector3d(1, -2, 3).normalized())
            .toRotationMatrix();

    EXPECT_NO_THROW({ FixedJoint(3.0, 0.5, 2.5, 0.9, rest); });
}

// ---------------------------------------------------------------------------
// JointVariant dispatch
// ---------------------------------------------------------------------------

JointVariant free_variant() { return JointVariant{FreeJoint(3.0, 0.5)}; }
JointVariant hinge_variant()
{
    return JointVariant{HingeJoint(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX())};
}
JointVariant fixed_variant()
{
    return JointVariant{FixedJoint(3.0, 0.5, 2.5, 0.9, Eigen::Matrix3d::Identity())};
}

std::vector<JointVariant> all_joints()
{
    return {free_variant(), hinge_variant(), fixed_variant()};
}

TEST(JointVariant, IsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<JointVariant>);
    static_assert(std::is_copy_assignable_v<JointVariant>);
    static_assert(std::is_move_assignable_v<JointVariant>);

    JointVariant a = free_variant();
    const JointVariant b = fixed_variant();

    a = b;

    EXPECT_TRUE(std::holds_alternative<FixedJoint>(a));
}

TEST(JointVariant, WorksInStandardContainers)
{
    std::vector<JointVariant> joints = all_joints();

    ASSERT_EQ(joints.size(), 3u);
    joints.erase(joints.begin());

    EXPECT_EQ(joints.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<HingeJoint>(joints.front()));
}

TEST(ValidateJoint, AcceptsEveryAlternativeOnFullRods)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);

    for (auto& joint : all_joints())
    {
        EXPECT_NO_THROW({ validate(joint, one, two); });
    }
}

TEST(ValidateJoint, AcceptsFreeJointOnTranslationOnlySystems)
{
    TranslationOnlySystem one;
    TranslationOnlySystem two;
    JointVariant joint = free_variant();

    EXPECT_NO_THROW({ validate(joint, one, two); });
}

TEST(ValidateJointDeathTest, RejectsRotationalJointsOnTranslationOnlySystems)
{
    TranslationOnlySystem one;
    TranslationOnlySystem two;
    JointVariant hinge = hinge_variant();
    JointVariant fixed = fixed_variant();

    EXPECT_ASSERT_FAILURE(validate(hinge, one, two));
    EXPECT_ASSERT_FAILURE(validate(fixed, one, two));
}

// One system may be rotational while the other is not.
TEST(ValidateJointDeathTest, RejectsMixedPairForRotationalJoints)
{
    RodStub full = make_rod(0.0);
    TranslationOnlySystem partial;
    JointVariant hinge = hinge_variant();

    EXPECT_ASSERT_FAILURE(validate(hinge, full, partial));
}

TEST(ApplyForcesVariant, MatchesDirectFreeJointCall)
{
    RodStub variant_one = make_rod(0.0);
    RodStub variant_two = make_rod(0.7);
    RodStub direct_one = make_rod(0.0);
    RodStub direct_two = make_rod(0.7);

    JointVariant joint = free_variant();
    apply_forces(joint, variant_one, -1, variant_two, 0, 0.0);
    FreeJoint(3.0, 0.5).apply_forces(direct_one, -1, direct_two, 0, 0.0);

    EXPECT_TRUE(Near(variant_one.m_forces, direct_one.m_forces));
    EXPECT_TRUE(Near(variant_two.m_forces, direct_two.m_forces));
}

TEST(ApplyTorquesVariant, MatchesDirectFixedJointCall)
{
    RodStub variant_one = make_rod(0.0);
    RodStub variant_two = make_rod(0.7);
    RodStub direct_one = make_rod(0.0);
    RodStub direct_two = make_rod(0.7);

    JointVariant joint = fixed_variant();
    apply_torques(joint, variant_one, -1, variant_two, 0, 0.0);
    FixedJoint(3.0, 0.5, 2.5, 0.9, Eigen::Matrix3d::Identity())
        .apply_torques(direct_one, -1, direct_two, 0, 0.0);

    EXPECT_TRUE(Near(variant_one.m_torques, direct_one.m_torques));
    EXPECT_TRUE(Near(variant_two.m_torques, direct_two.m_torques));
}

TEST(ApplyTorquesVariant, FreeJointIsANoOp)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    JointVariant joint = free_variant();

    EXPECT_NO_THROW({ apply_torques(joint, one, -1, two, 0, 0.0); });
    EXPECT_TRUE(Near(one.m_torques, Vector3DStack::Zero(3, 3)));
}

TEST(ApplyVariantDeathTest, RejectsIncompatibleSystems)
{
    TranslationOnlySystem one;
    TranslationOnlySystem two;
    JointVariant hinge = hinge_variant();

    EXPECT_ASSERT_FAILURE(apply_torques(hinge, one, 0, two, 0, 0.0));
}

// A joint list applied in sequence composes each contribution.
TEST(ApplyVariant, ComposesAcrossAList)
{
    RodStub one = make_rod(0.0);
    RodStub two = make_rod(0.7);
    RodStub expected_one = make_rod(0.0);
    RodStub expected_two = make_rod(0.7);

    std::vector<JointVariant> joints{free_variant(), hinge_variant()};

    for (auto& joint : joints) validate(joint, one, two);
    for (auto& joint : joints) apply_forces(joint, one, -1, two, 0, 0.0);
    for (auto& joint : joints) apply_torques(joint, one, -1, two, 0, 0.0);

    FreeJoint(3.0, 0.5).apply_forces(expected_one, -1, expected_two, 0, 0.0);
    const HingeJoint hinge(3.0, 0.5, 2.5, Eigen::Vector3d::UnitX());
    hinge.apply_forces(expected_one, -1, expected_two, 0, 0.0);
    hinge.apply_torques(expected_one, -1, expected_two, 0, 0.0);

    EXPECT_TRUE(Near(one.m_forces, expected_one.m_forces));
    EXPECT_TRUE(Near(one.m_torques, expected_one.m_torques));
    EXPECT_TRUE(Near(two.m_torques, expected_two.m_torques));
}

// A joint may connect a system to itself at two different indices.
TEST(ApplyVariant, ConnectsASystemToItself)
{
    RodStub rod = make_rod(0.0);
    JointVariant joint = free_variant();

    EXPECT_NO_THROW({ apply_forces(joint, rod, 0, rod, -1, 0.0); });
    EXPECT_TRUE(Near(rod.m_forces.row(0), -rod.m_forces.row(3)));
}

}  // namespace
}  // namespace cosserat::physics
