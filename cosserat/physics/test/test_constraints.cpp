#include <cosserat/physics/constraints.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>
#include <variant>
#include <vector>

#include <cosserat/math/linalg.hpp>

namespace cosserat::physics {
namespace {

using math::rotation_matrix;

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
// Stub systems. Positions and velocities are per-node; directors and angular
// velocities are per-element, so the stacks differ in length by one.
// ---------------------------------------------------------------------------

struct RodStub
{
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_angular_velocities;
    Matrix3DStack m_directors;

    Vector3DStack& mutable_positions() { return m_positions; }
    Vector3DStack& mutable_velocities() { return m_velocities; }
    Vector3DStack& mutable_angular_velocities() { return m_angular_velocities; }
    Matrix3DStack& mutable_frames() { return m_directors; }
};

// Positions only: no directors, no angular velocities.
struct PositionOnlySystem
{
    Vector3DStack m_positions = Vector3DStack::Zero(5, 3);
    Vector3DStack m_velocities = Vector3DStack::Zero(5, 3);

    Vector3DStack& mutable_positions() { return m_positions; }
    Vector3DStack& mutable_velocities() { return m_velocities; }
};

static_assert(PositionConstrainableSystem<RodStub>);
static_assert(DirectorConstrainableSystem<RodStub>);
static_assert(ConstrainableSystem<RodStub>);
static_assert(PositionConstrainableSystem<PositionOnlySystem>);
static_assert(!DirectorConstrainableSystem<PositionOnlySystem>);
static_assert(!ConstrainableSystem<PositionOnlySystem>);

RodStub make_rod(Eigen::Index num_elements)
{
    const Eigen::Index nodes = num_elements + 1;
    RodStub rod;
    rod.m_positions = Vector3DStack::Zero(nodes, 3);
    rod.m_velocities = Vector3DStack::Zero(nodes, 3);
    rod.m_angular_velocities = Vector3DStack::Zero(num_elements, 3);
    for (Eigen::Index i = 0; i < nodes; ++i)
    {
        rod.m_positions.row(i) << 1.0 + i, 2.0 + i, 3.0 + i;
        rod.m_velocities.row(i) << 0.5 + i, 1.5 + i, 2.5 + i;
    }
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    for (Eigen::Index i = 0; i < num_elements; ++i)
    {
        rod.m_angular_velocities.row(i) << 1.0 + 0.5 * i, 2.0 - 0.3 * i, 0.7 + i;
        rod.m_directors.push_back(
            Eigen::AngleAxisd(0.4 * (i + 1), axis).toRotationMatrix());
    }
    return rod;
}

RodStub make_rod() { return make_rod(4); }

Vector3DStack stack_of(std::initializer_list<Eigen::Vector3d> rows)
{
    Vector3DStack stack(static_cast<Eigen::Index>(rows.size()), 3);
    Eigen::Index i = 0;
    for (const Eigen::Vector3d& row : rows) stack.row(i++) = row.transpose();
    return stack;
}

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

// ---------------------------------------------------------------------------
// FreeBoundaryCondition
// ---------------------------------------------------------------------------

TEST(FreeBoundaryCondition, LeavesEverythingUntouched)
{
    RodStub rod = make_rod();
    const RodStub original = rod;

    const FreeBoundaryCondition constraint;
    constraint.constrain_values(rod, 1.0);
    constraint.constrain_rates(rod, 1.0);

    EXPECT_TRUE(Near(rod.m_positions, original.m_positions));
    EXPECT_TRUE(Near(rod.m_velocities, original.m_velocities));
    EXPECT_TRUE(Near(rod.m_angular_velocities, original.m_angular_velocities));
    for (std::size_t i = 0; i < rod.m_directors.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.m_directors[i], original.m_directors[i]));
    }
}

// Unconstrained, so it accepts systems the other rules would reject.
TEST(FreeBoundaryCondition, AcceptsAnySystemType)
{
    PositionOnlySystem sys;
    const FreeBoundaryCondition constraint;

    EXPECT_NO_THROW({ constraint.constrain_values(sys, 0.0); });
    EXPECT_NO_THROW({ constraint.constrain_rates(sys, 0.0); });
}

// ---------------------------------------------------------------------------
// OneEndFixedBoundaryCondition
// ---------------------------------------------------------------------------

TEST(OneEndFixedBoundaryCondition, PinsFirstNodeAndElement)
{
    RodStub rod = make_rod();
    const Eigen::Vector3d target(9.0, 8.0, 7.0);
    const Eigen::Matrix3d orientation =
        Eigen::AngleAxisd(0.9, Eigen::Vector3d::UnitY()).toRotationMatrix();

    const OneEndFixedBoundaryCondition constraint(target, orientation);
    constraint.constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0), target.transpose()));
    EXPECT_TRUE(Near(rod.m_directors[0], orientation));
}

TEST(OneEndFixedBoundaryCondition, ZeroesFirstNodeAndElementRates)
{
    RodStub rod = make_rod();
    const OneEndFixedBoundaryCondition constraint(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());

    constraint.constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(0), Eigen::RowVector3d::Zero()));
}

TEST(OneEndFixedBoundaryCondition, LeavesRemainingEntriesAlone)
{
    RodStub rod = make_rod();
    const RodStub original = rod;
    const OneEndFixedBoundaryCondition constraint(
        Eigen::Vector3d(9, 8, 7), Eigen::Matrix3d::Identity());

    constraint.constrain_values(rod, 0.0);
    constraint.constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.bottomRows(4), original.m_positions.bottomRows(4)));
    EXPECT_TRUE(Near(rod.m_velocities.bottomRows(4),
                     original.m_velocities.bottomRows(4)));
    EXPECT_TRUE(Near(rod.m_angular_velocities.bottomRows(3),
                     original.m_angular_velocities.bottomRows(3)));
    for (std::size_t i = 1; i < rod.m_directors.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.m_directors[i], original.m_directors[i])) << "element " << i;
    }
}

TEST(OneEndFixedBoundaryCondition, IsIdempotent)
{
    RodStub once = make_rod();
    RodStub twice = make_rod();
    const OneEndFixedBoundaryCondition constraint(
        Eigen::Vector3d(9, 8, 7), Eigen::Matrix3d::Identity());

    constraint.constrain_values(once, 0.0);
    constraint.constrain_values(twice, 0.0);
    constraint.constrain_values(twice, 0.0);

    EXPECT_TRUE(Near(once.m_positions, twice.m_positions));
}

TEST(OneEndFixedBoundaryCondition, ExposesConstructorArguments)
{
    const Eigen::Vector3d target(1, 2, 3);
    const OneEndFixedBoundaryCondition constraint(target, Eigen::Matrix3d::Identity());

    EXPECT_TRUE(Near(constraint.fixed_position(), target));
    EXPECT_TRUE(Near(constraint.fixed_directors(), Eigen::Matrix3d::Identity()));
}

TEST(OneEndFixedBoundaryConditionDeathTest, RejectsNonFiniteTargets)
{
    EXPECT_ASSERT_FAILURE(OneEndFixedBoundaryCondition(
        Eigen::Vector3d(kNaN, 0, 0), Eigen::Matrix3d::Identity()));
    EXPECT_ASSERT_FAILURE(OneEndFixedBoundaryCondition(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Constant(kInf)));
}

TEST(OneEndFixedBoundaryConditionDeathTest, RejectsEmptySystem)
{
    RodStub empty;
    empty.m_positions = Vector3DStack::Zero(0, 3);
    empty.m_velocities = Vector3DStack::Zero(0, 3);
    empty.m_angular_velocities = Vector3DStack::Zero(0, 3);

    const OneEndFixedBoundaryCondition constraint(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity());

    EXPECT_ASSERT_FAILURE(constraint.constrain_values(empty, 0.0));
    EXPECT_ASSERT_FAILURE(constraint.constrain_rates(empty, 0.0));
}

// ---------------------------------------------------------------------------
// GeneralConstraint
// ---------------------------------------------------------------------------

GeneralConstraint make_general(
    std::array<bool, 3> translational, std::array<bool, 3> rotational)
{
    return GeneralConstraint(
        {0, -1},
        stack_of({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}}),
        {0, -1},
        translational,
        rotational);
}

GeneralConstraint make_general()
{
    return make_general({true, true, true}, {true, true, true});
}

GeneralConstraint make_general(std::array<bool, 3> translational)
{
    return make_general(translational, {true, true, true});
}

TEST(GeneralConstraint, FullSelectorPinsPositionsOutright)
{
    RodStub rod = make_rod();
    make_general().constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0), Eigen::RowVector3d(10, 20, 30)));
    EXPECT_TRUE(Near(rod.m_positions.row(4), Eigen::RowVector3d(40, 50, 60)));
}

// A false selector entry leaves that axis at its current value.
TEST(GeneralConstraint, PartialSelectorLeavesFreeAxesUnchanged)
{
    RodStub rod = make_rod();
    const Eigen::RowVector3d before_first = rod.m_positions.row(0);
    const Eigen::RowVector3d before_last = rod.m_positions.row(4);

    make_general({true, true, false}).constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0),
                     Eigen::RowVector3d(10, 20, before_first(2))));
    EXPECT_TRUE(Near(rod.m_positions.row(4),
                     Eigen::RowVector3d(40, 50, before_last(2))));
}

TEST(GeneralConstraint, EmptySelectorLeavesPositionsUntouched)
{
    RodStub rod = make_rod();
    const Vector3DStack original = rod.m_positions;

    make_general({false, false, false}).constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions, original));
}

TEST(GeneralConstraint, NegativeIndicesAddressTheEnd)
{
    RodStub rod = make_rod();
    const Vector3DStack original = rod.m_positions;

    make_general().constrain_values(rod, 0.0);

    // Only the first and last rows should have moved.
    EXPECT_TRUE(Near(rod.m_positions.middleRows(1, 3), original.middleRows(1, 3)));
    EXPECT_FALSE(Near(rod.m_positions.row(4), original.row(4)));
}

TEST(GeneralConstraint, FullSelectorZeroesConstrainedVelocities)
{
    RodStub rod = make_rod();
    make_general().constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_velocities.row(4), Eigen::RowVector3d::Zero()));
}

TEST(GeneralConstraint, PartialSelectorZeroesOnlyChosenVelocityAxes)
{
    RodStub rod = make_rod();
    const Eigen::RowVector3d before = rod.m_velocities.row(0);

    make_general({true, false, true}).constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0),
                     Eigen::RowVector3d(0.0, before(1), 0.0)));
}

TEST(GeneralConstraint, LeavesUnconstrainedRowsAlone)
{
    RodStub rod = make_rod();
    const RodStub original = rod;

    make_general().constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.middleRows(1, 3),
                     original.m_velocities.middleRows(1, 3)));
    EXPECT_TRUE(Near(rod.m_angular_velocities.middleRows(1, 2),
                     original.m_angular_velocities.middleRows(1, 2)));
}

// Rotational masking happens in the inertial frame, so the result must be
// computed by rotating out of the material frame, masking, and rotating back.
TEST(GeneralConstraint, RotationalMaskingActsInTheLabFrame)
{
    RodStub rod = make_rod();
    const RodStub original = rod;
    const std::array<bool, 3> rotational{true, false, true};

    make_general({true, true, true}, rotational).constrain_rates(rod, 0.0);

    for (Eigen::Index idx : {Eigen::Index{0}, Eigen::Index{3}})
    {
        const Eigen::Matrix3d& director = original.m_directors[idx];
        Eigen::Vector3d lab =
            director.transpose() * original.m_angular_velocities.row(idx).transpose();
        lab(0) = 0.0;
        lab(2) = 0.0;
        const Eigen::Vector3d expected = director * lab;

        EXPECT_TRUE(Near(rod.m_angular_velocities.row(idx), expected.transpose()))
            << "element " << idx;
    }
}

TEST(GeneralConstraint, FullRotationalSelectorZeroesAngularVelocity)
{
    RodStub rod = make_rod();
    make_general().constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_angular_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(3), Eigen::RowVector3d::Zero()));
}

// With no rotational axes constrained the angular velocity must round-trip
// through the two frame conversions unchanged.
TEST(GeneralConstraint, EmptyRotationalSelectorPreservesAngularVelocity)
{
    RodStub rod = make_rod();
    const Vector3DStack original = rod.m_angular_velocities;

    make_general({true, true, true}, {false, false, false}).constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_angular_velocities, original, 1e-13));
}

TEST(GeneralConstraint, ConstrainValuesWorksOnPositionOnlySystem)
{
    PositionOnlySystem sys;
    for (Eigen::Index i = 0; i < 5; ++i) sys.m_positions.row(i).setConstant(1.0 + i);

    make_general().constrain_values(sys, 0.0);

    EXPECT_TRUE(Near(sys.m_positions.row(0), Eigen::RowVector3d(10, 20, 30)));
}

TEST(GeneralConstraint, ExposesConstructorArguments)
{
    const GeneralConstraint constraint = make_general({true, false, true},
                                                      {false, true, false});

    EXPECT_EQ(constraint.position_indices(), (std::vector<std::int64_t>{0, -1}));
    EXPECT_EQ(constraint.director_indices(), (std::vector<std::int64_t>{0, -1}));
    EXPECT_EQ(constraint.fixed_positions().rows(), 2);
    EXPECT_TRUE(Near(constraint.translational_selector().matrix(),
                     Eigen::Vector3d(1.0, 0.0, 1.0)));
    EXPECT_TRUE(Near(constraint.rotational_selector().matrix(),
                     Eigen::Vector3d(0.0, 1.0, 0.0)));
}

TEST(GeneralConstraintDeathTest, RejectsMismatchedPositionCounts)
{
    const std::array<bool, 3> all{true, true, true};
    EXPECT_ASSERT_FAILURE(GeneralConstraint(
        {0, -1}, stack_of({{1.0, 2.0, 3.0}}), {}, all, all));
    EXPECT_ASSERT_FAILURE(GeneralConstraint(
        {0}, stack_of({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}), {}, all, all));
}

TEST(GeneralConstraintDeathTest, RejectsConstraintThatDoesNothing)
{
    const std::array<bool, 3> all{true, true, true};
    EXPECT_ASSERT_FAILURE(
        GeneralConstraint({}, Vector3DStack::Zero(0, 3), {}, all, all));
}

TEST(GeneralConstraintDeathTest, RejectsNonFiniteTargets)
{
    const std::array<bool, 3> all{true, true, true};
    EXPECT_ASSERT_FAILURE(GeneralConstraint(
        {0}, stack_of({{kNaN, 0.0, 0.0}}), {}, all, all));
}

TEST(GeneralConstraintDeathTest, RejectsOutOfRangeIndicesAtApply)
{
    RodStub rod = make_rod();
    const std::array<bool, 3> all{true, true, true};
    const GeneralConstraint constraint(
        {99}, stack_of({{1.0, 2.0, 3.0}}), {}, all, all);

    EXPECT_ASSERT_FAILURE(constraint.constrain_values(rod, 0.0));
}

// ---------------------------------------------------------------------------
// FixedConstraint
// ---------------------------------------------------------------------------

FixedConstraint make_fixed()
{
    Matrix3DStack directors{
        Eigen::Matrix3d::Identity(),
        Eigen::AngleAxisd(0.8, Eigen::Vector3d::UnitZ()).toRotationMatrix()};
    return FixedConstraint(
        {0, -1},
        stack_of({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}}),
        {0, -1},
        directors);
}

TEST(FixedConstraint, PinsPositionsAndDirectors)
{
    RodStub rod = make_rod();
    const FixedConstraint constraint = make_fixed();

    constraint.constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0), Eigen::RowVector3d(10, 20, 30)));
    EXPECT_TRUE(Near(rod.m_positions.row(4), Eigen::RowVector3d(40, 50, 60)));
    EXPECT_TRUE(Near(rod.m_directors[0], constraint.fixed_directors()[0]));
    EXPECT_TRUE(Near(rod.m_directors[3], constraint.fixed_directors()[1]));
}

TEST(FixedConstraint, ZeroesConstrainedRates)
{
    RodStub rod = make_rod();
    make_fixed().constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_velocities.row(4), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(3), Eigen::RowVector3d::Zero()));
}

TEST(FixedConstraint, LeavesUnconstrainedEntriesAlone)
{
    RodStub rod = make_rod();
    const RodStub original = rod;

    make_fixed().constrain_values(rod, 0.0);
    make_fixed().constrain_rates(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.middleRows(1, 3),
                     original.m_positions.middleRows(1, 3)));
    EXPECT_TRUE(Near(rod.m_velocities.middleRows(1, 3),
                     original.m_velocities.middleRows(1, 3)));
    for (std::size_t i = 1; i < 3; ++i)
    {
        EXPECT_TRUE(Near(rod.m_directors[i], original.m_directors[i]));
    }
}

// Rates match a fully-selecting GeneralConstraint; values do not, because only
// FixedConstraint pins the directors themselves.
TEST(FixedConstraint, RatesAgreeWithFullyConstrainedGeneral)
{
    RodStub fixed_rod = make_rod();
    RodStub general_rod = make_rod();

    make_fixed().constrain_rates(fixed_rod, 0.0);
    make_general().constrain_rates(general_rod, 0.0);

    EXPECT_TRUE(Near(fixed_rod.m_velocities, general_rod.m_velocities));
    EXPECT_TRUE(Near(fixed_rod.m_angular_velocities,
                     general_rod.m_angular_velocities, 1e-13));
}

TEST(FixedConstraint, PositionsOnlyLeavesDirectorsAlone)
{
    RodStub rod = make_rod();
    const RodStub original = rod;

    const FixedConstraint constraint(
        {2}, stack_of({{7.0, 7.0, 7.0}}), {}, Matrix3DStack{});
    constraint.constrain_values(rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(2), Eigen::RowVector3d(7, 7, 7)));
    for (std::size_t i = 0; i < rod.m_directors.size(); ++i)
    {
        EXPECT_TRUE(Near(rod.m_directors[i], original.m_directors[i]));
    }
}

TEST(FixedConstraint, IsIdempotent)
{
    RodStub once = make_rod();
    RodStub twice = make_rod();

    make_fixed().constrain_values(once, 0.0);
    make_fixed().constrain_values(twice, 0.0);
    make_fixed().constrain_values(twice, 0.0);

    EXPECT_TRUE(Near(once.m_positions, twice.m_positions));
}

TEST(FixedConstraintDeathTest, RejectsMismatchedCounts)
{
    Matrix3DStack one{Eigen::Matrix3d::Identity()};
    EXPECT_ASSERT_FAILURE(FixedConstraint(
        {0, -1}, stack_of({{1.0, 2.0, 3.0}}), {}, Matrix3DStack{}));
    EXPECT_ASSERT_FAILURE(FixedConstraint(
        {}, Vector3DStack::Zero(0, 3), {0, -1}, one));
}

TEST(FixedConstraintDeathTest, RejectsConstraintThatDoesNothing)
{
    EXPECT_ASSERT_FAILURE(FixedConstraint(
        {}, Vector3DStack::Zero(0, 3), {}, Matrix3DStack{}));
}

TEST(FixedConstraintDeathTest, RejectsNonFiniteTargets)
{
    Matrix3DStack bad{Eigen::Matrix3d::Constant(kNaN)};
    EXPECT_ASSERT_FAILURE(FixedConstraint(
        {}, Vector3DStack::Zero(0, 3), {0}, bad));
}

// ---------------------------------------------------------------------------
// HelicalBucklingBoundaryCondition
// ---------------------------------------------------------------------------

HelicalBucklingBoundaryCondition make_helical(double twisting_time)
{
    return HelicalBucklingBoundaryCondition(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 10.0),
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix(),
        Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY()).toRotationMatrix(),
        twisting_time,
        /*slack=*/1.0,
        /*number_of_rotations=*/3.0);
}

HelicalBucklingBoundaryCondition make_helical() { return make_helical(2.0); }

TEST(HelicalBucklingBoundaryCondition, FinalPositionsMoveInwardByHalfSlack)
{
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    EXPECT_TRUE(Near(constraint.final_start_position(),
                     Eigen::Vector3d(0.0, 0.0, 0.5)));
    EXPECT_TRUE(Near(constraint.final_end_position(),
                     Eigen::Vector3d(0.0, 0.0, 9.5)));
}

TEST(HelicalBucklingBoundaryCondition, VelocitiesAreHalfTheTotalRates)
{
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    // Each end supplies half of 3 turns over 2 seconds, along +z.
    const double expected_angular = (2.0 * 3.0 * std::numbers::pi / 2.0) / 2.0;
    const double expected_shrink = 1.0 / (2.0 * 2.0);

    EXPECT_TRUE(Near(constraint.angular_velocity(),
                     Eigen::Vector3d(0.0, 0.0, expected_angular)));
    EXPECT_TRUE(Near(constraint.shrink_velocity(),
                     Eigen::Vector3d(0.0, 0.0, expected_shrink)));
}

TEST(HelicalBucklingBoundaryCondition, FinalDirectorsRotateByHalfTheTotalTwist)
{
    const HelicalBucklingBoundaryCondition constraint = make_helical();
    const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    const double theta = 3.0 * std::numbers::pi;

    const Eigen::Matrix3d start =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Matrix3d end =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY()).toRotationMatrix();

    EXPECT_TRUE(Near(constraint.final_start_directors(),
                     rotation_matrix(theta, axis) * start));
    EXPECT_TRUE(Near(constraint.final_end_directors(),
                     rotation_matrix(-theta, axis) * end));
}

TEST(HelicalBucklingBoundaryCondition, DriveEndsOppositelyDuringTwisting)
{
    RodStub rod = make_rod();
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    constraint.constrain_rates(rod, 1.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0),
                     constraint.shrink_velocity().transpose()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(0),
                     constraint.angular_velocity().transpose()));
    EXPECT_TRUE(Near(rod.m_velocities.row(4),
                     (-constraint.shrink_velocity()).transpose()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(3),
                     (-constraint.angular_velocity()).transpose()));
}

// The boundary is closed: at exactly twisting_time the ends are still driven.
TEST(HelicalBucklingBoundaryCondition, StillDrivingAtExactlyTwistingTime)
{
    RodStub rod = make_rod();
    const HelicalBucklingBoundaryCondition constraint = make_helical(2.0);

    constraint.constrain_rates(rod, 2.0);

    EXPECT_TRUE(Near(rod.m_velocities.row(0),
                     constraint.shrink_velocity().transpose()));
}

TEST(HelicalBucklingBoundaryCondition, HoldsEndsStillAfterTwisting)
{
    RodStub rod = make_rod();
    const HelicalBucklingBoundaryCondition constraint = make_helical(2.0);

    constraint.constrain_rates(rod, 2.5);

    EXPECT_TRUE(Near(rod.m_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_velocities.row(4), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(0), Eigen::RowVector3d::Zero()));
    EXPECT_TRUE(Near(rod.m_angular_velocities.row(3), Eigen::RowVector3d::Zero()));
}

TEST(HelicalBucklingBoundaryCondition, ValuesAreFreeDuringTwisting)
{
    RodStub rod = make_rod();
    const Vector3DStack original = rod.m_positions;

    make_helical().constrain_values(rod, 1.0);

    EXPECT_TRUE(Near(rod.m_positions, original));
}

TEST(HelicalBucklingBoundaryCondition, PinsEndsAfterTwisting)
{
    RodStub rod = make_rod();
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    constraint.constrain_values(rod, 3.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0),
                     constraint.final_start_position().transpose()));
    EXPECT_TRUE(Near(rod.m_positions.row(4),
                     constraint.final_end_position().transpose()));
    EXPECT_TRUE(Near(rod.m_directors.front(), constraint.final_start_directors()));
    EXPECT_TRUE(Near(rod.m_directors.back(), constraint.final_end_directors()));
}

TEST(HelicalBucklingBoundaryCondition, LeavesInteriorAlone)
{
    RodStub rod = make_rod();
    const RodStub original = rod;
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    constraint.constrain_values(rod, 3.0);
    constraint.constrain_rates(rod, 3.0);

    EXPECT_TRUE(Near(rod.m_positions.middleRows(1, 3),
                     original.m_positions.middleRows(1, 3)));
    EXPECT_TRUE(Near(rod.m_velocities.middleRows(1, 3),
                     original.m_velocities.middleRows(1, 3)));
    EXPECT_TRUE(Near(rod.m_angular_velocities.middleRows(1, 2),
                     original.m_angular_velocities.middleRows(1, 2)));
}

TEST(HelicalBucklingBoundaryCondition, ZeroSlackLeavesEndPositionsInPlace)
{
    const HelicalBucklingBoundaryCondition constraint(
        Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 10),
        Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
        2.0, /*slack=*/0.0, 3.0);

    EXPECT_TRUE(Near(constraint.final_start_position(), Eigen::Vector3d(0, 0, 0)));
    EXPECT_TRUE(Near(constraint.final_end_position(), Eigen::Vector3d(0, 0, 10)));
    EXPECT_TRUE(Near(constraint.shrink_velocity(), Eigen::Vector3d::Zero()));
}

TEST(HelicalBucklingBoundaryCondition, ExposesConstructorArguments)
{
    const HelicalBucklingBoundaryCondition constraint = make_helical(2.0);

    EXPECT_DOUBLE_EQ(constraint.twisting_time(), 2.0);
    EXPECT_DOUBLE_EQ(constraint.slack(), 1.0);
    EXPECT_DOUBLE_EQ(constraint.number_of_rotations(), 3.0);
}

TEST(HelicalBucklingBoundaryConditionDeathTest, RejectsBadParameters)
{
    const Eigen::Vector3d start(0, 0, 0);
    const Eigen::Vector3d end(0, 0, 10);
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    EXPECT_ASSERT_FAILURE(HelicalBucklingBoundaryCondition(
        start, end, identity, identity, 0.0, 1.0, 3.0));
    EXPECT_ASSERT_FAILURE(HelicalBucklingBoundaryCondition(
        start, end, identity, identity, -1.0, 1.0, 3.0));
    EXPECT_ASSERT_FAILURE(HelicalBucklingBoundaryCondition(
        start, end, identity, identity, 2.0, kNaN, 3.0));
    EXPECT_ASSERT_FAILURE(HelicalBucklingBoundaryCondition(
        start, end, identity, identity, 2.0, 1.0, kInf));
}

TEST(HelicalBucklingBoundaryConditionDeathTest, RejectsCoincidentEnds)
{
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    EXPECT_ASSERT_FAILURE(HelicalBucklingBoundaryCondition(
        Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(1, 2, 3),
        identity, identity, 2.0, 1.0, 3.0));
}

TEST(HelicalBucklingBoundaryConditionDeathTest, RejectsNonFiniteTime)
{
    RodStub rod = make_rod();
    const HelicalBucklingBoundaryCondition constraint = make_helical();

    EXPECT_ASSERT_FAILURE(constraint.constrain_values(rod, kNaN));
    EXPECT_ASSERT_FAILURE(constraint.constrain_rates(rod, kInf));
}

// ---------------------------------------------------------------------------
// ConstraintVariant dispatch
// ---------------------------------------------------------------------------

ConstraintVariant free_variant() { return ConstraintVariant{FreeBoundaryCondition{}}; }
ConstraintVariant one_end_variant()
{
    return ConstraintVariant{OneEndFixedBoundaryCondition(
        Eigen::Vector3d(9, 8, 7), Eigen::Matrix3d::Identity())};
}
ConstraintVariant general_variant() { return ConstraintVariant{make_general()}; }
ConstraintVariant fixed_variant() { return ConstraintVariant{make_fixed()}; }
ConstraintVariant helical_variant() { return ConstraintVariant{make_helical()}; }

std::vector<ConstraintVariant> all_constraints()
{
    return {free_variant(), one_end_variant(), general_variant(),
            fixed_variant(), helical_variant()};
}

TEST(ConstraintVariant, IsCopyableAndAssignable)
{
    static_assert(std::is_copy_constructible_v<ConstraintVariant>);
    static_assert(std::is_copy_assignable_v<ConstraintVariant>);
    static_assert(std::is_move_assignable_v<ConstraintVariant>);

    ConstraintVariant a = free_variant();
    const ConstraintVariant b = fixed_variant();

    a = b;

    EXPECT_TRUE(std::holds_alternative<FixedConstraint>(a));
}

TEST(ConstraintVariant, WorksInStandardContainers)
{
    std::vector<ConstraintVariant> constraints = all_constraints();

    ASSERT_EQ(constraints.size(), 5u);
    constraints.erase(constraints.begin());

    EXPECT_EQ(constraints.size(), 4u);
    EXPECT_TRUE(
        std::holds_alternative<OneEndFixedBoundaryCondition>(constraints.front()));
}

TEST(ValidateConstraint, AcceptsEveryAlternativeOnAFullRod)
{
    RodStub rod = make_rod();
    for (auto& constraint : all_constraints())
    {
        EXPECT_NO_THROW({ validate(constraint, rod); });
    }
}

// Only FreeBoundaryCondition is unconstrained enough for a position-only body.
TEST(ValidateConstraint, AcceptsFreeOnPositionOnlySystem)
{
    PositionOnlySystem sys;
    ConstraintVariant constraint = free_variant();

    EXPECT_NO_THROW({ validate(constraint, sys); });
}

TEST(ValidateConstraintDeathTest, RejectsDirectorConstraintsOnPositionOnlySystem)
{
    PositionOnlySystem sys;
    ConstraintVariant one_end = one_end_variant();
    ConstraintVariant general = general_variant();
    ConstraintVariant fixed = fixed_variant();
    ConstraintVariant helical = helical_variant();

    EXPECT_ASSERT_FAILURE(validate(one_end, sys));
    EXPECT_ASSERT_FAILURE(validate(general, sys));
    EXPECT_ASSERT_FAILURE(validate(fixed, sys));
    EXPECT_ASSERT_FAILURE(validate(helical, sys));
}

TEST(ConstrainValuesVariant, MatchesDirectOneEndFixedCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();

    ConstraintVariant constraint = one_end_variant();
    constrain_values(constraint, through_variant, 0.0);
    OneEndFixedBoundaryCondition(Eigen::Vector3d(9, 8, 7), Eigen::Matrix3d::Identity())
        .constrain_values(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_positions, direct.m_positions));
    EXPECT_TRUE(Near(through_variant.m_directors[0], direct.m_directors[0]));
}

TEST(ConstrainRatesVariant, MatchesDirectGeneralCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();

    ConstraintVariant constraint = general_variant();
    constrain_rates(constraint, through_variant, 0.0);
    make_general().constrain_rates(direct, 0.0);

    EXPECT_TRUE(Near(through_variant.m_velocities, direct.m_velocities));
    EXPECT_TRUE(Near(through_variant.m_angular_velocities,
                     direct.m_angular_velocities));
}

TEST(ConstrainValuesVariant, MatchesDirectHelicalCall)
{
    RodStub through_variant = make_rod();
    RodStub direct = make_rod();

    ConstraintVariant constraint = helical_variant();
    constrain_values(constraint, through_variant, 5.0);
    make_helical().constrain_values(direct, 5.0);

    EXPECT_TRUE(Near(through_variant.m_positions, direct.m_positions));
    EXPECT_TRUE(Near(through_variant.m_directors.back(), direct.m_directors.back()));
}

TEST(ConstrainVariantDeathTest, RejectsIncompatibleSystem)
{
    PositionOnlySystem sys;
    ConstraintVariant one_end = one_end_variant();

    EXPECT_ASSERT_FAILURE(constrain_values(one_end, sys, 0.0));
    EXPECT_ASSERT_FAILURE(constrain_rates(one_end, sys, 0.0));
}

// A constraint list applied in sequence composes each contribution.
TEST(ConstrainVariant, ComposesAcrossAList)
{
    RodStub rod = make_rod();
    RodStub expected = make_rod();

    std::vector<ConstraintVariant> constraints{one_end_variant(), general_variant()};

    for (auto& constraint : constraints) validate(constraint, rod);
    for (auto& constraint : constraints) constrain_values(constraint, rod, 0.0);
    for (auto& constraint : constraints) constrain_rates(constraint, rod, 0.0);

    OneEndFixedBoundaryCondition direct_one_end(
        Eigen::Vector3d(9, 8, 7), Eigen::Matrix3d::Identity());
    direct_one_end.constrain_values(expected, 0.0);
    make_general().constrain_values(expected, 0.0);
    direct_one_end.constrain_rates(expected, 0.0);
    make_general().constrain_rates(expected, 0.0);

    EXPECT_TRUE(Near(rod.m_positions, expected.m_positions));
    EXPECT_TRUE(Near(rod.m_velocities, expected.m_velocities));
    EXPECT_TRUE(Near(rod.m_angular_velocities, expected.m_angular_velocities));
}

// Later constraints win where two constraints target the same node.
TEST(ConstrainVariant, LastConstraintWinsOnOverlap)
{
    RodStub rod = make_rod();

    ConstraintVariant one_end = one_end_variant();
    ConstraintVariant general = general_variant();

    constrain_values(one_end, rod, 0.0);
    constrain_values(general, rod, 0.0);

    EXPECT_TRUE(Near(rod.m_positions.row(0), Eigen::RowVector3d(10, 20, 30)));
}

}  // namespace
}  // namespace cosserat::physics
