#include <cosserat/math/minimum_distance.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <random>

namespace cosserat::math {
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

/** Brute-force closest approach, by sampling both segments densely. */
double sampled_minimum_distance(
    const Eigen::Vector3d& x1, const Eigen::Vector3d& e1,
    const Eigen::Vector3d& x2, const Eigen::Vector3d& e2, int samples = 400)
{
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= samples; ++i)
    {
        const double t = static_cast<double>(i) / samples;
        for (int j = 0; j <= samples; ++j)
        {
            const double s = static_cast<double>(j) / samples;
            best = std::min(best, ((x2 + s * e2) - (x1 + t * e1)).norm());
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Segment to segment
// ---------------------------------------------------------------------------

TEST(MinimumDistanceSegmentSegment, ParallelOffsetSegmentsAreTheOffsetApart)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(1, 0, 0);
    const Eigen::Vector3d x2(0, 2, 0), e2(1, 0, 0);

    const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

    EXPECT_NEAR(result.distance_vector.norm(), 2.0, kTol);
    EXPECT_TRUE(Near(result.distance_vector, Eigen::Vector3d(0, 2, 0)));
}

TEST(MinimumDistanceSegmentSegment, CrossingSegmentsTouch)
{
    // Two segments crossing at the origin, offset only in z.
    const Eigen::Vector3d x1(-1, 0, 0), e1(2, 0, 0);
    const Eigen::Vector3d x2(0, -1, 0.5), e2(0, 2, 0);

    const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

    EXPECT_NEAR(result.distance_vector.norm(), 0.5, kTol);
    EXPECT_TRUE(Near(result.contact_point_one, Eigen::Vector3d(0, 0, 0)));
    EXPECT_TRUE(Near(result.contact_point_two, Eigen::Vector3d(0, 0, 0.5)));
}

// When the unconstrained minimum falls off both segments, the answer is
// between two endpoints.
TEST(MinimumDistanceSegmentSegment, DisjointSegmentsMeetAtEndpoints)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(1, 0, 0);
    const Eigen::Vector3d x2(3, 0, 0), e2(1, 0, 0);

    const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

    EXPECT_NEAR(result.distance_vector.norm(), 2.0, kTol);
    EXPECT_TRUE(Near(result.contact_point_one, Eigen::Vector3d(1, 0, 0)));
    EXPECT_TRUE(Near(result.contact_point_two, Eigen::Vector3d(3, 0, 0)));
}

TEST(MinimumDistanceSegmentSegment, IdenticalSegmentsHaveZeroDistance)
{
    const Eigen::Vector3d x1(1, 2, 3), e1(1, 1, 0);

    const auto result = minimum_distance_segment_segment(x1, e1, x1, e1);

    EXPECT_LT(result.distance_vector.norm(), kTol);
}

TEST(MinimumDistanceSegmentSegment, TheDistanceVectorRunsFromOneToTwo)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(1, 0, 0);
    const Eigen::Vector3d x2(0, 0, 3), e2(1, 0, 0);

    const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

    EXPECT_TRUE(Near(result.distance_vector,
                     Eigen::Vector3d(result.contact_point_two
                                     - result.contact_point_one)));
    EXPECT_GT(result.distance_vector(2), 0.0);
}

// Swapping the arguments must negate the distance vector and swap the points.
TEST(MinimumDistanceSegmentSegment, IsAntisymmetricUnderSwappingArguments)
{
    const Eigen::Vector3d x1(0.3, -1.0, 0.2), e1(1.0, 0.4, -0.3);
    const Eigen::Vector3d x2(0.7, 0.8, 1.1), e2(-0.5, 1.0, 0.2);

    const auto forward = minimum_distance_segment_segment(x1, e1, x2, e2);
    const auto backward = minimum_distance_segment_segment(x2, e2, x1, e1);

    EXPECT_NEAR(forward.distance_vector.norm(), backward.distance_vector.norm(), 1e-12);
    EXPECT_TRUE(Near(forward.distance_vector,
                     Eigen::Vector3d(-backward.distance_vector), 1e-12));
}

// The analytic answer must agree with brute-force sampling.
TEST(MinimumDistanceSegmentSegment, AgreesWithDenseSampling)
{
    std::mt19937 generator(3);
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);

    for (int trial = 0; trial < 40; ++trial)
    {
        const Eigen::Vector3d x1(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d e1(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d x2(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d e2(uniform(generator), uniform(generator),
                                 uniform(generator));
        if (e1.norm() < 0.1 or e2.norm() < 0.1) continue;

        const double analytic =
            minimum_distance_segment_segment(x1, e1, x2, e2).distance_vector.norm();
        const double sampled = sampled_minimum_distance(x1, e1, x2, e2, 200);

        // Sampling can only overestimate, and only by the sample spacing.
        EXPECT_LE(analytic, sampled + 1e-9) << "trial " << trial;
        EXPECT_NEAR(analytic, sampled, 5e-2) << "trial " << trial;
    }
}

// Near-parallel segments take a separate branch, since the general solution is
// singular there.
TEST(MinimumDistanceSegmentSegment, HandlesNearlyParallelSegments)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(1, 0, 0);
    const Eigen::Vector3d x2(0, 1, 0);

    for (double skew : {0.0, 1e-12, 1e-9, 1e-7})
    {
        const Eigen::Vector3d e2(1, skew, 0);
        const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

        EXPECT_NEAR(result.distance_vector.norm(), 1.0, 1e-6) << "skew " << skew;
    }
}

TEST(MinimumDistanceSegmentSegment, ContactPointsLieOnTheirSegments)
{
    std::mt19937 generator(5);
    std::uniform_real_distribution<double> uniform(-1.5, 1.5);

    for (int trial = 0; trial < 40; ++trial)
    {
        const Eigen::Vector3d x1(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d e1(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d x2(uniform(generator), uniform(generator),
                                 uniform(generator));
        const Eigen::Vector3d e2(uniform(generator), uniform(generator),
                                 uniform(generator));
        if (e1.norm() < 0.1 or e2.norm() < 0.1) continue;

        const auto result = minimum_distance_segment_segment(x1, e1, x2, e2);

        // Projecting the contact point back onto its segment must give a
        // parameter inside [0, 1].
        const double t = (result.contact_point_one - x1).dot(e1) / e1.squaredNorm();
        const double s = (result.contact_point_two - x2).dot(e2) / e2.squaredNorm();
        EXPECT_GE(t, -1e-9) << "trial " << trial;
        EXPECT_LE(t, 1.0 + 1e-9) << "trial " << trial;
        EXPECT_GE(s, -1e-9) << "trial " << trial;
        EXPECT_LE(s, 1.0 + 1e-9) << "trial " << trial;
    }
}

TEST(MinimumDistanceSegmentSegmentDeathTest, RejectsDegenerateSegments)
{
    const Eigen::Vector3d x(0, 0, 0), good(1, 0, 0), zero(0, 0, 0);

    EXPECT_ASSERT_FAILURE(minimum_distance_segment_segment(x, zero, x, good));
    EXPECT_ASSERT_FAILURE(minimum_distance_segment_segment(x, good, x, zero));
}

// ---------------------------------------------------------------------------
// Segment to point
// ---------------------------------------------------------------------------

TEST(MinimumDistanceSegmentPoint, PerpendicularFootWhenTheProjectionLands)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(2, 0, 0);
    const Eigen::Vector3d point(1, 3, 0);

    const auto result = minimum_distance_segment_point(x1, e1, point);

    EXPECT_NEAR(result.distance_vector.norm(), 3.0, kTol);
    EXPECT_TRUE(Near(result.contact_point_one, Eigen::Vector3d(1, 0, 0)));
    EXPECT_TRUE(Near(result.contact_point_two, point));
}

// Past the end of the segment, the nearest point is the endpoint.
TEST(MinimumDistanceSegmentPoint, ClampsToTheEndpoints)
{
    const Eigen::Vector3d x1(0, 0, 0), e1(2, 0, 0);

    const auto before = minimum_distance_segment_point(x1, e1, Eigen::Vector3d(-3, 0, 0));
    EXPECT_TRUE(Near(before.contact_point_one, Eigen::Vector3d(0, 0, 0)));
    EXPECT_NEAR(before.distance_vector.norm(), 3.0, kTol);

    const auto after = minimum_distance_segment_point(x1, e1, Eigen::Vector3d(5, 0, 0));
    EXPECT_TRUE(Near(after.contact_point_one, Eigen::Vector3d(2, 0, 0)));
    EXPECT_NEAR(after.distance_vector.norm(), 3.0, kTol);
}

TEST(MinimumDistanceSegmentPoint, APointOnTheSegmentHasZeroDistance)
{
    const Eigen::Vector3d x1(1, 1, 1), e1(2, 0, 0);

    const auto result = minimum_distance_segment_point(x1, e1, Eigen::Vector3d(2, 1, 1));

    EXPECT_LT(result.distance_vector.norm(), kTol);
}

TEST(MinimumDistanceSegmentPointDeathTest, RejectsADegenerateSegment)
{
    EXPECT_ASSERT_FAILURE(minimum_distance_segment_point(
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()));
}

// ---------------------------------------------------------------------------
// Bounding boxes
// ---------------------------------------------------------------------------

TEST(BoundingBox, OverlappingBoxesIntersect)
{
    const AxisAlignedBox first{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1)};
    const AxisAlignedBox second{Eigen::Vector3d(0.5, 0.5, 0.5),
                                Eigen::Vector3d(2, 2, 2)};

    EXPECT_FALSE(boxes_not_intersecting(first, second));
    EXPECT_FALSE(boxes_not_intersecting(second, first));
}

// Separation on any single axis is enough to rule the pair out.
TEST(BoundingBox, SeparationOnOneAxisIsEnough)
{
    const AxisAlignedBox first{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1)};

    for (int axis = 0; axis < 3; ++axis)
    {
        Eigen::Vector3d lower = Eigen::Vector3d::Zero();
        Eigen::Vector3d upper = Eigen::Vector3d::Ones();
        lower(axis) = 5.0;
        upper(axis) = 6.0;

        EXPECT_TRUE(boxes_not_intersecting(first, AxisAlignedBox{lower, upper}))
            << "axis " << axis;
    }
}

TEST(BoundingBox, TouchingBoxesCountAsIntersecting)
{
    const AxisAlignedBox first{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1)};
    const AxisAlignedBox second{Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(2, 1, 1)};

    EXPECT_FALSE(boxes_not_intersecting(first, second));
}

TEST(BoundingBox, RodBoxContainsEveryNodePaddedByRadiusAndLength)
{
    Vector3DStack positions(4, 3);
    positions << 0, 0, 0,
                 1, 0, 0,
                 1, 1, 0,
                 1, 1, 1;
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(3, 0.1);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(3, 1.0);

    const AxisAlignedBox box = bounding_box_rod(positions, radii, lengths);

    EXPECT_TRUE(Near(box.lower, Eigen::Vector3d::Constant(-1.1)));
    EXPECT_TRUE(Near(box.upper, Eigen::Vector3d::Constant(2.1)));
}

TEST(BoundingBox, SphereBoxIsTheRadiusOnEveryAxis)
{
    const AxisAlignedBox box =
        bounding_box_sphere(Eigen::Vector3d(1, 2, 3), 0.5);

    EXPECT_TRUE(Near(box.lower, Eigen::Vector3d(0.5, 1.5, 2.5)));
    EXPECT_TRUE(Near(box.upper, Eigen::Vector3d(1.5, 2.5, 3.5)));
}

// An axis-aligned cylinder's box is its radius across and half its length
// along.
TEST(BoundingBox, AxisAlignedCylinderBoxIsExact)
{
    const AxisAlignedBox box = bounding_box_cylinder(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 0.2, 2.0);

    EXPECT_TRUE(Near(box.lower, Eigen::Vector3d(-0.2, -0.2, -1.0)));
    EXPECT_TRUE(Near(box.upper, Eigen::Vector3d(0.2, 0.2, 1.0)));
}

TEST(BoundingBox, TiltingACylinderWidensItAcrossTheAxis)
{
    const AxisAlignedBox upright = bounding_box_cylinder(
        Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), 0.2, 2.0);
    const Eigen::Matrix3d tilted =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const AxisAlignedBox leaning =
        bounding_box_cylinder(Eigen::Vector3d::Zero(), tilted, 0.2, 2.0);

    EXPECT_GT(leaning.upper(0), upright.upper(0));
    EXPECT_LT(leaning.upper(2), upright.upper(2));
}

// The reference's cylinder box is undersized: it takes the absolute value
// after summing the rotated extents, letting opposing components cancel. The
// consequence is stark enough to pin here, because it means the broad phase
// can prune away real contacts for a tilted cylinder.
TEST(BoundingBox, TheCylinderBoxDoesNotContainTheCylinder)
{
    const Eigen::Matrix3d tilted =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const AxisAlignedBox box =
        bounding_box_cylinder(Eigen::Vector3d::Zero(), tilted, 0.2, 2.0);

    // The centre of the cylinder's own end face, half a length along the axis.
    const Eigen::Vector3d end_face = 1.0 * tilted.row(2).transpose();

    EXPECT_FALSE((end_face.array() >= box.lower.array() - 1e-12).all()
                 and (end_face.array() <= box.upper.array() + 1e-12).all());

    // A correct box takes the absolute value inside the sum and does contain
    // it. Roughly twice as wide across the tilt.
    const Eigen::Vector3d local_extents(0.2, 0.2, 1.0);
    const Eigen::Vector3d correct = tilted.transpose().cwiseAbs() * local_extents;
    EXPECT_TRUE((end_face.cwiseAbs().array() <= correct.array() + 1e-12).all());
    EXPECT_GT(correct(0), 2.0 * box.upper(0));
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

Vector3DStack straight_nodes(Eigen::Index count, const Eigen::Vector3d& start,
                             const Eigen::Vector3d& step)
{
    Vector3DStack positions(count, 3);
    for (Eigen::Index i = 0; i < count; ++i)
    {
        positions.row(i) = (start + static_cast<double>(i) * step).transpose();
    }
    return positions;
}

TEST(Pruning, FarApartRodsArePruned)
{
    const Vector3DStack near_origin =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));
    const Vector3DStack far_away =
        straight_nodes(5, Eigen::Vector3d(100, 0, 0), Eigen::Vector3d(0.1, 0, 0));
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);

    EXPECT_TRUE(prune_rod_rod(near_origin, radii, lengths, far_away, radii, lengths));
}

TEST(Pruning, OverlappingRodsAreNotPruned)
{
    const Vector3DStack first =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));
    const Vector3DStack second =
        straight_nodes(5, Eigen::Vector3d(0.0, 0.01, 0), Eigen::Vector3d(0.1, 0, 0));
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);

    EXPECT_FALSE(prune_rod_rod(first, radii, lengths, second, radii, lengths));
}

// Pruning must be conservative: it may keep a pair that cannot touch, but must
// never discard one that can.
TEST(Pruning, NeverDiscardsAnOverlappingPair)
{
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);
    const Vector3DStack fixed =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));

    for (double offset = 0.0; offset < 1.0; offset += 0.02)
    {
        const Vector3DStack moving = straight_nodes(
            5, Eigen::Vector3d(0, offset, 0), Eigen::Vector3d(0.1, 0, 0));
        const bool pruned = prune_rod_rod(fixed, radii, lengths, moving, radii, lengths);
        const bool can_touch = offset <= 0.1;  // radii sum is 0.1

        if (can_touch)
        {
            EXPECT_FALSE(pruned) << "offset " << offset;
        }
    }
}

TEST(Pruning, RodAndCylinder)
{
    const Vector3DStack positions =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);

    EXPECT_TRUE(prune_rod_cylinder(positions, radii, lengths,
                                   Eigen::Vector3d(50, 0, 0),
                                   Eigen::Matrix3d::Identity(), 0.1, 0.5));
    EXPECT_FALSE(prune_rod_cylinder(positions, radii, lengths,
                                    Eigen::Vector3d(0.2, 0.02, 0),
                                    Eigen::Matrix3d::Identity(), 0.1, 0.5));
}

TEST(Pruning, RodAndSphere)
{
    const Vector3DStack positions =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);

    EXPECT_TRUE(prune_rod_sphere(positions, radii, Eigen::Vector3d(50, 0, 0), 0.1));
    EXPECT_FALSE(
        prune_rod_sphere(positions, radii, Eigen::Vector3d(0.2, 0.05, 0), 0.1));
}

// The sphere test pads the rod by its radius alone, where the rod-to-rod test
// also adds the element length. That makes it the tightest of the three.
TEST(Pruning, TheSphereTestIsTighterThanTheRodTest)
{
    const Vector3DStack positions =
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0));
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);

    // At 0.12 away the sphere test prunes, because the rod is padded only by
    // its radius (0.05) against the sphere's 0.02. The cylinder test does not,
    // because there the rod is also padded by its element length (0.1).
    const Eigen::Vector3d center(0.2, 0.12, 0);
    EXPECT_TRUE(prune_rod_sphere(positions, radii, center, 0.02));
    EXPECT_FALSE(prune_rod_cylinder(positions, radii, lengths, center,
                                    Eigen::Matrix3d::Identity(), 0.02, 0.04));
}

TEST(PruningDeathTest, RejectsEmptyRods)
{
    const Eigen::VectorXd radii = Eigen::VectorXd::Constant(4, 0.05);
    const Eigen::VectorXd lengths = Eigen::VectorXd::Constant(4, 0.1);

    EXPECT_ASSERT_FAILURE(
        bounding_box_rod(Vector3DStack(0, 3), radii, lengths));
    EXPECT_ASSERT_FAILURE(bounding_box_rod(
        straight_nodes(5, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0, 0)),
        Eigen::VectorXd(0), Eigen::VectorXd(0)));
}

}  // namespace
}  // namespace cosserat::physics
