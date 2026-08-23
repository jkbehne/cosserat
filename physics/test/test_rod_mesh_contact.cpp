/**
 * @file test_rod_mesh_contact.cpp
 * @brief Tests for @ref rod_mesh_contact.hpp.
 *
 * Mesh contact is not a port, so there is no reference implementation to diff
 * against. Everything here is instead checked against analytic fields, where
 * distances, normals and penetrations all have closed forms. A sphere and a
 * box between them exercise curved surfaces, flat faces, edges and corners.
 */

#include "physics/rod_mesh_contact.hpp"

#include "math/signed_distance_field.hpp"

#include "physics/rods.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <numbers>
#include <vector>

namespace cosserat::physics {
using namespace math;
namespace {

constexpr double kTol = 1e-12;

/**
 * @brief A minimal body satisfying @ref ContactableMeshBody.
 *
 * Holds the field by reference and everything else by value, which is enough
 * for the contact rule and keeps the tests free of a full rigid body.
 */
template<SignedDistanceField FieldType>
class TestMeshBody
{
private: // Members
    const FieldType* m_field;
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_angular_velocities;
    Matrix3DStack m_frames;
    Vector3DStack m_external_forces;
    Vector3DStack m_external_torques;

public: // Methods
    /**
     * @brief Builds a body around a field.
     * @param field The field, which must outlive the body.
     * @param center Position of the body in the lab frame.
     * @param frame Body frame; identity places field and lab coordinates alike.
     */
    TestMeshBody(
        const FieldType& field,
        const Eigen::Vector3d& center = Eigen::Vector3d::Zero(),
        const Eigen::Matrix3d& frame = Eigen::Matrix3d::Identity()
    ) : m_field(&field),
        m_positions(Vector3DStack::Zero(1, 3)),
        m_velocities(Vector3DStack::Zero(1, 3)),
        m_angular_velocities(Vector3DStack::Zero(1, 3)),
        m_frames(1, frame),
        m_external_forces(Vector3DStack::Zero(1, 3)),
        m_external_torques(Vector3DStack::Zero(1, 3))
    {
        m_positions.row(0) = center.transpose();
    }

    const Vector3DStack& positions() const {return m_positions;}
    const Vector3DStack& velocities() const {return m_velocities;}
    const Vector3DStack& angular_velocities() const {return m_angular_velocities;}
    const Matrix3DStack& frames() const {return m_frames;}
    const Vector3DStack& external_forces() const {return m_external_forces;}
    const Vector3DStack& external_torques() const {return m_external_torques;}

    FieldQuery field_query() const {return as_query(*m_field);}
    Eigen::AlignedBox3d field_domain() const {return m_field->domain();}

    Vector3DStack& mutable_velocities() {return m_velocities;}
    Vector3DStack& mutable_angular_velocities() {return m_angular_velocities;}
    Vector3DStack& mutable_external_forces() {return m_external_forces;}
    Vector3DStack& mutable_external_torques() {return m_external_torques;}
};

/** A straight rod along a direction, for driving into an obstacle. */
CosseratRod make_rod(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction,
    double length = 1.0,
    double radius = 0.05,
    std::int64_t elements = 8
)
{
    Eigen::Vector3d normal(0.0, 0.0, 1.0);
    if (std::abs(normal.dot(direction)) > 0.9) normal = Eigen::Vector3d(0.0, 1.0, 0.0);
    normal = (normal - normal.dot(direction) * direction).normalized();
    return straight_cosserat_rod(
        elements, start, direction, normal, length, radius, 1000.0, 1.0e6,
        false, 1e-12);
}

double max_force(const CosseratRod& rod)
{
    return rod.external_forces().cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------------
// The fields themselves
// ---------------------------------------------------------------------------

TEST(SignedDistanceFieldTest, TheAnalyticFieldsSatisfyTheConcept)
{
    static_assert(SignedDistanceField<AnalyticSphereField>);
    static_assert(SignedDistanceField<AnalyticBoxField>);
    SUCCEED();
}

TEST(SignedDistanceFieldTest, SphereDistanceIsExactOutsideAndIn)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 1.0 /* radius*/, 1.0 /* margin */);

    EXPECT_NEAR(field.signed_distance({2.0, 0.0, 0.0}).distance, 1.0, kTol);
    EXPECT_NEAR(field.signed_distance({1.0, 0.0, 0.0}).distance, 0.0, kTol);
    EXPECT_NEAR(field.signed_distance({0.5, 0.0, 0.0}).distance, -0.5, kTol);
    EXPECT_NEAR(field.signed_distance({0.0, 0.0, -3.0}).distance, 2.0, kTol);
}

// A distance field has unit gradient wherever it is differentiable.
TEST(SignedDistanceFieldTest, SphereGradientIsUnitAndPointsOutward)
{
    const AnalyticSphereField field(Eigen::Vector3d(1.0, 2.0, 3.0), 0.5 /* radius */, 1.0 /* margin */);

    for (const Eigen::Vector3d& probe : {
             Eigen::Vector3d(3.0, 2.0, 3.0), Eigen::Vector3d(1.0, 0.0, 3.0),
             Eigen::Vector3d(1.2, 2.1, 3.1)})
    {
        const SignedDistance sample = field.signed_distance(probe);
        EXPECT_NEAR(sample.gradient.norm(), 1.0, 1e-12);
        // Outward means along the offset from the centre.
        EXPECT_GT(sample.gradient.dot(probe - Eigen::Vector3d(1.0, 2.0, 3.0)), 0.0);
    }
}

// The centre is the medial axis: no direction is nearer than any other.
TEST(SignedDistanceFieldTest, TheSphereGradientDegeneratesAtItsCentre)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 1.0 /* radius */, 1.0 /* margin */);

    const SignedDistance sample = field.signed_distance(Eigen::Vector3d::Zero());
    EXPECT_NEAR(sample.distance, -1.0, kTol);
    EXPECT_LT(sample.gradient.norm(), mesh_gradient_tolerance);
}

TEST(SignedDistanceFieldTest, BoxDistanceIsExactOnFacesEdgesAndCorners)
{
    const AnalyticBoxField field(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 1.0 /* margin */);

    // Straight out from a face.
    EXPECT_NEAR(field.signed_distance({2.0, 0.0, 0.0}).distance, 1.0, kTol);
    // Diagonally out from an edge: two units of overhang.
    EXPECT_NEAR(field.signed_distance({2.0, 2.0, 0.0}).distance,
                std::sqrt(2.0), kTol);
    // Out from a corner: three.
    EXPECT_NEAR(field.signed_distance({2.0, 2.0, 2.0}).distance,
                std::sqrt(3.0), kTol);
    // Inside, the nearest face governs.
    EXPECT_NEAR(field.signed_distance({0.0, 0.0, 0.0}).distance, -1.0, kTol);
    EXPECT_NEAR(field.signed_distance({0.9, 0.0, 0.0}).distance, -0.1, kTol);
}

TEST(SignedDistanceFieldTest, BoxGradientIsUnitOutsideAndAxisAlignedOnAFace)
{
    const AnalyticBoxField field(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 1.0 /* margin */);

    const SignedDistance face = field.signed_distance({2.0, 0.0, 0.0});
    EXPECT_NEAR(face.gradient.norm(), 1.0, kTol);
    EXPECT_TRUE(face.gradient.isApprox(Eigen::Vector3d::UnitX()));

    const SignedDistance corner = field.signed_distance({2.0, 2.0, 2.0});
    EXPECT_NEAR(corner.gradient.norm(), 1.0, kTol);
    EXPECT_TRUE(corner.gradient.isApprox(Eigen::Vector3d::Ones().normalized()));
}

// ---------------------------------------------------------------------------
// Marching one element
// ---------------------------------------------------------------------------

TEST(MarchElementTest, ReportsNoContactWhenTheElementIsClear)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 1.0, 10.0);

    const detail::MeshContactSample sample = detail::march_element_against_field(
        as_query(field), {3.0, -1.0, 0.0}, {3.0, 1.0, 0.0}, 0.1);

    EXPECT_FALSE(sample.in_contact);
}

TEST(MarchElementTest, FindsContactAndTheExactPenetration)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 1.0, 10.0);
    const double radius = 0.2;

    // A segment passing 0.9 from the centre, so the closest approach to the
    // surface is 0.9 - 1.0 = -0.1, and the penetration is 0.2 - (-0.1) = 0.3.
    const detail::MeshContactSample sample = detail::march_element_against_field(
        as_query(field), {0.9, -1.0, 0.0}, {0.9, 1.0, 0.0}, radius);

    ASSERT_TRUE(sample.in_contact);
    EXPECT_NEAR(sample.penetration, 0.3, 1e-3);
    // Deepest at the midpoint, where the segment is nearest the centre.
    EXPECT_NEAR(sample.parameter, 0.5, 0.05);
    EXPECT_TRUE(sample.normal.isApprox(Eigen::Vector3d::UnitX(), 1e-2));
}

// Sphere tracing steps by the safe distance, so a clear element costs only a
// handful of queries however long it is.
TEST(MarchElementTest, AClearElementCostsFewQueries)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 1.0, 100.0);

    const detail::MeshContactSample sample = detail::march_element_against_field(
        as_query(field), {20.0, -5.0, 0.0}, {20.0, 5.0, 0.0}, 0.1);

    EXPECT_FALSE(sample.in_contact);
    EXPECT_LT(sample.queries, 10) << "the march is not stepping conservatively";
}

// The guarantee that matters: a thin feature between two naive samples must
// still be found. A small sphere sitting at the middle of a long element would
// be skipped entirely by endpoint-and-midpoint sampling offset slightly.
TEST(MarchElementTest, DoesNotSkipASmallFeatureMidElement)
{
    const AnalyticSphereField field(Eigen::Vector3d(0.0, 0.137, 0.0), 0.02, 50.0);

    const detail::MeshContactSample sample = detail::march_element_against_field(
        as_query(field), {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, 0.01);

    EXPECT_TRUE(sample.in_contact) << "a small feature was marched straight past";
    EXPECT_NEAR(sample.point(1), 0.137, 0.05);
}

TEST(MarchElementTest, FindsTheDeepestPointNotTheFirst)
{
    // A segment driven diagonally into a box, so penetration deepens along it.
    const AnalyticBoxField field(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 10.0);

    const detail::MeshContactSample sample = detail::march_element_against_field(
        as_query(field), {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.1);

    ASSERT_TRUE(sample.in_contact);
    // Deepest at the far end, which sits at the box centre.
    EXPECT_GT(sample.parameter, 0.9);
    EXPECT_NEAR(sample.penetration, 0.1 + 1.0, 0.1);
}

// ---------------------------------------------------------------------------
// The contact rule
// ---------------------------------------------------------------------------

TEST(RodMeshContactTest, StoresItsCoefficients)
{
    const RodMeshContact contact(1.0e4, 10.0, 5.0, 0.3);
    EXPECT_DOUBLE_EQ(contact.k(), 1.0e4);
    EXPECT_DOUBLE_EQ(contact.nu(), 10.0);
    EXPECT_DOUBLE_EQ(contact.velocity_damping_coefficient(), 5.0);
    EXPECT_DOUBLE_EQ(contact.friction_coefficient(), 0.3);
}

TEST(RodMeshContactTest, ARodClearOfTheBodyFeelsNothing)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
    TestMeshBody body(field);
    CosseratRod rod = make_rod({5.0, -0.5, 0.0}, Eigen::Vector3d::UnitY());

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    EXPECT_LT(max_force(rod), kTol);
    EXPECT_LT(body.external_forces().cwiseAbs().maxCoeff(), kTol);
}

TEST(RodMeshContactTest, AnOverlappingRodIsPushedOut)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
    TestMeshBody body(field);
    // A rod running along y, offset in x so it grazes into the sphere.
    CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    EXPECT_GT(max_force(rod), 0.0);
    // Pushed away from the sphere, so along +x.
    EXPECT_GT(rod.external_forces().col(0).sum(), 0.0);
}

// Contact is an internal interaction, so nothing is created overall.
TEST(RodMeshContactTest, TotalForceOnThePairVanishes)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
    TestMeshBody body(field);
    CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    const Eigen::RowVector3d total = rod.external_forces().colwise().sum()
        + body.external_forces().colwise().sum();
    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-9);
}

TEST(RodMeshContactTest, DeeperOverlapPushesHarder)
{
    double previous = 0.0;
    for (double offset : {0.54, 0.50, 0.46, 0.42})
    {
        const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
        TestMeshBody body(field);
        CosseratRod rod = make_rod({offset, -0.5, 0.0}, Eigen::Vector3d::UnitY());

        RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

        const double magnitude = max_force(rod);
        EXPECT_GT(magnitude, previous) << "offset " << offset;
        previous = magnitude;
    }
}

// The force is k times the penetration, split in half by the contact law and
// shared over the element's two nodes. Checked against the closed form.
TEST(RodMeshContactTest, TheForceMatchesTheAnalyticPenetration)
{
    const double sphere_radius = 0.5;
    const double rod_radius = 0.05;
    const double stiffness = 1.0e4;
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), sphere_radius, 1.0 /* margin */);
    TestMeshBody body(field);

    // A rod along y whose axis passes at x = 0.52, so its closest approach to
    // the sphere's surface is 0.52 - 0.5 = 0.02 and the overlap is
    // 0.05 - 0.02 = 0.03.
    CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY(),
                               1.0, rod_radius);
    RodMeshContact(stiffness, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    const double expected_penetration = rod_radius - (0.52 - sphere_radius);
    // Each contacting element contributes 2 * (0.5 * k * gamma) to the body.
    const double per_element = stiffness * expected_penetration;
    const double total_on_body = body.external_forces().row(0).norm();

    EXPECT_GT(total_on_body, 0.0);
    // The elements straddle the sphere at slightly different depths, so the
    // total is bounded by the deepest contributing the full amount.
    EXPECT_LT(total_on_body, per_element * double(rod.num_elements()));
}

TEST(RodMeshContactTest, DampingRespondsToApproachVelocity)
{
    const auto run = [](double approach) {
        const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
        TestMeshBody body(field);
        CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
        rod.mutable_velocities().col(0).setConstant(-approach);  // driving inward
        RodMeshContact(1.0e4, 100.0, 0.0, 0.0).apply_contact(rod, body, 0.0);
        return max_force(rod);
    };

    EXPECT_GT(run(1.0), run(0.0));
}

TEST(RodMeshContactTest, FrictionOpposesSlidingAcrossTheSurface)
{
    const auto run = [](double friction) {
        const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
        TestMeshBody body(field);
        CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
        // Sliding along its own axis, across the sphere.
        rod.mutable_velocities().col(1).setConstant(1.0);
        RodMeshContact(1.0e4, 0.0, 100.0, friction).apply_contact(rod, body, 0.0);
        return rod.external_forces().col(1).cwiseAbs().maxCoeff();
    };

    EXPECT_LT(run(0.0), run(0.5));
}

// A contact away from the body's centre has a moment arm, so the body spins.
TEST(RodMeshContactTest, AnOffCentreContactTorquesTheBody)
{
    const AnalyticBoxField field(Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d(0.5, 0.5, 0.5), 1.0 /* margin */);
    TestMeshBody body(field);
    // Pressing on the +x face, well off the mid plane in z.
    CosseratRod rod = make_rod({0.52, -0.5, 0.3}, Eigen::Vector3d::UnitY());

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    ASSERT_GT(body.external_forces().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_GT(body.external_torques().cwiseAbs().maxCoeff(), 0.0);
}

TEST(RodMeshContactTest, AContactThroughTheCentreProducesNoTorque)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
    TestMeshBody body(field);
    // Symmetric about the sphere's centre, so the moments cancel.
    CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());

    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    ASSERT_GT(body.external_forces().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_LT(body.external_torques().cwiseAbs().maxCoeff(), 1e-9);
}

// The field is in body coordinates, so moving the body must move the contact
// with it. Placing the rod and the body together at an offset must reproduce
// the result at the origin exactly.
TEST(RodMeshContactTest, ContactIsInvariantUnderTranslatingTheWholeScene)
{
    const Eigen::Vector3d shift(3.0, -2.0, 1.0);
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);

    TestMeshBody at_origin(field);
    CosseratRod rod_a = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_a, at_origin, 0.0);

    TestMeshBody moved(field, shift);
    CosseratRod rod_b = make_rod(Eigen::Vector3d(0.52, -0.5, 0.0) + shift,
                                 Eigen::Vector3d::UnitY());
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_b, moved, 0.0);

    EXPECT_LT((rod_a.external_forces() - rod_b.external_forces())
                  .cwiseAbs().maxCoeff(), 1e-9);
}

// Rotating the body rotates its field with it, because the query happens in
// body coordinates. A rod approaching a box along +x should feel the same
// push whether the box is square to the axes or turned about it.
TEST(RodMeshContactTest, TheFieldRotatesWithTheBody)
{
    const AnalyticBoxField field(Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d(0.5, 0.5, 2.0), 1.0 /* margin */);
    // Turned a quarter turn about x, which swaps the y and z extents.
    const Eigen::Matrix3d turned =
        Eigen::AngleAxisd(std::numbers::pi / 2.0, Eigen::Vector3d::UnitX())
            .toRotationMatrix();

    TestMeshBody square(field);
    TestMeshBody rotated(field, Eigen::Vector3d::Zero(), turned);

    CosseratRod rod_a = make_rod({0.52, -0.5, 1.2}, Eigen::Vector3d::UnitY());
    CosseratRod rod_b = make_rod({0.52, -0.5, 1.2}, Eigen::Vector3d::UnitY());
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_a, square, 0.0);
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_b, rotated, 0.0);

    // Square on, the box is 2 long in z so the rod at z = 1.2 is beside it and
    // in contact. Turned, that extent lies along y and the rod is clear.
    EXPECT_GT(max_force(rod_a), 0.0);
    EXPECT_LT(max_force(rod_b), kTol);
}

// A sphere's field is isotropic, so turning the body cannot change the physics
// at all. It does change the body frame the query passes through, which makes
// this the test that pins the round trip: the point goes into body
// coordinates, and the gradient has to come back out of them.
TEST(RodMeshContactTest, TurningAnIsotropicBodyChangesNothing)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);

    TestMeshBody square(field);
    CosseratRod rod_a = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_a, square, 0.0);

    for (double degrees : {30.0, 90.0, 145.0})
    {
        const Eigen::Matrix3d turned =
            Eigen::AngleAxisd(degrees * std::numbers::pi / 180.0,
                              Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
                .toRotationMatrix();
        TestMeshBody rotated(field, Eigen::Vector3d::Zero(), turned);
        CosseratRod rod_b = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
        RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod_b, rotated, 0.0);

        EXPECT_LT((rod_a.external_forces() - rod_b.external_forces())
                      .cwiseAbs().maxCoeff(), 1e-9) << "turned " << degrees;
    }
}

// The push must be along the lab-frame outward normal, whatever frame the body
// happens to be in. Leaving the gradient in body coordinates rotates the force
// by the body's orientation, which this catches directly.
TEST(RodMeshContactTest, TheForceIsAlongTheLabFrameOutwardNormal)
{
    const AnalyticSphereField field(Eigen::Vector3d::Zero(), 0.5, 1.0 /* margin */);
    const Eigen::Matrix3d turned =
        Eigen::AngleAxisd(std::numbers::pi / 2.0, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    TestMeshBody body(field, Eigen::Vector3d::Zero(), turned);

    // Approaching along +x, so the rod must be pushed along +x.
    CosseratRod rod = make_rod({0.52, -0.5, 0.0}, Eigen::Vector3d::UnitY());
    RodMeshContact(1.0e4, 0.0, 0.0, 0.0).apply_contact(rod, body, 0.0);

    const Eigen::Vector3d total =
        rod.external_forces().colwise().sum().transpose();
    ASSERT_GT(total.norm(), 0.0);
    // Not exact: the march lands on discrete parameters, so the contributing
    // elements are not perfectly symmetric about the sphere and leave a
    // residual of a few parts per million along the rod. The tolerance is well
    // inside that and nowhere near the quarter turn a frame error would give.
    EXPECT_NEAR(total.normalized().dot(Eigen::Vector3d::UnitX()), 1.0, 1e-4);
}

TEST(RodMeshContactDeathTest, RejectsBadCoefficients)
{
    EXPECT_DEATH({ RodMeshContact(0.0, 1.0, 0.0, 0.0); }, "");
    EXPECT_DEATH({ RodMeshContact(-1.0, 1.0, 0.0, 0.0); }, "");
    EXPECT_DEATH({ RodMeshContact(1.0e4, -1.0, 0.0, 0.0); }, "");
    EXPECT_DEATH({ RodMeshContact(1.0e4, 1.0, -1.0, 0.0); }, "");
    EXPECT_DEATH({ RodMeshContact(1.0e4, 1.0, 0.0, -1.0); }, "");
}

}  // namespace
}  // namespace cosserat::physics
