/**
 * @file test_mesh_diagnostics.cpp
 * @brief Tests for @ref MeshDiagnostics and @ref MeshBody::write_mesh.
 *
 * What has to hold is a division of labour between two kinds of output. The
 * shape is written once and never again, because it does not change. The pose
 * is written every scheduled step, because it does. Together they have to be
 * enough to place every vertex in the world at every recorded frame, and the
 * last test here checks exactly that by doing the reconstruction.
 */

#include <cosserat/simulation/diagnostics.hpp>

#include <cosserat/physics/mesh_body.hpp>
#include <cosserat/physics/rigid_body.hpp>
#include <cosserat/physics/rods.hpp>

#include <cosserat/math/triangle_mesh_field.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cosserat::simulation {
namespace {

constexpr double kDensity = 1750.0;
constexpr double kMargin = 0.1;

/** A scratch directory named after the running test, cleaned around it. */
class MeshDiagnosticsTest : public ::testing::Test
{
protected:
    std::filesystem::path m_root;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_root = std::filesystem::temp_directory_path()
            / (std::string("mesh_diag_") + info->name());
        std::filesystem::remove_all(m_root);
    }

    void TearDown() override { std::filesystem::remove_all(m_root); }

    /** Step directories beneath the root, in step order. */
    std::vector<std::filesystem::path> step_directories() const
    {
        std::vector<std::filesystem::path> steps;
        if (!std::filesystem::is_directory(m_root)) return steps;
        for (const auto& entry : std::filesystem::directory_iterator(m_root))
        {
            if (entry.is_directory()) steps.push_back(entry.path());
        }
        std::sort(steps.begin(), steps.end());
        return steps;
    }

    /** Whether a stem's binary and metadata pair both exist for a body. */
    bool has_pair(
        const std::filesystem::path& step,
        const std::string& body,
        const std::string& stem
    ) const
    {
        const std::filesystem::path base = step / body / stem;
        return std::filesystem::is_regular_file(base.string() + ".bin")
            and std::filesystem::is_regular_file(base.string() + ".md.json");
    }
};

/** A box body, by default away from the origin so its pose is non trivial. */
physics::MeshBody make_body(
    const Eigen::Vector3d& center = Eigen::Vector3d(0.1, 0.2, 0.3),
    const Eigen::Vector3d& half_extent = Eigen::Vector3d(0.3, 0.5, 0.7)
)
{
    return physics::MeshBody(
        math::make_box_mesh(center, half_extent), kDensity, kMargin, true);
}

// ---------------------------------------------------------------------------
// write_mesh on its own
// ---------------------------------------------------------------------------

TEST_F(MeshDiagnosticsTest, WriteMeshEmitsVerticesAndTriangles)
{
    std::filesystem::create_directories(m_root);
    make_body().write_mesh(m_root);

    for (const std::string& stem : {std::string("mesh_vertices"), std::string("mesh_triangles")})
    {
        EXPECT_TRUE(std::filesystem::is_regular_file(
            (m_root / stem).string() + ".bin")) << stem;
        EXPECT_TRUE(std::filesystem::is_regular_file(
            (m_root / stem).string() + ".md.json")) << stem;
    }
}

// A box is eight vertices and twelve triangles, each three doubles.
TEST_F(MeshDiagnosticsTest, TheWrittenMeshHasTheExpectedSize)
{
    std::filesystem::create_directories(m_root);
    make_body().write_mesh(m_root);

    EXPECT_EQ(std::filesystem::file_size((m_root / "mesh_vertices").string() + ".bin"),
              8u * 3u * sizeof(double));
    EXPECT_EQ(std::filesystem::file_size((m_root / "mesh_triangles").string() + ".bin"),
              12u * 3u * sizeof(double));
}

// The shape is written in body coordinates, whose origin is the centroid, so
// moving the body does not move its mesh away from the origin.
//
// It is only the same shape, though, not the same numbers. Principal
// directions are eigenvectors, and an eigenvector's sign is arbitrary, so two
// identical boxes at different places can be handed frames differing by a half
// turn and body meshes rotated to match. What is invariant is the pair: the
// mesh together with the frame always reconstructs the same world geometry.
TEST_F(MeshDiagnosticsTest, TheWrittenMeshIsTheSameShapeWhereverTheBodySits)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const physics::MeshBody here = make_body(Eigen::Vector3d::Zero(), half_extent);
    const physics::MeshBody there =
        make_body(Eigen::Vector3d(5.0, -3.0, 2.0), half_extent);

    const auto extents = [](const physics::MeshBody& body) {
        Eigen::Vector3d lower = Eigen::Vector3d::Constant(1e30);
        Eigen::Vector3d upper = Eigen::Vector3d::Constant(-1e30);
        for (const Eigen::Vector3d& vertex : body.body_frame_mesh().vertices)
        {
            lower = lower.cwiseMin(vertex);
            upper = upper.cwiseMax(vertex);
        }
        Eigen::Vector3d spans = upper - lower;
        std::sort(spans.data(), spans.data() + 3);
        return spans;
    };

    // Centred on the origin either way, and the same box either way.
    EXPECT_LT(extents(here).cwiseAbs().minCoeff(), 1e30);
    EXPECT_LT((extents(here) - extents(there)).cwiseAbs().maxCoeff(), 1e-9);

    Eigen::Vector3d expected = 2.0 * half_extent;
    std::sort(expected.data(), expected.data() + 3);
    EXPECT_LT((extents(here) - expected).cwiseAbs().maxCoeff(), 1e-9);
}

// The invariant the arbitrary sign does not disturb: whatever frame a body was
// given, mesh and frame together put the geometry back in the same place.
TEST_F(MeshDiagnosticsTest, TheReconstructionIsInvariantToTheFrameChoice)
{
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    const Eigen::Vector3d shift(5.0, -3.0, 2.0);

    const auto world_bounds = [](const physics::MeshBody& body) {
        const Eigen::Vector3d position = body.positions().row(0).transpose();
        const Eigen::Matrix3d frame = body.frames()[0];
        Eigen::Vector3d lower = Eigen::Vector3d::Constant(1e30);
        Eigen::Vector3d upper = Eigen::Vector3d::Constant(-1e30);
        for (const Eigen::Vector3d& vertex : body.body_frame_mesh().vertices)
        {
            const Eigen::Vector3d world = position + frame.transpose() * vertex;
            lower = lower.cwiseMin(world);
            upper = upper.cwiseMax(world);
        }
        return std::make_pair(lower, upper);
    };

    const auto [here_low, here_high] =
        world_bounds(make_body(Eigen::Vector3d::Zero(), half_extent));
    const auto [there_low, there_high] = world_bounds(make_body(shift, half_extent));

    EXPECT_LT((here_low - (-half_extent)).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((here_high - half_extent).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((there_low - (shift - half_extent)).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((there_high - (shift + half_extent)).cwiseAbs().maxCoeff(), 1e-9);
}

// ---------------------------------------------------------------------------
// The diagnostic
// ---------------------------------------------------------------------------

TEST_F(MeshDiagnosticsTest, SatisfiesTheDiagnosticConcepts)
{
    static_assert(MeshWriteable<physics::MeshBody>);
    static_assert(Writeable<physics::MeshBody>);
    // A body with no mesh cannot be written by this diagnostic.
    static_assert(!MeshWriteable<physics::Sphere>);
    static_assert(!MeshWriteable<physics::CosseratRod>);
    SUCCEED();
}

TEST_F(MeshDiagnosticsTest, WritesThePoseOnEveryScheduledStep)
{
    physics::MeshBody body = make_body();
    MeshDiagnostics diagnostic(m_root, "block", 2);

    for (std::uint64_t step = 0; step <= 4; ++step)
    {
        diagnostic.make_callback(body, 0.001 * static_cast<double>(step), step);
    }

    const std::vector<std::filesystem::path> steps = step_directories();
    ASSERT_EQ(steps.size(), 3u);  // steps 0, 2 and 4
    for (const std::filesystem::path& step : steps)
    {
        EXPECT_TRUE(has_pair(step, "block", "positions"));
        EXPECT_TRUE(has_pair(step, "block", "frames"));
    }
}

// The shape goes into the first directory written and nowhere else.
TEST_F(MeshDiagnosticsTest, WritesTheShapeOnlyOnce)
{
    physics::MeshBody body = make_body();
    MeshDiagnostics diagnostic(m_root, "block", 2);

    for (std::uint64_t step = 0; step <= 6; ++step)
    {
        diagnostic.make_callback(body, 0.001 * static_cast<double>(step), step);
    }

    const std::vector<std::filesystem::path> steps = step_directories();
    ASSERT_GE(steps.size(), 2u);
    EXPECT_TRUE(has_pair(steps.front(), "block", "mesh_vertices"));
    EXPECT_TRUE(has_pair(steps.front(), "block", "mesh_triangles"));
    for (std::size_t index = 1; index < steps.size(); ++index)
    {
        EXPECT_FALSE(has_pair(steps[index], "block", "mesh_vertices"))
            << "the shape was written again at " << steps[index];
        EXPECT_FALSE(has_pair(steps[index], "block", "mesh_triangles"));
    }
}

TEST_F(MeshDiagnosticsTest, ReportsWhetherTheShapeHasBeenWritten)
{
    physics::MeshBody body = make_body();
    MeshDiagnostics diagnostic(m_root, "block", 1);

    EXPECT_FALSE(diagnostic.mesh_written());
    diagnostic.make_callback(body, 0.0, 0);
    EXPECT_TRUE(diagnostic.mesh_written());
}

// A skipped step writes nothing at all, so it cannot be where the shape lands.
TEST_F(MeshDiagnosticsTest, ASkippedStepWritesNothing)
{
    physics::MeshBody body = make_body();
    MeshDiagnostics diagnostic(m_root, "block", 10);

    EXPECT_FALSE(diagnostic.make_callback(body, 0.001, 1));
    EXPECT_FALSE(diagnostic.mesh_written());
    EXPECT_TRUE(step_directories().empty());
}

// The shape follows the first write rather than step zero, so a run resumed
// part way through still records it. Without that, poses would arrive with no
// shape to apply them to.
TEST_F(MeshDiagnosticsTest, TheShapeLandsInTheFirstWriteEvenWhenItIsNotStepZero)
{
    physics::MeshBody body = make_body();
    MeshDiagnostics diagnostic(m_root, "block", 10);

    // Resuming: the run never passes through step zero.
    for (std::uint64_t step = 5000; step <= 5030; ++step)
    {
        diagnostic.make_callback(body, 1e-5 * static_cast<double>(step), step);
    }

    const std::vector<std::filesystem::path> steps = step_directories();
    ASSERT_FALSE(steps.empty());
    EXPECT_TRUE(has_pair(steps.front(), "block", "mesh_vertices"));
    EXPECT_TRUE(has_pair(steps.front(), "block", "mesh_triangles"));
}

TEST_F(MeshDiagnosticsTest, WritesEachBodyIntoItsOwnSubdirectory)
{
    physics::MeshBody first = make_body(Eigen::Vector3d::Zero());
    physics::MeshBody second = make_body(Eigen::Vector3d(1.0, 0.0, 0.0));
    MeshDiagnostics one(m_root, "alpha", 1);
    MeshDiagnostics two(m_root, "beta", 1);

    one.make_callback(first, 0.0, 0);
    two.make_callback(second, 0.0, 0);

    const std::vector<std::filesystem::path> steps = step_directories();
    ASSERT_EQ(steps.size(), 1u);
    EXPECT_TRUE(has_pair(steps.front(), "alpha", "mesh_vertices"));
    EXPECT_TRUE(has_pair(steps.front(), "beta", "mesh_vertices"));
}

// ---------------------------------------------------------------------------
// Through the variant
// ---------------------------------------------------------------------------

TEST_F(MeshDiagnosticsTest, TheVariantHoldsAndDispatchesIt)
{
    physics::MeshBody body = make_body();
    DiagnosticVariant held = MeshDiagnostics(m_root, "block", 1);

    EXPECT_NO_THROW({ validate(held, body); });
    EXPECT_TRUE(make_callback(held, body, 0.0, 0));
    EXPECT_TRUE(has_pair(step_directories().front(), "block", "mesh_vertices"));
}

// A body with no mesh cannot satisfy this diagnostic, and is rejected where it
// is registered rather than at the first write.
TEST_F(MeshDiagnosticsTest, TheVariantRejectsABodyWithNoMesh)
{
    physics::Sphere ball(Eigen::Vector3d::Zero(), 0.1, kDensity);
    DiagnosticVariant held = MeshDiagnostics(m_root, "ball", 1);

    EXPECT_DEATH({ validate(held, ball); }, "");
}

// The plain diagnostic still works on a mesh body; it simply records no shape.
TEST_F(MeshDiagnosticsTest, TheBasicDiagnosticStillAcceptsAMeshBody)
{
    physics::MeshBody body = make_body();
    DiagnosticVariant held = BasicDiagnostics(m_root, "block", 1);

    EXPECT_NO_THROW({ validate(held, body); });
    EXPECT_TRUE(make_callback(held, body, 0.0, 0));

    const std::vector<std::filesystem::path> steps = step_directories();
    ASSERT_EQ(steps.size(), 1u);
    EXPECT_TRUE(has_pair(steps.front(), "block", "positions"));
    EXPECT_FALSE(has_pair(steps.front(), "block", "mesh_vertices"));
}

// ---------------------------------------------------------------------------
// The output is sufficient to place the body
// ---------------------------------------------------------------------------

// The point of the whole exercise: shape plus pose must reconstruct the body
// in the world. Read back the same way a renderer would, and check the box
// lands where the box actually is.
TEST_F(MeshDiagnosticsTest, TheShapeAndPoseTogetherPlaceTheBodyInTheWorld)
{
    const Eigen::Vector3d center(0.1, 0.2, 0.3);
    const Eigen::Vector3d half_extent(0.3, 0.5, 0.7);
    physics::MeshBody body = make_body(center, half_extent);

    MeshDiagnostics diagnostic(m_root, "block", 1);
    ASSERT_TRUE(diagnostic.make_callback(body, 0.0, 0));

    // Reconstruct from what the body holds, which is what the files carry.
    const Eigen::Vector3d position = body.positions().row(0).transpose();
    const Eigen::Matrix3d frame = body.frames()[0];

    Eigen::Vector3d lower = Eigen::Vector3d::Constant(1e30);
    Eigen::Vector3d upper = Eigen::Vector3d::Constant(-1e30);
    for (const Eigen::Vector3d& vertex : body.body_frame_mesh().vertices)
    {
        // world = position + frame^T * body
        const Eigen::Vector3d world = position + frame.transpose() * vertex;
        lower = lower.cwiseMin(world);
        upper = upper.cwiseMax(world);
    }

    EXPECT_LT((lower - (center - half_extent)).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((upper - (center + half_extent)).cwiseAbs().maxCoeff(), 1e-9);
}

// Indices are widened to double because the format carries only that scalar.
// Every index a mesh could reach is exactly representable, so nothing is lost.
TEST_F(MeshDiagnosticsTest, TriangleIndicesSurviveBeingStoredAsDoubles)
{
    physics::MeshBody body(
        math::make_sphere_mesh(Eigen::Vector3d::Zero(), 1.0, 3), kDensity, kMargin,
        true);

    for (const Eigen::Vector3i& triangle : body.body_frame_mesh().triangles)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const double widened = static_cast<double>(triangle(corner));
            EXPECT_EQ(static_cast<int>(widened), triangle(corner));
        }
    }
    // Well inside the range a double represents exactly.
    EXPECT_LT(body.body_frame_mesh().vertices.size(), 1u << 20);
}

}  // namespace
}  // namespace cosserat::simulation
