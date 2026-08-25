#include <cosserat/utils/file_utils.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cosserat::utils {
namespace {

// nice_assert is assumed to abort. If it throws instead, compile with
// -DNICE_ASSERT_THROWS. If it compiles out under NDEBUG, guard these tests.
#ifdef NICE_ASSERT_THROWS
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_ANY_THROW({ stmt; })
#else
#define EXPECT_ASSERT_FAILURE(stmt) EXPECT_DEATH({ stmt; }, "")
#endif

// ---------------------------------------------------------------------------
// Fixture: every test gets its own scratch directory, removed afterwards.
// ---------------------------------------------------------------------------

class FileUtilsTest : public ::testing::Test
{
protected:
    std::filesystem::path m_directory;

    void SetUp() override
    {
        const ::testing::TestInfo* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        m_directory = std::filesystem::temp_directory_path()
            / (std::string("file_utils_") + info->test_suite_name() + "_"
               + info->name());
        std::filesystem::remove_all(m_directory);
        std::filesystem::create_directories(m_directory);
    }

    void TearDown() override { std::filesystem::remove_all(m_directory); }

    std::filesystem::path stem_for(const std::string& name) const
    {
        return m_directory / name;
    }
};

/** Reads a whole binary file back as doubles. */
std::vector<double> read_doubles(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open()) << "could not open " << path;

    const auto bytes = std::filesystem::file_size(path);
    std::vector<double> values(bytes / sizeof(double));
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(bytes));
    return values;
}

/** Reads and parses the metadata sidecar for a stem. */
nlohmann::json read_metadata(const std::filesystem::path& stem)
{
    std::filesystem::path path = stem;
    path += metadata_extension;

    std::ifstream stream(path);
    EXPECT_TRUE(stream.is_open()) << "could not open " << path;

    nlohmann::json metadata;
    stream >> metadata;
    return metadata;
}

/** Flattens a matrix in its own storage order, as the writer should. */
template<typename MatrixType>
std::vector<double> storage_order_values(const MatrixType& matrix)
{
    return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

// ---------------------------------------------------------------------------
// timestamp_string
// ---------------------------------------------------------------------------

TEST(TimestampString, FormatsTheUnixEpoch)
{
    const auto epoch = std::chrono::system_clock::time_point{};
    EXPECT_EQ(timestamp_string(epoch), "19700101_000000");
}

TEST(TimestampString, FormatsAKnownInstant)
{
    // 2024-01-15 09:30:00 UTC
    const auto instant = std::chrono::system_clock::time_point{
        std::chrono::seconds{1705311000}};
    EXPECT_EQ(timestamp_string(instant), "20240115_093000");
}

TEST(TimestampString, HasAFixedSortableShape)
{
    const std::string stamp = timestamp_string();

    ASSERT_EQ(stamp.size(), 15u);
    EXPECT_EQ(stamp[8], '_');
    for (std::size_t i = 0; i < stamp.size(); ++i)
    {
        if (i == 8) continue;
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(stamp[i])))
            << "at index " << i;
    }
}

// Truncation to whole seconds means sub-second differences collapse.
TEST(TimestampString, TruncatesToWholeSeconds)
{
    const auto instant = std::chrono::system_clock::time_point{
        std::chrono::seconds{1705311000}};
    const auto later = instant + std::chrono::milliseconds{999};

    EXPECT_EQ(timestamp_string(instant), timestamp_string(later));
}

TEST(TimestampString, IsLexicographicallyOrdered)
{
    const auto early = std::chrono::system_clock::time_point{
        std::chrono::seconds{1705311000}};
    const auto late = early + std::chrono::hours{25};

    EXPECT_LT(timestamp_string(early), timestamp_string(late));
}

TEST(TimestampString, CurrentTimeOverloadAgreesWithExplicitOne)
{
    const auto before = std::chrono::system_clock::now();
    const std::string stamp = timestamp_string();
    const auto after = std::chrono::system_clock::now();

    EXPECT_TRUE(stamp == timestamp_string(before) or stamp == timestamp_string(after))
        << "got " << stamp;
}

// ---------------------------------------------------------------------------
// make_step_directory
// ---------------------------------------------------------------------------

TEST_F(FileUtilsTest, MakeStepDirectoryBuildsTheExpectedLayout)
{
    const std::filesystem::path created =
        make_step_directory(m_directory, "snake", "20240115_093000", "rod_one", 7);

    EXPECT_EQ(created,
              m_directory / "snake_20240115_093000" / "rod_one" / "step_7");
    EXPECT_TRUE(std::filesystem::is_directory(created));
}

TEST_F(FileUtilsTest, MakeStepDirectoryCreatesEveryMissingComponent)
{
    const std::filesystem::path nested = m_directory / "a" / "b" / "c";
    const std::filesystem::path created =
        make_step_directory(nested, "run", "20240115_093000", "body", 0);

    EXPECT_TRUE(std::filesystem::is_directory(created));
    EXPECT_TRUE(std::filesystem::is_directory(nested / "run_20240115_093000"));
}

TEST_F(FileUtilsTest, MakeStepDirectoryIsIdempotent)
{
    const std::filesystem::path first =
        make_step_directory(m_directory, "run", "20240115_093000", "body", 3);
    const std::filesystem::path second =
        make_step_directory(m_directory, "run", "20240115_093000", "body", 3);

    EXPECT_EQ(first, second);
    EXPECT_TRUE(std::filesystem::is_directory(second));
}

TEST_F(FileUtilsTest, MakeStepDirectorySeparatesStepsAndBodies)
{
    const std::filesystem::path step_one =
        make_step_directory(m_directory, "run", "20240115_093000", "body", 1);
    const std::filesystem::path step_two =
        make_step_directory(m_directory, "run", "20240115_093000", "body", 2);
    const std::filesystem::path other_body =
        make_step_directory(m_directory, "run", "20240115_093000", "other", 1);

    EXPECT_NE(step_one, step_two);
    EXPECT_NE(step_one, other_body);
    EXPECT_EQ(step_one.parent_path(), step_two.parent_path());
}

TEST_F(FileUtilsTest, MakeStepDirectoryRejectsBadComponents)
{
    EXPECT_ASSERT_FAILURE(
        make_step_directory(m_directory, "", "20240115_093000", "body", 0));
    EXPECT_ASSERT_FAILURE(
        make_step_directory(m_directory, "run", "", "body", 0));
    EXPECT_ASSERT_FAILURE(
        make_step_directory(m_directory, "run", "20240115_093000", "", 0));
    EXPECT_ASSERT_FAILURE(
        make_step_directory(m_directory, "a/b", "20240115_093000", "body", 0));
    EXPECT_ASSERT_FAILURE(
        make_step_directory(m_directory, "run", "20240115_093000", "bo/dy", 0));
}

// ---------------------------------------------------------------------------
// write_matrix: single matrix
// ---------------------------------------------------------------------------

TEST_F(FileUtilsTest, WritesBothFilesForASingleMatrix)
{
    const std::filesystem::path stem = stem_for("velocity");
    Eigen::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0;

    write_matrix(stem, matrix);

    std::filesystem::path binary = stem;
    binary += binary_extension;
    std::filesystem::path metadata = stem;
    metadata += metadata_extension;

    EXPECT_TRUE(std::filesystem::is_regular_file(binary));
    EXPECT_TRUE(std::filesystem::is_regular_file(metadata));
    EXPECT_EQ(std::filesystem::file_size(binary), 6u * sizeof(double));
}

TEST_F(FileUtilsTest, BinaryPayloadIsEigensOwnStorageOrder)
{
    const std::filesystem::path stem = stem_for("state");
    Eigen::MatrixXd matrix(2, 3);
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0;

    write_matrix(stem, matrix);

    std::filesystem::path binary = stem;
    binary += binary_extension;

    // Eigen defaults to column major, so the file runs down the columns.
    EXPECT_EQ(read_doubles(binary),
              (std::vector<double>{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}));
    EXPECT_EQ(read_doubles(binary), storage_order_values(matrix));
}

TEST_F(FileUtilsTest, MetadataDescribesASingleColumnMajorMatrix)
{
    const std::filesystem::path stem = stem_for("state");
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(2, 3);

    write_matrix(stem, matrix);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("scalar_type"), scalar_type_name);
    EXPECT_EQ(metadata.at("rows"), 2);
    EXPECT_EQ(metadata.at("cols"), 3);
    EXPECT_EQ(metadata.at("batches"), 1);
    EXPECT_EQ(metadata.at("storage_order"), column_major_name);
}

TEST_F(FileUtilsTest, RowMajorMatricesAreRecordedAndWrittenAsSuch)
{
    const std::filesystem::path stem = stem_for("row_major");
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(2, 3);
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0;

    write_matrix(stem, matrix);

    std::filesystem::path binary = stem;
    binary += binary_extension;

    EXPECT_EQ(read_doubles(binary),
              (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    EXPECT_EQ(read_metadata(stem).at("storage_order"), row_major_name);
}

// Same values, different storage order: the metadata is what distinguishes
// the two files.
TEST_F(FileUtilsTest, StorageOrderChangesTheBytesNotTheShape)
{
    Eigen::MatrixXd column_major(2, 3);
    column_major << 1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major =
        column_major;

    const std::filesystem::path column_stem = stem_for("column");
    const std::filesystem::path row_stem = stem_for("row");
    write_matrix(column_stem, column_major);
    write_matrix(row_stem, row_major);

    std::filesystem::path column_binary = column_stem;
    column_binary += binary_extension;
    std::filesystem::path row_binary = row_stem;
    row_binary += binary_extension;

    EXPECT_NE(read_doubles(column_binary), read_doubles(row_binary));
    EXPECT_EQ(read_metadata(column_stem).at("rows"),
              read_metadata(row_stem).at("rows"));
    EXPECT_EQ(read_metadata(column_stem).at("cols"),
              read_metadata(row_stem).at("cols"));
}

TEST_F(FileUtilsTest, WritesFixedSizeMatrices)
{
    const std::filesystem::path stem = stem_for("frame");
    Eigen::Matrix3d matrix;
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0;

    write_matrix(stem, matrix);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 3);
    EXPECT_EQ(metadata.at("cols"), 3);
    EXPECT_EQ(metadata.at("batches"), 1);

    std::filesystem::path binary = stem;
    binary += binary_extension;
    EXPECT_EQ(read_doubles(binary), storage_order_values(matrix));
}

TEST_F(FileUtilsTest, WritesColumnVectors)
{
    const std::filesystem::path stem = stem_for("direction");
    const Eigen::Vector3d vector(1.0, 2.0, 3.0);

    write_matrix(stem, vector);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 3);
    EXPECT_EQ(metadata.at("cols"), 1);
    EXPECT_EQ(metadata.at("batches"), 1);
}

TEST_F(FileUtilsTest, WritesRowVectors)
{
    const std::filesystem::path stem = stem_for("row_direction");
    const Eigen::RowVector3d vector(1.0, 2.0, 3.0);

    write_matrix(stem, vector);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 1);
    EXPECT_EQ(metadata.at("cols"), 3);
}

TEST_F(FileUtilsTest, WritesDynamicVectors)
{
    const std::filesystem::path stem = stem_for("mass");
    const Eigen::VectorXd vector = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);

    write_matrix(stem, vector);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 5);
    EXPECT_EQ(metadata.at("cols"), 1);

    std::filesystem::path binary = stem;
    binary += binary_extension;
    EXPECT_EQ(read_doubles(binary),
              (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
}

TEST_F(FileUtilsTest, WritesSingleElementMatrices)
{
    const std::filesystem::path stem = stem_for("scalar");
    const Eigen::Matrix<double, 1, 1> matrix = Eigen::Matrix<double, 1, 1>::Constant(4.5);

    write_matrix(stem, matrix);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 1);
    EXPECT_EQ(metadata.at("cols"), 1);

    std::filesystem::path binary = stem;
    binary += binary_extension;
    EXPECT_EQ(read_doubles(binary), (std::vector<double>{4.5}));
}

// ---------------------------------------------------------------------------
// write_matrix: batches
// ---------------------------------------------------------------------------

TEST_F(FileUtilsTest, WritesBatchesBackToBackInVectorOrder)
{
    const std::filesystem::path stem = stem_for("frames");
    std::vector<Eigen::Matrix3d> batches;
    for (int i = 0; i < 3; ++i)
    {
        batches.push_back(Eigen::Matrix3d::Constant(static_cast<double>(i)));
    }

    write_matrix(stem, batches);

    std::filesystem::path binary = stem;
    binary += binary_extension;

    std::vector<double> expected;
    for (const Eigen::Matrix3d& matrix : batches)
    {
        const std::vector<double> block = storage_order_values(matrix);
        expected.insert(expected.end(), block.begin(), block.end());
    }
    EXPECT_EQ(read_doubles(binary), expected);
    EXPECT_EQ(std::filesystem::file_size(binary), 27u * sizeof(double));
}

TEST_F(FileUtilsTest, MetadataRecordsTheBatchCount)
{
    const std::filesystem::path stem = stem_for("frames");
    const std::vector<Eigen::Matrix3d> batches(4, Eigen::Matrix3d::Identity());

    write_matrix(stem, batches);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 3);
    EXPECT_EQ(metadata.at("cols"), 3);
    EXPECT_EQ(metadata.at("batches"), 4);
    EXPECT_EQ(metadata.at("storage_order"), column_major_name);
}

// A single-element batch differs from an unbatched write only in intent, so
// the two must produce identical files.
TEST_F(FileUtilsTest, SingleBatchMatchesUnbatchedWrite)
{
    Eigen::Matrix3d matrix;
    matrix << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0;

    const std::filesystem::path batched_stem = stem_for("batched");
    const std::filesystem::path plain_stem = stem_for("plain");
    write_matrix(batched_stem, std::vector<Eigen::Matrix3d>{matrix});
    write_matrix(plain_stem, matrix);

    std::filesystem::path batched_binary = batched_stem;
    batched_binary += binary_extension;
    std::filesystem::path plain_binary = plain_stem;
    plain_binary += binary_extension;

    EXPECT_EQ(read_doubles(batched_binary), read_doubles(plain_binary));
    EXPECT_EQ(read_metadata(batched_stem), read_metadata(plain_stem));
}

TEST_F(FileUtilsTest, WritesBatchesOfDynamicMatrices)
{
    const std::filesystem::path stem = stem_for("stack");
    const std::vector<Eigen::MatrixXd> batches(2, Eigen::MatrixXd::Ones(4, 3));

    write_matrix(stem, batches);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 4);
    EXPECT_EQ(metadata.at("cols"), 3);
    EXPECT_EQ(metadata.at("batches"), 2);
}

TEST_F(FileUtilsTest, WritesBatchesOfVectors)
{
    const std::filesystem::path stem = stem_for("directions");
    const std::vector<Eigen::Vector3d> batches(5, Eigen::Vector3d(1.0, 2.0, 3.0));

    write_matrix(stem, batches);
    const nlohmann::json metadata = read_metadata(stem);

    EXPECT_EQ(metadata.at("rows"), 3);
    EXPECT_EQ(metadata.at("cols"), 1);
    EXPECT_EQ(metadata.at("batches"), 5);
}

TEST_F(FileUtilsTest, RejectsAnEmptyBatch)
{
    const std::filesystem::path stem = stem_for("empty");
    const std::vector<Eigen::Matrix3d> batches;

    EXPECT_ASSERT_FAILURE(write_matrix(stem, batches));
}

// A ragged batch cannot be described by one shape, so it must not be written.
TEST_F(FileUtilsTest, RejectsRaggedBatches)
{
    const std::filesystem::path stem = stem_for("ragged");
    std::vector<Eigen::MatrixXd> batches;
    batches.push_back(Eigen::MatrixXd::Zero(2, 3));
    batches.push_back(Eigen::MatrixXd::Zero(3, 3));

    EXPECT_ASSERT_FAILURE(write_matrix(stem, batches));
}

// ---------------------------------------------------------------------------
// write_matrix: paths and failure modes
// ---------------------------------------------------------------------------

TEST_F(FileUtilsTest, CreatesMissingParentDirectories)
{
    const std::filesystem::path stem = m_directory / "deep" / "deeper" / "state";
    const Eigen::Vector3d vector(1.0, 2.0, 3.0);

    write_matrix(stem, vector);

    std::filesystem::path binary = stem;
    binary += binary_extension;
    EXPECT_TRUE(std::filesystem::is_regular_file(binary));
}

TEST_F(FileUtilsTest, WritesIntoADirectoryFromMakeStepDirectory)
{
    const std::filesystem::path directory =
        make_step_directory(m_directory, "run", "20240115_093000", "rod", 12);
    const std::filesystem::path stem = directory / "positions";

    write_matrix(stem, Eigen::MatrixXd::Ones(3, 4));

    std::filesystem::path binary = stem;
    binary += binary_extension;
    EXPECT_TRUE(std::filesystem::is_regular_file(binary));
    EXPECT_EQ(std::filesystem::file_size(binary), 12u * sizeof(double));
}

// Rewriting a stem must truncate, not append.
TEST_F(FileUtilsTest, OverwritesRatherThanAppends)
{
    const std::filesystem::path stem = stem_for("state");
    write_matrix(stem, Eigen::MatrixXd::Ones(10, 10));
    write_matrix(stem, Eigen::MatrixXd::Ones(2, 2));

    std::filesystem::path binary = stem;
    binary += binary_extension;

    EXPECT_EQ(std::filesystem::file_size(binary), 4u * sizeof(double));
    EXPECT_EQ(read_metadata(stem).at("rows"), 2);
}

// Otherwise the output would be named velocity.bin.bin.
TEST_F(FileUtilsTest, RejectsAStemThatAlreadyHasAnExtension)
{
    const Eigen::Vector3d vector(1.0, 2.0, 3.0);

    EXPECT_ASSERT_FAILURE(write_matrix(stem_for("velocity.bin"), vector));
    EXPECT_ASSERT_FAILURE(write_matrix(stem_for("velocity.md.json"), vector));
}

TEST_F(FileUtilsTest, RejectsAStemWithoutAFilename)
{
    const Eigen::Vector3d vector(1.0, 2.0, 3.0);
    std::filesystem::path directory_stem = m_directory;
    directory_stem += std::filesystem::path::preferred_separator;

    EXPECT_ASSERT_FAILURE(write_matrix(directory_stem, vector));
}

TEST_F(FileUtilsTest, RejectsMatricesWithNoRowsOrColumns)
{
    const std::filesystem::path stem = stem_for("empty");

    EXPECT_ASSERT_FAILURE(write_matrix(stem, Eigen::MatrixXd(0, 3)));
    EXPECT_ASSERT_FAILURE(write_matrix(stem, Eigen::MatrixXd(3, 0)));
}

// Plain matrices and equivalent expressions must produce identical files;
// the two overloads differ only in whether an evaluation is needed.
TEST_F(FileUtilsTest, ExpressionsAndPlainMatricesAgree)
{
    const Eigen::MatrixXd plain = Eigen::MatrixXd::Ones(3, 4) * 2.0;

    const std::filesystem::path plain_stem = stem_for("plain");
    const std::filesystem::path expression_stem = stem_for("expression");
    write_matrix(plain_stem, plain);
    write_matrix(expression_stem, Eigen::MatrixXd::Ones(3, 4) * 2.0);

    std::filesystem::path plain_binary = plain_stem;
    plain_binary += binary_extension;
    std::filesystem::path expression_binary = expression_stem;
    expression_binary += binary_extension;

    EXPECT_EQ(read_doubles(plain_binary), read_doubles(expression_binary));
    EXPECT_EQ(read_metadata(plain_stem), read_metadata(expression_stem));
}

TEST_F(FileUtilsTest, WritesArithmeticAndBlockExpressions)
{
    Eigen::MatrixXd source(3, 4);
    source << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

    const std::filesystem::path sum_stem = stem_for("sum");
    write_matrix(sum_stem, source + source);
    EXPECT_EQ(read_metadata(sum_stem).at("rows"), 3);
    EXPECT_EQ(read_metadata(sum_stem).at("cols"), 4);

    const std::filesystem::path block_stem = stem_for("block");
    write_matrix(block_stem, source.leftCols(2));
    EXPECT_EQ(read_metadata(block_stem).at("rows"), 3);
    EXPECT_EQ(read_metadata(block_stem).at("cols"), 2);

    const std::filesystem::path transpose_stem = stem_for("transposed");
    write_matrix(transpose_stem, source.transpose());
    EXPECT_EQ(read_metadata(transpose_stem).at("rows"), 4);
    EXPECT_EQ(read_metadata(transpose_stem).at("cols"), 3);
}

TEST_F(FileUtilsTest, MetadataIsValidJsonWithExactlyTheExpectedKeys)
{
    const std::filesystem::path stem = stem_for("state");
    write_matrix(stem, Eigen::Matrix3d::Identity());

    const nlohmann::json metadata = read_metadata(stem);

    ASSERT_TRUE(metadata.is_object());
    EXPECT_EQ(metadata.size(), 5u);
    for (const char* key :
         {"scalar_type", "rows", "cols", "batches", "storage_order"})
    {
        EXPECT_TRUE(metadata.contains(key)) << "missing " << key;
    }
}
}  // namespace
}  // namespace cosserat::utils
