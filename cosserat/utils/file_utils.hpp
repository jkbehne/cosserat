#pragma once

/**
 * @file file_utils.hpp
 * @brief Writing simulation state to disk as raw binary plus JSON metadata.
 *
 * Each state matrix is written as a pair of files sharing a stem: a @c .bin
 * holding the raw @c double values exactly as Eigen stores them, and a
 * @c .md.json describing how to read them back. Splitting the two keeps the
 * binary payload free of any header, so it can be memory-mapped or read with a
 * single bulk call, while the metadata stays human-readable.
 *
 * The intended directory layout is
 *
 * @verbatim
 * base_directory/simulation_name_timestamp/body_name/step_n/
 * @endverbatim
 *
 * which @ref utils::make_step_directory builds and creates.
 *
 * Batched states, such as one rotation matrix per rod element, are written as a
 * single binary file holding the batches back to back in vector order. The
 * metadata records how many batches are present so a reader can recover the
 * outer dimension.
 *
 * @note Only @c double is supported. The scalar type is still recorded in the
 *       metadata so a reader can reject a file it was not expecting.
 */

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <cosserat/utils/assertions.hpp>

namespace cosserat::utils {

/** @brief Extension used for the raw binary payload. */
inline constexpr const char* binary_extension = ".bin";

/** @brief Extension used for the JSON metadata sidecar. */
inline constexpr const char* metadata_extension = ".md.json";

/** @brief Value recorded in the metadata for the scalar type. */
inline constexpr const char* scalar_type_name = "double";

/** @brief Value recorded in the metadata for row-major storage. */
inline constexpr const char* row_major_name = "row_major";

/** @brief Value recorded in the metadata for column-major storage. */
inline constexpr const char* column_major_name = "column_major";

/**
 * @brief Formats a time point as a compact, sortable, human-readable stamp.
 *
 * The result has the form @c YYYYMMDD_HHMMSS and is expressed in UTC, which
 * keeps directory names sortable and free of the ambiguity a daylight-saving
 * transition would introduce in local time.
 *
 * @param time_point Instant to format.
 * @return A fifteen-character stamp, for example @c 20240115_093000.
 *
 * @note Resolution is one second, so two runs started within the same second
 *       produce the same stamp.
 */
std::string timestamp_string(std::chrono::system_clock::time_point time_point);

/**
 * @brief Formats the current time as a stamp.
 *
 * Equivalent to calling @ref timestamp_string with
 * @c std::chrono::system_clock::now().
 *
 * @return A fifteen-character stamp, for example @c 20240115_093000.
 */
std::string timestamp_string();

/**
 * @brief Builds and creates the output directory for one body at one step.
 *
 * The resulting path is
 * @c base_directory/simulation_name_timestamp/body_name/step_n, and every
 * missing component is created.
 *
 * @param base_directory Root the run writes beneath; may already exist.
 * @param simulation_name Name of the run; must be non-empty and contain no
 *        path separator.
 * @param timestamp Stamp identifying the run, typically from
 *        @ref timestamp_string. Must be non-empty and contain no path
 *        separator.
 * @param body_name Name of the body being written; must be non-empty and
 *        contain no path separator.
 * @param step Step index, rendered as @c step_n.
 * @return The created directory.
 */
std::filesystem::path make_step_directory(
    const std::filesystem::path& base_directory,
    const std::string& simulation_name,
    const std::string& timestamp,
    const std::string& body_name,
    std::uint64_t step
);

namespace detail {

/**
 * @brief Writes contiguous blocks of doubles plus their metadata sidecar.
 *
 * Shared by the @ref utils::write_matrix overloads, which differ only in how
 * they collect the block pointers. Each block is written in full, in order, so
 * the binary file is the concatenation of the batches.
 *
 * @param stem Path without an extension; @c .bin and @c .md.json are appended.
 *        Missing parent directories are created.
 * @param blocks One pointer per batch, each addressing
 *        @p block_element_count contiguous doubles.
 * @param block_element_count Number of doubles in each block, that is
 *        @c rows * @c cols.
 * @param rows Row count of a single batch.
 * @param cols Column count of a single batch.
 * @param row_major True when each batch is stored row by row.
 *
 * @note Exposed because the calling templates live in this header. It is not
 *       part of the intended interface.
 */
void write_blocks(
    const std::filesystem::path& stem,
    const std::vector<const double*>& blocks,
    std::size_t block_element_count,
    Eigen::Index rows,
    Eigen::Index cols,
    bool row_major
);

} // End namespace detail

/**
 * @brief Writes a single matrix and its metadata.
 *
 * The binary payload is the matrix exactly as Eigen holds it, so a
 * column-major matrix is written column by column and a row-major one row by
 * row. The metadata records which, along with the dimensions and a batch count
 * of one.
 *
 * @tparam Rows Compile-time row count, possibly @c Eigen::Dynamic.
 * @tparam Cols Compile-time column count, possibly @c Eigen::Dynamic.
 * @tparam Options Eigen storage options, which carry the storage order.
 * @tparam MaxRows Eigen maximum row count.
 * @tparam MaxCols Eigen maximum column count.
 * @param stem Path without an extension; @c .bin and @c .md.json are appended.
 * @param matrix Matrix to write; must have at least one row and one column.
 */
template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
void write_matrix(
    const std::filesystem::path& stem,
    const Eigen::Matrix<double, Rows, Cols, Options, MaxRows, MaxCols>& matrix
)
{
    using MatrixType = Eigen::Matrix<double, Rows, Cols, Options, MaxRows, MaxCols>;

    nice_assert(matrix.rows() > 0, "Cannot write a matrix with no rows");
    nice_assert(matrix.cols() > 0, "Cannot write a matrix with no columns");

    detail::write_blocks(
        stem,
        std::vector<const double*>{matrix.data()},
        static_cast<std::size_t>(matrix.size()),
        matrix.rows(),
        matrix.cols(),
        static_cast<bool>(MatrixType::IsRowMajor)
    );
}

/**
 * @brief Writes the result of a matrix expression and its metadata.
 *
 * Evaluates the expression into a plain matrix and writes that, so callers can
 * pass things like @c Eigen::MatrixXd::Ones(3, 4) or @c a + b directly. The
 * evaluated type keeps the expression's storage order.
 *
 * Plain matrices bind to the non-evaluating overload instead, so passing a
 * stored state matrix does not copy it.
 *
 * @tparam Derived Eigen expression type; its scalar must be @c double.
 * @param stem Path without an extension; @c .bin and @c .md.json are appended.
 * @param expression Expression to evaluate and write; the result must have at
 *        least one row and one column.
 */
template<typename Derived>
    requires std::same_as<typename Derived::Scalar, double>
void write_matrix(
    const std::filesystem::path& stem, const Eigen::MatrixBase<Derived>& expression
)
{
    using PlainType = typename Derived::PlainObject;

    const PlainType evaluated = expression.derived();
    nice_assert(evaluated.rows() > 0, "Cannot write a matrix with no rows");
    nice_assert(evaluated.cols() > 0, "Cannot write a matrix with no columns");

    detail::write_blocks(
        stem,
        std::vector<const double*>{evaluated.data()},
        static_cast<std::size_t>(evaluated.size()),
        evaluated.rows(),
        evaluated.cols(),
        static_cast<bool>(PlainType::IsRowMajor)
    );
}

/**
 * @brief Writes a batch of equally shaped matrices and their metadata.
 *
 * The batches are written back to back in vector order, each in Eigen's own
 * storage order, so the binary file is their concatenation. The metadata
 * records the shape of one batch together with how many there are.
 *
 * @tparam Rows Compile-time row count, possibly @c Eigen::Dynamic.
 * @tparam Cols Compile-time column count, possibly @c Eigen::Dynamic.
 * @tparam Options Eigen storage options, which carry the storage order.
 * @tparam MaxRows Eigen maximum row count.
 * @tparam MaxCols Eigen maximum column count.
 * @param stem Path without an extension; @c .bin and @c .md.json are appended.
 * @param matrices Batches to write. Must be non-empty, and every batch must
 *        share the dimensions of the first.
 */
template<int Rows, int Cols, int Options, int MaxRows, int MaxCols>
void write_matrix(
    const std::filesystem::path& stem,
    const std::vector<Eigen::Matrix<double, Rows, Cols, Options, MaxRows, MaxCols>>&
        matrices
)
{
    using MatrixType = Eigen::Matrix<double, Rows, Cols, Options, MaxRows, MaxCols>;

    nice_assert(not matrices.empty(), "Cannot write an empty batch of matrices");

    const Eigen::Index rows = matrices.front().rows();
    const Eigen::Index cols = matrices.front().cols();
    nice_assert(rows > 0, "Cannot write matrices with no rows");
    nice_assert(cols > 0, "Cannot write matrices with no columns");

    std::vector<const double*> blocks;
    blocks.reserve(matrices.size());
    for (const MatrixType& matrix : matrices)
    {
        // A ragged batch cannot be described by a single shape, and would be
        // silently unreadable.
        nice_assert(
            matrix.rows() == rows and matrix.cols() == cols,
            "Every matrix in a batch must share the dimensions of the first"
        );
        blocks.push_back(matrix.data());
    }

    detail::write_blocks(
        stem,
        blocks,
        static_cast<std::size_t>(rows * cols),
        rows,
        cols,
        static_cast<bool>(MatrixType::IsRowMajor)
    );
}
} // End namespace cosserat::utils
