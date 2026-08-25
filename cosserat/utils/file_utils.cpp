/**
 * @file file_utils.cpp
 * @brief Non-template implementations for the simulation output utilities.
 */

#include <cosserat/utils/file_utils.hpp>

#include <nlohmann/json.hpp>

#include <format>
#include <fstream>
#include <ios>

namespace cosserat::utils {

namespace {

/**
 * @brief Fails unless a name is usable as a single directory component.
 *
 * @param name Candidate component.
 * @param message Message reported when the check fails.
 */
void assert_path_component(const std::string& name, const char* message)
{
    nice_assert(not name.empty(), message);
    nice_assert(
        name.find('/') == std::string::npos
            and name.find('\\') == std::string::npos,
        message
    );
}

} // End anonymous namespace

std::string timestamp_string(std::chrono::system_clock::time_point time_point)
{
    // Truncated to seconds so the stamp has a fixed width, and left in UTC so
    // that it stays sortable across a daylight-saving transition.
    const auto seconds = std::chrono::floor<std::chrono::seconds>(time_point);
    return std::format("{:%Y%m%d_%H%M%S}", seconds);
}

std::string timestamp_string()
{
    return timestamp_string(std::chrono::system_clock::now());
}

std::filesystem::path make_step_directory(
    const std::filesystem::path& base_directory,
    const std::string& simulation_name,
    const std::string& timestamp,
    const std::string& body_name,
    std::uint64_t step
)
{
    assert_path_component(
        simulation_name,
        "simulation_name must be non-empty and free of path separators"
    );
    assert_path_component(
        timestamp, "timestamp must be non-empty and free of path separators"
    );
    assert_path_component(
        body_name, "body_name must be non-empty and free of path separators"
    );

    const std::filesystem::path directory = base_directory
        / (simulation_name + "_" + timestamp) / body_name
        / ("step_" + std::to_string(step));

    std::error_code error;
    std::filesystem::create_directories(directory, error);
    nice_assert(
        not error, "Failed to create the output directory: " + error.message()
    );

    return directory;
}

namespace detail {

void write_blocks(
    const std::filesystem::path& stem,
    const std::vector<const double*>& blocks,
    std::size_t block_element_count,
    Eigen::Index rows,
    Eigen::Index cols,
    bool row_major
)
{
    // Appending to a stem that already carries an extension would produce
    // names such as velocity.bin.bin, so treat it as a mistake.
    nice_assert(
        not stem.has_extension(),
        "Output stem must not already carry an extension"
    );
    nice_assert(stem.has_filename(), "Output stem must name a file");
    nice_assert(not blocks.empty(), "Cannot write an empty batch of matrices");
    nice_assert(
        block_element_count == static_cast<std::size_t>(rows * cols),
        "Block element count must equal rows times columns"
    );

    const std::filesystem::path parent = stem.parent_path();
    if (not parent.empty())
    {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        nice_assert(
            not error,
            "Failed to create the output directory: " + error.message()
        );
    }

    std::filesystem::path binary_path = stem;
    binary_path += binary_extension;

    std::ofstream binary_stream(binary_path, std::ios::binary | std::ios::trunc);
    nice_assert(
        binary_stream.is_open(),
        "Failed to open the binary output file: " + binary_path.string()
    );

    const std::streamsize block_bytes =
        static_cast<std::streamsize>(block_element_count * sizeof(double));
    for (const double* block : blocks)
    {
        nice_assert(block != nullptr, "Encountered a null matrix data pointer");
        binary_stream.write(reinterpret_cast<const char*>(block), block_bytes);
    }

    binary_stream.flush();
    nice_assert(
        binary_stream.good(),
        "Failed while writing the binary output file: " + binary_path.string()
    );
    binary_stream.close();

    nlohmann::json metadata;
    metadata["scalar_type"] = scalar_type_name;
    metadata["rows"] = rows;
    metadata["cols"] = cols;
    metadata["batches"] = static_cast<std::uint64_t>(blocks.size());
    metadata["storage_order"] = row_major ? row_major_name : column_major_name;

    std::filesystem::path metadata_path = stem;
    metadata_path += metadata_extension;

    std::ofstream metadata_stream(metadata_path, std::ios::trunc);
    nice_assert(
        metadata_stream.is_open(),
        "Failed to open the metadata output file: " + metadata_path.string()
    );

    metadata_stream << metadata.dump(4) << "\n";
    metadata_stream.flush();
    nice_assert(
        metadata_stream.good(),
        "Failed while writing the metadata output file: " + metadata_path.string()
    );
}
} // End namespace detail
} // End namespace cosserat::utils
