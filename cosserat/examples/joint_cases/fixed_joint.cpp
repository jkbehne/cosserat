/**
 * @file fixed_joint.cpp
 * @brief Two rods rigidly joined end to end, driven by a sinusoidal tip load.
 *
 * A port of PyElastica's @c examples/JointCases/fixed_joint.py, and the
 * smallest scenario that exercises the whole library end to end: two bodies, a
 * boundary condition, a joint, an external forcing rule, damping, diagnostics
 * and the time stepper.
 *
 * @section scenario The scenario
 *
 * Two identical straight rods are laid end to end along @c +z. The first
 * starts at the origin; the second starts where the first finishes, one base
 * length further along. Each is a uniform circular rod of the given length,
 * radius, density and Young's modulus, discretised into @c num_elements
 * elements. The shear modulus is left to the library's default of
 * @f$ E/3 @f$, which corresponds to a Poisson ratio of one half.
 *
 * @verbatim
 *        z
 *        ^
 *        |    tip of rod 2, sinusoidal load applied here
 *        |   ___
 *        |  |   |  rod 2   (free)
 *        |  |___|
 *        |  =====  fixed joint: tip of rod 1 to base of rod 2
 *        |  |   |
 *        |  |   |  rod 1   (clamped at its base)
 *        |  |___|
 *        +--#####------------------> x
 *           clamped base
 * @endverbatim
 *
 * Three rules act on the pair:
 *
 * - The **base of rod 1 is clamped**. A @c OneEndFixedBoundaryCondition pins
 *   node zero's position and element zero's orientation to whatever they were
 *   at construction, and zeroes their rates every step. Without it the whole
 *   assembly would simply drift.
 * - The two rods are **rigidly joined**. A @c FixedJoint links the tip of rod
 *   one to the base of rod two, resisting both separation and relative
 *   rotation with a spring and damper on each. Its rest rotation is the
 *   identity, so the joint tries to keep the two rods collinear rather than at
 *   some fixed angle.
 * - The **tip of rod 2 is driven**. An @c EndpointForceSinusoidal ramps a load
 *   in after @c force_onset_time and then rotates it in the plane spanned by
 *   the normal and the rod's tangent, which sets the free end swinging.
 *
 * Both rods are damped so the response settles rather than ringing forever.
 *
 * @section indices How the joint addresses the rods
 *
 * A joint pulls on a node but twists an element, and for a rod those are
 * different domains. The connection is therefore given an element index and a
 * flag saying which end of that element to attach to, and the node follows:
 * element @c i spans nodes @c i and @c i+1.
 *
 * Here rod one is joined at element @c -1, meaning its last, at that element's
 * top, which is its final node; rod two at element @c 0, at that element's
 * bottom, which is its first node. That is the tip of one meeting the base of
 * the other.
 *
 * @section output What comes out
 *
 * Every @c diagnostic_steps steps each rod writes its positions and frames
 * beneath @c diagnostic_base_path, in a directory named for the step and the
 * simulation time, with one subdirectory per rod. The step counter is zero
 * padded so the directories sort into step order. A frame is also written
 * before the first step, so the initial configuration is on disk.
 *
 * Logging goes to the console by default. Pass @c --log_directory to mirror it
 * to a file as well.
 *
 * @section running Running it
 *
 * Every parameter has a default, so the example runs with no arguments at all:
 *
 * @verbatim
 *   ./fixed_joint
 *   ./fixed_joint --end_time 2.0 --dt 5e-5 --log_level debug
 *   ./fixed_joint --diagnostic_base_path ./out --diagnostic_steps 500
 *   ./fixed_joint --help
 * @endverbatim
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <Eigen/Dense>

#include <cosserat/physics/constraints.hpp>
#include <cosserat/physics/damping.hpp>
#include <cosserat/physics/forces.hpp>

#include <cosserat/simulation/diagnostics.hpp>
#include <cosserat/simulation/logging.hpp>
#include <cosserat/simulation/simulation_graph.hpp>
#include <cosserat/simulation/solver.hpp>

#include <cosserat/utils/assertions.hpp>
#include <cosserat/utils/file_utils.hpp>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

using namespace cosserat;
using namespace cosserat::physics;
using SolverType = simulation::Solver<simulation::SimulationGraph>;

constexpr int DEFAULT_NUM_ELEMENTS = 10;

constexpr double DEFAULT_BASE_LENGTH = 0.2;
constexpr double DEFAULT_BASE_RADIUS = 0.007;
constexpr double DEFAULT_DENSITY = 1750.0;
constexpr double DEFAULT_YOUNGS_MODULUS = 3e7;
constexpr double DEFAULT_START_FORCE_MAG = 0.0;
constexpr double DEFAULT_END_FORCE_MAG = 5e-3;
constexpr double DEFAULT_FORCE_ONSET_TIME = 0.2;
constexpr double DEFAULT_JOINT_STIFFNESS = 1e5;
constexpr double DEFAULT_JOINT_DAMP = 0.0;
constexpr double DEFAULT_JOINT_ROT_STIFFNESS = 1e1;
constexpr double DEFAULT_JOINT_ROT_DAMP = 0.0;
constexpr double DEFAULT_DAMP_CONSTANT = 0.4;
constexpr double DEFAULT_DT = 1e-4;
constexpr double DEFAULT_START_TIME = 0.0;
constexpr double DEFAULT_END_TIME = 10.0;

constexpr int DEFAULT_DIAGNOSTIC_STEPS = 1000;
const std::string DEFAULT_DIAGNOSTIC_BASE_PATH = "/tmp/cosserat_logs/joint_cases/fixed_joint";

const std::string DEFAULT_LOG_LEVEL = "info";
const std::string DEFAULT_LOG_DIR = "";
const std::string DEFAULT_LOG_NAME = "";

constexpr double tolerance = 1e-12;

using ParseReturnType = std::pair<cxxopts::ParseResult, cxxopts::Options>;

/**
 * @brief Declares every command line option and parses the arguments.
 *
 * The parsed result and the options are returned together so that the caller
 * can print the help text, which lives on the options rather than the result.
 *
 * @param argc Argument count from @c main.
 * @param argv Argument vector from @c main.
 * @return The parse result paired with the options that produced it.
 *
 * @throws cxxopts::exceptions::exception on an unrecognised or malformed
 *         argument. @c main catches it.
 */
ParseReturnType parse_arguments(int argc, char* argv[])
{
    cxxopts::Options options("joint_cases.fixed_joint", "A fixed joint example");
    options.add_options()
    (
        "n,num_elements",
        "Number of rod elements",
        cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NUM_ELEMENTS))
    )
    (
        "l,base_length",
        "Base length of the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_BASE_LENGTH))
    )
    (
        "r,base_radius",
        "Base radius of the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_BASE_RADIUS))
    )
    (
        "density",
        "Density of the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_DENSITY))
    )
    (
        "youngs_modulus",
        "Young's modulus for the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_YOUNGS_MODULUS))
    )
    (
        "start_force_mag",
        "Force for the first link of the second rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_START_FORCE_MAG))
    )
    (
        "end_force_mag",
        "Force for the last link of the second rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_END_FORCE_MAG))
    )
    (
        "force_onset_time",
        "Onset time of the force on the second rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_FORCE_ONSET_TIME))
    )
    (
        "joint_stiffness",
        "Stiffness for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_STIFFNESS))
    )
    (
        "joint_damping",
        "Damping constant for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_DAMP))
    )
    (
        "joint_rotation_stiffness",
        "Rotational stiffness for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_ROT_STIFFNESS))
    )
    (
        "joint_rotation_damping",
        "Rotational damping costant for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_ROT_DAMP))
    )
    (
        "damping_constant",
        "Damping constant for the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_DAMP_CONSTANT))
    )
    (
        "dt",
        "Time different for the integration steps",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_DT))
    )
    (
        "start_time",
        "Starting time for the simulation",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_START_TIME))
    )
    (
        "end_time",
        "End time for the simulation",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_END_TIME))
    )
    (
        // Declared as int because it is read back as one. Declaring it double
        // and reading as<int>() throws std::bad_cast: cxxopts stores the value
        // under the type it was declared with and does not convert.
        "s,diagnostic_steps",
        "How many timesteps to wait between writing diagnostic logs",
        cxxopts::value<int>()->default_value(std::to_string(DEFAULT_DIAGNOSTIC_STEPS))
    )
    (
        "d,diagnostic_base_path",
        "Directory to write diagnostic output beneath",
        cxxopts::value<std::string>()->default_value(DEFAULT_DIAGNOSTIC_BASE_PATH)
    )
    (
        // No short name. A short name must be a single character; cxxopts
        // accepts a longer one but then renders it as "--ll" in the help while
        // the real long name goes undocumented.
        "log_level",
        "Logging level (needs to be lower case), e.g., debug, info, error",
        cxxopts::value<std::string>()->default_value(DEFAULT_LOG_LEVEL)
    )
    (
        "log_directory",
        "Logging directory - only specify if you want logs saved as .log files",
        cxxopts::value<std::string>()->default_value(DEFAULT_LOG_DIR)
    )
    (
        "log_name",
        "Name of logging file - only specify if log_directory is specified",
        cxxopts::value<std::string>()->default_value(DEFAULT_LOG_NAME)
    )
    ("h,help", "Show command line argument options");
    return std::make_pair(options.parse(argc, argv), options);
}

/**
 * @brief Builds the two-rod scenario and integrates it.
 *
 * @param parsed Parsed command line arguments.
 * @param options The options, used only to print the help text.
 * @return Zero on success.
 */
int run_simulation(
    const cxxopts::ParseResult& parsed,
    const cxxopts::Options& options
)
{
    if (parsed.count("help") != 0)
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Logging first, so that everything below is recorded.
    const auto log_level = parsed["log_level"].as<std::string>();
    const auto log_directory = parsed["log_directory"].as<std::string>();
    const auto log_name = parsed["log_name"].as<std::string>();
    if (not log_directory.empty())
    {
        if (not log_name.empty()) simulation::setup_logging(log_level, log_directory, log_name);
        else simulation::setup_logging(log_level, log_directory);
    }
    else simulation::setup_logging(log_level);

    const auto num_elements = parsed["num_elements"].as<int>();
    const auto base_length = parsed["base_length"].as<double>();
    const auto base_radius = parsed["base_radius"].as<double>();
    const auto density = parsed["density"].as<double>();
    const auto youngs_modulus = parsed["youngs_modulus"].as<double>();

    // Both rods run along +z. The normal fixes the roll of the material frame
    // and is also the direction the tip load starts out along.
    const Eigen::Vector3d direction {0.0, 0.0, 1.0};
    const Eigen::Vector3d normal {0.0, 1.0, 0.0};
    const Eigen::Vector3d rod1_start = Eigen::Vector3d::Zero();
    const Eigen::Vector3d rod2_start = rod1_start + base_length * direction;

    auto make = [&](const Eigen::Vector3d& s)
    {
        return std::make_shared<BodyVariant>(
            straight_cosserat_rod(
                num_elements, s, direction, normal, base_length, base_radius, density,
                youngs_modulus, false /* respect_radii */, tolerance
            )
        );
    };
    auto rod1_ptr = make(rod1_start);
    auto rod2_ptr = make(rod2_start);
    const std::string rod1_name = "rod1";
    const std::string rod2_name = "rod2";

    simulation::SimulationGraph sim;
    sim.add_body(rod1_name, BodyVariantWrapper(rod1_ptr));
    sim.add_body(rod2_name, BodyVariantWrapper(rod2_ptr));

    // Clamp rod one where it currently is, rather than at a hardcoded pose, so
    // the constraint stays correct if the start position ever changes.
    Eigen::Vector3d initial_pos;
    Eigen::Matrix3d initial_frame;
    std::visit(
        [&](const auto& b)
        {initial_pos = b.positions().row(0).transpose(); initial_frame = b.frames()[0]; },
        *rod1_ptr
    );
    sim.add_constraint(rod1_name, OneEndFixedBoundaryCondition(initial_pos, initial_frame));

    const auto joint_stiffness = parsed["joint_stiffness"].as<double>();
    const auto joint_damping = parsed["joint_damping"].as<double>();
    const auto joint_rotation_stiffness = parsed["joint_rotation_stiffness"].as<double>();
    const auto joint_rotation_damping = parsed["joint_rotation_damping"].as<double>();
    sim.add_connection(
        rod1_name,
        rod2_name,
        -1, /* last element of rod1 */
        0, /* first element of rod2 */
        false, /* joint is on the top node of that element of rod1 */
        true, /* joint is on the bottom node of that element of rod2 */
        FixedJoint(
            joint_stiffness,
            joint_damping,
            joint_rotation_stiffness,
            joint_rotation_damping,
            Eigen::Matrix3d::Identity() /* rotation matrix at rest */
        )
    );

    const auto start_force_mag = parsed["start_force_mag"].as<double>();
    const auto end_force_mag = parsed["end_force_mag"].as<double>();
    const auto force_onset_time = parsed["force_onset_time"].as<double>();
    sim.add_forcing_to(
        rod2_name,
        EndpointForceSinusoidal(
            normal, direction, start_force_mag, end_force_mag, force_onset_time
        )
    );

    const auto dt = parsed["dt"].as<double>();
    const auto damp_constant = parsed["damping_constant"].as<double>();
    sim.dampen(rod1_name, LegacyAnalyticalDamper(damp_constant, dt));
    sim.dampen(rod2_name, LegacyAnalyticalDamper(damp_constant, dt));

    const auto base_path_str = parsed["diagnostic_base_path"].as<std::string>();
    const auto base_path = std::filesystem::path(base_path_str);
    const auto skip_steps = parsed["diagnostic_steps"].as<int>();
    utils::nice_assert(
        skip_steps > 0,
        "diagnostic_steps must be at least one; a negative value would wrap "
        "when widened to the unsigned interval the diagnostics take"
    );
    sim.collect_diagnostics(
        rod1_name,
        simulation::BasicDiagnostics(
            base_path, rod1_name, static_cast<std::uint64_t>(skip_steps)
        )
    );
    sim.collect_diagnostics(
        rod2_name,
        simulation::BasicDiagnostics(
            base_path, rod2_name, static_cast<std::uint64_t>(skip_steps)
        )
    );

    sim.finalize();

    const auto start_time = parsed["start_time"].as<double>();
    const auto end_time = parsed["end_time"].as<double>();
    spdlog::info(
        "Solving from {} to {} with dt {}, writing to {}",
        start_time, end_time, dt, base_path.string()
    );


    SolverType solver(dt);
    const double finished = solver.full_solve(sim, start_time, end_time);
    spdlog::info("Finished at simulation time {}", finished);
    return 0;
}

/**
 * @brief Entry point.
 *
 * Parsing is wrapped so that a malformed argument prints a message and exits
 * rather than terminating on an uncaught exception.
 */
int main(int argc, char* argv[])
{
    try
    {
        const auto parsed = parse_arguments(argc, argv);
        const auto result = run_simulation(parsed.first, parsed.second);
        return result;
    }
    catch (const cxxopts::exceptions::exception& error)
    {
        std::cerr << "Could not parse the command line: " << error.what() << "\n"
                  << "Run with --help to see the available options." << std::endl;
        return EXIT_FAILURE;
    }
}
