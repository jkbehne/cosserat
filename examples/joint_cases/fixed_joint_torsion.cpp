/**
 * @file fixed_joint_torsion.cpp
 * @brief Two perpendicular rods rigidly joined, driven by a twisting torque.
 *
 * A port of PyElastica's @c examples/JointCases/fixed_joint_torsion.py. It is
 * the companion to @c fixed_joint.cpp and shares its shape, but exercises the
 * part of a fixed joint that the simpler example never touches: the
 * constraint on relative *orientation*, rather than on relative position.
 *
 * @section torsion_scenario The scenario
 *
 * Rod one runs along @c +z from the origin. Rod two starts where rod one ends,
 * but runs along @c +y, so the pair forms a right angle. A fixed joint holds
 * the corner, and a uniform torque about @c +z is applied along the whole of
 * rod two, twisting it about the world vertical. Because rod two lies
 * perpendicular to that axis, the torque tries to sweep it around the corner,
 * and everything resisting is the joint's rotational spring plus the rods'
 * own bending and twisting stiffness.
 *
 * @verbatim
 *        z
 *        ^
 *        |         torque about +z applied along rod 2
 *        |            ,--.
 *        |   #########|   |========>  rod 2 (free, runs along +y)
 *        |   ^        `--'
 *        |   fixed joint holds the right angle
 *        |   |
 *        |   |  rod 1 (clamped at its base, runs along +z)
 *        |   |
 *        +---#####------------------> y
 *            clamped base
 * @endverbatim
 *
 * @section torsion_rest Why the rest rotation matters here
 *
 * A fixed joint restores the relative orientation of its two ends toward a
 * prescribed rest rotation. In @c fixed_joint.cpp the two rods are collinear
 * and that rest rotation is the identity. Here they are perpendicular, so the
 * identity would be the wrong target: the joint would fight to straighten the
 * corner from the first step.
 *
 * The rest rotation is therefore taken from the rods as built, exactly as the
 * reference does with its @c get_relative_rotation_two_systems helper:
 *
 * @f[ \mathbf{C}_{12} = \mathbf{Q}_1 \mathbf{Q}_2^{T} @f]
 *
 * using the frame of the joined element on each side. See
 * @ref relative_rotation.
 *
 * @warning Leaving the rest rotation as the identity here does not merely
 *          apply the wrong restoring torque, it applies **no torque at all**.
 *          For this geometry @f$ \mathbf{Q}_1 \mathbf{Q}_2^{T} @f$ is a half
 *          turn, and a half turn is the degenerate case for recovering a
 *          rotation vector: the antisymmetric part of the matrix vanishes, so
 *          the extracted vector is zero however far the joint is twisted. The
 *          rotational constraint disappears silently and the joint behaves as
 *          if only its translational spring existed. Measured against a
 *          deliberately twisted rod two, the correct rest rotation gives a
 *          restoring torque growing linearly with the twist angle, while the
 *          identity gives exactly zero at every angle.
 *
 * @section torsion_shear The shear modulus
 *
 * The reference sets @f$ G = E / (1 + \nu) @f$ with a Poisson ratio of one
 * half, giving @f$ E/1.5 @f$. That is not the library default of @f$ E/3 @f$,
 * and the difference is not cosmetic in this example: the twisting stiffness
 * of an element is @f$ G I_3 @f$, so the default would halve exactly the
 * quantity under test.
 *
 * @ref straight_rod_with_shear_modulus therefore builds the rods through the
 * @c CosseratRod constructor that accepts both moduli, rather than through
 * @c straight_cosserat_rod, which only takes Young's modulus. If a two-moduli
 * overload of @c straight_cosserat_rod ever appears, that helper can go.
 *
 * @section torsion_output What comes out
 *
 * Identical in shape to @c fixed_joint.cpp: every @c diagnostic_steps steps
 * each rod writes its positions, frames and radii beneath
 * @c diagnostic_base_path, one directory per step and one subdirectory per
 * rod, with a frame written before the first step as well.
 *
 * The interesting quantity here is the orientation of each rod's far element
 * over time, which the reference plots. The frames are on disk, so that is a
 * post-processing question rather than something this executable does.
 *
 * @section torsion_running Running it
 *
 * Every parameter has a default matching the reference, so it runs bare. Note
 * that the reference timestep is @c 1e-5 over ten seconds, which is a million
 * steps; shorten @c --end_time while experimenting.
 *
 * @verbatim
 *   ./fixed_joint_torsion
 *   ./fixed_joint_torsion --end_time 0.5 --log_level debug
 *   ./fixed_joint_torsion --torque_magnitude 1e-2 --joint_rotation_stiffness 1e4
 *   ./fixed_joint_torsion --help
 * @endverbatim
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "physics/constraints.hpp"
#include "physics/damping.hpp"
#include "physics/forces.hpp"
#include "physics/joints.hpp"
#include "physics/rods.hpp"

#include "simulation/diagnostics.hpp"
#include "simulation/logging.hpp"
#include "simulation/simulation_graph.hpp"
#include "simulation/solver.hpp"

#include "utils/assertions.hpp"
#include "utils/file_utils.hpp"

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
constexpr double DEFAULT_POISSON_RATIO = 0.5;

constexpr double DEFAULT_TORQUE_MAGNITUDE = 5e-3;

constexpr double DEFAULT_JOINT_STIFFNESS = 1e5;
constexpr double DEFAULT_JOINT_DAMP = 1.0;
constexpr double DEFAULT_JOINT_ROT_STIFFNESS = 1e3;
constexpr double DEFAULT_JOINT_ROT_DAMP = 1e-3;

constexpr double DEFAULT_DAMP_CONSTANT = 0.4;
constexpr double DEFAULT_DT = 1e-5;
constexpr double DEFAULT_START_TIME = 0.0;
constexpr double DEFAULT_END_TIME = 10.0;

constexpr int DEFAULT_DIAGNOSTIC_STEPS = 1000;
const std::string DEFAULT_DIAGNOSTIC_BASE_PATH =
    "/tmp/cosserat_logs/joint_cases/fixed_joint_torsion";

const std::string DEFAULT_LOG_LEVEL = "info";
const std::string DEFAULT_LOG_DIR = "";
const std::string DEFAULT_LOG_NAME = "";

constexpr double tolerance = 1e-12;

using ParseReturnType = std::pair<cxxopts::ParseResult, cxxopts::Options>;

/**
 * @brief The rotation carrying one system's frame onto another's.
 *
 * Mirrors the reference's @c get_relative_rotation_two_systems: given the
 * frames of the two elements a joint will act between, the relative rotation
 * is @f$ \mathbf{Q}_1 \mathbf{Q}_2^{T} @f$. Passing it to a fixed joint as the
 * rest rotation makes the joint hold whatever relative orientation the two
 * bodies were built in, rather than trying to align them.
 *
 * @param frame_one Frame of the element on the first system.
 * @param frame_two Frame of the element on the second system.
 * @return The relative rotation, which is itself a rotation matrix.
 */
Eigen::Matrix3d relative_rotation(
    const Eigen::Matrix3d& frame_one, const Eigen::Matrix3d& frame_two
)
{
    const Eigen::Matrix3d relative = frame_one * frame_two.transpose();
    utils::nice_assert(
        math::is_orthogonal(relative, 1e-10),
        "the relative rotation of two frames must itself be a rotation"
    );
    return relative;
}

/**
 * @brief Declares every command line option and parses the arguments.
 *
 * @param argc Argument count from @c main.
 * @param argv Argument vector from @c main.
 * @return The parse result paired with the options that produced it.
 *
 * @throws cxxopts::exceptions::exception on a malformed argument, caught by
 *         @c main.
 */
ParseReturnType parse_arguments(int argc, char* argv[])
{
    cxxopts::Options options(
        "joint_cases.fixed_joint_torsion", "A fixed joint under torsion"
    );
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
        // The shear modulus follows as E/(1+nu), which for the default ratio of
        // one half is E/1.5 rather than the library default of E/3.
        "poisson_ratio",
        "Poisson ratio, from which the shear modulus E/(1+nu) is taken",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_POISSON_RATIO))
    )
    (
        "t,torque_magnitude",
        "Magnitude of the uniform torque applied along the second rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_TORQUE_MAGNITUDE))
    )
    (
        "joint_stiffness",
        "Translational stiffness for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_STIFFNESS))
    )
    (
        "joint_damping",
        "Translational damping constant for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_DAMP))
    )
    (
        "joint_rotation_stiffness",
        "Rotational stiffness for the joint",
        cxxopts::value<double>()->default_value(
            std::to_string(DEFAULT_JOINT_ROT_STIFFNESS))
    )
    (
        "joint_rotation_damping",
        "Rotational damping constant for the joint",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_JOINT_ROT_DAMP))
    )
    (
        "damping_constant",
        "Damping constant for the rods",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_DAMP_CONSTANT))
    )
    (
        "dt",
        "Time step for the integration",
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
 * @brief Builds the two perpendicular rods and integrates them.
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
    const auto poisson_ratio = parsed["poisson_ratio"].as<double>();
    utils::nice_assert(
        poisson_ratio > -1.0, "poisson_ratio must exceed -1 for a positive shear modulus"
    );
    const double shear_modulus = youngs_modulus / (1.0 + poisson_ratio);

    // The right angle that makes this a torsion case: rod one runs along +z,
    // rod two along +y, starting where rod one ends.
    const Eigen::Vector3d direction_rod1 {0.0, 0.0, 1.0};
    const Eigen::Vector3d normal_rod1 {0.0, 1.0, 0.0};
    const Eigen::Vector3d direction_rod2 {0.0, 1.0, 0.0};
    const Eigen::Vector3d normal_rod2 {0.0, 0.0, 1.0};

    const Eigen::Vector3d start_rod1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d start_rod2 = start_rod1 + base_length * direction_rod1;

    auto rod1_ptr = std::make_shared<BodyVariant>(
        physics::straight_cosserat_rod(
            num_elements, start_rod1, direction_rod1, normal_rod1,
            base_length, base_radius, density, youngs_modulus, shear_modulus,
            false /* respect radii */, tolerance
        )
    );
    auto rod2_ptr = std::make_shared<BodyVariant>(
        physics::straight_cosserat_rod(
            num_elements, start_rod2, direction_rod2, normal_rod2,
            base_length, base_radius, density, youngs_modulus, shear_modulus,
            false /* respect radii */, tolerance
        )
    );
    const std::string rod1_name = "rod1";
    const std::string rod2_name = "rod2";

    simulation::SimulationGraph sim;
    sim.add_body(rod1_name, BodyVariantWrapper(rod1_ptr));
    sim.add_body(rod2_name, BodyVariantWrapper(rod2_ptr));

    // Clamp rod one where it currently is, rather than at a hardcoded pose.
    Eigen::Vector3d initial_pos;
    Eigen::Matrix3d initial_frame;
    std::visit(
        [&](const auto& body)
        {initial_pos = body.positions().row(0).transpose(); initial_frame = body.frames()[0]; },
        *rod1_ptr
    );
    sim.add_constraint(rod1_name, OneEndFixedBoundaryCondition(initial_pos, initial_frame));

    // The joint must hold the right angle the rods were built with, so its rest
    // rotation comes from the two frames it joins. See the file notes on why
    // leaving this as the identity silently removes the constraint entirely.
    Eigen::Matrix3d frame_rod1_tip;
    Eigen::Matrix3d frame_rod2_base;
    std::visit([&](const auto& body) {frame_rod1_tip = body.frames().back(); }, *rod1_ptr);
    std::visit([&](const auto& body) {frame_rod2_base = body.frames().front(); }, *rod2_ptr);
    const Eigen::Matrix3d rest_rotation =
        relative_rotation(frame_rod1_tip, frame_rod2_base);

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
            rest_rotation
        )
    );

    // A uniform torque about the world vertical, spread over rod two. Because
    // rod two lies perpendicular to that axis, this sweeps it around the joint.
    const auto torque_magnitude = parsed["torque_magnitude"].as<double>();
    const Eigen::Vector3d torque_direction {0.0, 0.0, 1.0};
    sim.add_forcing_to(rod2_name, UniformTorque(torque_magnitude * torque_direction));

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
        "Solving from {} to {} with dt {} ({} steps), writing to {}",
        start_time, end_time, dt,
        static_cast<std::int64_t>(std::round((end_time - start_time) / dt)),
        base_path.string()
    );
    spdlog::info(
        "Young's modulus {}, Poisson ratio {}, so shear modulus {}",
        youngs_modulus, poisson_ratio, shear_modulus
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
        return run_simulation(parsed.first, parsed.second);
    }
    catch (const cxxopts::exceptions::exception& error)
    {
        std::cerr << "Could not parse the command line: " << error.what() << "\n"
                  << "Run with --help to see the available options." << std::endl;
        return EXIT_FAILURE;
    }
}
