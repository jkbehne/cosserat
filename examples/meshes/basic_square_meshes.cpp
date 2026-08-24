/**
 * @file basic_square_meshes.cpp
 * @brief A rod falling under gravity onto a meshed ground and a meshed block.
 *
 * The first example built on mesh bodies rather than analytic primitives.
 * Everything the rod touches here is an arbitrary closed triangle mesh, seen
 * through a signed distance field, so this exercises the whole path from a
 * mesh on disk to a contact force: mass properties, principal frame, distance
 * field, and @ref RodMeshContact.
 *
 * Both obstacles happen to be boxes, which makes the expected outcome easy to
 * reason about, but nothing in the setup knows that. Swapping either for a
 * loaded mesh of any shape changes only the call that builds it.
 *
 * @section bsm_scenario The scenario
 *
 * A large flat box is the ground, its top face at @c z = 0. A smaller box sits
 * on it a little way along @c +x. A rod is released just above, leaning up
 * toward the block, with gravity the only applied load. It falls a short
 * distance, its lower end meets the ground and its upper end catches the
 * block, and it settles as a ramp propped against the block's near top edge.
 *
 * @verbatim
 *     z
 *     ^                      rod, released here
 *     |                  ,-'
 *     |              ,-'          ___________
 *     |          ,-'             |           |
 *     |      ,-'                 |   block   |
 *     |  ,-'                     |           |
 *     +--########################+###########+###########-> x
 *        ground (top face at z = 0)
 * @endverbatim
 *
 * @section bsm_pinned Why both boxes are pinned
 *
 * Both meshes are held in place by a @ref FixedConstraint, which makes them
 * static obstacles rather than participants. That is not decoration: the only
 * contact rule that understands a distance field is @ref RodMeshContact, which
 * acts between a rod and a mesh. There is no mesh against mesh contact, so an
 * unpinned block would fall straight through the ground it is sitting on.
 *
 * A mesh body is otherwise a fully capable rigid body. It has real mass and
 * inertia derived from its own geometry, and it responds to forces, dampers,
 * constraints and joints exactly as a sphere or a cylinder does. Only its
 * contact coverage is partial, and only that keeps this example static.
 *
 * @section bsm_settling What the parameters have to do
 *
 * The **contact damping** has to be large enough to absorb the impact. A
 * penalty contact with too little damping is very nearly a lossless spring:
 * measured here, at a hundredth of the default the rod returns to its release
 * height on every bounce and never settles at all, while at the default it
 * comes to rest within a tenth of a second. Rod damping is not a substitute,
 * because it drags on absolute velocity rather than dissipating at the
 * contact.
 *
 * The **friction** has to be non zero or nothing ever stops moving
 * horizontally. Both contact surfaces here are flat and level, so their
 * normals are vertical and nothing opposes sliding; without friction the rod
 * lands correctly and then slides across the ground at constant speed forever,
 * which is right but not useful.
 *
 * @section bsm_expected What to expect
 *
 * With the defaults the rod comes to rest at about @c t = 0.5 with its lower
 * node on the ground near @c x = -0.07, bearing on the block's near top edge
 * around @c x = 0.07, and its far end cantilevered over the block. A node
 * resting on a surface sits one rod radius above it, so the settled height on
 * the ground is @c 0.007 less a few microns of penetration.
 *
 * @section bsm_running Running it
 *
 * @verbatim
 *   ./basic_square_meshes
 *   ./basic_square_meshes --end_time 0.5 --log_level debug
 *   ./basic_square_meshes --contact_damping 0.01   # watch it bounce forever
 *   ./basic_square_meshes --friction 0.0           # watch it slide forever
 *   ./basic_square_meshes --help
 * @endverbatim
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "math/triangle_mesh_field.hpp"

#include "physics/bodies.hpp"
#include "physics/constraints.hpp"
#include "physics/contacts.hpp"
#include "physics/forces.hpp"
#include "physics/mesh_body.hpp"
#include "physics/rod_mesh_contact.hpp"
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

constexpr int DEFAULT_NUM_ELEMENTS = 12;

constexpr double DEFAULT_BASE_LENGTH = 0.2;
constexpr double DEFAULT_BASE_RADIUS = 0.007;
constexpr double DEFAULT_DENSITY = 1750.0;
constexpr double DEFAULT_YOUNGS_MODULUS = 3e7;

/** @brief Half extents of the ground slab, whose top face lands at z = 0. */
constexpr double DEFAULT_GROUND_HALF_WIDTH = 0.5;
constexpr double DEFAULT_GROUND_HALF_THICKNESS = 0.05;

/** @brief The block the rod comes to rest against. */
constexpr double DEFAULT_BLOCK_CENTER_X = 0.13;
constexpr double DEFAULT_BLOCK_HALF_LENGTH = 0.06;
constexpr double DEFAULT_BLOCK_HALF_WIDTH = 0.05;
constexpr double DEFAULT_BLOCK_HALF_HEIGHT = 0.04;

/** @brief Where the rod is released, and how far it leans toward the block. */
constexpr double DEFAULT_ROD_START_X = -0.053;
constexpr double DEFAULT_ROD_START_Z = 0.03;
constexpr double DEFAULT_ROD_TILT_DEGREES = 23.6;

constexpr double DEFAULT_CONTACT_STIFFNESS = 1e3;
constexpr double DEFAULT_CONTACT_DAMPING = 1.0;
constexpr double DEFAULT_VELOCITY_DAMPING = 10.0;
constexpr double DEFAULT_FRICTION = 0.6;

/** @brief Padding on each field's domain; must exceed the rod's radius. */
constexpr double DEFAULT_FIELD_MARGIN = 0.05;

constexpr double DEFAULT_DT = 1e-5;
constexpr double DEFAULT_START_TIME = 0.0;
constexpr double DEFAULT_END_TIME = 1.0;

constexpr int DEFAULT_DIAGNOSTIC_STEPS = 500;
const std::string DEFAULT_DIAGNOSTIC_BASE_PATH =
    "/tmp/cosserat_logs/meshes/basic_square_meshes";

const std::string DEFAULT_LOG_LEVEL = "info";
const std::string DEFAULT_LOG_DIR = "";
const std::string DEFAULT_LOG_NAME = "";

using ParseReturnType = std::pair<cxxopts::ParseResult, cxxopts::Options>;

/**
 * @brief Pins a rigid body exactly where it currently sits.
 *
 * A rigid body has a single node and a single element, so both index lists
 * hold only zero. Used to make the two meshes immovable; see
 * @ref bsm_pinned.
 *
 * @param body The body to hold still.
 * @return A constraint fixing its current position and orientation.
 */
FixedConstraint pin_in_place(const RigidBody& body)
{
    return FixedConstraint(
        std::vector<std::int64_t>{0}, body.positions(),
        std::vector<std::int64_t>{0}, body.frames()
    );
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
        "meshes.basic_square_meshes", "A rod falling onto meshed obstacles"
    );
    options.add_options()
    (
        "n,num_elements",
        "Number of rod elements",
        cxxopts::value<int>()->default_value(std::to_string(DEFAULT_NUM_ELEMENTS))
    )
    (
        "l,base_length",
        "Length of the rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_BASE_LENGTH))
    )
    (
        "r,base_radius",
        "Radius of the rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_BASE_RADIUS))
    )
    (
        "density",
        "Density of the rod and of both meshes",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_DENSITY))
    )
    (
        "youngs_modulus",
        "Young's modulus for the rod",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_YOUNGS_MODULUS))
    )
    (
        "block_center_x",
        "Where along x the block sits",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_BLOCK_CENTER_X))
    )
    (
        "block_half_height",
        "Half the block's height, so its top face is at twice this",
        cxxopts::value<double>()->default_value(
            std::to_string(DEFAULT_BLOCK_HALF_HEIGHT))
    )
    (
        "rod_start_x",
        "Where along x the rod's lower end is released",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_ROD_START_X))
    )
    (
        "rod_start_z",
        "Height the rod's lower end is released from",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_ROD_START_Z))
    )
    (
        "rod_tilt_degrees",
        "How far the rod leans up toward the block; negative leans away",
        cxxopts::value<double>()->default_value(
            std::to_string(DEFAULT_ROD_TILT_DEGREES))
    )
    (
        "contact_stiffness",
        "Spring constant for both contacts",
        cxxopts::value<double>()->default_value(
            std::to_string(DEFAULT_CONTACT_STIFFNESS))
    )
    (
        // Small values do not merely settle slowly, they never settle at all;
        // see the notes on parameters above.
        "contact_damping",
        "Damping for both contacts, which is what absorbs the impact",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_CONTACT_DAMPING))
    )
    (
        "velocity_damping",
        "Damping in the slip direction, approximating stiction when large",
        cxxopts::value<double>()->default_value(
            std::to_string(DEFAULT_VELOCITY_DAMPING))
    )
    (
        "friction",
        "Coulomb friction coefficient; at zero the rod slides forever",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_FRICTION))
    )
    (
        "field_margin",
        "Padding on each field's domain; must exceed the rod's radius",
        cxxopts::value<double>()->default_value(std::to_string(DEFAULT_FIELD_MARGIN))
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
 * @brief Builds the ground, the block and the rod, and integrates them.
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

    const auto density = parsed["density"].as<double>();
    const auto field_margin = parsed["field_margin"].as<double>();
    const auto base_radius = parsed["base_radius"].as<double>();
    utils::nice_assert(
        field_margin > base_radius,
        "the field margin must exceed the rod's radius, or a rod touching a "
        "surface from outside falls off the edge of the field's domain"
    );

    // The ground slab, positioned so its top face is exactly z = 0. That makes
    // every height below readable: a rod resting on it sits at its own radius.
    const double ground_half_thickness = DEFAULT_GROUND_HALF_THICKNESS;
    auto ground = std::make_shared<BodyVariant>(MeshBody(
        math::make_box_mesh(
            Eigen::Vector3d(0.0, 0.0, -ground_half_thickness),
            Eigen::Vector3d(DEFAULT_GROUND_HALF_WIDTH, DEFAULT_GROUND_HALF_WIDTH,
                            ground_half_thickness)
        ),
        density, field_margin, true
    ));

    // The block, sitting on the ground rather than intersecting it.
    const auto block_half_height = parsed["block_half_height"].as<double>();
    const auto block_center_x = parsed["block_center_x"].as<double>();
    auto block = std::make_shared<BodyVariant>(MeshBody(
        math::make_box_mesh(
            Eigen::Vector3d(block_center_x, 0.0, block_half_height),
            Eigen::Vector3d(DEFAULT_BLOCK_HALF_LENGTH, DEFAULT_BLOCK_HALF_WIDTH,
                            block_half_height)
        ),
        density, field_margin, true
    ));

    // Leaning up toward the block, so that it comes to rest bridging the two.
    const auto tilt_degrees = parsed["rod_tilt_degrees"].as<double>();
    const double tilt = tilt_degrees * std::numbers::pi / 180.0;
    const Eigen::Vector3d direction(std::cos(tilt), 0.0, std::sin(tilt));
    const Eigen::Vector3d normal(0.0, 1.0, 0.0);
    const Eigen::Vector3d rod_start(
        parsed["rod_start_x"].as<double>(), 0.0, parsed["rod_start_z"].as<double>());

    auto rod = std::make_shared<BodyVariant>(straight_cosserat_rod(
        parsed["num_elements"].as<int>(), rod_start, direction, normal,
        parsed["base_length"].as<double>(), base_radius, density,
        parsed["youngs_modulus"].as<double>(), false /* respect_radii */, 1e-12
    ));

    const std::string ground_name = "ground";
    const std::string block_name = "block";
    const std::string rod_name = "rod";

    simulation::SimulationGraph sim;
    sim.add_body(ground_name, BodyVariantWrapper(ground));
    sim.add_body(block_name, BodyVariantWrapper(block));
    sim.add_body(rod_name, BodyVariantWrapper(rod));

    // Held still, because nothing yet makes one mesh rest on another.
    sim.add_constraint(ground_name, pin_in_place(std::get<MeshBody>(*ground)));
    sim.add_constraint(block_name, pin_in_place(std::get<MeshBody>(*block)));

    // Gravity is the only applied load.
    sim.add_forcing_to(rod_name, GravityForceZ{});

    const auto contact_stiffness = parsed["contact_stiffness"].as<double>();
    const auto contact_damping = parsed["contact_damping"].as<double>();
    const auto velocity_damping = parsed["velocity_damping"].as<double>();
    const auto friction = parsed["friction"].as<double>();
    for (const std::string& obstacle : {ground_name, block_name})
    {
        sim.add_contact(
            rod_name, obstacle,
            RodMeshContact(
                contact_stiffness, contact_damping, velocity_damping, friction)
        );
    }

    const auto base_path =
        std::filesystem::path(parsed["diagnostic_base_path"].as<std::string>());
    const auto skip_steps = parsed["diagnostic_steps"].as<int>();
    utils::nice_assert(skip_steps > 0, "diagnostic_steps must be at least one");
    for (const std::string& name : {ground_name, block_name, rod_name})
    {
        sim.collect_diagnostics(
            name,
            simulation::BasicDiagnostics(
                base_path, name, static_cast<std::uint64_t>(skip_steps))
        );
    }

    sim.finalize();

    const auto dt = parsed["dt"].as<double>();
    const auto start_time = parsed["start_time"].as<double>();
    const auto end_time = parsed["end_time"].as<double>();

    const MeshBody& ground_body = std::get<MeshBody>(*ground);
    const MeshBody& block_body = std::get<MeshBody>(*block);
    spdlog::info(
        "Ground: {} triangles, mass {:.4f} kg, top face at z = {:.4f}",
        ground_body.field().num_triangles(), ground_body.masses()(0), 0.0
    );
    spdlog::info(
        "Block: {} triangles, mass {:.4f} kg, top face at z = {:.4f}",
        block_body.field().num_triangles(), block_body.masses()(0),
        2.0 * block_half_height
    );
    spdlog::info(
        "Solving from {} to {} with dt {} ({} steps), writing to {}",
        start_time, end_time, dt,
        static_cast<std::int64_t>(std::round((end_time - start_time) / dt)),
        base_path.string()
    );

    SolverType solver(dt);
    const double finished = solver.full_solve(sim, start_time, end_time);

    const CosseratRod& settled = std::get<CosseratRod>(*rod);
    spdlog::info("Finished at simulation time {}", finished);
    spdlog::info(
        "Rod rests between z = {:.5f} and {:.5f}, moving at {:.5f} m/s",
        settled.positions().col(2).minCoeff(),
        settled.positions().col(2).maxCoeff(),
        settled.velocities().rowwise().norm().maxCoeff()
    );
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
