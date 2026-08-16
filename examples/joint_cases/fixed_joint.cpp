#include <string>

#include <Eigen/Dense>

#include "physics/constraints.hpp"
#include "physics/forces.hpp"

#include "simulation/diagnostics.hpp"
#include "simulation/simulation_graph.hpp"
#include "simulation/solver.hpp"

namespace cosserat::simulation {
class SimulationGraph;
} // End namespace cosserat::simulation

using namespace cosserat;

int main()
{
    const int num_elements = 10;
    const Eigen::Vector3d direction {0.0, 0.0, 1.0};
    const Eigen::Vector3d normal {0.0, 1.0, 0.0};

    const double base_length = 0.2;
    const double base_radius = 0.007;
    const double density = 1750.0;
    const double young_modulus = 3e7;
    const double poisson_ratio = 0.5;
    const double shear_modulus = young_modulus / (1.0 + poisson_ratio);

    const Eigen::Vector3d rod1_start = Eigen::Vector3d::Zero();
    const Eigen::Vector3d rod2_start = rod1_start + base_length * direction;

    auto rod1 = physics::straight_cosserat_rod(
        num_elements,
        rod1_start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        young_modulus,
        shear_modulus
    );
    const std::string rod1_name = "rod1";

    simulation::SimulationGraph sim;
    sim.add_body(rod1_name, std::move(rod1));

    auto rod2 = physics::straight_cosserat_rod(
        num_elements,
        rod2_start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        young_modulus,
        shear_modulus
    );
    const std::string rod2_name = "rod2";
    sim.add_body(rod2_name, std::move(rod2));

    sim.constrain(
        rod1_name,
        physics::OneEndFixedBC(
            {0}, /* constrained_positions */
            {0} /* constrained frames */
        ) /* constraint */
    );
    sim.connect(
        rod1_name, /* first body */
        rod2_name, /* second body */
        physics::FixedJoint(
            1e5, /* stiffness */
            0.0, /* damping */
            1e1, /* rotation_stiffness */
            0.0 /* rotation_damping */
        ) /* joint */
    );
    sim.add_forcing_to(
        rod2_name,
        physics::EndpointForceSinusoidal(
            normal, /* normal_dir */
            direction, /* tangent_dir */
            0.0, /* first_link_mag */
            5e-3, /* last_link_mag */
            0.2 /* onset_time_ */
        ) /* force */
    );
    const double dt = 1e-4;
    sim.dampen(
        rod1_name,
        physics::AnalyticLinearDamper(
            0.4, /* damping_constant */
            dt
        )
    );
    sim.dampen(
        rod2_name,
        physics::AnalyticLinearDamper(
            0.4, /* damping_constant */
            dt
        )
    );

    sim.collect_diagnostics(
        rod1_name,
        simulation::BasicDiagnostics(
            1000 /* steps_to_skip */
        )
    );
    sim.collect_diagnostics(
        rod2_name,
        simulation::BasicDiagnostics(
            1000 /* steps_to_skip */
        )
    );
    sim.finalize();

    using SolverType = simulation::Solver<simulation::SimulationGraph>;
    SolverType solver(dt);
    const double start_time = 0.0;
    const double end_time = 10.0;
    sovler.full_solve(sim, start_time, end_time);

    sim.write_diagnostics();
}
