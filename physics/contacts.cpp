#include "physics/contacts.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "math/finite_difference.hpp"
#include "math/linalg.hpp"

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/** @brief Fails unless a coefficient is finite and strictly positive. */
void assert_positive(double value, const std::string& name)
{
    nice_assert(
        std::isfinite(value) and value > 0.0,
        "Expected " + name + " to be finite and greater than zero"
    );
}

/** @brief Fails unless a coefficient is finite and not negative. */
void assert_non_negative(double value, const std::string& name)
{
    nice_assert(
        std::isfinite(value) and value >= 0.0,
        "Expected " + name + " to be finite and not negative"
    );
}

/** @brief Guard against dividing by a vanishing magnitude. */
constexpr double unit_epsilon = 1e-14;

} // End anonymous namespace

namespace detail {

Vector3DStack element_start_positions(const Vector3DStack& positions)
{
    nice_assert(positions.rows() >= 2, "A rod needs at least two nodes");
    return positions.topRows(positions.rows() - 1);
}

Vector3DStack element_center_positions(const Vector3DStack& positions)
{
    nice_assert(positions.rows() >= 2, "A rod needs at least two nodes");
    // The midpoint of each adjacent node pair is exactly a row-wise average.
    return math::row_average(positions);
}

Vector3DStack node_to_element_mass_or_force(const Vector3DStack& nodal)
{
    nice_assert(nodal.rows() >= 2, "A rod needs at least two nodes");
    const Eigen::Index num_elements = nodal.rows() - 1;

    Vector3DStack element = math::row_average(nodal);
    // The two end elements pick up an extra half of the terminal node, so the
    // total carried by the elements matches the total on the nodes.
    element.row(0) += 0.5 * nodal.row(0);
    element.row(num_elements - 1) += 0.5 * nodal.row(nodal.rows() - 1);
    return element;
}

Vector3DStack node_to_element_velocity(
    const Eigen::VectorXd& masses, const Vector3DStack& nodal
)
{
    nice_assert(nodal.rows() >= 2, "A rod needs at least two nodes");
    nice_assert(masses.size() == nodal.rows(), "Masses and nodes must agree");

    const Eigen::Index num_elements = nodal.rows() - 1;
    Vector3DStack element(num_elements, 3);
    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        const double total = masses(idx) + masses(idx + 1);
        nice_assert(total > 0.0, "Adjacent node masses sum to zero");
        element.row(idx) =
            (masses(idx + 1) * nodal.row(idx + 1) + masses(idx) * nodal.row(idx))
            / total;
    }
    return element;
}

void elements_to_nodes_inplace(const Vector3DStack& element, Vector3DStack& nodal)
{
    nice_assert(
        nodal.rows() == element.rows() + 1, "Nodes must be one more than elements"
    );
    // Half of each element's share to the node on either side, which is
    // exactly the average kernel.
    nodal += math::row_average_kernel(element);
}

Eigen::VectorXd find_slipping_elements(
    const Vector3DStack& slip_velocity, double threshold
)
{
    assert_positive(threshold, "slip velocity threshold");

    const Eigen::VectorXd speed = math::row_norms(slip_velocity);
    Eigen::VectorXd slip_function = Eigen::VectorXd::Ones(speed.size());
    for (Eigen::Index idx = 0; idx < speed.size(); ++idx)
    {
        if (std::abs(speed(idx)) > threshold)
        {
            slip_function(idx) =
                std::abs(1.0 - std::min(1.0, speed(idx) / threshold - 1.0));
        }
    }
    return slip_function;
}

void contact_forces_rod_rod(
    const Vector3DStack& element_start_one,
    const Eigen::VectorXd& radii_one,
    const Eigen::VectorXd& lengths_one,
    const Vector3DStack& tangents_one,
    const Vector3DStack& velocities_one,
    const Vector3DStack& internal_forces_one,
    Vector3DStack& external_forces_one,
    const Vector3DStack& element_start_two,
    const Eigen::VectorXd& radii_two,
    const Eigen::VectorXd& lengths_two,
    const Vector3DStack& tangents_two,
    const Vector3DStack& velocities_two,
    const Vector3DStack& internal_forces_two,
    Vector3DStack& external_forces_two,
    double contact_k,
    double contact_nu
)
{
    const Eigen::Index count_one = element_start_one.rows();
    const Eigen::Index count_two = element_start_two.rows();

    for (Eigen::Index i = 0; i < count_one; ++i)
    {
        const Eigen::Vector3d start_one = element_start_one.row(i).transpose();
        const Eigen::Vector3d edge_one =
            lengths_one(i) * tangents_one.row(i).transpose();

        for (Eigen::Index j = 0; j < count_two; ++j)
        {
            const double radii_sum = radii_one(i) + radii_two(j);
            const double lengths_sum = lengths_one(i) + lengths_two(j);

            const Eigen::Vector3d start_two = element_start_two.row(j).transpose();

            // Cheap rejection before the full closest-approach test.
            if ((start_one - start_two).norm() >= radii_sum + lengths_sum) continue;

            const Eigen::Vector3d edge_two =
                lengths_two(j) * tangents_two.row(j).transpose();

            Eigen::Vector3d distance_vector =
                math::minimum_distance_segment_segment(
                    start_one, edge_one, start_two, edge_two
                ).distance_vector;
            const double distance = distance_vector.norm();
            // Exactly intersecting axes leave the normal undefined; see
            // contact_normal_tolerance.
            if (distance < contact_normal_tolerance) continue;
            distance_vector /= distance;

            const double gamma = radii_sum - distance;
            if (gamma < contact_separation_tolerance) continue;

            // The net force already pressing each element toward the other,
            // whose compressive part is cancelled below. The reference marks
            // this term as one it intends to remove.
            const Eigen::Vector3d elemental_forces_one =
                0.5 * (external_forces_one.row(i) + external_forces_one.row(i + 1)
                       + internal_forces_one.row(i) + internal_forces_one.row(i + 1))
                          .transpose();
            const Eigen::Vector3d elemental_forces_two =
                0.5 * (external_forces_two.row(j) + external_forces_two.row(j + 1)
                       + internal_forces_two.row(j) + internal_forces_two.row(j + 1))
                          .transpose();
            const Eigen::Vector3d equilibrium_forces =
                -elemental_forces_one + elemental_forces_two;
            const double normal_force =
                std::abs(std::min(equilibrium_forces.dot(distance_vector), 0.0));

            const double mask = gamma > 0.0 ? 1.0 : 0.0;
            const double contact_force = contact_k * gamma;

            const Eigen::Vector3d interpenetration_velocity =
                0.5 * ((velocities_one.row(i) + velocities_one.row(i + 1))
                       - (velocities_two.row(j) + velocities_two.row(j + 1)))
                          .transpose();
            const double contact_damping_force =
                contact_nu * interpenetration_velocity.dot(distance_vector);

            const Eigen::Vector3d net_contact_force =
                (normal_force + 0.5 * mask * (contact_damping_force + contact_force))
                * distance_vector;

            // The end elements of each rod split their share unevenly, which
            // keeps the total applied to a rod the same wherever contact lands.
            if (i == 0)
            {
                external_forces_one.row(i) -= (net_contact_force * 2.0 / 3.0).transpose();
                external_forces_one.row(i + 1) -= (net_contact_force * 4.0 / 3.0).transpose();
            }
            else if (i == count_one - 1)
            {
                external_forces_one.row(i) -= (net_contact_force * 4.0 / 3.0).transpose();
                external_forces_one.row(i + 1) -= (net_contact_force * 2.0 / 3.0).transpose();
            }
            else
            {
                external_forces_one.row(i) -= net_contact_force.transpose();
                external_forces_one.row(i + 1) -= net_contact_force.transpose();
            }

            if (j == 0)
            {
                external_forces_two.row(j) += (net_contact_force * 2.0 / 3.0).transpose();
                external_forces_two.row(j + 1) += (net_contact_force * 4.0 / 3.0).transpose();
            }
            else if (j == count_two - 1)
            {
                external_forces_two.row(j) += (net_contact_force * 4.0 / 3.0).transpose();
                external_forces_two.row(j + 1) += (net_contact_force * 2.0 / 3.0).transpose();
            }
            else
            {
                external_forces_two.row(j) += net_contact_force.transpose();
                external_forces_two.row(j + 1) += net_contact_force.transpose();
            }
        }
    }
}

void contact_forces_self_rod(
    const Vector3DStack& element_start,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& lengths,
    const Vector3DStack& tangents,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces,
    double contact_k,
    double contact_nu
)
{
    const Eigen::Index count = element_start.rows();

    for (Eigen::Index i = 0; i < count; ++i)
    {
        // Neighbouring elements are always touching, so a window either side
        // is excluded. Its width is roughly the arc a rod of this radius needs
        // to bend back onto itself.
        nice_assert(lengths(i) > 0.0, "Element length must be positive");
        const auto skip = static_cast<Eigen::Index>(
            1 + static_cast<std::int64_t>(
                    std::ceil(0.8 * std::numbers::pi * radii(i) / lengths(i))));

        const Eigen::Vector3d start_i = element_start.row(i).transpose();
        const Eigen::Vector3d edge_i = lengths(i) * tangents.row(i).transpose();

        for (Eigen::Index j = i - skip; j >= 0; --j)
        {
            const double radii_sum = radii(i) + radii(j);
            const double lengths_sum = lengths(i) + lengths(j);

            const Eigen::Vector3d start_j = element_start.row(j).transpose();
            if ((start_i - start_j).norm() >= radii_sum + lengths_sum) continue;

            const Eigen::Vector3d edge_j = lengths(j) * tangents.row(j).transpose();

            Eigen::Vector3d distance_vector =
                math::minimum_distance_segment_segment(
                    start_i, edge_i, start_j, edge_j
                ).distance_vector;
            const double distance = distance_vector.norm();
            // Exactly intersecting axes leave the normal undefined; see
            // contact_normal_tolerance.
            if (distance < contact_normal_tolerance) continue;
            distance_vector /= distance;

            const double gamma = radii_sum - distance;
            if (gamma < contact_separation_tolerance) continue;

            const double mask = gamma > 0.0 ? 1.0 : 0.0;
            const double contact_force = contact_k * gamma;

            const Eigen::Vector3d interpenetration_velocity =
                0.5 * ((velocities.row(i) + velocities.row(i + 1))
                       - (velocities.row(j) + velocities.row(j + 1)))
                          .transpose();
            const double contact_damping_force =
                contact_nu * interpenetration_velocity.dot(distance_vector);

            // No equilibrium-force term here, unlike rod-to-rod contact.
            const Eigen::Vector3d net_contact_force =
                (0.5 * mask * (contact_damping_force + contact_force)) * distance_vector;

            if (i == count - 1)
            {
                external_forces.row(i) -= (net_contact_force * 4.0 / 3.0).transpose();
                external_forces.row(i + 1) -= (net_contact_force * 2.0 / 3.0).transpose();
            }
            else
            {
                external_forces.row(i) -= net_contact_force.transpose();
                external_forces.row(i + 1) -= net_contact_force.transpose();
            }

            if (j == 0)
            {
                external_forces.row(j) += (net_contact_force * 2.0 / 3.0).transpose();
                external_forces.row(j + 1) += (net_contact_force * 4.0 / 3.0).transpose();
            }
            else
            {
                external_forces.row(j) += net_contact_force.transpose();
                external_forces.row(j + 1) += net_contact_force.transpose();
            }
        }
    }
}

namespace {

/**
 * @brief The friction a slipping contact adds, capped by a Coulomb limit.
 *
 * Shared by the rod-to-cylinder and rod-to-sphere kernels, which compute it
 * identically.
 */
Eigen::Vector3d slip_friction_force(
    const Eigen::Vector3d& interpenetration_velocity,
    const Eigen::Vector3d& normal_interpenetration_velocity,
    const Eigen::Vector3d& net_contact_force,
    double velocity_damping_coefficient,
    double friction_coefficient
)
{
    const Eigen::Vector3d slip_velocity =
        interpenetration_velocity - normal_interpenetration_velocity;
    const double slip_speed = slip_velocity.norm();
    const Eigen::Vector3d slip_direction = slip_velocity / (slip_speed + unit_epsilon);

    const double damping_in_slip = velocity_damping_coefficient * slip_speed;
    const double coulomb_limit = friction_coefficient * net_contact_force.norm();

    return -std::min(damping_in_slip, coulomb_limit) * slip_direction;
}

} // End anonymous namespace

void contact_forces_rod_cylinder(
    const Vector3DStack& element_center,
    const Vector3DStack& element_edge,
    const Eigen::Vector3d& cylinder_center,
    const Eigen::Vector3d& cylinder_tip,
    const Eigen::Vector3d& cylinder_edge,
    const Eigen::VectorXd& radii_sum,
    const Eigen::VectorXd& lengths_sum,
    const Vector3DStack& internal_forces_rod,
    Vector3DStack& external_forces_rod,
    Vector3DStack& external_forces_cylinder,
    Vector3DStack& external_torques_cylinder,
    const Eigen::Matrix3d& cylinder_frame,
    const Vector3DStack& velocities_rod,
    const Eigen::Vector3d& velocity_cylinder,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
)
{
    (void)internal_forces_rod;

    const Eigen::Index count = element_center.rows();
    Eigen::Vector3d total_forces = Eigen::Vector3d::Zero();
    Eigen::Vector3d total_torques = Eigen::Vector3d::Zero();

    for (Eigen::Index i = 0; i < count; ++i)
    {
        const Eigen::Vector3d center = element_center.row(i).transpose();
        if ((center - cylinder_tip).norm() >= radii_sum(i) + lengths_sum(i)) continue;

        const auto closest = math::minimum_distance_segment_segment(
            center, element_edge.row(i).transpose(), cylinder_tip, cylinder_edge
        );

        Eigen::Vector3d distance_vector = closest.distance_vector;
        const double distance = distance_vector.norm();
        // Exactly intersecting axes leave the normal undefined; see
        // contact_normal_tolerance.
        if (distance < contact_normal_tolerance) continue;
        distance_vector /= distance;

        const double gamma = radii_sum(i) - distance;
        if (gamma < contact_separation_tolerance) continue;

        const double mask = gamma > 0.0 ? 1.0 : 0.0;
        const Eigen::Vector3d contact_force = contact_k * gamma * distance_vector;

        const Eigen::Vector3d interpenetration_velocity =
            velocity_cylinder
            - 0.5 * (velocities_rod.row(i) + velocities_rod.row(i + 1)).transpose();
        const Eigen::Vector3d normal_interpenetration_velocity =
            interpenetration_velocity.dot(distance_vector) * distance_vector;
        const Eigen::Vector3d contact_damping_force =
            -contact_nu * normal_interpenetration_velocity;

        Eigen::Vector3d net_contact_force =
            0.5 * mask * (contact_damping_force + contact_force);
        net_contact_force += slip_friction_force(
            interpenetration_velocity, normal_interpenetration_velocity,
            net_contact_force, velocity_damping_coefficient, friction_coefficient);

        const Eigen::Vector3d moment_arm =
            closest.contact_point_two - cylinder_center;

        if (i == 0)
        {
            external_forces_rod.row(i) -= (2.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(i + 1) -= (4.0 / 3.0 * net_contact_force).transpose();
        }
        else if (i == count - 1)
        {
            external_forces_rod.row(i) -= (4.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(i + 1) -= (2.0 / 3.0 * net_contact_force).transpose();
        }
        else
        {
            external_forces_rod.row(i) -= net_contact_force.transpose();
            external_forces_rod.row(i + 1) -= net_contact_force.transpose();
        }

        total_forces += 2.0 * net_contact_force;
        total_torques += moment_arm.cross(2.0 * net_contact_force);
    }

    external_forces_cylinder.row(0) += total_forces.transpose();
    // Into the cylinder's own frame, which is where its torques live.
    external_torques_cylinder.row(0) += (cylinder_frame * total_torques).transpose();
}

void contact_forces_rod_sphere(
    const Vector3DStack& element_center,
    const Vector3DStack& element_edge,
    const Eigen::Vector3d& sphere_center,
    const Eigen::VectorXd& radii_sum,
    const Eigen::VectorXd& lengths_sum,
    Vector3DStack& external_forces_rod,
    Vector3DStack& external_forces_sphere,
    const Vector3DStack& velocities_rod,
    const Eigen::Vector3d& velocity_sphere,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
)
{
    const Eigen::Index count = element_center.rows();
    Eigen::Vector3d total_forces = Eigen::Vector3d::Zero();

    for (Eigen::Index i = 0; i < count; ++i)
    {
        const Eigen::Vector3d center = element_center.row(i).transpose();
        if ((center - sphere_center).norm() >= radii_sum(i) + lengths_sum(i)) continue;

        Eigen::Vector3d distance_vector =
            math::minimum_distance_segment_point(
                center, element_edge.row(i).transpose(), sphere_center
            ).distance_vector;
        const double distance = distance_vector.norm();
        // Exactly intersecting axes leave the normal undefined; see
        // contact_normal_tolerance.
        if (distance < contact_normal_tolerance) continue;
        distance_vector /= distance;

        const double gamma = radii_sum(i) - distance;
        if (gamma < contact_separation_tolerance) continue;

        const double mask = gamma > 0.0 ? 1.0 : 0.0;
        const Eigen::Vector3d contact_force = contact_k * gamma * distance_vector;

        const Eigen::Vector3d interpenetration_velocity =
            velocity_sphere
            - 0.5 * (velocities_rod.row(i) + velocities_rod.row(i + 1)).transpose();
        const Eigen::Vector3d normal_interpenetration_velocity =
            interpenetration_velocity.dot(distance_vector) * distance_vector;
        const Eigen::Vector3d contact_damping_force =
            -contact_nu * normal_interpenetration_velocity;

        Eigen::Vector3d net_contact_force =
            0.5 * mask * (contact_damping_force + contact_force);
        net_contact_force += slip_friction_force(
            interpenetration_velocity, normal_interpenetration_velocity,
            net_contact_force, velocity_damping_coefficient, friction_coefficient);

        if (i == 0)
        {
            external_forces_rod.row(i) -= (2.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(i + 1) -= (4.0 / 3.0 * net_contact_force).transpose();
        }
        else if (i == count - 1)
        {
            external_forces_rod.row(i) -= (4.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(i + 1) -= (2.0 / 3.0 * net_contact_force).transpose();
        }
        else
        {
            external_forces_rod.row(i) -= net_contact_force.transpose();
            external_forces_rod.row(i + 1) -= net_contact_force.transpose();
        }

        total_forces += 2.0 * net_contact_force;
    }

    // Force only: the reference applies no torque to a sphere, even though a
    // contact away from its centre has a moment arm.
    external_forces_sphere.row(0) += total_forces.transpose();
}

PlaneContactResult contact_forces_rod_plane(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double k,
    double nu,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& masses,
    const Vector3DStack& positions,
    const Vector3DStack& velocities,
    const Vector3DStack& internal_forces,
    Vector3DStack& external_forces
)
{
    const Vector3DStack element_total_forces =
        node_to_element_mass_or_force(internal_forces + external_forces);

    // Cancel whatever is already pressing each element into the surface, but
    // never pull an element that is being lifted away.
    const Eigen::VectorXd force_along_normal = element_total_forces * plane_normal;
    Vector3DStack forces_along_normal = force_along_normal * plane_normal.transpose();
    for (Eigen::Index idx = 0; idx < force_along_normal.size(); ++idx)
    {
        if (force_along_normal(idx) > 0.0) forces_along_normal.row(idx).setZero();
    }
    Vector3DStack plane_response_force = -forces_along_normal;

    const Vector3DStack element_position = math::row_average(positions);
    const Eigen::VectorXd distance_from_plane =
        (element_position.rowwise() - plane_origin.transpose()) * plane_normal;
    const Eigen::VectorXd penetration =
        (distance_from_plane - radii).cwiseMin(0.0);

    const Vector3DStack elastic_force =
        -k * (penetration * plane_normal.transpose());

    const Vector3DStack element_velocity =
        node_to_element_velocity(masses, velocities);
    const Eigen::VectorXd normal_velocity = element_velocity * plane_normal;
    const Vector3DStack damping_force =
        -nu * (normal_velocity * plane_normal.transpose());

    Vector3DStack total_response =
        plane_response_force + elastic_force + damping_force;

    PlaneContactResult result;
    result.out_of_contact =
        ((distance_from_plane - radii).array() > surface_tolerance);
    for (Eigen::Index idx = 0; idx < result.out_of_contact.size(); ++idx)
    {
        if (result.out_of_contact(idx))
        {
            plane_response_force.row(idx).setZero();
            total_response.row(idx).setZero();
        }
    }

    elements_to_nodes_inplace(total_response, external_forces);

    result.response_magnitude = math::row_norms(plane_response_force);
    return result;
}

void contact_forces_rod_plane_with_anisotropic_friction(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double slip_velocity_tol,
    double k,
    double nu,
    double kinetic_mu_forward,
    double kinetic_mu_backward,
    double kinetic_mu_sideways,
    double static_mu_forward,
    double static_mu_backward,
    double static_mu_sideways,
    const Eigen::VectorXd& radii,
    const Eigen::VectorXd& masses,
    const Vector3DStack& tangents,
    const Vector3DStack& positions,
    const Matrix3DStack& frames,
    const Vector3DStack& velocities,
    const Vector3DStack& angular_velocities,
    const Vector3DStack& internal_forces,
    Vector3DStack& external_forces,
    const Vector3DStack& internal_torques,
    Vector3DStack& external_torques
)
{
    const PlaneContactResult plane_result = contact_forces_rod_plane(
        plane_origin, plane_normal, k, nu, radii, masses, positions, velocities,
        internal_forces, external_forces);
    const Eigen::VectorXd& response_magnitude = plane_result.response_magnitude;
    const auto& out_of_contact = plane_result.out_of_contact;

    const Eigen::Index count = tangents.rows();

    // The rod's axis projected into the plane, which is the direction friction
    // resists along.
    const Eigen::VectorXd tangent_along_normal = tangents * plane_normal;
    Vector3DStack in_plane_tangent =
        tangents - tangent_along_normal * plane_normal.transpose();
    const Eigen::VectorXd in_plane_magnitude = math::row_norms(in_plane_tangent);
    Vector3DStack axial_direction(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        axial_direction.row(idx) =
            in_plane_tangent.row(idx) / (in_plane_magnitude(idx) + unit_epsilon);
    }

    const Vector3DStack element_velocity =
        node_to_element_velocity(masses, velocities);

    const Eigen::VectorXd speed_along_axis =
        math::batched_dot_product(element_velocity, axial_direction);
    Vector3DStack velocity_along_axis(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        velocity_along_axis.row(idx) = speed_along_axis(idx) * axial_direction.row(idx);
    }

    // Friction differs forward and backward, selected by the sign of travel.
    Eigen::VectorXd kinetic_mu(count);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const double sign = speed_along_axis(idx) > 0.0
            ? 1.0 : (speed_along_axis(idx) < 0.0 ? -1.0 : 0.0);
        kinetic_mu(idx) = 0.5 * (kinetic_mu_forward * (1.0 + sign)
                                 + kinetic_mu_backward * (1.0 - sign));
    }

    const Eigen::VectorXd slip_along_axis =
        find_slipping_elements(velocity_along_axis, slip_velocity_tol);

    // Rolling is across the axis, in the plane.
    Vector3DStack rolling_direction(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        rolling_direction.row(idx) =
            Eigen::Vector3d(axial_direction.row(idx).transpose().cross(plane_normal))
                .transpose();
    }

    Vector3DStack torque_arm(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        torque_arm.row(idx) = (-plane_normal * radii(idx)).transpose();
    }

    const Eigen::VectorXd velocity_along_rolling =
        math::batched_dot_product(element_velocity, rolling_direction);

    // Surface speed contributed by the element spinning about its own axis.
    Vector3DStack rotation_velocity(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const Eigen::Matrix3d& frame = frames[static_cast<std::size_t>(idx)];
        const Eigen::Vector3d arm_in_frame = frame * torque_arm.row(idx).transpose();
        const Eigen::Vector3d omega = angular_velocities.row(idx).transpose();
        rotation_velocity.row(idx) =
            (frame.transpose() * omega.cross(arm_in_frame)).transpose();
    }
    const Eigen::VectorXd rotation_along_rolling =
        math::batched_dot_product(rotation_velocity, rolling_direction);

    const Eigen::VectorXd slip_speed_rolling =
        velocity_along_rolling + rotation_along_rolling;
    Vector3DStack slip_velocity_rolling(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        slip_velocity_rolling.row(idx) =
            slip_speed_rolling(idx) * rolling_direction.row(idx);
    }
    const Eigen::VectorXd slip_along_rolling =
        find_slipping_elements(slip_velocity_rolling, slip_velocity_tol);

    Vector3DStack unitized_total_velocity = slip_velocity_rolling + velocity_along_axis;
    const Eigen::VectorXd total_speed =
        math::row_norms(Vector3DStack(unitized_total_velocity.array() + unit_epsilon));
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        unitized_total_velocity.row(idx) /= total_speed(idx);
    }

    // --- kinetic friction, along the axis -----------------------------------
    const Eigen::VectorXd unit_dot_axial =
        math::batched_dot_product(unitized_total_velocity, axial_direction);
    Vector3DStack kinetic_axial(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        kinetic_axial.row(idx) = -((1.0 - slip_along_axis(idx)) * kinetic_mu(idx)
                                   * response_magnitude(idx) * unit_dot_axial(idx))
            * axial_direction.row(idx);
        if (out_of_contact(idx)) kinetic_axial.row(idx).setZero();
    }
    elements_to_nodes_inplace(kinetic_axial, external_forces);

    // --- kinetic friction, rolling ------------------------------------------
    const Eigen::VectorXd unit_dot_rolling =
        math::batched_dot_product(unitized_total_velocity, rolling_direction);
    Vector3DStack kinetic_rolling(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        kinetic_rolling.row(idx) = -((1.0 - slip_along_rolling(idx)) * kinetic_mu_sideways
                                     * response_magnitude(idx) * unit_dot_rolling(idx))
            * rolling_direction.row(idx);
        if (out_of_contact(idx)) kinetic_rolling.row(idx).setZero();
    }
    elements_to_nodes_inplace(kinetic_rolling, external_forces);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const Eigen::Matrix3d& frame = frames[static_cast<std::size_t>(idx)];
        const Eigen::Vector3d moment =
            Eigen::Vector3d(torque_arm.row(idx).transpose())
                .cross(Eigen::Vector3d(kinetic_rolling.row(idx).transpose()));
        external_torques.row(idx) += (frame * moment).transpose();
    }

    // --- static friction, along the axis ------------------------------------
    // Recomputed from the forces as they now stand, including the kinetic
    // terms just added.
    const Vector3DStack element_total_forces =
        node_to_element_mass_or_force(internal_forces + external_forces);
    const Eigen::VectorXd force_along_axis =
        math::batched_dot_product(element_total_forces, axial_direction);

    Vector3DStack static_axial(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const double sign = force_along_axis(idx) > 0.0
            ? 1.0 : (force_along_axis(idx) < 0.0 ? -1.0 : 0.0);
        const double static_mu = 0.5 * (static_mu_forward * (1.0 + sign)
                                        + static_mu_backward * (1.0 - sign));
        const double max_friction =
            slip_along_axis(idx) * static_mu * response_magnitude(idx);
        static_axial.row(idx) =
            -(std::min(std::abs(force_along_axis(idx)), max_friction) * sign)
            * axial_direction.row(idx);
        if (out_of_contact(idx)) static_axial.row(idx).setZero();
    }
    elements_to_nodes_inplace(static_axial, external_forces);

    // --- static friction, rolling -------------------------------------------
    Vector3DStack total_torques(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const Eigen::Matrix3d& frame = frames[static_cast<std::size_t>(idx)];
        const Eigen::Vector3d torque =
            (internal_torques.row(idx) + external_torques.row(idx)).transpose();
        total_torques.row(idx) = (frame.transpose() * torque).transpose();
    }
    const Eigen::VectorXd torque_along_axis =
        math::batched_dot_product(total_torques, axial_direction);
    const Eigen::VectorXd force_along_rolling =
        math::batched_dot_product(element_total_forces, rolling_direction);

    Vector3DStack static_rolling(count, 3);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        nice_assert(radii(idx) > 0.0, "Element radius must be positive");
        const double noslip_force =
            -((radii(idx) * force_along_rolling(idx) - 2.0 * torque_along_axis(idx))
              / 3.0 / radii(idx));
        const double max_friction =
            slip_along_rolling(idx) * static_mu_sideways * response_magnitude(idx);
        const double sign =
            noslip_force > 0.0 ? 1.0 : (noslip_force < 0.0 ? -1.0 : 0.0);
        static_rolling.row(idx) =
            (std::min(std::abs(noslip_force), max_friction) * sign)
            * rolling_direction.row(idx);
        if (out_of_contact(idx)) static_rolling.row(idx).setZero();
    }
    elements_to_nodes_inplace(static_rolling, external_forces);
    for (Eigen::Index idx = 0; idx < count; ++idx)
    {
        const Eigen::Matrix3d& frame = frames[static_cast<std::size_t>(idx)];
        const Eigen::Vector3d moment =
            Eigen::Vector3d(torque_arm.row(idx).transpose())
                .cross(Eigen::Vector3d(static_rolling.row(idx).transpose()));
        external_torques.row(idx) += (frame * moment).transpose();
    }
}

PlaneContactResult contact_forces_cylinder_plane(
    const Eigen::Vector3d& plane_origin,
    const Eigen::Vector3d& plane_normal,
    double k,
    double nu,
    double length,
    const Vector3DStack& positions,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces
)
{
    const Eigen::VectorXd force_along_normal = external_forces * plane_normal;
    Vector3DStack forces_along_normal = force_along_normal * plane_normal.transpose();
    for (Eigen::Index idx = 0; idx < force_along_normal.size(); ++idx)
    {
        if (force_along_normal(idx) > 0.0) forces_along_normal.row(idx).setZero();
    }
    Vector3DStack plane_response_force = -forces_along_normal;

    // The cylinder is treated as a point with a half-length clearance rather
    // than as an oriented capsule, so its tilt does not enter here.
    const Eigen::VectorXd distance_from_plane =
        (positions.rowwise() - plane_origin.transpose()) * plane_normal;
    const Eigen::VectorXd penetration =
        (distance_from_plane.array() - 0.5 * length).cwiseMin(0.0);

    const Vector3DStack elastic_force = -k * (penetration * plane_normal.transpose());
    const Eigen::VectorXd normal_velocity = velocities * plane_normal;
    const Vector3DStack damping_force =
        -nu * (normal_velocity * plane_normal.transpose());

    Vector3DStack total_response =
        plane_response_force + elastic_force + damping_force;

    PlaneContactResult result;
    result.out_of_contact =
        ((distance_from_plane.array() - 0.5 * length) > surface_tolerance);
    for (Eigen::Index idx = 0; idx < result.out_of_contact.size(); ++idx)
    {
        if (result.out_of_contact(idx))
        {
            plane_response_force.row(idx).setZero();
            total_response.row(idx).setZero();
        }
    }

    external_forces += total_response;

    result.response_magnitude = math::row_norms(plane_response_force);
    return result;
}
} // End namespace detail

// ---------------------------------------------------------------------------
// Contact rules
// ---------------------------------------------------------------------------

RodRodContact::RodRodContact(double k, double nu) : m_k(k), m_nu(nu)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
}

double RodRodContact::k() const {return m_k;}
double RodRodContact::nu() const {return m_nu;}

RodSelfContact::RodSelfContact(double k, double nu) : m_k(k), m_nu(nu)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
}

double RodSelfContact::k() const {return m_k;}
double RodSelfContact::nu() const {return m_nu;}

RodCylinderContact::RodCylinderContact(
    double k, double nu, double velocity_damping_coefficient, double friction_coefficient
) : m_k(k),
    m_nu(nu),
    m_velocity_damping_coefficient(velocity_damping_coefficient),
    m_friction_coefficient(friction_coefficient)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
    assert_non_negative(velocity_damping_coefficient, "velocity damping coefficient");
    assert_non_negative(friction_coefficient, "friction coefficient");
}

double RodCylinderContact::k() const {return m_k;}
double RodCylinderContact::nu() const {return m_nu;}

RodSphereContact::RodSphereContact(
    double k, double nu, double velocity_damping_coefficient, double friction_coefficient
) : m_k(k),
    m_nu(nu),
    m_velocity_damping_coefficient(velocity_damping_coefficient),
    m_friction_coefficient(friction_coefficient)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
    assert_non_negative(velocity_damping_coefficient, "velocity damping coefficient");
    assert_non_negative(friction_coefficient, "friction coefficient");
}

double RodSphereContact::k() const {return m_k;}
double RodSphereContact::nu() const {return m_nu;}

RodPlaneContact::RodPlaneContact(double k, double nu) : m_k(k), m_nu(nu)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
}

double RodPlaneContact::k() const {return m_k;}
double RodPlaneContact::nu() const {return m_nu;}

RodPlaneContactWithAnisotropicFriction::RodPlaneContactWithAnisotropicFriction(
    double k,
    double nu,
    double slip_velocity_tol,
    const Eigen::Vector3d& static_mu,
    const Eigen::Vector3d& kinetic_mu
) : m_k(k),
    m_nu(nu),
    m_slip_velocity_tol(slip_velocity_tol),
    m_static_mu(static_mu),
    m_kinetic_mu(kinetic_mu)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
    assert_positive(slip_velocity_tol, "slip velocity tolerance");
    nice_assert(
        static_mu.array().isFinite().all() and (static_mu.array() >= 0.0).all(),
        "Static friction coefficients must be finite and not negative"
    );
    nice_assert(
        kinetic_mu.array().isFinite().all() and (kinetic_mu.array() >= 0.0).all(),
        "Kinetic friction coefficients must be finite and not negative"
    );
}

double RodPlaneContactWithAnisotropicFriction::k() const {return m_k;}
double RodPlaneContactWithAnisotropicFriction::nu() const {return m_nu;}

const Eigen::Vector3d& RodPlaneContactWithAnisotropicFriction::static_mu() const
{
    return m_static_mu;
}

const Eigen::Vector3d& RodPlaneContactWithAnisotropicFriction::kinetic_mu() const
{
    return m_kinetic_mu;
}

CylinderPlaneContact::CylinderPlaneContact(double k, double nu) : m_k(k), m_nu(nu)
{
    assert_positive(k, "contact stiffness");
    assert_non_negative(nu, "contact damping");
}

double CylinderPlaneContact::k() const {return m_k;}
double CylinderPlaneContact::nu() const {return m_nu;}
} // End namespace cosserat::physics
