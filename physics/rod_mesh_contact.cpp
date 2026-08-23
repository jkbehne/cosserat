#include "physics/rod_mesh_contact.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "utils/assertions.hpp"

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/** @brief Guard against dividing by a vanishing slip speed. */
constexpr double unit_epsilon = 1e-14;

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

/**
 * @brief The friction a slipping contact adds, capped by a Coulomb limit.
 *
 * The same regularized model the rod-to-cylinder and rod-to-sphere kernels
 * use, so a rod sliding on a mesh resists exactly as it would on a cylinder.
 * It needs only a normal, which is why swapping in a field gradient works
 * unchanged.
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

namespace detail {

MeshContactSample march_element_against_field(
    const math::FieldQuery& query,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& finish,
    double radius
)
{
    MeshContactSample best;
    best.penetration = -std::numeric_limits<double>::infinity();

    const Eigen::Vector3d edge = finish - start;
    const double length = edge.norm();
    nice_assert(length > 0.0, "a rod element must have positive length");

    double parameter = 0.0;
    while (parameter <= 1.0)
    {
        const Eigen::Vector3d point = start + parameter * edge;
        const math::SignedDistance sample = query(point);
        ++best.queries;

        const double penetration = radius - sample.distance;
        if (penetration > best.penetration)
        {
            best.penetration = penetration;
            best.parameter = parameter;
            best.point = point;
            best.normal = sample.gradient;
        }

        if (penetration >= mesh_separation_tolerance)
        {
            // Inside a contact region: scan on at a fixed rate so the deepest
            // point is found, not merely the first one touched.
            parameter += march_fine_step;
        }
        else
        {
            // Clear of the surface. A distance field is 1-Lipschitz, so no
            // surface lies within (distance - radius) of here and stepping
            // that far cannot skip a contact. See the header notes.
            const double safe = march_safety_factor * (sample.distance - radius) / length;
            parameter += std::max(safe, march_minimum_step);
        }
    }

    best.in_contact = best.penetration >= mesh_separation_tolerance;
    return best;
}

std::int64_t contact_forces_rod_field(
    const Vector3DStack& positions,
    const Eigen::VectorXd& radii,
    const Vector3DStack& velocities,
    Vector3DStack& external_forces_rod,
    const math::FieldQuery& query,
    const Eigen::AlignedBox3d& domain,
    const Eigen::Vector3d& body_center,
    const Eigen::Matrix3d& body_frame,
    const Eigen::Vector3d& body_velocity,
    const Eigen::Vector3d& body_angular_velocity,
    Vector3DStack& external_forces_body,
    Vector3DStack& external_torques_body,
    double contact_k,
    double contact_nu,
    double velocity_damping_coefficient,
    double friction_coefficient
)
{
    const Eigen::Index count = radii.size();
    nice_assert(
        positions.rows() == count + 1, "a rod needs one more node than elements"
    );
    nice_assert(
        velocities.rows() == count + 1, "velocities must be given per node"
    );

    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d total_torque = Eigen::Vector3d::Zero();
    std::int64_t queries = 0;

    // The field lives in body coordinates and never moves with the body, so
    // the rod is carried into that frame rather than the field out of it.
    const auto to_body = [&](const Eigen::Vector3d& lab)
    {
        return Eigen::Vector3d(body_frame * (lab - body_center));
    };

    for (Eigen::Index element = 0; element < count; ++element)
    {
        const Eigen::Vector3d start = to_body(positions.row(element).transpose());
        const Eigen::Vector3d finish = to_body(positions.row(element + 1).transpose());

        // The domain is finite, so a query outside it is undefined rather than
        // far away. Growing the box by the radius keeps an element that only
        // reaches in by its thickness. This doubles as the broad phase.
        Eigen::AlignedBox3d padded = domain;
        const Eigen::Vector3d pad = Eigen::Vector3d::Constant(radii(element));
        padded.min() -= pad;
        padded.max() += pad;
        Eigen::AlignedBox3d element_box;
        element_box.extend(start);
        element_box.extend(finish);
        if (not padded.intersects(element_box)) continue;

        const MeshContactSample sample =
            march_element_against_field(query, start, finish, radii(element));
        queries += sample.queries;
        if (not sample.in_contact) continue;

        // On the medial axis the nearest surface point is not unique and the
        // gradient collapses, so there is no direction to push along.
        const double gradient_length = sample.normal.norm();
        if (gradient_length < mesh_gradient_tolerance) continue;

        // Outward from the surface, in the lab frame. The rod is pushed along
        // this; the body along its negation.
        const Eigen::Vector3d outward_body = sample.normal / gradient_length;
        const Eigen::Vector3d outward = body_frame.transpose() * outward_body;

        // Expressed like the other kernels, which point from the rod toward
        // the other body so the sign conventions below match theirs.
        const Eigen::Vector3d toward_body = -outward;

        const double mask = sample.penetration > 0.0 ? 1.0 : 0.0;
        const Eigen::Vector3d contact_force = contact_k * sample.penetration * toward_body;

        // Velocity of the body's material at the contact point: its linear
        // velocity plus the spin about its centre. Angular velocity is stored
        // in body coordinates, so the cross product is taken there.
        const Eigen::Vector3d contact_point_lab =
            body_center + body_frame.transpose() * sample.point;
        const Eigen::Vector3d moment_arm = contact_point_lab - body_center;
        const Eigen::Vector3d spin_velocity = body_frame.transpose()
            * body_angular_velocity.cross(body_frame * moment_arm);
        const Eigen::Vector3d body_point_velocity = body_velocity + spin_velocity;

        const Eigen::Vector3d rod_velocity =
            0.5 * (velocities.row(element) + velocities.row(element + 1)).transpose();
        const Eigen::Vector3d interpenetration_velocity = body_point_velocity - rod_velocity;
        const Eigen::Vector3d normal_interpenetration_velocity =
            interpenetration_velocity.dot(toward_body) * toward_body;
        const Eigen::Vector3d contact_damping_force =
            -contact_nu * normal_interpenetration_velocity;

        Eigen::Vector3d net_contact_force =
            0.5 * mask * (contact_damping_force + contact_force);
        net_contact_force += slip_friction_force(
            interpenetration_velocity, normal_interpenetration_velocity, net_contact_force,
            velocity_damping_coefficient, friction_coefficient
        );

        // Split onto the element's two nodes, weighting the ends as the other
        // contact kernels do so a rod feels the same total wherever it touches.
        if (element == 0)
        {
            external_forces_rod.row(element) -= (2.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(element + 1) -= (4.0 / 3.0 * net_contact_force).transpose();
        }
        else if (element == count - 1)
        {
            external_forces_rod.row(element) -= (4.0 / 3.0 * net_contact_force).transpose();
            external_forces_rod.row(element + 1) -= (2.0 / 3.0 * net_contact_force).transpose();
        }
        else
        {
            external_forces_rod.row(element) -= net_contact_force.transpose();
            external_forces_rod.row(element + 1) -= net_contact_force.transpose();
        }

        total_force += 2.0 * net_contact_force;
        total_torque += moment_arm.cross(2.0 * net_contact_force);
    }

    external_forces_body.row(0) += total_force.transpose();
    // Torques on a rigid body are held in its own frame.
    external_torques_body.row(0) += (body_frame * total_torque).transpose();
    return queries;
}
} // End namespace detail

RodMeshContact::RodMeshContact(
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

double RodMeshContact::k() const {return m_k;}
double RodMeshContact::nu() const {return m_nu;}

double RodMeshContact::velocity_damping_coefficient() const
{
    return m_velocity_damping_coefficient;
}

double RodMeshContact::friction_coefficient() const {return m_friction_coefficient;}
} // End namespace cosserat::physics
