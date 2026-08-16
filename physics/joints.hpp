#pragma once

/**
 * @file joints.hpp
 * @brief Joints connecting two Cosserat rod or rigid-body systems.
 *
 * Each joint is an independent value type constructed from its own parameters
 * alone. It knows nothing about any particular body: the two systems and the
 * indices at which they are connected are supplied to @c apply_forces and
 * @c apply_torques, so one joint may be reused across any number of compatible
 * pairs.
 *
 * All joints restrain relative translation with a spring-damper acting between
 * the two connected nodes. @ref HingeJoint and @ref FixedJoint additionally
 * restrain relative rotation, differing in how much of it they remove: a hinge
 * leaves one axis free, whereas a fixed joint pins the full relative
 * orientation to a prescribed rest rotation.
 *
 * @note Indices are interpreted against the stack they address, so the same
 *       index means a node in @c apply_forces and an element in
 *       @c apply_torques. Negative indices count back from the end, which is
 *       why -1 correctly names the last node and the last element even though
 *       those are different absolute positions.
 *
 * @note This header includes @c constraints.hpp for @c resolve_index alone.
 *       That helper is not specific to boundary conditions and would sit more
 *       naturally in a shared indexing header.
 *
 * @see Zhang et al., Nature Communications (2019), for the joint formulation.
 */

#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "math/indexing.hpp"
#include "math/linalg.hpp"
#include "math/types.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

/**
 * @brief A system whose connected node can exchange forces through a joint.
 *
 * Positions and velocities are read at the connection node, and the resulting
 * force is accumulated into the external force stack. All three carry one row
 * per node.
 */
template<typename T>
concept ForceJointableSystem = requires(T sys)
{
    {sys.positions()} -> std::same_as<Vector3DStack&>;
    {sys.velocities()} -> std::same_as<Vector3DStack&>;
    {sys.external_forces()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A system whose connected element can exchange torques through a joint.
 *
 * Frames and angular velocities are read at the connection element, and the
 * resulting torque is accumulated into the external torque stack. Angular
 * velocities and torques are expressed in the material frame.
 */
template<typename T>
concept TorqueJointableSystem = requires(T sys)
{
    {sys.frames()} -> std::same_as<const Matrix3DStack&>;
    {sys.angular_velocities()} -> std::same_as<Vector3DStack&>;
    {sys.external_torques()} -> std::same_as<Vector3DStack&>;
};

/** @brief A system that can be connected by any of the joints here. */
template<typename T>
concept JointableSystem = ForceJointableSystem<T> and TorqueJointableSystem<T>;

/**
 * @brief Smallest direction vector length treated as defining an axis.
 */
inline constexpr double joint_direction_tolerance = 1e-12;

/**
 * @brief Largest departure from orthonormality tolerated in a rest rotation.
 */
inline constexpr double joint_rotation_tolerance = 1e-8;

using math::inverse_rotate;

// ---------------------------------------------------------------------------
// Free joint
// ---------------------------------------------------------------------------

/**
 * @brief Restrains relative translation only, leaving rotation free.
 *
 * A spring-damper acts along the vector separating the two connected nodes:
 *
 * @f[ \mathbf{F} = k (\mathbf{x}_2 - \mathbf{x}_1)
 *                + \nu (\mathbf{v}_2 - \mathbf{v}_1) @f]
 *
 * The force is added to system one and subtracted from system two, so the pair
 * exchanges equal and opposite forces. Relative rotation is unconstrained,
 * which makes this a spherical or ball joint.
 *
 * @see HingeJoint, FixedJoint for joints that also restrain rotation.
 */
struct FreeJoint
{
private: // Members
    double m_stiffness;
    double m_damping;

public: // Methods
    /**
     * @brief Builds the joint from its spring-damper coefficients.
     * @param stiffness Translational stiffness; finite and non-negative.
     * @param damping Translational damping; finite and non-negative.
     */
    FreeJoint(double stiffness, double damping);

    /**
     * @brief Adds equal and opposite spring-damper forces at the two nodes.
     * @tparam SystemOne Any @ref ForceJointableSystem.
     * @tparam SystemTwo Any @ref ForceJointableSystem.
     * @param system_one First connected system.
     * @param index_one Connection node on the first system; may be negative.
     * @param system_two Second connected system.
     * @param index_two Connection node on the second system; may be negative.
     */
    template<ForceJointableSystem SystemOne, ForceJointableSystem SystemTwo>
    void apply_forces(
        SystemOne& system_one,
        std::int64_t index_one,
        SystemTwo& system_two,
        std::int64_t index_two,
        double
    ) const
    {
        Vector3DStack& positions_one = system_one.positions();
        Vector3DStack& positions_two = system_two.positions();
        Vector3DStack& forces_one = system_one.external_forces();
        Vector3DStack& forces_two = system_two.external_forces();

        utils::nice_assert(
            forces_one.rows() == positions_one.rows()
            and forces_two.rows() == positions_two.rows(),
            "External force rows must match position rows"
        );

        const Eigen::Index one = resolve_index(index_one, positions_one.rows());
        const Eigen::Index two = resolve_index(index_two, positions_two.rows());

        const Eigen::RowVector3d separation =
            positions_two.row(two) - positions_one.row(one);
        const Eigen::RowVector3d relative_velocity =
            system_two.velocities().row(two) - system_one.velocities().row(one);

        const Eigen::RowVector3d contact_force =
            m_stiffness * separation + m_damping * relative_velocity;

        forces_one.row(one) += contact_force;
        forces_two.row(two) -= contact_force;
    }

    /** @brief Does nothing; a free joint leaves relative rotation free. */
    template<typename SystemOne, typename SystemTwo>
    void apply_torques(SystemOne&&, std::int64_t, SystemTwo&&, std::int64_t, double) const
    {}

    /** @brief Translational stiffness supplied at construction. */
    double stiffness() const;
    /** @brief Translational damping supplied at construction. */
    double damping() const;
};

/** @brief Alias for @ref FreeJoint. */
using BallJoint = FreeJoint;
/** @brief Alias for @ref FreeJoint. */
using SphericalJoint = FreeJoint;

// ---------------------------------------------------------------------------
// Hinge joint
// ---------------------------------------------------------------------------

/**
 * @brief Restrains relative translation and rotation about all but one axis.
 *
 * Translation is handled exactly as in @ref FreeJoint. Rotation is restrained
 * by penalising any component of the second system's tangent director that
 * leaves the constraint plane: the out-of-plane part of the tangent is
 * projected onto the plane normal, and a restoring torque proportional to
 * their cross product drives the tangent back into the plane. Rotation within
 * the plane is left free, which is what makes the joint a hinge.
 *
 * Equal and opposite torques are applied to the two systems after being
 * rotated into their respective material frames.
 */
struct HingeJoint
{
private: // Members
    FreeJoint m_free_joint;
    double m_rotational_stiffness;
    Eigen::Vector3d m_normal_direction;

public: // Methods
    /**
     * @brief Builds the joint.
     * @param stiffness Translational stiffness; finite and non-negative.
     * @param damping Translational damping; finite and non-negative.
     * @param rotational_stiffness Restoring torque stiffness; finite and
     *        non-negative.
     * @param normal_direction Normal of the plane rotation is confined to. It
     *        is normalised internally and must be longer than
     *        @ref joint_direction_tolerance.
     */
    HingeJoint(
        double stiffness,
        double damping,
        double rotational_stiffness,
        Eigen::Vector3d normal_direction
    );

    /**
     * @brief Adds the same spring-damper forces as @ref FreeJoint.
     * @tparam SystemOne Any @ref ForceJointableSystem.
     * @tparam SystemTwo Any @ref ForceJointableSystem.
     * @param system_one First connected system.
     * @param index_one Connection node on the first system; may be negative.
     * @param system_two Second connected system.
     * @param index_two Connection node on the second system; may be negative.
     * @param time Current simulation time.
     */
    template<ForceJointableSystem SystemOne, ForceJointableSystem SystemTwo>
    void apply_forces(
        SystemOne& system_one,
        std::int64_t index_one,
        SystemTwo& system_two,
        std::int64_t index_two,
        double time
    ) const
    {
        m_free_joint.apply_forces(system_one, index_one, system_two, index_two, time);
    }

    /**
     * @brief Adds restoring torques driving the tangent into the hinge plane.
     * @tparam SystemOne Any @ref TorqueJointableSystem.
     * @tparam SystemTwo Any @ref TorqueJointableSystem.
     * @param system_one First connected system.
     * @param index_one Connection element on the first system; may be negative.
     * @param system_two Second connected system.
     * @param index_two Connection element on the second system; may be
     *        negative.
     */
    template<TorqueJointableSystem SystemOne, TorqueJointableSystem SystemTwo>
    void apply_torques(
        SystemOne& system_one,
        std::int64_t index_one,
        SystemTwo& system_two,
        std::int64_t index_two,
        double
    ) const
    {
        const Matrix3DStack& frames_one = system_one.frames();
        const Matrix3DStack& frames_two = system_two.frames();
        Vector3DStack& torques_one = system_one.external_torques();
        Vector3DStack& torques_two = system_two.external_torques();

        utils::nice_assert(
            static_cast<Eigen::Index>(frames_one.size()) == torques_one.rows()
            and static_cast<Eigen::Index>(frames_two.size()) == torques_two.rows(),
            "External torque rows must match frame counts"
        );

        const Eigen::Index one = resolve_index(index_one, torques_one.rows());
        const Eigen::Index two = resolve_index(index_two, torques_two.rows());

        // Row 2 of the director matrix is the element tangent in lab frame.
        const Eigen::Vector3d tangent = frames_two[two].row(2).transpose();
        const Eigen::Vector3d out_of_plane =
            -tangent.dot(m_normal_direction) * m_normal_direction;
        const Eigen::Vector3d torque =
            m_rotational_stiffness * tangent.cross(out_of_plane);

        torques_one.row(one) -= (frames_one[one] * torque).transpose();
        torques_two.row(two) += (frames_two[two] * torque).transpose();
    }

    /** @brief Translational stiffness supplied at construction. */
    double stiffness() const;
    /** @brief Translational damping supplied at construction. */
    double damping() const;
    /** @brief Restoring torque stiffness supplied at construction. */
    double rotational_stiffness() const;
    /** @brief Unit normal of the plane rotation is confined to. */
    const Eigen::Vector3d& normal_direction() const;
};

// ---------------------------------------------------------------------------
// Fixed joint
// ---------------------------------------------------------------------------

/**
 * @brief Restrains relative translation and the full relative rotation.
 *
 * Translation is handled exactly as in @ref FreeJoint. Rotation is restrained
 * by a rotational spring-damper acting in the inertial frame: the deviation of
 * the current relative rotation from a prescribed rest rotation is converted
 * to a rotation vector, damped by the difference in angular velocity, and
 * applied as equal and opposite torques.
 *
 * Rather than aligning the two systems' frames directly, a desired rest
 * rotation from system one to system two is enforced. Supplying the identity
 * recovers direct alignment.
 */
struct FixedJoint
{
private: // Members
    FreeJoint m_free_joint;
    double m_rotational_stiffness;
    double m_rotational_damping;
    Eigen::Matrix3d m_rest_rotation_matrix;

public: // Methods
    /**
     * @brief Builds the joint.
     * @param stiffness Translational stiffness; finite and non-negative.
     * @param damping Translational damping; finite and non-negative.
     * @param rotational_stiffness Rotational stiffness; finite and
     *        non-negative.
     * @param rotational_damping Rotational damping; finite and non-negative.
     * @param rest_rotation_matrix Desired rotation from system one to system
     *        two at the connected elements. Must be a rotation matrix to
     *        within @ref joint_rotation_tolerance; pass the identity to align
     *        the two frames directly.
     */
    FixedJoint(
        double stiffness,
        double damping,
        double rotational_stiffness,
        double rotational_damping,
        Eigen::Matrix3d rest_rotation_matrix
    );

    /**
     * @brief Adds the same spring-damper forces as @ref FreeJoint.
     * @tparam SystemOne Any @ref ForceJointableSystem.
     * @tparam SystemTwo Any @ref ForceJointableSystem.
     * @param system_one First connected system.
     * @param index_one Connection node on the first system; may be negative.
     * @param system_two Second connected system.
     * @param index_two Connection node on the second system; may be negative.
     * @param time Current simulation time.
     */
    template<ForceJointableSystem SystemOne, ForceJointableSystem SystemTwo>
    void apply_forces(
        SystemOne& system_one,
        std::int64_t index_one,
        SystemTwo& system_two,
        std::int64_t index_two,
        double time
    ) const
    {
        m_free_joint.apply_forces(system_one, index_one, system_two, index_two, time);
    }

    /**
     * @brief Adds torques driving the relative rotation to its rest value.
     * @tparam SystemOne Any @ref TorqueJointableSystem.
     * @tparam SystemTwo Any @ref TorqueJointableSystem.
     * @param system_one First connected system.
     * @param index_one Connection element on the first system; may be negative.
     * @param system_two Second connected system.
     * @param index_two Connection element on the second system; may be
     *        negative.
     */
    template<TorqueJointableSystem SystemOne, TorqueJointableSystem SystemTwo>
    void apply_torques(
        SystemOne& system_one,
        std::int64_t index_one,
        SystemTwo& system_two,
        std::int64_t index_two,
        double
    ) const
    {
        const Matrix3DStack& frames_one = system_one.frames();
        const Matrix3DStack& frames_two = system_two.frames();
        Vector3DStack& torques_one = system_one.external_torques();
        Vector3DStack& torques_two = system_two.external_torques();

        utils::nice_assert(
            static_cast<Eigen::Index>(frames_one.size()) == torques_one.rows()
            and static_cast<Eigen::Index>(frames_two.size()) == torques_two.rows(),
            "External torque rows must match frame counts"
        );

        const Eigen::Index one = resolve_index(index_one, torques_one.rows());
        const Eigen::Index two = resolve_index(index_two, torques_two.rows());

        const Eigen::Matrix3d& director_one = frames_one[one];
        const Eigen::Matrix3d& director_two = frames_two[two];

        // Relative rotation from system one to system two, then how far that
        // sits from the rest rotation we want to hold.
        const Eigen::Matrix3d relative_rotation =
            director_one * director_two.transpose();
        const Eigen::Matrix3d deviation =
            relative_rotation.transpose() * m_rest_rotation_matrix;

        const Eigen::Vector3d rotation_vector =
            inverse_rotate(Eigen::Matrix3d::Identity(), deviation.transpose());
        const Eigen::Vector3d rotation_vector_inertial =
            director_two.transpose() * rotation_vector;

        // Angular velocity difference, taken after moving both into the
        // inertial frame so the two material frames do not bias it.
        const Eigen::Vector3d deviation_omega =
            director_two.transpose()
                * system_two.angular_velocities().row(two).transpose()
            - director_one.transpose()
                * system_one.angular_velocities().row(one).transpose();

        const Eigen::Vector3d torque =
            m_rotational_stiffness * rotation_vector_inertial
            - m_rotational_damping * deviation_omega;

        torques_one.row(one) -= (director_one * torque).transpose();
        torques_two.row(two) += (director_two * torque).transpose();
    }

    /** @brief Translational stiffness supplied at construction. */
    double stiffness() const;
    /** @brief Translational damping supplied at construction. */
    double damping() const;
    /** @brief Rotational stiffness supplied at construction. */
    double rotational_stiffness() const;
    /** @brief Rotational damping supplied at construction. */
    double rotational_damping() const;
    /** @brief Desired rotation from system one to system two. */
    const Eigen::Matrix3d& rest_rotation_matrix() const;
};

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

/** @brief Any one of the joints, held by value. */
using JointVariant = std::variant<FreeJoint, HingeJoint, FixedJoint>;

/**
 * @brief Fails if the held joint cannot connect the given pair of systems.
 *
 * Each joint restricts its entry points to the concepts it needs, so probing
 * whether both calls are well formed decides compatibility without naming the
 * concepts here.
 *
 * @tparam SystemOne Type of the first system.
 * @tparam SystemTwo Type of the second system.
 * @param joint_var Joint to check.
 * @param system_one First system; not modified.
 * @param system_two Second system; not modified.
 */
template<typename SystemOne, typename SystemTwo>
void validate(JointVariant& joint_var, SystemOne& system_one, SystemTwo& system_two)
{
    std::visit([&](auto& joint)
    {
        const std::int64_t index = 0;
        const double time = 0.0;
        if constexpr (
            not requires {
                joint.apply_forces(system_one, index, system_two, index, time);
                joint.apply_torques(system_one, index, system_two, index, time);
            }
        )
        {
            utils::nice_assert(false, "Joint is incompatible with these systems");
        }
    }, joint_var);
}

/**
 * @brief Applies the held joint's forces to the connected nodes.
 * @tparam SystemOne Type of the first system.
 * @tparam SystemTwo Type of the second system.
 * @param joint_var Joint to apply.
 * @param system_one First connected system.
 * @param index_one Connection node on the first system.
 * @param system_two Second connected system.
 * @param index_two Connection node on the second system.
 * @param time Current simulation time.
 */
template<typename SystemOne, typename SystemTwo>
void apply_forces(
    JointVariant& joint_var,
    SystemOne& system_one,
    std::int64_t index_one,
    SystemTwo& system_two,
    std::int64_t index_two,
    double time
)
{
    std::visit([&](auto& joint)
    {
        if constexpr (
            requires {joint.apply_forces(system_one, index_one, system_two, index_two, time);}
        )
        {
            joint.apply_forces(system_one, index_one, system_two, index_two, time);
        }
        else utils::nice_assert(false, "Joint is incompatible with these systems");
    }, joint_var);
}

/**
 * @brief Applies the held joint's torques to the connected elements.
 * @tparam SystemOne Type of the first system.
 * @tparam SystemTwo Type of the second system.
 * @param joint_var Joint to apply.
 * @param system_one First connected system.
 * @param index_one Connection element on the first system.
 * @param system_two Second connected system.
 * @param index_two Connection element on the second system.
 * @param time Current simulation time.
 */
template<typename SystemOne, typename SystemTwo>
void apply_torques(
    JointVariant& joint_var,
    SystemOne& system_one,
    std::int64_t index_one,
    SystemTwo& system_two,
    std::int64_t index_two,
    double time
)
{
    std::visit([&](auto& joint)
    {
        if constexpr (
            requires {joint.apply_torques(system_one, index_one, system_two, index_two, time);}
        )
        {
            joint.apply_torques(system_one, index_one, system_two, index_two, time);
        }
        else utils::nice_assert(false, "Joint is incompatible with these systems");
    }, joint_var);
}
} // End namespace cosserat::physics
