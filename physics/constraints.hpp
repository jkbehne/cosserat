#pragma once

/**
 * @file constraints.hpp
 * @brief Boundary conditions for Cosserat rod systems.
 *
 * Each constraint is an independent value type constructed from its own
 * parameters alone; it knows nothing about any particular body and reads
 * everything it needs from the system handed to @c constrain_values or
 * @c constrain_rates. One constraint may therefore be applied to any number of
 * compatible systems.
 *
 * A constraint is applied through two entry points, mirroring the two stages of
 * a timestep:
 *
 * - @c constrain_values pins configuration, meaning node positions and element
 *   directors.
 * - @c constrain_rates pins the corresponding time derivatives, meaning
 *   translational and angular velocities.
 *
 * Constraining a position is equivalent to removing translational degrees of
 * freedom; constraining a director removes rotational degrees of freedom.
 *
 * @note Constraints write to element frames, so they need mutable access to
 *       the data that @c forces.hpp reads through its const @c frames()
 *       accessor. A rod satisfying both headers therefore exposes the same
 *       storage twice: @c frames() returning a const reference for readers,
 *       and @c mutable_frames() returning a mutable one for writers.
 */

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "math/indexing.hpp"
#include "math/types.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

/**
 * @brief A system whose node positions and velocities can be constrained.
 *
 * Both stacks carry one row per node.
 */
template<typename T>
concept PositionConstrainableSystem = requires(T sys)
{
    {sys.positions()} -> std::same_as<Vector3DStack&>;
    {sys.velocities()} -> std::same_as<Vector3DStack&>;
};

/**
 * @brief A system whose element directors and angular velocities can be
 *        constrained.
 *
 * Directors carry one rotation matrix per element, and angular velocities one
 * row per element. Angular velocities are expressed in the material frame.
 */
template<typename T>
concept DirectorConstrainableSystem = requires(T sys)
{
    {sys.mutable_frames()} -> std::same_as<Matrix3DStack&>;
    {sys.angular_velocities()} -> std::same_as<Vector3DStack&>;
};

/** @brief A system offering both translational and rotational constraint. */
template<typename T>
concept ConstrainableSystem =
    PositionConstrainableSystem<T> and DirectorConstrainableSystem<T>;

// ---------------------------------------------------------------------------
// Free boundary condition
// ---------------------------------------------------------------------------

/**
 * @brief A boundary condition that constrains nothing.
 *
 * Useful as an explicit default in a constraint list, and as the template a
 * new boundary condition can be modelled on. Accepts any system type, since it
 * touches nothing.
 */
struct FreeBoundaryCondition
{
public: // Methods
    /** @brief Does nothing. */
    template<typename T>
    void constrain_values(T&&, double) const {}

    /** @brief Does nothing. */
    template<typename T>
    void constrain_rates(T&&, double) const {}
};

// ---------------------------------------------------------------------------
// One end fixed
// ---------------------------------------------------------------------------

/**
 * @brief Clamps the first node and first element of a rod.
 *
 * The first node is held at a fixed position and the first element at a fixed
 * orientation, with both corresponding rates driven to zero. This is the usual
 * cantilever boundary condition.
 *
 * @see FixedConstraint for clamping arbitrary nodes and elements.
 */
struct OneEndFixedBoundaryCondition
{
private: // Members
    Eigen::Vector3d m_fixed_position;
    Eigen::Matrix3d m_fixed_directors;

public: // Methods
    /**
     * @brief Builds the boundary condition from the target configuration.
     * @param fixed_position Position held by the first node; must be finite.
     * @param fixed_directors Orientation held by the first element; must be
     *        finite.
     */
    OneEndFixedBoundaryCondition(
        Eigen::Vector3d fixed_position, Eigen::Matrix3d fixed_directors
    );

    /**
     * @brief Pins the first node position and first element orientation.
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<ConstrainableSystem SystemType>
    void constrain_values(SystemType& system, double) const
    {
        Vector3DStack& positions = system.positions();
        Matrix3DStack& directors = system.mutable_frames();
        utils::nice_assert(positions.rows() > 0, "Need at least one node");
        utils::nice_assert(not directors.empty(), "Need at least one element");

        positions.row(0) = m_fixed_position.transpose();
        directors[0] = m_fixed_directors;
    }

    /**
     * @brief Drives the first node and element rates to zero.
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<ConstrainableSystem SystemType>
    void constrain_rates(SystemType& system, double) const
    {
        Vector3DStack& velocities = system.velocities();
        Vector3DStack& omegas = system.angular_velocities();
        utils::nice_assert(velocities.rows() > 0, "Need at least one node");
        utils::nice_assert(omegas.rows() > 0, "Need at least one element");

        velocities.row(0).setZero();
        omegas.row(0).setZero();
    }

    /** @brief The position held by the first node. */
    const Eigen::Vector3d& fixed_position() const;
    /** @brief The orientation held by the first element. */
    const Eigen::Matrix3d& fixed_directors() const;
};

// ---------------------------------------------------------------------------
// General constraint
// ---------------------------------------------------------------------------

/**
 * @brief Constrains selected degrees of freedom at chosen nodes and elements.
 *
 * A selector is a triple of booleans, one per axis. Where an entry is true the
 * corresponding degree of freedom is removed; where it is false the system
 * remains free along that axis.
 *
 * Translational selectors act in the inertial frame and are applied directly
 * to node positions and velocities. Rotational selectors also act in the
 * inertial frame: angular velocities are rotated out of the material frame,
 * masked, and rotated back, so constraining the z entry removes rotation about
 * the lab z axis regardless of how the element is currently oriented.
 *
 * This constraint pins positions but never directors. Use @ref FixedConstraint
 * when element orientations must be held at prescribed values.
 *
 * @see FixedConstraint
 */
struct GeneralConstraint
{
private: // Members
    std::vector<std::int64_t> m_position_indices;
    Vector3DStack m_fixed_positions;
    std::vector<std::int64_t> m_director_indices;
    // Stored as 1.0 where constrained and 0.0 where free, so the masks can be
    // applied arithmetically rather than branched on.
    Eigen::Array3d m_translational_selector;
    Eigen::Array3d m_rotational_selector;

public: // Methods
    /**
     * @brief Builds the constraint.
     * @param position_indices Nodes to constrain; may be negative to count
     *        back from the end.
     * @param fixed_positions Target position for each constrained node, one
     *        row per entry of @p position_indices.
     * @param director_indices Elements whose rotational rates are constrained;
     *        may be negative.
     * @param translational_selector Per-axis translational mask, in the
     *        inertial frame. True removes the degree of freedom.
     * @param rotational_selector Per-axis rotational mask, in the inertial
     *        frame. True removes the degree of freedom.
     */
    GeneralConstraint(
        std::vector<std::int64_t> position_indices,
        Vector3DStack fixed_positions,
        std::vector<std::int64_t> director_indices,
        std::array<bool, 3> translational_selector,
        std::array<bool, 3> rotational_selector
    );

    /**
     * @brief Blends constrained node positions toward their fixed values.
     *
     * Axes where the translational selector is false keep their current value.
     *
     * @tparam SystemType Any @ref PositionConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<PositionConstrainableSystem SystemType>
    void constrain_values(SystemType& system, double) const
    {
        if (m_position_indices.empty()) return;

        Vector3DStack& positions = system.positions();
        for (std::size_t entry = 0; entry < m_position_indices.size(); ++entry)
        {
            const Eigen::Index idx =
                resolve_index(m_position_indices[entry], positions.rows());
            const Eigen::Array3d current = positions.row(idx).transpose().array();
            const Eigen::Array3d target =
                m_fixed_positions.row(static_cast<Eigen::Index>(entry))
                    .transpose().array();

            positions.row(idx) =
                ((1.0 - m_translational_selector) * current
                 + m_translational_selector * target)
                    .matrix().transpose();
        }
    }

    /**
     * @brief Zeroes the constrained translational and rotational rates.
     *
     * Rotational masking is performed in the inertial frame, so the angular
     * velocity is rotated out of the material frame, masked, then rotated back.
     *
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<ConstrainableSystem SystemType>
    void constrain_rates(SystemType& system, double) const
    {
        if (not m_position_indices.empty())
        {
            Vector3DStack& velocities = system.velocities();
            for (const std::int64_t requested : m_position_indices)
            {
                const Eigen::Index idx =
                    resolve_index(requested, velocities.rows());
                velocities.row(idx).array() *=
                    (1.0 - m_translational_selector).transpose();
            }
        }

        if (not m_director_indices.empty())
        {
            Matrix3DStack& directors = system.mutable_frames();
            Vector3DStack& omegas = system.angular_velocities();
            utils::nice_assert(
                static_cast<Eigen::Index>(directors.size()) == omegas.rows(),
                "Expected one director per angular velocity row"
            );

            for (const std::int64_t requested : m_director_indices)
            {
                const Eigen::Index idx = resolve_index(requested, omegas.rows());
                const Eigen::Matrix3d& director = directors[idx];

                Eigen::Vector3d lab_frame =
                    director.transpose() * omegas.row(idx).transpose();
                lab_frame.array() *= (1.0 - m_rotational_selector);
                omegas.row(idx) = (director * lab_frame).transpose();
            }
        }
    }

    /** @brief Indices of the constrained nodes, as supplied. */
    const std::vector<std::int64_t>& position_indices() const;
    /** @brief Target position for each constrained node. */
    const Vector3DStack& fixed_positions() const;
    /** @brief Indices of the elements whose rotational rates are constrained. */
    const std::vector<std::int64_t>& director_indices() const;
    /** @brief Translational mask, 1.0 where constrained and 0.0 where free. */
    const Eigen::Array3d& translational_selector() const;
    /** @brief Rotational mask, 1.0 where constrained and 0.0 where free. */
    const Eigen::Array3d& rotational_selector() const;
};

// ---------------------------------------------------------------------------
// Fixed constraint
// ---------------------------------------------------------------------------

/**
 * @brief Clamps chosen nodes and elements in every degree of freedom.
 *
 * Equivalent to a @ref GeneralConstraint with all selector entries true, with
 * the addition that element directors are also held at prescribed values
 * rather than merely having their rates constrained. Positions, directors and
 * both rate stacks are pinned outright, so no frame conversion is needed.
 *
 * @see GeneralConstraint for constraining a subset of the degrees of freedom.
 */
struct FixedConstraint
{
private: // Members
    std::vector<std::int64_t> m_position_indices;
    Vector3DStack m_fixed_positions;
    std::vector<std::int64_t> m_director_indices;
    Matrix3DStack m_fixed_directors;

public: // Methods
    /**
     * @brief Builds the constraint.
     * @param position_indices Nodes to clamp; may be negative to count back
     *        from the end.
     * @param fixed_positions Target position for each clamped node, one row
     *        per entry of @p position_indices.
     * @param director_indices Elements to clamp; may be negative.
     * @param fixed_directors Target orientation for each clamped element, one
     *        entry per entry of @p director_indices.
     */
    FixedConstraint(
        std::vector<std::int64_t> position_indices,
        Vector3DStack fixed_positions,
        std::vector<std::int64_t> director_indices,
        Matrix3DStack fixed_directors
    );

    /**
     * @brief Pins the clamped node positions and element orientations.
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<ConstrainableSystem SystemType>
    void constrain_values(SystemType& system, double) const
    {
        if (not m_position_indices.empty())
        {
            Vector3DStack& positions = system.positions();
            for (std::size_t entry = 0; entry < m_position_indices.size(); ++entry)
            {
                const Eigen::Index idx =
                    resolve_index(m_position_indices[entry], positions.rows());
                positions.row(idx) =
                    m_fixed_positions.row(static_cast<Eigen::Index>(entry));
            }
        }

        if (not m_director_indices.empty())
        {
            Matrix3DStack& directors = system.mutable_frames();
            for (std::size_t entry = 0; entry < m_director_indices.size(); ++entry)
            {
                const Eigen::Index idx = resolve_index(
                    m_director_indices[entry],
                    static_cast<Eigen::Index>(directors.size())
                );
                directors[idx] = m_fixed_directors[entry];
            }
        }
    }

    /**
     * @brief Drives the clamped rates to zero.
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     */
    template<ConstrainableSystem SystemType>
    void constrain_rates(SystemType& system, double) const
    {
        if (not m_position_indices.empty())
        {
            Vector3DStack& velocities = system.velocities();
            for (const std::int64_t requested : m_position_indices)
            {
                velocities.row(resolve_index(requested, velocities.rows()))
                    .setZero();
            }
        }

        if (not m_director_indices.empty())
        {
            Vector3DStack& omegas = system.angular_velocities();
            for (const std::int64_t requested : m_director_indices)
            {
                omegas.row(resolve_index(requested, omegas.rows())).setZero();
            }
        }
    }

    /** @brief Indices of the clamped nodes, as supplied. */
    const std::vector<std::int64_t>& position_indices() const;
    /** @brief Target position for each clamped node. */
    const Vector3DStack& fixed_positions() const;
    /** @brief Indices of the clamped elements, as supplied. */
    const std::vector<std::int64_t>& director_indices() const;
    /** @brief Target orientation for each clamped element. */
    const Matrix3DStack& fixed_directors() const;
};

// ---------------------------------------------------------------------------
// Helical buckling
// ---------------------------------------------------------------------------

/**
 * @brief Applies twist and slack to both ends of a rod.
 *
 * Reproduces the helical buckling setup of Gazzola et al., RSoS (2018). While
 * @f$ t \le @f$ @c twisting_time the two ends are driven with equal and
 * opposite angular and shrink velocities; afterwards they are pinned at the
 * fully twisted, fully slackened configuration and held still.
 *
 * Angular velocity is applied as half the total rotation rate at each end, and
 * shrink velocity as half the total slack rate, so the two ends together
 * deliver @c number_of_rotations turns and @c slack of shortening over the
 * twisting interval.
 */
struct HelicalBucklingBoundaryCondition
{
private: // Members
    double m_twisting_time;
    double m_slack;
    double m_number_of_rotations;
    Eigen::Vector3d m_final_start_position;
    Eigen::Vector3d m_final_end_position;
    Eigen::Vector3d m_angular_velocity;
    Eigen::Vector3d m_shrink_velocity;
    Eigen::Matrix3d m_final_start_directors;
    Eigen::Matrix3d m_final_end_directors;

public: // Methods
    /**
     * @brief Builds the boundary condition and precomputes the end states.
     * @param position_start Initial position of the first node.
     * @param position_end Initial position of the last node. Must differ from
     *        @p position_start.
     * @param director_start Initial orientation of the first element.
     * @param director_end Initial orientation of the last element.
     * @param twisting_time Time over which the twist is applied; must be
     *        finite and positive.
     * @param slack Total shortening applied to the rod; must be finite.
     * @param number_of_rotations Total turns applied; must be finite.
     */
    HelicalBucklingBoundaryCondition(
        Eigen::Vector3d position_start,
        Eigen::Vector3d position_end,
        Eigen::Matrix3d director_start,
        Eigen::Matrix3d director_end,
        double twisting_time,
        double slack,
        double number_of_rotations
    );

    /**
     * @brief Pins both ends once twisting is complete.
     *
     * Does nothing while @f$ t \le @f$ @c twisting_time, leaving the ends free
     * to be carried by the rates.
     *
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     * @param time Current simulation time.
     */
    template<ConstrainableSystem SystemType>
    void constrain_values(SystemType& system, double time) const
    {
        utils::nice_assert(std::isfinite(time), "Expected time to be finite");
        if (time <= m_twisting_time) return;

        Vector3DStack& positions = system.positions();
        Matrix3DStack& directors = system.mutable_frames();
        utils::nice_assert(positions.rows() > 0, "Need at least one node");
        utils::nice_assert(not directors.empty(), "Need at least one element");

        positions.row(0) = m_final_start_position.transpose();
        positions.row(positions.rows() - 1) = m_final_end_position.transpose();
        directors.front() = m_final_start_directors;
        directors.back() = m_final_end_directors;
    }

    /**
     * @brief Drives the ends during twisting, then holds them still.
     * @tparam SystemType Any @ref ConstrainableSystem.
     * @param system System to constrain in place.
     * @param time Current simulation time.
     */
    template<ConstrainableSystem SystemType>
    void constrain_rates(SystemType& system, double time) const
    {
        utils::nice_assert(std::isfinite(time), "Expected time to be finite");

        Vector3DStack& velocities = system.velocities();
        Vector3DStack& omegas = system.angular_velocities();
        utils::nice_assert(velocities.rows() > 0, "Need at least one node");
        utils::nice_assert(omegas.rows() > 0, "Need at least one element");

        const Eigen::Index last_node = velocities.rows() - 1;
        const Eigen::Index last_element = omegas.rows() - 1;

        if (time > m_twisting_time)
        {
            velocities.row(0).setZero();
            omegas.row(0).setZero();
            velocities.row(last_node).setZero();
            omegas.row(last_element).setZero();
            return;
        }

        velocities.row(0) = m_shrink_velocity.transpose();
        omegas.row(0) = m_angular_velocity.transpose();
        velocities.row(last_node) = -m_shrink_velocity.transpose();
        omegas.row(last_element) = -m_angular_velocity.transpose();
    }

    /** @brief Time over which the twist is applied. */
    double twisting_time() const;
    /** @brief Total shortening applied to the rod. */
    double slack() const;
    /** @brief Total turns applied to the rod. */
    double number_of_rotations() const;
    /** @brief First node position once twisting is complete. */
    const Eigen::Vector3d& final_start_position() const;
    /** @brief Last node position once twisting is complete. */
    const Eigen::Vector3d& final_end_position() const;
    /** @brief Angular velocity applied to the first end during twisting. */
    const Eigen::Vector3d& angular_velocity() const;
    /** @brief Shrink velocity applied to the first end during twisting. */
    const Eigen::Vector3d& shrink_velocity() const;
    /** @brief First element orientation once twisting is complete. */
    const Eigen::Matrix3d& final_start_directors() const;
    /** @brief Last element orientation once twisting is complete. */
    const Eigen::Matrix3d& final_end_directors() const;
};

// ---------------------------------------------------------------------------
// Variant dispatch
// ---------------------------------------------------------------------------

/** @brief Any one of the boundary conditions, held by value. */
using ConstraintVariant = std::variant<
    FreeBoundaryCondition,
    OneEndFixedBoundaryCondition,
    GeneralConstraint,
    FixedConstraint,
    HelicalBucklingBoundaryCondition
>;

/**
 * @brief Fails if the held constraint cannot be applied to the given system.
 *
 * Each constraint restricts its entry points to the concept it needs, so
 * probing whether both calls are well formed decides compatibility without
 * naming the concept here.
 *
 * @tparam BodyType The system type to check against.
 * @param constraint_var Constraint to check.
 * @param system System to check against; not modified.
 */
template<typename BodyType>
void validate(ConstraintVariant& constraint_var, BodyType& system)
{
    std::visit([&](auto& constraint)
    {
        const double time = 0.0;
        if constexpr (
            not requires {
                constraint.constrain_values(system, time);
                constraint.constrain_rates(system, time);
            }
        )
        {
            utils::nice_assert(false, "Constraint is incompatible with this system");
        }
    }, constraint_var);
}

/**
 * @brief Applies the held constraint to the system configuration.
 * @tparam BodyType The system type to constrain.
 * @param constraint_var Constraint to apply.
 * @param system System to constrain in place.
 * @param time Current simulation time.
 */
template<typename BodyType>
void constrain_values(ConstraintVariant& constraint_var, BodyType& system, double time)
{
    std::visit([&](auto& constraint)
    {
        if constexpr (requires {constraint.constrain_values(system, time);})
        {
            constraint.constrain_values(system, time);
        }
        else utils::nice_assert(false, "Constraint is incompatible with this system");
    }, constraint_var);
}

/**
 * @brief Applies the held constraint to the system rates.
 * @tparam BodyType The system type to constrain.
 * @param constraint_var Constraint to apply.
 * @param system System to constrain in place.
 * @param time Current simulation time.
 */
template<typename BodyType>
void constrain_rates(ConstraintVariant& constraint_var, BodyType& system, double time)
{
    std::visit([&](auto& constraint)
    {
        if constexpr (requires {constraint.constrain_rates(system, time);})
        {
            constraint.constrain_rates(system, time);
        }
        else utils::nice_assert(false, "Constraint is incompatible with this system");
    }, constraint_var);
}
} // End namespace cosserat::physics
