#pragma once

/**
 * @file quantities.hpp
 * @brief Scalar quantities summarising the state of a whole simulation.
 *
 * Free functions rather than objects, because each is a pure reduction of a
 * collection of bodies to one number and carries no state worth keeping. They
 * are useful for logging, for assertions in tests, and for deciding whether a
 * scene has stopped moving, but none of that is their concern: they compute a
 * number and return it.
 *
 * @section quantities_which What each one is for
 *
 * @ref total_kinetic_energy is the quantity that is zero exactly when nothing
 * is moving, accounts for rotation as well as translation, and decays smoothly
 * as motion dies away. It is the one to reach for first.
 *
 * @ref max_speed is cruder and blind to a body spinning without translating,
 * but it is a speed, so a threshold on it can be read straight off a plot of
 * the motion.
 *
 * @ref max_axial_elastic_strain measures how hard the rods are being
 * stretched. It suits a problem whose motion is axial, a cable or a tether
 * being the obvious case, and reads almost nothing for anything else: a rod is
 * far stiffer along its length than across it, so bending motion barely
 * registers. Measured on a rod settling onto an obstacle, it wandered inside a
 * narrow band the whole time while the kinetic energy fell four orders of
 * magnitude.
 *
 * @section quantities_bodies Which bodies are counted
 *
 * All of them, without exception, and that turns out to be what is wanted even
 * when some bodies are held in place. A constrained body has its rates zeroed
 * by @c constrain_rates before any of this runs, so it contributes nothing to
 * a sum or a maximum built on velocity and drops out of its own accord.
 *
 * Bodies that cannot answer a question are skipped rather than faked. A rigid
 * body has no notion of being stretched, so it contributes nothing to
 * @ref max_axial_elastic_strain; a collection holding nothing else therefore
 * measures zero.
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <variant>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "math/types.hpp"

#include "physics/bodies.hpp"

namespace cosserat::physics {

/**
 * @brief A finalized collection whose bodies can be walked.
 *
 * Named so that handing one of these functions something unfinalized, or
 * something that is not a collection at all, fails with a message about
 * @c final_systems rather than several screens of template substitution
 * failures from inside a fold.
 *
 * @tparam T Candidate collection type.
 */
template<typename T>
concept BodyCollection = requires(T system)
{
    // A range of handles, each of which can be asked for the body it holds.
    requires std::ranges::range<decltype(system.final_systems())>;
    {(*std::ranges::begin(system.final_systems())).body()};
};

namespace detail {

/**
 * @brief Whether a body carries the rates a motion measure needs.
 * @tparam T Candidate body type.
 */
template<typename T>
concept HasRates = requires(const T body)
{
    {body.velocities()} -> std::same_as<const Vector3DStack&>;
    {body.angular_velocities()} -> std::same_as<const Vector3DStack&>;
};

/**
 * @brief Whether a body carries mass and inertia as well as rates.
 * @tparam T Candidate body type.
 */
template<typename T>
concept HasInertia = HasRates<T> and requires(const T body)
{
    {body.masses()} -> std::same_as<const Eigen::VectorXd&>;
    {body.mass_2nd_moments()} -> std::same_as<const Matrix3DStack&>;
};

/**
 * @brief Whether a body carries shear and stretch strains.
 *
 * Rods do. A rigid body has no notion of being stretched.
 *
 * @tparam T Candidate body type.
 */
template<typename T>
concept HasStrains = requires(const T body)
{
    {body.sigmas()} -> std::same_as<const Vector3DStack&>;
    {body.rest_sigmas()} -> std::same_as<const Vector3DStack&>;
};

} // End namespace detail

// ---------------------------------------------------------------------------
// One body at a time
// ---------------------------------------------------------------------------

/**
 * @brief Kinetic energy of a single body's translation.
 *
 * @f[ \tfrac{1}{2}\sum_i m_i \|\mathbf{v}_i\|^2 @f]
 *
 * Masses sit on the nodes for a rod and on the single node of a rigid body, so
 * both are handled without distinguishing them.
 *
 * @tparam BodyType Any body carrying masses and velocities.
 * @param body The body to measure.
 * @return The translational kinetic energy, in joules. Never negative.
 */
template<detail::HasInertia BodyType>
double translational_kinetic_energy(const BodyType& body)
{
    const Eigen::VectorXd speeds_squared =
        body.velocities().rowwise().squaredNorm();
    return 0.5 * body.masses().dot(speeds_squared);
}

/**
 * @brief Kinetic energy of a single body's rotation.
 *
 * @f[ \tfrac{1}{2}\sum_j \boldsymbol{\omega}_j \cdot
 *     \mathbf{J}_j \boldsymbol{\omega}_j @f]
 *
 * Angular velocities and second moments are both held in the material frame,
 * so no change of basis is needed between them.
 *
 * @tparam BodyType Any body carrying second moments and angular velocities.
 * @param body The body to measure.
 * @return The rotational kinetic energy, in joules. Never negative for a
 *         positive definite inertia, which every non degenerate body has.
 */
template<detail::HasInertia BodyType>
double rotational_kinetic_energy(const BodyType& body)
{
    double total = 0.0;
    const Matrix3DStack& moments = body.mass_2nd_moments();
    const Vector3DStack& omegas = body.angular_velocities();
    for (Eigen::Index element = 0; element < omegas.rows(); ++element)
    {
        const Eigen::Vector3d omega = omegas.row(element).transpose();
        total +=
            0.5 * omega.dot(moments[static_cast<std::size_t>(element)] * omega);
    }
    return total;
}

/**
 * @brief Total kinetic energy of a single body.
 *
 * @tparam BodyType Any body carrying mass, inertia and rates.
 * @param body The body to measure.
 * @return Translational plus rotational kinetic energy, in joules.
 */
template<detail::HasInertia BodyType>
double kinetic_energy(const BodyType& body)
{
    return translational_kinetic_energy(body) + rotational_kinetic_energy(body);
}

/**
 * @brief The fastest any node of a single body is moving.
 *
 * @tparam BodyType Any body carrying velocities.
 * @param body The body to measure.
 * @return The largest nodal speed, in metres per second, or zero for a body
 *         with no nodes.
 */
template<detail::HasRates BodyType>
double max_speed(const BodyType& body)
{
    if (body.velocities().rows() == 0) return 0.0;
    return body.velocities().rowwise().norm().maxCoeff();
}

/**
 * @brief The largest axial elastic strain anywhere along a single rod.
 *
 * The axial strain is the third component of the shear and stretch strain,
 *
 * @f[ \sigma_3 = e\,(\mathbf{Q}\mathbf{t})_3 - 1 @f]
 *
 * and the elastic part is what remains once the rest strain is taken off,
 * since that is the part the constitutive law turns into stress. Where there
 * is no shear it reduces to the dilatation less one.
 *
 * @tparam BodyType Any body carrying shear and stretch strains.
 * @param body The rod to measure.
 * @return The largest magnitude, dimensionless, or zero for a rod with no
 *         elements.
 */
template<detail::HasStrains BodyType>
double max_axial_elastic_strain(const BodyType& body)
{
    if (body.sigmas().rows() == 0) return 0.0;
    const Eigen::VectorXd elastic = body.sigmas().col(2) - body.rest_sigmas().col(2);
    return elastic.cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------------
// A whole collection
// ---------------------------------------------------------------------------

/**
 * @brief Total kinetic energy of every body in a collection.
 *
 * @tparam SystemType Any @ref BodyCollection.
 * @param system The finalized collection to measure.
 * @return The total kinetic energy, in joules. Zero exactly when nothing is
 *         moving.
 */
template<BodyCollection SystemType>
double total_kinetic_energy(SystemType& system)
{
    double total = 0.0;
    for (auto& handle : system.final_systems())
    {
        std::visit([&total](const auto& body)
        {
            using BodyType = std::decay_t<decltype(body)>;
            if constexpr (detail::HasInertia<BodyType>)
            {
                total += kinetic_energy(body);
            }
        }, handle.body());
    }
    return total;
}

/**
 * @brief The fastest any node of any body in a collection is moving.
 *
 * @tparam SystemType Any @ref BodyCollection.
 * @param system The finalized collection to measure.
 * @return The largest nodal speed, in metres per second, or zero if nothing in
 *         the collection carries velocities.
 */
template<BodyCollection SystemType>
double max_speed(SystemType& system)
{
    double fastest = 0.0;
    for (auto& handle : system.final_systems())
    {
        std::visit([&fastest](const auto& body)
        {
            using BodyType = std::decay_t<decltype(body)>;
            if constexpr (detail::HasRates<BodyType>)
            {
                fastest = std::max(fastest, max_speed(body));
            }
        }, handle.body());
    }
    return fastest;
}

/**
 * @brief The largest axial elastic strain over every rod in a collection.
 *
 * Bodies with no strains are skipped, so a collection holding only rigid
 * bodies measures zero.
 *
 * @tparam SystemType Any @ref BodyCollection.
 * @param system The finalized collection to measure.
 * @return The largest magnitude, dimensionless, or zero if the collection
 *         holds no rods.
 */
template<BodyCollection SystemType>
double max_axial_elastic_strain(SystemType& system)
{
    double largest = 0.0;
    for (auto& handle : system.final_systems())
    {
        std::visit([&largest](const auto& body)
        {
            using BodyType = std::decay_t<decltype(body)>;
            if constexpr (detail::HasStrains<BodyType>)
            {
                largest = std::max(largest, max_axial_elastic_strain(body));
            }
        }, handle.body());
    }
    return largest;
}
} // End namespace cosserat::physics
