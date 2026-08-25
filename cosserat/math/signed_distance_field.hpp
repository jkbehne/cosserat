#pragma once

/**
 * @file signed_distance_field.hpp
 * @brief Signed distance fields, as the geometry side of mesh contact.
 *
 * A rod meets a mesh through a scalar field rather than through triangles.
 * Anything that can answer "how far is this point from the surface, and which
 * way is out" is enough to run contact against, which is what
 * @ref SignedDistanceField requires. The rod itself is never tessellated: it
 * stays a chain of capsules, and contact is a point query minus a radius.
 *
 * @f[ \gamma = r_e - d(\mathbf{x}), \qquad \hat{\mathbf{n}} = \nabla d(\mathbf{x}) @f]
 *
 * @section sdf_convention Sign and gradient convention
 *
 * Distance is **positive outside** the body and negative within, so the
 * gradient points away from the surface, out into free space. Contact rules
 * rely on that: the gradient at a penetrating point is the direction the rod
 * has to move to escape.
 *
 * @section sdf_backends Backends
 *
 * Three are provided here. @ref AnalyticSphereField and @ref AnalyticBoxField
 * are exact and closed form, which makes them the right thing to test contact
 * against: every quantity has a known answer. @ref TriangleMeshField wraps a
 * bounding volume hierarchy over real triangles.
 *
 * A discretised field, such as Discregrid's @c CubicLagrangeDiscreteGrid, fits
 * the same concept and is roughly two orders of magnitude faster per query
 * because its cost does not depend on triangle count. Sampling a
 * @ref TriangleMeshField is how such a grid is built, so the two agree by
 * construction and can be diffed against one another. See
 * @ref sdf_discrete_backend for the adapter.
 *
 * @section sdf_requirements What a mesh has to satisfy
 *
 * Signs come from the angle weighted pseudonormal of Bærentzen and Aanæs, so
 * the mesh must be watertight with consistently outward normals. A triangle
 * soup or a mesh with holes still yields the correct *distance*, but the sign
 * is meaningless, and contact would then push the rod in arbitrary directions.
 * @ref TriangleMeshField cannot verify this, so validate at load time.
 */

#include <functional>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace cosserat::math {

/**
 * @brief What a distance query returns.
 */
struct SignedDistance
{
public: // Members
    /** @brief Signed distance to the surface; positive outside. */
    double distance = 0.0;

    /**
     * @brief Gradient of the distance, pointing away from the surface.
     *
     * Unit length wherever the field is differentiable. It degenerates on the
     * medial axis, where the nearest surface point is not unique, and callers
     * are expected to check rather than normalise blindly.
     */
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
};

/**
 * @brief Anything that can be queried for a signed distance and a gradient.
 *
 * @tparam T Candidate field type.
 */
template<typename T>
concept SignedDistanceField = requires(const T field, const Eigen::Vector3d& point)
{
    {field.signed_distance(point)} -> std::same_as<SignedDistance>;
    {field.domain()} -> std::same_as<Eigen::AlignedBox3d>;
};

/**
 * @brief A field query with its type erased, for passing into a kernel.
 *
 * The contact kernels take one of these rather than a template parameter, so
 * they can live in a translation unit instead of a header. The indirect call
 * costs a couple of nanoseconds against a query costing tens to thousands, so
 * the trade is heavily in favour of not templating.
 */
using FieldQuery = std::function<SignedDistance(const Eigen::Vector3d&)>;

/**
 * @brief Wraps any field into a @ref FieldQuery.
 *
 * @tparam FieldType Any @ref SignedDistanceField.
 * @param field The field, which must outlive the returned query.
 * @return A callable performing the same query.
 */
template<SignedDistanceField FieldType>
FieldQuery as_query(const FieldType& field)
{
    return [&field](const Eigen::Vector3d& point) {return field.signed_distance(point);};
}

// ---------------------------------------------------------------------------
// Analytic fields
// ---------------------------------------------------------------------------

/**
 * @brief An exact field around a sphere.
 *
 * Closed form everywhere except the centre, which is its medial axis. Useful
 * as ground truth: a contact rule run against this can be checked against
 * hand-computed answers.
 */
class AnalyticSphereField
{
private: // Members
    Eigen::Vector3d m_center;
    double m_radius;
    double m_margin;

public: // Methods
    /**
     * @brief Builds the field.
     * @param center Centre of the sphere.
     * @param radius Radius; must be finite and positive.
     * @param margin How far beyond the surface the domain extends.
     */
    AnalyticSphereField(const Eigen::Vector3d& center, double radius, double margin);

    /**
     * @brief Distance and gradient at a point.
     * @param point Query position.
     * @return The signed distance, with a zero gradient exactly at the centre.
     */
    SignedDistance signed_distance(const Eigen::Vector3d& point) const;

    /** @brief The box the field is defined over. */
    Eigen::AlignedBox3d domain() const;

    /** @brief Centre of the sphere. */
    const Eigen::Vector3d& center() const;
    /** @brief Radius of the sphere. */
    double radius() const;
    /** @brief Margin of the sphere. */
    double margin() const;
};

/**
 * @brief An exact field around an axis aligned box.
 *
 * Exact outside and on the surface. Inside, it reports the distance to the
 * nearest face, which is the standard box distance function and is exact for a
 * convex box.
 */
class AnalyticBoxField
{
private: // Members
    Eigen::Vector3d m_center;
    Eigen::Vector3d m_half_extent;
    double m_margin;

public: // Methods
    /**
     * @brief Builds the field.
     * @param center Centre of the box.
     * @param half_extent Half the box's size on each axis; all positive.
     * @param margin How far beyond the surface the domain extends.
     */
    AnalyticBoxField(
        const Eigen::Vector3d& center, const Eigen::Vector3d& half_extent, double margin
    );

    /**
     * @brief Distance and gradient at a point.
     * @param point Query position.
     * @return The signed distance.
     */
    SignedDistance signed_distance(const Eigen::Vector3d& point) const;

    /** @brief The box the field is defined over. */
    Eigen::AlignedBox3d domain() const;

    /** @brief The center of the box. */
    const Eigen::Vector3d& center() const;
    /** @brief Half the box's size on each axis; all positive. */
    const Eigen::Vector3d& half_extent() const;
    /** @brief How far beyond the surface the domain extends. */
    double margin() const;
};

/**
 * @page sdf_discrete_backend Using a discretised field
 *
 * A grid backed field satisfies @ref SignedDistanceField with a thin adapter.
 * With Discregrid, whose @c interpolate returns the value and writes the
 * gradient through an out parameter:
 *
 * @code
 * class DiscreteField
 * {
 * private:
 *     std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid> m_grid;
 *
 * public:
 *     SignedDistance signed_distance(const Eigen::Vector3d& point) const
 *     {
 *         SignedDistance result;
 *         result.distance = m_grid->interpolate(0u, point, &result.gradient);
 *         return result;
 *     }
 *
 *     Eigen::AlignedBox3d domain() const {return m_grid->domain();}
 * };
 * @endcode
 *
 * Build the grid by sampling an exact field, which is what makes the two
 * comparable:
 *
 * @code
 * TriangleMeshField exact(vertices, triangles);
 * auto sample = [&](const Eigen::Vector3d& x) {return exact.signed_distance(x).distance;};
 * grid->addFunction(sample, true);
 * @endcode
 *
 * Prefer cubic cells over trilinear ones. A trilinear gradient is constant
 * within a cell and therefore jumps across cell boundaries, and a contact
 * force whose direction jumps will excite the rod as it slides. Discregrid's
 * cubic Serendipity cells give continuous gradients.
 *
 * Two things a grid needs that an exact field does not. Its domain must extend
 * beyond the mesh by at least the largest rod radius plus a margin, or a rod
 * touching the surface from outside falls off the edge of the field. And
 * features smaller than the cell spacing are rounded away, so size the grid to
 * the smallest feature that matters mechanically.
 */
} // End namespace cosserat::physics
