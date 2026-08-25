#include <cosserat/physics/rigid_body.hpp>

#include <cmath>
#include <concepts>
#include <utility>

#include <cosserat/utils/assertions.hpp>

namespace cosserat::physics {
using utils::nice_assert;

namespace {

/**
 * @brief One of the container types a rigid body stores its state in.
 *
 * Constrains the variadic writer below so a mistyped argument is rejected at
 * the call rather than deep inside the recursion.
 */
template<typename T>
concept RigidBodyStorage = std::same_as<T, Vector3DStack>
    or std::same_as<T, Matrix3DStack> or std::same_as<T, Eigen::VectorXd>;

/** @brief Recursion base case for @ref write_storage. */
inline void write_storage() {}

/**
 * @brief Writes one stack to its stem, then recurses on the rest.
 *
 * Deliberately local to this translation unit rather than shared with the rod,
 * whose equivalent lives in its own source file. Hoisting both into a common
 * header would remove the duplication if a third body type appears.
 *
 * @param storage Stack to write.
 * @param stem Output path without an extension.
 * @param rest Remaining pairs.
 */
template<RigidBodyStorage Storage, typename... Args>
void write_storage(
    const Storage& storage, const std::filesystem::path& stem, const Args&... rest
)
{
    utils::write_matrix(stem, storage);
    write_storage(rest...);
}

/**
 * @brief Writes a whole list of stacks given as (storage, stem) pairs.
 * @param args Pairs of stack and output stem.
 */
template<typename... Args>
void write_storages(const Args&... args)
{
    static_assert(sizeof...(Args) > 0, "Must have at least one argument");
    static_assert(sizeof...(Args) % 2 == 0, "Arguments must come in pairs");
    write_storage(args...);
}

/** @brief Fails unless a scalar is finite and strictly positive. */
void assert_positive(double value, const std::string& name)
{
    nice_assert(
        std::isfinite(value) and value > 0.0,
        "Expected " + name + " to be finite and greater than zero"
    );
}

/** @brief Wraps a single vector as a one-row stack. */
Vector3DStack as_stack(const Eigen::Vector3d& vector)
{
    Vector3DStack stack(1, 3);
    stack.row(0) = vector.transpose();
    return stack;
}

/** @brief Wraps a single scalar as a one-entry vector. */
Eigen::VectorXd as_vector(double value)
{
    Eigen::VectorXd vector(1);
    vector(0) = value;
    return vector;
}

} // End anonymous namespace

// RigidBody constructor
RigidBody::RigidBody(
    Eigen::Vector3d position,
    Eigen::Matrix3d frame,
    double radius,
    double length,
    double density,
    double volume,
    Eigen::Vector3d inertia_diagonal
)
{
    nice_assert(
        position.array().isFinite().all(), "position must contain only finite values"
    );
    assert_positive(radius, "radius");
    assert_positive(length, "length");
    assert_positive(density, "density");
    assert_positive(volume, "volume");
    nice_assert(
        (inertia_diagonal.array() > tolerance).all(),
        "Mass 2nd moment is close to singular"
    );

    m_positions = as_stack(position);
    m_velocities = Vector3DStack::Zero(1, 3);
    m_accelerations = Vector3DStack::Zero(1, 3);
    // A rigid body generates no internal loads; these stay zero for its life.
    m_internal_forces = Vector3DStack::Zero(1, 3);
    m_external_forces = Vector3DStack::Zero(1, 3);
    m_masses = as_vector(volume * density);

    m_frames = Matrix3DStack{std::move(frame)};
    m_mass_2nd_moments = Matrix3DStack{
        Eigen::Matrix3d(inertia_diagonal.asDiagonal())};
    m_inv_mass_2nd_moments = Matrix3DStack{
        Eigen::Matrix3d(inertia_diagonal.cwiseInverse().asDiagonal())};
    m_angular_velocities = Vector3DStack::Zero(1, 3);
    m_angular_accelerations = Vector3DStack::Zero(1, 3);
    m_internal_torques = Vector3DStack::Zero(1, 3);
    m_external_torques = Vector3DStack::Zero(1, 3);

    m_radii = as_vector(radius);
    m_densities = as_vector(density);
    m_volumes = as_vector(volume);
    m_lengths = as_vector(length);
    // Nothing about a rigid body deforms, so the rest configuration is the
    // current one and the dilatation is identically one.
    m_rest_lengths = as_vector(length);
    m_dilatations = as_vector(1.0);

    assert_frame_validity();
    assert_is_proper();
}

// RigidBody public methods

void RigidBody::compute_internal_forces_and_torques(double)
{
    // Intentionally empty. A rigid body has no strain, so no internal loads.
}

void RigidBody::update_accelerations(double, double)
{
    nice_assert((m_masses.array() > 0.0).all(), "Zeros in masses");

    // No internal forces to add: a = F_ext / m.
    m_accelerations = m_external_forces / m_masses(0);

    // (J w) x w, the transport term that a rod folds into its internal torques.
    const Vector3DStack j_omega = math::batched_matrix_vector(
        m_mass_2nd_moments, m_angular_velocities.transpose()
    );
    const Vector3DStack lagrangian_transport =
        math::batched_cross_product(j_omega, m_angular_velocities);

    // No dilatation factor here: a rigid body's dilatation is always one.
    m_angular_accelerations = math::batched_matrix_vector(
        m_inv_mass_2nd_moments,
        (lagrangian_transport + m_external_torques).transpose()
    );
}

void RigidBody::zero_out_external_forces_and_torques(double)
{
    m_external_forces = Vector3DStack::Zero(1, 3);
    m_external_torques = Vector3DStack::Zero(1, 3);
}

Eigen::Vector3d RigidBody::position_center_of_mass() const
{
    return m_positions.row(0).transpose();
}

double RigidBody::translational_energy() const
{
    const Eigen::Vector3d velocity = m_velocities.row(0).transpose();
    return 0.5 * m_masses(0) * velocity.dot(velocity);
}

double RigidBody::rotational_energy() const
{
    const Vector3DStack j_omega = math::batched_matrix_vector(
        m_mass_2nd_moments, m_angular_velocities.transpose()
    );
    return 0.5 * math::batched_dot_product(m_angular_velocities, j_omega).sum();
}

void RigidBody::write_debug(const std::filesystem::path& write_path) const
{
    write_storages(
        m_positions, write_path / "positions",
        m_velocities, write_path / "velocities",
        m_accelerations, write_path / "accelerations",
        m_internal_forces, write_path / "internal_forces",
        m_external_forces, write_path / "external_forces",
        m_masses, write_path / "masses",
        m_frames, write_path / "frames",
        m_mass_2nd_moments, write_path / "mass_2nd_moments",
        m_inv_mass_2nd_moments, write_path / "inverse_mass_2nd_moments",
        m_angular_velocities, write_path / "angular_velocities",
        m_angular_accelerations, write_path / "angular_accelerations",
        m_internal_torques, write_path / "internal_torques",
        m_external_torques, write_path / "external_torques",
        m_radii, write_path / "radii",
        m_densities, write_path / "densities",
        m_volumes, write_path / "volumes",
        m_lengths, write_path / "lengths",
        m_rest_lengths, write_path / "rest_lengths",
        m_dilatations, write_path / "dilatations"
    );
}

void RigidBody::write(const std::filesystem::path& write_path) const
{
    write_storages(
        m_positions, write_path / "positions",
        m_frames, write_path / "frames"
    );
}

// RigidBody (private) assertion methods
void RigidBody::assert_frame_validity() const
{
    nice_assert(m_frames.size() == 1, "Rigid body must have exactly one frame");
    nice_assert(
        math::is_orthogonal(m_frames[0], tolerance),
        "Frame is not orthogonal matrix"
    );
}

void RigidBody::assert_is_proper() const
{
    nice_assert(m_positions.rows() == 1, "Expected one position row");
    nice_assert(m_velocities.rows() == 1, "Expected one velocity row");
    nice_assert(m_accelerations.rows() == 1, "Expected one acceleration row");
    nice_assert(m_internal_forces.rows() == 1, "Expected one internal force row");
    nice_assert(m_external_forces.rows() == 1, "Expected one external force row");
    nice_assert(m_masses.size() == 1, "Expected one mass entry");
    nice_assert(m_frames.size() == 1, "Expected one frame");
    nice_assert(m_mass_2nd_moments.size() == 1, "Expected one mass 2nd moment");
    nice_assert(
        m_inv_mass_2nd_moments.size() == 1, "Expected one inverse mass 2nd moment"
    );
    nice_assert(
        m_angular_velocities.rows() == 1, "Expected one angular velocity row"
    );
    nice_assert(
        m_angular_accelerations.rows() == 1, "Expected one angular acceleration row"
    );
    nice_assert(m_internal_torques.rows() == 1, "Expected one internal torque row");
    nice_assert(m_external_torques.rows() == 1, "Expected one external torque row");
    nice_assert(m_radii.size() == 1, "Expected one radius entry");
    nice_assert(m_densities.size() == 1, "Expected one density entry");
    nice_assert(m_volumes.size() == 1, "Expected one volume entry");
    nice_assert(m_lengths.size() == 1, "Expected one length entry");
    nice_assert(m_rest_lengths.size() == 1, "Expected one rest length entry");
    nice_assert(m_dilatations.size() == 1, "Expected one dilatation entry");
}

// RigidBody (public-const) accessors
std::int64_t RigidBody::num_nodes() const {return 1;}
std::int64_t RigidBody::num_elements() const {return 1;}
const Vector3DStack& RigidBody::positions() const {return m_positions;}
const Vector3DStack& RigidBody::velocities() const {return m_velocities;}
const Vector3DStack& RigidBody::accelerations() const {return m_accelerations;}
const Vector3DStack& RigidBody::internal_forces() const {return m_internal_forces;}
const Vector3DStack& RigidBody::external_forces() const {return m_external_forces;}
const Eigen::VectorXd& RigidBody::masses() const {return m_masses;}
const Matrix3DStack& RigidBody::frames() const {return m_frames;}
const Matrix3DStack& RigidBody::mass_2nd_moments() const {return m_mass_2nd_moments;}
const Matrix3DStack& RigidBody::inv_mass_2nd_moments() const
{
    return m_inv_mass_2nd_moments;
}
const Vector3DStack& RigidBody::angular_velocities() const
{
    return m_angular_velocities;
}
const Vector3DStack& RigidBody::angular_accelerations() const
{
    return m_angular_accelerations;
}
const Vector3DStack& RigidBody::internal_torques() const {return m_internal_torques;}
const Vector3DStack& RigidBody::external_torques() const {return m_external_torques;}
const Eigen::VectorXd& RigidBody::radii() const {return m_radii;}
const Eigen::VectorXd& RigidBody::densities() const {return m_densities;}
const Eigen::VectorXd& RigidBody::volumes() const {return m_volumes;}
const Eigen::VectorXd& RigidBody::lengths() const {return m_lengths;}
const Eigen::VectorXd& RigidBody::rest_lengths() const {return m_rest_lengths;}
const Eigen::VectorXd& RigidBody::dilatations() const {return m_dilatations;}

Eigen::Vector3d RigidBody::tangent() const
{
    return m_frames[0].row(2).transpose();
}
double RigidBody::total_mass() const {return m_masses(0);}
double RigidBody::radius() const {return m_radii(0);}
double RigidBody::length() const {return m_lengths(0);}
double RigidBody::volume() const {return m_volumes(0);}
double RigidBody::density() const {return m_densities(0);}

// RigidBody (public-mutable) accessors
Vector3DStack& RigidBody::mutable_positions() {return m_positions;}
Vector3DStack& RigidBody::mutable_velocities() {return m_velocities;}
Matrix3DStack& RigidBody::mutable_frames() {return m_frames;}
Vector3DStack& RigidBody::mutable_angular_velocities() {return m_angular_velocities;}
Vector3DStack& RigidBody::mutable_external_forces() {return m_external_forces;}
Vector3DStack& RigidBody::mutable_external_torques() {return m_external_torques;}

// Sphere
namespace {

/** @brief Body frame of a sphere: normal, binormal and tangent along x, y, z. */
Eigen::Matrix3d sphere_frame()
{
    const Eigen::Vector3d normal(1.0, 0.0, 0.0);
    const Eigen::Vector3d tangent(0.0, 0.0, 1.0);
    const Eigen::Vector3d binormal = tangent.cross(normal);

    Eigen::Matrix3d frame;
    frame.row(0) = normal.transpose();
    frame.row(1) = binormal.transpose();
    frame.row(2) = tangent.transpose();
    return frame;
}

} // End anonymous namespace

Sphere::Sphere(const Eigen::Vector3d& center, double base_radius, double density)
    : RigidBody(
        center,
        sphere_frame(),
        base_radius,
        // The reference implementation treats the sphere's diameter as its
        // characteristic length.
        2.0 * base_radius,
        density,
        (4.0 / 3.0) * std::numbers::pi * base_radius * base_radius * base_radius,
        // Isotropic and exact for a uniform solid sphere.
        Eigen::Vector3d::Constant(
            0.4 * ((4.0 / 3.0) * std::numbers::pi * base_radius * base_radius
                   * base_radius * density)
            * base_radius * base_radius))
{
}

// Cylinder
namespace {

/**
 * @brief Body frame of a cylinder from its axis and roll reference.
 *
 * Rows are the normal, the binormal formed as tangent crossed with normal, and
 * the tangent, matching the convention @ref straight_cosserat_rod uses for a
 * rod's element frames.
 */
Eigen::Matrix3d cylinder_frame(
    const Eigen::Vector3d& direction, const Eigen::Vector3d& normal
)
{
    const Eigen::Vector3d binormal = direction.cross(normal);

    Eigen::Matrix3d frame;
    frame.row(0) = normal.transpose();
    frame.row(1) = binormal.transpose();
    frame.row(2) = direction.transpose();
    return frame;
}

/**
 * @brief Mass second moment of inertia of a cylinder, in the reference's form.
 *
 * The two transverse entries are the cross-section's second moment of area
 * scaled by density and length, and the axial entry is twice that. See the
 * warning on @ref Cylinder for how this differs from a true solid cylinder.
 */
Eigen::Vector3d cylinder_inertia_diagonal(
    double base_length, double base_radius, double density
)
{
    const double area = std::numbers::pi * base_radius * base_radius;
    const double transverse = area * area / (4.0 * std::numbers::pi);
    const double axial = 2.0 * transverse;
    return Eigen::Vector3d(transverse, transverse, axial) * density * base_length;
}

} // End anonymous namespace

Cylinder::Cylinder(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction,
    const Eigen::Vector3d& normal,
    double base_length,
    double base_radius,
    double density,
    double tolerance_
)
    : RigidBody(
        // The body is placed at its midpoint, half a length from the start.
        Eigen::Vector3d(start + 0.5 * base_length * direction),
        cylinder_frame(direction, normal),
        base_radius,
        base_length,
        density,
        std::numbers::pi * base_radius * base_radius * base_length,
        cylinder_inertia_diagonal(base_length, base_radius, density))
{
    // Checked after the base is built so that the shape's own inputs are
    // reported rather than the derived frame. The base already rejects a
    // non-orthogonal frame, which is what a bad pair of directions produces.
    nice_assert(
        math::is_unit_vector(direction, tolerance_), "direction must be unit vector"
    );
    nice_assert(math::is_unit_vector(normal, tolerance_), "normal must be unit vector");
    nice_assert(
        std::abs(normal.dot(direction)) < tolerance_,
        "direction and normal must be orthogonal"
    );
}
} // End namespace cosserat::physics
