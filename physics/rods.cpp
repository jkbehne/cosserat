#include "physics/rods.hpp"

#include <concepts>
#include <utility>

#include "utils/assertions.hpp"

namespace cosserat::physics {
using utils::nice_assert;

namespace detail {

/**
 * @brief One of the three container types a rod stores its state in.
 *
 * Constrains the variadic helpers below so that a mistyped argument is
 * rejected at the call rather than deep inside the recursion.
 */
template<typename T>
concept CosseratStorage = std::same_as<T, Vector3DStack>
    or std::same_as<T, Matrix3DStack> or std::same_as<T, Eigen::VectorXd>;

/** @brief Recursion base case for @ref assert_storage_size. */
inline void assert_storage_size() {}

/**
 * @brief Checks one stack's entry count, then recurses on the rest.
 *
 * Entry count means rows for the two Eigen stacks and element count for the
 * vector of matrices, so the three storage types can be checked uniformly.
 *
 * @param storage Stack to check.
 * @param expected_size Required number of entries.
 * @param name Name reported in the failure message.
 * @param rest Remaining triples.
 */
template<CosseratStorage Storage, typename... Args>
void assert_storage_size(
    const Storage& storage,
    const std::int64_t expected_size,
    const std::string& name,
    const Args&... rest
)
{
    std::int64_t storage_size = -1;
    if constexpr (std::is_same_v<Storage, Matrix3DStack>)
    {
        storage_size = static_cast<std::int64_t>(storage.size());
    }
    else if constexpr (std::is_same_v<Storage, Vector3DStack>)
    {
        storage_size = storage.rows();
    }
    else if constexpr (std::is_same_v<Storage, Eigen::VectorXd>)
    {
        storage_size = storage.rows();
    }
    nice_assert(
        storage_size == expected_size,
        "Expected " + name + " to have " + std::to_string(expected_size) + " entries"
    );
    assert_storage_size(rest...);
}

/**
 * @brief Checks a whole list of stacks given as (storage, size, name) triples.
 * @param args Triples of stack, expected entry count and name.
 */
template<typename... Args>
void assert_storage_sizes(const Args&... args)
{
    static_assert(sizeof...(Args) > 0, "Must have at least one argument");
    static_assert(sizeof...(Args) % 3 == 0, "Arguments must be in threes");
    assert_storage_size(args...);
}

/** @brief Recursion base case for @ref write_storage. */
inline void write_storage() {}

/**
 * @brief Writes one stack to its stem, then recurses on the rest.
 * @param storage Stack to write.
 * @param stem Output path without an extension.
 * @param rest Remaining pairs.
 */
template<CosseratStorage Storage, typename... Args>
void write_storage(
    const Storage& storage,
    const std::filesystem::path& stem,
    const Args&... rest
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
} // End namespace detail

// CosseratRod static methods
void CosseratRod::assert_size_and_lower_bound(
    const Eigen::VectorXd& vector,
    std::int64_t exp_length,
    double lower_bound,
    const std::string& name
)
{
    nice_assert(
        vector.size() == exp_length, "Expected " + name + " to have "
        + std::to_string(exp_length) + " elements, but got " + std::to_string(vector.size())
    );
    nice_assert(
        (vector.array() > lower_bound).all(), "Expected " + name + " to have lower bound of "
        + std::to_string(lower_bound)
    );
}

Matrix3DStack CosseratRod::build_shearing_matrices(
    double youngs_modulus, double shear_modulus, const Eigen::VectorXd& A0
)
{
    const auto num_elements = A0.rows();
    Matrix3DStack shearing_matrices;
    shearing_matrices.reserve(num_elements);
    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        const auto A0_i = A0(idx);
        const Eigen::Vector3d diag {
            alpha_c * shear_modulus * A0_i,
            alpha_c * shear_modulus * A0_i,
            youngs_modulus * A0_i
        };
        shearing_matrices.push_back(diag.asDiagonal());
    }
    return shearing_matrices;
}

Matrix3DStack CosseratRod::build_bending_matrices(
    double youngs_modulus, double shear_modulus, const Vector3DStack& I0
)
{
    const auto num_elements = I0.rows();
    Matrix3DStack bending_matrices;
    bending_matrices.reserve(num_elements);
    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        const Eigen::Vector3d diag {
            youngs_modulus * I0(idx, 0),
            youngs_modulus * I0(idx, 1),
            shear_modulus * I0(idx, 2)
        };
        bending_matrices.push_back(diag.asDiagonal());
    }
    return bending_matrices;
}

// CosseratRod constructors
CosseratRod::CosseratRod(
    std::int64_t num_nodes,
    Vector3DStack positions,
    Vector3DStack velocities,
    Vector3DStack accelerations,
    Vector3DStack internal_forces,
    Vector3DStack external_forces,
    Eigen::VectorXd masses,
    Matrix3DStack frames,
    Matrix3DStack mass_2nd_moments,
    Matrix3DStack inv_mass_2nd_moments,
    Matrix3DStack bending_matrices,
    Matrix3DStack shearing_matrices,
    Vector3DStack angular_velocities,
    Vector3DStack angular_accelerations,
    Vector3DStack internal_torques,
    Vector3DStack external_torques,
    Vector3DStack tangents,
    Vector3DStack sigmas,
    Vector3DStack rest_sigmas,
    Vector3DStack internal_stresses,
    Eigen::VectorXd radii,
    Eigen::VectorXd densities,
    Eigen::VectorXd volumes,
    Eigen::VectorXd lengths,
    Eigen::VectorXd rest_lengths,
    Eigen::VectorXd dilatations,
    Eigen::VectorXd dilatation_rates,
    Vector3DStack kappas,
    Vector3DStack rest_kappas,
    Vector3DStack internal_couples,
    Eigen::VectorXd voronoi_dilatations,
    Eigen::VectorXd voronoi_rest_lengths
) : m_num_nodes(num_nodes),
    m_num_elements(num_nodes - 1),
    m_num_voronoi(num_nodes - 2),
    m_positions(std::move(positions)),
    m_velocities(std::move(velocities)),
    m_accelerations(std::move(accelerations)),
    m_internal_forces(std::move(internal_forces)),
    m_external_forces(std::move(external_forces)),
    m_masses(std::move(masses)),
    m_frames(std::move(frames)),
    m_mass_2nd_moments(std::move(mass_2nd_moments)),
    m_inv_mass_2nd_moments(std::move(inv_mass_2nd_moments)),
    m_shearing_matrices(std::move(shearing_matrices)),
    m_angular_velocities(std::move(angular_velocities)),
    m_angular_accelerations(std::move(angular_accelerations)),
    m_internal_torques(std::move(internal_torques)),
    m_external_torques(std::move(external_torques)),
    m_tangents(std::move(tangents)),
    m_sigmas(std::move(sigmas)),
    m_rest_sigmas(std::move(rest_sigmas)),
    m_internal_stresses(std::move(internal_stresses)),
    m_radii(std::move(radii)),
    m_densities(std::move(densities)),
    m_volumes(std::move(volumes)),
    m_lengths(std::move(lengths)),
    m_rest_lengths(std::move(rest_lengths)),
    m_dilatations(std::move(dilatations)),
    m_dilatation_rates(std::move(dilatation_rates)),
    m_bending_matrices(std::move(bending_matrices)),
    m_kappas(std::move(kappas)),
    m_rest_kappas(std::move(rest_kappas)),
    m_internal_couples(std::move(internal_couples)),
    m_voronoi_dilatations(std::move(voronoi_dilatations)),
    m_voronoi_rest_lengths(std::move(voronoi_rest_lengths)),
    m_respect_radii(true)
{
    nice_assert(m_num_voronoi > 0, "Expected at least one interior element");
    assert_is_proper();
    assert_volume_radii_density_validity();
    assert_frame_validity();
    assert_rest_lengths_validity();
}

CosseratRod::CosseratRod(
    Vector3DStack positions,
    Matrix3DStack frames,
    Matrix3DStack bending_matrices,
    Matrix3DStack shearing_matrices,
    Vector3DStack rest_sigmas,
    Eigen::VectorXd densities,
    Eigen::VectorXd volumes_or_radii,
    Eigen::VectorXd rest_lengths,
    Vector3DStack rest_kappas,
    bool respect_radii
)
{
    build_geometry(
        std::move(positions),
        std::move(volumes_or_radii),
        std::move(densities),
        std::move(frames),
        std::move(rest_lengths),
        respect_radii
    );
    build_mass_2nd_moments();
    build_shearing_bending_matrices(
        std::move(shearing_matrices), std::move(bending_matrices)
    );
    build_mass();
    build_rest_state(std::move(rest_sigmas), std::move(rest_kappas));
    build_zero_dynamics();
    assert_is_proper();
}

CosseratRod::CosseratRod(
    Vector3DStack positions,
    Matrix3DStack frames,
    Vector3DStack rest_sigmas,
    Eigen::VectorXd densities,
    Eigen::VectorXd volumes_or_radii,
    Eigen::VectorXd rest_lengths,
    Vector3DStack rest_kappas,
    bool respect_radii,
    double youngs_modulus,
    double shear_modulus
)
{
    build_geometry(
        std::move(positions),
        std::move(volumes_or_radii),
        std::move(densities),
        std::move(frames),
        std::move(rest_lengths),
        respect_radii
    );
    const auto result = build_mass_2nd_moments();
    build_shearing_bending_matrices(result, youngs_modulus, shear_modulus);
    build_mass();
    build_rest_state(std::move(rest_sigmas), std::move(rest_kappas));
    build_zero_dynamics();
    assert_is_proper();
}

CosseratRod::CosseratRod(
    Vector3DStack positions,
    Matrix3DStack frames,
    Vector3DStack rest_sigmas,
    Eigen::VectorXd densities,
    Eigen::VectorXd volumes_or_radii,
    Eigen::VectorXd rest_lengths,
    Vector3DStack rest_kappas,
    bool respect_radii,
    double youngs_modulus
) : CosseratRod(
    positions,
    frames,
    rest_sigmas,
    densities,
    volumes_or_radii,
    rest_lengths,
    rest_kappas,
    respect_radii,
    youngs_modulus,
    youngs_modulus / 3.0
) {}

// CosseratRod public methods

void CosseratRod::compute_internal_forces_and_torques(double)
{
    compute_internal_forces();
    compute_internal_torques();
}

void CosseratRod::update_accelerations(double, double)
{
    nice_assert(m_internal_forces.rows() == m_num_nodes, "Incorrect internal forces size");
    nice_assert(m_external_forces.rows() == m_num_nodes, "Incorrect external forces size");
    nice_assert((m_masses.array() > 0.0).all(), "Zeros in masses");
    m_accelerations = (m_internal_forces + m_external_forces).array().colwise()
        / m_masses.array();

    nice_assert(
        static_cast<std::int64_t>(m_inv_mass_2nd_moments.size()) == m_num_elements,
        "Incorrect inverse 2nd moments size"
    );
    nice_assert(
        m_internal_torques.rows() == m_num_elements, "Incorrect internal torques size"
    );
    nice_assert(
        m_external_torques.rows() == m_num_elements, "Incorrect external torques size"
    );
    nice_assert(m_dilatations.size() == m_num_elements, "Incorrect dilatations size");
    const auto product = math::batched_matrix_vector(
        m_inv_mass_2nd_moments, (m_internal_torques + m_external_torques).transpose()
    );
    m_angular_accelerations = product.array().colwise() * m_dilatations.array();
}

void CosseratRod::zero_out_external_forces_and_torques(double)
{
    m_external_forces = Vector3DStack::Zero(m_num_nodes, 3);
    m_external_torques = Vector3DStack::Zero(m_num_elements, 3);
}

void CosseratRod::write_debug(const std::filesystem::path& write_path) const
{
    detail::write_storages(
        m_positions, write_path / "positions",
        m_velocities, write_path / "velocities",
        m_accelerations, write_path / "accelerations",
        m_internal_forces, write_path / "internal_forces",
        m_external_forces, write_path / "external_forces",
        m_masses, write_path / "masses",
        m_frames, write_path / "frames",
        m_mass_2nd_moments, write_path / "mass_2nd_moments",
        m_inv_mass_2nd_moments, write_path / "inverse_mass_2nd_moments",
        m_bending_matrices, write_path / "bending_matrices",
        m_shearing_matrices, write_path / "shearing_matrices",
        m_angular_velocities, write_path / "angular_velocities",
        m_angular_accelerations, write_path / "angular_accelerations",
        m_internal_torques, write_path / "internal_torques",
        m_external_torques, write_path / "external_torques",
        m_tangents, write_path / "tangents",
        m_sigmas, write_path / "sigmas",
        m_rest_sigmas, write_path / "rest_sigmas",
        m_radii, write_path / "radii",
        m_densities, write_path / "densities",
        m_volumes, write_path / "volumes",
        m_lengths, write_path / "lengths",
        m_rest_lengths, write_path / "rest_lengths",
        m_dilatations, write_path / "dilatations",
        m_dilatation_rates, write_path / "dilatation_rates",
        m_internal_stresses, write_path / "internal_stresses",
        m_kappas, write_path / "kappas",
        m_rest_kappas, write_path / "rest_kappas",
        m_internal_couples, write_path / "internal_couples",
        m_voronoi_dilatations, write_path / "voronoi_dilatations",
        m_voronoi_rest_lengths, write_path / "voronoi_rest_lengths"
    );
}

void CosseratRod::write(const std::filesystem::path& write_path) const
{
    detail::write_storages(
        m_positions, write_path / "positions",
        m_frames, write_path / "frames"
    );
}

// CosseratRod (private) build methods
void CosseratRod::build_geometry(
    Vector3DStack positions,
    Eigen::VectorXd volumes_or_radii,
    Eigen::VectorXd densities,
    Matrix3DStack frames,
    Eigen::VectorXd rest_lengths,
    bool respect_radii
)
{
    const auto num_nodes = positions.rows();
    nice_assert(num_nodes >= 3, "Need at least 3 nodes");
    m_num_nodes = num_nodes;
    m_num_elements = num_nodes - 1;
    m_num_voronoi = num_nodes - 2;

    m_positions = std::move(positions);
    if (respect_radii) m_radii = std::move(volumes_or_radii);
    else m_volumes = std::move(volumes_or_radii);
    m_respect_radii = respect_radii;
    m_densities = std::move(densities);
    // Only one of radii and volumes is supplied; the other is derived by
    // compute_geometry below, so it cannot be validated yet.
    if (respect_radii)
    {
        assert_size_and_lower_bound(m_radii, m_num_elements, tolerance, "radii");
    }
    else
    {
        assert_size_and_lower_bound(m_volumes, m_num_elements, tolerance, "volumes");
    }
    assert_size_and_lower_bound(m_densities, m_num_elements, tolerance, "densities");

    m_rest_lengths = std::move(rest_lengths);
    assert_rest_lengths_validity();
    build_voronoi_rest_lengths();
    m_frames = std::move(frames);
    assert_frame_validity();

    // m_lengths, m_tangents, m_volumes, m_radii, m_dilatations, m_voronoi_dilatations,
    // and m_sigmas valid after
    compute_shear_stretch_strains();
    // Both radii and volumes exist now that compute_geometry has derived the
    // one that was not supplied.
    assert_volume_radii_density_validity();
    compute_bending_twist_strains(); // m_kappas valid after
}

CosseratRod::Mass2ndMomentResult CosseratRod::build_mass_2nd_moments()
{
    // Purely geometric: area and the second moments of area of a disk.
    const Eigen::VectorXd A0 = std::numbers::pi * m_radii.array().square();
    const Eigen::VectorXd I0_1 = (0.25 / std::numbers::pi) * A0.array().square();
    Vector3DStack I0(m_num_elements, 3);
    I0.col(0) = I0_1;
    I0.col(1) = I0_1;
    I0.col(2) = 2.0 * I0_1;

    // The mass second moment of inertia is that scaled by density and rest
    // length. The stiffness matrices need the unscaled form, so the scaling is
    // kept local rather than written back into I0.
    const Eigen::VectorXd scalars = m_densities.cwiseProduct(m_rest_lengths);
    const Vector3DStack mass_2nd_moment_temp = I0.array().colwise() * scalars.array();

    Matrix3DStack mass_2nd_moment_stack;
    Matrix3DStack inv_mass_2nd_moment_stack;
    mass_2nd_moment_stack.reserve(static_cast<std::size_t>(m_num_elements));
    inv_mass_2nd_moment_stack.reserve(static_cast<std::size_t>(m_num_elements));
    for (Eigen::Index idx = 0; idx < m_num_elements; ++idx)
    {
        const auto row_vec = mass_2nd_moment_temp.row(idx);
        nice_assert(
            (row_vec.array() > tolerance).all(), "Mass 2nd moment is close to singular"
        );
        mass_2nd_moment_stack.push_back(row_vec.asDiagonal());
        inv_mass_2nd_moment_stack.push_back(row_vec.cwiseInverse().asDiagonal());
    }
    m_mass_2nd_moments = std::move(mass_2nd_moment_stack);
    m_inv_mass_2nd_moments = std::move(inv_mass_2nd_moment_stack);

    Mass2ndMomentResult result;
    result.A0 = A0;
    result.I0 = std::move(I0);
    return result;
}

void CosseratRod::build_shearing_bending_matrices(
    Matrix3DStack shearing_matrices, Matrix3DStack bending_matrices
)
{
    m_shearing_matrices = std::move(shearing_matrices);
    nice_assert(
        static_cast<std::int64_t>(bending_matrices.size()) == m_num_elements,
        "Expected initial bending matrices to have "
        + std::to_string(m_num_elements) + " elements, but got "
        + std::to_string(bending_matrices.size())
    );
    Matrix3DStack voronoi_bending_temp;
    voronoi_bending_temp.reserve(m_num_voronoi);
    for (Eigen::Index idx = 0; idx < m_num_voronoi; ++idx)
    {
        const auto l0 = m_rest_lengths(idx);
        const auto l1 = m_rest_lengths(idx + 1);
        const auto scale = 1.0 / (l0 + l1);
        voronoi_bending_temp.push_back(
            scale * (l0 * bending_matrices[idx] + l1 * bending_matrices[idx + 1])
        );
    }
    m_bending_matrices = std::move(voronoi_bending_temp);
}

void CosseratRod::build_shearing_bending_matrices(
    const Mass2ndMomentResult& mass_result, double youngs_modulus, double shear_modulus
)
{
    build_shearing_bending_matrices(
        build_shearing_matrices(youngs_modulus, shear_modulus, mass_result.A0),
        build_bending_matrices(youngs_modulus, shear_modulus, mass_result.I0)
    );
}

void CosseratRod::build_mass()
{
    m_masses = Eigen::VectorXd(m_num_nodes);
    for (Eigen::Index idx = 0; idx < m_num_nodes; ++idx)
    {
        if (idx == 0) m_masses(idx) = 0.5 * m_densities(idx) * m_volumes(idx);
        else if (idx == m_num_nodes - 1)
        {
            m_masses(idx) = 0.5 * m_densities(idx - 1) * m_volumes(idx - 1);
        }
        else
        {
            const auto m1 = m_densities(idx) * m_volumes(idx);
            const auto m2 = m_densities(idx - 1) * m_volumes(idx - 1);
            m_masses(idx) = 0.5 * (m1 + m2);
        }
    }
}

void CosseratRod::build_voronoi_rest_lengths()
{
    Eigen::VectorXd rest_voronoi_temp(m_num_voronoi);
    for (Eigen::Index idx = 0; idx < m_num_voronoi; ++idx)
    {
        rest_voronoi_temp(idx) =
            0.5 * (m_rest_lengths(idx) + m_rest_lengths(idx + 1));
    }
    m_voronoi_rest_lengths = std::move(rest_voronoi_temp);
}

void CosseratRod::build_rest_state(Vector3DStack rest_sigmas, Vector3DStack rest_kappas)
{
    if (rest_sigmas.rows() > 0) m_rest_sigmas = std::move(rest_sigmas);
    else m_rest_sigmas = Vector3DStack::Zero(m_num_elements, 3);
    if (rest_kappas.rows() > 0) m_rest_kappas = std::move(rest_kappas);
    else m_rest_kappas = Vector3DStack::Zero(m_num_voronoi, 3);
}

void CosseratRod::build_zero_dynamics()
{
    m_velocities = Vector3DStack::Zero(m_num_nodes, 3);
    m_angular_velocities = Vector3DStack::Zero(m_num_elements, 3);
    m_accelerations = Vector3DStack::Zero(m_num_nodes, 3);
    m_angular_accelerations = Vector3DStack::Zero(m_num_elements, 3);
    m_internal_forces = Vector3DStack::Zero(m_num_nodes, 3);
    m_internal_torques = Vector3DStack::Zero(m_num_elements, 3);
    m_external_forces = Vector3DStack::Zero(m_num_nodes, 3);
    m_external_torques = Vector3DStack::Zero(m_num_elements, 3);
    m_dilatation_rates = Eigen::VectorXd::Zero(m_num_elements);
    m_internal_stresses = Vector3DStack::Zero(m_num_elements, 3);
    m_internal_couples = Vector3DStack::Zero(m_num_voronoi, 3);
}

// CosseratRod (private) assertion methods
void CosseratRod::assert_frame_validity() const
{
    nice_assert(
        static_cast<std::int64_t>(m_frames.size()) == m_num_elements,
        "Incorrect size for frames"
    );
    // Only orthogonality is checked. A frame's third row equals the tangent
    // only when the shear strain is exactly zero, which is untrue of any
    // deformed rod and of any rod built with a non-zero rest shear strain.
    for (std::size_t idx = 0; idx < m_frames.size(); ++idx)
    {
        nice_assert(
            math::is_orthogonal(m_frames[idx], tolerance),
            "Frame is not orthogonal matrix"
        );
    }
}

void CosseratRod::assert_volume_radii_density_validity() const
{
    assert_size_and_lower_bound(m_radii, m_num_elements, tolerance, "radii");
    assert_size_and_lower_bound(m_volumes, m_num_elements, tolerance, "volumes");
    assert_size_and_lower_bound(m_densities, m_num_elements, tolerance, "densities");
}

void CosseratRod::assert_rest_lengths_validity() const
{
    assert_size_and_lower_bound(m_rest_lengths, m_num_elements, tolerance, "rest lengths");
}

void CosseratRod::assert_is_proper() const
{
    detail::assert_storage_sizes(
        m_positions, m_num_nodes, "positions",
        m_velocities, m_num_nodes, "velocities",
        m_accelerations, m_num_nodes, "accelerations",
        m_internal_forces, m_num_nodes, "internal forces",
        m_external_forces, m_num_nodes, "external forces",
        m_masses, m_num_nodes, "masses",
        m_mass_2nd_moments, m_num_elements, "mass 2nd moments",
        m_inv_mass_2nd_moments, m_num_elements, "inverse mass 2nd moments",
        m_shearing_matrices, m_num_elements, "shearing matrices",
        m_angular_velocities, m_num_elements, "angular velocities",
        m_angular_accelerations, m_num_elements, "angular accelerations",
        m_internal_torques, m_num_elements, "internal torques",
        m_external_torques, m_num_elements, "external torques",
        m_tangents, m_num_elements, "tangents",
        m_sigmas, m_num_elements, "sigmas",
        m_rest_sigmas, m_num_elements, "rest sigmas",
        m_lengths, m_num_elements, "lengths",
        m_dilatations, m_num_elements, "dilatations",
        m_dilatation_rates, m_num_elements, "dilatation rates",
        m_internal_stresses, m_num_elements, "internal stresses",
        m_bending_matrices, m_num_voronoi, "bending matrices",
        m_kappas, m_num_voronoi, "kappas",
        m_rest_kappas, m_num_voronoi, "rest kappas",
        m_internal_couples, m_num_voronoi, "internal couples",
        m_voronoi_dilatations, m_num_voronoi, "voronoi dilatations",
        m_voronoi_rest_lengths, m_num_voronoi, "voronoi rest lengths"
    );
}

// CosseratRod (private) compute methods
void CosseratRod::compute_geometry()
{
    const auto pos_diff = math::row_difference(m_positions);
    m_lengths = math::row_norms(pos_diff);
    nice_assert(
        (m_lengths.array() > 0.0).all(), "m_lengths entries must be greater than zero"
    );
    m_tangents = pos_diff.array().colwise() / m_lengths.array();
    if (m_respect_radii)
    {

        nice_assert(m_radii.size() == m_num_elements, "Incorrect size for radii");
        m_volumes = std::numbers::pi * m_lengths.array() * m_radii.array().square();
    }
    else
    {
        nice_assert(m_volumes.size() == m_num_elements, "Incorrect size for volumes");
        m_radii = (m_volumes.array() / (std::numbers::pi * m_lengths.array())).sqrt();
    }
}

void CosseratRod::compute_dilatations()
{
    compute_geometry();
    assert_rest_lengths_validity();
    m_dilatations = m_lengths.array() / m_rest_lengths.array();
    const Eigen::VectorXd voronoi_lengths = math::row_average(m_lengths);
    m_voronoi_dilatations = voronoi_lengths.array() / m_voronoi_rest_lengths.array();
}

void CosseratRod::compute_shear_stretch_strains()
{
    compute_dilatations();
    const auto unit_z = Eigen::Vector3d::UnitZ();
    const Vector3DStackT tangents_T = m_tangents.transpose();
    nice_assert(
        static_cast<std::int64_t>(m_frames.size()) == m_num_elements,
        "Incorrect size for m_frames"
    );
    const auto product = math::batched_matrix_vector(m_frames, tangents_T);
    const auto left = product.array().colwise() * m_dilatations.array();
    m_sigmas = left.array().rowwise() - unit_z.transpose().array();
}

void CosseratRod::compute_bending_twist_strains()
{
    const Eigen::Index num_interior_idxs = m_num_voronoi;
    Vector3DStack inverse_rotations(num_interior_idxs, 3);
    for (Eigen::Index idx = 0; idx < num_interior_idxs; ++idx)
    {
        inverse_rotations.row(idx) = math::inverse_rotate(
            m_frames[idx], m_frames[idx + 1]
        ).transpose();
    }
    m_kappas = inverse_rotations.array().colwise() / m_voronoi_rest_lengths.array();
}

void CosseratRod::compute_internal_forces()
{
    compute_internal_shear_stretch_stresses_from_model();
    const auto cosserat_internal_stress = math::batched_matrix_vector<
        false /* ignore size mismatch */, true /* tranpose matrices */
    >(m_frames, m_internal_stresses.transpose());
    nice_assert(
        cosserat_internal_stress.rows() == m_dilatations.rows(),
        "Expected cosserat stress and dilatations to have the same number of rows"
    );
    const Vector3DStack cosserat_is = cosserat_internal_stress.array().colwise()
        / m_dilatations.array();
    m_internal_forces = math::row_difference_kernel(cosserat_is);
}

void CosseratRod::compute_internal_shear_stretch_stresses_from_model()
{
    compute_shear_stretch_strains();
    const Vector3DStackT pointwise = (m_sigmas - m_rest_sigmas).transpose();
    m_internal_stresses = math::batched_matrix_vector(m_shearing_matrices, pointwise);
}

void CosseratRod::compute_internal_torques()
{
    compute_internal_bending_twist_stresses_from_model();
    compute_dilatation_rates();

    nice_assert(
        m_voronoi_dilatations.rows() == m_num_voronoi, "Incorrect voronoi dilatations size"
    );
    nice_assert(
        (m_voronoi_dilatations.array() != 0.0).all(), "Zeros in Voronoi dilatations"
    );
    nice_assert(
        m_internal_couples.rows() == m_num_voronoi, "Incorrect internal couples size"
    );
    const auto voronoi_term = 1.0 / m_voronoi_dilatations.array().cube();
    const Vector3DStack product = m_internal_couples.array().colwise()
        * voronoi_term.array();
    const auto bend_twist_couple_2D = math::row_difference_kernel(product);

    nice_assert(m_kappas.rows() == m_num_voronoi, "Incorrect kappas size");
    const auto batch_cross = math::batched_cross_product(m_kappas, m_internal_couples);
    nice_assert(
        m_voronoi_rest_lengths.rows() == m_num_voronoi, "Incorrect voronoi rest lengths size"
    );
    const Vector3DStack scaled = batch_cross.array().colwise()
        * (m_voronoi_rest_lengths.array() * voronoi_term);
    const auto bend_twist_couple_3D = math::row_average_kernel(scaled);

    nice_assert(
        static_cast<std::int64_t>(m_frames.size()) == m_num_elements,
        "Size of frames is incorrect"
    );
    nice_assert(m_tangents.rows() == m_num_elements, "Size of tangents is incorrect");
    const auto ssc_product = math::batched_matrix_vector(
        m_frames, m_tangents.transpose()
    );
    nice_assert(
        m_internal_stresses.rows() == m_num_elements,
        "Size of internal stress is incorrect"
    );
    const auto ssc_cross = math::batched_cross_product(
        ssc_product, m_internal_stresses
    );
    const Vector3DStack shear_stretch_couple = ssc_cross.array().colwise()
        * m_rest_lengths.array();

    nice_assert(
        static_cast<std::int64_t>(m_mass_2nd_moments.size()) == m_num_elements,
        "Incorrect mass 2nd moments size"
    );
    const auto joe_product = math::batched_matrix_vector(
        m_mass_2nd_moments, m_angular_velocities.transpose()
    );
    nice_assert(m_dilatations.rows() == m_num_elements, "Incorrect dilatations size");
    const Vector3DStack full_joe = joe_product.array().colwise()
        / m_dilatations.array();

    const Vector3DStack lg_transport = math::batched_cross_product(
        full_joe, m_angular_velocities
    );

    nice_assert(
        m_dilatation_rates.rows() == m_num_elements, "Incorrect dilatation rates size"
    );
    nice_assert((m_dilatations.array() != 0.0).all(), "Zeros in dilatations");
    const Vector3DStack unsteady = full_joe.array().colwise()
        * (m_dilatation_rates.array() * m_dilatations.cwiseInverse().array());

    m_internal_torques = bend_twist_couple_2D + bend_twist_couple_3D
        + shear_stretch_couple + lg_transport + unsteady;
}

void CosseratRod::compute_internal_bending_twist_stresses_from_model()
{
    compute_bending_twist_strains();
    const auto pointwise = (m_kappas - m_rest_kappas).transpose();
    m_internal_couples = math::batched_matrix_vector(m_bending_matrices, pointwise);
}

void CosseratRod::compute_dilatation_rates()
{
    const auto num_nodes = m_positions.rows();
    const auto num_elements = num_nodes - 1;
    nice_assert(num_elements >= 1, "Need at least one element");

    const auto r_dot_v = math::batched_dot_product(m_positions, m_velocities);
    const auto r_plus_dot_v = math::batched_dot_product(
        m_positions.bottomRows(num_elements), m_velocities.topRows(num_elements)
    );
    const auto r_dot_v_plus = math::batched_dot_product(
        m_positions.topRows(num_elements), m_velocities.bottomRows(num_elements)
    );

    nice_assert(
        m_dilatation_rates.size() == num_elements,
        "Dilatation rates should be the same size as the number of elements"
    );
    nice_assert(
        (m_lengths.size() == num_elements) and (m_rest_lengths.size() == num_elements),
        "Expected lengths and rest lengths to have same size as dilatation rates"
    );

    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        const auto scalar = 1.0 / (m_lengths(idx) * m_rest_lengths(idx));
        const auto sum = r_dot_v(idx) + r_dot_v(idx + 1);
        const auto diff = sum - (r_dot_v_plus(idx) + r_plus_dot_v(idx));
        m_dilatation_rates(idx) = scalar * diff;
    }
}

// CosseratRod (public-const) accesssors
std::int64_t CosseratRod::num_nodes() const {return m_num_nodes;}
std::int64_t CosseratRod::num_elements() const {return m_num_elements;}
std::int64_t CosseratRod::num_voronoi() const {return m_num_voronoi;}
const Vector3DStack& CosseratRod::positions() const {return m_positions;}
const Vector3DStack& CosseratRod::velocities() const {return m_velocities;}
const Vector3DStack& CosseratRod::accelerations() const {return m_accelerations;}
const Vector3DStack& CosseratRod::internal_forces() const {return m_internal_forces;}
const Vector3DStack& CosseratRod::external_forces() const {return m_external_forces;}
const Eigen::VectorXd& CosseratRod::masses() const {return m_masses;}
const Matrix3DStack& CosseratRod::frames() const {return m_frames;}
const Matrix3DStack& CosseratRod::mass_2nd_moments() const {return m_mass_2nd_moments;}
const Matrix3DStack& CosseratRod::inv_mass_2nd_moments() const {return m_inv_mass_2nd_moments;}
const Matrix3DStack& CosseratRod::bending_matrices() const {return m_bending_matrices;}
const Matrix3DStack& CosseratRod::shearing_matrices() const {return m_shearing_matrices;}
const Vector3DStack& CosseratRod::angular_velocities() const {return m_angular_velocities;}
const Vector3DStack& CosseratRod::angular_accelerations() const {return m_angular_accelerations;}
const Vector3DStack& CosseratRod::internal_torques() const {return m_internal_torques;}
const Vector3DStack& CosseratRod::external_torques() const {return m_external_torques;}
const Vector3DStack& CosseratRod::tangents() const {return m_tangents;}
const Vector3DStack& CosseratRod::sigmas() const {return m_sigmas;}
const Vector3DStack& CosseratRod::rest_sigmas() const {return m_rest_sigmas;}
const Vector3DStack& CosseratRod::internal_stresses() const {return m_internal_stresses;}
const Eigen::VectorXd& CosseratRod::radii() const {return m_radii;}
const Eigen::VectorXd& CosseratRod::densities() const {return m_densities;}
const Eigen::VectorXd& CosseratRod::volumes() const {return m_volumes;}
const Eigen::VectorXd& CosseratRod::lengths() const {return m_lengths;}
const Eigen::VectorXd& CosseratRod::rest_lengths() const {return m_rest_lengths;}
const Eigen::VectorXd& CosseratRod::dilatations() const {return m_dilatations;}
const Eigen::VectorXd& CosseratRod::dilatation_rates() const {return m_dilatation_rates;}
const Vector3DStack& CosseratRod::kappas() const {return m_kappas;}
const Vector3DStack& CosseratRod::rest_kappas() const {return m_rest_kappas;}
const Vector3DStack& CosseratRod::internal_couples() const {return m_internal_couples;}
const Eigen::VectorXd& CosseratRod::voronoi_dilatations() const {return m_voronoi_dilatations;}
const Eigen::VectorXd& CosseratRod::voronoi_rest_lengths() const {return m_voronoi_rest_lengths;}
bool CosseratRod::respect_radii() const {return m_respect_radii;}

// CosseratRod (public-mutable) accessors
Vector3DStack& CosseratRod::mutable_positions() {return m_positions;}
Vector3DStack& CosseratRod::mutable_velocities() {return m_velocities;}
Matrix3DStack& CosseratRod::mutable_frames() {return m_frames;}
Vector3DStack& CosseratRod::mutable_angular_velocities() {return m_angular_velocities;}
Vector3DStack& CosseratRod::mutable_external_forces() {return m_external_forces;}
Vector3DStack& CosseratRod::mutable_external_torques() {return m_external_torques;}

// Utility function for constructing straight rods
CosseratRod straight_cosserat_rod(
    std::int64_t num_elements,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& direction,
    Eigen::Vector3d normal,
    double base_length,
    double base_radius,
    double density,
    double youngs_modulus,
    bool respect_radii,
    double tolerance
)
{
    nice_assert(num_elements >= 3, "Rod must have at least 3 elements");
    nice_assert(base_length > 0.0, "base_length should be greater than zero");
    nice_assert(base_radius > 0.0, "base_radius should be greater than zero");
    nice_assert(density > 0.0, "density should be greater than zero");
    nice_assert(math::is_unit_vector(direction, tolerance), "direction must be unit vector");
    nice_assert(math::is_unit_vector(normal, tolerance), "normal should be unit vector");
    nice_assert(
        std::abs(normal.dot(direction)) < tolerance, "tangent and normal must be orthogonal"
    );

    const Eigen::Vector3d end = start + base_length * direction;
    Vector3DStack positions(num_elements + 1, 3);
    positions.col(0) = Eigen::VectorXd::LinSpaced(num_elements + 1, start(0), end(0));
    positions.col(1) = Eigen::VectorXd::LinSpaced(num_elements + 1, start(1), end(1));
    positions.col(2) = Eigen::VectorXd::LinSpaced(num_elements + 1, start(2), end(2));

    const Eigen::Vector3d cross = direction.cross(normal);
    Eigen::Matrix3d Q;
    Q.row(0) = normal.transpose();
    Q.row(1) = cross.transpose();
    Q.row(2) = direction.transpose();
    Matrix3DStack frames;
    frames.reserve(static_cast<std::size_t>(num_elements));
    for (Eigen::Index idx = 0; idx < num_elements; ++idx)
    {
        frames.push_back(Q);
    }

    Eigen::VectorXd densities = Eigen::VectorXd::Constant(num_elements, density);
    const auto rest_length = base_length / static_cast<double>(num_elements);
    Eigen::VectorXd rest_lengths = Eigen::VectorXd::Constant(num_elements, rest_length);

    // Volume is taken over the rest length, matching the reference
    // implementation, so a rod built at its rest configuration has unit
    // dilatation whichever quantity is held fixed.
    Eigen::VectorXd volumes_or_radii = respect_radii
        ? Eigen::VectorXd(Eigen::VectorXd::Constant(num_elements, base_radius))
        : Eigen::VectorXd(Eigen::VectorXd::Constant(
              num_elements, std::numbers::pi * base_radius * base_radius * rest_length));

    return CosseratRod(
        std::move(positions),
        std::move(frames),
        {}, /* rest_sigmas */
        std::move(densities),
        std::move(volumes_or_radii),
        std::move(rest_lengths),
        {}, /* rest_kappas */
        respect_radii,
        youngs_modulus
    );
}
} // End namespace cosserat::physics
