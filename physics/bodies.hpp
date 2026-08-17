#pragma once

#include <cstdint>
#include <numbers>

#include <Eigen/Core>

#include "math/linalg.hpp"
#include "math/finite_difference.hpp"
#include "math/types.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

class CosseratRod
{
private: // Members
    std::int64_t m_num_nodes;
    std::int64_t m_num_elements;
    std::int64_t m_num_voronoi;

    // All should have num_nodes entries
    Vector3DStack m_positions;
    Vector3DStack m_velocities;
    Vector3DStack m_accelerations;
    Vector3DStack m_internal_forces;
    Vector3DStack m_external_forces;
    Eigen::VectorXd m_masses;

    // All should have num_elements entries
    Matrix3DStack m_frames;
    Matrix3DStack m_mass_2nd_moments;
    Matrix3DStack m_inv_mass_2nd_moments;
    Matrix3DStack m_bending_matrices;
    Matrix3DStack m_shearing_matrices;
    Vector3DStack m_angular_velocities;
    Vector3DStack m_angular_accelerations;
    Vector3DStack m_internal_torques;
    Vector3DStack m_external_torques;
    Vector3DStack m_tangents;
    Vector3DStack m_sigmas;
    Vector3DStack m_rest_sigmas;
    Eigen::VectorXd m_radii;
    Eigen::VectorXd m_densities;
    Eigen::VectorXd m_volumes;
    Eigen::VectorXd m_lengths;
    Eigen::VectorXd m_rest_lengths;
    Eigen::VectorXd m_dilatations;
    Eigen::VectorXd m_dilatation_rates;

    // All should have num_voronoi entries
    Vector3DStack m_kappas;
    Vector3DStack m_rest_kappas;
    Eigen::VectorXd m_voronoi_dilatations;
    Eigen::VectorXd m_voronoi_rest_lengths;

    // Unclear
    Vector3DStack m_internal_stresses;
    Vector3DStack m_internal_couples;

    bool m_respect_radii;

public: // Methods
    void compute_internal_forces_and_torques(double)
    {
        compute_internal_forces();
        compute_internal_torques();
    }

private: // Methods
    void compute_geometry()
    {
        const auto pos_diff = math::row_difference(m_positions);
        m_lengths = math::row_norms(pos_diff);
        utils::nice_assert(
            (m_lengths.array() > 0.0).all(),
            "m_lengths needs to have all entries greater than zero"
        );
        m_tangents = pos_diff.array().rowwise() / m_lengths.transpose().array();
        if (m_respect_radii)
        {
            utils::nice_assert(
                m_radii.size() == m_lengths.size(),
                "Expected radii and lengths to have the same number of elements"
            );
            m_volumes = std::numbers::pi * m_lengths.array() * m_radii.array().square();
        }
        else
        {
            utils::nice_assert(
                m_volumes.size() == m_lengths.size(),
                "Expected volumes and lengths to have the same number of elements"
            );
            m_radii = (m_volumes.array() / (std::numbers::pi * m_lengths.array())).sqrt();
        }
    }

    void compute_dilatations()
    {
        compute_geometry();
        utils::nice_assert(
            m_lengths.size() == m_rest_lengths.size(),
            "Expected lengths and rest_length to have the same number of elements"
        );
        utils::nice_assert(
            (m_rest_lengths.array() > 0.0).all(),
            "Expected all entries of rest_lengths to be greater than zero"
        );
        m_dilatations = m_lengths.array() / m_rest_lengths.array();
        const Eigen::VectorXd voronoi_lengths = math::row_average(m_lengths);
        utils::nice_assert(
            m_voronoi_rest_lengths.size() == voronoi_lengths.size(),
            "Expected Voronoi lengths and rest Voronoi lengths to have the same size"
        );
        m_voronoi_dilatations = voronoi_lengths.array() / m_voronoi_rest_lengths.array();
    }

    void compute_shear_stretch_strains()
    {
        compute_dilatations();
        const auto unit_z = Eigen::Vector3d::UnitZ();
        const Vector3DStackT tangents_T = m_tangents.transpose();
        const auto product = math::batched_matrix_vector(m_frames, tangents_T);
        const auto left = product.array().rowwise() * m_dilatations.transpose().array();
        m_sigmas = left.array().rowwise() - unit_z.transpose().array();
    }

    void compute_bending_twist_strains()
    {
        const Eigen::Index num_interior_idxs = m_frames.size() - 1;
        utils::nice_assert(num_interior_idxs >=  1, "Need at least one interior index");
        Vector3DStack inverse_rotations(num_interior_idxs, 3);
        for (Eigen::Index idx = 0; idx < num_interior_idxs; ++idx)
        {
            inverse_rotations.row(idx) = math::inverse_rotate(
                m_frames[idx], m_frames[idx + 1]
            ).transpose();
        }
        m_kappas = inverse_rotations.array().rowwise() / m_voronoi_rest_lengths.transpose().array();
    }

    void compute_internal_forces()
    {
        compute_internal_shear_stretch_stresses_from_model();
        const auto cosserat_internal_stress = math::batched_matrix_vector<
            false /* ignore size mismatch */, true /* tranpose matrices */
        >(m_frames, m_internal_stresses);
        utils::nice_assert(
            cosserat_internal_stress.rows() == m_dilatations.rows(),
            "Expected cosserat stress and dilatations to have the same number of rows"
        );
        const Vector3DStack cosserat_is = cosserat_internal_stress.array().rowwise()
            / m_dilatations.transpose().array();
        m_internal_forces = math::row_difference_kernel(cosserat_is);
    }

    void compute_internal_shear_stretch_stresses_from_model()
    {
        compute_shear_stretch_strains();
        const Vector3DStackT pointwise = (m_sigmas - m_rest_sigmas).transpose();
        m_internal_stresses = math::batched_matrix_vector(m_shearing_matrices, pointwise);
    }

    void compute_internal_torques()
    {
        compute_internal_bending_twist_stresses_from_model();
        compute_dilatation_rates();

        const auto voronoi_term = 1.0 / m_voronoi_dilatations.array().cube();
        const Vector3DStack product = m_internal_couples.array().rowwise()
            * voronoi_term.transpose().array();
        const auto bend_twist_couple_2D = math::row_difference_kernel(product);
    }

    void compute_internal_bending_twist_stresses_from_model()
    {
        compute_bending_twist_strains();
        const auto pointwise = (m_kappas - m_rest_kappas).transpose();
        m_internal_couples = math::batched_matrix_vector(m_bending_matrices, pointwise);
    }

    void compute_dilatation_rates()
    {
        const auto num_nodes = m_positions.rows();
        const auto num_elements = num_nodes - 1;
        utils::nice_assert(num_elements >= 1, "Need at least one element");

        const auto r_dot_v = math::batched_dot_product(m_positions, m_velocities);
        const auto r_plus_dot_v = math::batched_dot_product(
            m_positions.bottomRows(num_elements), m_velocities.topRows(num_elements)
        );
        const auto r_dot_v_plus = math::batched_dot_product(
            m_positions.topRows(num_elements), m_velocities.bottomRows(num_elements)
        );

        utils::nice_assert(
            m_dilatation_rates.size() == num_elements,
            "Dilatation rates should be the same size as the number of elements"
        );
        utils::nice_assert(
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
};

} // End namespace cosserat::physics
