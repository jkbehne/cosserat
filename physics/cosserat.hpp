// #pragma once

// #include <functional>
// #include <vector>

// #include <Eigen/Dense>
// #include <Eigen/Geometry>

// namespace math {
// namespace cosserat {

// // Type definitions
// using Row3DMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3>;
// using Vector3DStack = std::vector<Eigen::Vector3d>;
// using Matrix3DStack = std::vector<Eigen::Matrix3d>;

// using TaperFunc = std::function<double(const double)>;
// using GrowthRateFunc = std::function<double(const double, const double)>;

// struct RodDynamicState
// {
// public: // Members
//     // Properties with num_vertices entries
//     Vector3DStack positions;
//     Vector3DStack velocities;
//     Vector3DStack accelerations;
//     Vector3DStack internal_forces;
//     Vector3DStack external_forces;
//     std::vector<double> masses;

//     // Properties with num_edges entries
//     Vector3DStack angular_velocities;
//     Vector3DStack angular_accelerations;
//     Vector3DStack internal_torques;
//     Vector3DStack external_torques;
//     Vector3DStack tangents;
//     Vector3DStack sigmas;
//     Vector3DStack rest_sigmas;
//     Matrix3DStack rotations;
//     Matrix3DStack intertia_2nd_moments;
//     Matrix3DStack inverse_intertia_2nd_moments;
//     Matrix3DStack bend_matrix;
//     Matrix3DStack shear_matrix;
//     std::vector<double> rest_lengths;
//     std::vector<double> densities;
//     std::vector<double> volumes;
//     std::vector<double> lengths;
//     std::vector<double> radii;
//     std::vector<double> dialations;
//     std::vector<double> dialation_rates;

//     // Properties with num_voronoi entries
//     Vector3DStack kappas;
//     Vector3DStack rest_kappas;
//     std::vector<double> rest_vornoi_lengths;
//     std::vector<double> voronoi_dialations;

//     // Matrix-valued properties
//     self.internal_stress = internal_stress
//     self.internal_couple = internal_couple
// };

// class Rod
// {
// private: // Members
//     std::vector<Eigen::Vector3d> curve_points;
//     std::vector<Eigen::Quaterniond> quaternions;
//     std::vector<double> radii;

// public: // Constructor
//     Rod(
//         std::vector<Eigen::Vector3d>, std::vector<Eigen::Quaterniond>, std::vector<double>
//     );

// public: // Methods
//     std::vector<double> get_segment_lengths() const;
//     void resample(const double);
//     void apply_taper(const TaperFunc&);

//     const std::vector<Eigen::Vector3d>& get_curve() const;
//     const std::vector<Eigen::Quaterniond>& get_quaternions() const;
//     const std::vector<double>& get_radii() const;

//     size_t num_vertices() const;
//     size_t num_edges() const;
//     size_t num_interiors() const;

// public: // Static methods
//     static std::vector<double> material_coords_from_lengths(
//         const std::vector<double>&,
//         const bool
//     );
// };

// class GrowingRod
// {
// private: // Members
//     std::vector<Rod> rods;
//     const double growth_zone_length;
//     const bool grow_in_place;

// public: // Constructor
//     GrowingRod(Rod, const double, const bool);

// public: // Methods
//     void growth_step(
//         const std::vector<Eigen::Vector3d>& Delta,
//         const GrowthRateFunc& gr_func,
//         const TaperFunc& taper_func
//     )
//     {
//         auto& init_rod = rods.back();

//         const auto num_points = init_rod.num_vertices();
//         const auto num_edges = init_rod.num_edges();
//         const auto num_interiors = init_rod.num_interiors();
//         utils::nice_assert(num_interiors > 0, "Need at least one interior to update");


//     }
// };
// } // End namespace cosserat
// } // End namespace math
