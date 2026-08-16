// #include "cosserat.hpp"

// #include "utils/assertions.hpp"

// namespace math {
// namespace cosserat {

// // Rod constructor
// Rod::Rod(
//     std::vector<Eigen::Vector3d> points,
//     std::vector<Eigen::Quaterniond> quats,
//     std::vector<double> radii_
// )
// : curve_points(std::move(points)), quaternions(std::move(quats)), radii(std::move(radii_))
// {
//     const auto num_points = curve_points.size();
//     utils::nice_assert(num_points >= 2, "Number of points in rod must be >=2");
//     utils::nice_assert(
//         quaternions.size() == num_points - 1, "Expected 1 less quaternion than points"
//     );
//     utils::nice_assert(
//         radii.size() == num_points, "Expected to have a radius for each curve point"
//     );
// }

// // Rod public methods
// std::vector<double> Rod::get_segment_lengths() const
// {
//     const auto num_points = curve_points.size();
//     std::vector<double> lengths;
//     lengths.reserve(num_points - 1);
//     for (int idx = 0; idx < num_points - 1; ++idx)
//     {
//         const Eigen::Vector3d diff = curve_points[idx + 1] - curve_points[idx];
//         const double length = diff.norm();
//         lengths.push_back(length);
//     }
//     return lengths;
// }

// void Rod::resample(const double length_threshold)
// {
//     const auto lengths = get_segment_lengths();
//     std::vector<int> idxs;
//     for (int idx = 0; idx < lengths.size(); ++idx)
//     {
//         if (lengths[idx] > length_threshold) {idxs.push_back(idx);}
//     }

//     for (int idx = 0; idx < idxs.size(); ++idx)
//     {
//         const auto base_idx = idxs[idx] + idx + 1;
//         auto curve_iter = curve_points.begin() + base_idx;
//         const Eigen::Vector3d midpoint = 0.5 * (curve_points[base_idx] + curve_points[base_idx - 1]);
//         curve_points.emplace(curve_iter, midpoint);

//         auto quat_iter = quaternions.begin() + base_idx - 1;
//         const Eigen::Quaterniond quat_copy = *quat_iter;
//         quaternions.emplace(quat_iter, quat_copy);

//         auto radii_iter = radii.begin() + base_idx - 1;
//         const auto radii_copy = *radii_iter;
//         radii.emplace(radii_iter, radii_copy);
//     }
// }

// void Rod::apply_taper(const TaperFunc& taper)
// {
//     const auto lengths = get_segment_lengths();
//     const auto num_points = lengths.size() + 1;
//     utils::nice_assert(
//         radii.size() == num_points, "Expected to have as many radii as curve points"
//     );
//     const auto norm_coords = material_coords_from_lengths(lengths, true /* normalize */);
//     for (int idx = 0; idx < norm_coords.size(); ++idx)
//     {
//         radii[idx] = taper(norm_coords[idx]);
//     }
// }

// // Rod "getter" methods
// const std::vector<Eigen::Vector3d>& Rod::get_curve() const {return curve_points;}
// const std::vector<Eigen::Quaterniond>& Rod::get_quaternions() const {return quaternions;}
// const std::vector<double>& Rod::get_radii() const {return radii;}

// // Rod convenience methods
// size_t Rod::num_vertices() const {return curve_points.size();}
// size_t Rod::num_edges() const {return num_vertices() - 1;}
// size_t Rod::num_interiors() const
// {
//     const auto Nedges = num_edges();
//     if (Nedges > 0) return Nedges - 1;
//     return 0;
// }

// // Rod static methods
// std::vector<double> Rod::material_coords_from_lengths(
//     const std::vector<double>& lengths,
//     const bool normalize
// )
// {
//     utils::nice_assert(lengths.size() >= 1, "Must have at least one length");
//     const auto num_points = lengths.size() + 1;
//     std::vector<double> coords;
//     coords.reserve(num_points);
//     double sum = 0.0;
//     for (int idx = 0; idx < num_points; ++idx)
//     {
//         coords.push_back(sum);
//         if (idx < lengths.size()) sum += lengths[idx];
//     }
//     if (normalize)
//     {
//         utils::nice_assert(sum > 0.0, "Sum of lengths < 0");
//         for (int idx = 0; idx < num_points; ++idx)
//         {
//             coords[idx] /= sum;
//         }
//     }
//     return coords;
// }
// } // End namespace cosserat
// } // End namespace math
