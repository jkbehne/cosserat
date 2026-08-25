#pragma once

#include <concepts>

namespace cosserat::physics {

template<unsigned int CoordIdx>
struct CoordTag
{
public: // Constexpr
    static constexpr unsigned int idx = CoordIdx;
};

struct XTag : CoordTag<0> {};
struct YTag : CoordTag<1> {};
struct ZTag : CoordTag<2> {};

template<typename T>
concept IsXYZ = std::same_as<T, XTag>
             or std::same_as<T, YTag>
             or std::same_as<T, ZTag>;
} // End namespace cosserat::physics
