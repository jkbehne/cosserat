#pragma once

#include <concepts>
#include <variant>

namespace cosserat::utils {
template <typename T>
concept MutableReference = std::is_lvalue_reference_v<T> and not
    std::is_const_v<std::remove_reference_t<T>>;

// The magic "Overloaded" helper struct
template<class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
// Deduction guide required for C++17, but implicit in C++20 and newer
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>; 

template <typename T>
struct is_variant : std::false_type {};

template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_variant_v = is_variant<std::remove_cvref_t<T>>::value;

template <typename T>
concept Variant = is_variant_v<T>;
} // End namespace cosserat::utils
