#pragma once

#include <concepts>
#include <cstdlib>
#include <iostream>
#include <source_location>

namespace cosserat::utils {
template<typename T>
concept streamable = requires(std::ostream& stream, const T& val)
{
    {stream << val} -> std::convertible_to<std::ostream&>;
};

template<std::same_as<bool> ConditionType, streamable MessageType>
void nice_assert(
    [[maybe_unused]] const ConditionType condition,
    [[maybe_unused]] const MessageType& message,
    [[maybe_unused]] const std::source_location location = std::source_location::current()
)
{
    #ifndef NDEBUG
    if (not condition) [[unlikely]]
    {
        std::cerr << "Assertion Failed: ";
        std::cerr << message;
        std::cerr << "\n" << "File: " << location.file_name() << "\n"
                  << "Line: " << location.line() << "\n"
                  << "Function: " << location.function_name() << "\n";
        std::cout.flush();
        std::abort();
    }
    #endif
}
} // End namespace cosserat::utils
