#include "physics/bodies.hpp"

#include <utility>

#include "physics/dynamics_kinematics.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {
BodyVariantWrapper::BodyVariantWrapper(BodyVariantPtr body) : m_body(std::move(body))
{
    utils::nice_assert(m_body != nullptr, "Body pointer can't be null");
}

void BodyVariantWrapper::update_kinematics(double time, double scale)
{
    std::visit([&](auto& body){dynamics::update_kinematics(body, time, scale);}, *m_body);
}

void BodyVariantWrapper::update_dynamics(double time, double scale)
{
    std::visit([&](auto& body){dynamics::update_dynamics(body, time, scale);}, *m_body);
}

void BodyVariantWrapper::update_acceleration(double time, double scale)
{
    std::visit([&](auto& body){body.update_accelerations(time, scale);}, *m_body);
}

void BodyVariantWrapper::compute_internal_forces_and_torques(double time)
{
    std::visit([&](auto& body){body.compute_internal_forces_and_torques(time);}, *m_body);
}

void BodyVariantWrapper::zero_out_external_forces_and_torques(double time)
{
    std::visit([&](auto& body){body.zero_out_external_forces_and_torques(time);}, *m_body);
}
} // End namespace cosserat::physics
