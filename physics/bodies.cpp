#include "physics/bodies.hpp"

#include <utility>

#include "physics/dynamics_kinematics.hpp"

#include "utils/assertions.hpp"

namespace cosserat::physics {

BodyVariantWrapper::BodyVariantWrapper(BodyVariantPtr body) : m_body(std::move(body))
{
    utils::nice_assert(m_body != nullptr, "Body pointer can't be null");
}

BodyVariant& BodyVariantWrapper::variant()
{
    // The constructor rejects a null body, so this can only fire on a handle
    // that has been moved from.
    utils::nice_assert(m_body != nullptr, "Body handle is empty");
    return *m_body;
}

void BodyVariantWrapper::update_kinematics(double time, double scale)
{
    std::visit(
        [&](auto& body){dynamics::update_kinematics(body, time, scale);}, variant()
    );
}

void BodyVariantWrapper::update_dynamics(double time, double scale)
{
    std::visit(
        [&](auto& body){dynamics::update_dynamics(body, time, scale);}, variant()
    );
}

void BodyVariantWrapper::update_accelerations(double time, double dt)
{
    std::visit([&](auto& body){body.update_accelerations(time, dt);}, variant());
}

void BodyVariantWrapper::compute_internal_forces_and_torques(double time)
{
    std::visit(
        [&](auto& body){body.compute_internal_forces_and_torques(time);}, variant()
    );
}

void BodyVariantWrapper::zero_out_external_forces_and_torques(double time)
{
    std::visit(
        [&](auto& body){body.zero_out_external_forces_and_torques(time);}, variant()
    );
}

BodyVariant& BodyVariantWrapper::body() {return variant();}

const BodyVariant& BodyVariantWrapper::body() const
{
    utils::nice_assert(m_body != nullptr, "Body handle is empty");
    return *m_body;
}

bool BodyVariantWrapper::refers_to_same_body_as(const BodyVariantWrapper& other) const
{
    return m_body == other.m_body;
}
} // End namespace cosserat::physics
