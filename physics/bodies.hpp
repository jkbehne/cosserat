#pragma once

#include <memory>
#include <variant>

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

namespace cosserat::physics {

using BodyVariant = std::variant<CosseratRod, RigidBody, Sphere, Cylinder>;
using BodyVariantPtr = std::shared_ptr<BodyVariant>;

class BodyVariantWrapper
{
private: // Members
    BodyVariantPtr m_body;

public: // Members
    explicit BodyVariantWrapper(BodyVariantPtr body);

    void update_kinematics(double time, double scale);

    void update_dynamics(double time, double scale);

    void update_acceleration(double time, double scale);

    void compute_internal_forces_and_torques(double time);

    void zero_out_external_forces_and_torques(double time);
};
} // End namespace cosserat::physics
