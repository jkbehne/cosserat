#pragma once

#include <variant>

#include "physics/rigid_body.hpp"
#include "physics/rods.hpp"

namespace cosserat::physics {
using BodyVariant = std::variant<CosseratRod, RigidBody, Sphere, Cylinder>;
} // End namespace cosserat::physics
