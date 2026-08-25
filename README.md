# Cosserat

A C++20 library for simulating slender elastic bodies as Cosserat rods, with
rigid obstacles, contact, and the machinery to run a scene until it stops
moving.

A Cosserat rod is a curve carrying an orientation at every point. That extra
orientation is what lets a single one-dimensional object bend, twist, stretch
and shear, which is why the model suits hair, cables, catheters, muscle,
tentacles, and anything else long and thin that does not stay straight. The
formulation here follows Gazzola et al., *Forward and inverse problems in the
mechanics of soft filaments*, Royal Society Open Science 5(6), 2018, and began
as a port of the reference implementation, [PyElastica][pyelastica].

It is no longer only a port. The rod mechanics, joints, boundary conditions and
rod-to-rod contact are faithful to the reference and checked against it
numerically. Everything to do with arbitrary triangle meshes — mesh rigid
bodies, signed distance fields, mesh contact, mass properties from geometry —
has no counterpart there and is checked against closed-form answers instead.

[pyelastica]: https://github.com/GazzolaLab/PyElastica

---

## Contents

- [Why this exists](#why-this-exists)
- [Getting started](#getting-started)
- [A first program](#a-first-program)
- [How the library is laid out](#how-the-library-is-laid-out)
- [The physics](#the-physics)
- [Mesh bodies and contact](#mesh-bodies-and-contact)
- [Running a simulation](#running-a-simulation)
- [Stopping a simulation](#stopping-a-simulation)
- [Output and visualisation](#output-and-visualisation)
- [Examples](#examples)
- [Building and testing](#building-and-testing)
- [Installing](#installing)
- [Design notes](#design-notes)
- [Verification](#verification)
- [Known limitations](#known-limitations)
- [References](#references)

---

## Why this exists

The reference implementation is Python and NumPy with Numba on the hot paths.
It is excellent, and it is the definition of correct behaviour as far as this
project is concerned. What it does not offer is a C++ library you can link
into something else, and it has no notion of an arbitrary mesh obstacle.

This library provides both. The parts that correspond to the reference agree
with it to roughly fourteen significant digits over long runs; the parts that
do not correspond to anything there are new, and are documented as such
wherever they depart.

---

## Getting started

Nothing needs fetching. Every dependency is vendored in the repository, so a
fresh clone builds immediately:

```sh
git clone <repository-url> cosserat
cd cosserat
./install.sh ~/.local
```

That builds the library and installs it under `~/.local/cosserat`. Point it
anywhere; a system prefix needs `sudo`:

```sh
sudo ./install.sh /usr/local
```

Run `./install.sh --help` for the options. The useful ones are `--with-tests`,
which runs the suite before installing anything, and `--build-type Debug`,
which keeps the assertions live.

**Requirements.** A C++20 compiler (GCC 13 or Clang 16 onward), CMake 3.20 or
newer, and nothing else. Doxygen is used if present and skipped if not.

**Platforms.** Built and tested on Linux with GCC and on macOS with Clang. The
two disagree about enough to be worth naming: GCC raises two warnings from
inside Eigen that no amount of `SYSTEM` marking suppresses, because they come
from the optimiser rather than the front end, and they are demoted for GCC
only; and the relative run path that lets an installed tree be moved is spelled
`$ORIGIN` on ELF and `@loader_path` on Mach-O.

---

## A first program

A rod, given a shove, damped, and left to settle:

```cpp
#include <cosserat/physics/bodies.hpp>
#include <cosserat/physics/damping.hpp>
#include <cosserat/physics/rods.hpp>
#include <cosserat/simulation/simulation_graph.hpp>
#include <cosserat/simulation/solver.hpp>
#include <cosserat/simulation/stop_criteria.hpp>

#include <cstdio>
#include <memory>

int main()
{
    using namespace cosserat;

    // A metre of rod, fifty millimetres across, lying along +z.
    auto rod = std::make_shared<physics::BodyVariant>(
        physics::straight_cosserat_rod(
            /* elements */ 8,
            /* start    */ Eigen::Vector3d::Zero(),
            /* direction*/ Eigen::Vector3d::UnitZ(),
            /* normal   */ Eigen::Vector3d::UnitY(),
            /* length   */ 1.0,
            /* radius   */ 0.05,
            /* density  */ 1000.0,
            /* youngs   */ 1.0e6,
            /* respect_radii */ false,
            /* tolerance     */ 1e-12));

    simulation::SimulationGraph graph;
    graph.add_body("rod", physics::BodyVariantWrapper(rod));
    graph.dampen("rod", physics::UniformAnalyticalDamper(5.0, 1e-3));
    graph.finalize();

    std::get<physics::CosseratRod>(*rod)
        .mutable_velocities().col(0).setConstant(1.0);

    // Stop once nothing moves faster than a millimetre a second for a
    // twentieth of a second, or at twenty seconds regardless.
    auto criterion = simulation::settled_when_slow<simulation::SimulationGraph>(
        /* threshold           */ 1e-3,
        /* required_time_below */ 0.05,
        /* max_time            */ 20.0);

    simulation::Solver<simulation::SimulationGraph> solver(1e-3);
    const double reached = solver.full_solve(graph, criterion, 0.0);

    std::printf("%s at t = %.3f\n",
                criterion.settled() ? "settled" : "gave up", reached);
}
```

Build it against an installed copy:

```sh
g++ -std=c++20 main.cpp \
    -I$HOME/.local/cosserat/include \
    -L$HOME/.local/cosserat/lib \
    -lsimulation -lphysics -lmath -lutils \
    -Wl,-rpath,$HOME/.local/cosserat/lib
```

or from CMake:

```cmake
find_package(cosserat REQUIRED)
target_link_libraries(my_target PRIVATE
    cosserat::simulation cosserat::physics cosserat::math cosserat::utils)
```

---

## How the library is laid out

Four libraries, each depending only on those above it.

| Library | Holds | Depends on |
|---|---|---|
| `utils` | assertions, binary file writing | — |
| `math` | linear algebra, geometry, distance fields | `utils` |
| `physics` | bodies, forces, constraints, contact | `math` |
| `simulation` | the graph, the solver, diagnostics | `physics` |

The split is about dependency direction rather than subject matter. Anything
purely geometric goes in `math`, even when it exists only to serve the physics:
a signed distance field over a triangle mesh is geometry, so it lives there,
while the contact force computed from it is physics and lives one level up.

Headers are spelled the same inside the library and out:

```cpp
#include <cosserat/physics/rods.hpp>
```

which is why one include directory is enough for a consumer.

### The public headers

**`cosserat/math/`** — `linalg`, `functions`, `finite_difference`, `indexing`,
`types`, `minimum_distance`, `signed_distance_field`, `triangle_mesh_field`.

**`cosserat/physics/`** — `rods`, `rigid_body`, `mesh_body`, `bodies`,
`forces`, `damping`, `constraints`, `joints`, `contacts`, `rod_mesh_contact`,
`mass_properties`, `quantities`, `dynamics_kinematics`, `plane`, `coords`.

**`cosserat/simulation/`** — `simulation_graph`, `solver`, `stop_criteria`,
`diagnostics`, `logging`.

---

## The physics

### State

A rod of `n` elements carries `n + 1` nodes and `n` elements. Positions,
velocities, accelerations and the external force accumulator live on the nodes;
orientations, angular velocities and the torque accumulator live on the
elements. That split is what the discretisation asks for and it is worth
knowing before reading any of the kernels.

Orientation is a rotation matrix per element, stored so that its rows are the
material basis expressed in lab coordinates. A frame therefore carries a lab
vector into the material frame, and its transpose carries a material vector out.

### Strain

Two strain measures. Shear and stretch,

$$\boldsymbol{\sigma} = e\,(\mathbf{Q}\mathbf{t}) - \hat{\mathbf{z}}$$

whose third component is the axial strain and reduces to the dilatation less
one where there is no shear; and curvature $\boldsymbol{\kappa}$, the rotation
rate of the material frame along the rod, which carries both bending and twist.

Internal loads follow from the strains through the shearing and bending
matrices, and both are taken relative to a rest state, so a rod built curved
rests curved.

### What acts on a rod

**Forces** — gravity, endpoint loads, uniform and sinusoidal torques.
**Dampers** — several analytical forms; damping is applied inside the rate
constraint step rather than as a phase of its own, so a damper cannot scale
away a velocity a boundary condition has just pinned.
**Constraints** — fixing positions and orientations at chosen nodes and
elements, from a single clamped end to an arbitrary selection.
**Joints** — free, hinge, fixed, and fixed with a prescribed rest rotation, so
two rods can be held at an angle rather than only in line.
**Contact** — rod to rod, rod to itself, rod to sphere, rod to cylinder, rod to
plane with and without anisotropic friction, cylinder to plane, and rod to
mesh.

---

## Mesh bodies and contact

This is the part with no counterpart in the reference implementation.

### A body from a mesh

`MeshBody` takes a closed triangle mesh and a density and works out everything
else. Volume, centre of mass and the full inertia tensor come from integrating
over the solid the surface bounds, by the algorithm of Mirtich (1996) in the
triangle-specialised form given by Eberly. The mesh is then carried into its
principal frame, because a rigid body storing three moments of inertia is
implicitly expressed in the frame where the inertia tensor is diagonal.

```cpp
auto mesh = math::make_box_mesh(centre, half_extents);
physics::MeshBody obstacle(mesh, /* density */ 1750.0,
                           /* field margin */ 0.05, /* validate */ true);
```

The mesh must be closed and wound outward. Both are checked, including the case
every edge-based test misses: a mesh wound uniformly inward is perfectly
consistent and merely inside out, and only the sign of the enclosed volume
tells it apart.

### Contact through a distance field

The rod is never tessellated. The mesh becomes a signed distance field and the
rod stays a chain of capsules, so contact is a point query minus a radius:

$$\gamma = r_e - d(\mathbf{x}), \qquad \hat{\mathbf{n}} = \nabla d(\mathbf{x})$$

Tessellating the rod would be worse in three separate ways: the rod deforms
every step, so any acceleration structure over it would need rebuilding a
million times a run; triangle-against-triangle is far more expensive than a
point query; and it would discard the per-element radius, which for a
volume-preserving rod varies along the rod and changes as it stretches.

Each element is **marched** rather than sampled. A signed distance function is
1-Lipschitz, so from a point at distance $d$ no surface lies within $d - r$, and
stepping that far cannot skip a contact. This is sphere tracing, from Hart
(1996), and it makes a clear element cost a handful of queries however long it
is.

The field is built in body coordinates and never rebuilt: a query point is
carried into the body frame and the resulting normal carried back out, so a
moving, rotating obstacle costs nothing extra.

### Choosing a backend

Two are supported behind one concept. `TriangleMeshField` queries the triangles
directly through a bounding volume hierarchy — exact, no precomputation, no
memory budget. A discretised grid such as Discregrid's satisfies the same
concept and answers in constant time regardless of triangle count.

Measured here, exact queries cost about 1 µs at 1,280 triangles, 2.9 µs at
20,480 and 14.5 µs at 327,680, against roughly 0.5 µs for a grid at any size.
The grid wins past about a hundred thousand triangles; below that it costs
discretisation error, a build step and tens to hundreds of megabytes to save a
factor of two or three. The exact backend is the default for that reason.

---

## Running a simulation

A `SimulationGraph` holds the bodies and everything that couples them. Bodies
are registered by name; rules are attached to those names; `finalize()` closes
the graph and validates that every rule can act on the bodies it was given.

```cpp
simulation::SimulationGraph graph;
graph.add_body("rod", physics::BodyVariantWrapper(rod));
graph.add_body("ground", physics::BodyVariantWrapper(ground));

graph.add_constraint("ground", pin_in_place(ground_body));
graph.add_forcing_to("rod", physics::GravityForceZ{});
graph.add_contact("rod", "ground",
                  physics::RodMeshContact(1e3, 1.0, 10.0, 0.6));
graph.collect_diagnostics("rod",
    simulation::BasicDiagnostics(output_path, "rod", 500));

graph.finalize();
```

Validation happens where the rule is registered rather than at the first step,
so a contact rule handed a body it cannot act on fails while the scene is being
described and not several thousand steps into a run.

### The solver

Position Verlet, second order and symplectic. One step, in order:

```
half kinematic step        positions and frames advance by dt/2
constrain_values(t)        boundary conditions pin configuration
compute_internal(t)        each body's internal loads
synchronize(t)             joints, forces and contacts accumulate
dynamic step               accelerations, then rates advance by dt
constrain_rates(t)         boundary conditions, then damping
half kinematic step        positions and frames advance by dt/2
constrain_values(t)        configuration pinned again
apply_callbacks(t, step)   diagnostics observe the finished step
zero out external loads    accumulators cleared
```

Three of those orderings carry consequences. Internal forces are computed
before `synchronize`, because contact between two rods reads them. Damping runs
inside `constrain_rates`, so it cannot undo a pinned rate. The external
accumulators are cleared last, after the callbacks, so a diagnostic records the
loads that produced the step it is writing.

A solver runs once. It carries the step count and the origin of its run, and
reusing one would either continue an old run under the guise of a new one or
discard the state that makes the returned times meaningful.

---

## Stopping a simulation

Either on the clock:

```cpp
solver.full_solve(graph, start_time, end_time);
```

or on the state:

```cpp
auto criterion = simulation::settled_when_slow<simulation::SimulationGraph>(
    /* threshold */ 1e-2, /* required_time_below */ 0.25, /* max_time */ 10.0);
const double reached = solver.full_solve(graph, criterion, 0.0);
if (not criterion.settled()) { /* it ran out of time, still moving */ }
```

The criterion watches a quantity that is zero at rest and stops once that
quantity has stayed below a threshold for long enough. Two parameters, both
physical: the threshold says what counts as motion rather than numerical
residue, and the required time says how long the quiet must last before it is
believed. A threshold alone will not do, because contact leaves a scene
jittering rather than dead and the measure dips below any sensible threshold at
the turn of every rock.

### Which quantity

`max_material_point_speed` is the default, and it is a speed rather than an
energy for two reasons. It is **mass independent**, so one threshold means the
same thing for every body in a scene — an energy threshold in a scene holding a
gram of rod and a hundred kilograms of ground is really a statement about the
ground. And its residual level is **more predictable**: across a sweep varying
rod radius, density and contact stiffness, the level a scene settles to varied
by a factor of nineteen for speed against a hundred and sixty for kinetic
energy.

It also counts spin. A single-node rigid body turning in place moves no node at
all, so a nodal speed reads zero for a tumbling obstacle.

Kinetic energy and axial strain are available too, in
`cosserat/physics/quantities.hpp`, along with the per-body forms.

### Choosing a threshold

Above the level the scene settles to, or the run never stops. That level is a
property of the scene and worth measuring once: run to a fixed time, watch the
quantity, and pick a threshold above the floor it reaches. On the worked example
here the material speed comes to rest at a few millimetres a second, and the
default threshold is a centimetre a second.

The required time must exceed the longest period of anything still oscillating.
A pendulum's speed passes through zero twice a swing, so a shorter one can land
entirely inside a quiet part and call a swinging rod settled.

---

## Output and visualisation

Diagnostics write raw little-endian `float64` alongside a small JSON descriptor,
one directory per recorded step and one subdirectory per body:

```
step_000000000_st_0.000/
    rod/     positions.bin  frames.bin  radii.bin  (+ .md.json each)
    ground/  positions.bin  frames.bin  mesh_vertices.bin  mesh_triangles.bin
step_000002000_st_0.020/
    rod/     positions.bin  frames.bin  radii.bin
    ground/  positions.bin  frames.bin
```

A mesh body writes its triangles **once**, on the first step it records, and its
pose on every step after. A rigid body's shape is constant in its own frame, so
there is nothing to repeat; what varies is the pose, and both position and
orientation are needed to place a vertex:

$$\mathbf{v}_{world} = \mathbf{c}(t) + \mathbf{Q}(t)^{T}\mathbf{v}_{body}$$

### The Python side

`cosserat_python_tools` reads that output and renders it:

```sh
python3 scripts/make_rod_animation.py \
    --rod_data_dir /tmp/cosserat_logs/meshes/basic_square_meshes \
    --output_file_name scene.mp4
```

Rods and mesh bodies are recognised by what each wrote and drawn together —
a rod is swept into a tube, a mesh body is drawn as its triangles.

Optional arguments worth knowing: `--focus_height` crops the view to a band
above the lowest geometry, which helps when a wide flat ground would otherwise
set the scale of an equal-aspect view; `--elevation` and `--azimuth` aim the
camera, and a low elevation shows a rod resting on something far better than
looking down at it does.

---

## Examples

Built by default into `build/bin`. Every one takes `--help`.

**`joint_cases_fixed_joint`** — two rods joined end to end, driven by a
sinusoidal endpoint force. A port of the reference example of the same name.

**`joint_cases_fixed_joint_torsion`** — two rods meeting at a right angle,
driven by a uniform torque. The joint holds the corner through a rest rotation
taken from the rods as built. Also a port, and checked against it numerically.

**`meshes_basic_square_meshes`** — a rod falling under gravity onto a meshed
ground and a meshed block, coming to rest as a ramp propped against the block.
Nothing here corresponds to the reference: both obstacles are arbitrary closed
meshes seen through distance fields, and the run stops when the scene stops
moving rather than at a fixed time.

---

## Building and testing

```sh
cmake -S cosserat -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
cd build && ctest --output-on-failure
```

**Build in Debug when running the tests.** Assertions are compiled out under
`NDEBUG`, and a great many tests are death tests that check an assertion fires.
Against a Release build those tests do not fail informatively; they simply stop
testing anything.

| Option | Default | Effect |
|---|---|---|
| `BUILD_TESTING` | `ON` | build the test suite |
| `BUILD_EXAMPLES` | `ON` | build the example programs |
| `BUILD_DOCS` | `ON` | build documentation if Doxygen is present |
| `INSTALL_TESTS` | `OFF` | install test binaries alongside the examples |
| `ENABLE_OPTIMIZATION` | `ON` | optimisation flags on library and examples |
| `ENABLE_PROFILING` | `OFF` | profiling instrumentation |

Test and example targets are discovered rather than listed. A test at
`physics/test/test_rods.cpp` becomes `test_physics_rods`, and an example at
`examples/meshes/basic_square_meshes.cpp` becomes `meshes_basic_square_meshes`,
so a name is only ever unique within its directory and two directories may use
the same one.

Run a subset by label:

```sh
ctest -L physics
ctest -R RodMeshContact
```

---

## Installing

```sh
./install.sh /usr/local
```

produces

```
/usr/local/cosserat/lib          libutils.so, libmath.so, libphysics.so, libsimulation.so
/usr/local/cosserat/bin          the example programs
/usr/local/cosserat/include      cosserat/... plus the vendored headers
/usr/local/cosserat/lib/cmake    a find_package config
```

Everything sits under a `cosserat` directory inside the prefix, so an install
never scatters files through a shared `/usr/local/include`.

The installed binaries find their libraries through a **relative** run path, so
the whole `cosserat` directory can be moved or renamed without breaking. The
spelling of that path is platform specific — `$ORIGIN` on ELF, `@loader_path`
on Mach-O — and `install.sh` runs each installed program once after installing
to confirm it loads, because linking and loading are separate steps and a run
path wrong for the platform passes the first and fails the second.

The vendored dependencies our headers expose — Eigen, nlohmann/json, tsl
ordered-map, spdlog, cxxopts — are installed alongside. Without them the promise
that one include path suffices would be false: a consumer would get as far as
`<cosserat/physics/rods.hpp>` and fail on `<Eigen/Core>`.

---

## Design notes

**Compile-time polymorphism throughout.** No virtual functions anywhere. Bodies
and rules are held in `std::variant` and dispatched through `std::visit`;
requirements are expressed as concepts. A rule that cannot act on a body is a
compile error where it can be, and a clear assertion where it cannot.

**Const accessors, `mutable_` writers.** Every body exposes `positions()` and
`mutable_positions()`, so a mutation is visible at the call site.

**Assertions are load-bearing.** `nice_assert` checks preconditions that would
otherwise produce quietly wrong physics: a frame that is not a rotation, a mesh
that is not closed, a joint index that does not exist. They compile out under
`NDEBUG`.

**Deliberate departures are marked.** Where behaviour differs from the
reference implementation, the reason is documented with `@warning` rather than
silently corrected. Two examples: the reference's cylinder bounding box takes
an absolute value after summing rotated extents rather than inside the sum,
which undersizes the box, and that is reproduced for parity and documented; and
where two capsule axes intersect exactly, the reference produces a NaN normal,
while this library skips the pair.

---

## Verification

Three kinds of check, applied where each is possible.

**Numerical parity with the reference.** Both joint examples were run against
PyElastica step for step. Positions and frames agree to about 1e-13 over
hundreds of thousands of steps, and the disagreement does not grow:

```
fixed_joint          dt=1e-4, 5000 steps    worst disagreement  1.0e-13
fixed_joint_torsion  dt=1e-5, 20000 steps   worst disagreement  4.9e-13
```

The harness that produced those numbers is small and reusable, and is the right
first move when porting anything further.

**Closed-form answers where there is no reference.** A box of twelve triangles
*is* a box, so its distance field must agree with the analytic one to machine
precision, and its mass properties must match `V = 8abc` and
`I = m/3 \cdot diag(b^2+c^2, a^2+c^2, a^2+b^2)` exactly. Both do. A triangulated
sphere is only an approximation, so what is required there is that the error
falls as the mesh refines.

**Mutation testing on anything load-bearing.** A test that passes when the code
is deliberately broken is not testing what it claims to. Every significant
kernel here has had its central line inverted or removed to check the suite
notices. That practice has found real gaps: a contact test suite that passed
with the body-frame rotation removed, and a strain quantity whose rest-state
subtraction no test exercised because every rod in the suite rested unstrained.

The suite is **1,151 tests** across 29 executables, plus 200 on the Python side.

---

## Known limitations

**Mesh contact is rod-to-mesh only.** There is no mesh-to-mesh, mesh-to-sphere,
mesh-to-cylinder or mesh-to-plane contact, so two mesh bodies pass through one
another and a mesh body resting on a plane falls through it. Mesh obstacles are
therefore usually pinned with a constraint. A mesh body is otherwise a complete
rigid body — real mass and inertia from its own geometry, responding to forces,
dampers, constraints and joints — so only its contact coverage is partial.

**A mesh body's `radius()` and `length()` are nominal.** A rigid body requires
both to be positive, and an arbitrary mesh has neither, so the bounding sphere
stands in. Nothing reads them, because the primitive contact rules explicitly
exclude bodies carrying a distance field, but a new rule reading `radius()`
would silently get a meaningless number.

**Principal axes are only defined up to sign.** A body built from a mesh is
carried into its principal frame, and an eigenvector's sign is arbitrary, so an
axis-aligned box can come back with its frame turned by a half turn and its
vertices turned to match. The pair always reconstructs the same world geometry;
do not assume a mesh keeps the orientation it was modelled in.

**Joints attach at one node of a rigid body.** A rigid body has a single node
and element, so a joint can only attach there.

**Rendering is matplotlib.** The Python animation is a painter's algorithm with
no depth buffer, which cannot resolve interpenetrating geometry however finely
faces are subdivided. It is fine for scenes of a few separated bodies and is
not a general renderer.

---

## References

Gazzola, Dudte, McCormick, Mahadevan, *Forward and inverse problems in the
mechanics of soft filaments*, Royal Society Open Science 5(6), 2018. The
formulation this library implements.

Mirtich, *Fast and Accurate Computation of Polyhedral Mass Properties*, Journal
of Graphics Tools 1(2), 1996, pp. 31–50. Mass, centre of mass and inertia from
a closed mesh. Eberly's *Polyhedral Mass Properties (Revisited)* (2009) gives
the triangle-specialised form used here.

Bærentzen & Aanæs, *Signed distance computation using the angle weighted
pseudonormal*, IEEE TVCG 11(3), 2005. Why a signed distance needs a closed,
consistently wound surface, and how the sign is determined near edges and
vertices.

Hart, *Sphere tracing: a geometric method for the antialiased ray tracing of
implicit surfaces*, The Visual Computer 12(10), 1996. The marching used to test
a rod element against a distance field without missing thin features.

Macklin et al., *Local Optimization for Robust Signed Distance Field
Collision*, 2020. On why point-sampling a distance field for collision is
fragile, and what to do instead.

Preclik, Popa & Rüde, *Regularizing a Time-Stepping Method for Rigid Multibody
Dynamics*, ECCOMAS Multibody Dynamics, 2011. The regularised friction model used
by every contact rule here.

Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005. Standard
reference for the segment and capsule distance routines.
