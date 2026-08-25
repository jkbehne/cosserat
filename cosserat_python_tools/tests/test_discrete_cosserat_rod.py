"""
@file test_discrete_cosserat_rod.py
@brief Tests for @ref discrete_cosserat_rod.

The mesh is verified by geometric properties rather than by comparing vertex
arrays, because the vertex values depend on the circumferential resolution and
on the seam position while the properties do not. The two that matter are the
radius of each ring and the miter condition at a bend, and both are checked
directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cosserat_python_tools.rods.discrete_cosserat_rod import (
    DiscreteCosseratRod,
    FRAME_ORTHONORMAL_TOLERANCE,
    plot_rod,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def straight_rod(
    num_elements: int = 8, length: float = 1.0, radius: float = 0.05
) -> DiscreteCosseratRod:
    """
    @brief A straight rod along +z with identity frames and uniform radius.

    @param num_elements Number of elements.
    @param length Total length.
    @param radius Uniform element radius.
    @return The rod.
    """
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, length, num_elements + 1)
    frames = np.tile(np.eye(3), (num_elements, 1, 1))
    radii = np.full(num_elements, radius)
    return DiscreteCosseratRod(positions, frames, radii)


def elbow_rod(turn_degrees: float, radius: float = 0.1) -> DiscreteCosseratRod:
    """
    @brief Two elements meeting at a single bend in the xz plane.

    The bend is placed at the middle node so the miter has somewhere to act.

    @param turn_degrees Angle the rod turns through at the joint.
    @param radius Uniform element radius.
    @return The rod, whose middle node is index 1.
    """
    turn = np.radians(turn_degrees)
    first = np.array([np.sin(-turn / 2.0), 0.0, np.cos(turn / 2.0)])
    second = np.array([np.sin(turn / 2.0), 0.0, np.cos(turn / 2.0)])

    positions = np.zeros((3, 3))
    positions[0] = -first
    positions[1] = np.zeros(3)
    positions[2] = second

    # Frames whose tangent row follows each element.
    frames = np.stack([frame_with_tangent(first), frame_with_tangent(second)])
    radii = np.full(2, radius)
    return DiscreteCosseratRod(positions, frames, radii)


def frame_with_tangent(tangent: np.ndarray) -> np.ndarray:
    """
    @brief An orthonormal frame whose third row is the given tangent.

    @param tangent Unit vector to use as the material tangent.
    @return A rotation matrix with rows normal, binormal, tangent.
    """
    tangent = tangent / np.linalg.norm(tangent)
    seed = np.array([0.0, 1.0, 0.0])
    if abs(seed @ tangent) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    normal = seed - (seed @ tangent) * tangent
    normal /= np.linalg.norm(normal)
    binormal = np.cross(tangent, normal)
    return np.stack([normal, binormal, tangent])


def radius_about_axis(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """
    @brief Perpendicular distance from each point to a line.

    @param points Points, shape @c (..., 3).
    @param origin A point on the line.
    @param axis Unit direction of the line.
    @return Distances, shape @c (...).
    """
    offsets = points - origin
    along = offsets @ axis
    perpendicular = offsets - along[..., None] * axis
    return np.linalg.norm(perpendicular, axis=-1)


def ring_vertices(rod: DiscreteCosseratRod, node: int, resolution: int = 64):
    """
    @brief The vertices of one ring, without the duplicated seam column.

    @param rod The rod.
    @param node Index of the node whose ring is wanted.
    @param resolution Vertices around the ring.
    @return An array of shape @c (resolution, 3).
    """
    x_grid, y_grid, z_grid = rod.to_mesh(resolution, close_seam=False)
    return np.stack([x_grid[node], y_grid[node], z_grid[node]], axis=-1)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_accepts_a_well_formed_rod():
    rod = straight_rod(num_elements=5)
    assert rod.num_nodes == 6
    assert rod.num_elements == 5
    assert rod.positions.shape == (6, 3)
    assert rod.radii.shape == (5,)


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((4, 2)),  # not three columns
        np.zeros((4,)),  # not two dimensional
        np.zeros((4, 3, 1)),  # too many dimensions
    ],
)
def test_rejects_badly_shaped_positions(positions):
    with pytest.raises(ValueError, match="positions must have shape"):
        DiscreteCosseratRod(positions, np.tile(np.eye(3), (3, 1, 1)), np.ones(3))


def test_rejects_a_rod_with_fewer_than_two_nodes():
    with pytest.raises(ValueError, match="at least two nodes"):
        DiscreteCosseratRod(np.zeros((1, 3)), np.zeros((0, 3, 3)), np.zeros(0))


def test_rejects_a_radius_count_that_does_not_match_the_nodes():
    positions = np.zeros((5, 3))
    positions[:, 2] = np.arange(5)
    frames = np.tile(np.eye(3), (4, 1, 1))
    # Five nodes means four elements, so five radii is the classic off by one.
    with pytest.raises(ValueError, match="implies 4 elements"):
        DiscreteCosseratRod(positions, frames, np.ones(5))


def test_rejects_radii_that_are_not_one_dimensional():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    with pytest.raises(ValueError, match="one dimensional"):
        DiscreteCosseratRod(positions, np.tile(np.eye(3), (2, 1, 1)), np.ones((2, 1)))


def test_rejects_a_frame_count_that_does_not_match_the_nodes():
    positions = np.zeros((5, 3))
    positions[:, 2] = np.arange(5)
    with pytest.raises(ValueError, match=r"frames must have shape \(4, 3, 3\)"):
        DiscreteCosseratRod(positions, np.tile(np.eye(3), (3, 1, 1)), np.ones(4))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_positive_radii(bad):
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    radii = np.array([0.1, bad])
    with pytest.raises(ValueError, match="strictly positive"):
        DiscreteCosseratRod(positions, np.tile(np.eye(3), (2, 1, 1)), radii)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite_input(bad):
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.tile(np.eye(3), (2, 1, 1))

    broken_positions = positions.copy()
    broken_positions[1, 0] = bad
    with pytest.raises(ValueError, match="positions must be finite"):
        DiscreteCosseratRod(broken_positions, frames, np.ones(2))

    broken_radii = np.array([0.1, bad])
    with pytest.raises(ValueError, match="radii must be finite"):
        DiscreteCosseratRod(positions, frames, broken_radii)

    broken_frames = frames.copy()
    broken_frames[0, 0, 0] = bad
    with pytest.raises(ValueError, match="frames must be finite"):
        DiscreteCosseratRod(positions, broken_frames, np.ones(2))


def test_rejects_coincident_nodes():
    positions = np.array([[0.0, 0, 0], [0.0, 0, 1], [0.0, 0, 1]])
    with pytest.raises(ValueError, match="element 1 has zero length"):
        DiscreteCosseratRod(positions, np.tile(np.eye(3), (2, 1, 1)), np.ones(2))


# scipy quietly orthonormalises whatever it is given, so a frame that is not a
# rotation would be repaired rather than reported without this check.
def test_rejects_frames_that_are_not_orthonormal():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.tile(np.eye(3), (2, 1, 1)).astype(float)
    frames[1, 0, 0] = 2.0

    with pytest.raises(ValueError, match="frame 1 is not orthonormal"):
        DiscreteCosseratRod(positions, frames, np.ones(2))


def test_accepts_frames_within_the_orthonormality_tolerance():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.tile(np.eye(3), (2, 1, 1)).astype(float)
    frames[1, 0, 0] += 0.1 * FRAME_ORTHONORMAL_TOLERANCE

    DiscreteCosseratRod(positions, frames, np.ones(2))  # must not raise


def test_rejects_reflections():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.tile(np.eye(3), (2, 1, 1)).astype(float)
    frames[1] = np.diag([1.0, 1.0, -1.0])  # orthonormal, but determinant -1

    with pytest.raises(ValueError, match="must be rotations, not reflections"):
        DiscreteCosseratRod(positions, frames, np.ones(2))


# ---------------------------------------------------------------------------
# Node radii
# ---------------------------------------------------------------------------


def test_node_radii_have_one_entry_per_node():
    rod = straight_rod(num_elements=7)
    assert rod.node_radii.shape == (rod.num_nodes,)


def test_end_nodes_take_their_single_neighbouring_element():
    positions = np.zeros((4, 3))
    positions[:, 2] = np.arange(4)
    radii = np.array([0.3, 0.2, 0.1])
    rod = DiscreteCosseratRod(positions, np.tile(np.eye(3), (3, 1, 1)), radii)

    assert rod.node_radii[0] == pytest.approx(0.3)
    assert rod.node_radii[-1] == pytest.approx(0.1)


def test_interior_nodes_average_their_two_elements():
    positions = np.zeros((4, 3))
    positions[:, 2] = np.arange(4)
    radii = np.array([0.3, 0.2, 0.1])
    rod = DiscreteCosseratRod(positions, np.tile(np.eye(3), (3, 1, 1)), radii)

    assert rod.node_radii[1] == pytest.approx(0.25)
    assert rod.node_radii[2] == pytest.approx(0.15)


# The point of averaging rather than assigning across: an element's rendered
# midpoint radius comes back equal to its own radius, with no half element
# shift. Exact for a linear ramp.
def test_a_linear_radius_ramp_renders_each_element_at_its_own_radius():
    num_elements = 10
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, 1.0, num_elements + 1)
    radii = np.linspace(0.20, 0.05, num_elements)
    rod = DiscreteCosseratRod(
        positions, np.tile(np.eye(3), (num_elements, 1, 1)), radii
    )

    midpoints = 0.5 * (rod.node_radii[:-1] + rod.node_radii[1:])
    # The ends cannot be exact: they have no neighbour on one side.
    assert midpoints[1:-1] == pytest.approx(radii[1:-1])


def test_a_uniform_rod_has_uniform_node_radii():
    rod = straight_rod(num_elements=6, radius=0.07)
    assert rod.node_radii == pytest.approx(np.full(rod.num_nodes, 0.07))


# ---------------------------------------------------------------------------
# Node frames
# ---------------------------------------------------------------------------


def test_node_frames_have_one_entry_per_node():
    rod = straight_rod(num_elements=5)
    assert len(rod.node_rotations) == rod.num_nodes
    assert rod.node_quaternions.shape == (rod.num_nodes, 4)


def test_end_node_frames_copy_their_single_element():
    rod = straight_rod(num_elements=4)
    element = rod.element_rotations.as_matrix()
    node = rod.node_rotations.as_matrix()

    assert node[0] == pytest.approx(element[0])
    assert node[-1] == pytest.approx(element[-1])


# Half way between two frames about a common axis is half the angle.
def test_interior_node_frames_are_half_way_between_their_elements():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.stack(
        [
            np.eye(3),
            Rotation.from_rotvec([0.0, 0.0, np.radians(90.0)]).as_matrix(),
        ]
    )
    rod = DiscreteCosseratRod(positions, frames, np.ones(2))

    middle = rod.node_rotations[1]
    angle = np.linalg.norm(middle.as_rotvec())
    assert angle == pytest.approx(np.radians(45.0))


# Component-wise averaging of two rotation matrices is not a rotation. Slerp is,
# which is why the frames are carried as quaternions.
def test_interior_node_frames_stay_rotations_under_a_large_turn():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    frames = np.stack(
        [
            np.eye(3),
            Rotation.from_rotvec([0.0, 0.0, np.radians(170.0)]).as_matrix(),
        ]
    )
    rod = DiscreteCosseratRod(positions, frames, np.ones(2))

    middle = rod.node_rotations[1].as_matrix()
    assert middle @ middle.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(middle) == pytest.approx(1.0)

    naive = 0.5 * (frames[0] + frames[1])
    assert not np.allclose(naive @ naive.T, np.eye(3), atol=1e-6)


def test_slerp_takes_the_shortest_path():
    positions = np.zeros((3, 3))
    positions[:, 2] = np.arange(3)
    # 350 degrees the long way is 10 degrees the short way.
    frames = np.stack(
        [
            np.eye(3),
            Rotation.from_rotvec([0.0, 0.0, np.radians(350.0)]).as_matrix(),
        ]
    )
    rod = DiscreteCosseratRod(positions, frames, np.ones(2))

    angle = np.degrees(np.linalg.norm(rod.node_rotations[1].as_rotvec()))
    assert angle == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Mesh shape and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", [3, 8, 16, 33])
def test_mesh_grids_have_one_row_per_node_and_one_column_per_vertex(resolution):
    rod = straight_rod(num_elements=6)

    x_open, y_open, z_open = rod.to_mesh(resolution, close_seam=False)
    for grid in (x_open, y_open, z_open):
        assert grid.shape == (rod.num_nodes, resolution)

    x_closed, y_closed, z_closed = rod.to_mesh(resolution, close_seam=True)
    for grid in (x_closed, y_closed, z_closed):
        assert grid.shape == (rod.num_nodes, resolution + 1)


def test_closing_the_seam_duplicates_the_first_column():
    rod = straight_rod(num_elements=4)
    x_grid, y_grid, z_grid = rod.to_mesh(12, close_seam=True)

    for grid in (x_grid, y_grid, z_grid):
        assert grid[:, 0] == pytest.approx(grid[:, -1])


def test_rejects_too_few_vertices_around_the_rod():
    rod = straight_rod()
    with pytest.raises(ValueError, match="at least three vertices"):
        rod.to_mesh(2)


def test_mesh_is_finite():
    rod = straight_rod(num_elements=5)
    for grid in rod.to_mesh(16):
        assert np.isfinite(grid).all()


# ---------------------------------------------------------------------------
# Mesh geometry
# ---------------------------------------------------------------------------


def test_a_straight_rod_is_a_cylinder_of_the_requested_radius():
    rod = straight_rod(num_elements=8, radius=0.05)
    x_grid, y_grid, _ = rod.to_mesh(32)

    distances = np.hypot(x_grid, y_grid)
    assert distances == pytest.approx(0.05, abs=1e-12)


def test_every_ring_sits_at_its_node():
    rod = straight_rod(num_elements=6, length=2.0)
    _, _, z_grid = rod.to_mesh(16)

    assert z_grid.mean(axis=1) == pytest.approx(rod.positions[:, 2])


def test_each_ring_takes_its_own_node_radius():
    num_elements = 6
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, 1.0, num_elements + 1)
    radii = np.linspace(0.20, 0.05, num_elements)
    rod = DiscreteCosseratRod(
        positions, np.tile(np.eye(3), (num_elements, 1, 1)), radii
    )

    x_grid, y_grid, _ = rod.to_mesh(32, close_seam=False)
    ring_radii = np.hypot(x_grid, y_grid).mean(axis=1)
    assert ring_radii == pytest.approx(rod.node_radii)


def test_a_tapered_rod_narrows_monotonically():
    num_elements = 6
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, 1.0, num_elements + 1)
    radii = np.linspace(0.20, 0.05, num_elements)
    rod = DiscreteCosseratRod(
        positions, np.tile(np.eye(3), (num_elements, 1, 1)), radii
    )

    x_grid, y_grid, _ = rod.to_mesh(24, close_seam=False)
    ring_radii = np.hypot(x_grid, y_grid).mean(axis=1)
    assert (np.diff(ring_radii) < 0.0).all()


def test_translating_the_rod_translates_the_mesh():
    rod = straight_rod(num_elements=5)
    shift = np.array([1.0, -2.0, 3.0])
    moved = DiscreteCosseratRod(rod.positions + shift, rod.element_rotations.as_matrix(),
                                rod.radii)

    original = np.stack(rod.to_mesh(16), axis=-1)
    translated = np.stack(moved.to_mesh(16), axis=-1)
    assert translated == pytest.approx(original + shift)


# ---------------------------------------------------------------------------
# Twist
#
# A circular tube is rotationally symmetric, so twist cannot change the surface.
# It does change which material point sits at which vertex, which is what makes
# a texture or a stripe follow the twist.
# ---------------------------------------------------------------------------


def test_twisting_rotates_the_vertices_around_the_same_surface():
    rod = straight_rod(num_elements=6, radius=0.05)

    twisted_frames = np.stack(
        [
            Rotation.from_rotvec([0.0, 0.0, np.radians(30.0 * index)]).as_matrix()
            @ np.eye(3)
            for index in range(rod.num_elements)
        ]
    )
    twisted = DiscreteCosseratRod(rod.positions, twisted_frames, rod.radii)

    plain_x, plain_y, _ = rod.to_mesh(64, close_seam=False)
    twist_x, twist_y, _ = twisted.to_mesh(64, close_seam=False)

    # The surface is unchanged: every vertex is still on the same cylinder.
    assert np.hypot(twist_x, twist_y) == pytest.approx(0.05, abs=1e-12)
    # But the vertices have moved around it.
    assert not np.allclose(twist_x[3], plain_x[3], atol=1e-6)


def test_the_seam_follows_the_material_normal():
    num_elements = 4
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, 1.0, num_elements + 1)

    turn = np.radians(45.0)
    frames = np.tile(
        Rotation.from_rotvec([0.0, 0.0, turn]).as_matrix(), (num_elements, 1, 1)
    )
    rod = DiscreteCosseratRod(positions, frames, np.full(num_elements, 0.1))

    # Row 0 of the frame is the material normal, and the ring angle is measured
    # from it, so the seam vertex lies along it.
    x_grid, y_grid, _ = rod.to_mesh(16, close_seam=False)
    seam = np.array([x_grid[2, 0], y_grid[2, 0], 0.0])
    expected = 0.1 * frames[2][0]
    assert seam == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Miter joints
#
# The defining property: viewed along either adjacent tangent, a mitered ring
# projects to a circle of exactly the node radius. That is what makes the two
# tube segments meet without overlapping inside the turn.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("turn_degrees", [10.0, 30.0, 60.0, 90.0, 140.0])
def test_a_mitered_ring_projects_to_a_circle_along_both_tangents(turn_degrees):
    radius = 0.1
    rod = elbow_rod(turn_degrees, radius=radius)
    ring = ring_vertices(rod, node=1, resolution=128)

    tangents = rod.element_tangents
    for tangent in (tangents[0], tangents[1]):
        distances = radius_about_axis(ring, rod.positions[1], tangent)
        assert distances == pytest.approx(radius, abs=1e-12)


@pytest.mark.parametrize("turn_degrees", [15.0, 45.0, 90.0])
def test_the_mitered_ring_lies_in_the_bisecting_plane(turn_degrees):
    rod = elbow_rod(turn_degrees)
    ring = ring_vertices(rod, node=1, resolution=64)

    tangents = rod.element_tangents
    bisector = tangents[0] + tangents[1]
    bisector /= np.linalg.norm(bisector)

    offsets = ring - rod.positions[1]
    assert offsets @ bisector == pytest.approx(0.0, abs=1e-12)


# The stretch is exactly 1/cos(theta/2) in the plane of the bend, and none at
# all across it.
@pytest.mark.parametrize("turn_degrees", [20.0, 50.0, 100.0])
def test_the_ring_is_stretched_only_in_the_plane_of_the_bend(turn_degrees):
    radius = 0.1
    rod = elbow_rod(turn_degrees, radius=radius)
    ring = ring_vertices(rod, node=1, resolution=256)

    offsets = ring - rod.positions[1]
    extents = np.abs(offsets).max(axis=0)

    half = np.radians(turn_degrees) / 2.0
    # The elbow bends in the xz plane, so x is stretched and y is not.
    assert extents[0] == pytest.approx(radius / np.cos(half), abs=1e-9)
    assert extents[1] == pytest.approx(radius, abs=1e-9)


def test_a_straight_rod_is_not_stretched_at_all():
    rod = straight_rod(num_elements=5, radius=0.05)
    x_grid, y_grid, _ = rod.to_mesh(64, close_seam=False)

    # Every vertex is exactly on the cylinder, so no ring was stretched.
    assert np.hypot(x_grid, y_grid) == pytest.approx(0.05, abs=1e-13)


def test_the_end_rings_are_square_to_the_rod():
    rod = elbow_rod(60.0)
    tangents = rod.element_tangents

    for node, tangent in ((0, tangents[0]), (2, tangents[1])):
        ring = ring_vertices(rod, node=node, resolution=32)
        offsets = ring - rod.positions[node]
        assert offsets @ tangent == pytest.approx(0.0, abs=1e-12)


def test_rejects_a_rod_that_doubles_back_on_itself():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0 + 1e-13]])
    # Nudge so the zero length check passes and the 180 degree check is reached.
    positions[2, 2] = 0.0
    frames = np.stack(
        [frame_with_tangent(np.array([0.0, 0.0, 1.0])),
         frame_with_tangent(np.array([0.0, 0.0, -1.0]))]
    )
    rod = DiscreteCosseratRod(positions, frames, np.full(2, 0.05))

    with pytest.raises(ValueError, match="doubles back on itself"):
        rod.to_mesh(16)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_element_tangents_follow_the_centreline():
    rod = elbow_rod(60.0)
    tangents = rod.element_tangents

    assert tangents.shape == (2, 3)
    assert np.linalg.norm(tangents, axis=1) == pytest.approx(1.0)
    expected = np.diff(rod.positions, axis=0)
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    assert tangents == pytest.approx(expected)


def test_lengths_match_the_node_spacing():
    rod = straight_rod(num_elements=5, length=2.5)
    assert rod.lengths == pytest.approx(np.full(5, 0.5))


def test_quaternions_round_trip_back_to_the_input_frames():
    rod = elbow_rod(40.0)
    recovered = Rotation.from_quat(rod.element_quaternions).as_matrix()
    assert recovered == pytest.approx(rod.element_rotations.as_matrix())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_rod_draws_without_error():
    import matplotlib

    matplotlib.use("Agg")
    rod = elbow_rod(45.0)

    axes = plot_rod(rod, num_circumferential=12)
    assert axes is not None


# A rod is long and thin, and matplotlib stretches each axis independently by
# default, which distorts the shape. The helper equalises them.
def test_plot_rod_uses_one_scale_for_every_axis():
    import matplotlib

    matplotlib.use("Agg")
    rod = straight_rod(num_elements=6, length=2.0, radius=0.02)

    axes = plot_rod(rod, num_circumferential=12)
    spans = [
        axes.get_xlim()[1] - axes.get_xlim()[0],
        axes.get_ylim()[1] - axes.get_ylim()[0],
        axes.get_zlim()[1] - axes.get_zlim()[0],
    ]
    assert spans[0] == pytest.approx(spans[1])
    assert spans[1] == pytest.approx(spans[2])
    # And the span covers the long axis of the rod.
    assert spans[2] >= 2.0


def test_plot_rod_accepts_an_existing_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")

    returned = plot_rod(straight_rod(), axes=axes)
    assert returned is axes
