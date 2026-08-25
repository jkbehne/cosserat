"""@file test_rigid_triangle_mesh.py

@brief Tests for @ref rigid_triangle_mesh and for loading mesh bodies.

Three things have to hold together. A posed mesh has to place its vertices in
the world correctly. The loader has to tell a mesh body from a rod by what each
one wrote, and has to carry a shape written on one step forward to every step
after it. And the animator has to draw a triangle mesh at all, which it cannot
do through the grid path a rod uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosserat_python_tools.meshes.rigid_triangle_mesh import RigidTriangleMesh
from cosserat_python_tools.utils.binary_file_utils import (
    BINARY_EXTENSION,
    COLUMN_MAJOR,
    METADATA_EXTENSION,
    get_bodies_from_dir,
    get_bodies_from_step_dir,
    has_mesh_shape,
    is_mesh_like,
    is_rod_like,
)
from cosserat_python_tools.utils.mesh_animation import (
    SURFACE_KIND,
    TRIANGLE_KIND,
    as_drawable,
    drawable_points,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def unit_cube() -> tuple[np.ndarray, np.ndarray]:
    """@brief A closed cube of half extent one, as vertices and triangles.

    @return The eight vertices and the twelve triangles.
    """
    vertices = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    faces = [
        (0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ]
    return vertices, np.array(faces, dtype=np.int64)


def write_pair(
    stem: Path, matrix: np.ndarray, storage_order: str = COLUMN_MAJOR
) -> Path:
    """@brief Writes one matrix as the binary and metadata pair on disk.

    @param stem Path without an extension.
    @param matrix Two dimensional array to store.
    @param storage_order Order recorded in the metadata.
    @return The stem, for chaining.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    order = "F" if storage_order == COLUMN_MAJOR else "C"
    stem.with_name(stem.name + BINARY_EXTENSION).write_bytes(
        np.asarray(matrix, dtype=np.float64).flatten(order=order).tobytes()
    )
    metadata: dict[str, Any] = {
        "scalar_type": "double",
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "batches": 1,
        "storage_order": storage_order,
    }
    stem.with_name(stem.name + METADATA_EXTENSION).write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return stem


def write_mesh_body(
    body_dir: Path,
    position: np.ndarray,
    frame: np.ndarray | None = None,
    with_shape: bool = True,
) -> None:
    """@brief Writes what a mesh body records for one step.

    @param body_dir Directory for the body within a step.
    @param position Position of the body's origin.
    @param frame Body frame; identity when omitted.
    @param with_shape Whether to write the triangles as well as the pose.
    """
    frame = np.eye(3) if frame is None else frame
    write_pair(body_dir / "positions", np.asarray(position).reshape(1, 3))
    write_pair(body_dir / "frames", np.asarray(frame).reshape(3, 3))
    if with_shape:
        vertices, triangles = unit_cube()
        write_pair(body_dir / "mesh_vertices", vertices)
        write_pair(body_dir / "mesh_triangles", triangles.astype(float))


def write_rod_body(body_dir: Path, num_elements: int = 4) -> None:
    """@brief Writes what a rod records for one step.

    @param body_dir Directory for the body within a step.
    @param num_elements Elements in the rod.
    """
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = np.linspace(0.0, 1.0, num_elements + 1)
    write_pair(body_dir / "positions", positions)
    frames = np.tile(np.eye(3), (num_elements, 1))
    stem = body_dir / "frames"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_name(stem.name + BINARY_EXTENSION).write_bytes(
        np.concatenate(
            [np.eye(3).flatten(order="F") for _ in range(num_elements)]
        ).tobytes()
    )
    stem.with_name(stem.name + METADATA_EXTENSION).write_text(
        json.dumps({
            "scalar_type": "double", "rows": 3, "cols": 3,
            "batches": num_elements, "storage_order": COLUMN_MAJOR,
        }), encoding="utf-8")
    del frames
    write_pair(body_dir / "radii", np.full((num_elements, 1), 0.05))


def step_name(step: int, time: float) -> str:
    """@brief The directory name the writer would produce for a step.

    @param step Step index.
    @param time Simulation time.
    @return The directory name.
    """
    return f"step_{step:09d}_st_{time:.3f}"


# ---------------------------------------------------------------------------
# RigidTriangleMesh
# ---------------------------------------------------------------------------


def test_a_mesh_at_the_origin_is_its_own_body_frame() -> None:
    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(vertices, triangles, np.zeros(3), np.eye(3))

    world, faces = mesh.to_triangles()
    np.testing.assert_allclose(world, vertices)
    np.testing.assert_array_equal(faces, triangles)


def test_translating_the_pose_translates_every_vertex() -> None:
    vertices, triangles = unit_cube()
    shift = np.array([2.0, -3.0, 1.0])
    mesh = RigidTriangleMesh(vertices, triangles, shift, np.eye(3))

    world, _ = mesh.to_triangles()
    np.testing.assert_allclose(world, vertices + shift)


# The stored frame carries a world vector into the body, so its transpose
# carries a body vector out. Getting that backwards turns the body the wrong
# way, which a symmetric shape would hide, so this uses an asymmetric one.
def test_the_frame_is_applied_as_its_transpose() -> None:
    vertices = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    triangles = np.array([[0, 1, 2]])
    turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    mesh = RigidTriangleMesh(vertices, triangles, np.zeros(3), turn)
    world, _ = mesh.to_triangles()

    expected = np.array([turn.T @ vertex for vertex in vertices])
    np.testing.assert_allclose(world, expected, atol=1e-12)


def test_indices_may_arrive_as_floats() -> None:
    """The binary format carries only doubles, so indices come back as floats."""
    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(
        vertices, triangles.astype(float), np.zeros(3), np.eye(3))

    assert mesh.triangles.dtype.kind == "i"
    np.testing.assert_array_equal(mesh.triangles, triangles)


def test_with_pose_keeps_the_shape_and_changes_the_placement() -> None:
    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(vertices, triangles, np.zeros(3), np.eye(3))

    moved = mesh.with_pose(np.array([5.0, 0.0, 0.0]), np.eye(3))

    np.testing.assert_allclose(moved.vertices, mesh.vertices)
    np.testing.assert_array_equal(moved.triangles, mesh.triangles)
    lower, upper = moved.world_bounds()
    np.testing.assert_allclose(lower, [4.0, -1.0, -1.0])
    np.testing.assert_allclose(upper, [6.0, 1.0, 1.0])


def test_world_bounds_follow_the_pose() -> None:
    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(
        vertices, triangles, np.array([1.0, 2.0, 3.0]), np.eye(3))

    lower, upper = mesh.world_bounds()
    np.testing.assert_allclose(lower, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(upper, [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "vertices,triangles,message",
    [
        (np.zeros((8, 2)), np.zeros((1, 3)), "vertices must have shape"),
        (np.zeros((2, 3)), np.zeros((1, 3)), "at least three vertices"),
        (np.zeros((8, 3)), np.zeros((1, 2)), "triangles must have shape"),
        (np.zeros((8, 3)), np.zeros((0, 3)), "at least one triangle"),
    ],
)
def test_rejects_badly_shaped_geometry(vertices, triangles, message) -> None:
    with pytest.raises(ValueError, match=message):
        RigidTriangleMesh(vertices, triangles, np.zeros(3), np.eye(3))


def test_rejects_an_index_outside_the_vertices() -> None:
    vertices, _ = unit_cube()
    with pytest.raises(ValueError, match="outside the 8 vertices"):
        RigidTriangleMesh(
            vertices, np.array([[0, 1, 99]]), np.zeros(3), np.eye(3))


def test_rejects_indices_that_are_not_whole_numbers() -> None:
    vertices, _ = unit_cube()
    with pytest.raises(ValueError, match="whole numbers"):
        RigidTriangleMesh(
            vertices, np.array([[0.0, 1.0, 2.5]]), np.zeros(3), np.eye(3))


def test_rejects_a_frame_that_is_not_a_rotation() -> None:
    vertices, triangles = unit_cube()
    with pytest.raises(ValueError, match="not a rotation"):
        RigidTriangleMesh(
            vertices, triangles, np.zeros(3), 2.0 * np.eye(3))


def test_rejects_non_finite_input() -> None:
    vertices, triangles = unit_cube()
    broken = vertices.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="vertices must be finite"):
        RigidTriangleMesh(broken, triangles, np.zeros(3), np.eye(3))


# ---------------------------------------------------------------------------
# Telling one body from another
# ---------------------------------------------------------------------------


def test_a_rod_is_recognised_by_its_radii(tmp_path: Path) -> None:
    body = tmp_path / "rod"
    write_rod_body(body)

    assert is_rod_like(body)
    assert not is_mesh_like(body)


def test_a_mesh_body_is_recognised_by_a_pose_without_radii(tmp_path: Path) -> None:
    body = tmp_path / "block"
    write_mesh_body(body, np.zeros(3))

    assert is_mesh_like(body)
    assert not is_rod_like(body)
    assert has_mesh_shape(body)


# On any step after the first, a mesh body writes only its pose. It is still a
# mesh body, and must still be recognised as one.
def test_a_mesh_body_without_its_shape_is_still_recognised(tmp_path: Path) -> None:
    body = tmp_path / "block"
    write_mesh_body(body, np.zeros(3), with_shape=False)

    assert is_mesh_like(body)
    assert not has_mesh_shape(body)


# ---------------------------------------------------------------------------
# Loading a run
# ---------------------------------------------------------------------------


def write_run(
    root: Path, num_steps: int = 4, shape_on_first: bool = True
) -> Path:
    """@brief Writes a run holding one rod and one mesh body.

    @param root Directory to write the step directories beneath.
    @param num_steps Number of steps to write.
    @param shape_on_first Whether the mesh body writes its shape at all.
    @return The root, for chaining.
    """
    for step in range(num_steps):
        step_dir = root / step_name(step * 10, 0.01 * step)
        write_rod_body(step_dir / "rod")
        write_mesh_body(
            step_dir / "block",
            np.array([0.1 * step, 0.0, 0.0]),
            with_shape=shape_on_first and step == 0,
        )
    return root


def test_loads_rods_and_mesh_bodies_together(tmp_path: Path) -> None:
    write_run(tmp_path)

    frames = get_bodies_from_dir(tmp_path)

    assert len(frames) == 4
    for _, bodies in frames:
        kinds = sorted(type(body).__name__ for body in bodies)
        assert kinds == ["DiscreteCosseratRod", "RigidTriangleMesh"]


# The shape is written only on the first step recorded, so every later frame
# depends on it having been carried forward.
def test_the_shape_is_carried_forward_to_every_later_step(tmp_path: Path) -> None:
    write_run(tmp_path)

    frames = get_bodies_from_dir(tmp_path)

    for _, bodies in frames:
        mesh = next(b for b in bodies if isinstance(b, RigidTriangleMesh))
        assert mesh.num_vertices == 8
        assert mesh.num_triangles == 12


def test_each_frame_gets_its_own_pose(tmp_path: Path) -> None:
    write_run(tmp_path)

    frames = get_bodies_from_dir(tmp_path)

    seen = []
    for _, bodies in frames:
        mesh = next(b for b in bodies if isinstance(b, RigidTriangleMesh))
        seen.append(float(mesh.position[0]))
    np.testing.assert_allclose(seen, [0.0, 0.1, 0.2, 0.3])


def test_frames_come_back_in_step_order(tmp_path: Path) -> None:
    write_run(tmp_path, num_steps=12)

    times = [time for time, _ in get_bodies_from_dir(tmp_path)]

    assert times == sorted(times)


# A mesh body whose shape was never written cannot be placed, so it is skipped
# rather than guessed at.
def test_a_mesh_body_with_no_shape_anywhere_is_skipped(tmp_path: Path) -> None:
    write_run(tmp_path, shape_on_first=False)

    frames = get_bodies_from_dir(tmp_path)

    for _, bodies in frames:
        assert all(not isinstance(b, RigidTriangleMesh) for b in bodies)
        assert len(bodies) == 1


def test_a_step_with_nothing_drawable_is_rejected(tmp_path: Path) -> None:
    step_dir = tmp_path / step_name(0, 0.0)
    (step_dir / "mystery").mkdir(parents=True)

    with pytest.raises(ValueError, match="Couldn't find any drawable body"):
        get_bodies_from_step_dir(step_dir, {})


def test_a_step_with_nothing_drawable_can_be_tolerated(tmp_path: Path) -> None:
    step_dir = tmp_path / step_name(0, 0.0)
    (step_dir / "mystery").mkdir(parents=True)

    time, bodies = get_bodies_from_step_dir(step_dir, {}, raise_if_empty=False)

    assert time == pytest.approx(0.0)
    assert bodies == []


def test_a_tree_with_no_steps_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()

    with pytest.raises(ValueError, match=r"No step_\* directories"):
        get_bodies_from_dir(tmp_path)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def test_a_mesh_resolves_to_triangles_and_a_rod_to_a_surface(tmp_path: Path) -> None:
    write_run(tmp_path)
    _, bodies = get_bodies_from_dir(tmp_path)[0]

    kinds = {}
    for body in bodies:
        kind, _ = as_drawable(body, num_circumferential=8)
        kinds[type(body).__name__] = kind

    assert kinds["RigidTriangleMesh"] == TRIANGLE_KIND
    assert kinds["DiscreteCosseratRod"] == SURFACE_KIND


def test_a_triangle_drawable_carries_world_vertices_and_indices() -> None:
    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(
        vertices, triangles, np.array([1.0, 0.0, 0.0]), np.eye(3))

    kind, (world, faces) = as_drawable(mesh)

    assert kind == TRIANGLE_KIND
    assert world.shape == (8, 3)
    assert faces.shape == (12, 3)
    np.testing.assert_allclose(world, vertices + np.array([1.0, 0.0, 0.0]))


# Bounding a scene must not care which kind each body is.
def test_drawable_points_bounds_both_kinds(tmp_path: Path) -> None:
    write_run(tmp_path)
    _, bodies = get_bodies_from_dir(tmp_path)[0]

    for body in bodies:
        points = drawable_points(body, num_circumferential=8)
        assert points.ndim == 2
        assert points.shape[1] == 3
        assert np.isfinite(points).all()


def test_an_unrecognised_object_is_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        as_drawable(object())


# ---------------------------------------------------------------------------
# Drawing a scene as one depth sorted whole
# ---------------------------------------------------------------------------


def test_both_kinds_of_body_become_triangles() -> None:
    """Everything is drawn together, so everything has to be triangles."""
    from cosserat_python_tools.utils.mesh_animation import faces_of

    vertices, triangles = unit_cube()
    mesh = RigidTriangleMesh(vertices, triangles, np.zeros(3), np.eye(3))

    mesh_faces = faces_of(mesh)
    assert mesh_faces.shape == (12, 3, 3)

    # A surface grid is cut into two triangles per quad.
    grid = tuple(np.zeros((4, 5)) for _ in range(3))
    grid_faces = faces_of(grid)
    assert grid_faces.shape == (2 * 3 * 4, 3, 3)


def test_subdivision_splits_only_the_oversized_faces() -> None:
    from cosserat_python_tools.utils.mesh_animation import subdivide_faces

    big = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
    small = np.array([[[0.0, 0, 0], [0.01, 0, 0], [0.0, 0.01, 0]]])

    assert len(subdivide_faces(small, 0.1)) == 1
    assert len(subdivide_faces(big, 0.1)) > 1
    # Disabled by a non positive limit.
    assert len(subdivide_faces(big, 0.0)) == 1


def test_subdivision_preserves_the_surface() -> None:
    """Splitting a triangle must not move it: same plane, same total area."""
    from cosserat_python_tools.utils.mesh_animation import subdivide_faces

    face = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
    split = subdivide_faces(face, 0.2)

    def area(faces):
        edge_one = faces[:, 1] - faces[:, 0]
        edge_two = faces[:, 2] - faces[:, 0]
        return 0.5 * np.linalg.norm(np.cross(edge_one, edge_two), axis=1).sum()

    assert area(split) == pytest.approx(area(face))
    # Every piece still lies in the original plane.
    assert np.allclose(split[..., 2], 0.0)


def test_subdivision_terminates_on_a_pathological_limit() -> None:
    """A limit no triangle can meet costs bounded work, not unbounded."""
    from cosserat_python_tools.utils.mesh_animation import subdivide_faces

    face = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
    split = subdivide_faces(face, 1e-9)

    assert len(split) == 4 ** 8  # MAX_SUBDIVISION_PASSES, then it gives up
