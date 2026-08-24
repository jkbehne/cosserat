"""@file binary_file_utils.py

@brief Reading simulation state written by the C++ @c utils::file_utils module.

Each state matrix is stored on disk as two files sharing a stem: a @c .bin
holding raw little-endian doubles exactly as Eigen laid them out, and a
@c .md.json describing the shape, the batch count and the storage order.

This module reverses that: given the pair, it returns a NumPy array of the
right shape and dtype. Arrays are squeezed, so a stored Eigen column vector
comes back one-dimensional rather than as an @c (n, 1) column.

On top of that, the module walks the directory tree the diagnostics produce:

@verbatim
  base/step_000000000_st_0.000/rod1/positions.bin
                                   /positions.md.json
                                   /frames.bin ...
                              /rod2/...
      step_000001000_st_0.100/...
@endverbatim

@ref get_cosserat_rods_from_dir turns that tree into the sequence of
@c (time, rods) pairs the animation utilities consume, ordered by step. The
step number is zero padded by the writer so that sorting names and sorting
steps agree, and this module relies on that: change the padding width and the
frames come back out of order.

@note Only @c double is written by the C++ side. Other scalar types are
      rejected rather than guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from cosserat_python_tools.meshes.rigid_triangle_mesh import (
    RigidTriangleMesh
)
from cosserat_python_tools.rods.discrete_cosserat_rod import (
    DiscreteCosseratRod
)

## @brief Extension used for the raw binary payload.
BINARY_EXTENSION: Final[str] = ".bin"

## @brief Extension used for the JSON metadata sidecar.
METADATA_EXTENSION: Final[str] = ".md.json"

## @brief Metadata scalar names mapped to their NumPy dtypes.
SCALAR_DTYPES: Final[dict[str, np.dtype]] = {
    "double": np.dtype(np.float64),
}

## @brief Metadata keys every sidecar must provide.
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "scalar_type",
    "rows",
    "cols",
    "batches",
    "storage_order",
)

## @brief Storage order names the reader understands.
ROW_MAJOR: Final[str] = "row_major"
COLUMN_MAJOR: Final[str] = "column_major"

## @brief Leading field of a step directory's name, as written by the C++ side.
STEP_DIRECTORY_PREFIX: Final[str] = "step"

## @brief Stems a body must have written for it to be read back as a rod.
ROD_STEM_NAMES: Final[tuple[str, ...]] = ("positions", "frames", "radii")

## @brief Stems every rigid body writes, whatever its shape.
POSE_STEM_NAMES: Final[tuple[str, ...]] = ("positions", "frames")

## @brief Stems a mesh body writes once, on the first step it records.
MESH_SHAPE_STEM_NAMES: Final[tuple[str, ...]] = ("mesh_vertices", "mesh_triangles")


def read_metadata(metadata_path: Path | str) -> dict[str, Any]:
    """@brief Loads and validates a metadata sidecar.

    @param metadata_path Path to the @c .md.json file.
    @return The parsed metadata as a dictionary.
    @throws FileNotFoundError If the file does not exist.
    @throws KeyError If a required key is missing.
    @throws ValueError If the scalar type, storage order or dimensions are not
            ones this reader can handle.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        metadata: dict[str, Any] = json.load(stream)

    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing:
        raise KeyError(f"Metadata {path} is missing required keys: {missing}")

    scalar_type = metadata["scalar_type"]
    if scalar_type not in SCALAR_DTYPES:
        raise ValueError(
            f"Metadata {path} has unsupported scalar_type {scalar_type!r}; "
            f"expected one of {sorted(SCALAR_DTYPES)}"
        )

    storage_order = metadata["storage_order"]
    if storage_order not in (ROW_MAJOR, COLUMN_MAJOR):
        raise ValueError(
            f"Metadata {path} has unsupported storage_order {storage_order!r}; "
            f"expected {ROW_MAJOR!r} or {COLUMN_MAJOR!r}"
        )

    for key in ("rows", "cols", "batches"):
        value = metadata[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Metadata {path} has non-integer {key}: {value!r}")
        if value < 1:
            raise ValueError(f"Metadata {path} has non-positive {key}: {value}")

    return metadata


def load_matrix(
    binary_path: Path | str, metadata_path: Path | str
) -> NDArray[np.float64]:
    """@brief Reads a binary payload back into a NumPy array.

    The array is reshaped according to the metadata and then squeezed, so a
    single @c (3, 1) Eigen vector returns with shape @c (3,) and a single
    @c (4, 3) matrix returns with shape @c (4, 3). A batched write of @c n
    matrices returns with shape @c (n, rows, cols), again squeezed.

    @param binary_path Path to the @c .bin file.
    @param metadata_path Path to the matching @c .md.json file.
    @return The reconstructed array, squeezed.
    @throws FileNotFoundError If either file does not exist.
    @throws KeyError If the metadata is missing a required key.
    @throws ValueError If the metadata is unusable, or if the binary file size
            does not match the shape the metadata describes.
    """
    metadata = read_metadata(metadata_path)

    path = Path(binary_path)
    if not path.is_file():
        raise FileNotFoundError(f"Binary file does not exist: {path}")

    dtype = SCALAR_DTYPES[metadata["scalar_type"]]
    rows: int = metadata["rows"]
    cols: int = metadata["cols"]
    batches: int = metadata["batches"]
    storage_order: str = metadata["storage_order"]

    expected_elements = batches * rows * cols
    expected_bytes = expected_elements * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Binary file {path} holds {actual_bytes} bytes but metadata "
            f"describes {batches} x {rows} x {cols} of {dtype.name}, "
            f"which needs {expected_bytes}"
        )

    flat = np.fromfile(path, dtype=dtype)
    if flat.size != expected_elements:
        raise ValueError(
            f"Binary file {path} yielded {flat.size} elements, expected "
            f"{expected_elements}"
        )

    if storage_order == ROW_MAJOR:
        # Each batch already runs row by row, which is NumPy's own default.
        reshaped = flat.reshape(batches, rows, cols)
    else:
        # Each batch runs column by column, so in memory it is a (cols, rows)
        # C-ordered block; transposing the last two axes restores (rows, cols).
        reshaped = flat.reshape(batches, cols, rows).transpose(0, 2, 1)

    # Made contiguous before squeezing, not after: the column-major branch
    # returns a transposed view, and ascontiguousarray would promote a squeezed
    # zero-dimensional result back to shape (1,).
    return np.squeeze(np.ascontiguousarray(reshaped))


def load_matrix_from_stem(stem: Path | str) -> NDArray[np.float64]:
    """@brief Reads a matrix given the stem shared by its two files.

    Convenience wrapper that appends @ref BINARY_EXTENSION and
    @ref METADATA_EXTENSION to @p stem and calls @ref load_matrix.

    @param stem Path without an extension, as passed to the C++ writer.
    @return The reconstructed array, squeezed.
    @throws FileNotFoundError If either file does not exist.
    @throws ValueError If the metadata and binary file disagree.
    """
    stem_path = Path(stem)
    return load_matrix(
        stem_path.with_name(stem_path.name + BINARY_EXTENSION),
        stem_path.with_name(stem_path.name + METADATA_EXTENSION),
    )


def find_stems(directory: Path | str) -> list[Path]:
    """@brief Lists the stems of every matrix pair in a directory.

    A stem is reported only when both its binary and metadata files are
    present, so a partially written step is skipped rather than half-read.

    @param directory Directory to scan; not searched recursively.
    @return Stems, sorted by name, with no extension attached.
    @throws NotADirectoryError If @p directory is not a directory.
    """
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    stems: list[Path] = []
    for metadata_file in sorted(path.glob(f"*{METADATA_EXTENSION}")):
        name = metadata_file.name[: -len(METADATA_EXTENSION)]
        stem = metadata_file.with_name(name)
        if stem.with_name(name + BINARY_EXTENSION).is_file():
            stems.append(stem)
    return stems


def load_directory(directory: Path | str) -> dict[str, NDArray[np.float64]]:
    """@brief Reads every matrix pair in a directory into a dictionary.

    @param directory Directory to scan; not searched recursively.
    @return Mapping from stem name to the reconstructed array.
    @throws NotADirectoryError If @p directory is not a directory.
    @throws ValueError If any pair in the directory is inconsistent.
    """
    return {stem.name: load_matrix_from_stem(stem) for stem in find_stems(directory)}

def get_cosserat_rods_from_step_dir(
    step_dir: Path | str,
    raise_if_no_rods: bool = True,
) -> tuple[float, list[DiscreteCosseratRod]]:
    """@brief Reads every rod written into one step directory.

    A step directory holds one subdirectory per body, each containing the
    matrix pairs that body wrote. A body is treated as a rod when it has all
    three of @c positions, @c frames and @c radii; anything else is skipped,
    which is how rigid bodies are passed over, since they write no radii.

    Bodies are visited in name order rather than in whatever order the
    filesystem hands back, so a body occupies the same position in the returned
    list at every step. Callers that style bodies by index, such as giving each
    rod its own colour, depend on that being stable across frames.

    @param step_dir Directory named @c step_<number>_st_<time>, as written by
           the C++ diagnostics.
    @param raise_if_no_rods When true, a step containing no rod-like body is an
           error rather than an empty result.
    @return The simulation time parsed from the directory name, paired with the
            rods found inside it.
    @throws NotADirectoryError If @p step_dir is not a directory.
    @throws ValueError If the directory holds no body subdirectories, if its
            name does not match the expected form, or if @p raise_if_no_rods is
            set and nothing rod-like was found.

    @note The time comes from the directory name, which the writer rounds to
          three decimals. It is therefore a label rather than the exact
          simulation time, which is fine for ordering and for animation but
          should not be treated as authoritative.
    """
    step_path = Path(step_dir).expanduser()
    if not step_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {step_path}")

    rod_list: list[DiscreteCosseratRod] = []
    body_dirs = sorted(
        body_dir for body_dir in step_path.iterdir() if body_dir.is_dir()
    )
    if len(body_dirs) < 1:
        raise ValueError(
            f"Expected at least one body directory in directory {step_path}"
        )

    for body_dir in body_dirs:
        available_stems = find_stems(body_dir)
        is_rod_like = all(
            body_dir / name in available_stems for name in ROD_STEM_NAMES
        )
        if is_rod_like:
            rod_list.append(
                DiscreteCosseratRod(
                    positions=load_matrix_from_stem(body_dir / "positions"),
                    frames=load_matrix_from_stem(body_dir / "frames"),
                    radii=load_matrix_from_stem(body_dir / "radii"),
                )
            )

    if len(rod_list) == 0 and raise_if_no_rods:
        raise ValueError(f"Couldn't find any rod-like object in {step_path}")

    return parse_step_time(step_path), rod_list


def parse_step_time(step_dir: Path | str) -> float:
    """@brief Recovers the simulation time from a step directory's name.

    The writer names each step directory @c step_<number>_st_<time>, so the
    time is the final underscore-separated field.

    @param step_dir The directory, or just its name.
    @return The time the name records.
    @throws ValueError If the name does not have the expected four fields, or
            if the final field is not a number.
    """
    name = Path(step_dir).name
    fields = name.split("_")
    if len(fields) != 4 or fields[0] != STEP_DIRECTORY_PREFIX or fields[2] != "st":
        raise ValueError(
            f"Expected a directory named step_<number>_st_<time>, got {name!r}"
        )
    try:
        return float(fields[-1])
    except ValueError as error:
        raise ValueError(
            f"Step directory {name!r} does not end in a numeric time"
        ) from error


def get_cosserat_rods_from_dir(
    directory: Path | str,
    raise_if_no_rods: bool = True,
) -> list[tuple[float, list[DiscreteCosseratRod]]]:
    """@brief Reads every step directory beneath a diagnostics root.

    The result is ordered by step directory name, which is also step order:
    the writer zero pads the step number precisely so that sorting by name and
    sorting by step agree. Reading in @c iterdir order instead returns the
    steps in whatever order the filesystem stored them, which is not sorted,
    and produces a frame sequence that jumps back and forth in time.

    The returned list of @c (time, rods) pairs is exactly the frame sequence
    the animation utilities consume.

    @param directory Root the diagnostics wrote beneath, holding one
           subdirectory per recorded step.
    @param raise_if_no_rods When true, a step containing no rod-like body is an
           error rather than an empty entry.
    @return One @c (time, rods) pair per step, in step order.
    @throws NotADirectoryError If @p directory is not a directory.
    @throws ValueError If no step directories are present, or if any step
            cannot be read.
    """
    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Sorted, and filtered to the writer's naming scheme so that unrelated
    # directories sitting alongside the output are ignored rather than parsed.
    step_dirs = sorted(
        step_dir
        for step_dir in dir_path.glob(f"{STEP_DIRECTORY_PREFIX}_*")
        if step_dir.is_dir()
    )
    if not step_dirs:
        raise ValueError(
            f"No {STEP_DIRECTORY_PREFIX}_* directories found in {dir_path}"
        )

    return [
        get_cosserat_rods_from_step_dir(
            step_dir=step_dir, raise_if_no_rods=raise_if_no_rods
        )
        for step_dir in step_dirs
    ]


def is_rod_like(body_dir: Path) -> bool:
    """@brief Whether a body directory holds a rod.

    A rod writes its radii alongside its pose, which nothing else does.

    @param body_dir Directory for one body within one step.
    @return True when every stem a rod writes is present.
    """
    available = find_stems(body_dir)
    return all(body_dir / name in available for name in ROD_STEM_NAMES)


def is_mesh_like(body_dir: Path) -> bool:
    """@brief Whether a body directory holds a mesh body.

    A mesh body writes a pose like any rigid body and no radii, since it has no
    single radius to write. Its triangles appear only in the first step it
    records, so their absence here says nothing: what identifies it on a later
    step is the pose without the radii.

    @param body_dir Directory for one body within one step.
    @return True when the directory holds a pose but no radii.
    """
    available = find_stems(body_dir)
    has_pose = all(body_dir / name in available for name in POSE_STEM_NAMES)
    has_radii = body_dir / "radii" in available
    return has_pose and not has_radii


def has_mesh_shape(body_dir: Path) -> bool:
    """@brief Whether a body directory carries the written triangles.

    @param body_dir Directory for one body within one step.
    @return True when both shape stems are present.
    """
    available = find_stems(body_dir)
    return all(body_dir / name in available for name in MESH_SHAPE_STEM_NAMES)


def load_mesh_shape(
    body_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """@brief Reads the body frame triangles a mesh body wrote.

    @param body_dir Directory for one body within one step.
    @return The vertices and the vertex indices, both as written.
    @throws FileNotFoundError If either file is missing.
    """
    return (
        load_matrix_from_stem(body_dir / "mesh_vertices"),
        load_matrix_from_stem(body_dir / "mesh_triangles"),
    )


def load_pose(body_dir: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """@brief Reads the position and orientation of a rigid body.

    A rigid body has one node and one element, so the arrays come back squeezed
    to a single position and a single frame.

    @param body_dir Directory for one body within one step.
    @return The position, shape @c (3,), and the frame, shape @c (3, 3).
    """
    return (
        load_matrix_from_stem(body_dir / "positions").reshape(3),
        load_matrix_from_stem(body_dir / "frames").reshape(3, 3),
    )


def get_bodies_from_step_dir(
    step_dir: Path | str,
    mesh_shapes: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    raise_if_empty: bool = True,
) -> tuple[float, list[object]]:
    """@brief Reads every drawable body from one step directory.

    Rods are read whole, since everything they need is written every step. A
    mesh body is read as its pose applied to a shape from @p mesh_shapes, which
    the caller accumulates as it walks the steps in order; a mesh body seen
    before its shape has been recorded is skipped rather than guessed at.

    @param step_dir Directory named @c step_<number>_st_<time>.
    @param mesh_shapes Shapes already found, keyed by body name. A shape found
           in this directory is added to it.
    @param raise_if_empty When true, a step with nothing drawable is an error.
    @return The simulation time parsed from the directory name, paired with the
            bodies found, ordered by body name.
    @throws NotADirectoryError If @p step_dir is not a directory.
    @throws ValueError If the directory holds no body subdirectories, or if
            @p raise_if_empty is set and nothing drawable was found.
    """
    step_path = Path(step_dir).expanduser()
    if not step_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {step_path}")

    body_dirs = sorted(
        body_dir for body_dir in step_path.iterdir() if body_dir.is_dir()
    )
    if len(body_dirs) < 1:
        raise ValueError(
            f"Expected at least one body directory in directory {step_path}"
        )

    bodies: list[object] = []
    for body_dir in body_dirs:
        if is_rod_like(body_dir):
            bodies.append(
                DiscreteCosseratRod(
                    positions=load_matrix_from_stem(body_dir / "positions"),
                    frames=load_matrix_from_stem(body_dir / "frames"),
                    radii=load_matrix_from_stem(body_dir / "radii"),
                )
            )
            continue

        if not is_mesh_like(body_dir):
            continue

        # The triangles are written once, on the first step recorded, so a
        # shape found here is kept for every step after.
        if has_mesh_shape(body_dir):
            mesh_shapes[body_dir.name] = load_mesh_shape(body_dir)

        shape = mesh_shapes.get(body_dir.name)
        if shape is None:
            continue

        vertices, triangles = shape
        position, frame = load_pose(body_dir)
        bodies.append(RigidTriangleMesh(vertices, triangles, position, frame))

    if len(bodies) == 0 and raise_if_empty:
        raise ValueError(f"Couldn't find any drawable body in {step_path}")

    return parse_step_time(step_path), bodies


def get_bodies_from_dir(
    directory: Path | str,
    raise_if_empty: bool = True,
) -> list[tuple[float, list[object]]]:
    """@brief Reads every step of a run, rods and mesh bodies alike.

    Walks the step directories in step order, which is also name order because
    the writer zero pads the step number. Order matters twice over here: it is
    the frame order of any animation, and it is what puts a mesh body's
    triangles, written only on the first step recorded, in hand before the
    later poses that need them.

    The result is the frame sequence the animation utilities consume.

    @param directory Root the diagnostics wrote beneath.
    @param raise_if_empty When true, a step with nothing drawable is an error.
    @return One @c (time, bodies) pair per step, in step order.
    @throws NotADirectoryError If @p directory is not a directory.
    @throws ValueError If no step directories are present, or if any step
            cannot be read.
    """
    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    step_dirs = sorted(
        step_dir
        for step_dir in dir_path.glob(f"{STEP_DIRECTORY_PREFIX}_*")
        if step_dir.is_dir()
    )
    if not step_dirs:
        raise ValueError(
            f"No {STEP_DIRECTORY_PREFIX}_* directories found in {dir_path}"
        )

    mesh_shapes: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    return [
        get_bodies_from_step_dir(
            step_dir=step_dir,
            mesh_shapes=mesh_shapes,
            raise_if_empty=raise_if_empty,
        )
        for step_dir in step_dirs
    ]
