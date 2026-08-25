"""@file test_binary_file_utils.py

@brief pytest suite for @ref binary_file_utils.

Files are constructed here rather than produced by the C++ writer, so the
suite runs standalone. The byte layouts written below mirror what Eigen
produces: column-major blocks run down the columns, row-major blocks run
across the rows, and batches are concatenated in order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosserat_python_tools.utils.binary_file_utils import (
    BINARY_EXTENSION,
    COLUMN_MAJOR,
    METADATA_EXTENSION,
    ROW_MAJOR,
    find_stems,
    get_cosserat_rods_from_dir,
    get_cosserat_rods_from_step_dir,
    load_directory,
    load_matrix,
    load_matrix_from_stem,
    parse_step_time,
    read_metadata,
)


def write_pair(
    stem: Path,
    values: np.ndarray,
    rows: int,
    cols: int,
    batches: int,
    storage_order: str,
    scalar_type: str = "double",
) -> Path:
    """@brief Writes a binary and metadata pair from an explicit flat buffer.

    @param stem Path without an extension.
    @param values Flat buffer written verbatim to the binary file.
    @param rows Row count recorded in the metadata.
    @param cols Column count recorded in the metadata.
    @param batches Batch count recorded in the metadata.
    @param storage_order Storage order recorded in the metadata.
    @param scalar_type Scalar name recorded in the metadata.
    @return The stem, for chaining.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_name(stem.name + BINARY_EXTENSION).write_bytes(
        np.asarray(values, dtype=np.float64).tobytes()
    )
    metadata: dict[str, Any] = {
        "scalar_type": scalar_type,
        "rows": rows,
        "cols": cols,
        "batches": batches,
        "storage_order": storage_order,
    }
    stem.with_name(stem.name + METADATA_EXTENSION).write_text(
        json.dumps(metadata, indent=4) + "\n", encoding="utf-8"
    )
    return stem


def write_matrix_pair(stem: Path, matrix: np.ndarray, storage_order: str) -> Path:
    """@brief Writes a single matrix in the requested storage order.

    @param stem Path without an extension.
    @param matrix Two-dimensional array to store.
    @param storage_order Either row-major or column-major.
    @return The stem, for chaining.
    """
    order = "C" if storage_order == ROW_MAJOR else "F"
    return write_pair(
        stem,
        matrix.flatten(order=order),
        matrix.shape[0],
        matrix.shape[1],
        1,
        storage_order,
    )


def write_batch_pair(stem: Path, batch: np.ndarray, storage_order: str) -> Path:
    """@brief Writes a stack of matrices as consecutive blocks.

    @param stem Path without an extension.
    @param batch Three-dimensional array indexed as (batch, row, col).
    @param storage_order Either row-major or column-major.
    @return The stem, for chaining.
    """
    order = "C" if storage_order == ROW_MAJOR else "F"
    flat = np.concatenate([block.flatten(order=order) for block in batch])
    return write_pair(
        stem, flat, batch.shape[1], batch.shape[2], batch.shape[0], storage_order
    )


# ---------------------------------------------------------------------------
# read_metadata
# ---------------------------------------------------------------------------


def test_read_metadata_returns_every_field(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.zeros((2, 3)), COLUMN_MAJOR)

    metadata = read_metadata(stem.with_name(stem.name + METADATA_EXTENSION))

    assert metadata["scalar_type"] == "double"
    assert metadata["rows"] == 2
    assert metadata["cols"] == 3
    assert metadata["batches"] == 1
    assert metadata["storage_order"] == COLUMN_MAJOR


def test_read_metadata_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "absent.md.json")


def test_read_metadata_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "partial.md.json"
    path.write_text(json.dumps({"scalar_type": "double", "rows": 2}), encoding="utf-8")

    with pytest.raises(KeyError):
        read_metadata(path)


def test_read_metadata_rejects_unsupported_scalar_type(tmp_path: Path) -> None:
    stem = write_pair(
        tmp_path / "floats", np.zeros(6), 2, 3, 1, COLUMN_MAJOR, scalar_type="float"
    )

    with pytest.raises(ValueError, match="scalar_type"):
        read_metadata(stem.with_name(stem.name + METADATA_EXTENSION))


def test_read_metadata_rejects_unknown_storage_order(tmp_path: Path) -> None:
    stem = write_pair(tmp_path / "odd", np.zeros(6), 2, 3, 1, "diagonal_major")

    with pytest.raises(ValueError, match="storage_order"):
        read_metadata(stem.with_name(stem.name + METADATA_EXTENSION))


@pytest.mark.parametrize("key", ["rows", "cols", "batches"])
def test_read_metadata_rejects_non_positive_dimensions(tmp_path: Path, key: str) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.zeros((2, 3)), COLUMN_MAJOR)
    metadata_path = stem.with_name(stem.name + METADATA_EXTENSION)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[key] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        read_metadata(metadata_path)


# ---------------------------------------------------------------------------
# load_matrix: single matrices
# ---------------------------------------------------------------------------


def test_loads_column_major_matrix(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stem = write_matrix_pair(tmp_path / "state", matrix, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, matrix)


def test_loads_row_major_matrix(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stem = write_matrix_pair(tmp_path / "state", matrix, ROW_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, matrix)


def test_storage_orders_agree_on_the_same_matrix(tmp_path: Path) -> None:
    """The two orders differ only in bytes on disk, not in what is recovered."""
    matrix = np.arange(12, dtype=np.float64).reshape(3, 4)
    column_stem = write_matrix_pair(tmp_path / "column", matrix, COLUMN_MAJOR)
    row_stem = write_matrix_pair(tmp_path / "row", matrix, ROW_MAJOR)

    column_bytes = column_stem.with_name(column_stem.name + BINARY_EXTENSION).read_bytes()
    row_bytes = row_stem.with_name(row_stem.name + BINARY_EXTENSION).read_bytes()
    assert column_bytes != row_bytes

    np.testing.assert_allclose(
        load_matrix_from_stem(column_stem), load_matrix_from_stem(row_stem)
    )


def test_column_vector_is_squeezed_to_one_dimension(tmp_path: Path) -> None:
    vector = np.array([[1.0], [2.0], [3.0]])
    stem = write_matrix_pair(tmp_path / "direction", vector, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_row_vector_is_squeezed_to_one_dimension(tmp_path: Path) -> None:
    vector = np.array([[1.0, 2.0, 3.0]])
    stem = write_matrix_pair(tmp_path / "row_direction", vector, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_single_element_squeezes_to_a_scalar_array(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "scalar", np.array([[4.5]]), COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == ()
    assert float(result) == pytest.approx(4.5)


def test_result_dtype_is_float64(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.zeros((2, 2)), COLUMN_MAJOR)

    assert load_matrix_from_stem(stem).dtype == np.float64


def test_result_is_c_contiguous(tmp_path: Path) -> None:
    """Column-major input is transposed internally; the result must be normal."""
    matrix = np.arange(12, dtype=np.float64).reshape(3, 4)
    stem = write_matrix_pair(tmp_path / "state", matrix, COLUMN_MAJOR)

    assert load_matrix_from_stem(stem).flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# load_matrix: batches
# ---------------------------------------------------------------------------


def test_loads_column_major_batches(tmp_path: Path) -> None:
    batch = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    stem = write_batch_pair(tmp_path / "frames", batch, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (3, 3, 3)
    np.testing.assert_allclose(result, batch)


def test_loads_row_major_batches(tmp_path: Path) -> None:
    batch = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    stem = write_batch_pair(tmp_path / "frames", batch, ROW_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (2, 4, 3)
    np.testing.assert_allclose(result, batch)


def test_batches_are_recovered_in_order(tmp_path: Path) -> None:
    batch = np.stack([np.full((2, 2), fill) for fill in (0.0, 1.0, 2.0, 3.0)])
    stem = write_batch_pair(tmp_path / "frames", batch, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    for index, fill in enumerate((0.0, 1.0, 2.0, 3.0)):
        np.testing.assert_allclose(result[index], np.full((2, 2), fill))


def test_batch_of_vectors_squeezes_the_trailing_axis(tmp_path: Path) -> None:
    batch = np.arange(15, dtype=np.float64).reshape(5, 3, 1)
    stem = write_batch_pair(tmp_path / "directions", batch, COLUMN_MAJOR)

    result = load_matrix_from_stem(stem)

    assert result.shape == (5, 3)
    np.testing.assert_allclose(result, batch.reshape(5, 3))


def test_single_batch_matches_unbatched(tmp_path: Path) -> None:
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3)
    batched = write_batch_pair(tmp_path / "batched", matrix[None, ...], COLUMN_MAJOR)
    plain = write_matrix_pair(tmp_path / "plain", matrix, COLUMN_MAJOR)

    np.testing.assert_allclose(
        load_matrix_from_stem(batched), load_matrix_from_stem(plain)
    )


# ---------------------------------------------------------------------------
# load_matrix: failure modes
# ---------------------------------------------------------------------------


def test_rejects_missing_binary_file(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.zeros((2, 3)), COLUMN_MAJOR)
    stem.with_name(stem.name + BINARY_EXTENSION).unlink()

    with pytest.raises(FileNotFoundError):
        load_matrix_from_stem(stem)


def test_rejects_missing_metadata_file(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.zeros((2, 3)), COLUMN_MAJOR)
    stem.with_name(stem.name + METADATA_EXTENSION).unlink()

    with pytest.raises(FileNotFoundError):
        load_matrix_from_stem(stem)


def test_rejects_size_mismatch(tmp_path: Path) -> None:
    """Metadata claiming a bigger matrix than the payload holds must not pass."""
    stem = write_pair(tmp_path / "truncated", np.zeros(5), 2, 3, 1, COLUMN_MAJOR)

    with pytest.raises(ValueError, match="bytes"):
        load_matrix_from_stem(stem)


def test_rejects_trailing_bytes(tmp_path: Path) -> None:
    stem = write_pair(tmp_path / "overlong", np.zeros(7), 2, 3, 1, COLUMN_MAJOR)

    with pytest.raises(ValueError, match="bytes"):
        load_matrix_from_stem(stem)


def test_rejects_wrong_batch_count(tmp_path: Path) -> None:
    stem = write_pair(tmp_path / "frames", np.zeros(18), 3, 3, 3, COLUMN_MAJOR)

    with pytest.raises(ValueError, match="bytes"):
        load_matrix_from_stem(stem)


def test_load_matrix_accepts_explicit_paths(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.eye(3), COLUMN_MAJOR)

    result = load_matrix(
        stem.with_name(stem.name + BINARY_EXTENSION),
        stem.with_name(stem.name + METADATA_EXTENSION),
    )

    np.testing.assert_allclose(result, np.eye(3))


def test_load_matrix_accepts_string_paths(tmp_path: Path) -> None:
    stem = write_matrix_pair(tmp_path / "state", np.eye(2), COLUMN_MAJOR)

    result = load_matrix(
        str(stem) + BINARY_EXTENSION, str(stem) + METADATA_EXTENSION
    )

    np.testing.assert_allclose(result, np.eye(2))


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def test_find_stems_lists_complete_pairs_only(tmp_path: Path) -> None:
    write_matrix_pair(tmp_path / "alpha", np.zeros((2, 2)), COLUMN_MAJOR)
    write_matrix_pair(tmp_path / "beta", np.zeros((2, 2)), COLUMN_MAJOR)

    # A metadata file with no payload is a partially written step.
    (tmp_path / f"orphan{METADATA_EXTENSION}").write_text("{}", encoding="utf-8")

    names = [stem.name for stem in find_stems(tmp_path)]

    assert names == ["alpha", "beta"]


def test_find_stems_rejects_non_directory(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        find_stems(path)


def test_load_directory_reads_every_pair(tmp_path: Path) -> None:
    write_matrix_pair(tmp_path / "positions", np.ones((3, 3)), COLUMN_MAJOR)
    write_batch_pair(
        tmp_path / "frames", np.zeros((4, 3, 3)), COLUMN_MAJOR
    )

    loaded = load_directory(tmp_path)

    assert set(loaded) == {"positions", "frames"}
    assert loaded["positions"].shape == (3, 3)
    assert loaded["frames"].shape == (4, 3, 3)


def test_load_directory_is_empty_for_an_empty_directory(tmp_path: Path) -> None:
    assert load_directory(tmp_path) == {}


# ---------------------------------------------------------------------------
# Step directory helpers
# ---------------------------------------------------------------------------


def write_rod_body(body_dir: Path, num_elements: int = 6, z_offset: float = 0.0) -> Path:
    """@brief Writes the three matrix pairs that make a body readable as a rod.

    Mirrors what the C++ writer emits for a rod: node positions as one matrix,
    element frames as a batch of three by three blocks, and element radii as a
    column vector.

    @param body_dir Directory to write into.
    @param num_elements Elements in the rod.
    @param z_offset Height to place the rod's base at.
    @return The directory, for chaining.
    """
    positions = np.zeros((num_elements + 1, 3))
    positions[:, 2] = z_offset + np.linspace(0.0, 1.0, num_elements + 1)
    write_matrix_pair(body_dir / "positions", positions, COLUMN_MAJOR)

    frames = np.tile(np.eye(3), (num_elements, 1, 1))
    write_batch_pair(body_dir / "frames", frames, COLUMN_MAJOR)

    radii = np.full((num_elements, 1), 0.05)
    write_matrix_pair(body_dir / "radii", radii, COLUMN_MAJOR)
    return body_dir


def write_step_tree(
    root: Path,
    num_steps: int = 4,
    body_names: tuple[str, ...] = ("rod1", "rod2"),
    time_step: float = 0.1,
) -> Path:
    """@brief Writes a tree of step directories, each holding several rods.

    @param root Directory to write the step directories beneath.
    @param num_steps Number of steps to write.
    @param body_names Body subdirectory to create in every step.
    @param time_step Simulation time between consecutive steps.
    @return The root, for chaining.
    """
    for step in range(num_steps):
        step_dir = root / f"step_{step:09d}_st_{time_step * step:.3f}"
        for index, name in enumerate(body_names):
            write_rod_body(step_dir / name, z_offset=float(index))
    return root


# ---------------------------------------------------------------------------
# parse_step_time
# ---------------------------------------------------------------------------


def test_parse_step_time_reads_the_trailing_field() -> None:
    assert parse_step_time(Path("step_000000042_st_0.420")) == pytest.approx(0.420)


def test_parse_step_time_accepts_a_bare_name() -> None:
    assert parse_step_time("step_000001000_st_12.345") == pytest.approx(12.345)


def test_parse_step_time_accepts_a_negative_time() -> None:
    assert parse_step_time("step_000000001_st_-1.250") == pytest.approx(-1.25)


@pytest.mark.parametrize(
    "name",
    [
        "step_000000000_st",  # too few fields
        "step_000000000_st_0.000_extra",  # too many
        "frame_000000000_st_0.000",  # wrong prefix
        "step_000000000_tt_0.000",  # wrong separator
    ],
)
def test_parse_step_time_rejects_a_malformed_name(name: str) -> None:
    with pytest.raises(ValueError, match="step_<number>_st_<time>"):
        parse_step_time(name)


def test_parse_step_time_rejects_a_non_numeric_time() -> None:
    with pytest.raises(ValueError, match="numeric time"):
        parse_step_time("step_000000000_st_later")


# ---------------------------------------------------------------------------
# get_cosserat_rods_from_step_dir
# ---------------------------------------------------------------------------


def test_reads_every_rod_in_a_step(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000003_st_0.300"
    write_rod_body(step_dir / "rod1")
    write_rod_body(step_dir / "rod2")

    time, rods = get_cosserat_rods_from_step_dir(step_dir)

    assert time == pytest.approx(0.3)
    assert len(rods) == 2
    assert all(rod.num_elements == 6 for rod in rods)


# Bodies are read in name order, not in whatever order the filesystem returns,
# so a body keeps its position in the list at every step. Anything styling
# bodies by index depends on that.
def test_bodies_are_returned_in_name_order(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    for index, name in enumerate(("zulu", "alpha", "mike")):
        write_rod_body(step_dir / name, z_offset=float(index))

    _, rods = get_cosserat_rods_from_step_dir(step_dir)

    # alpha was written at z=1, mike at z=2, zulu at z=0.
    bases = [float(rod.positions[0, 2]) for rod in rods]
    assert bases == [1.0, 2.0, 0.0]


# A body that wrote no radii, such as a rigid body, is not a rod and is passed
# over rather than failing the whole step.
def test_bodies_without_radii_are_skipped(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    write_rod_body(step_dir / "rod")
    sphere = step_dir / "sphere"
    write_matrix_pair(sphere / "positions", np.zeros((1, 3)), COLUMN_MAJOR)
    write_batch_pair(sphere / "frames", np.tile(np.eye(3), (1, 1, 1)), COLUMN_MAJOR)

    _, rods = get_cosserat_rods_from_step_dir(step_dir)

    assert len(rods) == 1


def test_a_step_with_no_rods_raises_by_default(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    write_matrix_pair(step_dir / "sphere" / "positions", np.zeros((1, 3)), COLUMN_MAJOR)

    with pytest.raises(ValueError, match="Couldn't find any rod-like object"):
        get_cosserat_rods_from_step_dir(step_dir)


def test_a_step_with_no_rods_can_be_tolerated(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    write_matrix_pair(step_dir / "sphere" / "positions", np.zeros((1, 3)), COLUMN_MAJOR)

    time, rods = get_cosserat_rods_from_step_dir(step_dir, raise_if_no_rods=False)

    assert time == pytest.approx(0.0)
    assert rods == []


def test_a_step_with_no_body_directories_is_rejected(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    step_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="at least one body directory"):
        get_cosserat_rods_from_step_dir(step_dir)


def test_a_missing_step_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        get_cosserat_rods_from_step_dir(tmp_path / "nope")


# A half written body, with metadata but no payload, is not rod-like, because
# find_stems only reports complete pairs.
def test_a_partially_written_body_is_not_rod_like(tmp_path: Path) -> None:
    step_dir = tmp_path / "step_000000000_st_0.000"
    write_rod_body(step_dir / "rod")
    (step_dir / "rod" / ("radii" + BINARY_EXTENSION)).unlink()

    with pytest.raises(ValueError, match="Couldn't find any rod-like object"):
        get_cosserat_rods_from_step_dir(step_dir)


# ---------------------------------------------------------------------------
# get_cosserat_rods_from_dir
# ---------------------------------------------------------------------------


def test_reads_every_step_in_the_tree(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=5)

    frames = get_cosserat_rods_from_dir(tmp_path)

    assert len(frames) == 5
    assert all(len(rods) == 2 for _, rods in frames)


# The whole reason the writer zero pads the step number. Reading in iterdir
# order returns the steps in filesystem order, which is not sorted, and gives a
# frame sequence that jumps back and forth in time.
def test_steps_come_back_in_step_order(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=12)

    times = [time for time, _ in get_cosserat_rods_from_dir(tmp_path)]

    assert times == sorted(times)
    assert times == pytest.approx([0.1 * step for step in range(12)])


# Ten steps is where name order and numeric order would diverge without the
# padding, so a scrambled reader shows up here even on a small tree.
def test_step_order_survives_crossing_a_power_of_ten(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=11)

    times = [time for time, _ in get_cosserat_rods_from_dir(tmp_path)]

    assert times == sorted(times)
    assert times[-1] == pytest.approx(1.0)


def test_directories_that_are_not_steps_are_ignored(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=3)
    (tmp_path / "logs").mkdir()
    (tmp_path / "notes.txt").write_text("not a step", encoding="utf-8")

    frames = get_cosserat_rods_from_dir(tmp_path)

    assert len(frames) == 3


def test_a_tree_with_no_steps_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()

    with pytest.raises(ValueError, match="No step_\\* directories"):
        get_cosserat_rods_from_dir(tmp_path)


def test_a_missing_tree_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        get_cosserat_rods_from_dir(tmp_path / "nope")


def test_tolerating_empty_steps_propagates_to_every_step(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=2)
    empty = tmp_path / "step_000000002_st_0.200"
    write_matrix_pair(empty / "sphere" / "positions", np.zeros((1, 3)), COLUMN_MAJOR)

    frames = get_cosserat_rods_from_dir(tmp_path, raise_if_no_rods=False)

    assert [len(rods) for _, rods in frames] == [2, 2, 0]


def test_a_string_path_is_accepted(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=2)

    assert len(get_cosserat_rods_from_dir(str(tmp_path))) == 2


# The rods come back fully constructed, so their meshes can be built directly.
def test_the_rods_returned_can_be_meshed(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=2, body_names=("rod",))

    _, rods = get_cosserat_rods_from_dir(tmp_path)[0]
    x_grid, y_grid, z_grid = rods[0].to_mesh(12)

    assert x_grid.shape == (rods[0].num_nodes, 13)
    assert np.isfinite(x_grid).all()
    assert np.hypot(x_grid, y_grid) == pytest.approx(0.05, abs=1e-12)


# The shape of the result is exactly what the animation utilities take as their
# frame sequence, which is the point of the whole reader.
def test_the_result_is_a_usable_frame_sequence(tmp_path: Path) -> None:
    write_step_tree(tmp_path, num_steps=4)

    frames = get_cosserat_rods_from_dir(tmp_path)

    assert isinstance(frames, list)
    for entry in frames:
        time, rods = entry
        assert isinstance(time, float)
        assert isinstance(rods, list)
        assert all(hasattr(rod, "to_mesh") for rod in rods)

    times = [time for time, _ in frames]
    intervals = np.diff(times)
    assert intervals == pytest.approx(intervals[0])
