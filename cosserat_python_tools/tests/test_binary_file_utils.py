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
    load_directory,
    load_matrix,
    load_matrix_from_stem,
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
