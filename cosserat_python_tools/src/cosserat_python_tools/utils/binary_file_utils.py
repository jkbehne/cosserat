"""@file binary_file_utils.py

@brief Reading simulation state written by the C++ @c utils::file_utils module.

Each state matrix is stored on disk as two files sharing a stem: a @c .bin
holding raw little-endian doubles exactly as Eigen laid them out, and a
@c .md.json describing the shape, the batch count and the storage order.

This module reverses that: given the pair, it returns a NumPy array of the
right shape and dtype. Arrays are squeezed, so a stored Eigen column vector
comes back one-dimensional rather than as an @c (n, 1) column.

@note Only @c double is written by the C++ side. Other scalar types are
      rejected rather than guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

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
