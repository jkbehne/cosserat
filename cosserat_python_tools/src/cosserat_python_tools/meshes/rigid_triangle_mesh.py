"""@file rigid_triangle_mesh.py

@brief A rigid body whose shape is a triangle mesh, as written by the C++ side.

A mesh body records its geometry differently from a rod, because the two things
change on different timescales. Its triangles are fixed in its own frame and so
are written once for a whole run; its pose is not, and is written every
recorded step. Placing a vertex needs both:

@f[ \\mathbf{v}_{world} = \\mathbf{c}(t) + \\mathbf{Q}(t)^{T}\\mathbf{v}_{body} @f]

@ref RigidTriangleMesh holds one such pairing: a shape, and the pose to view it
through. @ref RigidTriangleMesh.with_pose makes a new view of the same shape
without copying the triangles, which is how one loaded mesh serves every frame
of an animation.

@note The stored frame carries a lab vector into the body, so its transpose
      carries a body vector out. Do not assume it is the identity even for a
      mesh that was modelled axis aligned: a mesh body is canonicalised into
      its principal frame on construction, and the sign of an eigenvector is
      arbitrary, so an axis aligned box can come back with its frame turned by
      a half turn and its vertices turned to match. The pair always
      reconstructs the same world geometry.
"""

from __future__ import annotations

from typing import Final, Tuple

import numpy as np
from numpy.typing import NDArray

## @brief Tolerance on the orthonormality check applied to a pose frame.
FRAME_ORTHONORMAL_TOLERANCE: Final[float] = 1e-8


class RigidTriangleMesh:
    """@brief A triangle mesh together with the rigid pose it is seen through.

    The shape is held in body coordinates exactly as it was written, and is
    never modified. World coordinates are produced on demand by
    @ref to_triangles.
    """

    def __init__(
        self,
        vertices: NDArray[np.float64],
        triangles: NDArray[np.int64],
        position: NDArray[np.float64],
        frame: NDArray[np.float64],
    ) -> None:
        """@brief Builds a posed mesh.

        @param vertices Vertices in body coordinates, shape @c (n, 3).
        @param triangles Vertex indices, shape @c (m, 3). Accepted as floats,
               which is how the binary format carries them, and converted.
        @param position Position of the body's origin in the world, shape
               @c (3,). For a mesh body this is its centre of mass.
        @param frame Rotation carrying a world vector into the body, shape
               @c (3, 3).
        @throws ValueError If any shape is wrong, if an index falls outside the
                vertices, if anything is not finite, or if the frame is not a
                rotation.
        """
        vertices = np.asarray(vertices, dtype=float)
        # Written as doubles because the binary format carries only that
        # scalar; every index a mesh could reach is exactly representable.
        triangles = np.asarray(triangles)
        position = np.asarray(position, dtype=float).reshape(-1)
        frame = np.asarray(frame, dtype=float)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n, 3), got {vertices.shape}"
            )
        if vertices.shape[0] < 3:
            raise ValueError(
                f"a mesh needs at least three vertices, got {vertices.shape[0]}"
            )
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (m, 3), got {triangles.shape}"
            )
        if triangles.shape[0] < 1:
            raise ValueError("a mesh needs at least one triangle")
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if frame.shape != (3, 3):
            raise ValueError(f"frame must have shape (3, 3), got {frame.shape}")

        if not np.isfinite(vertices).all():
            raise ValueError("mesh vertices must be finite")
        if not np.isfinite(position).all():
            raise ValueError("the position must be finite")
        if not np.isfinite(frame).all():
            raise ValueError("the frame must be finite")

        rounded = np.rint(triangles.astype(float))
        if not np.allclose(rounded, triangles.astype(float), atol=0.0):
            raise ValueError("triangle indices must be whole numbers")
        triangles = rounded.astype(np.int64)
        if triangles.min() < 0 or triangles.max() >= vertices.shape[0]:
            raise ValueError(
                f"triangle indices run {triangles.min()}..{triangles.max()}, "
                f"outside the {vertices.shape[0]} vertices given"
            )

        deviation = np.abs(frame @ frame.T - np.eye(3)).max()
        if deviation > FRAME_ORTHONORMAL_TOLERANCE:
            raise ValueError(
                f"the frame is not a rotation: orthonormality is off by "
                f"{deviation:.3e}"
            )

        self._vertices = vertices
        self._triangles = triangles
        self._position = position
        self._frame = frame

    # -- accessors ----------------------------------------------------------

    @property
    def vertices(self) -> NDArray[np.float64]:
        """@brief Vertices in body coordinates, shape @c (n, 3)."""
        return self._vertices

    @property
    def triangles(self) -> NDArray[np.int64]:
        """@brief Vertex indices, shape @c (m, 3)."""
        return self._triangles

    @property
    def position(self) -> NDArray[np.float64]:
        """@brief Position of the body's origin in the world, shape @c (3,)."""
        return self._position

    @property
    def frame(self) -> NDArray[np.float64]:
        """@brief Rotation carrying a world vector into the body."""
        return self._frame

    @property
    def num_vertices(self) -> int:
        """@brief Number of vertices."""
        return int(self._vertices.shape[0])

    @property
    def num_triangles(self) -> int:
        """@brief Number of triangles."""
        return int(self._triangles.shape[0])

    # -- posing -------------------------------------------------------------

    def with_pose(
        self, position: NDArray[np.float64], frame: NDArray[np.float64]
    ) -> "RigidTriangleMesh":
        """@brief The same shape seen through a different pose.

        The vertex and triangle arrays are shared rather than copied, so
        holding one of these per frame of a long animation costs a pose each,
        not a mesh each.

        @param position New position of the body's origin.
        @param frame New rotation carrying a world vector into the body.
        @return A mesh with this shape and that pose.
        """
        return RigidTriangleMesh(self._vertices, self._triangles, position, frame)

    # -- drawing ------------------------------------------------------------

    def to_triangles(self) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """@brief The mesh in world coordinates, ready to draw.

        Applies the pose to every vertex. The triangles are unchanged, since a
        rigid motion does not renumber anything.

        @return The world vertices, shape @c (n, 3), with the vertex indices.
        """
        # world = position + Q^T v, written as a right multiplication so the
        # whole vertex array transforms in one operation.
        world = self._position + self._vertices @ self._frame
        return world, self._triangles

    def world_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """@brief The corner to corner extent of the posed mesh.

        @return The lower and upper corners, each shape @c (3,).
        """
        world, _ = self.to_triangles()
        return world.min(axis=0), world.max(axis=0)

    def __repr__(self) -> str:
        """@brief A short description, for debugging."""
        return (
            f"RigidTriangleMesh({self.num_vertices} vertices, "
            f"{self.num_triangles} triangles, at {np.round(self._position, 4)})"
        )
