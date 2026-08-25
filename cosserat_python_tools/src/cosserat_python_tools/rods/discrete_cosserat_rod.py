"""
@file discrete_cosserat_rod.py
@brief Turns a discrete Cosserat rod state into a renderable tube mesh.

A Cosserat rod is stored on two interleaved domains. Positions live on the
@c num_nodes nodes; radii and material frames live on the @c num_elements
elements between them, of which there is always one fewer. Meshing has to
reconcile the two, because a tube is naturally built as a ring of vertices at
every node.

@section dcr_radius Radii

Element radii are averaged onto the nodes, with the two end nodes taking their
single neighbour. Assigning element @c k's radius directly to node @c k instead
would shift the whole radius profile half an element along the rod, so every
element would render with the average of two of its neighbours' radii rather
than its own. On a stretched, volume-preserving rod that bias is several
percent. Averaging removes it: for a linearly varying radius the interpolated
midpoint of an element comes back exactly equal to that element's radius.

The cost is smoothing. A genuine step in radius between two neighbouring
elements is rounded off rather than rendered as a step.

@section dcr_frames Frames

Frames are converted to quaternions on construction, because the node frames
have to be interpolated from the two elements on either side and rotation
matrices cannot be averaged component-wise -- the result is not a rotation.
Quaternions interpolate correctly under slerp, which is what is used here, along
the shortest path between the two element frames.

Carrying the frames at all is what makes twist visible. A circular cross-section
is rotationally symmetric, so a rod twisted by any amount renders identically
unless the surface parameterisation follows the material frame. Here the ring
angle is measured from the material normal, so a texture or a stripe applied to
the mesh will follow the twist.

@section dcr_miter Joints

At a bend, a ring placed perpendicular to either adjacent element leaves the two
tube segments overlapping on the inside of the turn. Each ring is therefore
mitered: placed in the plane bisecting the two adjacent tangents and stretched
in the plane of the bend by @c 1/cos(theta/2). The result projects to a circle
of exactly the requested radius when viewed along either adjacent tangent, which
is the defining property of a mitered joint.

The construction holds until the mitered rings themselves overlap, which happens
only past roughly @c 2*atan(l/2r) of turn within a single element -- of order
110 degrees for a typical rod. A turn of exactly 180 degrees is degenerate and
is rejected.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


## @brief Tolerance on the orthonormality check applied to input frames.
FRAME_ORTHONORMAL_TOLERANCE = 1e-8

## @brief Lengths below this are treated as degenerate.
LENGTH_TOLERANCE = 1e-12


class DiscreteCosseratRod:
    """
    @brief One time slice of a Cosserat rod, ready to be turned into a mesh.

    Holds the node positions, the per-element radii and the per-element frames
    as quaternions, together with the node-domain quantities derived from them.
    Everything is computed once on construction, so @ref to_mesh is cheap to
    call repeatedly at different circumferential resolutions.
    """

    def __init__(
        self,
        positions: np.ndarray,
        frames: np.ndarray,
        radii: np.ndarray,
    ) -> None:
        """
        @brief Builds a rod from one time slice of simulation output.

        @param positions Node positions, shape @c (num_nodes, 3). At least two
               nodes are required, and consecutive nodes must be distinct.
        @param frames Element frames, shape @c (num_elements, 3, 3), where
               @c num_elements is @c num_nodes-1. Each frame maps a lab-frame
               vector into the material frame, so its rows are the material
               normal, binormal and tangent expressed in lab coordinates. Each
               must be orthonormal to within
               @ref FRAME_ORTHONORMAL_TOLERANCE.
        @param radii Element radii, shape @c (num_elements,). All must be
               finite and strictly positive.

        @throws ValueError if any shape, count or value is not as described.
        """
        positions = np.asarray(positions, dtype=float)
        frames = np.asarray(frames, dtype=float)
        radii = np.asarray(radii, dtype=float)

        self._validate(positions, frames, radii)

        self._positions = positions
        self._radii = radii
        # Quaternions rather than matrices, so the node frames can be slerped
        # from the elements either side. See the module notes.
        self._element_rotations = Rotation.from_matrix(frames)

        self._node_radii = self._average_radii_onto_nodes(radii)
        self._node_rotations = self._slerp_frames_onto_nodes(self._element_rotations)

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate(
        positions: np.ndarray, frames: np.ndarray, radii: np.ndarray
    ) -> None:
        """
        @brief Fails unless the three arrays describe a consistent rod.

        @param positions Candidate node positions.
        @param frames Candidate element frames.
        @param radii Candidate element radii.

        @throws ValueError with a message naming the offending array.
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (num_nodes, 3), got {positions.shape}"
            )
        num_nodes = positions.shape[0]
        if num_nodes < 2:
            raise ValueError(f"a rod needs at least two nodes, got {num_nodes}")
        num_elements = num_nodes - 1

        if radii.ndim != 1:
            raise ValueError(f"radii must be one dimensional, got shape {radii.shape}")
        if radii.shape[0] != num_elements:
            raise ValueError(
                f"{num_nodes} nodes implies {num_elements} elements, "
                f"but got {radii.shape[0]} radii"
            )

        if frames.shape != (num_elements, 3, 3):
            raise ValueError(
                f"frames must have shape ({num_elements}, 3, 3), got {frames.shape}"
            )

        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if not np.isfinite(radii).all():
            raise ValueError("radii must be finite")
        if not np.isfinite(frames).all():
            raise ValueError("frames must be finite")

        if not (radii > 0.0).all():
            raise ValueError("every radius must be strictly positive")

        # Consecutive nodes must be distinct, or the element has no direction.
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if (segment_lengths <= LENGTH_TOLERANCE).any():
            bad = int(np.argmin(segment_lengths))
            raise ValueError(
                f"element {bad} has zero length; consecutive nodes must differ"
            )

        # scipy silently orthonormalises whatever it is handed, so a frame that
        # has drifted or was never a rotation would be quietly repaired rather
        # than reported. Check before converting.
        products = np.einsum("nij,nkj->nik", frames, frames)
        identity = np.broadcast_to(np.eye(3), products.shape)
        deviation = np.abs(products - identity).max(axis=(1, 2))
        if (deviation > FRAME_ORTHONORMAL_TOLERANCE).any():
            worst = int(np.argmax(deviation))
            raise ValueError(
                f"frame {worst} is not orthonormal: deviation {deviation[worst]:.3e} "
                f"exceeds {FRAME_ORTHONORMAL_TOLERANCE:.1e}"
            )

        determinants = np.linalg.det(frames)
        if (determinants < 0.0).any():
            worst = int(np.argmin(determinants))
            raise ValueError(
                f"frame {worst} has determinant {determinants[worst]:.3f}; frames "
                "must be rotations, not reflections"
            )

    # -- node-domain quantities ---------------------------------------------

    @staticmethod
    def _average_radii_onto_nodes(radii: np.ndarray) -> np.ndarray:
        """
        @brief Averages element radii onto the nodes.

        Interior node @c k takes the mean of elements @c k-1 and @c k; the two
        end nodes take their single neighbouring element. See the module notes
        for why the radii are not simply assigned across.

        @param radii Element radii, shape @c (num_elements,).
        @return Node radii, shape @c (num_elements + 1,).
        """
        node_radii = np.empty(radii.shape[0] + 1, dtype=float)
        node_radii[0] = radii[0]
        node_radii[-1] = radii[-1]
        node_radii[1:-1] = 0.5 * (radii[:-1] + radii[1:])
        return node_radii

    @staticmethod
    def _slerp_frames_onto_nodes(element_rotations: Rotation) -> Rotation:
        """
        @brief Interpolates element frames onto the nodes by slerp.

        Interior node @c k takes the rotation half way between elements @c k-1
        and @c k along the shortest path; the two end nodes take their single
        neighbouring element.

        Implemented as a relative rotation, halved as a rotation vector and
        composed back on. That is exactly slerp at the midpoint, and unlike
        building a scipy @c Slerp per node it runs over the whole rod at once.

        @param element_rotations Element frames as rotations.
        @return Node frames as rotations, one longer than the input.
        """
        previous = element_rotations[:-1]
        following = element_rotations[1:]

        # The shortest-path rotation taking each element frame to the next.
        relative = (following * previous.inv()).as_rotvec()
        interior = Rotation.from_rotvec(0.5 * relative) * previous

        return Rotation.concatenate(
            [element_rotations[0], interior, element_rotations[-1]]
        )

    # -- accessors ----------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """@brief Number of nodes."""
        return self._positions.shape[0]

    @property
    def num_elements(self) -> int:
        """@brief Number of elements, one fewer than the nodes."""
        return self._radii.shape[0]

    @property
    def positions(self) -> np.ndarray:
        """@brief Node positions, shape @c (num_nodes, 3)."""
        return self._positions

    @property
    def radii(self) -> np.ndarray:
        """@brief Element radii, shape @c (num_elements,)."""
        return self._radii

    @property
    def node_radii(self) -> np.ndarray:
        """@brief Radii averaged onto the nodes, shape @c (num_nodes,)."""
        return self._node_radii

    @property
    def element_rotations(self) -> Rotation:
        """@brief Element frames as rotations, length @c num_elements."""
        return self._element_rotations

    @property
    def node_rotations(self) -> Rotation:
        """@brief Frames slerped onto the nodes, length @c num_nodes."""
        return self._node_rotations

    @property
    def element_quaternions(self) -> np.ndarray:
        """@brief Element frames as quaternions, shape @c (num_elements, 4)."""
        return self._element_rotations.as_quat()

    @property
    def node_quaternions(self) -> np.ndarray:
        """@brief Node frames as quaternions, shape @c (num_nodes, 4)."""
        return self._node_rotations.as_quat()

    @property
    def element_tangents(self) -> np.ndarray:
        """
        @brief Unit tangents of the centreline, shape @c (num_elements, 3).

        Taken from the node positions, so these are the directions the tube
        segments actually run along. They are not in general the material
        tangents, which tilt away from the centreline by the shear strain.
        """
        segments = np.diff(self._positions, axis=0)
        return segments / np.linalg.norm(segments, axis=1)[:, None]

    @property
    def lengths(self) -> np.ndarray:
        """@brief Element lengths, shape @c (num_elements,)."""
        return np.linalg.norm(np.diff(self._positions, axis=0), axis=1)

    # -- meshing ------------------------------------------------------------

    def _miter_normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Ring plane normals, half-angle cosines and bend directions.

        Interior node @c k gets the bisector of the tangents either side; the
        two end nodes get their single tangent, which makes their rings square
        to the rod and leaves them unstretched.

        @return A triple of the per-node ring normals with shape
                @c (num_nodes, 3), the cosines of the half turn angle with
                shape @c (num_nodes,), and the in-plane bend directions with
                shape @c (num_nodes, 3). The bend direction is zero wherever
                there is no bend.

        @throws ValueError if any interior node turns through 180 degrees, at
                which point the bisector vanishes and no ring plane exists.
        """
        tangents = self.element_tangents
        num_nodes = self.num_nodes

        normals = np.empty((num_nodes, 3), dtype=float)
        cos_half = np.ones(num_nodes, dtype=float)
        bend_directions = np.zeros((num_nodes, 3), dtype=float)

        normals[0] = tangents[0]
        normals[-1] = tangents[-1]

        if num_nodes > 2:
            previous = tangents[:-1]
            following = tangents[1:]

            bisectors = previous + following
            bisector_lengths = np.linalg.norm(bisectors, axis=1)
            if (bisector_lengths <= LENGTH_TOLERANCE).any():
                worst = int(np.argmin(bisector_lengths)) + 1
                raise ValueError(
                    f"node {worst} doubles back on itself; a 180 degree turn has "
                    "no bisecting plane and cannot be mitered"
                )
            bisectors = bisectors / bisector_lengths[:, None]
            normals[1:-1] = bisectors

            # The bisector makes the same angle with each tangent, so either
            # gives the half angle.
            cos_half[1:-1] = np.einsum("nj,nj->n", bisectors, following)

            # The difference of two unit vectors is perpendicular to their sum,
            # so this lies in the ring plane and points along the bend.
            differences = following - previous
            difference_lengths = np.linalg.norm(differences, axis=1)
            straight = difference_lengths <= LENGTH_TOLERANCE
            safe_lengths = np.where(straight, 1.0, difference_lengths)
            bend_directions[1:-1] = np.where(
                straight[:, None], 0.0, differences / safe_lengths[:, None]
            )

        return normals, cos_half, bend_directions

    def to_mesh(
        self, num_circumferential: int = 16, close_seam: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        @brief Builds the tube surface as three coordinate grids.

        One ring of vertices is placed at every node, mitered into the plane
        bisecting the adjacent elements and sized by the node radius. The ring
        angle is measured from the material normal, so the parameterisation
        follows the rod's twist.

        The three returned arrays are laid out for @c matplotlib's
        @c plot_surface and for any consumer expecting a rectangular grid:
        rows run along the rod and columns run around it.

        @param num_circumferential Number of vertices around each ring; at
               least three.
        @param close_seam When true a duplicate of the first column is appended
               so the surface closes visually. Renderers that build faces from
               indices generally want this off, since the duplicated column
               would be a seam of degenerate quads.
        @return A triple @c (X, Y, Z), each of shape
                @c (num_nodes, num_circumferential + 1) when the seam is closed
                and @c (num_nodes, num_circumferential) when it is not.

        @throws ValueError if @p num_circumferential is less than three, or if
                the rod doubles back on itself at any node.
        """
        if num_circumferential < 3:
            raise ValueError(
                f"need at least three vertices around the rod, got {num_circumferential}"
            )

        normals, cos_half, bend_directions = self._miter_normals()
        node_frames = self._node_rotations.as_matrix()

        # The frame's rows are the material basis in lab coordinates, so row 0
        # is the normal that the ring angle is measured from.
        material_normals = node_frames[:, 0, :]

        # Square the reference direction to the ring plane. The material normal
        # is already very nearly perpendicular to it, so this is a small
        # correction, but it guarantees an orthonormal basis in the plane.
        in_plane = material_normals - (
            np.einsum("nj,nj->n", material_normals, normals)[:, None] * normals
        )
        in_plane_lengths = np.linalg.norm(in_plane, axis=1)
        if (in_plane_lengths <= LENGTH_TOLERANCE).any():
            worst = int(np.argmin(in_plane_lengths))
            raise ValueError(
                f"the frame at node {worst} is edge on to its ring plane, so the "
                "ring has no reference direction"
            )
        first_axis = in_plane / in_plane_lengths[:, None]
        second_axis = np.cross(normals, first_axis)

        angles = np.linspace(0.0, 2.0 * np.pi, num_circumferential, endpoint=False)
        cosines = np.cos(angles)[None, :, None]
        sines = np.sin(angles)[None, :, None]

        # A circle of the node radius, lying in the ring plane.
        rings = self._node_radii[:, None, None] * (
            cosines * first_axis[:, None, :] + sines * second_axis[:, None, :]
        )

        # Stretch along the bend so the ring is the intersection of the two
        # adjacent cylinders rather than a circle that pinches inside the turn.
        stretch = (1.0 / cos_half - 1.0)[:, None, None]
        along_bend = np.einsum("ntj,nj->nt", rings, bend_directions)[:, :, None]
        rings = rings + stretch * along_bend * bend_directions[:, None, :]

        vertices = self._positions[:, None, :] + rings

        if close_seam:
            vertices = np.concatenate([vertices, vertices[:, :1, :]], axis=1)

        return vertices[..., 0], vertices[..., 1], vertices[..., 2]


def plot_rod(
    rod: DiscreteCosseratRod,
    num_circumferential: int = 24,
    axes=None,
    **surface_kwargs,
):
    """
    @brief Draws a rod as a surface on a three dimensional matplotlib axes.

    Plotting the mesh is otherwise a one line call, but a rod is long and thin
    and matplotlib does not equalise the axis scales by default. Left alone it
    stretches each axis to fill the box, which turns a slender rod into a fat
    blob and makes a bend look like the wrong angle. This sets a common scale
    on all three axes so the shape is faithful.

    @param rod The rod to draw.
    @param num_circumferential Vertices around each ring.
    @param axes An existing three dimensional axes to draw on. A new figure is
           created when omitted.
    @param surface_kwargs Passed through to @c plot_surface.
    @return The axes drawn on.
    """
    import matplotlib.pyplot as plt

    x_grid, y_grid, z_grid = rod.to_mesh(num_circumferential, close_seam=True)

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d")

    surface_kwargs.setdefault("linewidth", 0)
    surface_kwargs.setdefault("antialiased", True)
    axes.plot_surface(x_grid, y_grid, z_grid, **surface_kwargs)

    # One scale for every axis, centred on the rod, so proportions are honest.
    stacked = np.stack([x_grid, y_grid, z_grid])
    centres = stacked.reshape(3, -1).mean(axis=1)
    half_span = 0.5 * max(
        float(grid.max() - grid.min()) for grid in (x_grid, y_grid, z_grid)
    )
    half_span = max(half_span, LENGTH_TOLERANCE)
    for setter, centre in zip(
        (axes.set_xlim, axes.set_ylim, axes.set_zlim), centres
    ):
        setter(centre - half_span, centre + half_span)
    axes.set_box_aspect((1.0, 1.0, 1.0))

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    return axes
