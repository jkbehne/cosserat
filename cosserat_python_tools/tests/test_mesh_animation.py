"""
@file test_mesh_animation.py
@brief Tests for @ref mesh_animation.

Where possible the output video is checked with @c ffprobe rather than by
trusting that the writer did what was asked, so the frame count, dimensions and
frame rate are read back off the encoded file.

The axis limits are verified through the frame hook, which sees the axes after
each frame is drawn. That is the property the module exists to guarantee: if the
limits moved between frames the camera would chase the motion.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cosserat_python_tools.rods.discrete_cosserat_rod import DiscreteCosseratRod
from cosserat_python_tools.utils.mesh_animation import (
    AnimationResult,
    Bounds,
    animate_meshes,
    as_grids,
    compute_global_bounds,
    default_title,
    ffmpeg_available,
    frames_per_second,
    uniform_frame_interval,
)


needs_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg is not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cylinder_grids(
    centre_x: float = 0.0, radius: float = 0.2, height: float = 1.0, rings: int = 6
):
    """
    @brief A simple tube as three coordinate grids.

    @param centre_x Offset along x, so a mesh can be made to move.
    @param radius Tube radius.
    @param height Tube height along z.
    @param rings Number of rings along the tube.
    @return The triple @c (X, Y, Z).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 12)
    heights = np.linspace(0.0, height, rings)
    angle_grid, height_grid = np.meshgrid(angles, heights)
    return (
        centre_x + radius * np.cos(angle_grid),
        radius * np.sin(angle_grid),
        height_grid,
    )


class ToMeshObject:
    """@brief A minimal stand-in for a mesh object exposing @c to_mesh."""

    def __init__(self, centre_x: float = 0.0) -> None:
        """@param centre_x Offset along x."""
        self.centre_x = centre_x
        self.calls: list = []

    def to_mesh(self, num_circumferential: int = 12, close_seam: bool = True):
        """
        @brief Returns the grids, recording how it was called.

        @param num_circumferential Vertices around the tube.
        @param close_seam Accepted for signature compatibility.
        @return The triple @c (X, Y, Z).
        """
        self.calls.append((num_circumferential, close_seam))
        angles = np.linspace(0.0, 2.0 * np.pi, num_circumferential)
        heights = np.linspace(0.0, 1.0, 5)
        angle_grid, height_grid = np.meshgrid(angles, heights)
        return (
            self.centre_x + 0.2 * np.cos(angle_grid),
            0.2 * np.sin(angle_grid),
            height_grid,
        )


def moving_frames(count: int = 8, interval: float = 0.05):
    """
    @brief Frames of one tube sliding along x.

    @param count Number of frames.
    @param interval Simulation time between frames.
    @return A list of @c (time, meshes) pairs.
    """
    return [
        (index * interval, [cylinder_grids(centre_x=0.1 * index)])
        for index in range(count)
    ]


def probe(path: Path) -> dict:
    """
    @brief Reads stream properties back off an encoded file.

    @param path The video file.
    @return The first video stream's properties.
    """
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(output.stdout)["streams"][0]


has_ffprobe = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe is not available"
)


# ---------------------------------------------------------------------------
# as_grids
# ---------------------------------------------------------------------------


def test_accepts_a_ready_made_triple():
    grids = cylinder_grids()
    x_grid, y_grid, z_grid = as_grids(grids)
    assert x_grid.shape == grids[0].shape
    assert z_grid is not None


def test_accepts_an_object_exposing_to_mesh():
    mesh = ToMeshObject()
    x_grid, _, _ = as_grids(mesh)
    assert x_grid.shape == (5, 12)


def test_forwards_options_to_to_mesh():
    mesh = ToMeshObject()
    as_grids(mesh, num_circumferential=7, close_seam=False)
    assert mesh.calls == [(7, False)]


def test_rejects_something_that_is_not_a_mesh():
    with pytest.raises(TypeError, match="must expose to_mesh"):
        as_grids(42)


def test_rejects_grids_that_disagree_in_shape():
    x_grid, y_grid, z_grid = cylinder_grids()
    with pytest.raises(ValueError, match="must agree in shape"):
        as_grids((x_grid, y_grid[:-1], z_grid))


def test_rejects_grids_that_are_not_two_dimensional():
    with pytest.raises(ValueError, match="two dimensional"):
        as_grids((np.zeros(4), np.zeros(4), np.zeros(4)))


def test_rejects_non_finite_grids():
    x_grid, y_grid, z_grid = cylinder_grids()
    x_grid = x_grid.copy()
    x_grid[0, 0] = np.nan
    with pytest.raises(ValueError, match="the X grid holds non-finite"):
        as_grids((x_grid, y_grid, z_grid))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_uniform_interval_of_evenly_spaced_times():
    assert uniform_frame_interval([0.0, 0.25, 0.5, 0.75]) == pytest.approx(0.25)


def test_uniform_interval_needs_two_frames():
    with pytest.raises(ValueError, match="at least two frames"):
        uniform_frame_interval([0.0])


def test_uniform_interval_rejects_times_that_do_not_increase():
    with pytest.raises(ValueError, match="must strictly increase"):
        uniform_frame_interval([0.0, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="must strictly increase"):
        uniform_frame_interval([0.0, 1.0, 0.5])


# Encoding unevenly sampled frames at a constant rate would silently make the
# motion look fast wherever the sampling was sparse, so it is refused.
def test_uniform_interval_rejects_uneven_spacing():
    with pytest.raises(ValueError, match="not uniformly spaced"):
        uniform_frame_interval([0.0, 0.1, 0.5, 0.6])


def test_frames_per_second_plays_in_real_time_by_default():
    times = [0.0, 0.02, 0.04, 0.06]
    assert frames_per_second(times) == pytest.approx(50.0)


def test_slowdown_reduces_the_frame_rate_proportionally():
    times = [0.0, 0.02, 0.04, 0.06]
    assert frames_per_second(times, slowdown=4.0) == pytest.approx(12.5)
    assert frames_per_second(times, slowdown=0.5) == pytest.approx(100.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_frames_per_second_rejects_a_bad_slowdown(bad):
    with pytest.raises(ValueError, match="slowdown must be finite and positive"):
        frames_per_second([0.0, 0.1], slowdown=bad)


# ---------------------------------------------------------------------------
# Bounds
#
# Fixing the axis limits across the whole sequence is the thing that stops the
# camera chasing the motion, so the box has to contain every frame.
# ---------------------------------------------------------------------------


def test_bounds_contain_every_frame():
    frames = moving_frames(count=8)
    bounds = compute_global_bounds(frames, padding=0.0)

    for _, meshes in frames:
        for mesh in meshes:
            for grid, centre in zip(as_grids(mesh), bounds.centre):
                assert grid.min() >= centre - bounds.half_extent - 1e-9
                assert grid.max() <= centre + bounds.half_extent + 1e-9


def test_bounds_use_one_extent_for_every_axis():
    # A long thin tube: without a common extent the short axes would be blown
    # up to fill the figure and the shape would be distorted.
    frames = [(0.0, [cylinder_grids(radius=0.05, height=4.0)])]
    bounds = compute_global_bounds(frames, padding=0.0)

    spans = [high - low for low, high in bounds.limits()]
    assert spans[0] == pytest.approx(spans[1])
    assert spans[1] == pytest.approx(spans[2])
    assert spans[2] == pytest.approx(4.0)


def test_padding_grows_the_box():
    frames = moving_frames(count=4)
    tight = compute_global_bounds(frames, padding=0.0)
    loose = compute_global_bounds(frames, padding=0.5)

    assert loose.half_extent == pytest.approx(1.5 * tight.half_extent)
    assert loose.centre == pytest.approx(tight.centre)


def test_bounds_reject_negative_padding():
    with pytest.raises(ValueError, match="padding must not be negative"):
        compute_global_bounds(moving_frames(), padding=-0.1)


def test_bounds_reject_a_sequence_with_no_meshes():
    with pytest.raises(ValueError, match="nothing to bound"):
        compute_global_bounds([(0.0, []), (0.1, [])])


def test_a_flat_scene_still_has_a_drawable_box():
    flat = (np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))
    bounds = compute_global_bounds([(0.0, [flat])], padding=0.0)
    assert bounds.half_extent > 0.0


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def test_accepts_a_callable_provider():
    frames = moving_frames(count=6)
    bounds_from_list = compute_global_bounds(frames)
    bounds_from_provider = compute_global_bounds(
        lambda index: frames[index], num_frames=6
    )

    assert bounds_from_provider.centre == pytest.approx(bounds_from_list.centre)
    assert bounds_from_provider.half_extent == pytest.approx(
        bounds_from_list.half_extent
    )


def test_a_provider_needs_a_frame_count():
    frames = moving_frames()
    with pytest.raises(ValueError, match="num_frames is required"):
        compute_global_bounds(lambda index: frames[index])


def test_a_contradictory_frame_count_is_rejected():
    with pytest.raises(ValueError, match="contradicts"):
        compute_global_bounds(moving_frames(count=4), num_frames=9)


def test_an_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="at least one frame"):
        compute_global_bounds([])


def test_a_malformed_frame_is_rejected():
    with pytest.raises(ValueError, match=r"must be a \(time, meshes\) pair"):
        compute_global_bounds([cylinder_grids()])


def test_a_non_finite_frame_time_is_rejected():
    with pytest.raises(ValueError, match="is not finite"):
        compute_global_bounds([(np.nan, [cylinder_grids()])])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_writes_a_video(tmp_path):
    output = tmp_path / "rod.mp4"
    result = animate_meshes(moving_frames(count=6), output, fps=10.0)

    assert output.exists()
    assert output.stat().st_size > 0
    assert isinstance(result, AnimationResult)
    assert result.num_frames == 6
    assert result.fps == pytest.approx(10.0)
    assert result.duration_seconds == pytest.approx(0.6)


@needs_ffmpeg
@has_ffprobe
def test_the_encoded_file_has_the_frames_and_rate_asked_for(tmp_path):
    output = tmp_path / "rod.mp4"
    animate_meshes(moving_frames(count=9), output, fps=12.0)

    stream = probe(output)
    assert int(stream["nb_frames"]) == 9
    numerator, denominator = stream["r_frame_rate"].split("/")
    assert int(numerator) / int(denominator) == pytest.approx(12.0)


@needs_ffmpeg
@has_ffprobe
def test_figure_size_and_dpi_set_the_pixel_dimensions(tmp_path):
    output = tmp_path / "rod.mp4"
    animate_meshes(
        moving_frames(count=3), output, fps=5.0, figure_size=(4.0, 3.0), dpi=100
    )

    stream = probe(output)
    assert (stream["width"], stream["height"]) == (400, 300)


@needs_ffmpeg
def test_the_frame_rate_is_inferred_from_the_times(tmp_path):
    output = tmp_path / "rod.mp4"
    # Frames 0.05 s apart played four times slower than real time.
    result = animate_meshes(
        moving_frames(count=5, interval=0.05), output, slowdown=4.0
    )
    assert result.fps == pytest.approx(5.0)


@needs_ffmpeg
def test_a_single_frame_needs_an_explicit_rate(tmp_path):
    output = tmp_path / "rod.mp4"
    single = [(0.0, [cylinder_grids()])]

    with pytest.raises(ValueError, match="cannot be inferred from a single frame"):
        animate_meshes(single, output)

    result = animate_meshes(single, output, fps=1.0)
    assert result.num_frames == 1


@needs_ffmpeg
def test_writes_a_gif(tmp_path):
    output = tmp_path / "rod.gif"
    result = animate_meshes(moving_frames(count=4), output, fps=8.0)

    assert output.exists()
    assert output.stat().st_size > 0
    assert result.output_path.suffix == ".gif"


def test_rejects_an_unsupported_container(tmp_path):
    with pytest.raises(ValueError, match="unsupported output format"):
        animate_meshes(moving_frames(count=2), tmp_path / "rod.xyz", fps=5.0)


@needs_ffmpeg
def test_creates_the_output_directory(tmp_path):
    output = tmp_path / "nested" / "deeper" / "rod.mp4"
    animate_meshes(moving_frames(count=3), output, fps=5.0)
    assert output.exists()


@needs_ffmpeg
def test_renders_several_meshes_per_frame(tmp_path):
    frames = [
        (
            0.05 * index,
            [cylinder_grids(centre_x=0.0), cylinder_grids(centre_x=1.0)],
        )
        for index in range(4)
    ]
    output = tmp_path / "pair.mp4"
    result = animate_meshes(frames, output, fps=5.0)

    assert result.num_frames == 4
    # The bounds must span both bodies, not just the first.
    low, high = result.bounds.limits()[0]
    assert low < -0.2 and high > 1.2


@needs_ffmpeg
def test_renders_objects_exposing_to_mesh(tmp_path):
    meshes = [ToMeshObject(centre_x=0.1 * index) for index in range(4)]
    frames = [(0.05 * index, [mesh]) for index, mesh in enumerate(meshes)]
    output = tmp_path / "objects.mp4"

    animate_meshes(frames, output, fps=5.0, to_mesh_kwargs={"num_circumferential": 9})

    # Once for the bounds pass and once for rendering.
    assert meshes[0].calls == [(9, True), (9, True)]


# ---------------------------------------------------------------------------
# The axis limits, which are the whole point
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_the_axis_limits_never_move(tmp_path):
    seen = []

    def record(axes, index, time):
        del index, time
        seen.append((axes.get_xlim(), axes.get_ylim(), axes.get_zlim()))

    # A tube that slides a long way, which is exactly when autoscaling would
    # make the camera chase it.
    frames = [(0.05 * index, [cylinder_grids(centre_x=index)]) for index in range(6)]
    animate_meshes(frames, tmp_path / "slide.mp4", fps=5.0, frame_hook=record)

    assert len(seen) == 6
    for limits in seen[1:]:
        assert limits == pytest.approx(np.array(seen[0]))


@needs_ffmpeg
def test_the_limits_match_the_computed_bounds(tmp_path):
    seen = []
    frames = moving_frames(count=4)

    result = animate_meshes(
        frames,
        tmp_path / "bounds.mp4",
        fps=5.0,
        frame_hook=lambda axes, index, time: seen.append(axes.get_xlim()),
    )

    assert seen[0] == pytest.approx(result.bounds.limits()[0])


@needs_ffmpeg
def test_every_axis_gets_the_same_span(tmp_path):
    spans = []

    def record(axes, index, time):
        del index, time
        spans.append(
            [
                axes.get_xlim()[1] - axes.get_xlim()[0],
                axes.get_ylim()[1] - axes.get_ylim()[0],
                axes.get_zlim()[1] - axes.get_zlim()[0],
            ]
        )

    frames = [(0.05 * i, [cylinder_grids(radius=0.05, height=3.0)]) for i in range(3)]
    animate_meshes(frames, tmp_path / "aspect.mp4", fps=5.0, frame_hook=record)

    for span in spans:
        assert span[0] == pytest.approx(span[1])
        assert span[1] == pytest.approx(span[2])


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_the_default_title_shows_the_time():
    assert default_title(1.5, 3) == "t = 1.500 s"


@needs_ffmpeg
def test_a_custom_title_is_used(tmp_path):
    titles = []

    def record(axes, index, time):
        del index, time
        titles.append(axes.get_title())

    animate_meshes(
        moving_frames(count=3),
        tmp_path / "titled.mp4",
        fps=5.0,
        title_formatter=lambda time, index: f"frame {index} at {time:.2f}",
        frame_hook=record,
    )

    assert titles == ["frame 0 at 0.00", "frame 1 at 0.05", "frame 2 at 0.10"]


@needs_ffmpeg
def test_the_title_can_be_turned_off(tmp_path):
    titles = []
    animate_meshes(
        moving_frames(count=2),
        tmp_path / "untitled.mp4",
        fps=5.0,
        title_formatter=None,
        frame_hook=lambda axes, index, time: titles.append(axes.get_title()),
    )
    assert titles == ["", ""]


@needs_ffmpeg
def test_per_mesh_styling_cycles_across_the_meshes(tmp_path):
    counts = []
    palettes = []

    def record(axes, index, time):
        del index, time
        counts.append(len(axes.collections))
        # Every body's faces share one collection, so the styling shows up as
        # the set of face colours within it rather than as separate artists.
        colours = set()
        for collection in axes.collections:
            for colour in collection.get_facecolors():
                colours.add(tuple(np.round(colour, 4)))
        palettes.append(colours)

    frames = [
        (0.05 * index, [cylinder_grids(0.0), cylinder_grids(1.0)])
        for index in range(3)
    ]
    animate_meshes(
        frames,
        tmp_path / "styled.mp4",
        fps=5.0,
        surface_kwargs=[{"color": "tab:blue"}, {"color": "tab:orange"}],
        frame_hook=record,
    )

    # The whole scene is drawn in one collection, so that matplotlib depth
    # sorts it as one thing, and the old one is cleared away each frame.
    assert counts == [1, 1, 1]
    # Both styles still reach the output: two bodies, two distinct colours.
    assert all(len(palette) == 2 for palette in palettes), palettes


@needs_ffmpeg
def test_progress_is_reported_once_per_frame(tmp_path):
    seen = []
    animate_meshes(
        moving_frames(count=5),
        tmp_path / "progress.mp4",
        fps=5.0,
        progress=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


# ---------------------------------------------------------------------------
# Integration with DiscreteCosseratRod
# ---------------------------------------------------------------------------


def cosserat_rod(phase: float, num_elements: int = 12):
    """
    @brief A coiling, tapering rod, for the integration test.

    @param phase Rotates the coil, so successive frames differ.
    @param num_elements Number of elements.
    @return A @c DiscreteCosseratRod.
    """

    parameter = np.linspace(0.0, 2.0 * np.pi, num_elements + 1)
    positions = np.stack(
        [
            0.3 * np.cos(parameter + phase),
            0.3 * np.sin(parameter + phase),
            np.linspace(0.0, 1.0, num_elements + 1),
        ],
        axis=1,
    )
    segments = np.diff(positions, axis=0)
    tangents = segments / np.linalg.norm(segments, axis=1)[:, None]

    frames = []
    for tangent in tangents:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(seed @ tangent) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        normal = seed - (seed @ tangent) * tangent
        normal /= np.linalg.norm(normal)
        frames.append(np.stack([normal, np.cross(tangent, normal), tangent]))

    radii = np.linspace(0.06, 0.02, num_elements)
    return DiscreteCosseratRod(positions, np.array(frames), radii)


@needs_ffmpeg
@has_ffprobe
def test_animates_cosserat_rods_end_to_end(tmp_path):
    frames = [(0.01 * index, [cosserat_rod(0.2 * index)]) for index in range(8)]
    output = tmp_path / "cosserat.mp4"

    result = animate_meshes(
        frames,
        output,
        slowdown=10.0,
        to_mesh_kwargs={"num_circumferential": 12},
        surface_kwargs={"cmap": "viridis"},
    )

    assert result.num_frames == 8
    # 0.01 s between frames, played ten times slower, is ten frames a second.
    assert result.fps == pytest.approx(10.0)
    assert int(probe(output)["nb_frames"]) == 8


@needs_ffmpeg
def test_two_rods_share_one_set_of_bounds(tmp_path):
    frames = [
        (0.01 * index, [cosserat_rod(0.2 * index), cosserat_rod(0.2 * index + np.pi)])
        for index in range(4)
    ]
    result = animate_meshes(frames, tmp_path / "two.mp4", fps=5.0)

    # Both coils are inside the same box.
    assert result.bounds.half_extent > 0.3
