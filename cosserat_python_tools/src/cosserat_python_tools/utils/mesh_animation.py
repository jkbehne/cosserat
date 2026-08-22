"""
@file mesh_animation.py
@brief Turns a time-ordered sequence of mesh collections into a video.

A simulation frame is a time together with the meshes visible at that time, so
an animation is a sequence of those. This module renders each frame with
matplotlib and encodes the result, which is enough for reviewing a run and for
most presentation use.

@section anim_bounds Why the bounds are computed up front

The single thing that separates a usable animation from an unusable one is
fixing the axis limits across the whole sequence. Left to itself matplotlib
rescales every frame to fit whatever it is currently drawing, so the camera
appears to breathe and a body that is barely moving looks like it is pulsing.
Every frame is therefore visited once before rendering to find the bounds that
contain all of them, and those limits are then held constant.

The same reasoning applies to a colour scale: if you colour by a scalar, its
normalisation has to be global too, or a constant field shimmers. Pass explicit
@c vmin and @c vmax through @c surface_kwargs when doing that.

@section anim_timing Timing

Frames from a simulation are usually uniformly spaced in simulation time, since
they are written every N steps. Given that, the frame rate follows from how
fast you want the result to play: @ref frames_per_second turns a slowdown
factor into an fps, so asking for four times slow motion is a number you can
reason about rather than one you have to work out.

If the times are not uniformly spaced, encoding at a constant frame rate
silently misrepresents the dynamics -- motion looks fast wherever the sampling
was sparse. That case is rejected rather than guessed at.

@section anim_cost What it costs

Rendering is the whole budget, and a three dimensional surface cannot be
blitted: matplotlib has no in place update for @c plot_surface, so each frame
is drawn from scratch. Measured on a rod of forty elements at twenty eight
vertices around, that is roughly 100 ms per mesh per frame, so a few hundred
frames of a two body scene takes under a minute. Larger scenes are better
served by a dedicated renderer.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


## @brief A mesh as three coordinate grids, laid out for @c plot_surface.
Grids = Tuple[np.ndarray, np.ndarray, np.ndarray]

## @brief Anything this module can draw: grids, or an object offering to_mesh.
MeshLike = Any

## @brief One frame: a simulation time and the meshes visible at it.
Frame = Tuple[float, Sequence[MeshLike]]

## @brief Produces the frame at an index, for sequences too large to hold.
FrameProvider = Callable[[int], Frame]

## @brief Relative tolerance on the uniformity of the frame spacing.
FRAME_INTERVAL_TOLERANCE = 1e-6

## @brief Container extensions mapped to the writer that handles them.
WRITER_FOR_SUFFIX: Dict[str, str] = {
    ".mp4": "ffmpeg",
    ".mkv": "ffmpeg",
    ".mov": "ffmpeg",
    ".avi": "ffmpeg",
    ".webm": "ffmpeg",
    ".gif": "pillow",
}


@dataclass(frozen=True)
class Bounds:
    """
    @brief An axis aligned box, held as a centre and a single half extent.

    One half extent rather than three, because the three axes are given a
    common scale: a rod is long and thin, and letting each axis fill the figure
    independently distorts the shape and makes bends read as the wrong angle.
    """

    ## @brief Centre of the box.
    centre: np.ndarray
    ## @brief Half the longest side, applied to every axis.
    half_extent: float

    def limits(self) -> Tuple[Tuple[float, float], ...]:
        """
        @brief The box as three (low, high) pairs.
        @return One pair per axis, in x, y, z order.
        """
        return tuple(
            (float(value - self.half_extent), float(value + self.half_extent))
            for value in self.centre
        )


@dataclass(frozen=True)
class AnimationResult:
    """
    @brief What an animation run produced.
    """

    ## @brief Path the video was written to.
    output_path: Path
    ## @brief Number of frames encoded.
    num_frames: int
    ## @brief Frame rate used.
    fps: float
    ## @brief Playback length in seconds.
    duration_seconds: float
    ## @brief Bounds the axes were held at.
    bounds: Bounds


def as_grids(mesh: MeshLike, **to_mesh_kwargs) -> Grids:
    """
    @brief Normalises anything drawable into three coordinate grids.

    Accepts either a mesh object exposing @c to_mesh, such as a
    @c DiscreteCosseratRod, or a ready made triple of grids.

    @param mesh The mesh to normalise.
    @param to_mesh_kwargs Forwarded to @c to_mesh when the object provides it.
    @return The triple @c (X, Y, Z).

    @throws TypeError if the object is neither of those things.
    @throws ValueError if the grids disagree in shape, are not two
            dimensional, or hold non-finite values.
    """
    if hasattr(mesh, "to_mesh"):
        grids = mesh.to_mesh(**to_mesh_kwargs)
    else:
        grids = mesh

    try:
        x_grid, y_grid, z_grid = (np.asarray(part, dtype=float) for part in grids)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "a mesh must expose to_mesh() or be a triple of coordinate grids, "
            f"got {type(mesh).__name__}"
        ) from error

    if x_grid.ndim != 2:
        raise ValueError(
            f"coordinate grids must be two dimensional, got shape {x_grid.shape}"
        )
    if not (x_grid.shape == y_grid.shape == z_grid.shape):
        raise ValueError(
            "the three coordinate grids must agree in shape, got "
            f"{x_grid.shape}, {y_grid.shape}, {z_grid.shape}"
        )
    for name, grid in (("X", x_grid), ("Y", y_grid), ("Z", z_grid)):
        if not np.isfinite(grid).all():
            raise ValueError(f"the {name} grid holds non-finite values")

    return x_grid, y_grid, z_grid


def uniform_frame_interval(
    times: Sequence[float], tolerance: float = FRAME_INTERVAL_TOLERANCE
) -> float:
    """
    @brief The spacing between frames, requiring it to be uniform.

    @param times Frame times, which must be strictly increasing.
    @param tolerance Relative tolerance on the spread of the intervals.
    @return The common interval between frames.

    @throws ValueError if there are fewer than two times, if they are not
            strictly increasing, or if the spacing varies by more than
            @p tolerance. Encoding non-uniform frames at a constant rate would
            quietly misrepresent the dynamics, so it is refused rather than
            approximated.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("at least two frames are needed to infer a frame interval")
    if not np.isfinite(times).all():
        raise ValueError("frame times must be finite")

    intervals = np.diff(times)
    if (intervals <= 0.0).any():
        worst = int(np.argmin(intervals))
        raise ValueError(
            f"frame times must strictly increase, but frame {worst + 1} is not "
            f"after frame {worst}"
        )

    mean_interval = float(intervals.mean())
    spread = float(intervals.max() - intervals.min())
    if spread > tolerance * mean_interval:
        raise ValueError(
            f"frame times are not uniformly spaced (intervals range over "
            f"{spread:.3e} about a mean of {mean_interval:.3e}). Pass fps "
            "explicitly to encode them anyway, or resample onto a uniform grid"
        )
    return mean_interval


def frames_per_second(times: Sequence[float], slowdown: float = 1.0) -> float:
    """
    @brief Frame rate that plays a sequence at the requested speed.

    A slowdown of one plays in real time, so one second of simulation takes one
    second to watch; two plays at half speed, and so on.

    @param times Frame times, uniformly spaced and strictly increasing.
    @param slowdown How many seconds of playback per second of simulation;
           must be finite and positive.
    @return The frame rate in frames per second.

    @throws ValueError if @p slowdown is not positive or the times are unusable.
    """
    if not np.isfinite(slowdown) or slowdown <= 0.0:
        raise ValueError(f"slowdown must be finite and positive, got {slowdown}")
    interval = uniform_frame_interval(times)
    return 1.0 / (interval * slowdown)


class _FrameSource:
    """
    @brief Uniform access to frames held in a list or produced on demand.

    A list is convenient and a provider keeps memory flat for long runs, so
    both are accepted and everything downstream sees one interface.
    """

    def __init__(
        self,
        frames: Union[Sequence[Frame], FrameProvider],
        num_frames: Optional[int],
    ) -> None:
        """
        @brief Wraps a frame sequence or provider.

        @param frames Either a sequence of frames, or a callable taking a frame
               index and returning one.
        @param num_frames Required when @p frames is a callable, and rejected
               when it is a sequence, whose length already says.

        @throws ValueError if the count is missing, contradictory or not
                positive.
        """
        if callable(frames):
            if num_frames is None:
                raise ValueError(
                    "num_frames is required when frames is a callable, since a "
                    "provider has no length"
                )
            self._provider: FrameProvider = frames
            self._count = int(num_frames)
        else:
            sequence = list(frames)
            if num_frames is not None and num_frames != len(sequence):
                raise ValueError(
                    f"num_frames ({num_frames}) contradicts the {len(sequence)} "
                    "frames supplied"
                )
            self._provider = sequence.__getitem__
            self._count = len(sequence)

        if self._count < 1:
            raise ValueError("an animation needs at least one frame")

    def __len__(self) -> int:
        """@brief Number of frames."""
        return self._count

    def __getitem__(self, index: int) -> Frame:
        """
        @brief The frame at an index, validated.

        @param index Frame index.
        @return The time and the meshes at it.

        @throws ValueError if the frame is not a time paired with meshes.
        """
        frame = self._provider(index)
        try:
            time, meshes = frame
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"frame {index} must be a (time, meshes) pair, got {frame!r}"
            ) from error

        time = float(time)
        if not np.isfinite(time):
            raise ValueError(f"the time of frame {index} is not finite")
        if isinstance(meshes, (str, bytes)):
            raise ValueError(f"the meshes of frame {index} must be a collection")
        return time, list(meshes)

    def times(self) -> np.ndarray:
        """
        @brief Every frame time, in order.
        @return An array of length @c len(self).
        """
        return np.array([self[index][0] for index in range(self._count)], dtype=float)


def compute_global_bounds(
    frames: Union[Sequence[Frame], FrameProvider],
    num_frames: Optional[int] = None,
    padding: float = 0.05,
    to_mesh_kwargs: Optional[Dict[str, Any]] = None,
) -> Bounds:
    """
    @brief The box containing every mesh of every frame.

    Visiting all frames once before rendering is what lets the axis limits stay
    fixed, which is the difference between an animation of a moving body and an
    animation of a moving camera. See the module notes.

    @param frames A frame sequence, or a provider.
    @param num_frames Required when @p frames is a provider.
    @param padding Fraction of the extent to add as margin; must not be
           negative.
    @param to_mesh_kwargs Forwarded to each mesh's @c to_mesh.
    @return The bounds, with a common half extent for all three axes.

    @throws ValueError if no frame holds any mesh, or if @p padding is
            negative.
    """
    if padding < 0.0:
        raise ValueError(f"padding must not be negative, got {padding}")

    source = _FrameSource(frames, num_frames)
    to_mesh_kwargs = to_mesh_kwargs or {}

    lowest = np.full(3, np.inf)
    highest = np.full(3, -np.inf)
    seen_any = False

    for index in range(len(source)):
        _, meshes = source[index]
        for mesh in meshes:
            x_grid, y_grid, z_grid = as_grids(mesh, **to_mesh_kwargs)
            stacked = np.stack(
                [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1
            )
            lowest = np.minimum(lowest, stacked.min(axis=0))
            highest = np.maximum(highest, stacked.max(axis=0))
            seen_any = True

    if not seen_any:
        raise ValueError("no frame contained any mesh, so there is nothing to bound")

    centre = 0.5 * (lowest + highest)
    half_extent = 0.5 * float((highest - lowest).max())
    # A single point or a perfectly flat scene would give a zero sized box,
    # which matplotlib cannot draw into.
    half_extent = max(half_extent, 1e-9)
    return Bounds(centre=centre, half_extent=half_extent * (1.0 + padding))


def _writer_for(output_path: Path, fps: float, bitrate: Optional[int]):
    """
    @brief Selects and builds the writer for an output container.

    @param output_path Where the video will be written; the suffix selects the
           writer.
    @param fps Frame rate to encode at.
    @param bitrate Bitrate for the video writers, or None for the default.
    @return A matplotlib writer instance.

    @throws ValueError if the suffix is not a container this module handles.
    @throws RuntimeError if the writer for that container is not installed,
            naming what is missing rather than failing deep inside matplotlib.
    """
    suffix = output_path.suffix.lower()
    if suffix not in WRITER_FOR_SUFFIX:
        raise ValueError(
            f"unsupported output format '{suffix}'. Supported: "
            + ", ".join(sorted(WRITER_FOR_SUFFIX))
        )

    name = WRITER_FOR_SUFFIX[suffix]
    if not animation.writers.is_available(name):
        detail = (
            "ffmpeg was not found on PATH; install it, or write a .gif instead"
            if name == "ffmpeg"
            else f"the '{name}' writer is unavailable"
        )
        raise RuntimeError(f"cannot write {suffix}: {detail}")

    writer_class = animation.writers[name]
    if name == "ffmpeg":
        return writer_class(fps=fps, bitrate=bitrate)
    return writer_class(fps=fps)


def _resolve_surface_kwargs(
    surface_kwargs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]],
    mesh_index: int,
) -> Dict[str, Any]:
    """
    @brief Per-mesh drawing options, cycling if a sequence was given.

    Passing a sequence lets separate bodies be styled differently, which is how
    two rods in one scene are told apart.

    @param surface_kwargs A single option dict, a sequence of them, or None.
    @param mesh_index Index of the mesh within its frame.
    @return The options for that mesh.
    """
    if surface_kwargs is None:
        return {}
    if isinstance(surface_kwargs, dict):
        return dict(surface_kwargs)
    options = list(surface_kwargs)
    if not options:
        return {}
    return dict(options[mesh_index % len(options)])


def default_title(time: float, index: int) -> str:
    """
    @brief The title drawn on each frame when none is supplied.

    @param time Simulation time of the frame.
    @param index Frame index; unused here.
    @return The title text.
    """
    del index
    return f"t = {time:.3f} s"


def animate_meshes(
    frames: Union[Sequence[Frame], FrameProvider],
    output_path: Union[str, Path],
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    slowdown: float = 1.0,
    padding: float = 0.05,
    figure_size: Tuple[float, float] = (6.0, 5.0),
    dpi: int = 100,
    surface_kwargs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    to_mesh_kwargs: Optional[Dict[str, Any]] = None,
    elevation: float = 20.0,
    azimuth: float = -60.0,
    axis_labels: Tuple[str, str, str] = ("x", "y", "z"),
    title_formatter: Optional[Callable[[float, int], str]] = default_title,
    bitrate: Optional[int] = 2400,
    frame_hook: Optional[Callable[[Any, int, float], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> AnimationResult:
    """
    @brief Renders a sequence of mesh collections to a video file.

    @param frames A sequence of @c (time, meshes) pairs, or a callable that
           returns one given an index.
    @param output_path Where to write. The suffix picks the encoder; see
           @ref WRITER_FOR_SUFFIX.
    @param num_frames Required when @p frames is a callable.
    @param fps Frame rate. When omitted it is derived from the frame times and
           @p slowdown, which requires uniformly spaced times.
    @param slowdown Seconds of playback per second of simulation. Ignored when
           @p fps is given.
    @param padding Fraction of margin around the scene.
    @param figure_size Figure size in inches.
    @param dpi Dots per inch, so pixel size is @p figure_size times this.
    @param surface_kwargs Options forwarded to @c plot_surface, either one dict
           for every mesh or a sequence cycled across the meshes in a frame.
    @param to_mesh_kwargs Options forwarded to each mesh's @c to_mesh.
    @param elevation Camera elevation in degrees.
    @param azimuth Camera azimuth in degrees.
    @param axis_labels Labels for the three axes.
    @param title_formatter Builds the per-frame title from the time and index.
           Pass None for no title.
    @param bitrate Bitrate for video containers; ignored for GIF.
    @param frame_hook Called with the axes, frame index and time after each
           frame is drawn, for annotations.
    @param progress Called with the frame index and total after each frame.
    @return What was produced, including the frame rate and bounds used.

    @throws ValueError for malformed frames, an unsupported container, or
            non-uniform times when the frame rate has to be inferred.
    @throws RuntimeError if the encoder for the chosen container is missing.
    """
    source = _FrameSource(frames, num_frames)
    output_path = Path(output_path)
    to_mesh_kwargs = to_mesh_kwargs or {}

    if fps is None:
        if len(source) < 2:
            raise ValueError(
                "fps cannot be inferred from a single frame; pass it explicitly"
            )
        fps = frames_per_second(source.times(), slowdown)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")

    bounds = compute_global_bounds(
        frames, num_frames, padding=padding, to_mesh_kwargs=to_mesh_kwargs
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _writer_for(output_path, fps, bitrate)

    # A bare Figure with an Agg canvas, rather than pyplot, so nothing is added
    # to the global figure registry and no display backend is needed.
    figure = Figure(figsize=figure_size, dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(projection="3d")

    limits = bounds.limits()

    with writer.saving(figure, str(output_path), dpi):
        for index in range(len(source)):
            time, meshes = source[index]

            # plot_surface has no in place update, so the axes are rebuilt each
            # frame. This is also why blitting is not available here.
            axes.clear()
            for mesh_index, mesh in enumerate(meshes):
                x_grid, y_grid, z_grid = as_grids(mesh, **to_mesh_kwargs)
                options = _resolve_surface_kwargs(surface_kwargs, mesh_index)
                options.setdefault("linewidth", 0)
                options.setdefault("antialiased", True)
                axes.plot_surface(x_grid, y_grid, z_grid, **options)

            # Held at the global bounds, so the camera does not chase the
            # motion. One extent for all three axes keeps proportions honest.
            axes.set_xlim(*limits[0])
            axes.set_ylim(*limits[1])
            axes.set_zlim(*limits[2])
            axes.set_box_aspect((1.0, 1.0, 1.0))
            axes.view_init(elev=elevation, azim=azimuth)

            axes.set_xlabel(axis_labels[0])
            axes.set_ylabel(axis_labels[1])
            axes.set_zlabel(axis_labels[2])
            if title_formatter is not None:
                axes.set_title(title_formatter(time, index))

            if frame_hook is not None:
                frame_hook(axes, index, time)

            writer.grab_frame()

            if progress is not None:
                progress(index + 1, len(source))

    return AnimationResult(
        output_path=output_path,
        num_frames=len(source),
        fps=float(fps),
        duration_seconds=len(source) / float(fps),
        bounds=bounds,
    )


def ffmpeg_available() -> bool:
    """
    @brief Whether video containers can be written.
    @return True when ffmpeg is on PATH and matplotlib can drive it.
    """
    return shutil.which("ffmpeg") is not None and animation.writers.is_available(
        "ffmpeg"
    )
