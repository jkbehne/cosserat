"""@file make_rod_animation.py

@brief Renders a run's diagnostic output as a video.

Reads whatever the simulation recorded, rods and rigid mesh bodies alike, and
animates all of it together. Mesh bodies need no special handling from the
caller: they are recognised by what they wrote and drawn alongside the rods.

@section mra_framing A note on framing

The view is a cube, because giving all three axes one scale is the only way the
geometry keeps its proportions. That is unkind to a scene built on a wide flat
ground: the ground sets the horizontal extent, the cube gives the vertical
extent to match, and much of the frame ends up as empty air above and below.

@ref main therefore takes an optional @c focus_height. Given one, the view is
cropped to that many metres above the lowest geometry, which for a ground based
scene is usually what you want to look at. The proportions stay honest either
way; only the crop changes.

The camera can be aimed with @c elevation and @c azimuth, both in degrees. A
low elevation looks at such a scene side on, which shows a rod resting on
something far more clearly than looking down at it does.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from cosserat_python_tools.utils.binary_file_utils import (
    get_bodies_from_dir,
)
from cosserat_python_tools.utils.mesh_animation import (
    Bounds,
    animate_meshes,
    drawable_points,
)

MIN_TIME_STEPS = 10


def main(
    rod_data_dir: str,
    output_file_name: str,
    focus_height: Optional[float] = None,
    elevation: Optional[float] = None,
    azimuth: Optional[float] = None,
    num_circumferential: int = 20,
) -> None:
    """@brief Turns a diagnostic directory into a video.

    @param rod_data_dir Root the diagnostics wrote beneath.
    @param output_file_name Video to write; the container follows the suffix.
    @param focus_height Crop the view to this many metres above the lowest
           geometry. Leave unset to frame everything.
    @param elevation Camera elevation in degrees; low values look side on.
    @param azimuth Camera azimuth in degrees.
    @param num_circumferential Vertices around each rod.
    """
    times_and_bodies = get_bodies_from_dir(
        directory=rod_data_dir, raise_if_empty=True,
    )
    times_and_bodies = sorted(times_and_bodies, key=lambda x: x[0])

    out_path = Path(output_file_name).expanduser()
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    to_mesh_kwargs = {"num_circumferential": num_circumferential}

    # Bodies are ordered by name within a frame, so cycling a palette gives
    # each one a stable colour for the whole run rather than a scene in which
    # everything is the same shade and nothing can be told apart.
    options = {
        "surface_kwargs": [
            {"color": colour}
            for colour in ("tab:blue", "0.75", "tab:orange", "tab:green",
                           "tab:red", "tab:purple")
        ]
    }
    if elevation is not None:
        options["elevation"] = float(elevation)
    if azimuth is not None:
        options["azimuth"] = float(azimuth)

    if focus_height is not None:
        # Crop to the band just above the lowest geometry. The scene's own
        # extent is needed for that, not the cube derived from it, whose floor
        # sits well below everything when the ground is wide and flat.
        lowest = np.full(3, np.inf)
        highest = np.full(3, -np.inf)
        for _, bodies in times_and_bodies:
            for body in bodies:
                points = drawable_points(body, **to_mesh_kwargs)
                lowest = np.minimum(lowest, points.min(axis=0))
                highest = np.maximum(highest, points.max(axis=0))

        half = 0.5 * float(focus_height)
        centre = 0.5 * (lowest + highest)
        centre[2] = lowest[2] + half
        options["bounds"] = Bounds(centre=centre, half_extent=half)

    animate_meshes(
        frames=times_and_bodies,
        output_path=out_path,
        to_mesh_kwargs=to_mesh_kwargs,
        **options,
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
