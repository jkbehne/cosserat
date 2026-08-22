from pathlib import Path

from cosserat_python_tools.utils.binary_file_utils import (
    get_cosserat_rods_from_dir,
)
from cosserat_python_tools.utils.mesh_animation import animate_meshes

MIN_TIME_STEPS = 10

def main(rod_data_dir: str, output_file_name: str) -> None:
    times_and_rods = get_cosserat_rods_from_dir(
        directory=rod_data_dir, raise_if_no_rods=True,
    )
    times_and_rods = sorted(times_and_rods, key=lambda x:x[0])
    out_path = Path(output_file_name).expanduser()
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    animate_meshes(frames=times_and_rods, output_path=out_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
