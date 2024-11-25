from .csv_to_ply import save_trajectory_from_ORB_SLAM, save_point_cloud_from_ORB_SLAM
import os
from pathlib import Path
from typing import Union

def convert_output_to_ply(example_demo_session_dir: Union[Path,str] = None, output_ply_dir: Union[Path,str] = None):
    save_to_original_folder = True if output_ply_dir else False

    example_demo_session_dir = Path(example_demo_session_dir)
    if not example_demo_session_dir.is_dir():
        raise ValueError("example_demo_session_dir invalid")
    
    demos_dir = example_demo_session_dir.joinpath('demos')

    # iterate through each folder
    for folder in demos_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.stem.startswith('gripper_calibration'):
            continue # gripper calibration does not need visualization

        # Create output folder
        if not save_to_original_folder:
            output_ply_dir = folder.joinpath('ply')
            output_ply_dir.mkdir(parents=True, exist_ok=True)

        for csv in folder.iterdir():
            if not csv.is_file() or not csv.name.endswith('.csv'):
                continue
            if csv.stem.endswith('absolute_camera_trajectory'):
                save_trajectory_from_ORB_SLAM(csv, output_ply_dir.joinpath(csv.stem + '.ply'))
            if csv.stem.endswith('point_cloud'):
                save_point_cloud_from_ORB_SLAM(csv, output_ply_dir.joinpath(csv.stem + '.ply'))


if __name__ == "__main__":
    # only for testing locally
    # need to delete '.' before imported module name
    convert_output_to_ply('/Users/max/Desktop/universal_manipulation_interface/example_demo_session')