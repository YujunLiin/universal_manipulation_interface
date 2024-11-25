import os
from pathlib import Path
from typing import Union
import os

import numpy as np

def save_point_cloud_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path,str] = "out.ply"):
    """Converts a comma separated list of map point coordinates into
    PLY format for viewing the generated map.

    Args:
        input_file (str or Path): Path to the input file which is expected to
        be a .csv file with the columns pos_x, pos_y, pos_z designating the
        coordinates of the points in the world reference frame.

        output_file (str or Path): Path to the output .ply file, format is
        described here: https://paulbourke.net/dataformats/ply/
    """

    coords = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    ply_header = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'end_header' % x.shape[0]

    np.savetxt(output_file, np.column_stack((x, y, z)), fmt='%f %f %f', header=ply_header, comments='')

def save_trajectory_from_ORB_SLAM(input_file: Union[Path, str], output_file: Union[Path,str] = "out_trajectory.ply"):
    """Converts the saved trajectory file from ORB-SLAM3 to a point cloud to then
    show alongside the mapped cloud.

    The input file is expected to be in the format (KITTI format with image path prepended, can ignore it):
    /path/to/image0.png R_00 R_01 R_02 t_0 R_10 R_11 R_12 t_1 R_20 R_21 R_22 t_2
    /path/to/image1.png R_00 R_01 R_02 t_0 R_10 R_11 R_12 t_1 R_20 R_21 R_22 t_2

    Where the R terms are the rotation and t terms are the translation terms
    of the homogeneous transformation matrix T_w_cam0.
    """
    coords = np.genfromtxt(input_file, delimiter=",", skip_header=1)

    x = coords[:, 5]
    y = coords[:, 6]
    z = coords[:, 7]

    # RGB values for each point on the trajectory, set to be light green
    r = np.ones_like(x) * 144
    g = np.ones_like(x) * 238
    b = np.ones_like(x) * 144

    ply_header = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header' % x.shape

    np.savetxt(output_file, np.column_stack((x, y, z, r, g, b)), fmt='%f %f %f %d %d %d', header=ply_header, comments='')


if __name__ == "__main__":
    # just for testing locally

    # input_map = "/Users/max/Desktop/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.04.02.335867/point_cloud.csv"
    # output_map = "/Users/max/Desktop/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.04.02.335867/point_cloud.ply"

    #############################no_mask_mirrors
    # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/no_mask_mirrors/camera_trajectory_mapping_video.csv"
    output_trajectory = "./test_tiny_dataset/no_mask_mirrors/camera_trajectory_mapping_video.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)

     # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/no_mask_mirrors/camera_trajectory_0.csv"
    output_trajectory = "./test_tiny_dataset/no_mask_mirrors/camera_trajectory_0.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)

    #############################mask_mirrors
    # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/mask_mirrors/camera_trajectory_mapping_video.csv"
    output_trajectory = "./test_tiny_dataset/mask_mirrors/camera_trajectory_mapping_video.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)

     # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/mask_mirrors/camera_trajectory_0.csv"
    output_trajectory = "./test_tiny_dataset/mask_mirrors/camera_trajectory_0.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)

    ############################# mask_gripper+mirrors
    # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/mask_gripper+mirrors/camera_trajectory_mapping_video.csv"
    output_trajectory = "./test_tiny_dataset/mask_gripper+mirrors/camera_trajectory_mapping_video.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)

     # no_mask_mirrors, mask_mirrors, mask_gripper+mirrors
    input_trajectory = "./test_tiny_dataset/mask_gripper+mirrors/camera_trajectory_0.csv"
    output_trajectory = "./test_tiny_dataset/mask_gripper+mirrors/camera_trajectory_0.ply"

    # # save_point_cloud_from_ORB_SLAM(input_map, output_map)
    save_trajectory_from_ORB_SLAM(input_trajectory, output_trajectory)