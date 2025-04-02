#!/usr/bin/env python3
# Developed by Neng Wang
# 	and the main_funciton 'prosess_one_seq' refers to Xieyuanli Chen’s gen_residual_images.py
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images

import os
# os.environ["OMP_NUM_THREADS"] = "4"
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from icecream import ic
import time 
import sys
sys.path.append("/home/wangneng/code/InsMOS_bev/utils/src/build")
import Array_Index

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
            pose_path: (Complete) filename for the pose file
        Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']
    
    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')
    
    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
    
    except FileNotFoundError:
        print('Calibrations are not avaialble.')
    
    return np.array(T_cam_velo)


def load_vertex(scan_path):
    """ Load 3D points of a scan. The fileformat is the .bin format used in
        the KITTI dataset.
        Args:
            scan_path: the (full) filename of the scan file
        Returns:
            A nx4 numpy array of homogeneous points (x, y, z, 1).
    """
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_files(folder):
    """ Load all files in a folder and sort.
    """
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths


def load_labels(label_path):
    """ Load semantic and instance labels in SemanticKitti format.
    """
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label


def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_yaml(path):
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(path))
    return config

def gen_bev(pointcloud,limit_range,grid_size):
    bev_height = int((limit_range[3]-limit_range[0])/grid_size)
    bev_width = int((limit_range[4]-limit_range[1])/grid_size)
    bev_img1 = limit_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
    bev_max = limit_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
    bev_min = limit_range[5]*np.ones([bev_height,bev_width],dtype=np.float32)

    bev_img = Array_Index.pointcloud2bevHeightSingleThread_1(pointcloud,limit_range,bev_img1,bev_max,bev_min,grid_size)
    # showbev(bev_img)
    return bev_img

def process_one_seq(config):
    # specify parameters
    num_frames = config['num_frames']
    limit_range = config['DATA']['POINT_CLOUD_RANGE']
    grid_size = config['DATA']['GRID_SIZE_BEV']

    print("grid size:",grid_size)
    num_last_n = config['num_last_n']

    # specify the output folders
    residual_image_folder = config['residual_image_folder']
    check_and_makedirs(residual_image_folder)


    # load poses
    pose_file = config['pose_file']
    print(pose_file)
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    calib_file = config['calib_file']
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_folder = config['scan_folder']
    scan_paths = load_files(scan_folder)

    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    # range_image_params = config['range_image']

    # generate residual images for the whole sequence
    frame = 0
    duration = 0
    for frame_idx in tqdm(range(len(scan_paths))):
        frame = frame + 1
        start_time = time.time()
        file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
        bev_height = int((limit_range[3]-limit_range[0])/grid_size)
        bev_width = int((limit_range[4]-limit_range[1])/grid_size)


        # for the first N frame we generate a dummy file
        if frame_idx < num_last_n:
            diff_image = np.array([0,0,0]) # [H,W] range (0 is no data)
            np.save(file_name, diff_image)
        else:
            # load current scan and generate current range image
            current_pose = poses[frame_idx]
            current_scan = load_vertex(scan_paths[frame_idx])

            current_bev = gen_bev(current_scan[:,:3],limit_range,grid_size)

            # load last scan, transform into the current coord and generate a transformed last range image
            last_pose = poses[frame_idx - num_last_n]
            last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
            last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T

            last_scan_transformed_bev = gen_bev(last_scan_transformed[:,:3],limit_range,grid_size)

            diff_image = current_bev - last_scan_transformed_bev

            # 保存索引
            ben_conv = np.zeros([bev_height*bev_width,3],dtype=np.float32)
            bev_index = Array_Index.bev2index(diff_image,ben_conv)
            bev_index_abs = abs(bev_index)
            bev_index_no_empty = bev_index[bev_index_abs[:,2]>0,:]

            # save residual image
            np.save(file_name, bev_index_no_empty)
            end_time = time.time()
            duration = duration + end_time - start_time
            # average_time = duration / frame
            # print("process frame count:",frame)
            # print("average time:",average_time)

if __name__ == '__main__':

    # load config file
    # config_filename = 'config/data_preparing_hesai32.yaml'
    config_filename = 'config/config.yaml'
    config = load_yaml(config_filename)

    # used for kitti-raw and kitti-road
    for seq in range(10, 11): # sequences id

        for i in range(10,14): # residual_image_i

            # Update the value in config to facilitate the iterative loop
            config['num_frames'] = -1
            config['num_last_n'] = i
            config['scan_folder'] = f"/home/wangneng/DataFast/kitti/sequences/{'%02d'%seq}/velodyne"
            config['pose_file'] = f"/home/wangneng/DataFast/kitti/sequences/{'%02d'%seq}/poses.txt"
            config['calib_file'] = f"//home/wangneng/DataFast/kitti/sequences/{'%02d'%seq}/calib.txt"
            # config['residual_image_folder'] = f"data/sequences/{'%02d'%seq}/residual_images_{i}"
            # config['visualization_folder'] = f"data/sequences/{'%02d'%seq}/visualization_{i}"
            config['residual_image_folder'] = f"/home/wangneng/DataFast/kitti/sequences/{'%02d'%seq}/residual_bev_images_{i}"
            # config['visualization_folder'] = f"/home/wangneng/DataFast/kitti/sequences/{'%02d'%seq}/visualization_{i}"
            ic(config)
            process_one_seq(config)

