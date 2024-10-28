#!/usr/bin/env python3

# Developed by Neng Wang
# Brief: visualizer based on open3D for bounding box
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import os
import open3d as o3d

def load_scan(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32).reshape((-1,4))
    return scan


def load_bounding_box_label(boundingbox_label_path):
    bounding_label=np.load(boundingbox_label_path,allow_pickle=True)
    if len(bounding_label)==0:
      bounding_label = []
      bounding_label.append(["nothing",0,1,np.array([0,0,0,0,0,0,0])])
    return bounding_label

data_dir = '../data/nuScenes_kitti/train/0001'
def main():
    scan_folder = os.path.join(data_dir, 'velodyne')
    label_folder = os.path.join(data_dir,'labels')
    bbox_folder = os.path.join(data_dir,'boundingbox')


    scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
    scan_paths.sort()
    label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_folder)) for f in fn]
    label_paths.sort()
    bbox_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(bbox_folder)) for f in fn]
    bbox_paths.sort()
    frame_idx=0
    scan = load_scan(scan_paths[frame_idx])
    bbox = load_bounding_box_label(bbox_paths[frame_idx])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
  
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd)
    print("bbox:",bbox)
    if len(bbox)!=0:
        for bounding_box_idx in range(len(bbox)):
            bbox_center = np.zeros(3,dtype=np.float32)
            bbox_extent = np.zeros(3,dtype=np.float32)
            bbox_center[0] = bbox[bounding_box_idx][3][0]
            bbox_center[1] = bbox[bounding_box_idx][3][1]
            bbox_center[2] = bbox[bounding_box_idx][3][2]
            bbox_extent[0] = bbox[bounding_box_idx][3][3]
            bbox_extent[1] = bbox[bounding_box_idx][3][4]
            bbox_extent[2] = bbox[bounding_box_idx][3][5]
            theta = bbox[bounding_box_idx][3][6]
            inst_lable =bbox[bounding_box_idx][1]
            
            Rotation_eye = np.eye(3)
            color=[0,0,0]
            obb =  o3d.geometry.OrientedBoundingBox(center=bbox_center,R=Rotation_eye,extent=bbox_extent)
            Rotation = obb.get_rotation_matrix_from_xyz((0,0,theta))
            obb_rotation =  o3d.geometry.OrientedBoundingBox(center=bbox_center,R=Rotation,extent=bbox_extent)
            #Rotation = obb.get_rotation_matrix_from_xyz((0,0,theta))
            obb_rotation.color = color
            vis.add_geometry(obb_rotation)

        
    while True:
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

if __name__=="__main__":
    main()