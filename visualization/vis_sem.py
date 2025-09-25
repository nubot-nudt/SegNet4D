#!/usr/bin/env python3

# Developed by Xieyuanli Chen and Neng Wang
# Brief: visualizer based on open3D for semantic segmentation
# This file is covered by the LICENSE file in the root of this project.

import sys
import yaml
import open3d as o3d
import pynput.keyboard as keyboard
import copy
import os
import numpy as np

def gen_color_map_semantic(sem_color_dict):
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    sem_color_lut = np.zeros((max_sem_key + 300, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      sem_color_lut[key] = np.array(value, np.float32) / 255.0
      temp = sem_color_lut[key][2]
      sem_color_lut[key][2] = sem_color_lut[key][0]
      sem_color_lut[key][0] = temp
      
    return sem_color_lut 

def load_files(folder):
  """ Load all files in a folder and sort.
  """
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths

def load_labels(label_path):
  label = np.fromfile(label_path, dtype=np.uint32)
  label = label.reshape((-1))
  # print("label:",label.shape)
  sem_label = label & 0xFFFF  # semantic label in lower half
  
  return sem_label

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

class vis_mos_results:
  """ LiDAR moving object segmentation results (LiDAR-MOS) visualizer
  Keyboard navigation:
    n: play next
    b: play back
    esc or q: exits
  """
  def __init__(self, config):
    # specify paths
    seq = str(config['seq']).zfill(4)
    self.nuscense_config = yaml.safe_load(open(config['nuscence_all_config']))
    dataset_root = config['dataset_root']
    pred_4d_root = config['prediction_4d_root']

    # specify folders
    scan_folder = os.path.join(dataset_root,seq, 'velodyne')
    gt_folder = os.path.join(pred_4d_root, 'sequences', seq,'predictions')
    
    # load files
    self.scan_files = load_files(scan_folder)
    self.scan_files = self.scan_files[2:]
    print("sequence len:",len(self.scan_files))
    self.gt_paths = load_files(gt_folder)
    
    # init frame
    self.current_points = load_vertex(self.scan_files[0])[:, :3]
    self.current_gt = load_labels(self.gt_paths[0])
    
    # init visualizer
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window()
  
    self.pcd = o3d.geometry.PointCloud()
    self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
    self.pcd.paint_uniform_color([0.2, 0.2, 0.2])
    colors = np.array(self.pcd.colors)

  
    self.pcd.colors = o3d.utility.Vector3dVector(colors)
  
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-45, -45, -5),
                                               max_bound=(45, 45, 5))
    self.pcd = self.pcd.crop(bbox)  # set view area
    self.vis.add_geometry(self.pcd)
  
    # init keyboard controller
    key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
    key_listener.start()
    
    # init frame index
    self.frame_idx = 0
    self.num_frames = len(self.scan_files)

  def on_press(self, key):
    try:
      if key.char == 'q':
        try:
          sys.exit(0)
        except SystemExit:
          os._exit(0)
        
      if key.char == 'n':
        if self.frame_idx < self.num_frames - 1:
          self.frame_idx += 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_gt= load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('Reach the end of this sequence!')
          
      if key.char == 'b':
        if self.frame_idx > 1:
          self.frame_idx -= 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_gt = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('At the start at this sequence!')
          
    except AttributeError:
      print('special key {0} pressed'.format(key))
      
  def on_release(self, key):
    try:
      if key.char == 'n':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_gt = load_labels(self.gt_paths[self.frame_idx])
    
      if key.char == 'b':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_gt = load_labels(self.gt_paths[self.frame_idx])
        
    except AttributeError:
      print('special key {0} pressed'.format(key))
  
  def run(self):
    current_points = self.current_points
    current_gt = self.current_gt
    if (len(current_points) == len(current_gt)):
      self.pcd.points = o3d.utility.Vector3dVector(current_points)
      self.pcd.paint_uniform_color([0.2, 0.2, 0.2])
      colors = np.array(self.pcd.colors)
  
      mapped_labels = copy.deepcopy(current_gt)
      for k, v in self.nuscense_config["learning_map"].items():
          mapped_labels[current_gt == k] = v

      color_dict = self.nuscense_config["color_map"]
      sem_color_lut =  gen_color_map_semantic(color_dict)
      colors = sem_color_lut[mapped_labels]

    
      self.pcd.colors = o3d.utility.Vector3dVector(colors)
      
      self.vis.update_geometry(self.pcd)
      self.vis.poll_events()
      self.vis.update_renderer()
  
  
if __name__ == '__main__':
  # load config file
  config_filename = './dataset_root.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  # init the mos visualizer
  visualizer = vis_mos_results(config)
  
  # run the visualizer
  while True:
    visualizer.run()