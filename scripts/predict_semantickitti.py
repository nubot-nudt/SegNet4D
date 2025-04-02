#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from pytorch_lightning import Trainer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

import sys
sys.path.append(".")
import time
import copy
import numpy as np
from pathlib import Path
import argparse

from easydict import EasyDict
import yaml
data_cfg = EasyDict()

import models.models as models
from dataloader.utils import load_poses, load_calib, load_files

from collections import defaultdict
from models.backbones_3d.voxel_generate import VoxelGenerate
from dataloader.src import Array_Index
from spconv.pytorch.utils import gather_features_by_pc_voxel_id


class DemoDataset(Dataset):
    def __init__(self, cfg, data_root,split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = data_root

        # Pose information
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATA"]["POSES"]

        # Semantic information
        self.semantic_mos_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_MOS_CONFIG_FILE"]))

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]

        self.split = split
        if self.split == "train":
            self.training = True 
            self.sequences = self.cfg["DATA"]["SPLIT"]["TRAIN"]
        elif self.split == "val":
            self.training = False
            self.sequences = self.cfg["DATA"]["SPLIT"]["VAL"]
        elif self.split == "test":
            self.training = False
            self.sequences = self.cfg["DATA"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")
        self.point_cloud_range = np.array(self.cfg["DATA"]["POINT_CLOUD_RANGE"],dtype=np.float32)
        self.point_cloud_range_min = np.array(self.cfg["DATA"]["POINT_CLOUD_RANGE_MIN"],dtype=np.float32)
        self.grid_size = self.cfg["DATA"]["VOXEL_SIZE"]
        self.grid_size_bev = self.cfg["DATA"]["GRID_SIZE_BEV"]

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)


        self.online_train = True

        self.filenames = {}

        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0

        for residual_idx in range(self.n_past_steps-1):
            exec("self.residual_files_" + str(residual_idx+1) + " = {}")

        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            path_to_seq = os.path.join(self.root_dir, seqstr)

            scan_path = os.path.join(path_to_seq, "velodyne")
            self.filenames[seq] = load_files(scan_path)
            if self.transform:
                self.poses[seq] = self.read_poses(path_to_seq)
                assert len(self.poses[seq]) == len(self.filenames[seq])
            else:
                self.poses[seq] = []

            # Get number of sequences based on number of past steps
            n_samples_sequence = max(
                0, len(self.filenames[seq]) - self.skip * (self.n_past_steps - 1)
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.skip * (self.n_past_steps - 1) + sample_idx
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load point clouds and get sequence

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        seq, scan_idx = self.idx_mapper[idx]

        # set past point clouds path
        from_idx = scan_idx - self.skip * (self.n_past_steps - 1)
        to_idx = scan_idx + 1
        past_indices = list(range(from_idx, to_idx, self.skip))
        past_files = self.filenames[seq][from_idx : to_idx : self.skip]
        
        data_dict = {}
        #load pointcloud
        list_past_point_clouds_raw = [self.read_point_cloud(f) for f in past_files]

        for i,pcd in enumerate(list_past_point_clouds_raw):
            if self.transform:
                from_pose = self.poses[seq][past_indices[i]]
                to_pose = self.poses[seq][past_indices[-1]]
                pcd[:,:3] = self.transform_point_cloud(pcd[:,:3], from_pose, to_pose)
                point_mask_max = self.mask_points_by_range(pcd,self.point_cloud_range)
                point_mask_min = self.mask_points_by_range(pcd,self.point_cloud_range_min)
                point_mask = point_mask_max & ~point_mask_min
                data_point_ranged = pcd[point_mask]
            list_past_point_clouds_raw[i] = data_point_ranged

        motion_feature = self.convert_pointclou2bev(list_past_point_clouds_raw)
        current_point_with_feature = np.hstack([list_past_point_clouds_raw[-1],motion_feature])
        current_point_with_feature_tensor = torch.tensor(current_point_with_feature,dtype=torch.float32)  

        meta = (seq,scan_idx,past_files)
        data_dict = {  
            "meta":meta,
            "current_point_with_feature_tensor":current_point_with_feature_tensor,  # [past_number ,N ,5]  5是x, y, z, r, t
            "batch_size_npast":self.n_past_steps, 
            "range_mask":point_mask
        }     
        return data_dict

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = np.linalg.inv(to_pose) @ from_pose
        NP = past_point_clouds.shape[0]
        xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def read_ground_label(self,filename):
        labels = np.fromfile(filename,dtype=np.uint32)
        labels = labels.reshape(-1)
        labels = labels & 0xFFFF
        return labels

    def mask_points_by_range(self,points, limit_range):
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] < limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] < limit_range[4]) \
                & (points[:, 2] >= limit_range[2]) & (points[:, 2] < limit_range[5])
        return mask

    def convert_pointclou2bev(self,list_pointcloud):

        current_point_cloud = list_pointcloud[-1]
        motion_feature = np.zeros([current_point_cloud.shape[0],self.n_past_steps-1],dtype=np.float16)
        bev_height = int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.grid_size[0])
        bev_width = int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.grid_size[1])
       
        bev_img_current = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
        bev_max_current = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
        bev_min_current= self.point_cloud_range[5]*np.ones([bev_height,bev_width],dtype=np.float32)

        current_bev = Array_Index.pointcloud2bevHeightSingleThread_1(current_point_cloud[:,:3],self.point_cloud_range,bev_img_current,bev_max_current,bev_min_current,self.grid_size[0])
        
        current_x_index = ((-current_point_cloud[:,0] - self.point_cloud_range[0])/self.grid_size[0]).astype(np.int32)
        current_y_index = ((-current_point_cloud[:,1] - self.point_cloud_range[1])/self.grid_size[1]).astype(np.int32)
   
        # 这里不行，搞一个mask出来
        current_x_index[current_x_index==1200] = 1199
        current_y_index[current_y_index==1000] = 999

        for i in range(0,self.n_past_steps-1): # 最后一帧是当前帧
            bev_img = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_max = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_min = self.point_cloud_range[5]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_frame = Array_Index.pointcloud2bevHeightSingleThread_1(list_pointcloud[self.n_past_steps-i-2][:,:3],self.point_cloud_range,bev_img,bev_max,bev_min,self.grid_size[0])
            
            residual_bev = current_bev - bev_frame
            motion_feature[:,i] = residual_bev[current_x_index,current_y_index]
        return motion_feature


    def timestamp_tensor(self,tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor
    
    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        print("pose_file:",pose_file)
        calib_file = os.path.join(path_to_seq, "calib.txt")
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        return poses

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))
        return point_cloud



    @staticmethod
    def collate_batch_test(batch):
        list_data_dict = [item for item in batch]
        return list_data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = torch.cat(val, dim=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        # coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coor_pad = torch.full((coor.shape[0],1),i)
                        coor_paded = torch.hstack([coor_pad,coor])

        
                        coors.append(coor_paded)
                    ret[key] = torch.cat(coors, dim=0)
                    
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = torch.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=torch.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
def parse_config():

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='config/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    return args


def to_original_labels(labels, semantic_config):
    original_labels = copy.deepcopy(labels)
    for k, v in semantic_config["learning_map_inv"].items():
        original_labels[labels == k] = v
    return original_labels

def softmax(x):
    exps = np.exp(x - np.max(x)) 
    return exps / np.sum(exps)

def main():

    args = parse_config()
    cfg = torch.load(args.ckpt,map_location='cuda:0')["hyper_parameters"]
    data_root = args.data_path

    cfg["TRAIN"]["BATCH_SIZE"] = 4
    # sequences = [11,12,13,14,15,16,17,18,19,20,21]
    sequences = [8]
    id = cfg["EXPERIMENT"]["ID"]
    cfg["DATA"]["ONLINE_TRAIN"] = True
    cfg["DATA"]["POSES"] = "poses.txt"
    cfg["DATA"]["POINT_CLOUD_RANGE_MIN"] = [-2.5, -1.5, -3, 2.5, 1.5, 1]
    print("pose:",cfg["DATA"]["POSES"])

    for seq_idx in sequences: 
        cfg["DATA"]["SPLIT"]["TEST"]  = [seq_idx]
        demo_dataset =  DemoDataset(cfg,data_root,split="test")
        demo_loader = DataLoader(
                dataset=demo_dataset,
                batch_size=cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=demo_dataset.collate_batch,
                shuffle=False,
                num_workers=cfg["DATA"]["NUM_WORKER"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
        )
        semantic_mos_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_MOS_CONFIG_FILE"]))
        semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
        semantic_config_all = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE_ALL"]))
        ignore_index = [
            key for key, ignore in semantic_mos_config["learning_ignore"].items() if ignore
        ]
        point_cloud_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        in_cahnnel = len(cfg["MODEL"]["POINT_FEATURE_ENCODING"]["src_feature_list"]) + cfg["MODEL"]["N_PAST_STEPS"] - 1
        voxel_generate = VoxelGenerate(cfg["DATA"]['VOXEL_SIZE']  ,point_cloud_range,100000,5,in_cahnnel) 
        model = models.SegNet4D.load_from_checkpoint(args.ckpt, hparams=cfg,map_location='cuda:0')
        model.cuda()
        model.eval()
        with torch.no_grad():
            Model_mode = 'test'
            for batch_idx, data_dict in enumerate(tqdm(demo_loader)):
                path_list = []

                seq,_,past_indice = data_dict['meta'][0]
                for j in range(0,len(data_dict['meta'])):
                    _,scan_idx,past_indice = data_dict['meta'][j]
                    path_list.append(scan_idx)
                if data_dict.get('voxels') != None:
                    data_dict["voxels"] = data_dict["voxels"].cuda()
                if data_dict.get('voxel_coords') != None:
                    data_dict["voxel_coords"] = data_dict["voxel_coords"].cuda()
                if data_dict.get('voxel_num_points') != None:
                    data_dict["voxel_num_points"] = data_dict["voxel_num_points"].cuda()
                if data_dict.get('pc_voxel_id') != None:
                    for j in range(0,len(data_dict["pc_voxel_id"])):
                        data_dict["pc_voxel_id"][j] = data_dict["pc_voxel_id"][j].cuda()
                if data_dict.get('current_point_with_feature_tensor') != None:
                    for j in range(0,len(data_dict["current_point_with_feature_tensor"])):
                        data_dict["current_point_with_feature_tensor"][j] = data_dict["current_point_with_feature_tensor"][j].cuda()

                path_mos = os.path.join("preb_out",id,"mos_preb","sequences",str(seq).zfill(2),"predictions")
                path_mos_confidence = os.path.join("preb_out",id,"confidence","sequences",str(seq).zfill(2),"predictions")
                path_bbox = os.path.join("preb_out",id,"bbox_preb","sequences",str(seq).zfill(2),"predictions")
                path_semantic = os.path.join("preb_out",id,"semantic_preb","sequences",str(seq).zfill(2),"predictions")
                path_semantic_all = os.path.join("preb_out",id,"multi_semantic_preb","sequences",str(seq).zfill(2),"predictions")
        
                os.makedirs(path_mos,exist_ok=True)
                os.makedirs(path_mos_confidence,exist_ok=True)
                os.makedirs(path_bbox,exist_ok=True)
                os.makedirs(path_semantic,exist_ok=True)
                os.makedirs(path_semantic_all,exist_ok=True)


                data_dict = voxel_generate(data_dict)
                batch_mos_feature, batch_semantic_feature, batch_semantic_feature_all,preb_dict_list,recall_dict = model.forward(data_dict,Model_mode)


       
                for idx in range(data_dict["batch_size"]):
                    batch_mask = (data_dict['voxel_coords'][:,0]==idx)
                    point_mask = data_dict['range_mask'][idx]
                    mos_feature = gather_features_by_pc_voxel_id(batch_mos_feature[batch_mask], data_dict['pc_voxel_id'][idx])
                    semantic_feature = gather_features_by_pc_voxel_id(batch_semantic_feature[batch_mask],data_dict['pc_voxel_id'][idx])
                    semantic_feature_all = gather_features_by_pc_voxel_id(batch_semantic_feature_all[batch_mask],data_dict['pc_voxel_id'][idx])
  
                    file_name_mos = os.path.join(path_mos,str(path_list[idx]).zfill(6)+".label")
                    file_name_moving = os.path.join(path_mos_confidence,str(path_list[idx]).zfill(6)+".npy")
                    file_name_bbox = os.path.join(path_bbox,str(path_list[idx]).zfill(6)+".npy")
                    
                    mos_feature[:, ignore_index] = -float("inf")
                    pred_softmax = F.softmax(mos_feature, dim=1)
                    #=========================================
                    pred_softmax_cpu = pred_softmax.detach().cpu().numpy()
                    moving_confidence = pred_softmax_cpu[:, 1:]
                    moving_confidence_label = np.zeros([point_mask.shape[0],2])
                    moving_confidence_label[point_mask] = moving_confidence
                    np.save(file_name_moving, moving_confidence_label)
                    #=======================================
                    pred_labels = torch.argmax(pred_softmax, axis=1).long() 
                    preb_mos_label = pred_labels.cpu().numpy()
                    full_mos_label = np.zeros(point_mask.shape[0],np.int32)
                    full_mos_label[point_mask] = preb_mos_label
                    full_mos_label = to_original_labels(full_mos_label, semantic_mos_config)
                    full_mos_label = full_mos_label.reshape((-1)).astype(np.int32)

                    full_mos_label.tofile(file_name_mos)


                    preb_dict = preb_dict_list[idx]
                    for key in preb_dict:
                        preb_dict[key] = preb_dict[key].cpu().numpy()

                        np.save(file_name_bbox,preb_dict)
                    #semantic output 
                    file_name_semantic = os.path.join(path_semantic,str(path_list[idx]).zfill(6)+".label")
                    semantic_feature[:, ignore_index] = -float("inf")
                    pred_semantic_softmax = F.softmax(semantic_feature, dim=1)
                    pred_semantic_labels = torch.argmax(pred_semantic_softmax, axis=1).long() 
                    preb_semantic_label = pred_semantic_labels.cpu().numpy()
                    full_semantic_label = np.zeros(point_mask.shape[0],np.int32)
                    full_semantic_label[point_mask] = preb_semantic_label
                    full_semantic_label = to_original_labels(full_semantic_label, semantic_config)
                    full_semantic_label = full_semantic_label.reshape((-1)).astype(np.int32)
                    full_semantic_label.tofile(file_name_semantic)

                    #multi_semantic output
                    file_name_semantic_all = os.path.join(path_semantic_all,str(path_list[idx]).zfill(6)+".label")
                    semantic_feature_all[:, ignore_index] = -float("inf")
                    pred_semantic_all_softmax = F.softmax(semantic_feature_all, dim=1)
                    pred_semantic_labels_all = torch.argmax(pred_semantic_all_softmax, axis=1).long() 
                    preb_semantic_label_all = pred_semantic_labels_all.cpu().numpy()
                    full_semantic_label_all = np.zeros(point_mask.shape[0],np.int32)
                    full_semantic_label_all[point_mask] = preb_semantic_label_all
                    full_semantic_label_all = to_original_labels(full_semantic_label_all, semantic_config_all)
                    full_semantic_label_all = full_semantic_label_all.reshape((-1)).astype(np.int32)
                    full_semantic_label_all.tofile(file_name_semantic_all)
                torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
