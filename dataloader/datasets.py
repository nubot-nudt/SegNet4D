#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# @file      datasets.py
# @author    Neng Wang 

import numpy as np
import yaml
import os
import copy
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .src import Array_Index

from .utils import load_poses, load_calib, load_files, mask_points_by_range
from .augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud, 
    random_rotation,
    random_scaling,
    random_flip,
    random_shift,
    random_jitter,
    random_elastic_aug
)
from collections import defaultdict

class KittiSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data"""

    def __init__(self, cfg):
        """Method to initizalize the KITTI dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(KittiSequentialModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""

        ########## Point dataset splits
        train_set = KittiSequentialDataset(self.cfg, split="train")

        val_set = KittiSequentialDataset(self.cfg, split="val")

        test_set = KittiSequentialDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_batch,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_batch,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_batch,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):
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

class KittiSequentialDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("DATA")

    
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATA"]["POSES"]

        # Semantic information
        self.semantic_mos_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_MOS_CONFIG_FILE"]))
        self.semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE_ALL"]))

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
  
        self.grid_size = self.cfg["DATA"]["VOXEL_SIZE"]
        self.grid_size_bev = self.cfg["DATA"]["GRID_SIZE_BEV"]
        self.online_train = self.cfg["DATA"]["ONLINE_TRAIN"]

        self.rotation = list(self.cfg["DATA_AUGMENTOR"]["WORLD_ROT_ANGLE"])
        self.scale = list(self.cfg["DATA_AUGMENTOR"]["WORLD_SCALE_RANGE"])

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)


        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        idx_valid = 0

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

            if self.online_train==False:
                for residual_idx in range(self.n_past_steps-1):
                    folder_name = "residual_bev_images_" + str(residual_idx+1)
                    exec("residual_path_" + str(residual_idx+1) + "=" + "os.path.join(path_to_seq, folder_name)")
                    exec("residual_files_" + str(residual_idx+1) + " = " + '[os.path.join(dp, f) for dp, dn, fn in '
                             'os.walk(os.path.expanduser(residual_path_' + str(residual_idx+1) + '))'
                             ' for f in fn]')
                    exec("residual_files_" + str(residual_idx+1) + ".sort()")
                    exec("self.residual_files_" + str(residual_idx+1) + "[seq]" + " = " + "residual_files_" + str(residual_idx+1))


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
        
        # load bounding box 
        bounding_box_file = [ os.path.join(self.root_dir, str(seq).zfill(2), "boundingbox_label_lshape", str(i).zfill(6) + ".npy")
            for i in past_indices ]

        data_dict = {}

        if self.online_train:
            list_bounding_box = [self.read_bounding_box_label(bounding_box_file[-1])] 
            gt_box = np.zeros([len(list_bounding_box[0]),8])
            for i, boxs in enumerate(list_bounding_box):
                gt_box[:len(boxs),0:7] = boxs[:,2:9]
                gt_box[:len(boxs),7] = boxs[:,0]

            list_past_point_clouds_raw = [self.read_point_cloud(f) for f in past_files]

            num_point_list=[]
            for i,pcd in enumerate(list_past_point_clouds_raw):
                if self.transform:
                    from_pose = self.poses[seq][past_indices[i]]
                    to_pose = self.poses[seq][past_indices[-1]]
                    pcd[:,:3] = self.transform_point_cloud(pcd[:,:3], from_pose, to_pose)
                    num_point_list.append(pcd.shape[0])
                list_past_point_clouds_raw[i] = pcd
            
   
            if self.training:
                gt_box_for_augment = gt_box[:,0:7]
                past_points_for_transform=np.concatenate(list_past_point_clouds_raw,axis=0)
                
                past_points_for_transform,gt_box_for_augment=random_flip(past_points_for_transform,gt_box_for_augment)
                past_points_for_transform,gt_box_for_augment=random_rotation(past_points_for_transform,gt_box_for_augment,self.rotation)
                past_points_for_transform,gt_box_for_augment=random_scaling(past_points_for_transform,gt_box_for_augment,self.scale)
                past_points_for_transform,gt_box_for_augment=random_shift(past_points_for_transform,gt_box_for_augment)

                for i in range(len(num_point_list)):
                    if i==0:
                        list_past_point_clouds_raw[i]=past_points_for_transform[0:num_point_list[0],:]
                    else:
                        list_past_point_clouds_raw[i]=past_points_for_transform[np.sum(num_point_list[0:i]):np.sum(num_point_list[0:i+1]),:]
                gt_box[:,0:7] = gt_box_for_augment

            # Load past labels
            label_files = [
                os.path.join(self.root_dir, str(seq).zfill(2), "labels", str(i).zfill(6) + ".label")
                for i in past_indices
            ]

            past_labels = [self.read_mos_labels(f) for f in label_files]
            semantic_labels = [self.read_semantic_labels(f) for f in label_files]

            list_past_point_clouds = []
            for i in range(0,len(list_past_point_clouds_raw)):
                data_point = np.hstack([list_past_point_clouds_raw[i],past_labels[i],semantic_labels[i]])
                point_mask = mask_points_by_range(data_point,self.point_cloud_range)
                data_point = data_point[point_mask]
                list_past_point_clouds.append(data_point[:,:4]) 
                past_labels[i] = torch.tensor(data_point[:,-2],dtype=torch.float32)
                semantic_labels[i] = torch.tensor(data_point[:,-1],dtype=torch.float32)

            #convert bev
            motion_feature = self.convert_pointclou2bev(list_past_point_clouds)
            current_point_with_feature = np.hstack([list_past_point_clouds[-1],motion_feature])
            current_point_with_feature_tensor = torch.tensor(current_point_with_feature,dtype=torch.float32)

            current_mos_label = past_labels[-1]
            current_semantic_label_all = semantic_labels[-1]

            current_semantic_label = current_semantic_label_all.clone()
            for k, v in self.semantic_config["learning_moving_map_inv"].items():
                current_semantic_label[current_semantic_label == k] = v

            gt_boxes = torch.tensor(gt_box.astype(np.float32))
            meta = (seq, scan_idx, past_files)
            data_dict = {
                "meta":meta,   
                "current_point_with_feature_tensor":current_point_with_feature_tensor,
                "mos_labels":current_mos_label,               
                "semantic_labels":current_semantic_label,
                "semantic_labels_all":current_semantic_label_all,
                "gt_boxes":gt_boxes,
            }
        else:
            list_bounding_box = [self.read_bounding_box_label(bounding_box_file[-1])] 
            gt_box = np.zeros([len(list_bounding_box[0]),8])
            for i, boxs in enumerate(list_bounding_box):
                gt_box[:len(boxs),0:7] = boxs[:,2:9]
                gt_box[:len(boxs),7] = boxs[:,0]

            curren_point_cloud = self.read_point_cloud(past_files[-1])

            residual_bev_list = []
            for idx in range(self.n_past_steps-1):
                exec("residual_file_" + str(idx+1) + " = " + "self.residual_files_" + str(idx+1) + "[seq][scan_idx]")
                residual_bev_idx = np.load(eval("residual_file_" + str(idx+1)),allow_pickle=True)
                assert residual_bev_idx.shape[0]!=3
                residual_bev_list.append(residual_bev_idx)

            # Load past labels
            label_files = [
                os.path.join(self.root_dir, str(seq).zfill(2), "labels", str(i).zfill(6) + ".label")
                for i in past_indices
            ]
            mos_labels = self.read_mos_labels(label_files[-1])
            semantic_labels = self.read_semantic_labels(label_files[-1])

            data_point_first = np.hstack([curren_point_cloud,mos_labels,semantic_labels])
            point_mask_first = mask_points_by_range(data_point_first,self.point_cloud_range)
            data_point_ranged_first = data_point_first[point_mask_first]
            curren_point_cloud_range_first = data_point_ranged_first[:,:4]
            mos_label_first = data_point_ranged_first[:,-2].reshape(-1,1)
            semantic_label_first = data_point_ranged_first[:,-1].reshape(-1,1)

            motion_feature = self.encoding_motion_feature(curren_point_cloud_range_first,residual_bev_list)
            current_point_with_feature = np.hstack([curren_point_cloud_range_first,motion_feature])


            gt_box_for_augment = gt_box[:,0:7]
            past_points_for_transform = current_point_with_feature[:,:3]
            past_points_for_transform,gt_box_for_augment=random_flip(past_points_for_transform,gt_box_for_augment)
            past_points_for_transform,gt_box_for_augment=random_rotation(past_points_for_transform,gt_box_for_augment,self.rotation)
            past_points_for_transform,gt_box_for_augment=random_scaling(past_points_for_transform,gt_box_for_augment,self.scale)
            past_points_for_transform,gt_box_for_augment=random_shift(past_points_for_transform,gt_box_for_augment)

            gt_box[:,0:7] = gt_box_for_augment
            current_point_with_feature[:,:3] = past_points_for_transform
            
            data_point = np.hstack([current_point_with_feature,mos_label_first,semantic_label_first])
            point_mask = mask_points_by_range(data_point,self.point_cloud_range)
            data_point_second = data_point[point_mask]
            curren_point_cloud_ranged = data_point_second[:,:-2] 
            current_mos_label = torch.tensor(data_point_second[:,-2])
            current_semantic_label_all = torch.tensor(data_point_second[:,-1],dtype=torch.float32)
            
            current_semantic_label = current_semantic_label_all.clone()
            for k, v in self.semantic_config["learning_moving_map_inv"].items():
                current_semantic_label[current_semantic_label == k] = v

            current_point_with_feature_tensor = torch.tensor(curren_point_cloud_ranged,dtype=torch.float32,requires_grad=False)
            gt_boxes = torch.tensor(gt_box.astype(np.float32))

            meta = (seq, scan_idx, past_files)
            data_dict = {
                "meta":meta,   
                "current_point_with_feature_tensor":current_point_with_feature_tensor,
                "mos_labels":current_mos_label,              
                "semantic_labels":current_semantic_label,
                "semantic_labels_all":current_semantic_label_all,
                "gt_boxes":gt_boxes,
            }

        return data_dict

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = np.linalg.inv(to_pose) @ from_pose
        NP = past_point_clouds.shape[0]
        xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def augment_data(self, past_point_clouds):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))
        return point_cloud

    def read_mos_labels(self, filename):
        """Load moving object labels from .label file"""
        if os.path.isfile(filename):
            labels = np.fromfile(filename, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  
            mapped_labels = copy.deepcopy(labels)
            for k, v in self.semantic_mos_config["learning_map"].items():
                mapped_labels[labels == k] = v
            selected_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            selected_labels = selected_labels.reshape((-1, 1))
            return selected_labels
        else:
            return torch.Tensor(1, 1).long()

    def convert_pointclou2bev(self,list_pointcloud):
        current_point_cloud = list_pointcloud[-1]
        motion_feature = np.zeros([current_point_cloud.shape[0],self.n_past_steps-1],dtype=np.float16)
        bev_height = int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.grid_size_bev)
        bev_width = int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.grid_size_bev)
       
        bev_img_current = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
        bev_max_current = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
        bev_min_current= self.point_cloud_range[5]*np.ones([bev_height,bev_width],dtype=np.float32)

        current_bev = Array_Index.pointcloud2bevHeightSingleThread_1(current_point_cloud[:,:3],self.point_cloud_range,bev_img_current,bev_max_current,bev_min_current,self.grid_size_bev)
        
        current_x_index = ((-current_point_cloud[:,0] - self.point_cloud_range[0])/self.grid_size_bev).astype(np.int32)
        current_y_index = ((-current_point_cloud[:,1] - self.point_cloud_range[1])/self.grid_size_bev).astype(np.int32)
        current_x_index[current_x_index==bev_height] = bev_height-1
        current_y_index[current_y_index==bev_width] = bev_width-1
        for i in range(0,self.n_past_steps-1): 
            bev_img = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_max = self.point_cloud_range[2]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_min = self.point_cloud_range[5]*np.ones([bev_height,bev_width],dtype=np.float32)
            bev_frame = Array_Index.pointcloud2bevHeightSingleThread_1(list_pointcloud[self.n_past_steps-i-2][:,:3],self.point_cloud_range,bev_img,bev_max,bev_min,self.grid_size_bev)
            
            residual_bev = current_bev - bev_frame
            motion_feature[:,i] = residual_bev[current_x_index,current_y_index]
        return motion_feature
    
    def encoding_motion_feature(self,current_pointcloud,residual_bev_idx):
        current_point_cloud = current_pointcloud
        motion_feature = np.zeros([current_point_cloud.shape[0],self.n_past_steps-1],dtype=np.float16)
        bev_height = int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.grid_size_bev)
        bev_width = int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.grid_size_bev)


        current_x_index = ((-current_point_cloud[:,0] - self.point_cloud_range[0])/self.grid_size_bev).astype(np.int32)
        current_y_index = ((-current_point_cloud[:,1] - self.point_cloud_range[1])/self.grid_size_bev).astype(np.int32)
        current_x_index[current_x_index==bev_height] = bev_height-1
        current_y_index[current_y_index==bev_width] = bev_width-1


        for i in range(0,len(residual_bev_idx)):
            residual_idx = residual_bev_idx[i]
            residual_bev= np.zeros([bev_height,bev_width],dtype=np.float32)
            bev_img_index_xy = residual_idx[:,:2]
            bev_img_index_xy = bev_img_index_xy.astype(int)
            residual_bev[bev_img_index_xy[:,0],bev_img_index_xy[:,1]] = residual_idx[:,2]
            motion_feature[:,i] = residual_bev[current_x_index,current_y_index]
        
        return motion_feature


    def read_semantic_labels(self, filename):
        """Load moving object labels from .label file"""
        if os.path.isfile(filename):
            labels = np.fromfile(filename, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for k, v in self.semantic_config["learning_map"].items():
                mapped_labels[labels == k] = v
            selected_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            selected_labels = selected_labels.reshape((-1, 1))
            return selected_labels
        else:
            return torch.Tensor(1, 1).long()

    def read_bounding_box_label(self,filename):
        """Load object boundingbox  from .npy file"""
        boundingbox_label_load = np.load(filename,allow_pickle=True)
        if len(boundingbox_label_load)==0: 
            boundingbox_label_load = []
            boundingbox_label_load.append([0,0,1,[0,0,0,0,0,0,0]]) 
        dynamic_falg = False
        boundingbox_label_list = []
        for i in range(0,len(boundingbox_label_load)):
            boundingbox_label = np.zeros(9,dtype=np.float)
            boundingbox_label[0] = boundingbox_label_load[i][1]
            boundingbox_label[1] = boundingbox_label_load[i][2]
            boundingbox_label[2:9] = boundingbox_label_load[i][3][:]
            #合并标签
            if boundingbox_label[0]==1 or boundingbox_label[0] ==3 or boundingbox_label[0]== 6:
                boundingbox_label[0]=1
            elif boundingbox_label[0]==8: 
                boundingbox_label[0] =2
            elif boundingbox_label[0]==9 or boundingbox_label[0]==10: 
                boundingbox_label[0] =3
            else:
                boundingbox_label[0] = 0 
            boundingbox_label_list.append(boundingbox_label)

            if boundingbox_label[1] >0:
                dynamic_falg = True
            
        if dynamic_falg ==False: 
            boundingbox_label_list.append([0,1,0,0,0,0,0,0,0])

        box_label_numpy = np.array(boundingbox_label_list)
        return box_label_numpy

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
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
