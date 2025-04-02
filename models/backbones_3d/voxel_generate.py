#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel

class VoxelGenerate(nn.Module):
    def __init__(self,voxel_size,point_cloud_range,max_number_of_voxel,max_point_per_voxel,num_point_feature):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range=point_cloud_range
        self.max_number_of_voxel=max_number_of_voxel
        self.max_point_per_voxel=max_point_per_voxel
        self.num_point_feature=num_point_feature
    
    def forward(self,batch):
        voxels_list = []
        voxel_coords_list = []
        voxel_num_points_list = []
        pc_voxel_id_list = []
        for i in range(0,batch["batch_size"]):
            
            self.voxel_generator = PointToVoxel(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz = self.point_cloud_range,
                    num_point_features=self.num_point_feature, # feature dim
                    max_num_voxels = self.max_number_of_voxel,
                    max_num_points_per_voxel = self.max_point_per_voxel,
                    device=batch["current_point_with_feature_tensor"][i].device
                )
            voxel_output=self.voxel_generator.generate_voxel_with_id(batch["current_point_with_feature_tensor"][i])
            voxels,voxel_coords,voxel_num_points,pc_voxel_id = voxel_output

            voxels_list.append(voxels.detach().clone())
            voxel_num_points_list.append(voxel_num_points.detach().clone())
            coor_pad = torch.full((voxel_coords.shape[0],1),i,device=voxel_coords.device)
            coor_paded = torch.hstack([coor_pad,voxel_coords.detach().clone()])
            voxel_coords_list.append(coor_paded.detach().clone())
            pc_voxel_id_list.append(pc_voxel_id.detach().clone())

            batch["current_point_with_feature_tensor"][i] = batch["current_point_with_feature_tensor"][i].detach()
            voxels = voxels.detach()
            voxel_coords = voxel_coords.detach()
            voxel_num_points = voxel_num_points.detach()
            pc_voxel_id = pc_voxel_id.detach()
        
        batch['voxels'] = torch.cat(voxels_list,dim=0)
        batch['voxel_num_points'] = torch.cat(voxel_num_points_list,dim=0)
        batch['voxel_coords'] = torch.cat(voxel_coords_list,dim=0)
        batch['pc_voxel_id'] = pc_voxel_id_list
        batch['xyz'] = batch['voxels'][:,0,:3].clone()
        batch['batch'] = batch['voxel_coords'][:,0].clone()

        return batch


