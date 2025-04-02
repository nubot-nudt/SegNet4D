#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# our instance aware backbone build upon the UNet-based Sparse Convolution library.

import functools
import torch
import torch.nn as nn
import time
import os
from typing import Set
import numpy as np

try:
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo
except:
    import spconv as spconv



from ..backbones_2d.height_compression import HeightCompression
from ..backbones_2d.base_bev_backbone import BaseBEVBackbone
from ..backbones_2d.center_head import CenterHead

from ..post_process import post_processing
from ..utils import Array_Index
from torch_scatter import scatter_mean

from .spherical_attention import SphereAttention

from .MSFM import MSF_Module

def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

class ResidualBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output



def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next

class Instance_Aware_Backbone(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,mos_class,semantic_class,multi_semantic_class, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.blocks1 = spconv.SparseSequential(
            ResidualBlock(32,32,norm_fn,indice_key='subm1'),
            ResidualBlock(32,32,norm_fn,indice_key='subm1')
        )
        window_size = np.array([0.3,0.3,0.3])
        quant_size_scale = 24
        window_size_sphere = np.array([2,2,80])
        self.atten_block1 = SphereAttention(
            dim=32,
            num_heads=2,    
            window_size = window_size,
            window_size_sphere = window_size_sphere,
            quant_size = window_size/quant_size_scale,
            quant_size_sphere = window_size_sphere/quant_size_scale,
            indice_key = 'sptr_1',
            rel_query=True,
            rel_key = True,
            rel_value = True,
            drop_path=0,
        )

        self.conv1 = spconv.SparseSequential(
                norm_fn(32),
                nn.ReLU(),
                spconv.SparseConv3d(32, 64, kernel_size=2, stride=2, bias=False, indice_key='spconv2', algo=ConvAlgo.Native)
        )
        self.blocks2 = spconv.SparseSequential(
            ResidualBlock(64,64,norm_fn,indice_key='subm2'),
            ResidualBlock(64,64,norm_fn,indice_key='subm2')
        )
        window_size = np.array([0.6,0.6,0.6])
        window_size_sphere = np.array([3,3,80])
        self.atten_block2 = SphereAttention(
            dim=64,
            num_heads=4,    
            window_size = window_size,
            window_size_sphere = window_size_sphere,
            quant_size = window_size/quant_size_scale,
            quant_size_sphere = window_size_sphere/quant_size_scale,
            indice_key = 'sptr_2',
            rel_query=True,
            rel_key = True,
            rel_value = True,
            drop_path=0.05,
        )

        self.conv2 = spconv.SparseSequential(
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseConv3d(64, 128, kernel_size=2, stride=2, bias=False, indice_key='spconv3', algo=ConvAlgo.Native)
        )

        self.blocks3 = spconv.SparseSequential(
            ResidualBlock(128,128,norm_fn,indice_key='subm3'),
            ResidualBlock(128,128,norm_fn,indice_key='subm3')
        )
        window_size = np.array([1.2,1.2,1.2])
        window_size_sphere = np.array([4.5,4.5,80])
        self.atten_block3 = SphereAttention(
            dim=128,
            num_heads=8,
            window_size = window_size,
            window_size_sphere = window_size_sphere,
            quant_size = window_size/quant_size_scale,
            quant_size_sphere = window_size_sphere/quant_size_scale,
            indice_key = 'sptr_3',
            rel_query=True,
            rel_key = True,
            rel_value = True,
            drop_path=0.1,
        )

        self.conv3 = spconv.SparseSequential(
                norm_fn(128),
                nn.ReLU(),
                spconv.SparseConv3d(128, 256, kernel_size=2, stride=2, bias=False, indice_key='spconv4', algo=ConvAlgo.Native)
        )
        self.blocks4 = spconv.SparseSequential(
            ResidualBlock(256,256,norm_fn,indice_key='subm4'),
            ResidualBlock(256,256,norm_fn,indice_key='subm4')
        )
        window_size = np.array([2.4,2.4,2.4])
        window_size_sphere = np.array([6.75,6.75,80])
        self.atten_block4 = SphereAttention(
            dim=256,
            num_heads=16,
            window_size = window_size,
            window_size_sphere = window_size_sphere,
            quant_size = window_size/quant_size_scale,
            quant_size_sphere = window_size_sphere/quant_size_scale,
            indice_key = 'sptr_4',
            rel_query=True,
            rel_key = True,
            rel_value = True,
            drop_path=0.15,
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)
            self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(256, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down1'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None

        self.post_process = model_cfg["MODEL"]["POST_PROCESSING"]
        self.num_class = model_cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"]
        self.point_cloud_range = np.array(model_cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        self.voxel_size = model_cfg["DATA"]['VOXEL_SIZE']  
        self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size=np.round(self.grid_size).astype(np.int64)
        self.to_bev = HeightCompression(model_cfg["MODEL"]["MAP_TO_BEV"])  
        self.bev_backbone = BaseBEVBackbone(model_cfg["MODEL"]["BACKBONE_2D"],input_channels=self.to_bev.num_bev_features)
        self.center_head = CenterHead(model_cfg["MODEL"]["DENSE_HEAD"],
                                        input_channels=model_cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"][0],
                                        num_class=model_cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"] if not model_cfg["MODEL"]["DENSE_HEAD"]["CLASS_AGNOSTIC"] else 1,
                                        class_names=model_cfg["MODEL"]["DENSE_HEAD"]["CLASE_NAME"],
                                        grid_size = self.grid_size,
                                        point_cloud_range=self.point_cloud_range,
                                        predict_boxes_when_training=model_cfg.get('ROI_HEAD', False))

        self.conv4 = spconv.SparseSequential(
                norm_fn(256),
                nn.ReLU(),
                spconv.SparseConv3d(256, 256, kernel_size=2, stride=2, bias=False, indice_key='spconv5', algo=ConvAlgo.Native)
        )
        self.blocks5 = spconv.SparseSequential(
            ResidualBlock(256,256,norm_fn,indice_key='subm5'),
            ResidualBlock(256,256,norm_fn,indice_key='subm5')
        )
        window_size = np.array([4.8,4.8,4.8])
        window_size_sphere = np.array([10.125,10.125,80])
        self.atten_block5 = SphereAttention(
            dim=256,
            num_heads=16,
            window_size = window_size,
            window_size_sphere = window_size_sphere,
            quant_size = window_size/quant_size_scale,
            quant_size_sphere = window_size_sphere/quant_size_scale,
            indice_key = 'sptr_5',
            rel_query=True,
            rel_key = True,
            rel_value = True,
            drop_path=0.2,
        )

        self.conv_up_instance_block= spconv.SubMConv3d(256+self.num_class, 256, 3, padding=1, indice_key='subm4')

        # ---------invert conv
        self.deconv4 = spconv.SparseSequential(
                norm_fn(256),
                nn.ReLU(),
                spconv.SparseInverseConv3d(256, 256, kernel_size=2, bias=False, indice_key='spconv5', algo=ConvAlgo.Native)
        )
        self.blocks_tail4 = spconv.SparseSequential(
            ResidualBlock(512,256,norm_fn,indice_key='subm5'),
            ResidualBlock(256,256,norm_fn,indice_key='subm5')
        )
        self.deconv3 = spconv.SparseSequential(
                norm_fn(256),
                nn.ReLU(),
                spconv.SparseInverseConv3d(256, 128, kernel_size=2, bias=False, indice_key='spconv4', algo=ConvAlgo.Native)
        )
        self.blocks_tail3 = spconv.SparseSequential(
            ResidualBlock(256,128,norm_fn,indice_key='subm4'),
            ResidualBlock(128,128,norm_fn,indice_key='subm4')
        )
        self.deconv2 = spconv.SparseSequential(
                norm_fn(128),
                nn.ReLU(),
                spconv.SparseInverseConv3d(128, 64, kernel_size=2, bias=False, indice_key='spconv3', algo=ConvAlgo.Native)
        )
        self.blocks_tail2 = spconv.SparseSequential(
            ResidualBlock(128,64,norm_fn,indice_key='subm3'),
            ResidualBlock(64,64,norm_fn,indice_key='subm3')
        )
        self.deconv1 = spconv.SparseSequential(
                norm_fn(64),
                nn.ReLU(),
                spconv.SparseInverseConv3d(64, 32, kernel_size=2, bias=False, indice_key='spconv2', algo=ConvAlgo.Native)
        )
        self.blocks_tail1 = spconv.SparseSequential(
            ResidualBlock(64,32,norm_fn,indice_key='subm2'),
            ResidualBlock(32,32,norm_fn,indice_key='subm2')
        )        
        self.sem_head = spconv.SparseSequential(
            norm_fn(32),
            nn.ReLU()
        )
        self.sem_linear = nn.Linear(32, semantic_class) # bias(default): True

        self.mos_head = spconv.SparseSequential(
            norm_fn(32),
            nn.GELU(),
            spconv.SubMConv3d(32, 16, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.GELU()
        )
        self.mos_linear = nn.Linear(16, mos_class)

        self.ms_fusion = MSF_Module(32,16,multi_semantic_class)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key,algo=ConvAlgo.Native),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False, algo=ConvAlgo.Native),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    @staticmethod
    def replace_feature_indice(self,out,new_features):
        sparse_feature = torch.tensor(out.features,device=out.features.device)
        for i in range(0,out.indices.shape[0]):
            sparse_feature[i,:] = out.features[i,:]+new_features.features[out.indices[i,1]*125*150+out.indices[i,1]*150+out.indices[i,3],:]
        return sparse_feature
    
    def forward(self, batch_dict,Model_mode):
        """
        Args:
            Model_mode: train || eval || test
            batch_dict:
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        """
        
        # -----------------batch_dict['voxel_features'] include original features and motion features of LiDAR points------------------------
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        # batch_size=  1 # the batch_size just for voxel processing
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size']
        )

        x = self.input_conv(input_sp_tensor)
        x_conv1 = self.blocks1(x)
        x_conv1 = x_conv1.replace_feature(self.atten_block1(x_conv1.features,batch_dict['xyz'],batch_dict['batch'])) 

        x_conv2 = self.conv1(x_conv1)
        x_conv2 = self.blocks2(x_conv2)
        indice_pairs = x_conv2.indice_dict['spconv2'].indice_pairs
        batch_dict['xyz'],batch_dict['batch'] = get_downsample_info(batch_dict['xyz'],batch_dict['batch'],indice_pairs)
        x_conv2 = x_conv2.replace_feature(self.atten_block2(x_conv2.features,batch_dict['xyz'],batch_dict['batch']))

        x_conv3 = self.conv2(x_conv2)
        x_conv3 = self.blocks3(x_conv3)
        indice_pairs = x_conv3.indice_dict['spconv3'].indice_pairs
        batch_dict['xyz'],batch_dict['batch'] = get_downsample_info(batch_dict['xyz'],batch_dict['batch'],indice_pairs)
        x_conv3 = x_conv3.replace_feature(self.atten_block3(x_conv3.features,batch_dict['xyz'],batch_dict['batch']))

        x_conv4 = self.conv3(x_conv3)
        x_conv4 = self.blocks4(x_conv4)
        indice_pairs = x_conv4.indice_dict['spconv4'].indice_pairs
        batch_dict['xyz'],batch_dict['batch'] = get_downsample_info(batch_dict['xyz'],batch_dict['batch'],indice_pairs)
        x_conv4 = x_conv4.replace_feature(self.atten_block4(x_conv4.features,batch_dict['xyz'],batch_dict['batch']))
        
        # --------------detecting instance------------------
        if self.conv_out is not None:
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8
        
        batch_dict = self.to_bev(batch_dict) 
        batch_dict = self.bev_backbone(batch_dict)
        batch_dict = self.center_head(batch_dict,Model_mode)

        # --------------nms... for refining the bounding box ----------------
        pred_dicts, recall_dicts = post_processing(batch_dict,self.post_process,self.num_class)

        
        x_conv5 = self.conv4(x_conv4)
        x_conv5 = self.blocks5(x_conv5)
        indice_pairs = x_conv5.indice_dict['spconv5'].indice_pairs
        batch_dict['xyz'],batch_dict['batch'] = get_downsample_info(batch_dict['xyz'],batch_dict['batch'],indice_pairs)
        x_conv5 = x_conv5.replace_feature(self.atten_block5(x_conv5.features,batch_dict['xyz'],batch_dict['batch']))

        x_output = self.deconv4(x_conv5)
        x_conv4 = x_conv4.replace_feature(torch.cat((x_conv4.features, x_output.features), dim=1))
        x_output = self.blocks_tail4(x_conv4)

        sparse_inv_bev_features = []
        sparse_inv_bev_coord = x_output.indices.detach().cpu().numpy()[:,[0,3,2,1]]

        ## upsample fusion: you also can integrate multi-scale instance information, more details can be found in the our conference paper, i.e., InsMOS.
        ## For efficiency, we only integrate the instance information on the min-scale features layer.
        for batch_i in range(x_output.batch_size):
            boxes = pred_dicts[batch_i]["pred_boxes"].clone().detach()
            boxes_label = pred_dicts[batch_i]["pred_labels"].clone().detach()
            boxes_label = boxes_label.view(-1,1)
            boxes[:,0] = (boxes[:,0] -self.point_cloud_range[0])/self.voxel_size[0]/batch_dict['encoded_spconv_tensor_stride']
            boxes[:,1] = (boxes[:,1] -self.point_cloud_range[1])/self.voxel_size[1]/batch_dict['encoded_spconv_tensor_stride']
            boxes[:,2] = (boxes[:,2] -self.point_cloud_range[2])/self.voxel_size[2]/batch_dict['encoded_spconv_tensor_stride']
            boxes[:,3] = boxes[:,3]/self.voxel_size[0]/batch_dict['encoded_spconv_tensor_stride']
            boxes[:,4] = boxes[:,4]/self.voxel_size[1]/batch_dict['encoded_spconv_tensor_stride']
            boxes[:,5] = boxes[:,5]/self.voxel_size[2]/batch_dict['encoded_spconv_tensor_stride']
            boxes = torch.hstack([boxes,boxes_label])
            sparse_inv_bev_coord_i = sparse_inv_bev_coord[sparse_inv_bev_coord[:,0]==batch_i]
            sparse_inv_bev_features_i = np.zeros((len(sparse_inv_bev_coord_i),self.num_class),dtype=int)
            features_instance_i = Array_Index.find_features_by_bbox_with_yaw(sparse_inv_bev_coord_i,boxes.detach().cpu().numpy(),sparse_inv_bev_features_i)
            sparse_inv_bev_features.append(features_instance_i)

        features_instance_tensor = torch.from_numpy(np.concatenate(sparse_inv_bev_features,axis=0))
        features_instance_tensor = features_instance_tensor.to(device=x_output.features.device)
        x_output = x_output.replace_feature(torch.cat([x_output.features,features_instance_tensor],dim=1)) 
        x_output=self.conv_up_instance_block(x_output)  

        x_output = self.deconv3(x_output)
        x_conv3 = x_conv3.replace_feature(torch.cat((x_conv3.features, x_output.features), dim=1))
        x_output = self.blocks_tail3(x_conv3)

        x_output = self.deconv2(x_output)
        x_conv2 = x_conv2.replace_feature(torch.cat((x_conv2.features, x_output.features), dim=1))
        x_output = self.blocks_tail2(x_conv2)

        x_output = self.deconv1(x_output)
        x_conv1 = x_conv1.replace_feature(torch.cat((x_conv1.features, x_output.features), dim=1))
        x_output = self.blocks_tail1(x_conv1)

        # two heads: MOS head and Semantic head
        x_output_mos = self.mos_head(x_output)
        mos_seg_preb = self.mos_linear(x_output_mos.features)

        x_output_semantic = self.sem_head(x_output)
        semantic_seg_preb = self.sem_linear(x_output_semantic.features)

        # fusion: MOS and Semantic predictions
        sem_all_seg_preb = self.ms_fusion(x_output_semantic,x_output_mos)
        
        if Model_mode =='train':
            return self.center_head.get_loss(),mos_seg_preb,semantic_seg_preb,sem_all_seg_preb
        else:
            return mos_seg_preb,semantic_seg_preb,sem_all_seg_preb,pred_dicts, recall_dicts
