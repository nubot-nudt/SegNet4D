#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.pytorch import ops

from typing import List


class SparseGlobalAvgPool(SparseModule):
    """
    """
    def __init__(self, is_mean: bool, name=None):
        super(SparseGlobalAvgPool, self).__init__(name=name)
        self.is_mean = is_mean

    def forward(self, input: spconv.SparseConvTensor):
        is_int8 = input.is_quantized
        assert not is_int8, "not implemented"
        assert isinstance(input, spconv.SparseConvTensor)
        out_indices, counts = ops.global_pool_rearrange(input.indices, input.batch_size)
        counts_cpu = counts.cpu()

        counts_cpu_np = counts_cpu.numpy()
        res_features_list: List[torch.Tensor] = []
        for i in range(input.batch_size):
            real_inds = out_indices[i, :counts_cpu_np[i]]
            real_features = input.features[real_inds.long()]
            if self.is_mean:
                real_features_reduced = torch.mean(real_features, dim=0)
            else:
                real_features_reduced = torch.max(real_features, dim=0)[0]
            res_features_list.append(real_features_reduced)
        res = torch.stack(res_features_list)
        return res 

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
    
class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out
    
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = SparseGlobalAvgPool(is_mean=True)
        self.max_pool = SparseGlobalAvgPool(is_mean=False)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out_put = spconv.SparseConvTensor(x.features, x.indices, x.spatial_shape, x.batch_size)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x).unsqueeze(dim=-1).unsqueeze(dim=-1))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x).unsqueeze(dim=-1).unsqueeze(dim=-1))))
        out = self.sigmoid(avg_out + max_out)
        out = out.squeeze(dim=-1).squeeze(dim=-1)
        out_multi = torch.zeros([x.features.shape[0],x.features.shape[1]],device=x.features.device)
        for i in range(x.batch_size):
            out_multi[x.indices[:,0]==i] = out[i]
        x = x.replace_feature(x.features*out_multi)
        return x
    

    
class fusion_module_MGA_3D(nn.Module):
    def __init__(self, channel_a, channel_m):
        super(fusion_module_MGA_3D, self).__init__()
        self.conv1x1_channel_wise = nn.Conv2d(channel_a, channel_a, 1, bias=True)
        self.conv1x1_spatial = spconv.SubMConv3d(channel_m, 1, 1, padding=1, bias=True, indice_key='subm1')
        self.avg_pool = SparseGlobalAvgPool(is_mean=True)

    def forward(self, img_feat, flow_feat):
        """
            flow_feat_map:  [N, 1]  
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = self.conv1x1_spatial(flow_feat)  # -> [bxn,1]
        flow_feat_map = flow_feat_map.replace_feature(nn.Sigmoid()(flow_feat_map.features))
        flow_feat_map = flow_feat_map.replace_feature(flow_feat_map.features * img_feat.features) # -> [bxn,channel_a]

        # channel-wise attention
        feat_vec = self.avg_pool(flow_feat_map) # -> [bx1,channel_a]
        feat_vec = feat_vec.unsqueeze(dim=-1).unsqueeze(dim=-1)  #-> [bx1,channel_a,1,1]
        feat_vec = self.conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        feat_vec = feat_vec.squeeze(dim=-1).squeeze(dim=-1)  #-> [bx1,channel_a]

        for i in range(flow_feat_map.batch_size):
            flow_feat_map.features[flow_feat_map.indices[:,0]==i] = flow_feat_map.features[flow_feat_map.indices[:,0]==i] * feat_vec[i]  # [bxn,channel_a] x [bx1,channel_a]

        flow_feat_map = flow_feat_map.replace_feature(flow_feat_map.features)

        img_feat = img_feat.replace_feature(flow_feat_map.features + img_feat.features)
        return img_feat


class MSF_Module(nn.Module):
    def __init__(self,channel_sem,channel_mos,sem_class_all):
        super(MSF_Module,self).__init__()
        self.channel_mos = channel_mos
        self.channel_sem = channel_sem
        self.fusion_module = fusion_module_MGA_3D(self.channel_sem,self.channel_mos)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.conv = spconv.SubMConv3d(channel_sem, 32, 3, padding=1, bias=True, indice_key='subm1')
        self.resblock = SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='subm1')
        self.channel_atten = ChannelAttention3D(32)
        self.resblock_2 = SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='subm1')
        self.output_layer = spconv.SparseSequential(norm_fn(32),nn.ReLU())
        self.linear = nn.Linear(32, sem_class_all)
        

    
    def forward(self,feat_sem,feat_mos):
        feat = self.fusion_module(feat_sem,feat_mos)
        feat = self.conv(feat)
        feat = self.resblock(feat)
        feat = self.channel_atten(feat)
        feat = self.resblock_2(feat)
        feat = self.output_layer(feat)
        output = self.linear(feat.features)
        return output


