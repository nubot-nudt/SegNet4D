from multiprocessing.util import spawnv_passfds
import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg["NUM_BEV_FEATURES"]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']

        spatial_features = encoded_spconv_tensor.dense()
        # print("spatial_features shape:",spatial_features.shape)
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W) 
        # print("spatial_features shape:",spatial_features.shape)
        batch_dict['spatial_features'] = spatial_features
        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        return batch_dict
