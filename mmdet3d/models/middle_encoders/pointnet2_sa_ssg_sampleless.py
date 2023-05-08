# Copyright (c) OpenMMLab. All rights reserved.
import torch

from torch import nn as nn
# from .base_pointnet import BasePointNet
from mmdet3d.models.backbones import base_pointnet
from mmdet3d.models.layers import PointFPModule, build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.models.backbones.base_pointnet import BasePointNet


@MODELS.register_module()
class PointNet2SASSG_SL(BasePointNet):
    """PointNet2 with Single-scale grouping without sampling methods.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(64, 32, 16, 16),
                 sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                              (128, 128, 256)),
                 fp_channels=((256, 256), (256, 256)),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointRefereceSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            skip_channel_list.append(sa_out_channel)
            sa_in_channel = sa_out_channel

        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    def forward(self, voxels, mean_point_xyz):  #points,mean_point_xyz
        """Forward pass.

        Args:
               - voxels : The coordinates of each point clouds inside the voxel. (V,P,D)
               - mean_point_xyz :  Mean point of Voxel Point clouds. (B,V,3)


        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of
                    each fp features.
                - fp_features (list[torch.Tensor]): The features
                    from each Feature Propagate Layers.
                - voxel_feature(torch.Tensor): The voxel-wise feature  concatination of (fp_xyz, fp_features) (B,V,P,D)
                
        """

        # print("tyep mean_point_xyz", mean_point_xyz.shape)
        # print("shape of voxel tensor", voxels.shape)
        # print("shape of mean_point_xyz tensor", mean_point_xyz.size)
        
        v,p,d = voxels.shape
        # print("origional voxel shape" ,voxels.shape)
        batch_szie = mean_point_xyz.shape[0]
        point_xyz = voxels.view(v*p,d).expand(batch_szie,-1,-1)  # B,V*P,D  #reshape to get total number of points from all voxels

        # .view(-1, v,p,d)
        # print("point_xyz shape after unsqueeze", point_xyz.shape)

        if(mean_point_xyz.dtype == torch.float16):
            mean_point_xyz = mean_point_xyz.type(torch.float32)

        xyz, features = self._split_point_feats(point_xyz)
        sa_xyz = [xyz]
        sa_features = [features]
        # sa_indices = [indices]
        for i in range(self.num_sa):
    
            ## the first SA module takes the voxel mean coordinates as a target point  
            cur_xyz, cur_features = self.SA_modules[i] ( 
                            sa_xyz[i], sa_features[i], target_xyz=mean_point_xyz )
        
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
           
        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        # fp_indices = [sa_indices[-1]]       

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            # fp_indices.append(sa_indices[self.num_sa - i - 1])

               # print("fp_features shape:",fp_features[-1].shape)

        fp_features[-1] = fp_features[-1].permute(0,2,1)
        voxel_feature = torch.cat((fp_xyz[-1],fp_features[-1]),dim=2).view(batch_szie,v,p,-1)
        # print("fp_xyz[-1] ", fp_xyz[-1].shape)
        # print("fp_features[-1] ", fp_features[-1].shape)
        # print("voxel_feature ", voxel_feature.shape)

        # voxel_feature = voxel_feature.expand(batch_szie,-1,-1,-1)
        # print("voxel_feature ", voxel_feature.shape)
        # print(type(voxel_feature))

        ret = dict(
            # fp_xyz=fp_xyz,
            # fp_features=fp_features,
            # fp_indices=fp_indices,
            # sa_xyz=sa_xyz,
            # sa_features=sa_features,
            # sa_indices=sa_indices
            voxels = voxel_feature
            )
        return ret



