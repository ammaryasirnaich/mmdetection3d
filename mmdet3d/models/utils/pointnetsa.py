import warnings
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple
from ..builder import BACKBONES
from ...utils import get_root_logger
from mmcv.runner import BaseModule
from mmcv.ops.group_points import *
from mmcv.ops.furthest_point_sample import *
from mmcv.ops.gather_points import *


from typing import List, Tuple

from mmcv.runner import auto_fp16
from torch import nn as nn
from ..builder import BACKBONES
from mmdet3d.ops.pointnet_modules.point_sa_module import *



class VoxelPoinetEmbedding(BasePointSAModule):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with no sampler method like FPS. 
    Afer using the input from voxelization using mean of point clouds as 
    reference point cloud with raduii to gather points around the reference 

    Args:
    mlp_channels (list[int]): Specify of the pointnet before
        the global pooling for each scale.
    num_point (int, optional): Number of points.
        Default: None.
    radius (float, optional): Radius to group with.
        Default: None.
    num_sample (int, optional): Number of samples in each ball query.
        Default: None.
    norm_cfg (dict, optional): Type of normalization method.
        Default: dict(type='BN2d').
    use_xyz (bool, optional): Whether to use xyz.
        Default: True.
    pool_mod (str, optional): Type of pooling method.
        Default: 'max_pool'.
    normalize_xyz (bool, optional): Whether to normalize local XYZ
        with radius. Default: False.
  
'''
    
    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',  
                 normalize_xyz=False):
        super().__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)
        
    


    def forward(self, points_xyz, features=None, indices=None, focal_point=None):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) features of each point.
                Default: None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Default: None.
            focal_point (Tensor, optional): (B, M, 3) new coords of after downsampling 
            using mean function on voxel points.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []
    
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, focal_point, features)

            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return focal_point, torch.cat(new_features_list, dim=1)
 
       




# class SimplePointNetSASSG(BasePointNet):
#     """PointNet with Single-scale grouping.

#     Args:
#         in_channels (int): Input channels of point cloud.
#         num_points (tuple[int]): The number of points which each SA
#             module samples.
#         radius (tuple[float]): Sampling radii of each SA module.
#         num_samples (tuple[int]): The number of samples for ball
#             query in each SA module.
#         sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
#         fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
#         norm_cfg (dict): Config of normalization layer.
#         sa_cfg (dict): Config of set abstraction module, which may contain
#             the following keys and values:

#             - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
#             - use_xyz (bool): Whether to use xyz as a part of features.
#             - normalize_xyz (bool): Whether to normalize xyz with radii in
#               each SA module.
#     """

#     def __init__(self,
#                  in_channels,
#                  num_points=(2048, 1024, 512, 256),
#                  radius=(0.2, 0.4, 0.8, 1.2),
#                  num_samples=(64, 32, 16, 16),
#                  sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
#                               (128, 128, 256)),
#                  fp_channels=((256, 256), (256, 256)),
#                  norm_cfg=dict(type='BN2d'),
#                  sa_cfg=dict(
#                      type='PointSAModule',
#                      pool_mod='max',
#                      use_xyz=True,
#                      normalize_xyz=True),
#                  init_cfg=None):
#         super().__init__(init_cfg=init_cfg)
#         self.num_sa = len(sa_channels)
#         self.num_fp = len(fp_channels)

#         assert len(num_points) == len(radius) == len(num_samples) == len(
#             sa_channels)
#         assert len(sa_channels) >= len(fp_channels)

#         self.SA_modules = nn.ModuleList()
#         sa_in_channel = in_channels - 3  # number of channels without xyz
#         skip_channel_list = [sa_in_channel]

#         for sa_index in range(self.num_sa):
#             cur_sa_mlps = list(sa_channels[sa_index])
#             cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
#             sa_out_channel = cur_sa_mlps[-1]

#             self.SA_modules.append(
#                 build_sa_module(
#                     num_point=num_points[sa_index],
#                     radius=radius[sa_index],
#                     num_sample=num_samples[sa_index],
#                     mlp_channels=cur_sa_mlps,
#                     norm_cfg=norm_cfg,
#                     cfg=sa_cfg))
#             skip_channel_list.append(sa_out_channel)
#             sa_in_channel = sa_out_channel

#         self.FP_modules = nn.ModuleList()

#         fp_source_channel = skip_channel_list.pop()
#         fp_target_channel = skip_channel_list.pop()
#         for fp_index in range(len(fp_channels)):
#             cur_fp_mlps = list(fp_channels[fp_index])
#             cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
#             self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
#             if fp_index != len(fp_channels) - 1:
#                 fp_source_channel = cur_fp_mlps[-1]
#                 fp_target_channel = skip_channel_list.pop()

#     @auto_fp16(apply_to=('points', ))
#     def forward(self, points,vox_cood):
#         """Forward pass.

#         Args:
#             points (torch.Tensor): point coordinates with features,
#                 with shape (B, N, 3 + input_feature_dim).

#         Returns:
#                 xyz, FeaturesxD  
#         """
#         # xyz, features = self._split_point_feats(points)
#         xyz =  points[:,1:4]
#         features = points[:,:4]

#         batch, num_points = xyz.shape[:2]
#         indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
#             batch, 1).long()

#         sa_xyz = [xyz]
#         sa_features = [features]
#         sa_indices = [indices]

#         for i in range(self.num_sa):
#             cur_xyz, cur_features, cur_indices = self.SA_modules[i](
#                 sa_xyz[i], sa_features[i])
#             sa_xyz.append(cur_xyz)
#             sa_features.append(cur_features)
#             sa_indices.append(
#                 torch.gather(sa_indices[-1], 1, cur_indices.long()))

#         fp_xyz = [sa_xyz[-1]]
#         fp_features = [sa_features[-1]]
#         fp_indices = [sa_indices[-1]]

#         for i in range(self.num_fp):
#             fp_features.append(self.FP_modules[i](
#                 sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
#                 sa_features[self.num_sa - i - 1], fp_features[-1]))
#             fp_xyz.append(sa_xyz[self.num_sa - i - 1])
#             fp_indices.append(sa_indices[self.num_sa - i - 1])

#         ret = dict(
#             fp_xyz=fp_xyz,
#             fp_features=fp_features,
#             fp_indices=fp_indices,
#             sa_xyz=sa_xyz,
#             sa_features=sa_features,
#             sa_indices=sa_indices)
#         return ret
