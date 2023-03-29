
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.models import build_backbone
from mmdet3d.models.builder import build_voxel_encoder, build_middle_encoder,  DETECTORS
from mmdet3d.core.voxel.voxel_generator import VoxelGenerator
from mmcv.ops import Voxelization
import mmdet3d.models.builder


def vfe_feature_encoder(): 
    ### configuration for VFE encoder 
    hardsimple_feature_net_cfg = dict(type='HardSimpleVFE')
    hardsimple_feature_net = build_voxel_encoder(hardsimple_feature_net_cfg)

    point_xyz = torch.rand([97297, 20, 4])
    num_voxels = torch.randint(1, 100, [97297])
    voxel_coord = torch.randint(0, 100, [97297, 3])
    mean_point_xyz = hardsimple_feature_net(point_xyz, num_voxels, voxel_coord)
   
    mean_point_xyz = mean_point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32)
    point_xyz = point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32)

    return mean_point_xyz , point_xyz ,voxel_coord
   

def test_3dConViT():
    if not torch.cuda.is_available():
        pytest.skip()

    voxel_layer =dict()
    voxel_encoder =dict(type='HardSimpleVFE')
    middle_liyer =  dict(
                type='PointNet2SASSG_SL',
                in_channels=4,
                num_points=(32,16), # irrelavent
                radius=(0.8, 1.2),
                num_samples=(16,8),
                sa_channels=((8, 16), (16, 16)),
                fp_channels=((16, 16), (16, 16)),
                norm_cfg=dict(type='BN2d'))
    
    mean_point_xyz , point_xyz ,voxel_coord = vfe_feature_encoder()
    
    # voxelization_layer = build_voxel_encoder(voxel_encoder)
    # voxelization_layer.cuda()
    # x = voxelization_layer()


    ### using custome pointnet embedding
    embedding_encoder_layer = build_middle_encoder(middle_liyer)
    embedding_encoder_layer.cuda()
    feat_dic = embedding_encoder_layer(point_xyz,mean_point_xyz[:,:,:3])


    ### backdone
    cfg_backbone =  dict(
                type='ConViT3DDecoder',
                in_chans=16,
                embed_dim=96,
                depth = 12, # stochastic depth decay rule
                num_heads=12 ,
                mlp_ratio=4,
                qkv_bias=False ,
                qk_scale=None ,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0, 
                hybrid_backbone=None ,
                global_pool=None,
                local_up_to_layer=10 ,
                locality_strength=1,
                use_pos_embed=True,
                init_cfg=None,
                pretrained=None,
                fp_channels = ((576,16),(16,16)), ) # (head*embed_dim , output_dim)
    backbone = build_backbone(cfg_backbone)
    backbone.cuda()

    x = backbone(feat_dic)
    print("Pass")


if __name__ == "__main__":
    test_3dConViT()