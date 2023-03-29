# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.models import build_backbone
from mmdet3d.models.builder import build_voxel_encoder,DETECTORS
from mmdet3d.core.voxel.voxel_generator import VoxelGenerator
from mmcv.ops import Voxelization
import mmdet3d.models.builder





def vfe_feature_encoder(): 
    ### configuration for VFE encoder 
    hardsimple_feature_net_cfg = dict(
        type='HardSimpleVFE',      
        )
    hardsimple_feature_net = build_voxel_encoder(hardsimple_feature_net_cfg)

    point_xyz = torch.rand([97297, 20, 4])
    num_voxels = torch.randint(1, 100, [97297])
    voxel_coord = torch.randint(0, 100, [97297, 3])
    mean_point_xyz = hardsimple_feature_net(point_xyz, num_voxels, voxel_coord)
   
    mean_point_xyz = mean_point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32)

    return mean_point_xyz , point_xyz ,voxel_coord
   

def test_3DConViT():
    if not torch.cuda.is_available():
        pytest.skip()

    voxel_layer =dict()
    voxel_encoder =dict(type='HardSimpleVFE')

    # test list config
    cfg_list = dict(
        type='MultiBackbone',
        num_streams=2,
        suffixes=['pe','convit'],
        backbones=[
            dict(
                type='PointNet2SASSG_SL',
                in_channels=4,
                num_points=(32,16), # irrelavent
                radius=(0.8, 1.2),
                num_samples=(16,8),
                sa_channels=((8, 16), (16, 16)),
                fp_channels=((16, 16), (16, 16)),
                norm_cfg=dict(type='BN2d')),       
            dict(
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
        ])

    # if not torch.cuda.is_available():
    #     pytest.skip() 

    # # test list config
    # cfg_list = dict(
    # type='MultiBackbone',
    # num_streams=2,
    # suffixes=['net0', 'net1'],
    # backbones=[
    #     dict(
    #         type='PointNet2SASSG_SL',
    #         in_channels=4,
    #         num_points=(256, 128, 64, 32),
    #         radius=(0.2, 0.4, 0.8, 1.2),
    #         num_samples=(64, 32, 16, 16),
    #         sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
    #                     (128, 128, 256)),
    #         fp_channels=((256, 256), (256, 256)),
    #         norm_cfg=dict(type='BN2d')),
    #     dict(
    #         type='PointNet2SASSG',
    #         in_channels=4,
    #         num_points=(256, 128, 64, 32),
    #         radius=(0.2, 0.4, 0.8, 1.2),
    #         num_samples=(64, 32, 16, 16),
    #         sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
    #                     (128, 128, 256)),
    #         fp_channels=((256, 256), (256, 256)),
    #         norm_cfg=dict(type='BN2d')),
    
    # ])

    self = build_backbone(cfg_list)
    self.cuda()

    assert len(self.backbone_list) == 2

    xyz = np.fromfile('/workspace/mmdetection3d/tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    x = dict(point_xyz=xyz[:, :, :4], point_feature=xyz[:, :, :3])

    ret_dict = self(x)

    # assert ret_dict['hd_feature'].shape == torch.Size([1, 256, 128])
    # assert ret_dict['fp_xyz_net0'][-1].shape == torch.Size([1, 128, 3])
    # assert ret_dict['fp_features_net0'][-1].shape == torch.Size([1, 256, 128])


    
    # self = build_backbone(cfg_list)

    # self.cuda()
    
    # mean_point_xyz , point_xyz, cntr_voxel_xyz = vfe_feature_encoder()
    # mean_point_xyz  = mean_point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32).contiguous()   # (B, N, 4)
    # point_xyz  = point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32).contiguous()   # (B, V*N, 4)
   

    # assert len(self.backbone_list) == 2


    # test forward
    # ret_dict = self(point_xyz, mean_point_xyz[...,:3] ) 
    print("Pass")

    # assert ret_dict['hd_feature'].shape == torch.Size([1, 256, 128])
    # assert ret_dict['fp_xyz_net0'][-1].shape == torch.Size([1, 128, 3])
    # assert ret_dict['fp_features_net0'][-1].shape == torch.Size([1, 256, 128])


if __name__ =="__main__":
    test_3DConViT()