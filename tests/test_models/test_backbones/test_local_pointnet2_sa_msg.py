# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.registry import MODELS


def test_pointnet2_sa_msg_1():
    if not torch.cuda.is_available():
       pytest.skip()
        # PN2MSG used in 3DSSD
    cfg = dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(256, 64,  32),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((8, 8, 16), (8, 8, 16), (8, 8, 8)),
        sa_channels=(((8, 8, 16), (8, 8, 16),(8, 8, 16)), 
                     ((16, 16, 32), (16, 16, 32), (16, 24, 32)),
                     ((32, 32, 64), (32, 24, 64), (32, 64, 64))),
        aggregation_channels=(16, 32, 64),
        fps_mods=(('D-FPS'), ('D-FPS'), ('F-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1)),
        dilated_group=(False, False, False),
        norm_cfg=dict(type='BN2d'),
        out_indices=(0, 1, 2),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False))

    # self = MODELS.build(cfg)
    # self.cuda()
    # assert self.SA_modules[0].mlps[0].layer0.conv.in_channels == 4
    # assert self.SA_modules[0].mlps[0].layer0.conv.out_channels == 8
    # assert self.SA_modules[0].mlps[1].layer1.conv.out_channels == 8
    # assert self.SA_modules[2].mlps[2].layer2.conv.out_channels == 64
    
        # importing kitti dataset
    xyz = np.fromfile('/workspace/mmdetection3d/tests/data/kitti/training/velodyne_reduced/000000.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 4).cuda()  # (B, N, 4)
    
    self = MODELS.build(cfg)
    self.cuda()
    ret_dict = self(xyz)
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']

        
    print(" print xyz original data", xyz.shape)

    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 4
    
    print("sa_xyz[0]:",sa_xyz[0].shape)
    print("sa_xyz[1]:",sa_xyz[1].shape)
    print("sa_xyz[2]:",sa_xyz[2].shape)
    print("sa_xyz[3]:",sa_xyz[3].shape)
    
    print("sa_features[0]:",sa_features[0].shape)
    print("sa_features[1]:",sa_features[1].shape)
    print("sa_features[2]:",sa_features[2].shape)
    print("sa_features[3]:",sa_features[3].shape)
    
    print("sa_indices[0]:",sa_indices[0].shape)
    print("sa_indices[1]:",sa_indices[1].shape)
    print("sa_indices[2]:",sa_indices[2].shape)
    print("sa_indices[3]:",sa_indices[3].shape)
    





def test_pointnet2_sa_msg():
    if not torch.cuda.is_available():
        pytest.skip()

    # PN2MSG used in segmentation
    cfg2 = dict(
        type='PointNet2SAMSG',
        in_channels=6,  # [xyz, rgb] for sunrgbd data instance
        # in_channels=4,  # [xyz, r] for kitti data instance
        num_points=(1024, 256, 64, 16),
        radii=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), 
                     ((64, 64, 128), (64, 96,128)),
                     ((128, 196, 256), (128, 196, 256)),
                     ((256, 256, 512), (256, 384, 512))),
        aggregation_channels=(None, None, None, None),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False))
    
    
    
    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    ##test forward
   
   
    self = MODELS.build(cfg2)
    self.cuda()
    ret_dict = self(xyz)
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']


    print(" print xyz original data", xyz.shape)
    print("len(sa_xyz)", len(sa_xyz))
    print("len(sa_features)", len(sa_features))
    print("len(sa_indices)", len(sa_indices))

    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 5
    
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 1024, 3])
    assert sa_xyz[2].shape == torch.Size([1, 256, 3])
    assert sa_xyz[3].shape == torch.Size([1, 64, 3])
    assert sa_xyz[4].shape == torch.Size([1, 16, 3])
    
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 96, 1024])
    assert sa_features[2].shape == torch.Size([1, 256, 256])
    assert sa_features[3].shape == torch.Size([1, 512, 64])
    assert sa_features[4].shape == torch.Size([1, 1024, 16])
    
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 1024])
    assert sa_indices[2].shape == torch.Size([1, 256])
    assert sa_indices[3].shape == torch.Size([1, 64])
    assert sa_indices[4].shape == torch.Size([1, 16])



def test_convit3d_pointnet2_sa_msg():
    cfg3 = dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, (256, 256)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        # aggregation_channels=(64, 128, 256),
        out_indices=(0, 1, 2),
        # aggregation_channels=(None, None, None),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        # norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False))
    
    # importing kitti dataset
    xyz = np.fromfile('/workspace/mmdetection3d/tests/data/kitti/training/velodyne_reduced/000000.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 4).cuda()  # (B, N, 4)
    
    self = MODELS.build(cfg3)
    self.cuda()
    ret_dict = self(xyz)
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']

        
    print(" print xyz original data", xyz.shape)

    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 4
    
    print("sa_xyz[0]:",sa_xyz[0].shape)
    print("sa_xyz[1]:",sa_xyz[1].shape)
    print("sa_xyz[2]:",sa_xyz[2].shape)
    print("sa_xyz[3]:",sa_xyz[3].shape)
    
    print("sa_features[0]:",sa_features[0].shape)
    print("sa_features[1]:",sa_features[1].shape)
    print("sa_features[2]:",sa_features[2].shape)
    print("sa_features[3]:",sa_features[3].shape)
    
    print("sa_indices[0]:",sa_indices[0].shape)
    print("sa_indices[1]:",sa_indices[1].shape)
    print("sa_indices[2]:",sa_indices[2].shape)
    print("sa_indices[3]:",sa_indices[3].shape)
    
    
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 1024, 3])
    assert sa_xyz[2].shape == torch.Size([1, 256, 3])
    assert sa_xyz[3].shape == torch.Size([1, 64, 3])
    assert sa_xyz[4].shape == torch.Size([1, 16, 3])
    
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 96, 1024])
    assert sa_features[2].shape == torch.Size([1, 256, 256])
    assert sa_features[3].shape == torch.Size([1, 512, 64])
    assert sa_features[4].shape == torch.Size([1, 1024, 16])
    
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 1024])
    assert sa_indices[2].shape == torch.Size([1, 256])
    assert sa_indices[3].shape == torch.Size([1, 64])
    assert sa_indices[4].shape == torch.Size([1, 16])
    





if __name__ =="__main__":
    # test_pointnet2_sa_msg()
    test_pointnet2_sa_msg_1()
    # test_convit3d_pointnet2_sa_msg()