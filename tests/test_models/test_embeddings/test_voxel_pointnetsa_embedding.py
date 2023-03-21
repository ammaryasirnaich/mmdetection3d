import torch

from mmdet3d.models.builder import build_voxel_encoder
import numpy as np
import pytest
import torch

from mmdet3d.models import build_backbone



def vfe_feature_encoder():
    
    ### configuration for VFE encoder 
    hardsimple_feature_net_cfg = dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[16],
        with_distance=False,
        voxel_size=(0.5, 0.5, 0.5),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        return_point_feats=False,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))

    hardsimple_feature_net = build_voxel_encoder(hardsimple_feature_net_cfg)

    features = torch.rand([97297, 20, 4])
    num_voxels = torch.randint(1, 100, [97297])
    coors = torch.randint(0, 100, [97297, 3])
    voxel_feats = hardsimple_feature_net(features, num_voxels, coors)
    return voxel_feats , coors
   

def voxel_point_encoding():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='PointNet2SASSG',
        in_channels=3,
        num_points=(32, 16),
        radius=(0.8, 1.2),
        num_samples=(16, 8),
        sa_channels=((8, 16), (16, 16)),
        fp_channels=((16, 16), (16, 16)))
    self = build_backbone(cfg)
    self.cuda()

    voxel_feats , coors = vfe_feature_encoder()

    # xyz = np.fromfile('/workspace/mmdetection3d/tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    # xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    
    # test forward
   
    coors = coors.view(1,-1,3).to('cuda:0', dtype=torch.float32)
    # sun_data = xyz[..., :3]

    print("coor shape", coors.shape)

    # print("sunrgbd", sun_data.shape)


    ret_dict = self(coors)
    # ret_dict = self(sun_data)
    fp_xyz = ret_dict['fp_xyz']
    fp_features = ret_dict['fp_features']
    fp_indices = ret_dict['fp_indices']
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']

    
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])





def test_pointnet2_sa_ssg():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='PointNet2SASSG',
        in_channels=6,
        num_points=(32, 16),
        radius=(0.8, 1.2),
        num_samples=(16, 8),
        sa_channels=((8, 16), (16, 16)),
        fp_channels=((16, 16), (16, 16)))
    self = build_backbone(cfg)
    self.cuda()

    assert self.SA_modules[0].mlps[0].layer0.conv.in_channels == 6
    assert self.SA_modules[0].mlps[0].layer0.conv.out_channels == 8
    assert self.SA_modules[0].mlps[0].layer1.conv.out_channels == 16
    assert self.SA_modules[1].mlps[0].layer1.conv.out_channels == 16
    assert self.FP_modules[0].mlps.layer0.conv.in_channels == 32
    assert self.FP_modules[0].mlps.layer0.conv.out_channels == 16
    assert self.FP_modules[1].mlps.layer0.conv.in_channels == 19

    xyz = np.fromfile('/workspace/mmdetection3d/tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz)
    fp_xyz = ret_dict['fp_xyz']
    fp_features = ret_dict['fp_features']
    fp_indices = ret_dict['fp_indices']
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_xyz[0].shape == torch.Size([1, 16, 3])
    assert fp_xyz[1].shape == torch.Size([1, 32, 3])
    assert fp_xyz[2].shape == torch.Size([1, 100, 3])
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert fp_indices[0].shape == torch.Size([1, 16])
    assert fp_indices[1].shape == torch.Size([1, 32])
    assert fp_indices[2].shape == torch.Size([1, 100])
    assert sa_xyz[0].shape == torch.Size([1, 100, 3])
    assert sa_xyz[1].shape == torch.Size([1, 32, 3])
    assert sa_xyz[2].shape == torch.Size([1, 16, 3])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])
    assert sa_indices[0].shape == torch.Size([1, 100])
    assert sa_indices[1].shape == torch.Size([1, 32])
    assert sa_indices[2].shape == torch.Size([1, 16])

    # test only xyz input without features
    cfg['in_channels'] = 3
    self = build_backbone(cfg)
    self.cuda()
    ret_dict = self(xyz[..., :3])
    assert len(fp_xyz) == len(fp_features) == len(fp_indices) == 3
    assert len(sa_xyz) == len(sa_features) == len(sa_indices) == 3
    assert fp_features[0].shape == torch.Size([1, 16, 16])
    assert fp_features[1].shape == torch.Size([1, 16, 32])
    assert fp_features[2].shape == torch.Size([1, 16, 100])
    assert sa_features[0].shape == torch.Size([1, 3, 100])
    assert sa_features[1].shape == torch.Size([1, 16, 32])
    assert sa_features[2].shape == torch.Size([1, 16, 16])






def point_embedding_backbone():
    if not torch.cuda.is_available():
        pytest.skip()

    cfg = dict(
        type='VoxelPoinetEmbedding',
        # in_channels=4,
        num_points=(32, 16),
        radius=(0.8, 1.2),
        num_samples=(16, 8),
        sa_channels=((8, 16), (16, 16)),
        fp_channels=((16, 16), (16, 16)))
    self = build_backbone(cfg)
    self.cuda()

    voxel_feats , coors = vfe_feature_encoder()

    print("voxel_feats", voxel_feats.shape)
   
    coors = coors.view(1,-1,3).to('cuda:0', dtype=torch.float32)
    voxel_feats  = voxel_feats.view(1,-1,4).to('cuda:0', dtype=torch.float32)


    print("coor shape", coors.shape)


    ret_dict = self(coors,voxel_feats)
    fp_xyz = ret_dict['fp_xyz']
    fp_features = ret_dict['fp_features']
    fp_indices = ret_dict['fp_indices']
    sa_xyz = ret_dict['sa_xyz']
    sa_features = ret_dict['sa_features']
    sa_indices = ret_dict['sa_indices']

  




if __name__ == "__main__":
    # test_hard_simple_VFE()
    voxel_feats, voxel_corrd = vfe_feature_encoder()
    print("feature shape", voxel_feats.shape)
    print("feature shape", voxel_corrd.shape)
    point_embedding_backbone()
    # test_pointnet2_sa_ssg()
    # point_embedding()
    print("End")
   
 
