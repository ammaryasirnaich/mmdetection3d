# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.models import build_backbone

def test_multi_backbone():
    if not torch.cuda.is_available():
        pytest.skip()

    # test list config
    cfg_list = dict(
        type='MultiBackbone',
        num_streams=4,
        suffixes=['net0', 'net1', 'net2', 'net3'],
        backbones=[
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d')),
            dict(
                type='PointNet2SASSG',
                in_channels=4,
                num_points=(256, 128, 64, 32),
                radius=(0.2, 0.4, 0.8, 1.2),
                num_samples=(64, 32, 16, 16),
                sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                             (128, 128, 256)),
                fp_channels=((256, 256), (256, 256)),
                norm_cfg=dict(type='BN2d'))
        ])


    self = build_backbone(cfg_list)
    self.cuda()

    assert len(self.backbone_list) == 4

    xyz = np.fromfile('/workspace/mmdetection3d/tests/data/sunrgbd/points/000001.bin', dtype=np.float32)
    xyz = torch.from_numpy(xyz).view(1, -1, 6).cuda()  # (B, N, 6)
    # test forward
    ret_dict = self(xyz[:, :, :4])

    assert ret_dict['hd_feature'].shape == torch.Size([1, 256, 128])
    assert ret_dict['fp_xyz_net0'][-1].shape == torch.Size([1, 128, 3])
    assert ret_dict['fp_features_net0'][-1].shape == torch.Size([1, 256, 128])


if __name__ =="__main__":
    test_multi_backbone()