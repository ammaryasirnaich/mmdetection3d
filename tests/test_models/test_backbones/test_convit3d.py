# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch


from mmdet3d.registry import MODELS



class convit3d_module_test:

    def __init__(self) -> None:
        self.point_xyz = torch.rand([45000, 20, 4]).cuda()
        self.num_voxels = torch.randint(1, 100, [45000])
        self.voxel_coord = torch.randint(0, 100, [45000, 3]).cuda()
        self.bck_point_xyz = torch.rand([45000, 20, 16]).cuda()
     
 
    def get_used_memory(self):
        free,total = torch.cuda.mem_get_info()
        used = (total-free)/(1024*1024*1024)
        return used


    def test_middle_encoder(self):
        hardsimple_feature_net_cfg = dict(
        type='HardSimpleVFE',      
        )
        hardsimple_feature_net = MODELS.build(hardsimple_feature_net_cfg)

        mean_point_xyz = hardsimple_feature_net(self.point_xyz, self.num_voxels, self.voxel_coord)
   
        mean_point_xyz = mean_point_xyz.view(1,-1,4).to('cuda:0', dtype=torch.float32)
        
        cfg = dict(type='PointNet2SASSG_SL',
                    in_channels=4,
                    num_points=(32,16), # irrelavent
                    radius=(0.8, 1.2),
                    num_samples=(16,8),
                    sa_channels=((8, 16), (16, 16)),
                    fp_channels=((16, 16), (16, 16)),
                    norm_cfg=dict(type='BN2d'))
          
        encoder = MODELS.build(cfg)
        encoder.cuda()

        ret_dict = encoder(self.point_xyz,mean_point_xyz[:,:,:3])
        print("keys", ret_dict.keys())
        print("Encoder test Pass")
        print("memory used in GB:", self.get_used_memory() )
   

    def test_convit3d_backbone(self):
        if not torch.cuda.is_available():
            pytest.skip()
        # DGCNNGF used in segmentation
            
        cfg = dict( type='ConViT3DDecoder',
                    num_classes=3, 
                    in_chans=16, #19
                    embed_dim=16, #19
                    depth = 6, #  Depths Transformer stage. Default 12
                    num_heads=4 ,  # 12
                    mlp_ratio=4,
                    qkv_bias=False ,
                    qk_scale=None ,
                    drop_rate=0,
                    attn_drop_rate=0,
                    drop_path_rate=0, 
                    hybrid_backbone=None ,
                    global_pool=None,
                    local_up_to_layer=4 ,  #Consider how many layers to work for local feature aggregation
                    locality_strength=1,
                    use_pos_embed=False,
                    init_cfg=None,
                    pretrained=None,
                    fp_output_channel = 16, 
                    )

        backbone = MODELS.build(cfg)
        backbone.cuda()

        feat_dic=[]
        self.bck_point_xyz = self.bck_point_xyz.expand(1,-1, -1,-1)
        # print("point_xyz shape",self.point_xyz.shape)
        feat_dic = dict(fp_features=self.bck_point_xyz)
        ret_dict = backbone(feat_dic, self.voxel_coord)
        print("keys", ret_dict.keys())
        print("Backbone test Pass")
        print("memory used in GB:", self.get_used_memory() )
        print("pass")

        # gf_points = ret_dict['gf_points']
        # fa_points = ret_dict['fa_points']

        # assert len(gf_points) == 4
        # assert gf_points[0].shape == torch.Size([1, 100, 6])
        # assert gf_points[1].shape == torch.Size([1, 100, 64])
        # assert gf_points[2].shape == torch.Size([1, 100, 64])
        # assert gf_points[3].shape == torch.Size([1, 100, 64])
        # assert fa_points.shape == torch.Size([1, 100, 1216])


if __name__ == "__main__":
    cpnt = convit3d_module_test()
    cpnt.test_middle_encoder()
    cpnt.test_convit3d_backbone()