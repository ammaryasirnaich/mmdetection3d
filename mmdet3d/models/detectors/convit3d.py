

# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor
import torch

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStage3DDetector
from mmcv.ops import Voxelization

from .votenet import VoteNet
from .point_rcnn import PointRCNN
from typing import Dict, Optional

@MODELS.register_module()
class ConVit3D(VoteNet):  #PointRCNN ,  VoteNet
    r"""for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: None,
                 backbone: ConfigType,
                 neck: OptConfigType = ConfigType,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.voxel_encoder = MODELS.build(voxel_encoder)
        
        if middle_encoder is not None:
            self.middle_encoder = MODELS.build(middle_encoder)
   

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
       
       
        # the voxel_feature (V,D(4)) is the mean voxel point(xyz)  
        voxel_dict = batch_inputs_dict['voxels']
       
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        
        batch_size = voxel_dict['coors'][-1, 0].item() + 1       
        voxel_features = voxel_features.expand(batch_size,-1,-1)  #(B,V,D)
                    
        x = self.backbone(voxel_features)

        # print("feature shape", x['sa_features'][-1].shape)
        
        x['raw_points']=torch.stack(batch_inputs_dict['points'])[:,:,:3]  # (N,D(3))


        if self.with_neck:
            fp_xyz = x['sa_xyz'][-1]
            x = self.neck(x,fp_xyz)
        return x
    

    