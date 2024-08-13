# mm3d/models/backbones/multi_modal_backbone.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmengine.structures import InstanceData

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.layers.fusion_layers.point_fusion import point_sample
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList



class MultiModalBackbone(Base3DDetector):
    '''
    img_backbone: Builds the image backbone (e.g., ResNet) using provided configuration.
    pts_backbone: Builds the point cloud backbone (e.g., PointNet++).
    img_neck and pts_neck: Optional necks for further processing.
    
    '''
    def __init__(self, 
                 img_backbone_cfg, 
                 pts_backbone_cfg, 
                 img_neck_cfg=None, 
                 pts_neck_cfg=None):
        super(MultiModalBackbone, self).__init__()
        self.img_backbone = MODELS.build(img_backbone_cfg)
        self.pts_backbone = MODELS.build(pts_backbone_cfg)
        
        if img_neck_cfg:
            self.img_neck = MODELS.build(img_neck_cfg)
        else:
            self.img_neck = None
        
        if pts_neck_cfg:
            self.pts_neck = MODELS.build(pts_neck_cfg)
        else:
            self.pts_neck = None

    def forward(self, img, points):
        img_feats = self.img_backbone(img)
        if self.img_neck:
            img_feats = self.img_neck(img_feats)
        
        pts_feats = self.pts_backbone(points)
        if self.pts_neck:
            pts_feats = self.pts_neck(pts_feats)
        
        return img_feats, pts_feats
