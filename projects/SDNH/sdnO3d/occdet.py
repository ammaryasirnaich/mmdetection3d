# mm3d/models/detectors/occupancy_detector.py
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization




from adaptive_scaling import AdaptiveResolutionScaling
from hierarchialfusion import HierarchicalFeatureFusion
from occupancy_head import OccupancyPredictionHead

from mmdet3d.models.detectors import Base3DDetector

class OccupancyDetector(Base3DDetector):
    def __init__(self, 
                 img_backbone, 
                 pts_backbone, 
                 img_neck, 
                 pts_neck, 
                 fusion_cfg, 
                 scaling_cfg, 
                 occupancy_head_cfg):
        super(OccupancyDetector, self).__init__()
        # self.backbone = MultiModalBackbone(
        #     img_backbone=img_backbone,
        #     pts_backbone=pts_backbone,
        #     img_neck=img_neck,
        #     pts_neck=pts_neck
        # )
        
        self.img_backbone = MODELS.build(img_backbone)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.pts_neck = MODELS.build(pts_neck)
        # self.fusion = HierarchicalFeatureFusion(**fusion_cfg)
        # self.scaling = AdaptiveResolutionScaling(**scaling_cfg)
        # self.occupancy_head = OccupancyPredictionHead(**occupancy_head_cfg)

    def forward(self, return_loss=True, **kwargs):
        img = kwargs['img']
        points = kwargs['points']
        img_feats, pts_feats = self.backbone(img, points)
        # fused_feats = self.fusion(img_feats, pts_feats)
        # scaled_feats = self.scaling(fused_feats)
        # occupancy_pred = self.occupancy_head(scaled_feats)
        
        # if return_loss:
        #     gt_occupancy = kwargs['gt_occupancy']
        #     loss = self.loss(occupancy_pred, gt_occupancy)
        #     return loss
        # else:
        #     return occupancy_pred

    # def loss(self, preds, targets):
    #     criterion = nn.BCELoss()
    #     loss = criterion(preds, targets)
    #     return {'loss_occupancy': loss}
