# mm3d/models/backbones/multi_modal_backbone.py

import torch.nn as nn
from mmdet3d.models import builder
from mmdet3d.models.backbones import BaseBackbone

class MultiModalBackbone(BaseBackbone):
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
        self.img_backbone = builder.build_backbone(img_backbone_cfg)
        self.pts_backbone = builder.build_backbone(pts_backbone_cfg)
        
        if img_neck_cfg:
            self.img_neck = builder.build_neck(img_neck_cfg)
        else:
            self.img_neck = None
        
        if pts_neck_cfg:
            self.pts_neck = builder.build_neck(pts_neck_cfg)
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
