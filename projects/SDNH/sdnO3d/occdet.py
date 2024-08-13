# mm3d/models/detectors/occupancy_detector.py

from adaptive_scaling import AdaptiveResolutionScaling
from hierarchialfusion import HierarchicalFeatureFusion
from occupancy_head import OccupancyPredictionHead
from mmbackbone import MultiModalBackbone

from mmdet3d.models.detectors import Base3DDetector

class OccupancyDetector(Base3DDetector):
    def __init__(self, 
                 img_backbone_cfg, 
                 pts_backbone_cfg, 
                 img_neck_cfg, 
                 pts_neck_cfg, 
                 fusion_cfg, 
                 scaling_cfg, 
                 occupancy_head_cfg):
        super(OccupancyDetector, self).__init__()
        self.backbone = MultiModalBackbone(
            img_backbone_cfg=img_backbone_cfg,
            pts_backbone_cfg=pts_backbone_cfg,
            img_neck_cfg=img_neck_cfg,
            pts_neck_cfg=pts_neck_cfg
        )
        self.fusion = HierarchicalFeatureFusion(**fusion_cfg)
        self.scaling = AdaptiveResolutionScaling(**scaling_cfg)
        self.occupancy_head = OccupancyPredictionHead(**occupancy_head_cfg)

    def forward(self, return_loss=True, **kwargs):
        img = kwargs['img']
        points = kwargs['points']
        img_feats, pts_feats = self.backbone(img, points)
        fused_feats = self.fusion(img_feats, pts_feats)
        scaled_feats = self.scaling(fused_feats)
        occupancy_pred = self.occupancy_head(scaled_feats)
        
        if return_loss:
            gt_occupancy = kwargs['gt_occupancy']
            loss = self.loss(occupancy_pred, gt_occupancy)
            return loss
        else:
            return occupancy_pred

    def loss(self, preds, targets):
        criterion = nn.BCELoss()
        loss = criterion(preds, targets)
        return {'loss_occupancy': loss}
