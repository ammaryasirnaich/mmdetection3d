import torch
from typing import Optional
from mmengine.structures import InstanceData
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
@MODELS.register_module()
class HFFModel(MVXTwoStageDetector):
    def __init__(self,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 pts_voxel_layer: Optional[dict] = None,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 voxel_encoder: Optional[dict] = None,
                 fusion_module: Optional[dict] = None,
                 decode_head: Optional[dict] = None,
                 mask_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None):
        super(HFFModel, self).__init__(
            img_backbone=img_backbone,
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_neck=img_neck,
            pretrained=pretrained
        )

        # Build custom modules
        self.voxel_encoder = MODELS.build(voxel_encoder) if voxel_encoder else None
        self.fusion_module = MODELS.build(fusion_module) if fusion_module else None
        self.decode_head = MODELS.build(decode_head) if decode_head else None
        self.mask_head = MODELS.build(mask_head) if mask_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img, pts=None):
        """Extract features from images and LiDAR data."""
        img_feats = self.extract_img_feat(img)
        pts_feats = self.extract_pts_feat(pts)

        fused_feats = self.fusion_module(img_feats, pts_feats) if self.fusion_module else None
        return fused_feats

    def forward_train(self, img_metas, img=None, pts=None, **kwargs):
        """Training forward function."""
        fused_feats = self.extract_feat(img, pts)
        decode_output = self.decode_head(fused_feats, img_metas)
        mask_output = self.mask_head(decode_output, img_metas)

        losses = self.compute_loss(decode_output, mask_output, **kwargs)
        return losses

    def forward_test(self, img_metas, img=None, pts=None, **kwargs):
        """Testing forward function."""
        fused_feats = self.extract_feat(img, pts)
        decode_output = self.decode_head(fused_feats, img_metas)
        mask_output = self.mask_head(decode_output, img_metas)

        return self.format_results(decode_output, mask_output)

    def compute_loss(self, decode_output, mask_output, **kwargs):
        loss_decode = self.decode_head.loss(**kwargs, decode_output=decode_output)
        loss_mask = self.mask_head.loss(mask_output, **kwargs)
        return dict(loss_decode=loss_decode, loss_mask=loss_mask)

    def format_results(self, decode_output, mask_output):
        return dict(decode_output=decode_output, mask_output=mask_output)