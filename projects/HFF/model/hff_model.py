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
                 voxel_encoder_SparseBEVTransformer: Optional[dict] = None,
                 fusion_module_MultiResolutionFusion: Optional[dict] = None,
                 decode_head_SparseVoxelDecoder: Optional[dict] = None,
                 mask_head_MaskTransformerHead: Optional[dict] = None,
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

        # Custom layers with type included in the variable names
        self.voxel_encoder_SparseBEVTransformer = MODELS.build(voxel_encoder_SparseBEVTransformer) if voxel_encoder_SparseBEVTransformer else None
        self.fusion_module_MultiResolutionFusion = MODELS.build(fusion_module_MultiResolutionFusion) if fusion_module_MultiResolutionFusion else None
        self.decode_head_SparseVoxelDecoder = MODELS.build(decode_head_SparseVoxelDecoder) if decode_head_SparseVoxelDecoder else None
        self.mask_head_MaskTransformerHead = MODELS.build(mask_head_MaskTransformerHead) if mask_head_MaskTransformerHead else None

        # Training and testing configuration
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img, pts=None):
        """Extract features from images and LiDAR data."""
        img_feats = self.extract_img_feat(img)
        pts_feats = self.extract_pts_feat(pts)

        # Fusion logic using the hierarchical feature fusion module
        fused_feats = self.fusion_module_MultiResolutionFusion(img_feats, pts_feats) if self.fusion_module_MultiResolutionFusion else None
        return fused_feats

    def forward_train(self,
                      img_metas,
                      img=None,
                      pts=None,
                      voxel_semantics=None,
                      voxel_instances=None,
                      instance_class_ids=None,
                      **kwargs):
        """Training forward function."""
        # Extract features from multi-modal inputs
        fused_feats = self.extract_feat(img, pts)

        # Run the voxel encoder and transformer-based head for 3D occupancy prediction
        decode_output = self.decode_head_SparseVoxelDecoder(fused_feats, img_metas)
        mask_output = self.mask_head_MaskTransformerHead(decode_output, img_metas)

        # Compute losses
        losses = self.compute_loss(decode_output, mask_output, voxel_semantics, voxel_instances, instance_class_ids)
        return losses

    def forward_test(self, img_metas, img=None, pts=None, **kwargs):
        """Testing forward function."""
        fused_feats = self.extract_feat(img, pts)

        # Run the voxel encoder and transformer-based head for 3D occupancy prediction
        decode_output = self.decode_head_SparseVoxelDecoder(fused_feats, img_metas)
        mask_output = self.mask_head_MaskTransformerHead(decode_output, img_metas)

        return self.format_results(decode_output, mask_output)

    def compute_loss(self, decode_output, mask_output, voxel_semantics, voxel_instances, instance_class_ids):
        """Calculate the losses for occupancy prediction."""
        loss_decode = self.decode_head_SparseVoxelDecoder.loss(voxel_semantics, voxel_instances, decode_output)
        loss_mask = self.mask_head_MaskTransformerHead.loss(mask_output, instance_class_ids)
        return dict(loss_decode=loss_decode, loss_mask=loss_mask)

    def format_results(self, decode_output, mask_output):
        """Format the outputs for inference."""
        return dict(decode_output=decode_output, mask_output=mask_output)
