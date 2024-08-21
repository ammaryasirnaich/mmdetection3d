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
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 img_point_encoder: Optional[dict] = None,
                 fusion_module: Optional[dict] = None,
                 mask_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 ):
        super(HFFModel, self).__init__(
            img_backbone=img_backbone,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_neck=img_neck,
            data_preprocessor= data_preprocessor
        )

        # Build custom modules
    
        self.sparse_bev_transformer = MODELS.build(img_point_encoder) if img_point_encoder else None
        self.multi_resolution_fusion = MODELS.build(fusion_module) if fusion_module else None
        self.mask_tnsform_head = MODELS.build(mask_head) if mask_head else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img,img_metas, pts=None):
        """Extract features from images and LiDAR data."""
        print(f'img features',img.shape)
        
        
        img_feats = self.extract_img_feat(img,img_metas)
        pts_feats = self.extract_pts_feat(pts)

        # fused_feats = self.multi_resolution_fusion(img_feats, pts_feats) if self.fusion_module else None
        fused_feats =None
        return fused_feats

    def forward_train(self, img_metas, img=None, pts=None, **kwargs):
        """Training forward function."""
        
        print(img.shape)
        
        fused_feats = self.extract_feat(img,img_metas, pts)
        # decode_output = self.sparse_voxel_decoder(fused_feats, img_metas)
        # mask_output = self.mask_tnsform_head(decode_output, img_metas)

        # losses = self.compute_loss(decode_output, mask_output, **kwargs)
        # return losses
        pass

    def forward_test(self, img_metas, img=None, pts=None, **kwargs):
        """Testing forward function."""
        fused_feats = self.extract_feat(img, pts)
        decode_output = self.sparse_voxel_decoder(fused_feats, img_metas)
        mask_output = self.mask_tnsform_head(decode_output, img_metas)

        return self.format_results(decode_output, mask_output)

    def compute_loss(self, decode_output, mask_output, **kwargs):
        # loss_decode = self.sparse_voxel_decoder.loss(**kwargs, decode_output=decode_output)
        # loss_mask = self.mask_hmask_tnsform_head.loss(mask_output, **kwargs)
        # return dict(loss_decode=loss_decode, loss_mask=loss_mask)
        pass

    def format_results(self, decode_output, mask_output):
        # return dict(decode_output=decode_output, mask_output=mask_output)
        pass