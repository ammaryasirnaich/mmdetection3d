import torch
from typing import Optional
from mmengine.structures import InstanceData
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList

@MODELS.register_module()
class HFFModel(Base3DDetector):
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
                 init_cfg: OptMultiConfig = None,
                 ):
        super(HFFModel, self).__init__(
              data_preprocessor= data_preprocessor,init_cfg=init_cfg )

        #        
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build( img_neck) if img_neck is not None else None
        
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

    
        self.sparse_bev_transformer = MODELS.build(img_point_encoder) if img_point_encoder else None
        self.multi_resolution_fusion = MODELS.build(fusion_module) if fusion_module else None
        self.mask_tnsform_head = MODELS.build(mask_head) if mask_head else None
        
        # later setting up
        # self.bbox_head = MODELS.build(bbox_head)
        
        self.init_weights()
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
