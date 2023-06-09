

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
                 middle_encoder: ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=None,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    # def __init__(self,
    #              voxel_encoder: ConfigType,
    #              middle_encoder: ConfigType,
    #              backbone: dict,
    #              neck: Optional[dict] = None,
    #              rpn_head: Optional[dict] = None,
    #              roi_head: Optional[dict] = None,
    #              train_cfg: Optional[dict] = None,
    #              test_cfg: Optional[dict] = None,
    #              init_cfg: Optional[dict] = None,
    #              data_preprocessor: Optional[dict] = None) -> Optional:
    #     super(PointRCNN, self).__init__(
    #         backbone=backbone,
    #         neck=neck,
    #         rpn_head=rpn_head,
    #         roi_head=roi_head,
    #         train_cfg=train_cfg,
    #         test_cfg=test_cfg,
    #         init_cfg=init_cfg,
    #         data_preprocessor=data_preprocessor)
        
        self.voxel_encoder = MODELS.build(voxel_encoder)
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
        
        print("voxel_features", voxel_features.shape)
        
        x = self.middle_encoder(voxel_dict['voxels'],voxel_features[:,:,:3]) # dic[voxels = voxel_feature] (B,V,P,D)       
        
       
        '''
        @ Note: Testing
        force regular structure
        '''
        # print("Origional Feture from  middle encoder", x.keys())
        # print("x['fp_features'] shape",x['fp_features'].shape)
        # x['fp_features'] =  x['fp_features'][:,:10000,:,:]
        # voxel_dict['coors'] = voxel_dict['coors'][:10000,:]
        # print("Setting lower limit for testing...:",x['fp_features'].shape)

        # print("pause")

    
        x = self.backbone(x,voxel_dict['coors'][:,1:])

        # voxel_raw_points = torch.stack(batch_inputs_dict['points'])
        x['raw_points']=torch.stack(batch_inputs_dict['points'])[:,:,:3]  # (N,D(3))

        if self.with_neck:
            x = self.neck(x)
        return x
    

    
