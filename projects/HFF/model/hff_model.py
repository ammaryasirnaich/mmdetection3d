import torch
from typing import Optional
from mmengine.structures import InstanceData
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.structures import Det3DDataSample
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np

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


    def extract_img_feat(self, x) -> torch.Tensor:
    #     self,
    #     x,
    #     points,
    #     lidar2image,
    #     camera_intrinsics,
    #     camera2lidar,
    #     img_aug_matrix,
    #     lidar_aug_matrix,
    #     img_metas,
    # ) -> torch.Tensor:
    

        print(x.size())
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # with torch.autocast(device_type='cuda', dtype=torch.float32):
         
        #     x = self.view_transform(
        #         x,
        #         points,
        #         lidar2image,
        #         camera_intrinsics,
        #         camera2lidar,
        #         img_aug_matrix,
        #         lidar_aug_matrix,
        #         img_metas,
        #     )
        
        return x
    
    
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        feats, coords, batch_size = self.pts_voxel_encoder(points)
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x
    
    
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        # if imgs is not None:
        #     imgs = imgs.contiguous()
        #     lidar2image, camera_intrinsics, camera2lidar = [], [], []
        #     img_aug_matrix, lidar_aug_matrix = [], []
        #     for i, meta in enumerate(batch_input_metas):
        #         lidar2image.append(meta['lidar2img'])
        #         camera_intrinsics.append(meta['cam2img'])
        #         camera2lidar.append(meta['cam2lidar'])
        #         img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
        #         lidar_aug_matrix.append(
        #             meta.get('lidar_aug_matrix', np.eye(4)))

        #     lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        #     camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
        #     camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        #     img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        #     lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
        #     img_feature = self.extract_img_feat(imgs, deepcopy(points),
        #                                         lidar2image, camera_intrinsics,
        #                                         camera2lidar, img_aug_matrix,
        #                                         lidar_aug_matrix,
        #                                         batch_input_metas)
        img_feature = self.extract_img_feat(imgs)
        features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        # x = self.pts_backbone(x)
        # x = self.pts_neck(x)

        return x

    
    
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        
        
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        # if self.with_bbox_head:
        #     outputs = self.bbox_head.predict(feats, batch_input_metas)

        # res = self.add_pred_to_datasample(batch_data_samples, outputs)
        res = None
        return res
    
    
    
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass
    
    
    
    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        # log_vars = []
        # for loss_name, loss_value in losses.items():
        #     if isinstance(loss_value, torch.Tensor):
        #         log_vars.append([loss_name, loss_value.mean()])
        #     elif is_list_of(loss_value, torch.Tensor):
        #         log_vars.append(
        #             [loss_name,
        #              sum(_loss.mean() for _loss in loss_value)])
        #     else:
        #         raise TypeError(
        #             f'{loss_name} is not a tensor or list of tensors')

        # loss = sum(value for key, value in log_vars if 'loss' in key)
        # log_vars.insert(0, ['loss', loss])
        # log_vars = OrderedDict(log_vars)  # type: ignore

        # for loss_name, loss_value in log_vars.items():
        #     # reduce loss when distributed training
        #     if dist.is_available() and dist.is_initialized():
        #         loss_value = loss_value.data.clone()
        #         dist.all_reduce(loss_value.div_(dist.get_world_size()))
        #     log_vars[loss_name] = loss_value.item()

        # return loss, log_vars  # type: ignore
        pass



    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None
    
    
    
    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        # batch_input_metas = [item.metainfo for item in batch_data_samples]
        # feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        # losses = dict()
        # if self.with_bbox_head:
        #     bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        # losses.update(bbox_loss)
        losses = None
        return losses
