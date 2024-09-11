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

# from .segmentation import SegmentationHead
from .splitshoot import LiftSplatShoot
from .adaptive_feature_refinement import Refine_Resolution_Adjacement

@MODELS.register_module()
class SDHFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        img_voxelization: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        refine_adj_cfg : Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
               
        ## point cloud module initialization
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
 
        # self.refine_resolution_adj = Refine_Resolution_Adjacement().cuda()
        self.refine_resolution_adj = MODELS.build(refine_adj_cfg)
             
        # Note update this part
        self.splitshoot=LiftSplatShoot(depth_bins=100, H=64, W=176,N=6, bev_channels=64).cuda() # depth_bins(100 meters), H=64, W=176 from resnetfpn
       
        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

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
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat( self, x, points, lidar2image, camera_intrinsics,
        camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas, ) -> torch.Tensor:
        """_summary_

        Args:
            x (_type_): _description_
            points (_type_): _description_
            lidar2image (_type_): _description_
            camera_intrinsics (_type_): _description_
            camera2lidar (_type_): _description_
            img_aug_matrix (_type_): _description_
            lidar_aug_matrix (_type_): _description_
            img_metas (_type_): _description_

        Returns:
            x : 2D-to-3D projected point clouds [B, N, C(3), H, W, D(100)]) 
            bev_feature : 3D point cloud projection on BEV plan [B, N, H, W, C]
            
        """
        
        B, N, C, H, W = x.size()

        x = x.view(B * N, C, H, W).contiguous()
        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]
                
        # BN, C, H, W = x.size()  ([24, 256, 64, 176])
        x, bev_feature = self.splitshoot(x, camera_intrinsics, lidar2image)   
        return x, bev_feature 
        
        # x = x.view(B, int(BN / B), C, H, W)
        
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
        
        '''
        
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                H, W = img.shape[-2:]

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        ####img_feats = self.extract_img_feat(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)


        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped
        '''
        
        # return x
        
        

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # with torch.autocast('cuda', enabled=False):
        #     points = [point.float() for point in points]
        #     feats, coords, sizes = self.voxelize(points)
        #     batch_size = coords[-1, 0] + 1
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        
        x = self.pts_backbone(x)    
        x = self.pts_neck(x)
        
           
        # self.forward_pts_train(img_feats, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas)
        
        return x


    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        print(imgs.shape)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            
            # Image feature extratir module
            img_feature,img_bev_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            # features.append(img_feature)
        
        # Point feature encoder model
        print(img_bev_feature.shape)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
         
        fused_feature, complexity_score = self.refine_resolution_adj(pts_feature,img_bev_feature)
        print(f'adaptive feature shape: {fused_feature.shape}')
        print(f'complexity_score shape: {complexity_score.shape}')
        
        # adaptive feature shape: torch.Size([4, 512, 200, 176])
        # complexity_score shape: torch.Size([4, 1, 200, 176])
        
        # SegmentationHead(input_dim, 10)
        # # Final segmentation
        # predictions = SegmentationHead(self.adaptive_feature)

        return fused_feature
    
    
    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.img_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if True:   #self.voxelize_reduce = True
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes



    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses


'''
  def forward_pts_train(self, mlvl_feats, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas):
        """
        voxel_semantics: [bs, 200, 200, 16], value in range [0, num_cls - 1]
        voxel_instances: [bs, 200, 200, 16], value in range [0, num_obj - 1]
        instance_class_ids: [[bs0_num_obj], [bs1_num_obj], ...], value in range [0, num_cls - 1]
        """
        outs = self.pts_bbox_head(mlvl_feats, img_metas)
        loss_inputs = [voxel_semantics, voxel_instances, instance_class_ids, outs]
        return self.pts_bbox_head.loss(*loss_inputs)
'''