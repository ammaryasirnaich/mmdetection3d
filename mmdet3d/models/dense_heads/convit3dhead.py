from typing import Dict, List, Optional, Tuple, Union
from mmdet3d.structures import BaseInstance3DBoxes

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures.bbox_3d import (DepthInstance3DBoxes,
                                        LiDARInstance3DBoxes,
                                        rotation_3d_in_axis)
from .base_conv_bbox_head import BaseConvBboxHead
from torch import nn

from .ssd_3d_head import SSD3DHead

# Convit3DHead
@MODELS.register_module()
class Convit3DHead(SSD3DHead):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
    """

    def __init__(self,
                 num_classes: int,
                 bbox_coder: Union[ConfigDict, dict],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pred_layer_cfg: Optional[dict] = None,
                 objectness_loss: Optional[dict] = None,
                 center_loss: Optional[dict] = None,
                 dir_class_loss: Optional[dict] = None,
                 dir_res_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 corner_loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super(SSD3DHead,self).__init__(
            num_classes,
            bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pred_layer_cfg=pred_layer_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=None,
            size_res_loss=size_res_loss,
            semantic_loss=None,
            init_cfg=init_cfg)
        
        self.corner_loss = MODELS.build(corner_loss)
        self.vote_loss = None
        
  

    def _extract_input(self, feat_dict: dict) -> Tuple:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """

        aggregated_points = feat_dict['sa_xyz'][-1]
        aggregated_features = feat_dict["sa_features"][-1].permute(0,2,1)
        aggregated_indices = feat_dict['sa_indices'][-1]

        self.num_candidates = aggregated_points.shape[1]

        # return aggregated_points, aggregated_features, feat_dict['raw_points']

        return aggregated_points, aggregated_features, aggregated_indices
    
    def loss_by_feat(
            self,
            points: List[torch.Tensor],
            bbox_preds_dict: dict,
            batch_gt_instances_3d: List[InstanceData],
            batch_pts_semantic_mask: Optional[List[torch.Tensor]] = None,
            batch_pts_instance_mask: Optional[List[torch.Tensor]] = None,
            batch_input_metas: List[dict] = None,
            ret_target: bool = False,
            **kwargs) -> dict:
        """Compute loss.

        Args:
            points (list[torch.Tensor]): Input points.
            bbox_preds_dict (dict): Predictions from forward of vote head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_pts_semantic_mask (list[tensor]): Semantic mask
                of points cloud. Defaults to None. Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Instance mask
                of points cloud. Defaults to None. Defaults to None.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            ret_target (bool): Return targets or not.  Defaults to False.

        Returns:
            dict: Losses of 3DSSD.
        """

        targets = self.get_targets(points, bbox_preds_dict,
                                   batch_gt_instances_3d,
                                   batch_pts_semantic_mask,
                                   batch_pts_instance_mask)
        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask, centerness_weights,
         box_loss_weights, heading_res_loss_weight) = targets

        # calculate centerness loss
        centerness_loss = self.loss_objectness(
            bbox_preds_dict['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights)

        # calculate center loss
        center_loss = self.loss_center(
            bbox_preds_dict['center_offset'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate direction class loss
        dir_class_loss = self.loss_dir_class(
            bbox_preds_dict['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.loss_dir_res(
            bbox_preds_dict['dir_res_norm'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=heading_res_loss_weight)

        # calculate size residual loss
        size_loss = self.loss_size_res(
            bbox_preds_dict['size'],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate corner loss
        one_hot_dir_class_targets = dir_class_targets.new_zeros(
            bbox_preds_dict['dir_class'].shape)
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1),
                                           1)
        pred_bbox3d = self.bbox_coder.decode(
            dict(
                center=bbox_preds_dict['center'],
                dir_res=bbox_preds_dict['dir_res'],
                dir_class=one_hot_dir_class_targets,
                size=bbox_preds_dict['size']))
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = batch_input_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1))

        if self.vote_loss is not None:
            # calculate vote loss
            vote_loss = self.vote_loss(
                bbox_preds_dict['vote_offset'].transpose(1, 2),
                vote_targets,
                weight=vote_mask.unsqueeze(-1))

            losses = dict(
                centerness_loss=centerness_loss,
                center_loss=center_loss,
                dir_class_loss=dir_class_loss,
                dir_res_loss=dir_res_loss,
                size_res_loss=size_loss,
                corner_loss=corner_loss,
                vote_loss=vote_loss)
        else:
             losses = dict(
                centerness_loss=centerness_loss,
                center_loss=center_loss,
                dir_class_loss=dir_class_loss,
                dir_res_loss=dir_res_loss,
                size_res_loss=size_loss,
                corner_loss=corner_loss) 

        return losses
    
    def get_targets(
        self,
        points: List[Tensor],
        bbox_preds_dict: dict = None,
        batch_gt_instances_3d: List[InstanceData] = None,
        batch_pts_semantic_mask: List[torch.Tensor] = None,
        batch_pts_instance_mask: List[torch.Tensor] = None,
    ) -> Tuple[Tensor]:
        """Generate targets of 3DSSD head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            bbox_preds_dict (dict): Bounding box predictions of
                vote head.  Defaults to None.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes`` and ``labels``
                attributes.  Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Semantic gt mask for
                point clouds.  Defaults to None.
            batch_pts_instance_mask (list[tensor]): Instance gt mask for
                point clouds. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Targets of 3DSSD head.
        """
        batch_gt_labels_3d = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        batch_gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]

        # find empty example
        for index in range(len(batch_gt_labels_3d)):
            if len(batch_gt_labels_3d[index]) == 0:
                fake_box = batch_gt_bboxes_3d[index].tensor.new_zeros(
                    1, batch_gt_bboxes_3d[index].tensor.shape[-1])
                batch_gt_bboxes_3d[index] = batch_gt_bboxes_3d[index].new_box(
                    fake_box)
                batch_gt_labels_3d[index] = batch_gt_labels_3d[
                    index].new_zeros(1)

        if batch_pts_semantic_mask is None:
            batch_pts_semantic_mask = [
                None for _ in range(len(batch_gt_labels_3d))
            ]
            batch_pts_instance_mask = [
                None for _ in range(len(batch_gt_labels_3d))
            ]

        aggregated_points = [
            bbox_preds_dict['aggregated_points'][i]
            for i in range(len(batch_gt_labels_3d))
        ]

        seed_points = [
            bbox_preds_dict['seed_points'][i, :self.num_candidates].detach()
            for i in range(len(batch_gt_labels_3d))
        ]

        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask) = multi_apply(
             self.get_targets_single, points, batch_gt_bboxes_3d,
             batch_gt_labels_3d, batch_pts_semantic_mask,
             batch_pts_instance_mask, aggregated_points, seed_points)

        center_targets = torch.stack(center_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets).detach()
        corner3d_targets = torch.stack(corner3d_targets)
        vote_targets = torch.stack(vote_targets)
        vote_mask = torch.stack(vote_mask)

        center_targets -= bbox_preds_dict['aggregated_points']

        centerness_weights = (positive_mask +
                              negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
        centerness_weights = centerness_weights / \
            (centerness_weights.sum() + 1e-6)
        vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * \
            box_loss_weights.unsqueeze(-1)

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask, centerness_weights, box_loss_weights,
                heading_res_loss_weight)


    def forward(self, feat_dict: dict) -> dict:
        """Forward pass.

        Note:
            The forward of convit3dhead uses two steps to predict the 3D boxex:

                1. Predict bbox and score.
                2. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            dict: Predictions of convit3d head.
        """
        aggregated_points, aggregated_features, aggregated_indices = self._extract_input(
            feat_dict)
        
      
        # print("seed points shape", seed_points.shape)
        # print("feature input to head (aggregated_points) shape", aggregated_points.shape)
        
        # # 1. generate vote_points from seed_points
        aggregated_features = aggregated_features.permute(0,2,1)
        
        results = dict(aggregated_features=aggregated_features)
        
        results['aggregated_indices'] = aggregated_indices
        results['aggregated_points'] = aggregated_points
        results['seed_points'] = aggregated_points
       
        
        # temp = torch.rand([4,512,64], device=aggregated_features.device)
        # temp = temp.permute(0,2,1)
        # print("device",aggregated_features.device)
        # print("aggregated_features",aggregated_features.shape)
        # print("temp",temp.shape)

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(aggregated_features)

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                aggregated_points)
        results.update(decode_res)

        # print("Keys", results.keys())
        
        return results


    def get_targets_single(self,
                           points: Tensor,
                           gt_bboxes_3d: BaseInstance3DBoxes,
                           gt_labels_3d: Tensor,
                           pts_semantic_mask: Optional[Tensor] = None,
                           pts_instance_mask: Optional[Tensor] = None,
                           aggregated_points: Optional[Tensor] = None,
                           seed_points: Optional[Tensor] = None,
                           **kwargs):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # Generate fake GT for empty scene
        if valid_gt.sum() == 0:
            vote_targets = points.new_zeros(self.num_candidates, 3)
            center_targets = points.new_zeros(self.num_candidates, 3)
            size_res_targets = points.new_zeros(self.num_candidates, 3)
            dir_class_targets = points.new_zeros(
                self.num_candidates, dtype=torch.int64)
            dir_res_targets = points.new_zeros(self.num_candidates)
            mask_targets = points.new_zeros(
                self.num_candidates, dtype=torch.int64)
            centerness_targets = points.new_zeros(self.num_candidates,
                                                  self.num_classes)
            corner3d_targets = points.new_zeros(self.num_candidates, 8, 3)
            vote_mask = points.new_zeros(self.num_candidates, dtype=torch.bool)
            positive_mask = points.new_zeros(
                self.num_candidates, dtype=torch.bool)
            negative_mask = points.new_ones(
                self.num_candidates, dtype=torch.bool)
            return (vote_targets, center_targets, size_res_targets,
                    dir_class_targets, dir_res_targets, mask_targets,
                    centerness_targets, corner3d_targets, vote_mask,
                    positive_mask, negative_mask)

        gt_corner3d = gt_bboxes_3d.corners

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, aggregated_points)

        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        corner3d_targets = gt_corner3d[assignment]

        top_center_targets = center_targets.clone()
        top_center_targets[:, 2] += size_res_targets[:, 2]
        dist = torch.norm(aggregated_points - top_center_targets, dim=1)
        dist_mask = dist < self.train_cfg.pos_distance_thr
        positive_mask = (points_mask.max(1)[0] > 0) * dist_mask
        negative_mask = (points_mask.max(1)[0] == 0)

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        if self.bbox_coder.with_rot:
            # TODO: Align points rotation implementation of
            # LiDARInstance3DBoxes and DepthInstance3DBoxes
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment],
                axis=2).squeeze(1)
        distance_front = torch.clamp(
            size_res_targets[:, 0] - canonical_xyz[:, 0], min=0)
        distance_back = torch.clamp(
            size_res_targets[:, 0] + canonical_xyz[:, 0], min=0)
        distance_left = torch.clamp(
            size_res_targets[:, 1] - canonical_xyz[:, 1], min=0)
        distance_right = torch.clamp(
            size_res_targets[:, 1] + canonical_xyz[:, 1], min=0)
        distance_top = torch.clamp(
            size_res_targets[:, 2] - canonical_xyz[:, 2], min=0)
        distance_bottom = torch.clamp(
            size_res_targets[:, 2] + canonical_xyz[:, 2], min=0)

        centerness_l = torch.min(distance_front, distance_back) / torch.max(
            distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(
            distance_left, distance_right)
        centerness_h = torch.min(distance_bottom, distance_top) / torch.max(
            distance_bottom, distance_top)
        centerness_targets = torch.clamp(
            centerness_l * centerness_w * centerness_h, min=0)
        centerness_targets = centerness_targets.pow(1 / 3.0)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        proposal_num = centerness_targets.shape[0]
        one_hot_centerness_targets = centerness_targets.new_zeros(
            (proposal_num, self.num_classes))
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(
            1) * one_hot_centerness_targets

        # Vote loss targets
        enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(
            self.train_cfg.expand_dims_length)
        enlarged_gt_bboxes_3d.tensor[:, 2] -= self.train_cfg.expand_dims_length
        vote_mask, vote_assignment = self._assign_targets_by_points_inside(
            enlarged_gt_bboxes_3d, seed_points)

        vote_targets = gt_bboxes_3d.gravity_center
        vote_targets = vote_targets[vote_assignment] - seed_points
        vote_mask = vote_mask.max(1)[0] > 0

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask)

