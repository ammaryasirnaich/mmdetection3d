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
class Convit3DHeadOld(SSD3DHead):
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
        aggregated_features = feat_dict['sa_features'][-1]
        aggregated_indices = feat_dict['sa_indices'][-1]
        seeds = feat_dict['raw_points']

        self.num_candidates = aggregated_points.shape[1]

        return aggregated_points, aggregated_features, aggregated_indices
    


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
        aggregated_points, aggregated_features, seed_points = self._extract_input(
            feat_dict)
        
      
        # print("seed points shape", seed_points.shape)
        # print("feature input to head (aggregated_points) shape", aggregated_points.shape)
        
        # # 1. generate vote_points from seed_points
        # vote_points, vote_features, vote_offset = self.vote_module(
        #     seed_points, seed_features)

        results = dict(seed_points=seed_points)

        # results = dict(
        #     seed_points=seed_points,
        #     seed_indices=seed_indices,
        #     seed_features=seed_features)

 
        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = aggregated_features
        # results['aggregated_indices'] = aggregated_indices
        print("aggregated_features",aggregated_features.shape)


        aggregated_features = aggregated_features.permute(0,2,1)

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