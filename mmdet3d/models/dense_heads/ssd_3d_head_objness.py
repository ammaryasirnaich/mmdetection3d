from __future__ import annotations

from typing import List, Optional

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads.ssd_3d_head import SSD3DHead


@MODELS.register_module()
class SSD3DHeadObjness(SSD3DHead):
    """SSD3DHead variant with explicit objectness targets for `obj_scores`.

    Motivation:
    - In the current `SSD3DHead`, `obj_scores` is supervised by `centerness_targets`.
      In your logs this produces a near-zero `centerness_loss` (~1e-6), meaning
      the score head receives little useful gradient.

    Change:
    - Supervise `obj_scores` with a *binary objectness target* (1 for the assigned
      class at positive proposals, 0 elsewhere), while using centerness as a
      *weight* for positives (negatives keep weight 1).

    Notes:
    - This keeps all other losses identical to `SSD3DHead`.
    - Use via `custom_imports` in a config (no edits to existing files needed).
    """

    def loss_by_feat(
        self,
        points: List[torch.Tensor],
        bbox_preds_dict: dict,
        batch_gt_instances_3d: List[InstanceData],
        batch_pts_semantic_mask: Optional[List[torch.Tensor]] = None,
        batch_pts_instance_mask: Optional[List[torch.Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None,
        ret_target: bool = False,
        **kwargs,
    ) -> dict:
        targets = self.get_targets(
            points,
            bbox_preds_dict,
            batch_gt_instances_3d,
            batch_pts_semantic_mask,
            batch_pts_instance_mask,
        )

        (
            vote_targets,
            center_targets,
            size_res_targets,
            dir_class_targets,
            dir_res_targets,
            mask_targets,
            centerness_targets,
            corner3d_targets,
            vote_mask,
            positive_mask,
            negative_mask,
            _centerness_weights,  # not used (we build our own)
            box_loss_weights,
            heading_res_loss_weight,
        ) = targets

        # ------------------------------------------------------------
        # Objectness supervision (replaces centerness-as-target)
        # ------------------------------------------------------------
        # NOTE: For `AnchorFreeBBoxCoder.split_pred`, `obj_scores` is kept as
        # the raw classification conv output: (B, C, N).
        # The loss in upstream heads uses `obj_scores.transpose(2, 1)` to get
        # (B, N, C). We follow that convention here to avoid shape bugs.
        obj_scores: Tensor = bbox_preds_dict["obj_scores"]  # (B, C, N)
        obj_scores_bnC = obj_scores.transpose(2, 1)  # (B, N, C)
        B, N, C = obj_scores_bnC.shape

        obj_targets = obj_scores.new_zeros((B, N, C))
        if positive_mask.any():
            b_idx, n_idx = positive_mask.nonzero(as_tuple=True)
            cls_idx = mask_targets[b_idx, n_idx].long().clamp(min=0, max=C - 1)
            obj_targets[b_idx, n_idx, cls_idx] = 1.0

        # weights:
        # - negatives: weight 1
        # - positives: weight = centerness (only on the assigned class channel)
        neg_w = negative_mask.float().unsqueeze(-1).expand(-1, -1, C)
        # `centerness_targets` is (B, N, num_classes) with non-zero only on the
        # assigned class channel; also guard to only weight true positives.
        pos_w = (
            centerness_targets.detach().clamp(min=0, max=1)
            * positive_mask.float().unsqueeze(-1)
        )
        obj_weights = neg_w + pos_w
        obj_weights = obj_weights / (obj_weights.sum() + 1e-6)

        objectness_loss = self.loss_objectness(
            obj_scores_bnC, obj_targets, weight=obj_weights
        )

        # ------------------------------------------------------------
        # Remaining losses (same as SSD3DHead)
        # ------------------------------------------------------------
        center_loss = self.loss_center(
            bbox_preds_dict["center_offset"],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1),
        )

        dir_class_loss = self.loss_dir_class(
            bbox_preds_dict["dir_class"].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights,
        )

        dir_res_loss = self.loss_dir_res(
            bbox_preds_dict["dir_res_norm"],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=heading_res_loss_weight,
        )

        size_loss = self.loss_size_res(
            bbox_preds_dict["size"],
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1),
        )

        # Corner loss: decode boxes and compare corners
        one_hot_dir_class_targets = dir_class_targets.new_zeros(
            bbox_preds_dict["dir_class"].shape
        )
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        pred_bbox3d = self.bbox_coder.decode(
            dict(
                center=bbox_preds_dict["center"],
                dir_res=bbox_preds_dict["dir_res"],
                dir_class=one_hot_dir_class_targets,
                size=bbox_preds_dict["size"],
            )
        )
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = batch_input_metas[0]["box_type_3d"](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5),
        )
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1),
        )

        losses = dict(
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss,
            corner_loss=corner_loss,
        )

        if self.vote_loss is not None:
            vote_loss = self.vote_loss(
                bbox_preds_dict["vote_offset"].transpose(1, 2),
                vote_targets,
                weight=vote_mask.unsqueeze(-1),
            )
            losses["vote_loss"] = vote_loss

        if ret_target:
            losses["targets"] = targets

        return losses

