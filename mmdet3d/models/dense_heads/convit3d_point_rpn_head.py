# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.structures.bbox_3d import (BaseInstance3DBoxes,
                                        DepthInstance3DBoxes,
                                        LiDARInstance3DBoxes)
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import InstanceList
from .point_rpn_head import PointRPNHead

@MODELS.register_module()
class ConVit3DRPNHead(PointRPNHead):
    """RPN module for PointRCNN.

    Args:
        num_classes (int): Number of classes.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        pred_layer_cfg (dict, optional): Config of classification and
            regression prediction layers. Defaults to None.
        enlarge_width (float, optional): Enlarge bbox for each side to ignore
            close points. Defaults to 0.1.
        cls_loss (dict, optional): Config of direction classification loss.
            Defaults to None.
        bbox_loss (dict, optional): Config of localization loss.
            Defaults to None.
        bbox_coder (dict, optional): Config dict of box coders.
            Defaults to None.
        init_cfg (dict, optional): Config of initialization. Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 train_cfg: dict,
                 test_cfg: dict,
                 pred_layer_cfg: Optional[dict] = None,
                 enlarge_width: float = 0.1,
                 cls_loss: Optional[dict] = None,
                 bbox_loss: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.enlarge_width = enlarge_width

        # build loss function
        self.bbox_loss = MODELS.build(bbox_loss)
        self.cls_loss = MODELS.build(cls_loss)

        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        # build pred conv
        self.cls_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.cls_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_cls_out_channels())

        self.reg_layers = self._make_fc_layers(
            fc_cfg=pred_layer_cfg.reg_linear_channels,
            input_channels=pred_layer_cfg.in_channels,
            output_channels=self._get_reg_out_channels())
    

    def forward(self, feat_dict: dict) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            tuple[list[torch.Tensor]]: Predicted boxes and classification
                scores.
        """
        point_features = feat_dict['sa_features'][-1]

        batch_size = point_features.shape[0]
        feat_cls = point_features.view(-1, point_features.shape[-1])
        feat_reg = point_features.view(-1, point_features.shape[-1])
        
        point_cls_preds = self.cls_layers(feat_cls).reshape(
            batch_size, -1, self._get_cls_out_channels())
        
        point_box_preds = self.reg_layers(feat_reg).reshape(
            batch_size, -1, self._get_reg_out_channels())
        return point_box_preds, point_cls_preds
