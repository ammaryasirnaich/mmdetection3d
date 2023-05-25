from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.ops import furthest_point_sample
from mmdet.models.utils import multi_apply
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.layers import VoteModule, aligned_3d_nms, build_sa_module
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample
from .base_conv_bbox_head import BaseConvBboxHead

from .vote_head import VoteHead
from .ssd_3d_head import SSD3DHead

@MODELS.register_module()
class Convit3DHead(SSD3DHead):
    r"""Bbox head of `Convit3DHead .
    Args:
        num_classes (int): The number of class.
        bbox_coder (ConfigDict, dict): Bbox coder for encoding and
            decoding boxes. Defaults to None.
        train_cfg (dict, optional): Config for training. Defaults to None.
        test_cfg (dict, optional): Config for testing. Defaults to None.
        pred_layer_cfg (dict, optional): Config of classification
            and regression prediction layers. Defaults to None.
        objectness_loss (dict, optional): Config of objectness loss.
            Defaults to None.
        center_loss (dict, optional): Config of center loss.
            Defaults to None.
        dir_class_loss (dict, optional): Config of direction
            classification loss. Defaults to None.
        dir_res_loss (dict, optional): Config of direction
            residual regression loss. Defaults to None.
        size_class_loss (dict, optional): Config of size
            classification loss. Defaults to None.
        size_res_loss (dict, optional): Config of size
            residual regression loss. Defaults to None.
        semantic_loss (dict, optional): Config of point-wise
            semantic segmentation loss. Defaults to None.
        iou_loss (dict, optional): Config of IOU loss for
            regression. Defaults to None.
        init_cfg (dict, optional): Config of model weight
            initialization. Defaults to None.
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
                 size_class_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 corner_loss: Optional[dict] = None,
                 iou_loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(Convit3DHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_objectness = MODELS.build(objectness_loss)
        self.loss_center = MODELS.build(center_loss)
        self.loss_dir_res = MODELS.build(dir_res_loss)
        self.loss_dir_class = MODELS.build(dir_class_loss)
        self.loss_size_res = MODELS.build(size_res_loss)
        if size_class_loss is not None:
            self.size_class_loss = MODELS.build(size_class_loss)
        if iou_loss is not None:
            self.iou_loss = MODELS.build(iou_loss)
        else:
            self.iou_loss = None
        self.corner_loss = MODELS.build(corner_loss)

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.fp16_enabled = False

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())
    
    def forward(self, feat_dict: dict) -> dict:


        print("Keys", feat_dict.keys())
        print("pass")
        return super().forward(feat_dict)

 