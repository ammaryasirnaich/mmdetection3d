from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample
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
        
  

    def _extract_input(self, feat_dict: dict) -> Tuple:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['sa_xyz'][-1]
        seed_features = feat_dict['sa_features'][-1]
        seed_indices = feat_dict['sa_indices'][-1]

        self.num_candidates = seed_points.shape[1]

        return seed_points, seed_features, seed_indices
    


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

        seed_points, seed_features, seed_indices = self._extract_input(
            feat_dict)

        # # 1. generate vote_points from seed_points
        # vote_points, vote_features, vote_offset = self.vote_module(
        #     seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            seed_features=seed_features)

 
        results['aggregated_points'] = seed_points
        results['aggregated_features'] = seed_features
        results['aggregated_indices'] = seed_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(seed_features)

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                seed_points)
        results.update(decode_res)

        print("Keys", results.keys())
        return results


        
