import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


# Register the ResNet module in the configuration file
from mmdet.models.backbones import ResNet
MODELS.register_module(module=ResNet)


def test_center_point():
    import mmdet3d.models

    # assert hasattr(mmdet3d.models, 'HFFModel')

    setup_seed(0)
    DefaultScope.get_instance('test_hff', scope_name='mmdet3d')
    hff_net_cfg = get_detector_cfg(
        'hff/hff_base_config.py'  # noqa
    )
    model = MODELS.build(hff_net_cfg)
    # assert hasattr(mmdet3d.models, 'HFFModel')
    num_gt_instance = 50
    packed_inputs = create_detector_inputs(
        with_img=True,with_points=True, num_gt_instance=num_gt_instance, points_feat_dim=5)
    
    
    

    for sample_id in range(len(packed_inputs['data_samples'])):
        det_sample = packed_inputs['data_samples'][sample_id]
        num_instances = len(det_sample.gt_instances_3d.bboxes_3d)
        bbox_3d_class = det_sample.gt_instances_3d.bboxes_3d.__class__
        det_sample.gt_instances_3d.bboxes_3d = bbox_3d_class(
            torch.rand(num_instances, 9), box_dim=9)

    if torch.cuda.is_available():

        model = model.cuda()
        # test simple_test

        batch_input_dict = model.data_preprocessor(packed_inputs, True)
        with torch.no_grad():
            torch.cuda.empty_cache()
            losses = model.forward(**batch_input_dict, mode='loss')
        # assert losses['task0.loss_heatmap'] >= 0
        # assert losses['task0.loss_bbox'] >= 0
        # assert losses['task1.loss_heatmap'] >= 0
        # assert losses['task1.loss_bbox'] >= 0
        # assert losses['task2.loss_heatmap'] >= 0
        # assert losses['task2.loss_bbox'] >= 0
        # assert losses['task3.loss_heatmap'] >= 0
        # assert losses['task3.loss_bbox'] >= 0
        # assert losses['task3.loss_bbox'] >= 0
        # assert losses['task4.loss_bbox'] >= 0
        # assert losses['task5.loss_heatmap'] >= 0
        # assert losses['task5.loss_bbox'] >= 0

        with torch.no_grad():
            results = model.forward(**data, mode='predict')
        # self.assertEqual(len(results), 1)
        # self.assertIn('bboxes_3d', results[0].pred_instances_3d)
        # self.assertIn('scores_3d', results[0].pred_instances_3d)
        # self.assertIn('labels_3d', results[0].pred_instances_3d)
    # TODO test_aug_test


if __name__ =="__main__":
    test_center_point()
    