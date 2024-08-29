import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)


class TestSDF():

    def test_3dssd(self):
        import mmdet3d.models

        # assert hasattr(mmdet3d.models, 'SSD3DNet')
        DefaultScope.get_instance('test_SDF', scope_name='mmdet3d')
        setup_seed(0)
        
    #     hff_net_cfg = get_detector_cfg(
    #  'hff/hff_base_config.py'  # noqa
    #     )
    #     model = MODELS.build(hff_net_cfg)
        
        sdf_net_cfg = get_detector_cfg('hff/hff_base_config.py')
        model = MODELS.build(sdf_net_cfg)
        num_gt_instance = 5
        packed_inputs = create_detector_inputs(with_img=True,with_points=True,
            num_gt_instance=num_gt_instance, num_classes=10)

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                torch.cuda.empty_cache()
                # results = model.forward(**data, mode='predict')
                results = model.forward(**data, mode='loss')
            # self.assertEqual(len(results), 1)
            # self.assertIn('bboxes_3d', results[0].pred_instances_3d)
            # self.assertIn('scores_3d', results[0].pred_instances_3d)
            # self.assertIn('labels_3d', results[0].pred_instances_3d)

            # losses = model.forward(**data, mode='loss')

            # self.assertGreater(losses['centerness_loss'], 0)

if __name__ == "__main__":
    model_test = TestSDF()
    model_test.test_3dssd()