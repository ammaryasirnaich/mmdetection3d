import unittest
import torch
from mmengine import DefaultScope
from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg, setup_seed)

class TestPointViT(unittest.TestCase):

    def test_pointvit(self):
        import mmdet3d.models

        assert hasattr(mmdet3d.models, 'ConVit3D')
        DefaultScope.get_instance('test_ConVit3D', scope_name='mmdet3d')
        setup_seed(0)
        voxel_net_cfg = get_detector_cfg('3dconvit/convit3D_pointNet_transformer_kitti.py')
        model = MODELS.build(voxel_net_cfg)
        num_gt_instance = 3
        packed_inputs = create_detector_inputs(
            num_gt_instance=num_gt_instance, num_classes=3)

        if torch.cuda.is_available():
            model = model.cuda()
            # test simple_test
            with torch.no_grad():
                data = model.data_preprocessor(packed_inputs, True)
                torch.cuda.empty_cache()
                results = model.forward(**data, mode='predict')
                num_layers = len(model.neck.blocks)
                last_layer = model.neck.blocks[-1]
                print("type of last layer",{type(last_layer)})
                "model.neck.blocks[-1] last layer"
                print(f"number of layers:",{num_layers})

            losses = model.forward(**data, mode='loss')
            self.assertGreater(losses['centerness_loss'], 0)

if __name__ == '__main__':
    unittest.main()
