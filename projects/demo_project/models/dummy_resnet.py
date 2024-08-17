from mmdet.models.backbones import ResNet

from mmdet3d.registry import MODELS

from mmdet3d.models import MVXTwoStageDetector


@MODELS.register_module()
class DummyTest(ResNet):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('DummyTest called!')
        super().__init__(**kwargs)


@MODELS.register_module()
class SparseOcc(MVXTwoStageDetector):
    print('SparseOcc called world!')
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_aug=None,
                 use_mask_camera=False,
                 **kwargs):

        super(SparseOcc, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.use_mask_camera = use_mask_camera
        self.fp16_enabled = False
        self.data_aug = data_aug

        self.memory = {}
        
