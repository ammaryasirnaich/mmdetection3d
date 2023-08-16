import torch
import numpy as np
import  matplotlib.pyplot as plt
from mmdet3d.visualization import Det3DLocalVisualizer
from mmengine import load

from mmengine.structures import InstanceData
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from mmdet3d.visualization.vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode)
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)




from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS


config="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead__14_August/convit3D_pointnet_transformer_ssdhead.py"
checkpoint="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead__14_August/epoch_80.pth"
file_name ='/workspace/mmdetection3d/demo/data/kitti/000008.bin' 

model = init_model(config, checkpoint, device='cuda:0')

    # init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# test a single point cloud sample
result, data = inference_detector(model, file_name)
points = data['inputs']['points']
data_input = dict(points=points)




# show the results
visualizer.add_datasample(
    'result',
    data_input,
    data_sample=result,
    draw_gt=False,
    show=True,
    wait_time=-1,
    out_file=None,
    pred_score_thr=0.3,
    vis_task='lidar_det')
