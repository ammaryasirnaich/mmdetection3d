import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)

from mmdet3d.apis.inference import *

# pcd_file = "/workspace/data/kitti_detection/kitti/training/velodyne/000173.bin",
config = "/workspace/mmdetection3d/work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py"
checkpoint ="/workspace/mmdetection3d/work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"

# config='/workspace/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-car.py'
# checkpoint='/workspace/data/kitti_detection/models_to_test/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth'
# file ='/workspace/mmdetection3d/demo/data/kitti/000008.bin'
pcd_file ='/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin'
points = np.fromfile(pcd_file).reshape(-1,4)
model = init_model(config,checkpoint)
results, data = inference_detector(model,points)





print("Pass")




