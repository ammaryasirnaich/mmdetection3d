import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)

from mmdet3d.apis.inference import *

import open3d as o3d





def get_pointcloud(point:np):
    pcd = o3d.geometry.PointCloud()
    xyz = point[:, :3]
    
    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Optionally, add colors based on reflectance values or other criteria
    # Here we simply use a grayscale color based on reflectance
    reflectance = point[:, 3]
    # colors = np.zeros((reflectance.shape[0], 3))
    # colors[:, 0] = colors[:, 1] = colors[:, 2] = reflectance / reflectance.max()
    colors = [[0.5, 0.5, 0.5] for i in range(points.shape[0])]
    pcd.colors = o3d.utility.Vector3dVector(colors)
        
    return pcd    



# pcd_file = "/workspace/data/kitti_detection/kitti/training/velodyne/000173.bin",

# config = "/workspace/mmdetection3d/work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py"
# checkpoint ="/workspace/mmdetection3d/work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-3class/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"



# config='/workspace/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-car.py'
# checkpoint='/workspace/data/kitti_detection/models_to_test/3dssd/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth'

rootpath="/workspace/data/kitti_detection/model_output_results/convit3D_kitti_24_June_2024/"
config=rootpath+'convit3D_pointNet_transformer_kitti.py'
checkpoint=rootpath+'epoch_80.pth'



file ='/workspace/mmdetection3d/demo/data/kitti/000008.bin'
pcd_file ='/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin'
points = np.fromfile(file, dtype=np.float32).reshape(-1,4)
model = init_model(config,checkpoint)
# _,data = inference_detector(model,points)


# num_gt_instance = 2
# packed_inputs = create_detector_inputs(
# num_points=10, num_gt_instance=num_gt_instance)
# packed_inputs['points'] = [points]


# if torch.cuda.is_available():
#     model = model.cuda()
#     # test simple_test
#     with torch.no_grad():
#         data = model.data_preprocessor(packed_inputs, True)
#         torch.cuda.empty_cache()
#         results = model.forward(**data, mode='predict')

# print(model)



# # If you want to save the visualization to a file
# o3d.io.write_point_cloud("attention_map.pcd", points)

# Visualize the point cloud
vis = o3d.visualization.Visualizer()
pcd = get_pointcloud(points)
# vis.draw_geometries([get_pointcloud(points)], window_name="Attention Map Visualization")
vis.create_window()
vis.add_geometry(pcd)
# vis.get_render_option().point_size = 2
vis.run()

    



