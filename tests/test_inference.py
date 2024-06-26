import unittest

import torch
from mmengine import DefaultScope

from mmdet3d.registry import MODELS
from mmdet3d.testing import (create_detector_inputs, get_detector_cfg,
                             setup_seed)

from mmdet3d.apis.inference import *

import open3d as o3d
import matplotlib.pyplot as plt



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



def get_pointcloud(point:np):
    pcd = o3d.geometry.PointCloud()
    xyz = point[:, :3]
    
    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
     ###@ giving color based on intensity
    intensity = points[:, 3]

    # Normalize intensity values to the range [0, 1]
    normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

    # Map normalized intensity values to colors using a colormap (e.g., 'plasma' colormap)
    color_map = plt.get_cmap('plasma')
    points_colors = color_map(normalized_intensity)[:, :3]  # Exclude alpha channel

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    return pcd
   
def demo_point_visualize():
    import open3d as o3d
    import numpy as np

    # Example point cloud data (randomly generated for demonstration)
    num_points = 10
    points = np.random.rand(num_points, 3)
    sizes = np.random.rand(num_points) * 0.05  # Random sizes for demonstration

    # Create an empty geometry list to store the spheres
    geometries = []

    # Create a sphere for each point with a specific size
    for i in range(num_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sizes[i])
        sphere.translate(points[i])
        sphere.paint_uniform_color([1, 0, 0])  # Optional: color the spheres red
        geometries.append(sphere)

    # Visualize all the spheres (representing points with different sizes)
    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with Variable Sizes")
    


if __name__ == "__main__":
    points = np.fromfile(file, dtype=np.float32).reshape(-1,4)
    # model = init_model(config,checkpoint)
    # _,data = inference_detector(model,points)


    # num_gt_instance = 2
    # packed_inputs = create_detector_inputs(
    # num_points=10, num_gt_instance=num_gt_instance)
    # packed_inputs['points'] = [points]

    # points = torch.from_numpy(points).to('cuda:0')
    # print(type(points))
    # print(points.is_cuda)
    # print(points.shape)
    
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     # test simple_test
    #     with torch.no_grad():
    #         data = model.data_preprocessor(packed_inputs, True)
    #         pnt = data['inputs']['points']
    #         points = data['inputs']['points']
    #         print(len(points))
    #         stack_points = torch.stack(points)
    #         print(stack_points.shape)
    #         print(type(stack_points))
    #         print(stack_points.is_cuda)
    #         torch.cuda.empty_cache()
            # results = model.forward(**data, mode='predict')
            

        # x = self.backbone(stack_points)
        # points = torch.stack(packed_inputs['points'])
        # x = model.backbone(points)
        # print("shape of backbone:", x.shape)    
        # if self.with_neck:
            # x = self.neck(x)

    # print(model)
    demo_point_visualize()


    # # Visualize the point cloud
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd_raw = get_pointcloud(points)
    # # vis.draw_geometries([get_pointcloud(points)], window_name="Attention Map Visualization")
    # vis.add_geometry(pcd_raw)

    # vis.get_render_option().point_size = 2
    # vis.run()

    



