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
import pickle


from mmdet3d.visualization.get3dInstancefrompkl import *


# import mmcv
# import numpy as np
# from mmengine import load

# from mmdet3d.visualization import Det3DLocalVisualizer

# info_file = load('/workspace/mmdetection3d/demo/data/kitti/000008.pkl')
# points = np.fromfile('/workspace/mmdetection3d/demo/data/kitti/000008.bin', dtype=np.float32)
# points = points.reshape(-1, 4)[:, :3]
# lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

# visualizer = Det3DLocalVisualizer()
# img = mmcv.imread('/workspace/mmdetection3d/demo/data/kitti/000008.png')
# img = mmcv.imconvert(img, 'bgr', 'rgb')
# visualizer.set_image(img)
# visualizer.draw_points_on_image(points, lidar2img)
# visualizer.show()



# python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --show --show-dir ${SHOW_DIR}




'''
Example 1 : with out using instance object
'''

# import pickle

# with open('demo/data/kitti/000008.pkl', 'rb') as f:
#     info_file = pickle.load(f)
# imuLidarTrans = info_file['data_list'][0]['lidar_points']['Tr_imu_to_velo']


# bboxes_3d = []

# for instance in info_file['data_list'][0]['instances']:
#     bboxes_3d.append(instance['bbox_3d'])

# # Define a single RGB color for all bounding boxes (e.g., green)
# single_bbox_color = [0, 255, 0]

# # Duplicate the color for all bounding boxes
# bbox_colors = [single_bbox_color] * len(bboxes_3d)

# points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
# points = points.reshape(-1, 4)
# visualizer = Det3DLocalVisualizer()
# # set point cloud in visualizer
# visualizer.set_points(points,vis_mode='add')


# # Draw and visualize each bounding box with the duplicated color
# for bbox, color in zip(bboxes_3d, bbox_colors):
#     visualizer.draw_bboxes_3d(BaseInstance3DBoxes(torch.tensor(bbox).unsqueeze(0)), bbox_color=np.array([color]))
# visualizer.show()


'''
Second example for the 3D Bounding box are disoriented
'''

# points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
# points = points.reshape(-1, 4)
# visualizer = Det3DLocalVisualizer()


# info_file = load('demo/data/kitti/000008.pkl')


# bboxes_3d = []
# labels_3d = []

# gt_instances_3d = InstanceData()

# for instance in info_file['data_list'][0]['instances']:
#     bboxes_3d.append(instance['bbox_3d'])
#     labels_3d.append(instance['bbox_label_3d'])
#     if 'axis_align_matrix' in info_file['data_list'][0]:
#         print("yes its there")

# bbox3d = torch.tensor(bboxes_3d)
# labels_3d = torch.tensor(labels_3d)

# gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox3d)
# gt_instances_3d.labels_3d = labels_3d


# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
# # gt_det3d_data_sample.metainfo = 

# data_input = dict(points=points)

# visualizer.add_datasample('3D Scene', data_input,
#                                    gt_det3d_data_sample,
#                                     vis_task='lidar_det')

# visualizer.show()
# print("pass")



'''

###using the config file for consistance with the help
###of pcd demo file



points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()


info_file = load('demo/data/kitti/000008.pkl')


bboxes_3d = []
labels_3d = []

gt_instances_3d = InstanceData()

for instance in info_file['data_list'][0]['instances']:
    bboxes_3d.append(instance['bbox_3d'])
    labels_3d.append(instance['bbox_label_3d'])

bbox3d = torch.tensor(bboxes_3d)
labels_3d = torch.tensor(labels_3d)

# convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
lidar2cam = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2cam'])
# print("lidar2cam",lidar2cam)
               

gt_instances_3d.bboxes_3d = CameraInstance3DBoxes(bbox3d).convert_to(Box3DMode.LIDAR,
                                                 np.linalg.inv(lidar2cam))
gt_instances_3d.labels_3d = labels_3d


gt_det3d_data_sample = Det3DDataSample()
gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
data_input = dict(points=points)

visualizer.add_datasample('3D Scene', data_input,
                                   gt_det3d_data_sample,
                                    vis_task='lidar_det')

visualizer.show()




# info_file = load('demo/data/kitti/kitti_infos_test.pkl')   # ['metainfo', 'data_list'
# info_file = load('demo/data/kitti/kitti_infos_train.pkl')   # ['metainfo', 'data_list'
# info_file = load('demo/data/kitti/000008.pkl')


'''


'''
## for single object displya



def get_3dbbox_from_pklfile(file:str):
    info_file = load(file)   # ['metainfo', 'data_list'
     

    # Traverse the data to extract 3D bounding box information
    print(info_file['data_list'][0].keys()) 

    bboxes_3d = []  

    ### sample_idx', 'images', 'lidar_points', 'instances', 'cam_instances
    for info_dic in info_file['data_list']:
            # print(type(info_dic))
            lidar_info_str= info_dic['lidar_points']['lidar_path']
            # print(type(lidar_info_dic))
            # print(lidar_info_str) 
            if(lidar_info_str == '000003.bin'):

                # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
                lidar2cam = np.array(info_dic['images']['CAM2']['lidar2cam'])
                print("lidar2cam",lidar2cam)
               
                # print(type(info_dic['instances']))
                # print(len(info_dic['instances']))
                for instance_dic in info_dic['instances']:
                    # print(len(instance_dic['bbox_3d']))
                    # print(instance_dic['bbox_3d'])
                    # box = torch.tensor([[13.22, 4.15, -1.5997, 4.15, 1.57, 1.73, 3.1]])
                    # print(box)
                    box = torch.tensor([instance_dic['bbox_3d']])

                    # print(box)
                    gt_bboxes_3d = CameraInstance3DBoxes(box).convert_to(Box3DMode.LIDAR,
                                                 np.linalg.inv(lidar2cam))
                    # print("after conversion")
                    # print(gt_bboxes_3d)
   
                        
                    bboxes_3d.append(gt_bboxes_3d)
                    # print(instance_dic['bbox_3d'])
                ## break after reading a single data instancce in the if conidition              
                break
                
    return  bboxes_3d


'''



'''

    
import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

# filename ='/workspace/data/kitti_detection/kitti/kitti_infos_test.pkl'  # ['metainfo', 'data_list'
filename ='/workspace/data/kitti_detection/kitti/kitti_infos_train.pkl'  # ['metainfo', 'data_list'
# filename = 'demo/data/kitti/000008.pkl'

# points = np.fromfile('/workspace/data/kitti_detection/kitti/training/velodyne_reduced/000002.bin', dtype=np.float32)
points = np.fromfile('/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)

gt_bboxes_3d= get_3dbbox_from_pklfile(filename)


print("No of entries",len(gt_bboxes_3d))
print(gt_bboxes_3d)


# gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
# print("After converting using DepthInstance3DBoxes:")
# print(gt_bboxes_3d)

bbox_color = [(0,225,0)]*len(gt_bboxes_3d)
# # Draw 3D bboxes
visualizer.draw_bboxes_3d(gt_bboxes_3d[0],bbox_color)
visualizer.show()
'''




'''
import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

points = np.fromfile('/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
## set point cloud in visualizer
visualizer.set_points(points)

# center_obj_point = np.random.rand(1, 4)
# center_obj_point = np.array([(13.22, 4.15, -1.5997,0)])

# visualizer.set_points(center_obj_point,points_color=(255,0,0),points_size=10)

info = data_infos[index]
rect = info['calib']['R0_rect'].astype(np.float32)
Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

gt_bboxes_3d = torch.tensor([[13.22, 4.15, -1.5997, 4.15, 1.57, 1.73, 3.1]])
gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            'LiDAR', np.linalg.inv(rect @ Trv2c))

# print(bboxes_3d)

# bboxes_3d = LiDARInstance3DBoxes(  # x , y ,z ,h ,w, l, radin
#     ## torch.tensor([[1.0, 1.75, 13.22, 4.15, 1.57, 1.73, 1.62]]))
#       torch.tensor([[13.22, 4.15, -1.5997, 4.15, 1.57, 1.73, 3.1]]))
                    

# bbox_color = [(0,225,0)]*len(bboxes_3d)
# # Draw 3D bboxes
# visualizer.draw_bboxes_3d(bboxes_3d,bbox_color)
visualizer.show()

'''





'''
Testing get3dInstance_frompkl
'''
import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()

lidarinstanceName= '000008.bin'
gt_instances_3d = get_3dInstance_from_pklfile(lidarinstanceName)


gt_det3d_data_sample = Det3DDataSample()
gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
data_input = dict(points=points)

visualizer.add_datasample('3D Scene', data_input,
                                   gt_det3d_data_sample,
                                    vis_task='lidar_det')

visualizer.show()
