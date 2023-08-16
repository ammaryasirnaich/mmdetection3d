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


# # https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/en/_static/image/cat_and_dog.png
# image = mmcv.imread('/workspace/mmengine/docs/en/_static/image/cat_and_dog.png',
#                     channel_order='rgb')
# visualizer = Visualizer(image=image)
# # single bbox formatted as [xyxy]
# visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]))
# # draw multiple bboxes
# visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
# visualizer.show()
# print("End")



# import mmcv
# from mmengine.visualization import Visualizer

# tensor = torch.load('./bev_image_0.pt')
# tensor = torch.load('./second_fpn0.pt')
# b,c,h,w = tensor.shape
# print("second_fpn shape:",tensor.shape)
# print( tensor.shape)
# img = tensor[0]

# img = img.permute(1,2,0).detach().cpu().numpy()
# img = img.mean(2)
# print(img.shape, img.min(), img.max())
# print(img[:,:,:3])
# img = img-img.min()
# img = img/img.max()
# plt.imshow(img)
# print("Pass")



# select a sample from the batch
# img = x[0]
# permute to match the desired memory format
# img = img.permute(1, 2, 0).numpy()
# print(img[:,:,:3])
# plt.imshow(img)



'''
Example 1 : with out using instance object
'''

# import pickle

# with open('demo/data/kitti/000008.pkl', 'rb') as f:
#     info_file = pickle.load(f)


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
#     visualizer.draw_bboxes_3d(LiDARInstance3DBoxes(torch.tensor(bbox).unsqueeze(0)), np.array([color]))
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

# bbox3d = torch.tensor(bboxes_3d)
# labels_3d = torch.tensor(labels_3d)

# gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox3d)
# gt_instances_3d.labels_3d = labels_3d

# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
# data_input = dict(points=points)

# visualizer.add_datasample('3D Scene', data_input,
#                                    gt_det3d_data_sample,
#                                     vis_task='lidar_det')

# visualizer.show()
# print("pass")





'''

using the config file for consistance with the help
of pcd demo file
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

# bbox3d = torch.tensor(bboxes_3d)
# labels_3d = torch.tensor(labels_3d)

# # gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox3d)
# # gt_instances_3d.labels_3d = labels_3d

# points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)

# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
# data_input = dict(points=points)

# visualizer.add_datasample('3D Scene', data_input,
#                                    gt_det3d_data_sample,
#                                     vis_task='lidar_det')

# visualizer.show()


from webcolors import name_to_rgb
color_name = 'green'

try:
    rgb_tuple = name_to_rgb(color_name)
    print(f"The RGB values for {color_name} are: {rgb_tuple}")
except ValueError:
    print("Color not found.")