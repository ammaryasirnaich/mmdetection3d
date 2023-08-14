import torch
import numpy as np
import  matplotlib.pyplot as plt
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine import load

# points = np.fromfile('/workspace/mmdetection3d/demo/data/kitti/000008.bin', dtype=np.float32)
# points = points.reshape(-1, 4)
# visualizer = Det3DLocalVisualizer()
# # set point cloud in visualizer
# visualizer.set_points(points)
# bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900, -1.5808]]))
# # Draw 3D bboxes
# visualizer.draw_bboxes_3d(bboxes_3d)
# visualizer.show()
# print("Pause")



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


import pickle

with open('demo/data/kitti/000008.pkl', 'rb') as f:
    info_file = pickle.load(f)


bboxes_3d = []

for instance in info_file['data_list'][0]['instances']:
    bboxes_3d.append(instance['bbox_3d'])

# Define a single RGB color for all bounding boxes (e.g., green)
single_bbox_color = [0, 255, 0]

# Duplicate the color for all bounding boxes
bbox_colors = [single_bbox_color] * len(bboxes_3d)

points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points,pcd_mode=2,vis_mode='add')

# Draw and visualize each bounding box with the duplicated color
for bbox, color in zip(bboxes_3d, bbox_colors):
    visualizer.draw_bboxes_3d(LiDARInstance3DBoxes(torch.tensor(bbox).unsqueeze(0)), np.array([color]))
visualizer.show()