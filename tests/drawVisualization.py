import torch
import numpy as np

# from mmdet3d.visualization import Det3DLocalVisualizer
# from mmdet3d.structures import LiDARInstance3DBoxes

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


# import torch
# import mmcv
# from mmengine.visualization import Visualizer

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