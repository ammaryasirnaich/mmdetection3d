import numpy as np
import torch as t
import numba
from numba import cuda
from numba import njit
import matplotlib.pyplot as plt
import cupy as cp

def load_kitti_pointcloud(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # Assumes XYZI format
    return points



def visualize_birds_eye_view_batch(birdseye_images_batch):
    batch_size = birdseye_images_batch.shape[0]
    height = birdseye_images_batch.shape[1]
    width = birdseye_images_batch.shape[2]
    num_channels = birdseye_images_batch.shape[3]
    
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            plt.imshow(birdseye_images_batch[batch_idx, :, :, channel_idx], cmap='viridis', origin='upper', vmin=0, vmax=1)  # Adjust vmin and vmax as needed
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Batch {batch_idx}, Channel {channel_idx}')
            plt.colorbar(label='Intensity')
            plt.show()
            print("pass")


def project_to_birds_eye_view_batch(point_clouds_batch, resolution=0.2, height_range=(-2, 2), image_size=(200, 200)):
    
    batch_size,num_points,num_channels = point_clouds_batch.shape
    x_min = t.min(point_clouds_batch[:, :, 0])
    y_min = t.min(point_clouds_batch[:, :, 1])
    x_range = t.max(point_clouds_batch[:, :, 0]) - x_min
    y_range = t.max(point_clouds_batch[:, :, 1]) - y_min

    x_bins = int(x_range / resolution)
    y_bins = int(y_range / resolution)

    hist_batch = t.zeros((batch_size, image_size[0], image_size[1], num_channels), dtype=t.float32,device='cuda')

    for batch_idx in range(batch_size):
        for point_idx in range(num_points):
            point = point_clouds_batch[batch_idx, point_idx, :]
            if height_range[0] <= point[2] <= height_range[1]:
                x_index = int((point[0] - x_min) / resolution)
                y_index = int((point[1] - y_min) / resolution)
                if 0 <= x_index < image_size[0] and 0 <= y_index < image_size[1]:
                    intensity_channels = point[3:]  # Exclude XYZ
                    hist_batch[batch_idx, x_index, y_index, :] = intensity_channels

    return hist_batch




batch_size = 4
num_channels = 4  # Adjust based on your data
point_clouds_batch = np.array([load_kitti_pointcloud('demo/data/kitti/000008.bin')] * batch_size)
point_clouds_batch = t.from_numpy(point_clouds_batch).cuda()

resolution = 0.2
height_range = (-3, 1)
image_size = (200, 200)

# Project the point clouds to bird's eye view images0
birdseye_images_batch = project_to_birds_eye_view_batch(point_clouds_batch, resolution, height_range, image_size=image_size)
# birdseye_images_batch = t.as_tensor( cp.asanyarray(birdseye_images_batch) )
print("birdseye_images_batch size", birdseye_images_batch.shape)

# Visualize the bird's eye view images for each batch and channel
visualize_birds_eye_view_batch(birdseye_images_batch.cpu())


print("pass")



