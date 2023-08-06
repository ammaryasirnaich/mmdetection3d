import numpy as t
import numba
from numba import jit
import matplotlib.pyplot as plt
import torch as t

# Numba JIT-compiled function for the projection
# @jit(nopython=True)
def project_to_birds_eye_view(points_3d, resolution=0.3, height_range=(-3, 1), image_size=(200, 200)):
    x_min = t.min(points_3d[:, 0])
    y_min = t.min(points_3d[:, 1])
    x_range = t.max(points_3d[:, 0]) - x_min
    y_range = t.max(points_3d[:, 1]) - y_min

    x_bins = int(x_range / resolution)
    y_bins = int(y_range / resolution)

    hist = t.zeros((x_bins, y_bins, points_3d.shape[1] - 2), dtype=t.float32, device="cuda")

    for point in points_3d:
        if height_range[0] <= point[2] <= height_range[1]:
            x_index = int((point[0] - x_min) / resolution)
            y_index = int((point[1] - y_min) / resolution)
            if 0 <= x_index < x_bins and 0 <= y_index < y_bins:
                hist[x_index, y_index, :] += point[3:]

    return hist

# Generate a sample 3D point cloud with higher channel dimensions (XYZ + additional attributes)
# num_points = 1000
num_channels = 4  # Adjust the number of channels as needed
# point_cloud = t.random.rand(num_points, num_channels) * t.array([10, 10, 4, 1, 2, 3, 4])  # Adjust the scaling as needed

point_cloud = t.load("/workspace/data/kitti_detection/pointclouds.pt")
b,n,c = point_cloud.shape
point_cloud = point_cloud.reshape(b*n,c)


# print("point_cloud shape", point_cloud.shape)



# # Project the point cloud to birds-eye view image
# birdseye_view = project_to_birds_eye_view(point_cloud)

# birdseye_view = birdseye_view.cpu().numpy()

# print("birdseye_view shape", birdseye_view.shape)

# # Plot the birds-eye view image for each channel
# fig, axes = plt.subplots(1, num_channels - 2, figsize=(15, 3))
# for i in range(num_channels - 2):
#     axes[i].imshow(birdseye_view[:, :, i], cmap='viridis', origin='upper')
#     axes[i].set_title(f'Channel {i + 2}')
# plt.tight_layout()
# plt.show()
# print("Pass")

import numpy as np
import matplotlib.pyplot as plt

def project_to_birds_eye_view_batch(point_clouds_batch, resolution=0.1, height_range=(-2, 2), image_size=(200, 200)):
    batch_size, num_points, num_channels = point_clouds_batch.shape

    x_min = np.min(point_clouds_batch[:, :, 0])
    y_min = np.min(point_clouds_batch[:, :, 1])
    x_range = np.max(point_clouds_batch[:, :, 0]) - x_min
    y_range = np.max(point_clouds_batch[:, :, 1]) - y_min

    x_bins = int(x_range / resolution)
    y_bins = int(y_range / resolution)

    hist_batch = np.zeros((batch_size, x_bins, y_bins, num_channels - 2), dtype=np.float32)

    for batch_idx in range(batch_size):
        for point_idx in range(num_points):
            point = point_clouds_batch[batch_idx, point_idx, :]
            if height_range[0] <= point[2] <= height_range[1]:
                x_index = int((point[0] - x_min) / resolution)
                y_index = int((point[1] - y_min) / resolution)
                if 0 <= x_index < x_bins and 0 <= y_index < y_bins:
                    # Update each channel individually
                    for channel_idx in range(3, num_channels - 1):  # Exclude XYZ and Intensity
                        hist_batch[batch_idx, x_index, y_index, channel_idx - 3] += point[channel_idx]

    return hist_batch

# Generate a batch of sample 3D point clouds with varying additional dimensions
batch_size = 4
num_points = 1000
num_channels = 10  # Adjust the number of channels as needed
point_clouds_batch = np.random.rand(batch_size, num_points, num_channels) * np.array([10, 10, 4, 1, 2, 3, 4, 5, 6, 7])  # Adjust the scaling as needed

# Project the point clouds to birds-eye view images
birdseye_images_batch = project_to_birds_eye_view_batch(point_clouds_batch)

# Plot the birds-eye view images for each batch and channel
for batch_idx in range(batch_size):
    for channel_idx in range(num_channels - 4):  # Exclude XYZ and Intensity
        plt.imshow(birdseye_images_batch[batch_idx, :, :, channel_idx], cmap='viridis', origin='upper')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Batch {batch_idx}, Channel {channel_idx + 3}')
        plt.colorbar(label='Intensity')
        plt.show()
        print("Pass")




