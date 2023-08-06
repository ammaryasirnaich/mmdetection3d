import torch as t
import numba
# from numba import jit
import matplotlib.pyplot as plt

# Numba JIT-compiled function for the projection
# @jit(nopython=True)
def bev_3D_to_2D(points_3d, resolution=0.1, height_range=(-3, 1), image_size=(800, 800)):
    x_min = t.min(points_3d[:, 0])
    y_min = t.min(points_3d[:, 1])
    x_range = t.max(points_3d[:, 0]) - x_min
    y_range = t.max(points_3d[:, 1]) - y_min

    x_bins = int(x_range / resolution)
    y_bins = int(y_range / resolution)

    hist = t.zeros((x_bins, y_bins, points_3d.shape[1] - 2), dtype=t.float32)

    for point in points_3d:
        if height_range[0] <= point[2] <= height_range[1]:
            x_index = int((point[0] - x_min) / resolution)
            y_index = int((point[1] - y_min) / resolution)
            if 0 <= x_index < x_bins and 0 <= y_index < y_bins:
                hist[x_index, y_index, :] += point[3:]

    return hist

# # Generate a sample 3D point cloud with higher channel dimensions (XYZ + additional attributes)
# num_points = 1000
# num_channels = 7  # Adjust the number of channels as needed
# point_cloud = t.random.rand(num_points, num_channels) * t.array([10, 10, 4, 1, 2, 3, 4])  # Adjust the scaling as needed

# # Project the point cloud to birds-eye view image
# birdseye_view = project_to_birds_eye_view(point_cloud)

# # Plot the birds-eye view image for each channel
# fig, axes = plt.subplots(1, num_channels - 2, figsize=(15, 3))
# for i in range(num_channels - 2):
#     axes[i].imshow(birdseye_view[:, :, i], cmap='viridis', origin='upper')
#     axes[i].set_title(f'Channel {i + 2}')
# plt.tight_layout()
# plt.show()
