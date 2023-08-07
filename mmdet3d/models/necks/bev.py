import torch
import numpy as np


def bev_3D_to_2D(point_clouds_batch, resolution=0.2, height_range=(-3, 1), image_size=(200, 200)):
    
    batch_size,num_points,num_channels = point_clouds_batch.shape
    x_min = torch.min(point_clouds_batch[:, :, 0])
    y_min = torch.min(point_clouds_batch[:, :, 1])
    x_range = torch.max(point_clouds_batch[:, :, 0]) - x_min
    y_range = torch.max(point_clouds_batch[:, :, 1]) - y_min

    x_bins = int(x_range / resolution)
    y_bins = int(y_range / resolution)

    hist_batch = torch.zeros((batch_size, image_size[0], image_size[1], num_channels-3), dtype=torch.float32,device='cuda')

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
