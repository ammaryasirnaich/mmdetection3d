
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')
from numba import njit , jit
import numba as nb
import numpy as np
from mmdet3d.registry import MODELS



@njit(parallel=True)
def pointcloud_to_bev(point_cloud, resolution, range_x, range_y, output_shape):
    num_pixels_x = output_shape[1]
    num_pixels_y = output_shape[0]
    bev = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float32)

    for i in nb.prange(point_cloud.shape[0]):
        x, y, z, intensity = point_cloud[i]
        if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
            x_idx = int((x - range_x[0]) / resolution)
            y_idx = int((y - range_y[0]) / resolution)
            if 0 <= x_idx < num_pixels_x and 0 <= y_idx < num_pixels_y:
                bev[y_idx, x_idx] = intensity

    return bev



# Bird's eye view parameters
resolution = 0.1  # Resolution of each cell in the BEV
range_x = (0, 70)  # Range of X coordinates for the BEV
range_y = (-40, 40)  # Range of Y coordinates for the BEV
output_shape = (600, 800)  # Output shape of the BEV (y,x)



data = torch.load('pointclouds.pt',map_location=torch.device('cpu'))
print("data", data.shape)
b,v,d = data.shape
point_cloud =   data.reshape(b*v,d).numpy()

# Convert point cloud to BEV
bev = pointcloud_to_bev(point_cloud, resolution, range_x, range_y, output_shape)

# Plot the bird's eye view
plt.imshow(bev, cmap='jet', extent=[range_x[0], range_x[1], range_y[0], range_y[1]], origin='lower')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bird\'s Eye View')
plt.show()

print("Pause")






