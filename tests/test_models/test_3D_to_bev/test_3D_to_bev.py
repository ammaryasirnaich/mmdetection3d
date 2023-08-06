
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')
from numba import njit , jit
import numba as nb
import numpy as np
from mmdet3d.registry import MODELS




# @njit(parallel=True)
# def pointcloud_to_bev(point_cloud, resolution, range_x, range_y, output_shape):
#     num_pixels_x = output_shape[1]
#     num_pixels_y = output_shape[0]
#     bev = np.zeros((num_pixels_y, num_pixels_x), dtype=np.float32)

#     for i in nb.prange(point_cloud.shape[0]):
#         x, y, z, intensity = point_cloud[i]
#         if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
#             x_idx = int((x - range_x[0]) / resolution)
#             y_idx = int((y - range_y[0]) / resolution)
#             if 0 <= x_idx < num_pixels_x and 0 <= y_idx < num_pixels_y:
#                 bev[y_idx, x_idx] = intensity

#     return bev



@njit
def kitti_to_bev(voxel_data, voxel_size, output_shape):
    # Calculate the dimensions of the bird's-eye view grid
    bev_width, bev_height = output_shape

    # Create an empty bird's-eye view grid
    birdseye_view = np.zeros((bev_height, bev_width), dtype=np.float32)

    # Calculate voxel grid origin (minimum x and y values)
    origin_x = np.min(voxel_data[:, 0])
    origin_y = np.min(voxel_data[:, 1])

    # Calculate voxel grid dimensions (number of cells in x and y directions)
    grid_dim_x = int(np.ceil((np.max(voxel_data[:, 0]) - origin_x) / voxel_size))
    grid_dim_y = int(np.ceil((np.max(voxel_data[:, 1]) - origin_y) / voxel_size))

    # Iterate over each voxel in the voxel grid
    for voxel in voxel_data:
        x, y, z, intensity = voxel

        # Calculate the corresponding position in the bird's-eye view grid
        bev_x = int((x - origin_x) / voxel_size)
        bev_y = int((y - origin_y) / voxel_size)

        # Update the bird's-eye view grid with the voxel intensity
        if bev_x >= 0 and bev_x < bev_width and bev_y >= 0 and bev_y < bev_height:
            if intensity > birdseye_view[bev_y, bev_x]:
                birdseye_view[bev_y, bev_x] = intensity

    return birdseye_view
# Bird's eye view parameters
resolution = 0.1  # Resolution of each cell in the BEV
range_x = (0, 70)  # Range of X coordinates for the BEV
range_y = (-40, 40)  # Range of Y coordinates for the BEV
output_shape = (600, 800)  # Output shape of the BEV (y,x)



# Example usage
# voxel_data = np.array([
#     [2, 1, 3, 0.5],
#     [5, 2, 4, 0.8],
#     [7, 8, 2, 0.6]
# ])

data = torch.load('pointclouds.pt',map_location=torch.device('cpu'))
print("data", data.shape)
b,v,d = data.shape
voxel_data =   data.reshape(b*v,d).numpy()

print("data",voxel_data.shape)
print("element", voxel_data[1,:])

voxel_size = 0.1  # Voxel size in meters
output_shape = (100, 100)  # Output BEV shape in pixels (width, height)
data = torch.load('pointclouds.pt',map_location=torch.device('cpu'))
print("data", data.shape)
b,v,d = data.shape
point_cloud =   data.reshape(b*v,d).numpy()

# Convert point cloud to BEV
bev = kitti_to_bev(point_cloud, resolution, range_x, range_y, output_shape)

# Plot the bird's eye view
plt.imshow(bev, cmap='jet', extent=[range_x[0], range_x[1], range_y[0], range_y[1]], origin='lower')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bird\'s Eye View')
plt.show()

print("Pause")

outputs = torch.concat((outputs,coors[:,1:]),dim=2)
print("Pass")





