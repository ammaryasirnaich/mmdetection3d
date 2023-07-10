
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
plt.matplotlib.use('TkAgg')
from numba import njit
import numpy as np
from mmdet3d.registry import MODELS



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

birdseye_view = kitti_to_bev(voxel_data, voxel_size, output_shape)
print("birdseye_view",birdseye_view.shape)

# Display the bird's-eye view
plt.imshow(birdseye_view, cmap='gray')
plt.show()
print("Pass")









# num_voxels = torch.randint(1, 100, [97297])
# features = torch.rand([97297, 20, 5])
# coors = torch.randint(0, 100, [97297, 4])


# hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
# hard_simple_VFE = MODELS.build(hard_simple_VFE_cfg)
# outputs = hard_simple_VFE(features, num_voxels, None)

# print("coors", coors[:,1:].shape)
# print("feature shape", outputs.shape)

# outputs = torch.concat((outputs,coors[:,1:]),dim=2)
# print("Pass")





