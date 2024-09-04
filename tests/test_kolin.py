import torch
# from kaolin.ops.conversions import pointclouds_to_voxelgrids
from kaolin import ops

from kaolin.ops.spc import Conv3d, scan_octrees, generate_points

# Example point cloud tensor with shape (num_points, 3)
pointclouds = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

# Convert the point cloud to a voxel grid of desired resolution
voxel_grid = ops.conversions.pointclouds_to_voxelgrids(pointclouds.unsqueeze(0), resolution=32)



# Add a channel dimension to the voxel grid
voxel_grid = voxel_grid.unsqueeze(1).cuda()  # (batch_size, 1, depth, height, width)


# print(voxel_grid.shape)

# # Assuming you have the voxel grid or structured point cloud
octrees, lengths,coalescent_features = ops.spc.feature_grids_to_spc(voxel_grid)

max_level, pyramids, exsum = scan_octrees(octrees, lengths)
point_hierarchies = generate_points(octrees, pyramids, exsum)

# # Define sparse convolution kernel
kernel_vectors = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.short).cuda()
# 
# # Perform convolution
conv_layer = Conv3d(in_channels=3, out_channels=16, kernel_vectors=kernel_vectors, jump=1).cuda()
out_features, out_level = conv_layer(octrees, point_hierarchies, max_level, pyramids, exsum)

print("pass")

