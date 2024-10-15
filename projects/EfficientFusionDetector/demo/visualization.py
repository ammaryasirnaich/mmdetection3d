import open3d as o3d
import torch
import torch.nn as nn
import numpy as np

# Original frustum creation function
def create_frustum(image_size, feature_size, dbound):
    """
    Creates the frustum grid based on image and feature size.
    """
    iH, iW = image_size
    fH, fW = feature_size

    # Create depth slices
    ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape

    # Create xs and ys for the frustum grid
    xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    # Stack xs, ys, and ds to form the frustum
    frustum = torch.stack((xs, ys, ds), -1)
    
    # Print diagnostic info about the frustum
    print(f"Frustum Shape: {frustum.shape}")
    print(f"Frustum Sample Values: {frustum[0,0,0]}, {frustum[-1,-1,-1]}")

    # Return as a non-trainable tensor (no gradient required)
    return nn.Parameter(frustum, requires_grad=False)

# Create individual voxel meshes
def create_voxel(center, size):
    voxel = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    voxel.translate(center - np.array([size / 2, size / 2, size / 2]))  # Shift to the correct center
    voxel.paint_uniform_color([0.5, 0.5, 0.8])  # Optional: color the voxel (light blue here)
    return voxel

# Visualization function using Open3D
def visualize_frustum_with_voxels(image_size, feature_size, xbound, ybound, zbound, dbound):
    # Generate the frustum using the provided function
    frustum = create_frustum(image_size, feature_size, dbound)

    # Extract xs, ys, and ds from the frustum tensor
    xs, ys, ds = frustum[..., 0], frustum[..., 1], frustum[..., 2]

    # Generate voxel grid points inside the frustum
    dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
    
    # Print diagnostic info about voxel grid step sizes and number of steps
    print(f"Voxel Step Sizes (dx): {dx}")
    print(f"Number of Steps (nx): {nx}")
    
    # Limit the number of points visualized (reduce number of voxels)
    limit = 500  # Cap the number of voxels to display
    xs_flat = xs.flatten()[:limit]
    ys_flat = ys.flatten()[:limit]
    ds_flat = ds.flatten()[:limit]

    # Create voxel centers using xs, ys, and ds (limited)
    voxel_centers = torch.stack([xs_flat, ys_flat, ds_flat], dim=-1).numpy()

    # Print voxel count and coordinates
    print(f"Number of voxels: {voxel_centers.shape[0]}")
    print(f"Voxel Centers Sample: {voxel_centers[:5]}")

    # Visualizing voxels inside the frustum
    voxel_size = dx.numpy().min()  # Use the minimum grid size for consistent voxel size
    voxel_list = []
    
    for center in voxel_centers:
        voxel = create_voxel(center, voxel_size)  # Create the voxel at each center
        voxel_list.append(voxel)

    # Visualize all the voxels
    if voxel_list:
        o3d.visualization.draw_geometries(voxel_list)
    else:
        print("No voxels to display.")

# Additional helper function from the original code
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # Step sizes in x, y, z directions
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])  # Mid-points
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])  # Number of steps
    return dx, bx, nx

# Example parameters for the frustum creation
image_size = [256, 704]  # Original image size
feature_size = [32, 88]  # Feature map size
xbound = [-54.0, 54.0, 5.0]  # Larger step size for x-axis to reduce voxel count
ybound = [-54.0, 54.0, 5.0]  # Larger step size for y-axis to reduce voxel count
zbound = [-10.0, 10.0, 5.0]  # Larger step size for z-axis (depth) to reduce voxel count
dbound = [1.0, 60.0, 0.5]   # Larger depth step size

# Calling the visualization function
visualize_frustum_with_voxels(image_size, feature_size, xbound, ybound, zbound, dbound)
