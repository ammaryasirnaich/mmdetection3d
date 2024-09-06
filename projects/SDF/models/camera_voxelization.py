import torch
from einops import rearrange

def voxelize_camera_features(camera_features, voxel_grid):
    """
    Voxelizes the dense camera features into the sparse voxel grid.

    Args:
    - camera_features (torch.Tensor): Tensor of shape [B, N, H, W, Depth, C, F], where:
        - C represents the 3D world coordinates (x, y, z),
        - F represents the feature dimensions.
    - voxel_grid (torch.Tensor): Predefined voxel grid coordinates [B, M, 3] (x, y, z).

    Returns:
    - voxelized_features (torch.Tensor): Tensor of shape [B, M, F], where F is the feature dimension.
    """
    B, N, H, W, Depth, C, F = camera_features.shape
    M = voxel_grid.shape[1]  # Number of voxels

    # Step 1: Flatten the camera features and world coordinates
    # Combine B, N, H, W, Depth into a single dimension [B*N, H*W*Depth]
    camera_feats_flat = rearrange(camera_features, 'B N H W D C F -> (B N) (H W D) (C F)')
    
    # Separate the world coordinates (C: x, y, z) and features (F)
    world_coords_flat = camera_feats_flat[..., :C]  # Shape: [B*N, H*W*Depth, C] (world coordinates)
    features_flat = camera_feats_flat[..., C:]  # Shape: [B*N, H*W*Depth, F] (features)

    # Repeat the voxel grid for each camera view and batch
    voxel_grid_repeated = voxel_grid.repeat_interleave(N, dim=0)  # Shape: [B*N, M, 3]

    # Step 2: Compute the distances between each point in world_coords_flat and the voxel grid
    distances = torch.cdist(world_coords_flat, voxel_grid_repeated)  # Shape: [B*N, H*W*Depth, M]

    # Step 3: Find the nearest voxel for each point based on the smallest distance
    nearest_voxels = torch.argmin(distances, dim=-1)  # Shape: [B*N, H*W*Depth]

    # Step 4: Initialize tensor for voxelized features and point counts per voxel
    voxelized_features = torch.zeros(B * N, M, F, device=camera_features.device)  # Shape: [B*N, M, F]
    voxel_counts = torch.zeros(B * N, M, device=camera_features.device)  # Shape: [B*N, M]

    # Step 5: Perform scatter-add to aggregate features based on voxel assignment
    indices = nearest_voxels.view(B * N, -1)  # Shape: [B*N, H*W*Depth]
    
    # Use scatter_add_ for efficient feature aggregation per voxel
    voxelized_features.scatter_add_(
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, F),  # Expand indices to match the feature dimensions
        src=features_flat
    )
    
    # Count the number of points per voxel using scatter_add_ without expanding indices
    voxel_counts.scatter_add_(
        dim=1,
        index=indices,  # Indices of nearest voxels
        src=torch.ones_like(indices, dtype=voxel_counts.dtype)
    )

    # Step 6: Normalize the features (mean pooling)
    voxelized_features /= voxel_counts.clamp(min=1).unsqueeze(-1)  # Avoid division by zero

    # Reshape back to [B, N, M, F]
    voxelized_features = rearrange(voxelized_features, '(b n) M F -> b n M F', b=B, n=N)

    return voxelized_features



# Example usage
if __name__ == "__main__":
    # Define test input dimensions
    B = 2  # Batch size
    N = 3  # Number of camera views
    H = 64  # Height of image
    W = 64  # Width of image
    Depth = 32  # Depth dimension
    C = 3  # Number of coordinate dimensions (x, y, z)
    F = 4  # Number of feature dimensions (e.g., intensity, color)
    M = 100  # Number of voxels

    # Generate random camera features [B, N, H, W, Depth, C, F]
    camera_features = torch.randn(B, N, H, W, Depth, C, F)

    # Generate random voxel grid coordinates [B, M, 3]
    voxel_grid = torch.randn(B, M, 3)

    # Call the voxelization function
    voxelized_features = voxelize_camera_features(camera_features, voxel_grid)
    
     # Check the output shape
    expected_output_shape = (B, N, M, F)  # Expected shape after voxelization
    print(voxelized_features.shape)
    print(expected_output_shape)
    assert voxelized_features.shape == expected_output_shape, f"Expected {expected_output_shape}, but got {voxelized_features.shape}"
    

