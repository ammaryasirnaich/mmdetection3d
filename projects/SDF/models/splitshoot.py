import torch
import torch.nn as nn
import torch.nn.functional as F



class LiftSplatShoot(nn.Module):
    def __init__(self, depth_bins):
        super(LiftSplatShoot, self).__init__()
        self.depth_bins = depth_bins  # Number of depth layers (D)
        
    def lift(self, image_features,intrinsics,extrinsics):
        """
        Lifts 2D image features to 3D space using depth predictions.
        :param image_features: Tensor of shape (B, C, H, W)
        :return: 3D point cloud
        """
        B, C, H, W = image_features.shape
        self.intrinsics = intrinsics  # Camera intrinsic matrix (3x3)
        self.extrinsics = extrinsics  # Camera extrinsic matrix (4x4)
        # Create mesh grid for image coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        x, y = x.to(image_features.device), y.to(image_features.device)

        # Flatten and add homogeneous coordinate (1)
        xy1 = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x).flatten()), dim=0)  # (3, H*W)
        xy1 = xy1.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)

        # Apply camera intrinsics to get rays
        rays = torch.inverse(self.intrinsics).unsqueeze(0).repeat(B, 1, 1) @ xy1  # (B, 3, H*W)

        # Predict depth distribution (e.g., softmax over depth bins)
        depth_probs = torch.softmax(image_features.view(B, C, -1), dim=2)  # (B, C, H*W)
        depths = torch.linspace(0.1, 1.0, self.depth_bins, device=image_features.device)  # Define depth bins
        depth_map = depths.view(1, 1, -1) * depth_probs  # (B, C, H*W)

        # Lift to 3D by scaling rays by depth
        point_cloud = (rays.unsqueeze(-1) * depth_map).view(B, 3, H, W, self.depth_bins)  # (B, 3, H, W, D)
        
        return point_cloud

    def splat(self, point_cloud):
        """
        Projects 3D point cloud to a BEV grid.
        :param point_cloud: Tensor of shape (B, 3, H, W, D)
        :return: BEV grid
        """
        # Simplified "splatting" process (e.g., projecting the 3D points onto a 2D grid)
        # This will vary based on your specific requirements
        # Convert 3D points to BEV (this can be complex, depending on your grid structure)
        BEV_grid = torch.max(point_cloud, dim=-1)[0]  # Max pooling along depth for BEV projection
        return BEV_grid

    def forward(self, image_features):
        # Step 1: Lift 2D image features to 3D space
        point_cloud = self.lift(image_features)
        
        # Step 2: Project 3D points to a 2D BEV grid
        # BEV_grid = self.splat(point_cloud)
        
        return point_cloud

# # Example usage
# depth_bins = 32  # Number of depth bins
# intrinsics = torch.eye(3)  # Replace with actual camera intrinsics
# extrinsics = torch.eye(4)  # Replace with actual camera extrinsics

# # Instantiate the model
# model = LiftSplatShoot(depth_bins, intrinsics, extrinsics)

# # Simulate image feature input
# image_features = torch.randn(1, depth_bins, 128, 352)  # Batch size of 1, with 128x352 image

# # Run the forward pass
# BEV_output = model(image_features)

# print(BEV_output.shape)  # Output BEV grid
