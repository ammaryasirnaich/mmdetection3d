import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftSplatShoot(nn.Module):
    def __init__(self, depth_bins, H, W):
        super(LiftSplatShoot, self).__init__()
        self.depth_bins = depth_bins  # Number of depth layers (D)

        # Precompute the depth bins (independent of input)
        self.depths = torch.linspace(0.1, 1.0, depth_bins)

        # Precompute the meshgrid (independent of input)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        self.xy1 = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x).flatten()), dim=0).float()  # (3, H*W)

    
    def lift(self, image_features, intrinsics, extrinsics):
        """
        Lifts 2D image features to 3D space using depth predictions.
        :param image_features: Tensor of shape (B*N, C, H, W)
        :param intrinsics: Tensor of shape (B, N, 4, 4)
        :param extrinsics: Tensor of shape (B, N, 4, 4)
        :return: 3D point cloud
        """
        BN, C, H, W = image_features.shape
        B = intrinsics.shape[0]
        N = intrinsics.shape[1]

        # Reshape image_features to separate batch and number of views
        image_features = image_features.view(B, N, C, H, W)

      # Use precomputed xy1
        xy1 = self.xy1.to(image_features.device)  # Ensure it's on the correct device
        xy1 = xy1.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 3, H*W)

        # Apply camera intrinsics to get rays
        intrinsics_inv = torch.inverse(intrinsics[:, :, :3, :3])  # Invert the intrinsic matrix
        rays = intrinsics_inv @ xy1  # (B, N, 3, H*W)

        # Predict depth distribution (e.g., softmax over depth bins)
        depth_probs = torch.softmax(image_features.view(B, N, C, -1), dim=3)  # (B, N, C, H*W)

        # Use precomputed depths and adjust depth_map to match the dimensions for broadcasting
        depth_map = depth_probs.unsqueeze(-1) * self.depths.view(1, 1, 1, -1).to(image_features.device)  # (B, N, C, H*W, D)

        # Lift to 3D by scaling rays by depth
        rays = rays.unsqueeze(-1)  # (B, N, 3, H*W, 1)
        point_cloud = rays * depth_map  # (B, N, 3, H*W, D)
        
        # Transform point cloud using camera extrinsics
        point_cloud_homogeneous = torch.cat([point_cloud, torch.ones_like(point_cloud[:, :, :1])], dim=2)  # (B, N, 4, H*W, D)
        point_cloud_homogeneous = point_cloud_homogeneous.permute(0, 1, 3, 4, 2)  # (B, N, H*W, D, 4)
        
        # Multiply with extrinsics (B, N, 1, 1, 4, 4) with (B, N, H*W, D, 4) -> (B, N, H*W, D, 4)
        extrinsics = extrinsics.unsqueeze(-3).unsqueeze(-3)  # (B, N, 1, 1, 4, 4)
        point_cloud_world = torch.matmul(extrinsics, point_cloud_homogeneous.unsqueeze(-1))  # (B, N, H*W, D, 4, 1)
        point_cloud_world = point_cloud_world.squeeze(-1).permute(0, 1, 4, 2, 3)[:, :, :3]  # (B, N, 3, H*W, D)
        point_cloud_world = point_cloud_world.view(B, N, 3, H, W, self.depth_bins)

        return point_cloud_world
    
    

    def splat(self, point_cloud):
        """
        Projects 3D point cloud to a BEV grid.
        :param point_cloud: Tensor of shape (B, N, 3, H, W, D)
        :return: BEV grid
        """
        # Simplified "splatting" process (e.g., projecting the 3D points onto a 2D grid)
        # This will vary based on your specific requirements
        BEV_grid = torch.max(point_cloud, dim=-1)[0]  # Max pooling along depth for BEV projection
        BEV_grid = BEV_grid.mean(dim=1)  # Average over camera views (N)
        return BEV_grid

    def forward(self, image_features,intrinsics, extrinsics):
        
        self.intrinsics = intrinsics  # Camera intrinsic matrix (B, N, 4, 4)
        self.extrinsics = extrinsics  # Camera extrinsic matrix (B, N, 4, 4)
        
        # Step 1: Lift 2D image features to 3D space
        #Output 3d project depth feature 
        point_cloud = self.lift(image_features, self.intrinsics, self.extrinsics) #[B, N, C(3)(coord), H, W, D]
        
        # Step 2: Project 3D points to a 2D BEV grid
        # BEV_grid = self.splat(point_cloud)
        
        return point_cloud



if __name__=="__main__":
    # Example usage
    depth_bins = 100  # Number of depth bins
    B, N, C, H, W = 4, 6, 3, 256, 704  # Example dimensions

    # Simulate camera intrinsics and extrinsics
    intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 4, 4)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 4, 4)

    # Simulate image feature input
    image_features = torch.randn(B * N, C, H, W)  # Batch size of 2, 6 views, 64 channels, 128x352 image

    # Instantiate the model
    model = LiftSplatShoot(depth_bins, H, W)

    # Run the forward pass
    point_cloud_world = model(image_features,intrinsics, extrinsics)

    print(point_cloud_world.shape)  # Output 3d project depth feature [B, N, C(3)(coord), H, W, D]
    print("Stop")
