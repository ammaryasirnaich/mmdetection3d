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
        self.H = H
        self.W = W

    def lift(self, image_features, intrinsics, extrinsics):
        """
        Lifts 2D image features to 3D space using depth predictions, but only keeps the coordinate information.
        :param image_features: Tensor of shape (B*N, C, H, W)
        :param intrinsics: Tensor of shape (B, N, 4, 4)
        :param extrinsics: Tensor of shape (B, N, 4, 4)
        :return: 3D point cloud with only coordinate information
        """
        BN, C, H, W = image_features.shape
        B = BN // intrinsics.shape[1]  # Calculate B from BN and N
        N = intrinsics.shape[1]

        # Ensure that H and W are consistent
        assert H == self.H and W == self.W, "Mismatch between input feature size and precomputed meshgrid."

        # Use precomputed xy1
        xy1 = self.xy1.to(image_features.device)  # Ensure it's on the correct device
        xy1 = xy1.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 3, H*W)

        # Apply camera intrinsics to get rays
        intrinsics_inv = torch.inverse(intrinsics[:, :, :3, :3])  # Invert the intrinsic matrix
        rays = intrinsics_inv @ xy1  # (B, N, 3, H*W)

        # Predict depth distribution (e.g., softmax over depth bins)
        depth_probs = torch.softmax(image_features.view(B, N, C, -1), dim=3)  # (B, N, C, H*W)

        # Use precomputed depths and adjust depth_map to match the dimensions for broadcasting
        depth_map = depth_probs.mean(dim=2, keepdim=True).unsqueeze(-1) * self.depths.view(1, 1, 1, -1).to(image_features.device)  # (B, N, 1, H*W, self.depth_bins)

        # Expand rays to match the depth_map's shape for proper broadcasting
        rays = rays.unsqueeze(-1)  # (B, N, 3, H*W, 1)
        rays = rays.expand(-1, -1, -1, -1, self.depth_bins)  # (B, N, 3, H*W, self.depth_bins)

        # Multiply rays by depth_map to lift to 3D coordinates
        point_cloud_coords = rays * depth_map  # (B, N, 3, H*W, self.depth_bins)

        # Reshape the point cloud to (B, N, 3, H, W, self.depth_bins)
        point_cloud_coords = point_cloud_coords.view(B, N, 3, self.H, self.W, self.depth_bins)

        return point_cloud_coords

    def splat(self, point_cloud):
        """
        Projects 3D point cloud to a BEV grid, considering only spatial (coordinate) information.
        :param point_cloud: Tensor of shape (B, N, 3, H, W, self.depth_bins)
        :return: BEV grid
        """
        # Simplified "splatting" process (e.g., projecting the 3D points onto a 2D grid)
        BEV_grid = torch.max(point_cloud, dim=-1)[0]  # Max pooling along depth for BEV projection
        BEV_grid = BEV_grid.mean(dim=1)  # Average over camera views (N)
        return BEV_grid

    def forward(self, image_features, intrinsics, extrinsics):
        # Step 1: Lift 2D image features to 3D space, but only keep the coordinate information
        point_cloud = self.lift(image_features, intrinsics, extrinsics)
        
        # Step 2: Project 3D points to a 2D BEV grid
        # BEV_grid = self.splat(point_cloud)
        
       
        # torch.Size([B, N, C, H, W, D]) 
         # torch.Size([B*N, C, H, W])
        # print(f'point cloud size {point_cloud.shape} and image feature {image_features.shape}')

        # Update it to flattent he dimension to be use easy
        
        return point_cloud, image_features



# Global level meshgrid precomputation
def create_meshgrid(H, W):
    """
    Create a meshgrid for image coordinates.
    :param H: Height of the image
    :param W: Width of the image
    :return: Tensor of shape (3, H*W)
    """
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    xy1 = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x).flatten()), dim=0).float()  # (3, H*W)
    return xy1





if __name__=="__main__":
    # Example usage
    depth_bins = 100  # Number of depth bins
    B, N, C, H, W = 4, 6, 256, 64, 176  # Example dimensions

    # Simulate camera intrinsics and extrinsics
    intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 4, 4)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B, N, 4, 4)

    # Simulate image feature input
    image_features = torch.randn(B * N, C, H, W)  # Batch size of 2, 6 views, 256 channels, 64x176 image

    # Instantiate the model
    model = LiftSplatShoot(depth_bins, H, W)

    # Run the forward pass
    point_cloud = model(image_features, intrinsics, extrinsics)

    print(point_cloud.shape)  # Output BEV grid
