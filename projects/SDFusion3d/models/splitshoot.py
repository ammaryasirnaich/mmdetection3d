import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftSplatShoot(nn.Module):
    def __init__(self, depth_bins, H, W, N, bev_channels=64):
        super(LiftSplatShoot, self).__init__()
        self.depth_bins = depth_bins  # Number of depth layers (D)
        self.bev_channels = bev_channels  # Number of BEV feature channels
        self.N = N  # Number of views

        # Precompute the depth bins (independent of input)
        self.depths = torch.linspace(1.0, 100.0, depth_bins).view(1, 1, -1, 1).cuda()

        # Precompute the meshgrid (independent of input)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        self.xy1 = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x).flatten()), dim=0).float().cuda()  # (3, H*W)
        self.H = H
        self.W = W

        # Learnable attention weights for depth and views
        self.view_attention = nn.Parameter(torch.ones(1, N, 1, H, W), requires_grad=True)  # For views
        self.depth_attention = nn.Parameter(torch.ones(1, self.depth_bins, 1, 1), requires_grad=True)  # For depth
        
        self.bev_conv = nn.Sequential(
            nn.Conv2d(3, self.bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.bev_channels),
            nn.ReLU(),
            nn.Conv2d(self.bev_channels, self.bev_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.bev_channels),
            nn.ReLU(),
        ).cuda()


    def lift(self, image_features, intrinsics, extrinsics):
        """
        Lifts 2D image features to 3D space using depth predictions, but only keeps the coordinate information.
        :param image_features: Tensor of shape (B*N, C, H, W)
        :param intrinsics: Tensor of shape (B, N, 4, 4)
        :param extrinsics: Tensor of shape (B, N, 4, 4)
        :return: 3D point cloud with only coordinate information
        """
        # print(f'Feature input to lift function:{image_features.shape}')
        BN, C, H, W = image_features.shape
        B = BN // intrinsics.shape[1]  # Calculate B from BN and N
        N = intrinsics.shape[1]

        # Flatten batch and view dimensions (combine B and N into a single dimension)
        image_features_flat = image_features
        intrinsics_flat = intrinsics.view(B * N, 4, 4)
        extrinsics_flat = extrinsics.view(B * N, 4, 4)

        # Use precomputed xy1
        xy1 = self.xy1 # Ensure it's on the correct device
        xy1 = xy1.unsqueeze(0).repeat(B * N, 1, 1)  # (B*N, 3, H*W)

        # Apply camera intrinsics to get rays
        intrinsics_inv = torch.inverse(intrinsics_flat[:, :3, :3])  # Invert the intrinsic matrix
        rays = intrinsics_inv @ xy1  # (B*N, 3, H*W)

        # Predict depth distribution (e.g., softmax over depth bins)
        depth_probs = torch.softmax(image_features_flat.view(B * N, C, -1), dim=-1)  # (B*N, C, H*W)

        # Use precomputed depth bins to create 3D points
        depths = self.depths # (1, 1, D, 1)
        point_cloud = depths * rays.unsqueeze(2)  # (B*N, 3, D, H*W)

        return point_cloud.view(B, N, 3, H, W, self.depth_bins)  # Unflatten to (B, N, 3, H, W, D)

    def splat(self, point_cloud):
        """
        Project 3D point cloud into 2D BEV grid.
        :param point_cloud: Tensor of shape (B, N, 3, H, W, D)
        :return: BEV grid of shape (B, bev_channels, H, W)
        """
        B, N, _, H, W, D = point_cloud.shape

        # Apply attention weights to depth and views
        depth_attention = torch.softmax(self.depth_attention, dim=1)  # Softmax over depth bins (D)
        view_attention = torch.softmax(self.view_attention, dim=1)    # Softmax over views (N)

        # Reshape view attention to match the point cloud (B, N, 1, H, W)
        view_attention = view_attention.expand(B, N, 1, H, W)

        # Weight each view and depth bin by its attention score
        weighted_point_cloud = point_cloud * depth_attention.view(1, 1, 1, 1, 1, D)  # Apply depth attention
        weighted_point_cloud = weighted_point_cloud * view_attention.unsqueeze(-1)  # Apply view attention

        # Aggregate over views and depth
        bev_features = weighted_point_cloud.sum(dim=[1, -1])  # Sum over N (views) and D (depth)

        # # Process features with learnable convolution layers
        # self.bev_conv = nn.Sequential(
        #     nn.Conv2d(3, self.bev_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.bev_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(self.bev_channels, self.bev_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.bev_channels),
        #     nn.ReLU(),
        # ).cuda()

        # Apply the convolution to produce the final BEV output with feature channels
        BEV_grid = self.bev_conv(bev_features)

        return BEV_grid.view(B, self.bev_channels, H, W)  # Unflatten to (B, bev_channels, H, W)

    def forward(self, image_features, intrinsics, extrinsics):
        # Step 1: Lift 2D image features to 3D space, but only keep the coordinate information
        point_cloud = self.lift(image_features, intrinsics, extrinsics)
        
        # Step 2: Project 3D points to a 2D BEV grid
        BEV_grid = self.splat(point_cloud)
        
        return point_cloud, BEV_grid

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

if __name__ == "__main__":
    # Example usage
    depth_bins = 100  # Number of depth bins
    B, N, C, H, W = 4, 6, 256, 64, 176  # Example dimensions

    # Simulate camera intrinsics and extrinsics
    intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).cuda()  # (B, N, 4, 4)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).cuda()  # (B, N, 4, 4)

    # Simulate image feature input
    image_features = torch.randn(B * N, C, H, W).cuda()  # Batch size of 2, 6 views, 256 channels, 64x176 image

    # Instantiate the model
    model = LiftSplatShoot(depth_bins, H, W, N).cuda()

    # Run the forward pass
    point_cloud, BEV_grid = model(image_features, intrinsics, extrinsics)

    print(f'Point cloud shape: {point_cloud.shape}')  # Output point cloud shape [B, N, D, H, W]
    print(f'BEV shape: {BEV_grid.shape}')  # Output BEV grid shape [B, C, H, W]