import torch
import torch.nn as nn
import torch.nn.functional as F

class LiftSplatShoot(nn.Module):
    def __init__(self, depth_bins, img_height, img_width, num_cams=6, bev_channels=64):
        super(LiftSplatShoot, self).__init__()
        self.depth_bins = depth_bins  # Number of depth layers (D)
        self.bev_channels = bev_channels  # Number of BEV feature channels
        self.img_height, self.img_width = img_height, img_width
        self.num_cams = num_cams

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Precompute the depth bins using logarithmic scaling for dynamic depth binning
        self.depth_bin_edges = torch.logspace(0, 2, steps=depth_bins + 1).to(self.device)  # Depth from 1 to 100

        # Precompute the meshgrid for image coordinates
        self.register_buffer('pixel_coords', self.create_meshgrid(img_height, img_width))

        # Learnable parameters for depth and view attention
        self.view_attention = nn.Parameter(torch.ones(self.num_cams, 1, 1), requires_grad=True)
        self.depth_attention = nn.Parameter(torch.ones(1, self.depth_bins, 1, 1), requires_grad=True)

        # Efficient multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.bev_channels, num_heads=4, batch_first=True)

        # Convolutional layers for BEV feature extraction
        self.bev_conv = nn.Sequential(
            nn.Conv2d(self.depth_bins, self.bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bev_channels),
            nn.ReLU(inplace=True),
        )

        # Example backbone (can be replaced with a pre-trained CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Additional layers can be added here
        )

    def create_meshgrid(self, height, width):
        """Creates a meshgrid of pixel coordinates."""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device)
        )
        ones = torch.ones_like(x_coords)
        pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # Shape: [3, H, W]
        return pixel_coords

    def forward(self, images, intrinsics, extrinsics):
        """
        images: Tensor of shape (B, N, C, H, W)
        intrinsics: Tensor of shape (B, N, 3, 3)
        extrinsics: Tensor of shape (B, N, 4, 4)
        """
        B, N, C, H, W = images.shape
        assert N == self.num_cams, "Number of camera views does not match."

        # Flatten batch and camera dimensions for backbone processing
        images = images.view(B * N, C, H, W)

        # Extract image features using the backbone
        with torch.cuda.amp.autocast():
            img_feats = self.backbone(images)  # Shape: [B*N, feat_channels, H', W']

        # Reshape back to separate batch and camera dimensions
        img_feats = img_feats.view(B, N, -1, img_feats.shape[-2], img_feats.shape[-1])  # [B, N, C', H', W']

        # Apply view attention
        view_attn = self.view_attention.unsqueeze(0)  # Shape: [1, N, 1, 1]
        img_feats = img_feats * view_attn  # Broadcasting over batch and spatial dimensions

        # Project image features into 3D space
        bev_feat_list = []
        for b in range(B):
            bev_feat = self.project_to_bev(img_feats[b], intrinsics[b], extrinsics[b])
            bev_feat_list.append(bev_feat)

        # Stack BEV features from all batches
        bev_feats = torch.stack(bev_feat_list, dim=0)  # Shape: [B, bev_channels, H_bev, W_bev]

        return bev_feats

    def project_to_bev(self, img_feats, intrinsics, extrinsics):
        """
        Projects image features into BEV space for a single batch.

        img_feats: Tensor of shape [N, C', H', W']
        intrinsics: Tensor of shape [N, 3, 3]
        extrinsics: Tensor of shape [N, 4, 4]
        """
        N, C, H, W = img_feats.shape
        D = self.depth_bins

        # Flatten spatial dimensions
        img_feats = img_feats.view(N, C, -1)  # Shape: [N, C, H'*W']

        # Apply depth attention
        depth_attn = self.depth_attention  # Shape: [1, D, 1, 1]
        img_feats = img_feats.unsqueeze(1) * depth_attn  # Shape: [N, D, C, H'*W']

        # Reshape for multi-head attention
        img_feats = img_feats.view(N * D, C, -1).permute(0, 2, 1)  # Shape: [N*D, H'*W', C]

        # Apply multi-head attention
        attn_output, _ = self.multihead_attention(img_feats, img_feats, img_feats)

        # Reshape back to spatial dimensions
        attn_output = attn_output.permute(0, 2, 1).view(N, D, C, H, W)

        # Aggregate features across cameras
        bev_feat = attn_output.sum(dim=0)  # Sum over cameras

        # Apply BEV convolutional layers
        bev_feat = self.bev_conv(bev_feat)

        return bev_feat

