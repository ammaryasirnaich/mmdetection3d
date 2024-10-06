import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AdaptiveMultiStageFusionBEV(nn.Module):
    def __init__(self, voxel_dim=512, image_dim=64, upscale_size=(200, 176)):
        super(AdaptiveMultiStageFusionBEV, self).__init__()

        # Image upscaler: Upsample the spatial dimensions and adjust channels to match voxel_dim
        self.image_upscaler = nn.Sequential(
            nn.Upsample(size=upscale_size, mode='bilinear', align_corners=True),  # Upscale image BEV spatial dimensions
            nn.Conv2d(image_dim, voxel_dim, kernel_size=1),  # Adjust image channels from image_dim (e.g., 64) to voxel_dim (e.g., 512)
            # nn.BatchNorm2d(voxel_dim)  # Batch normalization after channel adjustment
        )

        # Fully connected layers for calculating weights for voxel and image BEV features
        self.bev_lidar_fc = nn.Linear(voxel_dim, 1)  # Weight for LiDAR BEV
        self.bev_image_fc = nn.Linear(voxel_dim, 1)  # Weight for Image BEV
        
        # Bias terms for adaptive weighting
        self.bias_lidar = nn.Parameter(torch.zeros(1))
        self.bias_image = nn.Parameter(torch.zeros(1))
        
        # Batch normalization for the final fused BEV feature
        self.batch_norm = nn.BatchNorm2d(voxel_dim)

    def forward(self, bev_lidar_feature, bev_image_feature):
        """
        bev_lidar_feature: BEV feature from LiDAR sensor [B, C_voxel, H, W]
        bev_image_feature: BEV feature from Camera sensor [B, C_image, H', W']
        """
        # Step 1: Upscale the image BEV feature to match the LiDAR BEV feature size and adjust the channels
        bev_image_upscaled = self.image_upscaler(bev_image_feature)  # [B, C_voxel, H, W] after upscaling

        # Step 2: Flatten the BEV features spatially for weight calculation
        bev_lidar_flat = bev_lidar_feature.view(bev_lidar_feature.size(0), bev_lidar_feature.size(1), -1).mean(-1)  # [B, C_voxel]
        bev_image_flat = bev_image_upscaled.view(bev_image_upscaled.size(0), bev_image_upscaled.size(1), -1).mean(-1)  # [B, C_voxel]

        # Step 3: Calculate the weights for each BEV feature using FC layers
        lidar_weight = torch.sigmoid(self.bev_lidar_fc(bev_lidar_flat) + self.bias_lidar)  # [B, 1]
        image_weight = torch.sigmoid(self.bev_image_fc(bev_image_flat) + self.bias_image)  # [B, 1]

        # Step 4: Normalize the weights so that their sum equals 1
        total_weight = lidar_weight + image_weight
        lidar_weight = lidar_weight / total_weight  # Normalize [B, 1]
        image_weight = image_weight / total_weight  # Normalize [B, 1]

        # Step 5: Reshape the weights for broadcasting
        lidar_weight = lidar_weight.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        image_weight = image_weight.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        # Step 6: Fuse the BEV features from LiDAR and Camera using the calculated weights
        fused_feature = lidar_weight * bev_lidar_feature + image_weight * bev_image_upscaled

        # Step 7: Apply batch normalization to the fused BEV feature
        fused_feature = self.batch_norm(fused_feature)
        
        return fused_feature



if __name__ == "__main__":
    # Example usage for BEV features
    batch_size = 4
    voxel_dim = 512
    image_dim = 64
    height_voxel, width_voxel = 200, 176  # Spatial dimensions of LiDAR BEV features
    height_image, width_image = 50, 44  # Original spatial dimensions of Image BEV features (before upscaling)

    # Simulated input data (BEV features from LiDAR and Camera sensors)
    bev_lidar_feature = torch.randn(batch_size, voxel_dim, height_voxel, width_voxel)  # LiDAR BEV feature
    bev_image_feature = torch.randn(batch_size, image_dim, height_image, width_image)  # Camera BEV feature

    # Initialize the adaptive fusion model for BEV features with upscaling
    adaptive_fusion_model_bev = AdaptiveMultiStageFusionBEV(voxel_dim=voxel_dim, image_dim=image_dim, upscale_size=(height_voxel, width_voxel))

    # Perform forward pass
    fused_output = adaptive_fusion_model_bev(bev_lidar_feature, bev_image_feature)

    print(fused_output.shape)  # Should match LiDAR BEV shape, e.g., [B, 512, 200, 176]
