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



@MODELS.register_module()
class GatedFusionModule(nn.Module):
    def __init__(self, voxel_dim=512, image_dim=64, upscale_size=(180, 180)):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable alpha

        self.upscale_size = upscale_size
        self.image_upscaler = nn.Sequential(
             nn.Conv2d(image_dim, voxel_dim, kernel_size=1),  # Adjust image channels from image_dim (e.g., 64) to voxel_dim (e.g., 512)
            )
        
        # Gating mechanism: a small convolutional network for local gating
        in_channels = voxel_dim
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),  # Output single channel for gating
            nn.Sigmoid()  # Ensures gating values are between 0 and 1
        )
    
    def forward(self, bev_lidar_feature, bev_image_feature):
        # Pool the dense feature map
        """
        bev_lidar_feature: LiDAR feature map [B, C, H, W]
        bev_image_feature: Camera feature map [B, C, H, W]
        """
        # Ensure alpha stays in [0, 1] range using sigmoid
        alpha = torch.sigmoid(self.alpha)        
        bev_image_upscaled = F.interpolate(bev_image_feature, size=self.upscale_size, mode='bilinear', align_corners=True)
        bev_image_upscaled = self.image_upscaler(bev_image_upscaled)
        

        # Concatenate the features along the channel dimension for gating
        F_concat = torch.cat([bev_lidar_feature, bev_image_upscaled], dim=1)  # Concatenate along channel axis
        
        # Compute local gating values
        G_local = self.gate_conv(F_concat)  # Shape: [B, 1, H, W]
        
        # Compute the fused output using both global and local control
        F_fused = alpha * G_local * bev_lidar_feature + (1 - alpha) * (1 - G_local) * bev_image_upscaled
        
        return F_fused


# Example usage
if __name__ == "__main__":
    # Example feature maps from LiDAR (F_voxel) and Camera (F_dense)
    F_voxel = torch.randn(4, 512, 180, 180)  # LiDAR feature map
    F_dense = torch.randn(4, 64, 64, 176)  # Camera feature map

    # voxel_feature = torch.randn(4, 512, 180, 180).to("cuda")
    
    # Image feature shape: [batch_size, 64, 64, 176]
    # image_feature = torch.randn(4, 64, 64, 176).to("cuda")   #torch.Size([12, 256, 32, 88])
    
    # Initialize the Gated Fusion module
    gated_fusion_module = GatedFusionModule()
    
    # Forward pass through the fusion module
    F_fused = gated_fusion_module(F_voxel, F_dense)
    
    print("Fused feature map shape:", F_fused.shape)  # Should output: [B, C, H, W]





