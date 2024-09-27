import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AdaptiveWeight(nn.Module):
    def __init__(self, voxel_dim=512, image_dim=64, upscale_size=(200, 176)):
        super().__init__()
        # Upsample image feature's height and width based on upscale_size
        self.image_upscaler = nn.Sequential(
            nn.Upsample(size=upscale_size, mode='bilinear', align_corners=True),  # Upscale height and width
            nn.Conv2d(image_dim, voxel_dim, kernel_size=1)  # Adjust image channels from 64 to 512
        )

        # Linear layers to compute scalar weights for voxel and image features
        self.voxel_fc = nn.Linear(voxel_dim, 1)
        self.image_fc = nn.Linear(voxel_dim, 1)

        # Learnable parameters for fusion
        self.b_s = nn.Parameter(torch.zeros(1))
        self.b_d = nn.Parameter(torch.zeros(1))

    def forward(self, voxel_feature, image_feature):
        # Upscale the image feature's height and width to match the voxel feature dimensions
        image_feature_upscaled = self.image_upscaler(image_feature)

        # Flatten the spatial dimensions for voxel and image features
        voxel_flat = voxel_feature.view(voxel_feature.size(0), voxel_feature.size(1), -1)  # [B, 512, H * W]
        image_flat = image_feature_upscaled.view(image_feature_upscaled.size(0), image_feature_upscaled.size(1), -1)  # [B, 512, H * W]

        # Calculate voxel and image weights
        voxel_weight = torch.sigmoid(self.voxel_fc(voxel_flat.mean(-1)))  # [B, 1]
        image_weight = torch.sigmoid(self.image_fc(image_flat.mean(-1)))  # [B, 1]

        # Reshape weights for broadcasting: [B, 1, 1] -> [B, 1, H * W]
        voxel_weight = voxel_weight.unsqueeze(-1)  # [B, 1, 1] -> [B, 1, 1] -> [B, 1, H * W]
        image_weight = image_weight.unsqueeze(-1)  # Same for image_weight

        # Fusion
        fused_feature = (voxel_weight * voxel_flat + image_weight * image_flat).view_as(voxel_feature)
        
        return fused_feature
    
    

if __name__=="__main__":
    # Create random tensor inputs for voxel features and image features
    # Voxel feature shape: [batch_size, 512, 200, 176]
    # voxel_feature = torch.randn(4, 512, 200, 176)
    voxel_feature = torch.randn(4, 512, 180, 180)
    
    # Image feature shape: [batch_size, 64, 64, 176]
    image_feature = torch.randn(4, 64, 64, 176)   #torch.Size([12, 256, 32, 88])

    # Instantiate the AdaptiveWeight class
    model = AdaptiveWeight(voxel_dim=512, image_dim=64, upscale_size=(180, 180))
    
    # Run the forward method
    fused_feature = model(voxel_feature, image_feature)
    
    # Check the output shape
    print(f"Fused feature shape: {fused_feature.shape}")  # Expected: [4, 512, 200, 176]

    # Check for correctness: Assert that the output shape matches the expected shape
    # assert fused_feature.shape == (4, 512, 200, 176), f"Output shape mismatch: {fused_feature.shape}"
    assert fused_feature.shape == (4, 512, 180, 180), f"Output shape mismatch: {fused_feature.shape}"
    
    # Check if the tensor contains valid numbers (not NaN or Inf)
    assert not torch.isnan(fused_feature).any(), "Fused feature contains NaN values"
    assert not torch.isinf(fused_feature).any(), "Fused feature contains Inf values"

    print("Test passed successfully!")
    
    