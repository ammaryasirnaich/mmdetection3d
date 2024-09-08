# src/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaptiveWeight(nn.Module):
    def __init__(self, voxel_dim, image_dim,upscale_size=(200, 176)):
        super().__init__()
        self.image_upscaler = nn.Sequential(
            nn.Upsample(size=upscale_size, mode='bilinear', align_corners=True),
            nn.Conv2d(image_dim, voxel_dim, kernel_size=1)
        )
        self.voxel_fc = nn.Linear(voxel_dim, 1)
        self.image_fc = nn.Linear(voxel_dim, 1)
        self.b_s = nn.Parameter(torch.zeros(1))
        self.b_d = nn.Parameter(torch.zeros(1))
    
    def forward(self, voxel_feature, image_feature):
        # Upscale the image feature to match voxel feature dimensions
        
        print(f'Voxel_feature shape:{voxel_feature.shape}')
        print(f'Voxel_feature shape:{image_feature.shape}')
        # Voxel_feature shape:torch.Size([4, 512, 200, 176])
        # Voxel_feature shape:torch.Size([4, 64, 64, 176])
        
        
        upscaled_image_feature = self.image_upscaler(image_feature)  # [4, 512, 200, 176]

        # Flatten the features for adaptive weight computation
        voxel_flat = torch.mean(voxel_feature, dim=[2, 3])  # [4, 512]
        image_flat = torch.mean(upscaled_image_feature, dim=[2, 3])  # [4, 512]

        # Compute the adaptive weights
        voxel_weight = torch.sigmoid(self.voxel_fc(voxel_flat) + self.image_fc(image_flat) + self.b_s)  # [4, 1]
        image_weight = torch.sigmoid(self.image_fc(image_flat) + self.voxel_fc(voxel_flat) + self.b_d)  # [4, 1]
        
        # Normalize the weights
        total_weight = voxel_weight + image_weight
        voxel_weight = voxel_weight / total_weight
        image_weight = image_weight / total_weight
        
        # Reshape weights for broadcasting
        voxel_weight = voxel_weight.view(-1, 1, 1, 1)  # [4, 1, 1, 1]
        image_weight = image_weight.view(-1, 1, 1, 1)  # [4, 1, 1, 1]

        return voxel_weight, image_weight, upscaled_image_feature

def fuse_features(sparse_feature, dense_feature, sparse_weight, dense_weight):
        # Apply weights to the respective features
    weighted_voxel_feature = sparse_weight * sparse_feature  # Weighted voxel feature
    weighted_image_feature = dense_weight * dense_feature  # Weighted image feature

    # Combine the weighted features to produce the fused feature map
    fused_feature = weighted_voxel_feature + weighted_image_feature


    return fused_feature
