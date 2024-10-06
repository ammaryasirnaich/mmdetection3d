import torch.nn as nn
import torch.nn.functional as F
import torch


# from .fusion import AdaptiveWeight
# from .adaptive_feature_bevfusion import AdaptiveMultiStageFusionBEV
# from .adaptive_resolution_scaling_net import AdaptiveResolutionScalingNetwork
from einops import rearrange
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Refine_Resolution_Adjacement(nn.Module):
    def __init__(self,  
                 adaptive_fusion_cfg: OptConfigType = None,
                 adaptive_scale_net_cfg: OptConfigType = None,):
        
        super().__init__()
         
        # self.adaptive_weight = AdaptiveWeight(voxel_dim=512, image_dim=64,upscale_size=(200, 176)).cuda()
        # self.adaptive_resol_scale_model = AdaptiveResolutionScalingNetwork(in_channels=512, n_ref_points=4)
        
        # self.adaptive_weight = MODELS.build(adaptive_fusion_cfg)
        self.adaptive_fusion = MODELS.build(adaptive_fusion_cfg)
        self.adaptive_resol_scale_model = MODELS.build(adaptive_scale_net_cfg)
        
        
    def forward(self,bev_lidar_features,dense_camera_feature):
        # sparse_features : # [B, N, 256, H, W]
        # dense_feature   : #  [B, C, H, W]
                      
        # Adaptive Fusion between lidar and image features
        fused_feature = self.adaptive_fusion(bev_lidar_features[0], dense_camera_feature) #  [B, C, H, W] , [B, N, 256, H, W]
        output = self.adaptive_resol_scale_model(fused_feature) #  output [B, C, H, W] , comp_map 
        return output
        

if __name__=="__main__":
    batch_size = 4
    num_views = 6
    num_classes = 10
    height, width = 64, 176
    channel = 256
    voxel_channel=512
    bev_hight=180  #200
    bev_width=180  # 176
    dense_image_feature = torch.rand(batch_size, num_views, channel, height, width)
    sparse_feature      = torch.rand(batch_size, voxel_channel, bev_hight, bev_width)
    
    
    model = Refine_Resolution_Adjacement()
    out = model(sparse_feature,dense_image_feature)
    print(out.shape)
    print("pass")
    