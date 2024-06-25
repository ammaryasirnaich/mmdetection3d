from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .bev import bev_3D_to_2D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



"""
class RelPositionalEncoding3D(nn.Module):
    def __init__(self, input_dim, max_points):
        super(RelPositionalEncoding3D, self).__init__()
        self.input_dim = input_dim
        self.max_points = max_points

    def forward(self, points):
        
        batch_size, num_points, _ = points.size()
       # Compute relative positions
        points_1 = points.unsqueeze(1).expand(-1, num_points, -1, -1)
        points_2 = points.unsqueeze(2).expand(-1, -1, num_points, -1)
        positions = points_1 - points_2

        # Compute pairwise distances
        distances = torch.norm(positions, dim=-1)

        # Concatenate positions and distances
        encodings = torch.cat([positions, distances.unsqueeze(-1)], dim=-1)
        return encodings
"""

class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)       
        self.v = nn.Linear(dim, dim, bias=qkv_bias)       
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(4, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        # self.embd_3d_encodding = RelPositionalEncoding3D(3,dim)
       
        self.apply(self._init_weights)
        
        if use_local_init:
            # self.local_init(locality_strength=locality_strength) # for 2d image data
            # self.local_init_3d(locality_strength=locality_strength)  # for 3d point cloud
            self.local_init_3d_relaxed(locality_strength=locality_strength)  # for 3d point cloud

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self,  x, voxel_coord):
        "forward with scaled dot attention mechanism"
        B, N, C = x.shape

        if not hasattr(self, 'rel_indices') or self.rel_indices.shape[0]!=B:
            # self.rel_indices = self.embd_3d_encodding(voxel_coord)
            
            # print(f"shape of self.rel_indices is {self.rel_indices.shape}") 
            # print(f"shape of voxel_coord {voxel_coord.shape}")
            self.rel_indices = self.RelPositionalEncoding3D_test(voxel_coord)
            # print(f"shape of self.rel_indices is {test_shape.shape}")
           
        attn = self.get_attention(x)
        # print(f" x attn shape {x.shape}")
        attn = self.proj(attn) 
        attn = self.proj_drop(attn)

        return attn

    
    def RelPositionalEncoding3D_test(self, point_clouds):
  
        """
        Calculate the relative position vectors and distances for a batch of 3D point clouds.

        Args:
        point_clouds (torch.Tensor): Tensor of shape (batch_size, num_points, 3) containing
                                    the x, y, z coordinates for each point in each point cloud.

        Returns:
        torch.Tensor: A 4D tensor of shape (batch_size, num_points, num_points, 4) containing the 
                    relative position vectors and Euclidean distances between each pair of points 
                    in the point clouds.
        """
       
        # print(f"point_clouds shape {point_clouds.shape}")
       
        # Ensure the input is a floating point tensor for accurate calculations
        # point_clouds = point_clouds.float()
        

        # Expand point_clouds to shape [batch_size, num_points, 1, 3]
        point_clouds_expanded = point_clouds.unsqueeze(2)
        
        # Expand point_clouds again in another dimension to shape [batch_size, 1, num_points, 3]
        point_clouds_expanded_again = point_clouds.unsqueeze(1)
        
        # Subtract the expanded tensors to get relative positions: shape [batch_size, num_points, num_points, 3]
        relative_positions = point_clouds_expanded - point_clouds_expanded_again
        
        # Calculate the Euclidean distance: shape [batch_size, num_points, num_points]
        distances = torch.norm(relative_positions, dim=3, keepdim=True)
        
        # Concatenate the relative positions with their corresponding distances
        relative_positions_with_distances = torch.cat((relative_positions, distances), dim=3)
        
        return relative_positions_with_distances


    def get_attention(self, x):    
        B, N, C = x.shape      
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   
        
        q, k = qk[0], qk[1]

        # print("input shape to attention GPSA", x.shape)
       

        # '''
        # Memory Efficient Attention Pytorch: https://arxiv.org/abs/2112.05682
        # Self-attention Does Not Need O(n2) Memory
        # '''
        pos_score = self.rel_indices
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)  # V_pos*(r)
        pos_score = pos_score.softmax(dim=-1)


        I = torch.eye(k.shape[-2],k.shape[-2],device='cuda')
        # print("shape of !",I.shape)

        patch_score = F.scaled_dot_product_attention(q,k,I,scale=self.scale ,dropout_p=0.0)            
        gating = self.gating_param.view(1,-1,1,1)

        # print("Dimension mismatched")
        # print("patch_score shape",patch_score.shape)
        # print("pos_score shape", pos_score.shape)
        # print("gating shape", gating.shape)
        # print("self.rel_indices shaoe",self.rel_indices.shape)
        # print("q.shape: ", q.shape)
        # print("k.shape: ", k.shape)
        # print("v.shape: ", v.shape)
        # print("Wait")


        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        # print("attn shape", attn.shape)
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        # attn = attn.squeeze(0)
        # print("attn shape after unsqueeze", attn.shape)

        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return attn
   
    """ 
    def local_init(self, locality_strength=1.):
        
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    
        
    def local_init_3d(self, locality_strength=1.):
        position=0
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5) 
        kernel_size = int(self.num_heads**(1/3))
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                for h3 in range(kernel_size):    
                    # print(position,h1,h2,h3)
                    self.pos_proj.weight.data[position,3] = -1
                    self.pos_proj.weight.data[position,2] = 2*(h3-center)*locality_distance
                    self.pos_proj.weight.data[position,1] = 2*(h2-center)*locality_distance
                    self.pos_proj.weight.data[position,0] = 2*(h1-center)*locality_distance
                    position +=1
        self.pos_proj.weight.data *= locality_strength
        
    """
    
    def local_init_3d_relaxed(self,locality_strength=1.):
        '''
        This function will generate intial weights for number of heads = torch.math.ceil(num_heads**(1/3))**3
        and then randomly select weights for heads, num_heads
        note:
        for num_heads = 8, it works normaly, with kernel size of 2x2x2
        for num_heads >8 and  num_heads <=27, it generate weights for 27 heads with kernel size of 3x3x3
        and then randomly select weights to match the number of actual head.

    '''
        position=0
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        num_heads_exp = torch.math.ceil(self.num_heads**(1/3))**3
        kernel_size = int(torch.math.ceil(self.num_heads**(1/3)))
        # print(self.num_heads, '-->',num_heads_exp, 'kernel_size =',kernel_size)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        wdata = torch.zeros([num_heads_exp,4])


        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                for h3 in range(kernel_size):    
                    # print(position,h1,h2,h3)
                    wdata[position,3] = -1
                    wdata[position,2] = 2*(h3-center)*locality_distance
                    wdata[position,1] = 2*(h2-center)*locality_distance
                    wdata[position,0] = 2*(h1-center)*locality_distance
                    position +=1

        wdata *= locality_strength
        # print(wdata)
        idx = torch.randperm(wdata.shape[0])
        wdata = wdata[idx]
        # print(wdata)
        self.pos_proj.weight.data[:,0] = wdata[:self.num_heads,0]
        self.pos_proj.weight.data[:,1] = wdata[:self.num_heads,1]
        self.pos_proj.weight.data[:,2] = wdata[:self.num_heads,2]
        self.pos_proj.weight.data[:,3] = wdata[:self.num_heads,3]

        
        



class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_attn = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self,  x, _ ):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q,k,v,attn_mask=None,scale=self.scale ,dropout_p= self.drop_attn, is_causal=True).permute(0,2,1,3)
        B_t,N_t,H_t,D_t = attn.shape
        attn =attn.reshape(B_t,N_t,H_t*D_t)   

        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        
        return attn
    
class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # upsampling the features with the ratio of 4
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, voxel_coords):
        x = x + self.drop_path(self.attn(self.norm1(x),voxel_coords))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
@MODELS.register_module()
class VisionTransformer(nn.Module):
    """ 
    FullConViT3DNeck: Using End-to-End Transformers paradigam which behaves also as convolution for early layers and fully attention at later layers
    It uses full attention between query,key and value while using relative positional encoding
    
    """  
    """ 
    Args:
        in_chans: (int): The num of input channels. Defaults: 4. 
        num_classes:3 
        embed_dim (int): The feature dimension. Default: 96.
        depth(tuple[int]): Depths of each Swin Transformer stage. Default 12
        num_heads(tuple[int]): Parallel attention heads of Transformer stage. Default 12 
        mlp_ratio:4. 
        qkv_bias(bool, optional): If True, add a learnable bias to query, key, value. Default: False 
        qk_scale(float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        drop_rate:0. 
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.. 
        hybrid_backbone:None 
        norm_layer:nn.LayerNorm 
        global_pool:None
        local_up_to_layer (int): selecting initial layers to behave like convolution. Default10 
        locality_strength:1. 
        use_pos_embed:True
        init_cfg=None,
        pretrained=None

        ## new configuration
        radius=0.2, 
        nsample=64,

    """
    def __init__(self,
                # img_size=224 ,
                # patch_size=16 ,
                in_chans=4 ,
                num_classes=3 ,
                embed_dim=48 ,
                depth=12,
                num_heads=12 ,
                mlp_ratio=4,
                qkv_bias=False ,
                qk_scale=None ,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0, 
                hybrid_backbone=None ,
                norm_layer=nn.LayerNorm ,
                global_pool=None,
                local_up_to_layer=10 ,
                locality_strength=1,
                use_pos_embed=False,
                init_cfg=None,
                pretrained=None,
                use_patch_embed=False,
                fp_output_channel = 16, # embed_dim, num_classes
                rpn_feature_set = False,

                ):
        super(VisionTransformer,self).__init__()
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.entry_counter=0
        self.depth = depth
        self.fp_output_channel=fp_output_channel
        self.num_heads=num_heads
        self.use_patch_embed = use_patch_embed
              
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.i = 0
        self.rpn_feature_set = rpn_feature_set
        
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(self.depth)])
        
        self.norm = norm_layer(embed_dim)

        #Transformer head
        self.transformer_head = nn.Linear(self.embed_dim, self.fp_output_channel) #if num_classes > 0 else nn.Identity() 
        # self.transformer_head.apply(self._init_weights)
        
        
        # self.coordrefine = CoordinateRefinementModule(self.num_heads)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward_features(self, x, voxel_coors):
        # print("input to visualTransformer shape", x.shape)
        # print("voxel_coors to visualTransformer shape", voxel_coors.shape)
        x = x.permute(0,2,1)
        # x = self.pos_drop(x)            
        for u,blk in enumerate(self.blocks):
            # print("input after permute", x.shape)
            x = blk(x,voxel_coors)
        x = self.norm(x)

        return x

    def forward(self, feat_dict):
        x = feat_dict["sa_features"][-1]
        voxel_coors = feat_dict["sa_xyz"][-1]
        attend= self.forward_features(x, voxel_coors)
        #pass through transformer head
        # print("attend output shape before head",attend.shape)
        attend = self.transformer_head(attend)  
        # create new feature 
        
        if (self.rpn_feature_set):
            # feat_dict["fp_features"] = attend.contiguous()
            # feat_dict["fp_xyz"] = feat_dict["sa_xyz"][-1]
            feat_dict["sa_features"][-1] = attend.contiguous()
            x = torch.cat((feat_dict["sa_xyz"][-1][:,:,:3],feat_dict["sa_features"][-1]),dim=2)
            # print("cat fature",x.shape)
            x = bev_3D_to_2D(x).permute(0,3,1,2)
            # print("shape for rpn network x:", x.shape)
            # print(f'bev_image_{self.i}.pt')
            # torch.save(x, f'bev_image_{self.i}.pt')
            # self.i += 1
            return [x]
        else:
            feat_dict["sa_features"][-1] = attend.permute(0,2,1).contiguous()
            
        # print("fp_features shape",feat_dict["fp_features"].shape)
        # print("fp_xyz shape",feat_dict["fp_xyz"].shape)
        # print("attend output shape after permute",feat_dict["sa_features"][-1].shape)
        return feat_dict
    
    
    
'''
###### code analysis has to be done    
class CoordinateRefinementModule(nn.Module):
    def __init__(self, num_attention_heads):
        super(CoordinateRefinementModule, self).__init__()
        self.num_attention_heads = num_attention_heads

    def forward(self, transformer_output, centroid_points):
        """
        Args:
            transformer_output (Tensor): Output features from the last Local Transformer layer
                Shape: (batch_size, num_points, feature_dim)
            centroid_points (Tensor): Coordinates of the centroid points
                Shape: (batch_size, num_centroids, 3)

        Returns:
            refined_centroids (Tensor): Refined coordinates of the centroid points
                Shape: (batch_size, num_centroids, 3)
        """
        batch_size, num_points, feature_dim = transformer_output.shape
        _, num_centroids, _ = centroid_points.shape

        # Extract attention maps from the last Local Transformer layer
        attention_maps = transformer_output[:, :, :feature_dim // self.num_attention_heads]

        # Compute average attention map
        avg_attention_map = attention_maps.mean(dim=-1)  # (batch_size, num_points)

        # Select attention weights for the centroid points
        centroid_attention_weights = avg_attention_map[:, centroid_points[:, :, 0].long()]  # (batch_size, num_centroids)

        # Compute refined centroid coordinates as weighted average
        refined_centroids = torch.bmm(
            centroid_attention_weights.unsqueeze(1),
            centroid_points.permute(0, 2, 1)
        ).squeeze(1)  # (batch_size, num_centroids, (features)3)

        return refined_centroids
'''    
    