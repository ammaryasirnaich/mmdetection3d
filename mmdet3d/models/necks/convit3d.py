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

# class RelPositionalEncoding3D(nn.Module):
#     def __init__(self, input_dim, max_points):
#         super(RelPositionalEncoding3D, self).__init__()
#         self.input_dim = input_dim
#         self.max_points = max_points
        
#         self.position_encoding = nn.Embedding(self.max_points, self.input_dim)
#         

#     def forward(self, points):
#         '''
#         points: 3D point cloud (B,N,D)
#         return: relevant position encoding cooridnates(3) with pairwise eucliden distance(1) (B,N,N,4) 
        
#         '''
#         batch_size, num_points, _ = points.size()
        
#         # Compute relative coordinates
#         relative_coords = points[:, :, None, :] - points[:, None, :, :]
        
#         # Compute pairwise distances
#         distances = torch.sqrt(torch.sum(relative_coords ** 2, dim=-1))  # Euclidean distance
        
#         # Compute position encoding
#         position_indices = torch.arange(num_points, device=points.device).unsqueeze(0).expand(batch_size, -1)
        
#         # position_encodings = self.position_encoding(position_indices)
        
        
#         # Expand position encodings to match the shape of distances
#         position_encodings = position_encodings.unsqueeze(2).expand(-1, -1, num_points, -1)
        
#         # Concatenate position encodings with distances
#         encodings = torch.cat([position_encodings, distances.unsqueeze(-1)], dim=-1)
#         self.register_buffer("encodings", encodings)   
#         return encodings

class RelPositionalEncoding3D(nn.Module):
    def __init__(self, input_dim, max_points):
        super(RelPositionalEncoding3D, self).__init__()
        self.input_dim = input_dim
        self.max_points = max_points

        # self.position_encodings = nn.Parameter(torch.randn(max_points, input_dim))

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
        self.embd_3d_encodding = RelPositionalEncoding3D(3,dim)
        self.apply(self._init_weights)
        
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    '''      
    def forward(self,  x, voxel_coord):
        B, N, C = x.shape


        if not hasattr(self, 'rel_indices'):
            # self.get_patch_wise_relative_encoding(voxel_coord)
            # dumy_rel= torch.randn(4,64,64,4, device='cuda')
            self.rel_indices = self.embd_3d_encodding(voxel_coord)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    '''

    def forward(self,  x, voxel_coord):
        "forward with scaled dot attention mechanism"
        B, N, C = x.shape

        if not hasattr(self, 'rel_indices') or self.rel_indices.shape[0]!=B:
            self.rel_indices = self.embd_3d_encodding(voxel_coord)
           
        attn = self.get_attention(x)
        # print(" x attn", x.shape)
       
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        if(B!= attn.shape[0]):
            print("Batch mismatched occured in GPSA")
            print("Input batch shape:", x.shape, ", ouput batch shape:", attn.shape )
            print("Voxelcoord shape:", voxel_coord.shape)

        return attn


    def get_attention(self, x):

        # B, N, C = x.shape  
        # qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k = qk[0], qk[1]
        # pos_score = self.rel_indices
        # pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        # patch_score = (q @ k.transpose(-2, -1)) * self.scale


        # patch_score = patch_score.softmax(dim=-1)
        # pos_score = pos_score.softmax(dim=-1)

        # gating = self.gating_param.view(1,-1,1,1)


        # if(patch_score.shape!=pos_score.shape):
        #     print("patch_score shape", patch_score.shape)
        #     print("Pos_score shape",pos_score.shape)
        #     print("gating shape", gating.shape)
         

        # attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        # attn /= attn.sum(dim=-1).unsqueeze(-1)
        # attn = self.attn_drop(attn)

        # print("shape of input in get_attention",x.shape)
        # print("Q/k Dimension input :",self.dim,"output: ",self.dim*2)
      
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
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        pos_score = pos_score.softmax(dim=-1)


        I = torch.eye(k.shape[-2],k.shape[-2],device='cuda')
        # print("shape of !",I.shape)

        patch_score = F.scaled_dot_product_attention(q,k,I,scale=self.scale ,dropout_p=0.0)
        # patch_score = patch_score.softmax(dim=-1)

        # p_B,p_N,p_H,p_D = patch_score.shape
        # patch_score =patch_score.reshape(p_B,p_N,p_H*p_D)

        # s_B,s_N,s_H,s_D = pos_score.shape
        # pos_score =pos_score.reshape(s_B,s_N,s_H*s_D)
             
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
      
        # attn=attn.transpose(1,2).reshape(B,N,C)


        # print("attn shape after rearranging", attn.shape)

        return attn
    
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
        # B, N, C = x.shape

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)    
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # x = self.proj(x)
        # x = self.proj_drop(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q,k,v,attn_mask=None,scale=self.scale ,dropout_p= self.drop_attn, is_causal=True).permute(0,2,1,3)
        
        # print("attn shape after scaled attention", attn.shape)
        
        # attn = torch.einsum('bijk->bjik', attn)
        # print("attn shape", attn.shape)
        B_t,N_t,H_t,D_t = attn.shape
        attn =attn.reshape(B_t,N_t,H_t*D_t)   
        # attn = attn.transpose(1, 2).reshape(B, N, C) 
        # print("attn shape after reshape", attn.shape)
        

        attn = self.proj(attn)
        attn = self.proj_drop(attn)


        if(B!= attn.shape[0]):
            print("Batch mismatched occured in MHSA")
            print("Input batch shape:", x.shape, ", ouput batch shape:", attn.shape )
            print("Voxelcoord shape:", _.shape)
        
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
        mlp_hidden_dim = int(dim * mlp_ratio)
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
                fp_output_channel = 16 # embed_dim, num_classes
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

        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        #Transformer head
        self.transformer_head = nn.Linear(self.embed_dim, self.fp_output_channel) #if num_classes > 0 else nn.Identity()
       
        self.transformer_head .apply(self._init_weights)

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

    def forward(self, feat_dict, voxel_coors):
        x = feat_dict["sa_features"][-1]
        attend= self.forward_features(x, voxel_coors)

        # print("output shape from forward_features",attend.shape)

        #pass through transformer head
        attend = self.transformer_head(attend)
        
        # create new feature 
        # feat_dict["tranformer_features"]= attend 
        feat_dict["sa_features"][-1] = attend       
        return feat_dict
    
    