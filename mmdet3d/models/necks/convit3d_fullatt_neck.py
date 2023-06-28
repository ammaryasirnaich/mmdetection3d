# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn import *
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from mmengine.model.weight_init import trunc_normal_
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple

# from mmdet.models.utils.transformer import PatchEmbed, PatchMerging
from mmdet3d.models.backbones import PointNet2SASSG
import numpy as np




def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class RelPositionalEncoding3D(nn.Module):
    def __init__(self, input_dim, max_points):
        super(RelPositionalEncoding3D, self).__init__()
        self.input_dim = input_dim
        self.max_points = max_points
        
        self.position_encoding = nn.Embedding(self.max_points, self.input_dim)

    def forward(self, points):
        '''
        points: 3D point cloud (B,N,D)
        return: relevant position encoding cooridnates(3) with pairwise eucliden distance(1) (B,N,N,4) 
        
        '''
        batch_size, num_points, _ = points.size()
        
        # Compute relative coordinates
        relative_coords = points[:, :, None, :] - points[:, None, :, :]
        
        # Compute pairwise distances
        distances = torch.sqrt(torch.sum(relative_coords ** 2, dim=-1))  # Euclidean distance
        
        # Compute position encoding
        position_indices = torch.arange(num_points, device=points.device).unsqueeze(0).expand(batch_size, -1)
        position_encodings = self.position_encoding(position_indices)
        
        # Expand position encodings to match the shape of distances
        position_encodings = position_encodings.unsqueeze(2).expand(-1, -1, num_points, -1)
        
        # Concatenate position encodings with distances
        encodings = torch.cat([position_encodings, distances.unsqueeze(-1)], dim=-1)
        
        return encodings


class DropPath(BaseModule):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(BaseModule):
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


class GPSA(BaseModule):
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
        
    def forward(self, x, voxel_coord):
        
        # if x.shape[1]>16000:
        #     print("Voxel shape", x.shape)
        #     print("special check on large number of voxels")  
        #     if hasattr(self, 'rel_indices'): print("self.rel_indices.size(1):",self.rel_indices.size(0))
        # x : voxel-wise feature (B,V,P,D)
        # x = x.permute(2,0,1,3).squeeze(0) # taking only one point from each voxel
        # voxel_coords
        # rel_pos = pos[:, :, None, :] - pos[:, None, :, :]

        B, N, C = x.shape   # batch, num_of_points, features
        # print("shape of x" , x.shape)
        # print("shape of feature", voxel_coord.shape)
        if not hasattr(self, 'rel_indices'):   #or self.rel_indices.size(0)!=N:

            # self.get_rel_indices(N)
            # self.get_rel_indices_3d(num_patches=N)
            # self.get_patch_wise_relative_encoding(voxel_coord)
            rel_ind = self.embd_3d_encodding(voxel_coord)
            self.rel_indices = rel_ind

        attn = self.get_attention(x) 
        # v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("shape of attention", attn.shape)
        x = attn.transpose(1, 2).reshape(B, N, C)
        # print("shape of attention", x.shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        # print("shape of input in get_attention",x.shape)
        # print("Q/k Dimension input :",self.dim,"output: ",self.dim*2)
      
        B, N, C = x.shape      

        # print(" reshape dimension", B, N, 2, self.num_heads, C // self.num_heads) 

        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   
        
        q, k = qk[0], qk[1]

        # print("Q Dimension", q.size)

        print("self.rel_indices.shape: ",self.rel_indices.shape)
        
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        
        # print("+ R dimension", pos_score.shape)
        # print("pos_score dimensions", pos_score.shape)

        # default       
        # pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        # print("pos_score shape",pos_score.shape)
        # pos_score = pos_score.softmax(dim=-1)
        
       
        # patch_score = (q @ k.transpose(-2, -1)) * self.scale
        # patch_score = patch_score.softmax(dim=-1)

        '''
        Memory Efficient Attention Pytorch: https://arxiv.org/abs/2112.05682
        Self-attention Does Not Need O(n2) Memory
        '''
        
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        pos_score = pos_score.softmax(dim=-1)
        # print("pos_score shape",pos_score.shape)
        # print("shape of v", v.shape)
        # print("truncted-shape of v", v[:,:,:pos_score.size(-1),:].shape)
        pos_score = pos_score @ v[:,:,:pos_score.size(-1),:]
        # print("pos_score @ V shape",pos_score.shape)

        # p = q.shape[-2]
        # print("I shape" , q.dtype)
        # l1=torch.eye(p,p,dtype=torch.float32, device="cuda")
        # print("shape of l1" , l1.shape)
        # attn_qk = F.scaled_dot_product_attention(q,k,l1)
        # v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # patch_score = F.scaled_dot_product_attention(attn_qk,l1*torch.sqrt(attn_qk.size(-1)),v)
        
        patch_score = F.scaled_dot_product_attention(q,k,v,scale=self.scale ,dropout_p=0.0)

        gating = self.gating_param.view(1,-1,1,1)

        # print("patch_score shape ", patch_score.shape)
        # print("pos_score shape ", pos_score.shape)
        # print("shape of gating", gating.shape)

        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    #Note: To be inspected
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

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)

    
    #Note: To be inspected
    def get_patch_wise_relative_encoding(self,coord: torch.tensor):
        '''
        Arg:
        coord (tensor): (V,D) shape tensor containing the voxel cooridnates 

        Return:
        rel_indices (): (V,1024,D) shape tensor containing relative indices for each voxel relative to
        the patch/block containing 1024     
        '''
        # start =0
        # print("shape of input", coord.shape)
        last_limit = coord.shape[1]
        # print("last_limit",last_limit)
      
        stride = 1024
   
        if(stride>last_limit):
            relative = coord[:, 0:last_limit, None, :] - coord[:, None, 0:last_limit, :]
            relative_distance = relative.sum(dim=-1, dtype = torch.float32)
            B,V,P = relative_distance.shape
            # print("B,V,P",B,V,P)
            relative_distance = relative_distance.view(B,V,P,1)
            self.rel_indices  = torch.concat([relative,relative_distance],dim=3)
            # print("shape of indices",self.rel_indices.shape)  

        else:

            repeat_cycles = int(last_limit/stride)
            # print("repeat_cycles",repeat_cycles)

            relative = coord[:, 0:stride, None, :] - coord[:, None, 0:stride, :]

            # print("shape of relative" , relative.shape)
            
            leftover = last_limit-(stride*repeat_cycles)
            # print("remains of points", leftover)
            
            if(leftover!=0):
                relative = relative.repeat(1, repeat_cycles+1, 1, 1)
                relative = relative[:,:last_limit,:,:]
                # print("final shape after clipping", relative.shape)
            else:
                relative = relative.repeat(-1, repeat_cycles, 1, 1)
            # print("relative shape",relative.shape)
            relative_distance = relative.sum(dim=-1, dtype = torch.float32)
            # print("relative_distance shape",relative_distance.shape)
            # print("content value before view", relative_distance[1,:4])

            B,V,P = relative_distance.shape
            relative_distance = relative_distance.view(B,V,P,1)        
            # print("relative_distance shape",relative_distance.shape)

            # print("content value after view", relative_distance[1,:4,])
            self.rel_indices = relative.unsqueeze(0)
            self.rel_indices  = torch.concat([relative,relative_distance],dim=3)
            # print("dist_rel_indices shape ",self.rel_indices.shape)
            # print("Pass")
    


        
 
class MHSA(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_attn = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(self.drop_attn)
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

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N**.5)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        distances = indd**.5
        distances = distances.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        
        if return_map:
            return dist, attn_map
        else:
            return dist

            
    def forward(self, x, _):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q,k,v,scale=self.scale ,dropout_p= self.drop_attn)
        x = x.transpose(1, 2).reshape(B, N, C)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(BaseModule):

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
class FullConViT3DNeck(BaseModule):
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
                img_size=224 ,
                patch_size=16 ,
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
                fp_output_channel = 16 # embed_dim, num_classes

                ):
        
        super().__init__(init_cfg=init_cfg,
                        #  num_classes = num_classes,
                        #  local_up_to_layer=local_up_to_layer,
                        #  embed_dim=embed_dim,
                        #  locality_strength=locality_strength,
                        #  use_pos_embed=use_pos_embed,
                        #  fp_output_channel=fp_output_channel,
                        #  depth=depth,
                        #  num_heads=num_heads
                         )

        
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.entry_counter=0
        self.depth = depth
        self.fp_output_channel=fp_output_channel
        self.num_heads=num_heads

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(self.depth)])
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        # self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(self.embed_dim, self.fp_output_channel) #if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.cls_token, std=.02)
        self.head.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, feat_dic, voxel_coors): # 
        # self.entry_counter = self.entry_counter+1
        # print("No of Enteries into backbone:",self.entry_counter)

        # x = self.patch_embed(x)

        # embedding using single PointNet
        # Example, Branch   SA(512,0.4,[64,128,256]) , meansing using 512x4 points and using radius 0.4 and 
        # x = self.point_embed(x,coors)

        # xyz = point_embeddings["fp_xyz"][-1]  # (B,V*P,3)
        # features = point_embeddings["fp_features"][-1].permute(0,2,1).contiguous()  # (B,V*P,D)
        # print("xyz coordinates", xyz.shape)
        # print("features dimensions", features.shape)
        # x = torch.cat((xyz,features),dim=2)  # (B,V*P,3+D)
        # print(" combined values" , x.shape)

        # voxel = point_embeddings["voxels"]

        # B = xyz.shape[0]

        # pos = voxel_coors
        # x = point_embeddings_dic["voxels"]  # (B,V,P,D(xyz(3)+feature(16)))
        x = feat_dic["sa_features"][-1] # (B,V,P,D)
        # x = x[:,:,:1,:].squeeze(2)#
        x = x.permute(0,2,1)
       
        # print("x feature", x.shape)
        # print("Input feature to Block:", x.shape)
        # print("Input voxel to Block:", voxel_coors.shape)

        B = x.shape[0]

        # x = point_embeddings_dic["voxels"]   #.expand(B,-1,-1,-1)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
    
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            # print("No of Block#", u)
            # if u == self.local_up_to_layer :
            #     x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x,voxel_coors)
            # print("Output from Block:",u," is of shape", x.shape)

        x = self.norm(x)
        # print("Output after normalization", x.shape)

        #update the feature
        feat_dic["fp_features"] = x
        return feat_dic
    
    def forward(self, feat_dic, voxel_coors):
        # print("Input to ConViT Model:")
        # print("Voxel Feature of shape from pipline:",x["fp_features"].shape)
        feat_dic = self.forward_features(feat_dic, voxel_coors)

        # print(" shape of final output from the attention model", x.shape)
        # feat_dict=[]       
        # feat_dict['sa_xyz']= []
        # feat_dict['sa_features']=x
        # feat_dict['sa_indices']=[]
        # x = self.head(x)

        return feat_dic
    

 









     

   