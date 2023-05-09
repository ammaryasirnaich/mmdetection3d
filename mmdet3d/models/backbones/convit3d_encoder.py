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

# from mmcv.cnn.utils.weight_init import trunc_normal_
# from mmcv.runner import BaseModule, ModuleList, _load_checkpoint

# from mmcv.utils import to_2tuple

from mmengine.utils import to_2tuple
# from ..builder import BACKBONES
# from mmdet3d.models.builder import BACKBONES
# from ...utils import get_root_logger


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
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
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
        # x : voxel-wise feature (B,V,P,D)
        print("voxel feature shape of input", x.shape)
        print(" voxel coordinate shape",  voxel_coord.shape)

        x = x[:,:,:1,:].permute(2,0,1,3).squeeze(0) # taking only one point from each voxel
        print("reshaping input shape", x.shape)

        # voxel_coords
        # rel_pos = pos[:, :, None, :] - pos[:, None, :, :]

        B, N, C = x.shape   # batch, num_of_points, features
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            # self.get_rel_indices(N)
            # self.get_rel_indices_3d(num_patches=N)
            self.get_patch_wise_relative_encoding(voxel_coord)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape        
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1,-1)

        print("pos_score dimensions", pos_score.shape)

        pos_score = self.pos_proj(pos_score).permute(0,3,1,2) 
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
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


    
    def get_rel_indices_3d(self, patches_loc=None,num_patches=None,dim=3):
        if patches_loc is None:
            assert num_patches is not None
            grid_size = round(num_patches**(1/dim))
            if dim==2:
                xi,yi = np.meshgrid(np.arange(grid_size),np.arange(grid_size))
                patches_loc = np.c_[xi.reshape(-1),yi.reshape(-1)]
            else:
                xi,yi,zi = np.meshgrid(np.arange(grid_size),np.arange(grid_size),np.arange(grid_size))
                patches_loc = np.c_[xi.reshape(-1),yi.reshape(-1),zi.reshape(-1)]

            patches_loc = torch.tensor(patches_loc)

        else:
            num_patches = patches_loc.shape[0]
            dim = patches_loc.shape[1]
            
        rel_ind =[]
        for i in range(dim):
            Xi = patches_loc[:,i] - patches_loc[:,i][:,None]
            rel_ind.append(Xi)

        rel_ind = torch.stack(rel_ind).permute([1,2,0])
        rel_ind = torch.cat([rel_ind, ((rel_ind**2).sum(2)).unsqueeze(2)],dim=2)
        if dim==3: rel_ind = rel_ind[:,:,[2,0,1,3]]

        device = self.qk.weight.device
        self.rel_indices = rel_ind.to(device)

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
        last_limit = coord.shape[0]
        # print("last_limit",last_limit)
        stride = 1024
        repeat_cycles = int(last_limit/stride)

        relative = coord[ 0:stride, None, :] - coord[ None, 0:stride, :]

  

        # print("shape of relative" , relative.shape)
        leftover = last_limit-(stride*repeat_cycles)
        # print("remains of points", leftover)
        if(leftover!=0):
            relative = relative.repeat( repeat_cycles+1, 1, 1)
            relative = relative[:last_limit,:,:]
            # print("final shape after clipping", relative.shape)
        else:
            relative = relative.repeat( repeat_cycles, 1, 1)
        # print("global_rel_pos",relative.shape)
        # device = self.qk.weight.device
        # self.rel_indices = relative.to(device)
        # if(relative.is_cuda): print(" using cuda")

       

 
class MHSA(BaseModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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

            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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

        print("dim=",dim)

    def forward(self, x, voxel_coords):
        x = x + self.drop_path(self.attn(self.norm1(x),voxel_coords))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(BaseModule):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.apply(self._init_weights)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class HybridEmbed(BaseModule):
    """ CNN Feature Map Embedding, from timm
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, BaseModule)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



@MODELS.register_module()
class ConViT3DDecoder(BaseModule):
    """ 
    ConViT3DDecoder: Using End-to-End Transformers paradigam which behaves also as convolution for early layers and fully attention at later layers
    """
    
    """ ConViT3DDecoder
    Args:
        img_size (int | tuple[int]): The size of input image when  pretrain. Defaults: 224.
        patch_size: (int | tuple[int]): Patch size. Default: 4.
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
                num_classes=1000 ,
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
                fp_channels = ((576,16)) # (head*embed_dim , output_dim)

                ):
        
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed


        ### Voxel Encoder will be doing the embedding, we will get the embedding in the form of voxel-features
        print("embed_dim=",embed_dim)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches


        ## call the PointNet2SASSG_SL backbone for genearting point feature embedding    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
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


    def forward_features(self, point_embeddings_dic, voxel_coors): # 

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
        print("shape of voxel_coors", voxel_coors.shape)
        x = point_embeddings_dic["voxels"]  # (B,V,P,D)

        print("point_embeddings_dic[voxels]", x.shape)

        B = x.shape[0]
        # x = point_embeddings_dic["voxels"]   #.expand(B,-1,-1,-1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
    
        
        # to-do, similar like
        '''
        pos = coords.permute(0, 2, 1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos = rel_pos.sum(dim=-1)  
        fused_features = voxel_features + self.point_features(features, rel_pos)
        
        ''' 

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            if u == self.local_up_to_layer :
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x,voxel_coors)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, voxel_coors):

        x = self.forward_features(x, voxel_coors)
        
        x = self.head(x)
        return x
    

 









     

   