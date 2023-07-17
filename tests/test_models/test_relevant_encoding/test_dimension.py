import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import numpy as np



# B,N,C = (1,512,256)

# print(B,N,C)

# x = torch.randn([1,512,4,256],device='cuda')

# x = x.transpose(1, 2).reshape(B, N, C)

# print(x.shape)

B,N,C = (64,196,432)
x = torch.randn([64, 196, 9, 48],device='cuda')
x=x.transpose(1,2).reshape(B,N,C)
print(x.shape)

pos_score = torch.randn([4, 512, 4, 64],device='cuda')
patch_score =torch.randn([4, 512, 4, 64],device='cuda')
gating = torch.randn([1, 1, 4, 1],device='cuda')
attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
print(attn.shape)