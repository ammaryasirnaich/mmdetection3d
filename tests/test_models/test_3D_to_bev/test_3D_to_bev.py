from basic_block_2d import *
import torch.nn.functional as F

from mmdet3d.registry import MODELS


num_voxels = torch.randint(1, 100, [97297])
features = torch.rand([97297, 20, 5])
coors = torch.randint(0, 100, [97297, 4])


hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
hard_simple_VFE = MODELS.build(hard_simple_VFE_cfg)
outputs = hard_simple_VFE(features, num_voxels, None)

print("coors", coors[:,1:].shape)
print("feature shape", outputs.shape)

outputs = torch.concat((outputs,coors[:,1:]),dim=2)
print("output shape", outputs.shape)
batch_dict={}

# bevConvert = Conv2DCollapse(cfg,grid_size=[0.05,0.05,0.02])
bevConvert = Conv2DCollapse(in_channels=5,output_shape=[400,600],grid_size=[80,70,3])
batch_dict['voxel_features']=outputs

x = bevConvert(batch_dict)

print("output shape", x.shape)

print(bevConvert)





