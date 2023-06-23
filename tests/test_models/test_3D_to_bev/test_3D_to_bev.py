
import torch.nn.functional as F
import torch

from mmdet3d.registry import MODELS
from mmdet3d.models.utils.bev_pool import dev_pool


num_voxels = torch.randint(1, 100, [97297])
features = torch.rand([97297, 20, 5])
coors = torch.randint(0, 100, [97297, 4])


hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
hard_simple_VFE = MODELS.build(hard_simple_VFE_cfg)
outputs = hard_simple_VFE(features, num_voxels, None)

print("coors", coors[:,1:].shape)
print("feature shape", outputs.shape)

outputs = torch.concat((outputs,coors[:,1:]),dim=2)
print("Pass")





