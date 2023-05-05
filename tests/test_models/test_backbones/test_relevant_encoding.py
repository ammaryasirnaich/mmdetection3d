# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS




def test_hard_simple_VFE():

    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    hard_simple_VFE_cfg = dict(type='HardSimpleVFE', num_features=5)
    hard_simple_VFE = MODELS.build(hard_simple_VFE_cfg)
    
    # features = torch.rand([240000, 10, 5])
    # num_voxels = torch.randint(1, 10, [240000])
    # coord = hard_simple_VFE(features, num_voxels, None)

    # assert coord.shape == torch.Size([240000, 5])


    # featuresV = torch.rand([93943, 10, 5]).cuda()
    # num_voxelsV = torch.randint(1, 10, [93943]).cuda()
    # coord = hard_simple_VFE(featuresV, num_voxelsV, None)


    '''
        Memory check for the overall feature matrics
    '''
    # relative = torch.rand([1,93943, 1024, 3]).cuda()
    # print("shape of overall relative matrics", relative.shape)
    # relative = torch.rand([1, 27648, 1024, 3]).cuda()
    # print("shape of overall relative matrics", relative.shape)
    

    # relative = torch.rand([1,1024*3, 50, 3]).cuda()


    pos = torch.rand([1,93943, 3]).cuda()
    generate_relative_encoding(pos)

    # print("relative_first", relative.shape)
    # test = relative[:, 0:1024, None, :]
    # print("shape_first", test.shape)


    # temp1 = torch.empty((1,93943,1024,3), dtype=torch.int16).cuda()
    # print("temp shape",temp1.shape)

    # temp1 = torch.empty((1024,1024,50,3), dtype=torch.int16).cuda()
    # print("temp shape",temp1.shape)


    # pos = torch.rand([1,1024, 3]).cuda()
    # # print("relative_second", relative.shape)
    # # test2 = relative[:, None, :, :]
    # # print("shape_second", test2.shape)


    # rel_pos_1 = pos[:, :, None, :] - pos[:, None, :, :]
    # print("rel_pos shape", rel_pos_1.shape)
    # rel_pos_2 = pos[:, :, None, :] - pos[:, None, :, :]
    # print("rel_pos shape", rel_pos_2.shape)
    # rel_pos_3 = pos[:, :, None, :] - pos[:, None, :, :]
    # # rel_pos_4 = pos[:, :, None, :] - pos[:, None, :, :]
    # # rel_pos_5 = pos[:, :, None, :] - pos[:, None, :, :]
    # # rel_pos_6 = pos[:, :, None, :] - pos[:, None, :, :]
    # # rel_pos_7 = pos[:, :, None, :] - pos[:, None, :, :]
    # # rel_pos_8 = pos[:, :, None, :] - pos[:, None, :, :]

    # global_rel_pos = torch.concat([rel_pos_1,rel_pos_2,rel_pos_3],dim=1)
    # print("global_rel_pos",global_rel_pos.shape)

    # assert coord.shape == torch.Size([93943, 5])

def generate_relative_encoding(coord: torch.tensor):
    # start =0
    last_limit = coord.shape[1]
    print("last_limit",last_limit)
    stride = 1024
    repeat_cycles = int(last_limit/stride)

    relative = coord[:, 0:stride, None, :] - coord[:, None, 0:stride, :]
   
    print("shape of relative" , relative.shape)
    leftover = last_limit-(stride*repeat_cycles)
    print("remains of points", leftover)
    if(leftover!=0):
        relative = relative.repeat(1, repeat_cycles+1, 1, 1)
        relative = relative[:,:last_limit,:,:]
        print("final shape after clipping", relative.shape)
    else:
        relative = relative.repeat(1, repeat_cycles, 1, 1)
    print("global_rel_pos",relative.shape)
    return  relative


def efficient_rel_pos():   #coord:torch.tensor

    A = torch.rand([1,1024, 1024, 3]).cuda()
    # A = torch.empty((3,2))
    print("shape of A", A.shape)
    K=91

    B = A.repeat(1, K, 1, 1)
    print("shape of B", B.shape)

    print("pass")

    # assert outputs.shape == torch.Size([97297, 4])


if __name__ == "__main__":
    test_hard_simple_VFE()
    # efficient_rel_pos()