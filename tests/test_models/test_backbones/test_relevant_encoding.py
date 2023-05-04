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
    
    features = torch.rand([240000, 10, 5])
    num_voxels = torch.randint(1, 10, [240000])
    coord = hard_simple_VFE(features, num_voxels, None)

    assert coord.shape == torch.Size([240000, 5])


    featuresV = torch.rand([93943, 10, 5]).cuda()
    num_voxelsV = torch.randint(1, 10, [93943]).cuda()
    coord = hard_simple_VFE(featuresV, num_voxelsV, None)

    relative = torch.rand([1,1024*3, 50, 3]).cuda()

    # print("relative_first", relative.shape)
    # test = relative[:, 0:1024, None, :]
    # print("shape_first", test.shape)




    # print("relative_second", relative.shape)
    # test2 = relative[:, None, :, :]
    # print("shape_second", test2.shape)

    generate_relative_encoding(relative)

    # rel_pos_1 = relative[:, :, None, :] - relative[:, None, :, :]

    # print("rel_pos shape", rel_pos_1.shape)

    # rel_pos_2 = relative[:, :, None, :] - relative[:, None, :, :]



    # rel_pos_3 = relative[:, :, None, :] - relative[:, None, :, :]
    # rel_pos_4 = relative[:, :, None, :] - relative[:, None, :, :]


    # rel_pos_5 = relative[:, :, None, :] - relative[:, None, :, :]
    # rel_pos_6 = relative[:, :, None, :] - relative[:, None, :, :]
    # rel_pos_7 = relative[:, :, None, :] - relative[:, None, :, :]
    # rel_pos_8 = relative[:, :, None, :] - relative[:, None, :, :]


    # global_rel_pos = torch.concat([rel_pos_1,rel_pos_2],dim=1)
    # print("global_rel_pos",global_rel_pos.shape)



    # assert coord.shape == torch.Size([93943, 5])

def generate_relative_encoding(coord: torch.tensor):
    start =0
    last_limit = coord.shape[1]
    stride = 1024
    global_rel_pos = torch.empty((1,1024,50,3), dtype=torch.int16)
    # print("last limit", last_limit)
    cycles = int(last_limit/stride)
    # print("cycles", cycles)
    for i in range(cycles):
        end = stride*(i+1)
        # print("start",start, " ","End ", end)   
        rel_pos= coord[:, start:end, None, :] - coord[:, None, start:end, :]
        start = end
        if i==0:
            global_rel_pos = rel_pos
        else:
            # print("global_rel_pos",global_rel_pos.shape)
            # print("rel_pos",rel_pos.shape)
            global_rel_pos = torch.concat([global_rel_pos,rel_pos],dim=1)
    

    print("global_rel_pos",global_rel_pos.shape)

    return  global_rel_pos


        


    # print("mode", int((1024*2)/1024))


    
    



    print("pass")

    # assert outputs.shape == torch.Size([97297, 4])


if __name__ == "__main__":
    test_hard_simple_VFE()