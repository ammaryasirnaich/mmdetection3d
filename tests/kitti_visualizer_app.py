import torch
import numpy as np
import  matplotlib.pyplot as plt
from mmdet3d.visualization import Det3DLocalVisualizer
from mmengine import load

from mmengine.structures import InstanceData
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from mmdet3d.visualization.vis_utils import (proj_camera_bbox3d_to_img, proj_depth_bbox3d_to_img,
                        proj_lidar_bbox3d_to_img, to_depth_mode)
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)




from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS



# def get_gt_instance(filename:str):
#     info_file = load(filename)

#     bboxes_3d = []
#     labels_3d = []

#     gt_instances_3d = InstanceData()

#     for instance in info_file['data_list'][0]['instances']:
#         bboxes_3d.append(instance['bbox_3d'])
#         labels_3d.append(instance['bbox_label_3d'])

#     bbox3d = torch.tensor(bboxes_3d)
#     labels_3d = torch.tensor(labels_3d)

#     gt_instances_3d.bboxes_3d = LiDARInstance3DBoxes(bbox3d)
#     gt_instances_3d.labels_3d = labels_3d

#     # gt_det3d_data_sample = Det3DDataSample()
#     # gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
#     return gt_instances_3d



def get_3dbbox_from_pklfile(file:str):
    info_file = load(file)   # ['metainfo', 'data_list'
     
    gt_instances_3d = InstanceData()
    # Traverse the data to extract 3D bounding box information
    print(info_file['data_list'][0].keys()) 

    bboxes_3d = []
    labels_3d = []  

    ### sample_idx', 'images', 'lidar_points', 'instances', 'cam_instances
    for info_dic in info_file['data_list']:
            # print(type(info_dic))
            lidar_info_str= info_dic['lidar_points']['lidar_path']
            # print(type(lidar_info_dic))
            # print(lidar_info_str) 
            if(lidar_info_str == '000003.bin'):
                # print(type(info_dic['instances']))
                # print(len(info_dic['instances']))
                for instance_dic in info_dic['instances']:
                    
                    bboxes_3d.append(instance_dic['bbox_3d'])
                    labels_3d.append(instance_dic['bbox_label_3d'])
                    # print(instance_dic['bbox_3d'])
                bbox3d = torch.tensor(bboxes_3d)
                labels_3d = torch.tensor(labels_3d)
                gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox3d)  #CameraInstance3DBoxes , LiDARInstance3DBoxes, DepthInstance3DBoxes
                gt_instances_3d.labels_3d = labels_3d
                ## break after reading a single data instancce in the if conidition              
                return gt_instances_3d

                
   




config="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead__14_August/convit3D_pointnet_transformer_ssdhead.py"
checkpoint="/workspace/mmdetection3d/work_dirs/convit3D_PointNet_transformer_ssdhead__14_August/epoch_170.pth"
file_name ='/workspace/data/kitti_detection/kitti/testing/velodyne_reduced/000003.bin' 
annotatioFile ="/workspace/data/kitti_detection/kitti/kitti_infos_train.pkl"

model = init_model(config, checkpoint, device='cuda:0')

    # init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# test a single point cloud sample
result, data = inference_detector(model, file_name)
points = data['inputs']['points']




gt_instances_3d = get_3dbbox_from_pklfile(annotatioFile)
result.gt_instances_3d = gt_instances_3d


data_input = dict(points=points)


##show the results
visualizer.add_datasample(
    'result',
    data_input,
    data_sample=result,
    draw_gt=True,
    draw_pred = True,
    show=True,
    wait_time=-1,
    out_file=None,
    pred_score_thr=0.5,
    vis_task='lidar_det')





