import torch
import numpy as np

from mmengine import load

from mmengine.structures import InstanceData
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode, CameraInstance3DBoxes)
                              

## for single object displya

info_file = load('demo/data/kitti/kitti_infos_test.pkl')   # ['metainfo', 'data_list'
# info_file = load('demo/data/kitti/kitti_infos_train.pkl')   # ['metainfo', 'data_list'


def get_3dInstance_from_pklfile(lidarfileinstance:str):
    test_pkl_file = 'demo/data/kitti/kitti_infos_test.pkl'
    info_file = load(test_pkl_file)   # ['metainfo', 'data_list'
     

    # Traverse the data to extract 3D bounding box information
    print(info_file['data_list'][0].keys()) 

    
    bboxes_3d = []
    labels_3d = []
    gt_instances_3d = InstanceData()
    lidar2cam = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2cam'])

    ### sample_idx', 'images', 'lidar_points', 'instances', 'cam_instances
    for info_dic in info_file['data_list']:
            # print(type(info_dic))
            lidar_info_str= info_dic['lidar_points']['lidar_path']
            # print(type(lidar_info_dic))
            # print(lidar_info_str) 
            if(lidar_info_str == lidarfileinstance):

                # print(type(info_dic['instances']))
                # print(len(info_dic['instances']))
                for instance_dic in info_dic['instances']:
                    bboxes_3d.append(instance_dic['bbox_3d'])
                    labels_3d.append(instance_dic['bbox_label_3d'])

                bbox3d = torch.tensor(bboxes_3d)
                labels_3d = torch.tensor(labels_3d)  
                gt_instances_3d.bboxes_3d = CameraInstance3DBoxes(bbox3d).convert_to(Box3DMode.LIDAR,
                                                 np.linalg.inv(lidar2cam))
                gt_instances_3d.labels_3d = labels_3d
                return  gt_instances_3d

          
   

