import torch
import numpy as np

from mmengine import load

from mmengine.structures import InstanceData
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode, CameraInstance3DBoxes)
                              

def get_3dInstance_from_pklfile(lidarfileinstance:str):
    # test_pkl_file = '/workspace/data/kitti_detection/kitti/kitti_infos_test.pkl'
    # test_pkl_file = '/workspace/data/kitti_detection/kitti/kitti_infos_train.pkl'
    val_pkl_file = '/workspace/data/kitti_detection/kitti/kitti_infos_val.pkl'
    info_file = load(val_pkl_file)   # ['metainfo', 'data_list'
        
    bboxes_3d = []
    labels_3d = []
    gt_instances_3d = InstanceData()
    lidar2cam = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2cam'])


    for info_dic in info_file['data_list']:
            lidar_info_str= info_dic['lidar_points']['lidar_path']
            if(lidar_info_str == lidarfileinstance):
                if(len(info_dic['instances'])==0):
                     print("No instance found")
                     return

                for instance_dic in info_dic['instances']:
                    bboxes_3d.append(instance_dic['bbox_3d'])
                    labels_3d.append(instance_dic['bbox_label_3d'])

                bbox3d_cam = torch.tensor(bboxes_3d)
                # bbox3d_cam = CameraInstance3DBoxes(bbox3d_cam)
                
                # bbox3d_cam -> bbox3d_lidar
                gt_instances_3d.bboxes_3d = CameraInstance3DBoxes(bbox3d_cam).convert_to(Box3DMode.LIDAR)
                gt_instances_3d.labels_3d = torch.tensor(labels_3d) 
                return  gt_instances_3d

          
   

