import numpy as np
import pytest
import torch
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmengine.structures import InstanceData
from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import KittiMetric
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, Det3DDataSample, points_cam2img)

import jsonpickle
import json

def _init_evaluate_input(result_dict,anno_path):
    
    classes = ['Pedestrian', 'Cyclist', 'Car']

    # load annotations
    ann_dict = load(anno_path)
    kittimetric = KittiMetric(anno_path, metric=['mAP'])
    data_infos = kittimetric.convert_annos_to_kitti_annos(ann_dict)
    
    metric_dict = {}
    metrics= ['bbox']
    gt_annos=[]
    

    # gt_annos_temp =[]
    # for result in result_dict['pred_instances_3d']:    
    #     if(len(result['sample_idx'])>0 and int(result['sample_idx'][0])< len(data_infos)):
    #         sample = result['sample_idx'][0]
    #         print(sample)
    #         kitti_anno = data_infos[sample]['kitti_annos']
    #         gt_annos_temp.append(kitti_anno)
            
    
    gt_annos = [
        data_infos[result['sample_idx'][0]]['kitti_annos']
        for result in result_dict['pred_instances_3d']
    ]    
    # print('result_dict:',len(result_dict['pred_instances_3d']))
    # print('data_infos:',len(data_infos))
    # print('gt_annos length',len(gt_annos))
 
    for metric in metrics:
        ap_dict = kittimetric.kitti_evaluate(
            result_dict,
            gt_annos,
            metric=metric,
            logger=None,
            classes=classes)
        for result in ap_dict:
            metric_dict[result] = ap_dict[result]

    return metric_dict



def test_kitti_metric_mAP():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    
    # '/workspace/data/kitti_detection/kitti/kitti_infos_val.pkl'
    data_root ="/workspace/data/kitti_detection/kitti"
    info_path = data_root + '/kitti_infos_val.pkl'
    
    # pred_result_file = '/workspace/data/pred_instances_3d.pkl'
    pred_result_file = '/workspace/data/kitti_detection/model_output_results/prediction_3d_mmdet3_format.pkl'
    result_dict =load(pred_result_file)
 
    predictions = _init_evaluate_input(result_dict,info_path)
    print("finished")
    





if __name__ == "__main__":
    test_kitti_metric_mAP()
    # dumpingObjects()

    