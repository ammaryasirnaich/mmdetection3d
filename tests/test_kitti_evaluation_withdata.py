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
    
    gt_annos_temp =[]
    


    for result in result_dict:
        # sample = result['sample_idx'][0]
                
        if(len(result['sample_idx'])>0 and int(result['sample_idx'][0])< len(data_infos)):
            sample = result['sample_idx'][0]
            # print(sample)
            
            kitti_anno = data_infos[sample]['kitti_annos']
            gt_annos_temp.append(kitti_anno)
            
    
        gt_annos = [
        data_infos[result['sample_idx'][0]]['kitti_annos']
        for result in result_dict
    ]
    
 
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



def dumpingObjects():
    
    ann_file = '/workspace/data/kitti_detection/model_output_results/predictionresults'
    # ann_file = osp.dirname(rootPath)+"/test.json"
    
    results = {"alpha": 1, "beta": 2}   

    
    path = osp.dirname(ann_file)+"/test.json"
    with open(path, 'w') as fout:
        # json.dump(results, fout)
        metainfo = dict(sample_idx=0)
        predictions = Det3DDataSample()
        pred_instances_3d = InstanceData()
        pred_instances = InstanceData()
        pred_instances.bboxes = torch.tensor([[712.4, 143, 810.7, 307.92]])
        pred_instances.scores = torch.Tensor([0.9])
        pred_instances.labels = torch.Tensor([0])
        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(
                [[8.7314, -1.8559, -1.5997, 0.4800, 1.2000, 1.8900, 0.0100]]))

        pred_instances_3d.scores_3d = torch.Tensor([0.9])
        pred_instances_3d.labels_3d = torch.Tensor([0])

        predictions.pred_instances_3d = pred_instances_3d
        predictions.pred_instances = pred_instances
        predictions.set_metainfo(metainfo)
        predictions = predictions.to_dict()
        
        encoded_data = jsonpickle.encode(predictions, unpicklable=False)
        json.dump(encoded_data, fout)






def test_kitti_metric_mAP():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    
    # '/workspace/data/kitti_detection/kitti/kitti_infos_val.pkl'
    data_root ="/workspace/data/kitti_detection/kitti"
    info_path = data_root + '/kitti_infos_val.pkl'
    
    # pred_result_file = '/workspace/data/pred_instances_3d.pkl'
    pred_result_file = '/workspace/data/kitti_detection/model_output_results/prediction_3d_mmdet3_format.pk'
    result_dict =load(pred_result_file)
 
    predictions = _init_evaluate_input(result_dict,info_path)
    print("finished")
    





if __name__ == "__main__":
    test_kitti_metric_mAP()
    # dumpingObjects()

    