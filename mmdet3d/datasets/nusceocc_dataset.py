from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader



from projects.SPOC.loaders.ray_metrics import main_rayiou, main_raypq
from projects.SPOC.models.utils import sparse2dense
from projects.SPOC.loaders.ego_pose_dataset import EgoPoseDataset

from projects.SPOC.configs.r50_nuimg_704x256_8f import occ_class_names as occ3d_class_names
from projects.SPOC.configs.r50_nuimg_704x256_8f_openocc import occ_class_names as openocc_class_names
import glob
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    print("NuSceneOcc is called")    
    def __init__(self, occ_gt_root, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name


    def collect_sweeps(self, index, into_past=150, into_future=0):
        
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index].get('sweeps', [])
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            if curr_index > 0:
                all_sweeps_prev.append(self.data_infos[curr_index - 1].get('cams', {}))
            curr_index -= 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future and curr_index < len(self.data_infos):
            curr_sweeps = self.data_infos[curr_index].get('sweeps', [])
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index].get('cams', {}))
            curr_index += 1
        
        return all_sweeps_prev, all_sweeps_next
    def get_data_info(self, index):
            
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):

        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]

            
    def format_results(self, occ_results, submission_prefix, **kwargs):

        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
