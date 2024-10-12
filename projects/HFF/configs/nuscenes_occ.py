

_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(imports=['projects.HFF.model','projects.BEVFusion.bevfusion'],allow_failed_imports=False)
# custom_imports = dict(imports=['projects.HFF.model','projects.BEVFusion.bevfusion'],allow_failed_imports=False)
# 

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

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



metainfo = dict(classes=class_names)
dataset_type = 'NuSceneOcc'
data_root = '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/'
occ_gt_root ='/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/occ3d/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=False)


backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)
_num_frames_ = 8

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(
        type='Pack3DDetInputs',  # New formatting component replacing DefaultFormatBundle3D and Collect3D
        keys=[ 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels', 'img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar',
                    'cam2img', 'ori_cam2img', 'lidar2cam', 'cam2lidar','ori_lidar2img', 'img_aug_matrix', 
                    'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
                    'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
                    'lidar_aug_matrix', 'num_pts_feats')
            )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='Pack3DDetInputs',  # New formatting component replacing DefaultFormatBundle3D and Collect3D
        keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar',
                   'cam2img', 'ori_cam2img', 'lidar2cam', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                    'lidar_path', 'img_path', 'num_pts_feats')
    )
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            occ_gt_root=occ_gt_root,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        occ_gt_root=occ_gt_root,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]

