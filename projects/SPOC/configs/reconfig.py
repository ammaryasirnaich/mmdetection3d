_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.SPOC.models'], allow_failed_imports=False)


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

dataset_type = 'NuScenesSegDataset'
dataset_type = 'NuSceneOcc'
dataset_root = '/workspace/data/nusense/mini_dataset/'
occ_gt_root = '/workspace/data/nusense/mini_dataset/occ3d/'

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')


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

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)


backend_args = None

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]
val_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_val_sweep.pkl',
        pipeline=val_pipeline,
        test_mode=True))


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_test_sweep.pkl',
        pipeline=val_pipeline,
        test_mode=True))



val_evaluator = dict(
    type='NuScenesMetric',
    data_root=dataset_root,
    ann_file=dataset_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Optimizer Configuration
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=0.01,
        paramwise_cfg=dict(
            custom_keys={
                'img_backbone': dict(lr_mult=0.1),
                'sampling_offset': dict(lr_mult=0.1),})),
    clip_grad=dict(max_norm=35,norm_type=2) )

param_scheduler = [
    # Warmup Configuration
    dict(
        type='LinearLR',  # Linear warmup scheduler
        start_factor=1.0 / 3,
        by_epoch=True,
        begin=0,
        end=500,  # Equivalent to warmup_iters=500
        convert_to_iter_based=True  # Converts to iterations if by_epoch=True
    ),
    # Step LR Scheduler
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[22, 24],  # Equivalent to step=[22, 24]
        gamma=0.2  # Learning rate decay factor
    )
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

model = dict(
    type='SparseOcc',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_mask_camera=False,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
)