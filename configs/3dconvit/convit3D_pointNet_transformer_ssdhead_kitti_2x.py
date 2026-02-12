_base_ = [
    '../_base_/models/convit3D_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py',
]

# Dataset & pipeline overrides to follow the legacy SSD head experiment,
# adapted to the new KITTI base dataset.
# 2x: RepeatDataset times=2

dataset_type = 'KittiDataset'
data_root = '/home/naich/dataset/kitti_data/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=16384),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=16384),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points']),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

# Model override: ConVit3D + SSD3DHead as in the legacy config

voxel_size = [0.05, 0.05, 0.2]

model = dict(
    type='ConVit3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    voxel_encoder=None,
    middle_encoder=None,
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, (256, 256)),
        radii=(
            (0.2, 0.4, 0.8),
            (0.4, 0.8, 1.6),
            (1.6, 3.2, 4.8),
        ),
        num_samples=(
            (32, 32, 64),
            (32, 32, 64),
            (32, 32, 32),
        ),
        sa_channels=(
            ((16, 16, 32), (16, 16, 32), (32, 32, 64)),
            ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
            ((128, 128, 256), (128, 192, 256), (128, 256, 256)),
        ),
        aggregation_channels=(64, 128, 240),
        fps_mods=('D-FPS', 'FS', ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=(-1, -1, (512, -1)),
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    neck=dict(
        type='VisionTransformer',
        num_classes=3,
        embed_dim=240,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        hybrid_backbone=None,
        global_pool=None,
        local_up_to_layer=10,
        locality_strength=1,
        use_pos_embed=False,
        init_cfg=None,
        pretrained=None,
        use_patch_embed=False,
        fp_output_channel=256,
        rpn_feature_set=False),
    bbox_head=dict(
        type='SSD3DHead',
        num_classes=3,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder',
            num_dir_bins=12,
            with_rot=True),
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=(
                (256, 256, 256, 512),
                (256, 256, 512, 1024),
            ),
            norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128,),
            reg_conv_channels=(128,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=1.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss',
            reduction='sum',
            loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=1.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=1.0),
        vote_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=1.0)),
    train_cfg=dict(
        sample_mode='spec',
        pos_distance_thr=10.0,
        expand_dims_length=0.05),
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mode='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100))

# Runtime, optimizer and schedule, following the legacy experiment but
# expressed in the modern MMEngine-style runtime config.

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',interval=1, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='Det3DVisualizationHook',
        vis_task='lidar_det',
        draw=False),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_config = dict(
    interval=50,
    by_epoch=True,
    log_metric_by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

custom_hooks = [dict(type='EpochLossValuesLogging')]

epoch_num = 100
lr = 0.002

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[45, 60],
        gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop',
    max_epochs=epoch_num,val_interval=40)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
work_dir = './outputs/convit3D_pointNet_transformer_ssdhead_kitti_2x'
load_from = None
resume = True
resume_from = None
workflow = [('train', 1)]

fp16 = dict(loss_scale='dynamic')
