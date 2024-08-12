auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
checkpoint_config = dict(interval=1)
class_names = [
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
]
custom_hooks = [
    dict(type='EpochLossValuesLogging'),
]
data_prefix = dict(img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
data_root = '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/'
dataset_type = 'NuScenesDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=False, type='Det3DVisualizationHook', vis_task='lidar_det'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch_num = 80
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        sweeps_num=10,
        test_mode=True,
        type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
fp16 = dict(loss_scale='dynamic')
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'pytorch'
load_from = None
log_config = dict(
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50,
    log_metric_by_epoch=True)
log_level = 'INFO'
lr = 0.001
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    backbone=dict(
        aggregation_channels=(
            64,
            128,
            252,
        ),
        fps_mods=(
            'D-FPS',
            'FS',
            (
                'F-FPS',
                'D-FPS',
            ),
        ),
        fps_sample_range_lists=(
            -1,
            -1,
            (
                512,
                -1,
            ),
        ),
        in_channels=5,
        norm_cfg=dict(eps=0.001, momentum=0.1, type='BN2d'),
        num_points=(
            4096,
            1024,
            (
                256,
                256,
            ),
        ),
        num_samples=(
            (
                32,
                32,
                64,
            ),
            (
                32,
                32,
                64,
            ),
            (
                32,
                32,
                32,
            ),
        ),
        out_indices=(
            0,
            1,
            2,
        ),
        radii=(
            (
                0.2,
                0.4,
                0.8,
            ),
            (
                0.4,
                0.8,
                1.6,
            ),
            (
                1.6,
                3.2,
                4.8,
            ),
        ),
        sa_cfg=dict(
            normalize_xyz=False,
            pool_mod='max',
            type='PointSAModuleMSG',
            use_xyz=True),
        sa_channels=(
            (
                (
                    16,
                    16,
                    32,
                ),
                (
                    16,
                    16,
                    32,
                ),
                (
                    32,
                    32,
                    64,
                ),
            ),
            (
                (
                    64,
                    64,
                    128,
                ),
                (
                    64,
                    64,
                    128,
                ),
                (
                    64,
                    96,
                    128,
                ),
            ),
            (
                (
                    128,
                    128,
                    252,
                ),
                (
                    128,
                    192,
                    252,
                ),
                (
                    128,
                    256,
                    252,
                ),
            ),
        ),
        type='PointNet2SAMSG'),
    bbox_head=dict(
        bbox_coder=dict(
            num_dir_bins=12, type='AnchorFreeBBoxCoder', with_rot=True),
        center_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        corner_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        dir_class_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.CrossEntropyLoss'),
        dir_res_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        num_classes=3,
        objectness_loss=dict(
            loss_weight=1.0,
            reduction='sum',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        pred_layer_cfg=dict(
            bias=True,
            cls_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            in_channels=1536,
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN1d'),
            reg_conv_channels=(128, ),
            shared_conv_channels=(
                512,
                128,
            )),
        size_res_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        type='SSD3DHead',
        vote_aggregation_cfg=dict(
            bias=True,
            mlp_channels=(
                (
                    256,
                    256,
                    256,
                    512,
                ),
                (
                    256,
                    256,
                    512,
                    1024,
                ),
            ),
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN2d'),
            normalize_xyz=False,
            num_point=256,
            radii=(
                4.8,
                6.4,
            ),
            sample_nums=(
                16,
                32,
            ),
            type='PointSAModuleMSG',
            use_xyz=True),
        vote_loss=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.SmoothL1Loss'),
        vote_module_cfg=dict(
            conv_cfg=dict(type='Conv1d'),
            conv_channels=(128, ),
            gt_per_seed=1,
            in_channels=256,
            norm_cfg=dict(eps=0.001, momentum=0.1, type='BN1d'),
            num_points=256,
            vote_xyz_range=(
                3.0,
                3.0,
                2.0,
            ),
            with_res_feat=False)),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    middle_encoder=None,
    neck=dict(
        attn_drop_rate=0,
        depth=12,
        drop_path_rate=0,
        drop_rate=0,
        embed_dim=252,
        fp_output_channel=256,
        global_pool=None,
        hybrid_backbone=None,
        init_cfg=None,
        local_up_to_layer=10,
        locality_strength=1,
        mlp_ratio=4,
        num_classes=3,
        num_heads=9,
        pretrained=None,
        qk_scale=None,
        qkv_bias=False,
        rpn_feature_set=False,
        type='VisionTransformer',
        use_patch_embed=False,
        use_pos_embed=False),
    test_cfg=dict(
        max_output_num=100,
        nms_cfg=dict(iou_thr=0.1, type='nms'),
        per_class_proposal=True,
        sample_mode='spec',
        score_thr=0.0),
    train_cfg=dict(
        expand_dims_length=0.05, pos_distance_thr=10.0, sample_mode='spec'),
    type='ConVit3D',
    voxel_encoder=None)
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            20,
            23,
        ],
        type='MultiStepLR'),
]
point_cloud_range = [
    -50,
    -50,
    -5,
    50,
    50,
    3,
]
pointcloudchannel = 5
resume = True
resume_from = './work_dirs/convit3dNusence_pretrained'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root=
        '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                test_mode=True,
                type='LoadPointsFromMultiSweeps'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -50,
                            -50,
                            -5,
                            50,
                            50,
                            3,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root=
    '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(num_points=32768, type='PointSample'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -50,
                    -50,
                    -5,
                    50,
                    50,
                    3,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=80, type='EpochBasedTrainLoop', val_interval=40)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root=
        '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                type='LoadPointsFromMultiSweeps'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                rot_range=[
                    -0.3925,
                    0.3925,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -50,
                    -50,
                    -5,
                    50,
                    50,
                    3,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -50,
                    -50,
                    -5,
                    50,
                    50,
                    3,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ],
                type='ObjectNameFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(backend_args=None, sweeps_num=10, type='LoadPointsFromMultiSweeps'),
    dict(num_points=32768, type='PointSample'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
        ],
        type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root=
        '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                sweeps_num=10,
                test_mode=True,
                type='LoadPointsFromMultiSweeps'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -50,
                            -50,
                            -5,
                            50,
                            50,
                            3,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root=
    '/import/digitreasure/openmm_processed_dataset/nusense_dataset/nuscenses/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.25,
    0.25,
    8,
]
work_dir = './work_dirs/convit3dNusence_pretrained'
workflow = [
    (
        'train',
        1,
    ),
    (
        'val',
        1,
    ),
]
