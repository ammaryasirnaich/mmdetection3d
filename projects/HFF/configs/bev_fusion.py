_base_ = [ './nusce_dataset.py']
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)


# custom_imports = dict(imports=['projects.HFF.model'],allow_failed_imports=False)

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

backend_args = None

occ_size = [200, 200, 16]
voxel_size=[0.2, 0.2, 8]
point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0]

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]


model = dict(
    type='HFFModel', 
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
        ),
        mean=[123.675, 116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        bgr_to_rgb=True
    ), 
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', requires_grad=True),
        use_conv_for_no_stride=True
    ),
    
    img_point_encoder=dict(
        type='SparseBEVTransformer',
        # in_channels=768,  # Combined features from all sensors (camera, LiDAR)
        embed_dims=_dim_,
        num_layers=_num_layers_,
        num_frames=_num_frames_,
        num_points=_num_points_,
        # num_groups=_num_groups_,
        # num_queries=_num_queries_,
        num_levels=4,
        num_classes=len(occ_class_names),
        pc_range=point_cloud_range,
        # code_size=occ_size,
        # topk_training=_topk_training_,
        # topk_testing=_topk_testing_
    ),
    fusion_module=dict(
        type='MultiResolutionFusion',
        coarse_channels=256,
        intermediate_channels=256,
        fine_channels=256,
        adaptative_resolution=True,
        complexity_threshold=0.8
    ),
    mask_head=dict(
        type='MaskTransformerHead',
        num_queries=100,
        sparseocc_tranformer=dict(
            type='SparseOccTransformer',
            embed_dims=256,
            num_layers=6,
            num_queries=100,
            num_frames=8,
            num_points=2048,
            num_groups=4,
            num_levels=4,
            num_classes=18,
            pc_range=[-50, -50, -5.0, 50, 50, 3.0],
            occ_size=[200, 200, 16] ),
        # loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_dice=dict(type='DiceLoss', loss_weight=1.0)
    ),
    
    )

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=50),
#     checkpoint=dict(type='CheckpointHook', interval=1))

# del _base_.custom_hooks


# KeyError: "Duplicate key is not allowed among bases. Duplicate keys: {'log_processor', 'default_hooks'}"