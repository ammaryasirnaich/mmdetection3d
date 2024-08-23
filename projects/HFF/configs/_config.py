# _base_ = ['./nuscenes_occ.py','../../../configs/_base_/default_runtime.py']
_base_ = './multimodel_nusce_cfg.py'
# _base_ = ['/import/digitreasure/ammar_workspace/mmdetection3d/projects/HFF/configs/nuscenes_occ.py']

custom_imports = dict(imports=['projects.HFF.model'],allow_failed_imports=False)

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

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
            occ_size=[200, 200, 16]
        ),
        # loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_dice=dict(type='DiceLoss', loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(type='HungarianAssigner3D', cls_cost=dict(type='FocalLoss', weight=2.0)),
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0]
    ),
    test_cfg=dict(
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0],
        nms_type='nms',
        iou_thr=0.5
    )
)

# Optimizer configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,  # No weight decay for normalization layers
        bias_decay_mult=0.0,  # No weight decay for bias parameters
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2
    )
)

# Learning rate schedule configuration
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=1000,  # Equivalent to warmup_iters
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,  # Number of total epochs
        by_epoch=True,
        milestones=[8, 11],  # When to decay the learning rate
        gamma=0.1  # The factor by which the learning rate is reduced
    )
]

# Training loop configuration
train_cfg = dict( type='EpochBasedTrainLoop',max_epochs=12,  val_interval=1)
val_cfg = dict()
test_cfg = dict()



log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Dataset settings are inherited from the base config
