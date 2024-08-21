# hff_config.py

_base_ = './nuscenes_occ.py'
custom_imports = dict(imports=['projects.HFF.model'],allow_failed_imports=False)



voxel_size=[0.2, 0.2, 8]
point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0]

model = dict(
    type='HFFModel',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),)),
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
    voxel_encoder=dict(
        type='SparseBEVTransformer',
        in_channels=768,  # Combined features from all sensors (camera, LiDAR)
        embed_dims=256,
        num_layers=3,
        num_frames=8,
        num_points=2048,
        num_groups=4,
        num_levels=4,
        num_classes=18,
        pc_range=[-50, -50, -5.0, 50, 50, 3.0],
        occ_size=[200, 200, 16],
        topk_training=200,
        topk_testing=100
    ),
    decode_head=dict(
        type='SparseVoxelDecoder',
        in_channels=256,
        num_classes=18,
        topk_training=200,
        topk_testing=100,
        norm_cfg=dict(type='BN', requires_grad=True),
        pc_range=[-50, -50, -5.0, 50, 50, 3.0]
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
        transformer=dict(
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
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=1.0)
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

# Training hyperparameters
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Dataset settings are inherited from the base config
