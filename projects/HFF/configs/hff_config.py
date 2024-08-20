# hff_config.py

_base_ = './sparseocc_dataset.py'  # Dataset configuration is referenced here

model = dict(
    type='HFFModel',
    img_backbone=dict(
        type='ResNet',
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
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    pts_voxel_layer=dict(  # Voxelization layer for LiDAR data
        max_num_points=32,
        voxel_size=[0.2, 0.2, 8],
        max_voxels=(16000, 40000),
        point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0]
    ),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        in_channels=4,
        feat_channels=[64, 64, 128]
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,
        sparse_shape=[41, 1600, 1408],
        output_channels=256
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
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
        type='SparseBEVTransformer',  # Integrated from SparseOcc
        in_channels=512,  # Combined from both LiDAR and image features
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
        topk_training=200,  # Sparsification settings from SparseOcc
        topk_testing=100,
        norm_cfg=dict(type='BN', requires_grad=True),
        pc_range=[-50, -50, -5.0, 50, 50, 3.0]
    ),
    mask_head=dict(
        type='MaskTransformerHead',
        num_queries=100,
        transformer=dict(
            type='SparseOccTransformer',  # Integrated SparseOcc transformer
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
