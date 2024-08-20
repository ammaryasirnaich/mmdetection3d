# hff_model.py

_base_ = './sparseocc_dataset.py'

model = dict(
    type='HFFModel',  # This is a custom model type that you define
    backbone=dict(
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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    voxel_encoder=dict(
        type='HardSimpleVFE',
        in_channels=256,
        feat_channels=[128, 128, 256],
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0]
    ),
    decode_head=dict(
        type='SparseVoxelDecoder',
        in_channels=256,
        num_classes=len(class_names),
        top_k=200,  # Sparsification settings from SparseOcc
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    mask_head=dict(
        type='MaskTransformerHead',
        num_queries=100,
        transformer=dict(
            type='DetrTransformer',
            num_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        ),
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=1.0)
    ),
    train_cfg=dict(
        voxel_size=[0.2, 0.2, 8],
        point_cloud_range=[-50, -50, -5.0, 50, 50, 3.0],
        assigner=dict(type='HungarianAssigner3D', cls_cost=dict(type='FocalLoss', weight=2.0))
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
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Dataset settings are inherited from the base config
