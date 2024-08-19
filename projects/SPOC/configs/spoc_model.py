_base_ = ['./nusceneocc_dataset.py']


custom_imports = dict(imports=['projects.SPOC.models'],allow_failed_imports=False)


point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]


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




# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict()


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


# optimizer = dict(
#     type='AdamW',
#     lr=5e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#             'sampling_offset': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01
# )
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     by_epoch=True,
#     step=[22, 24],
#     gamma=0.2
# )


total_epochs = 24

auto_scale_lr = dict(enable=False, base_batch_size=32)
# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        # dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = False