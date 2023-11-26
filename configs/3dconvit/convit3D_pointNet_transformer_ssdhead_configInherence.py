_base_ = [
    '../_base_/models/convit3D_kitti.py', '../_base_/datasets/kitti-3d-3class.py'
]


point_cloud_range = [0, -40, -5, 70, 40, 3]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

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
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='PointSample', num_points=32768), # 16384*2
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=32768),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]


train_dataloader = dict(
    batch_size=2, dataset=dict(dataset=dict(pipeline=train_pipeline, )))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))






'''
Log settings
'''
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook',vis_task='lidar_det',draw=False)
    )

log_config = dict(
    interval=50,
    by_epoch=True,
    log_metric_by_epoch=True,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(interval=1)


# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# optimizer
lr = 0.001 # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=1e-2),
    clip_grad=dict(max_norm=35, norm_type=2),
)


# learning rate
param_scheduler = [
    # Use a linear warm-up at [0, 30) iterations
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=30),
    # Use a cosine learning rate at [30, 200) iterations
    dict(type='CosineAnnealingLR',
         T_max=150,
         by_epoch=False,
         begin=30,
         end=200)
]


# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=80,
#         by_epoch=True,
#         milestones=[45, 60],
#         gamma=0.1)
# ]



env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
work_dir = './work_dirs/convit3D_PointNet_transformer_ssdhead_mini_waymo'
load_from = None
resume_from = './work_dirs/convit3D_PointNet_transformer_ssdhead_mini_waymo'
workflow = [('train', 1),('val', 1)]  
# workflow = [('val', 1)]  



