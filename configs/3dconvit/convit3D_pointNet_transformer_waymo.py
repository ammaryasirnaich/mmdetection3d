_base_ = [
    '../_base_/models/convit3D_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py'
]

dataset_type = 'WaymoDataset'
# data_root = '/import/digitreasure/openmm_processed_dataset/waymo/waymo_mini/'
# data_root = '/import/digitreasure/openmm_processed_dataset/waymo/kitti_format/'

point_cloud_range = [-76.8, -51.2, -2, 76.8, 51.2, 4]
pointcloudchannel=5

backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectSample', db_sampler=db_sampler),
      dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=32768),  #16384
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
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
            dict(type='PointSample', num_points=32768),  #16384 ,32768
        ]),
      dict(type='Pack3DDetInputs', keys=['points'],
         meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp'])
]






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

# log_config = dict(
#     interval=1,
#     by_epoch=True,
#     log_metric_by_epoch=True,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook'),
#            dict(type='EpochLossValuesLogging')])

custom_hooks = [dict(type='EpochLossValuesLogging')]

checkpoint_config = dict(interval=1)


# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001 
epoch_num = 90

# optimizer
lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning rate
param_scheduler = [
     dict(
        type='MultiStepLR',
        begin=0,
        end=epoch_num,
        by_epoch=True,
        milestones=[245,60],
        gamma=0.1)
]




# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=epoch_num * 0.4,
#         eta_min=lr * 10,
#         begin=0,
#         end=epoch_num * 0.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=epoch_num * 0.6,
#         eta_min=lr * 1e-4,
#         begin=epoch_num * 0.4,
#         end=epoch_num * 1,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=epoch_num * 0.4,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=epoch_num * 0.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=epoch_num * 0.6,
#         eta_min=1,
#         begin=epoch_num * 0.4,
#         end=epoch_num * 1,
#         convert_to_iter_based=True)
# ]

# training schedule for 1x
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=40)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


train_dataloader = dict(
    batch_size=4 ,dataset=dict(dataset=dict(pipeline=train_pipeline, )))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
# lr = 0.0018 # max learning rate
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
#     clip_grad=dict(max_norm=35., norm_type=2))
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=24,
#         by_epoch=True,
#         milestones=[16, 22],
#         gamma=0.1)
# ]
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).


auto_scale_lr = dict(enable=False, base_batch_size=4)



log_level = 'INFO'
work_dir = './work_dirs/convit3D_waymmo_batch_8'
# work_dir = './work_dirs/logtesting'
resume=True
load_from = None
resume_from = './work_dirs/convit3D_waymmo_batch_8'
# resume_from = './work_dirs/logtesting'
workflow = [('train', 1),('val', 1)]  
# workflow = [('val', 1)]  
