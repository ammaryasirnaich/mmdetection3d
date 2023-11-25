_base_ = [
    '../_base_/models/convit3D_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
]

# dataset_type = 'WaymoDataset'
# data_root = '/import/digitreasure/openmm_processed_dataset/waymo/waymo_mini/'

point_cloud_range = [-76.8, -51.2, -2, 76.8, 51.2, 4]

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
    dict(type='PointSample', num_points=16384),
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
            dict(type='PointSample', num_points=16384),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4, dataset=dict(dataset=dict(pipeline=train_pipeline, )))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))





# fp16 settings
# fp16 =dict(loss_scale=512.)
fp16 = dict(loss_scale='dynamic')

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
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(interval=1)


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# trace_config = dict(type='tb_trace', dir_name= work_dir)
# schedule_config= dict(type="schedule", wait=1,warmup=1,active=2)

# profiler_config = dict(type='ProfilerHook',by_epoch=False,profile_iters=2,
#                     record_shapes=True, profile_memory=True, with_flops =True, 
#                         schedule = dict( wait=1,warmup=1,active=2),
#                         on_trace_ready=dict(type='tb_trace', dir_name= work_dir))
#                         # with_stack =True,



'''
Schedules settings
'''
# optimizer

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
lr = 0.0018 # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)



log_level = 'INFO'
work_dir = './work_dirs/convit3D_PointNet_transformer_ssdhead_large_waymmo'
load_from = None
resume_from = './work_dirs/convit3D_PointNet_transformer_ssdhead_large_waymo'
workflow = [('train', 1),('val', 1)]  
# workflow = [('val', 1)]  
