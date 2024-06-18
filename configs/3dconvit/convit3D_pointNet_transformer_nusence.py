_base_ = [
    #  '../_base_/models/pointpillars_hv_fpn_nus.py',
    '../_base_/models/convit3D_waymo.py',
    '../_base_/datasets/nus-3d.py',
     '../_base_/schedules/schedule-2x.py'
]

pointcloudchannel=5
voxel_size = [0.25,0.25,8]
point_cloud_range = [-50, -50, -5, 50, 50, 3]


train_pipeline = [
    dict(type='PointSample', num_points=32768),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
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
            dict(type='PointSample', num_points=32768),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
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

log_config = dict(
    interval=50,
    by_epoch=True,
    log_metric_by_epoch=True,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

custom_hooks = [dict(type='EpochLossValuesLogging')]

checkpoint_config = dict(interval=1)


# In practice PointPillars also uses a different schedule
# optimizer
epoch_num = 80

# optimizer
resume = True
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=40)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')



env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
work_dir = './work_dirs/convit3dNusence_14_june'
load_from = None
resume = True
resume_from = './work_dirs/convit3dNusence_14_june'
workflow = [('train', 1),('val', 1)]  
  



