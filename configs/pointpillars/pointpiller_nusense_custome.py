_base_ = './pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'

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

epoch_num = 41

train_dataloader = dict(batch_size=4)

train_cfg = dict(_delete_=True,type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=25)

log_level = 'INFO'
work_dir = './work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d'
load_from = None
resume = True
resume_from = './work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d'
# workflow = [('train', 1),('val', 1)]  
workflow = [('train', 1)] 
  