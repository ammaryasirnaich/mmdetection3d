from unittest import runner


_base_ = [
    '/workspace/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py',
    # ../_base_/datasets/kitti-3d-car.py ,../_base_/datasets/kitti-3d-2d-car.py'
]


checkpoint_config = dict(interval=1,max_keep_ckpts=2,save_last=True)


# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
       
    ])

# trace_config = dict(type='tb_trace', dir_name='/workspace')
# profiler_config =  dict(hooks=[dict(type='ProfilerHook',by_epoch=False,profile_iters=5, record_shapes=True)])
# profiler_config = dict(type='ProfilerHook',profile_iters=1, record_shapes=True, \
#     on_trace_ready=trace_config)


voxel_size = [0.2, 0.2, 0.4]
point_cloud_range=[0, -40, -3, 70.4, 40, 1]

#  type='DynamicVFE',  # HardVFE , IEVFE' , HardSimpleVFE

model = dict(
     # Type of the Detector, refer to mmdet3d.models.detectors 
  voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
  
    voxel_encoder=dict(
        type='IEVFE',      # HardVFE , IEVFE
        in_channels=4,
        feat_channels=[32, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=False,
        point_cloud_range=point_cloud_range),
   
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=138, #128
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),)



data_root = '/workspace/data/kitti_detection/kitti/'
work_dir = '/workspace'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')       
    ])


# fp16 =dict(loss_scale=512.)
fp16 = dict(loss_scale='dynamic')


# custom_hooks = [ dict(type='TensorboardImageLoggerHook') ]

# Set seed thus the results are more reproducible
# seed = 0
# gpu_ids = range(1)
# samples_per_gpu=1 # batch size per GPU
runner = dict(type='EpochBasedRunner', max_epochs=40)