voxel_size = [0.08, 0.08, 0.1]
point_cloud_range = [-76.8, -51.2, -2, 76.8, 51.2, 4]

model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(80000, 90000)),
    voxel_encoder=dict(
        type='HardVFE',      # HardVFE , IEVFE
        # in_channels=4,
        # feat_channels=[32, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=False,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='SparseEncoder',
        # in_channels=138,
        sparse_shape=[41, 1600, 1408],  #[61, 1280, 1920]
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=256, #384
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-76.8, -51.2, -0.0345, 76.8, 51.2, -0.0345],
                    [-76.8, -51.2, 0, 76.8, 51.2, 0],
                    [-76.8, -51.2, -0.1188, 76.8, 51.2, -0.1188]],
            sizes=[
                [4.73, 2.08, 1.77],  # car
                [0.91, 0.84, 1.74],  # pedestrian
                [1.81, 0.84, 1.77]  # cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.25,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=500))



# dataset settings
dataset_type = 'WaymoDataset'
data_root = '/workspace/data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4]))

# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=5)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=24, pipeline=eval_pipeline)

# custom_hooks = [ dict(type='TensorboardImageLoggerHook') ]
# Training settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/workspace/mmdetection3d/working_dir/Waymo_intensityAware_10pnt'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Mixed precision set to dynamic
# fp16 =dict(loss_scale=512.)
fp16 = dict(loss_scale='dynamic')

checkpoint_config = dict(interval=1, max_keep_ckpts=5, save_last=True)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

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
# The schedule is usually used by models trained on KITTI dataset
# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4

# the official AdamW optimizer implemented by PyTorch.
optimizer = dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# We use cyclic learning rate and momentum schedule following SECOND.Pytorch

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])

momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40  
# seed = 0
# gpu_ids = range(1)
# samples_per_gpu=1 # batch size per GPU
runner = dict(type='EpochBasedRunner', max_epochs=24)
# runner = dict(type='IterBasedRunner', max_iters=100)


