dataset_type = 'KittiDataset'
data_root = '/workspace/data/kitti_detection/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

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
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=16384),
    dict(type='PointShuffle'),
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
            dict(type='PointSample', num_points=16384)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
        
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

'''
Model parameter settings
'''

# voxel_size = [0.2, 0.2, 0.4]   # no of voxel generated 38799
voxel_size = [0.05, 0.05, 0.2]  # no of voxel generated 91600
# x=1408 , y=1600, z= 40

# voxel_size = [0.05, 0.05, 0.1]
# point_cloud_range=[70.4, 80, 4]
               #      x  ,  y, z
               
# voxel_size = [0.05, 0.05, 0.2]

model = dict(
    type='ConVit3D', # Type of the Detector, refer to mmdet3d.models.detectors 
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=9,  #35
            point_cloud_range= point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE',),      # HardVFE , IEVFE 
    middle_encoder = None,
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, 256, 64),
        radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        aggregation_channels=(None, None, None, None),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),

    neck =  dict(
                type='ConViT3DNeck',
                num_classes=3, 
                in_chans=1024, #19
                embed_dim=1024, #19
                depth = 6, #  Depths Transformer stage. Default 12
                num_heads=4 ,  # 12
                mlp_ratio=4,
                qkv_bias=False ,
                qk_scale=None ,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0, 
                hybrid_backbone=None ,
                global_pool=None,
                local_up_to_layer=4 ,  #Consider how many layers to work for local feature aggregation
                locality_strength=1,
                use_pos_embed=False,
                init_cfg=None,
                pretrained=None,
                fp_output_channel = 512, 
                ),
        
     bbox_head=dict(
        type='Convit3DHead',
        num_classes=3,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True),
        pred_layer_cfg=dict(
            in_channels=1024,    # feature dimention from ConViT3DDecoder
            shared_conv_channels=(512, 128),   # (512, 128)
            cls_conv_channels=(128, ),         # (128, ), 
            reg_conv_channels=(128, ),         # (128, ), 
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
          
            ),

    # model training and testing settings
      train_cfg=dict(
        by_epoch=True,
          max_epochs=40, 
          val_interval=1,
          sample_mode='spec', 
          pos_distance_thr=0.3, 
          neg_distance_thr=0.6,
          expand_dims_length=0.05
          ) ,
    
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.25,
        score_thr=0.0,
        min_bbox_size=0,
        max_output_num=100,
        nms_pre=100,
        max_num=50,
        per_class_proposal=True,)
        
    )

# custom_hooks = [ dict(type='TensorboardImageLoggerHook') ]
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/workspace/working_dir'
load_from = None
resume_from = None
workflow = [('train', 1)]   # , ('val', 1)

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'


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
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook', draw=True))

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(interval=1,max_keep_ckpts=3,save_last=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False




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
        end=80,
        by_epoch=True,
        milestones=[45, 60],
        gamma=0.1)
]


# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)


# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40  
# seed = 0
# gpu_ids = range(1)
# samples_per_gpu=1 # batch size per GPU
# runner = dict(type='EpochBasedRunner', max_epochs=40)
# runner = dict(type='IterBasedRunner', max_iters=100)