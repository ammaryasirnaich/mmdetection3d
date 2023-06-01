# dataset settings
dataset_type = 'KittiDataset'
data_root = '/workspace/data/kitti_detection/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/kitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15),
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
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
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

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


'''
Model parameter settings
'''

voxel_size = [0.2, 0.2, 0.4]   # no of voxel generated 38799
# voxel_size = [0.05, 0.05, 0.2]  # no of voxel generated 91600
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
    middle_encoder =  dict(
                type='PointNet2SASSG_SL',
                in_channels=4,
                num_points=(32,16), # irrelavent
                radius=(0.8, 1.2),
                num_samples=(16,8),
                sa_channels=((8, 16), (16, 16)),    #((8, 16), (16, 16)),
                fp_channels=((16, 16), (16, 16)),   #((16, 16), (16, 16))
                norm_cfg=dict(type='BN2d')),
    backbone =  dict(
                type='ConViT3DDecoder',
                num_classes=3, 
                in_chans=16, #19
                embed_dim=16, #19
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
                fp_output_channel = 16, 
                ),
        
    rpn_head=dict(
        type='PointRPNHead',
        num_classes=3,
        enlarge_width=0.1,
        pred_layer_cfg=dict(
            in_channels=16,
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256)),
        cls_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        bbox_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=1.0),
        bbox_coder=dict(
            type='PointXYZWHLRBBoxCoder',
            code_size=8,
            # code_size: (center residual (3), size regression (3),
            #             torch.cos(yaw) (1), torch.sin(yaw) (1)
            use_mean_size=True,
            mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6,
                                                            1.73]])),
    roi_head=dict(
        type='PointRCNNRoIHead',
        bbox_roi_extractor=dict(
            type='Single3DRoIPointExtractor',
            roi_layer=dict(type='RoIPointPool3d', num_sampled_points=512)),
        bbox_head=dict(
            type='PointRCNNBboxHead',
            num_classes=1,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            pred_layer_cfg=dict(
                in_channels=512,
                cls_conv_channels=(256, 256),
                reg_conv_channels=(256, 256),
                bias=True),
            in_channels=5,
            # 5 = 3 (xyz) + scores + depth
            mlp_channels=[128, 128],
            num_points=(128, 32, -1),
            radius=(0.2, 0.4, 100),
            num_samples=(16, 16, 16),
            sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
            with_corner_loss=True),
        depth_normalizer=70.0),

    # model training and testing settings
     train_cfg=dict(
        pos_distance_thr=10.0,
        rpn=dict(
            rpn_proposal=dict(
                use_rotate_nms=True,
                score_thr=None,
                iou_thr=0.8,
                nms_pre=9000,
                nms_post=512)),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.7,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True,
                iou_thr=0.85,
                nms_pre=9000,
                nms_post=512,
                score_thr=None)),
        rcnn=dict(use_rotate_nms=True, nms_thr=0.1, score_thr=0.1))
    
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
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


checkpoint_config = dict(interval=1,max_keep_ckpts=5,save_last=True)
# log_config = dict(
#     interval=50,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook')])

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


param_scheduler = [
    # learning rate scheduler
    # During the first 35 epochs, learning rate increases from 0 to lr * 10
    # during the next 45 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=lr * 10,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=lr * 1e-4,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 35 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 45 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=35,
        eta_min=0.85 / 0.95,
        begin=0,
        end=35,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=45,
        eta_min=1,
        begin=35,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True)
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