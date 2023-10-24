
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
    type= 'ConVit3D',        #'ConVit3D', # Type of the Detector, refer to mmdet3d.models.detectors 
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
       
        ),
    voxel_encoder=None,      # HardVFE , IEVFE ,dict(type='HardSimpleVFE',),
    middle_encoder = None,
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=4,
        num_points=(4096, 1024, (256, 256)),   #(4096, 512, (256, 256)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 240),
        fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=((-1), (-1), (512, -1)),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
            
      neck =  dict(
                type='VisionTransformer',   
                num_classes=3, 
                # in_chans=256, #1024
                embed_dim=240, #1024
                depth = 12, #  Depths Transformer stage. Default 12
                num_heads=12 ,  # 12
                mlp_ratio=4,
                qkv_bias=False ,
                qk_scale=None ,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0, 
                hybrid_backbone=None ,
                global_pool=None,
                local_up_to_layer=10 ,  #Consider how many layers to work for local feature aggregation
                locality_strength=1,
                use_pos_embed=False,
                init_cfg=None,
                pretrained=None,
                use_patch_embed=False,
                fp_output_channel = 256,
                rpn_feature_set = False,  
                ), 


   bbox_head=dict(
        type='SSD3DHead',    #SSD3DHead , TransHead
        num_classes=3,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True),
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
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
        vote_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0)),

    # model training and testing settings
   train_cfg=dict(
        sample_mode='spec', pos_distance_thr=10.0, expand_dims_length=0.05),
    
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mode='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100)
        )


# custom_hooks = [ dict(type='TensorboardImageLoggerHook') ]
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/convit3D_PointNet_transformer_ssdhead__01_Sep'
load_from = None
# resume_from = './work_dirs/convit3D_PointNet_transformer_ssdhead__01_Sep'
workflow = [('train', 1)]  

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
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook',draw=True)
    )

log_config = dict(
    interval=50,
    by_epoch=True,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(interval=5)


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = True

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


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=4)


