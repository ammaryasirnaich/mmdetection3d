# PointConvIt3D on NuScenes: base config.
# Uses SaVoteHead with 10 classes, 5-channel points, NuScenesDataset.
_base_ = [
    '../_base_/models/pointconvit3d_kitti.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/cyclic-40e.py',
]

# -----------------------------------------------------------------------------
# Dataset: NuScenes at specified path (ensure nuscenes_infos_train/val.pkl exist)
# -----------------------------------------------------------------------------
data_root = '/var/lib/containers/dataset/nusence_dataset/nuscenes'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# NuScenes uses 5-dim points (x, y, z, intensity, ring_index)
pointcloudchannel = 5
backend_args = None

# -----------------------------------------------------------------------------
# NuScenes 10-class mean sizes (l, w, h) in meters - placeholders; replace with
# values computed from nuscenes_infos_train.pkl for better performance.
# Order: car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle,
#        pedestrian, traffic_cone, barrier
# -----------------------------------------------------------------------------
nuscenes_mean_sizes = [
    [4.5, 1.8, 1.5],   # car
    [7.0, 2.5, 3.2],   # truck
    [10.0, 2.5, 3.0],  # trailer
    [10.5, 2.8, 3.5],  # bus
    [6.0, 2.5, 2.8],   # construction_vehicle
    [1.7, 0.6, 1.2],   # bicycle
    [2.2, 0.8, 1.4],   # motorcycle
    [0.6, 0.6, 1.7],   # pedestrian
    [0.5, 0.5, 0.8],   # traffic_cone
    [4.0, 0.5, 1.2],   # barrier
]

# -----------------------------------------------------------------------------
# Model: 5-channel backbone, 10-class neck, SaVoteHead with PartialBinBasedBBoxCoder
# Neck: deeper ViT for 10-class nuScenes (depth=18, more local layers, drop_path)
# -----------------------------------------------------------------------------
model = dict(
    backbone=dict(in_channels=pointcloudchannel),
    neck=dict(
        num_classes=10,
        depth=18,
        local_up_to_layer=14,
        drop_path_rate=0.1,
    ),
    bbox_head=dict(
        _delete_=True,
        type='SaVoteHead',
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=nuscenes_mean_sizes),
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            num_points=-1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            norm_feats=True,
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0),
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=5.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=512,
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
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            reduction='sum',
            loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss',
            reduction='sum',
            loss_weight=1.0,
            class_weight=[1.0] * 10),
    ),
    train_cfg=dict(
        pos_distance_thr=1.0, neg_distance_thr=2.0, sample_mode='spec'),
    test_cfg=dict(
        sample_mode='spec', nms_thr=0.35, score_thr=0.01,
        per_class_proposal=True),
)

# -----------------------------------------------------------------------------
# Dataloaders / evaluator: use nus-3d structure; only data_root is overridden
# via the variable above (nus-3d.py references data_root).
# with_velocity=False so GT boxes are 7-D (no velocity), matching mmcv
# points_in_boxes_* which expect 7-D; avoids AssertionError in SaVoteHead.
# -----------------------------------------------------------------------------
train_dataloader = dict(
    dataset=dict(data_root=data_root, with_velocity=False),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root, with_velocity=False),
)
test_dataloader = dict(
    dataset=dict(data_root=data_root, with_velocity=False),
)
val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + '/nuscenes_infos_val.pkl',
)
test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + '/nuscenes_infos_val.pkl',
)

# -----------------------------------------------------------------------------
# Runtime: work_dir, hooks, env (same style as KITTI config)
# -----------------------------------------------------------------------------
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='Det3DVisualizationHook', vis_task='lidar_det', draw=False),
)
log_config = dict(
    interval=50,
    by_epoch=True,
    log_metric_by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
custom_hooks = [dict(type='EpochLossValuesLogging')]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
work_dir = './outputs/pointconvit3d_nuscenes'
load_from = None
resume = True
workflow = [('train', 1)]

# -----------------------------------------------------------------------------
# Optimizer and schedule for 80 epochs (replaces cyclic-40e; warmup + cosine)
# -----------------------------------------------------------------------------
epoch_num = 80
lr = 0.002
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.01,
        betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=25, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num - 1,
        eta_min=2e-5,
        by_epoch=True,
        begin=1,
        end=epoch_num,
    ),
]
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=epoch_num,
    val_interval=10,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
