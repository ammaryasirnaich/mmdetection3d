# Plan-based improvements for PointConvIt3D NuScenes.
# Inherits base NuScenes config, applies optimizer/schedule/vote/semantic/NMS tweaks.
_base_ = ['../pointconvit3d_nuscenes_config.py']

work_dir = './outputs/pointconvit_nuscenes_plan_improvements'
resume = False
load_from = None

# -----------------------------------------------------------------------------
# Optimizer: weight decay, betas, gradient clipping (plan style)
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.002,
        weight_decay=0.01,
        betas=(0.95, 0.99)),
    clip_grad=dict(max_norm=25, norm_type=2),
)

# -----------------------------------------------------------------------------
# LR schedule: warmup + cosine
# -----------------------------------------------------------------------------
epoch_num = 170
lr = 0.002
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

# -----------------------------------------------------------------------------
# Training loop: more frequent validation
# -----------------------------------------------------------------------------
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=epoch_num,
    val_interval=10,
)

# -----------------------------------------------------------------------------
# Model overrides: vote loss, semantic balance, proposals, NMS (10-class)
# -----------------------------------------------------------------------------
model = dict(
    bbox_head=dict(
        vote_module_cfg=dict(
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=5.0)),
        vote_aggregation_cfg=dict(num_point=512),
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss',
            reduction='sum',
            loss_weight=1.0,
            # 10-class weights; tune per class or use uniform [1.0]*10
            class_weight=[1.0] * 10),
    ),
    test_cfg=dict(
        sample_mode='spec',
        nms_thr=0.35,
        score_thr=0.01,
        per_class_proposal=True),
)
