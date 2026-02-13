# Plan-based improvements: new config only (no edits to original configs).
# Inherits SaVoteHead setup from experiments, then applies optimizer, schedule,
# vote loss rebalance, class-balanced semantic, more proposals, and NMS tuning.
_base_ = ['pointconvit3d_kitti_experiments.py']

# Unique work_dir under outputs/
work_dir = './outputs/pointconvit_plan_improvements'
# resume: use --resume when running tools/train.py to auto-resume from latest checkpoint in work_dir
resume = False
load_from = None

# -----------------------------------------------------------------------------
# Optimizer (plan 3.9): weight decay, optional betas, gradient clipping
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
# LR schedule: warmup + cosine (plan 3.6, 3.8)
# -----------------------------------------------------------------------------
epoch_num = 100
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
# Model overrides (plan quick wins): vote loss, semantic balance, proposals, NMS
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
            class_weight=[3.0, 4.0, 1.0]),
    ),
    test_cfg=dict(
        sample_mode='spec',
        nms_thr=0.35,
        score_thr=0.01,
        per_class_proposal=True),
)
