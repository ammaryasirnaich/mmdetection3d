_base_ = ['./pointconvit3d_kitti_config.py']

# Save logs/checkpoints under outputs/ for this experiment
work_dir = './outputs/pointconvit3d_kitti_improved'

# Import the custom head module without editing any existing files.
custom_imports = dict(
    imports=['mmdet3d.models.dense_heads.ssd_3d_head_objness'],
    allow_failed_imports=False,
)

# --- Optimizer: add weight decay for better generalization ---
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
)

# --- LR schedule: warmup + cosine decay ---
# Keeps the same base lr (0.002) from the base config.
epoch_num = 100
lr = 0.002
param_scheduler = [
    # 1-epoch warmup to reduce early gradient spikes
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    # Smooth decay for the remaining epochs
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num - 1,
        eta_min=2e-5,
        by_epoch=True,
        begin=1,
        end=epoch_num,
    ),
]

# --- Model changes ---
model = dict(
    bbox_head=dict(
        # swap in the new head; all other settings are inherited
        type='SSD3DHeadObjness',
        # optional: slightly upweight score/objectness learning
        objectness_loss=dict(loss_weight=2.0),
    ),
    # less-aggressive NMS than 0.1; you should still sweep this for best Car AP
    test_cfg=dict(nms_cfg=dict(iou_thr=0.25)),
)

# --- Training loop ---
# More frequent validation to see Car AP improvements sooner.
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=epoch_num,
    val_interval=10,
)

