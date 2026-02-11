_base_ = [
    '../pointconvit3d_kitti_config.py',
]

# -----------------------------------------------------------------------------
# One-file experiment workflow:
# - Keep base configs untouched.
# - Edit ONLY this file as you move Task A -> B -> C -> D -> E.
# - Always change work_dir when you change the experiment.
# -----------------------------------------------------------------------------

# Set a unique work_dir per task run (do not overwrite prior results).
work_dir = './outputs/pointconvit/taskA_votehead_multiclass'
resume = False
load_from = None

# Toggle tasks as you move down the plan (A -> B -> C -> D -> E).
# Keep only ONE enabled at a time for clean comparisons.
ENABLE_TASK_B_DBSAMPLER_TUNE = False
ENABLE_TASK_C_CLASS_BALANCE = False
ENABLE_TASK_D_PROPOSAL_NMS = False
ENABLE_TASK_E_HEAD_SWAP = False

# -----------------------------------------------------------------------------
# Task A: VoteHead-style multi-class supervision for PointNet2SAMSG (+ ConViT neck)
#
# Important: Stock VoteHead expects fp_xyz/fp_features/fp_indices (PointNet2SASSG).
# This experiment uses SaVoteHead which reads sa_xyz/sa_features/sa_indices.
# -----------------------------------------------------------------------------

# KITTI 3-class size priors (l, w, h). These are reasonable initial values.
# You can later replace them with statistics computed from your KITTI train set.
kitti_mean_sizes = [
    [0.80, 0.60, 1.73],  # Pedestrian
    [1.76, 0.60, 1.73],  # Cyclist
    [3.90, 1.60, 1.56],  # Car
]

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='SaVoteHead',
        num_classes=3,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=3,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=kitti_mean_sizes),
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            # IMPORTANT:
            # VoteModule will truncate BOTH seed_points and seed_feats when
            # `num_points != -1`, but VoteHead stores the *original* seed_points
            # for loss computation. That causes a seed/vote shape mismatch.
            # Keep `-1` to use all seeds from the backbone output.
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
                loss_dst_weight=10.0)),
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
        # Semantic/class supervision (multi-class) - start unweighted.
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
    ),
    # VoteHead expects different train/test cfg keys than SSD3DHead.
    train_cfg=dict(pos_distance_thr=1.0, neg_distance_thr=2.0, sample_mode='spec'),
    test_cfg=dict(sample_mode='spec', nms_thr=0.25, score_thr=0.05, per_class_proposal=True),
)

# -----------------------------------------------------------------------------
# Task B: DB sampler tuning (oversample Pedestrian/Cyclist)
# Base config already has ObjectSample enabled; this only changes sampling ratios.
# -----------------------------------------------------------------------------
if ENABLE_TASK_B_DBSAMPLER_TUNE:
    work_dir = './work_dirs/pointconvit/taskB_dbsampler_tune'
    # `db_sampler` is inherited from `configs/_base_/datasets/kitti-3d-3class.py`
    # and is a normal Python dict, so we can update it in-place.
    db_sampler.update(
        dict(sample_groups=dict(Car=12, Pedestrian=12, Cyclist=12)))

# -----------------------------------------------------------------------------
# Task C: Class-balanced semantic loss
# -----------------------------------------------------------------------------
if ENABLE_TASK_C_CLASS_BALANCE:
    work_dir = './work_dirs/pointconvit/taskC_class_balance'
    # Example weights: upweight rare classes (Pedestrian, Cyclist) vs Car.
    model['bbox_head']['semantic_loss'] = dict(
        type='mmdet.CrossEntropyLoss',
        class_weight=[3.0, 4.0, 1.0],
        reduction='sum',
        loss_weight=1.0)

# -----------------------------------------------------------------------------
# Task D: Proposal + NMS tuning for small object recall
# -----------------------------------------------------------------------------
if ENABLE_TASK_D_PROPOSAL_NMS:
    work_dir = './outputs/pointconvit/taskD_proposal_nms'
    # More proposals can help Ped/Cyclist recall.
    model['bbox_head']['vote_aggregation_cfg']['num_point'] = 512
    # Less aggressive filtering.
    model['test_cfg']['nms_thr'] = 0.35
    model['test_cfg']['score_thr'] = 0.01

# -----------------------------------------------------------------------------
# Task E: Head/detector swap (larger change)
# -----------------------------------------------------------------------------
if ENABLE_TASK_E_HEAD_SWAP:
    # Intentionally left as a manual step; switching to GroupFree3D requires
    # changing the detector type and matching its expected inputs/config.
    raise NotImplementedError(
        'Task E requires switching detector/head (e.g., GroupFree3DNet). '
        'Implement as a separate experiment when you reach Task E.')

# -----------------------------------------------------------------------------
# Task B/C/D/E: implement by editing THIS file only.
#
# Suggested pattern:
# - Copy this file to a backup name OR just change work_dir and continue.
#
# Task B (db sampling): already enabled in `pointconvit3d_kitti_config.py`.
#   To oversample Ped/Cyclist, override `db_sampler.sample_groups` here.
#
# Task C (class balance): add class weights (or focal) to `semantic_loss`.
#   Example (CE weights): semantic_loss = dict(type='mmdet.CrossEntropyLoss',
#     class_weight=[w_ped, w_cyc, w_car], reduction='sum', loss_weight=1.0)
#
# Task D (proposal/NMS): tune `test_cfg.nms_thr`, `test_cfg.score_thr`, and/or
#   `vote_aggregation_cfg.num_point` (proposals).
#
# Task E (head swap): switch to GroupFree3DNet + GroupFree3DHead (bigger change).
# -----------------------------------------------------------------------------

