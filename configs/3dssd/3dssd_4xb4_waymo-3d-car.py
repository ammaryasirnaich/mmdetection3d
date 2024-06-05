_base_ = [
    '../_base_/models/3dssd.py', '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'WaymoDataset'
data_root = '/workspace/data/waymo/waymo_mini/'
             

# class_names = ['Car']
point_cloud_range = [-76.8, -51.2, -2, 76.8, 51.2, 4]

backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectSample', db_sampler=db_sampler),
      dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    # 3DSSD can get a higher performance without this transform
    # dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='PointSample', num_points=16384),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
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
            dict(type='PointSample', num_points=16384),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4, dataset=dict(dataset=dict(pipeline=train_pipeline, )))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# model settings
model = dict(
    bbox_head=dict(
        num_classes=3,
        
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)))

# optimizer
lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
