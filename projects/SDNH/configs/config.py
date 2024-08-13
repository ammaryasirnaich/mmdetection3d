#  Example configuration for a ResNet backbone in a config file
model = dict(
    type='MVXNet',
    pts_voxel_layer=dict(
        max_num_points=32,
        voxel_size=[0.16, 0.16, 4],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 4],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[496, 432]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[128, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(
            type='DeltaXYZWLHRBBoxCoder',
            code_size=9,
            norm_bbox=True),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # Add camera branch
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4))