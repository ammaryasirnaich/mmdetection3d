def project_image_features(self, img_feats, depth_map):
    # Assuming img_feats is a list of feature maps from FPN
    projected_features = []
    for feat in img_feats:
        # Project each feature map to 3D using depth estimation
        # Depth map is used to lift the 2D features to 3D voxel space
        proj_feat = lift_splat_shoot(feat, depth_map)
        projected_features.append(proj_feat)
    return projected_features



