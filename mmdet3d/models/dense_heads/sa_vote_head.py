# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Tuple

from mmdet3d.registry import MODELS
from .vote_head import VoteHead


@MODELS.register_module()
class SaVoteHead(VoteHead):
    """VoteHead variant that consumes SA outputs (sa_xyz/sa_features/sa_indices).

    This is a minimal adapter for pipelines whose backbone/neck produce only
    Set-Abstraction (SA) features (e.g. PointNet2SAMSG + custom neck), so we
    can use VoteHead-style multi-class supervision without requiring
    fp_xyz/fp_features/fp_indices (feature propagation outputs).
    """

    def _extract_input(self, feat_dict: dict) -> Tuple:
        # Prefer SA outputs when present (matches SSD3DHead usage).
        if ('sa_xyz' in feat_dict and 'sa_features' in feat_dict
                and 'sa_indices' in feat_dict):
            seed_points = feat_dict['sa_xyz'][-1]
            seed_features = feat_dict['sa_features'][-1]
            seed_indices = feat_dict['sa_indices'][-1]
            return seed_points, seed_features, seed_indices

        # Fallback to the default VoteHead behavior (fp_* or imvotenet keys).
        return super()._extract_input(feat_dict)

