# Copyright (c) OpenMMLab. All rights reserved.
"""
Plot ConViT gating (locality) and token-normalised nonlocality side by side, and
report correlation (analysis 5.2). Expectation: higher gating in early layers
correlates with lower token-normalised nonlocality (convolution-like behaviour).

Usage:
  python tools/plot_convit_locality_nonlocality_correlation.py <training_folder> [--output combined.jpeg]
  python tools/plot_convit_locality_nonlocality_correlation.py <training_folder> --use-fake-data  # no dataset

Run from repo root so config _base_ paths resolve. Uses conda env mmdet.
"""

import argparse
import os
import sys

# Ensure tools dir is on path for sibling imports
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# Reuse locality extraction (no model run)
from plot_convit_locality import (
    _discover_checkpoints,
    _extract_locality_per_checkpoint,
    _load_config_and_neck_layout,
)
# Nonlocality data (runs model on data)
from plot_convit_nonlocality import get_nonlocality_data


def _build_locality_data(training_folder: str):
    """Build data_by_layer for locality (gating): layer -> [(epoch, value), ...]."""
    depth, local_up_to_layer = _load_config_and_neck_layout(training_folder)
    checkpoints = _discover_checkpoints(training_folder)
    data_by_layer = {i: [] for i in range(depth)}
    for ckpt_path, epoch_from_list in checkpoints:
        epoch, values = _extract_locality_per_checkpoint(
            ckpt_path, depth, local_up_to_layer, epoch_fallback=epoch_from_list
        )
        if epoch is None:
            continue
        for layer_idx, val in enumerate(values):
            data_by_layer[layer_idx].append((epoch, val))
    return data_by_layer, depth


def _plot_dual_and_correlation(
    data_by_layer_locality,
    data_by_layer_nonlocality,
    data_uniform_baseline,
    depth: int,
    output_path: str,
    title: str = "ConViT 3D: locality vs nonlocality",
):
    """Dual-panel figure and optional correlation summary."""
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from scipy.stats import pearsonr
    except ImportError:
        pearsonr = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap((i + 0.5) / depth) for i in range(depth)]

    # Left: locality (gating) vs epoch
    for layer_idx in range(depth):
        points = data_by_layer_locality.get(layer_idx, [])
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        epochs = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        ax1.plot(
            epochs,
            values,
            color=colors[layer_idx],
            label=f"Layer {layer_idx + 1}",
        )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Locality (mean σ(gating))")
    ax1.set_title("Gating locality")
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax1.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax1.set_ylim(bottom=0)

    # Right: nonlocality (token) vs epoch
    for layer_idx in range(depth):
        points = data_by_layer_nonlocality.get(layer_idx, [])
        if not points:
            continue
        points = [(e, v) for e, v in points if not (isinstance(v, float) and math.isnan(v))]
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        epochs = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        ax2.plot(
            epochs,
            values,
            color=colors[layer_idx],
            label=f"Layer {layer_idx + 1}",
        )
    if data_uniform_baseline:
        pts = [(e, v) for e, v in data_uniform_baseline if not (isinstance(v, float) and math.isnan(v))]
        if pts:
            pts.sort(key=lambda x: x[0])
            ep = np.array([p[0] for p in pts])
            uv = np.array([p[1] for p in pts])
            ax2.plot(ep, uv, color="gray", linestyle="--", linewidth=1.5, label="Uniform (baseline)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Nonlocality (token-normalised)")
    ax2.set_title("Nonlocality (Eq. 8)")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax2.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax2.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_path, dpi=150, format="jpeg", bbox_inches="tight")
    plt.close()

    # Correlation: per layer, Pearson between gating and (negative) nonlocality across epochs
    if pearsonr is None:
        print("scipy not available; skipping correlation.")
        return

    print("\n--- Gating vs token-normalised nonlocality (per layer, across epochs) ---")
    print("Expected: negative correlation (high gating -> low nonlocality = convolution-like).")
    for layer_idx in range(depth):
        loc_pts = data_by_layer_locality.get(layer_idx, [])
        nl_pts = data_by_layer_nonlocality.get(layer_idx, [])
        if not loc_pts or not nl_pts:
            continue
        by_epoch_loc = {e: v for e, v in loc_pts}
        by_epoch_nl = {e: v for e, v in nl_pts if not (isinstance(v, float) and math.isnan(v))}
        common_epochs = sorted(set(by_epoch_loc) & set(by_epoch_nl))
        if len(common_epochs) < 2:
            continue
        g = np.array([by_epoch_loc[e] for e in common_epochs])
        n = np.array([by_epoch_nl[e] for e in common_epochs])
        r, p = pearsonr(g, n)
        print(f"  Layer {layer_idx + 1}: r = {r:.3f}, p = {p:.4f}  (n_epochs={len(common_epochs)})")
    print("---")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ConViT locality (gating) and token nonlocality side by side; report correlation."
    )
    parser.add_argument(
        "training_folder",
        type=str,
        help="Path to training folder (config + checkpoints)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="locality_nonlocality_combined.jpeg",
        help="Output path for the combined figure",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="ConViT 3D: locality vs nonlocality",
        help="Figure title",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use random points for nonlocality (no dataset required)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches for nonlocality (default 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: cuda if available",
    )
    args = parser.parse_args()

    training_folder = os.path.abspath(args.training_folder)
    if not os.path.isdir(training_folder):
        print(f"Error: training folder is not a directory: {training_folder}", file=sys.stderr)
        sys.exit(1)

    # Locality from checkpoints only (no model run)
    print("Loading locality (gating) from checkpoints...")
    data_by_layer_locality, depth = _build_locality_data(training_folder)

    # Nonlocality: run model with token-normalised distance
    print("Computing token-normalised nonlocality (this runs the model)...")
    data_by_layer_nonlocality, data_uniform_baseline, depth_nl = get_nonlocality_data(
        training_folder,
        use_fake_data=args.use_fake_data,
        num_batches=args.num_batches,
        distance_unit="token",
        device=args.device,
    )
    if depth_nl != depth:
        print(f"Warning: depth from config ({depth}) vs nonlocality ({depth_nl}) differ.", file=sys.stderr)

    output_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _plot_dual_and_correlation(
        data_by_layer_locality,
        data_by_layer_nonlocality,
        data_uniform_baseline,
        depth,
        output_abs,
        title=args.title,
    )
    print(f"Saved combined plot to {output_abs}")


if __name__ == "__main__":
    main()
