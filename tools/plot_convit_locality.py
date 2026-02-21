# Copyright (c) OpenMMLab. All rights reserved.
"""
Plot ConViT-style locality (mean sigma(gating)) per layer over training epochs.

Usage:
    python tools/plot_convit_locality.py <training_folder> [--output plot.jpeg] [--title "ConViT 3D"]

Expects training_folder to contain:
  - A single .py config file (dumped config with model.neck.depth and model.neck.local_up_to_layer)
  - Checkpoints epoch_1.pth, epoch_2.pth, ... (or iter_*.pth)
"""

import argparse
import glob
import os
import re
import sys
from typing import Optional

import torch


def _find_config(training_folder: str):
    """Find the single .py config in training_folder."""
    folder = os.path.abspath(training_folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Training folder is not a directory: {folder}")
    py_files = glob.glob(os.path.join(folder, "*.py"))
    # Prefer config.py if present
    for p in py_files:
        if os.path.basename(p) == "config.py":
            return p
    if len(py_files) == 0:
        raise FileNotFoundError(
            f"No .py config file found in {folder}. "
            "Expected a dumped config (e.g. pointconvit3d_kitti_improved.py)."
        )
    if len(py_files) > 1:
        # Prefer one that looks like a config (contains model.neck or neck=)
        for p in py_files:
            with open(p, "r") as f:
                if "neck" in f.read():
                    return p
        # Otherwise take the first
        return py_files[0]
    return py_files[0]


def _load_config_and_neck_layout(training_folder: str):
    """Load config from training folder and return depth, local_up_to_layer."""
    from mmengine import Config

    config_path = _find_config(training_folder)
    cfg = Config.fromfile(config_path)
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "neck"):
        raise ValueError(
            "Config must define model.neck (ConViT-style neck with depth and local_up_to_layer). "
            f"Loaded from: {config_path}"
        )
    neck = cfg.model.neck
    depth = getattr(neck, "depth", 12)
    local_up_to_layer = getattr(neck, "local_up_to_layer", 10)
    return depth, local_up_to_layer


def _discover_checkpoints(training_folder: str):
    """
    List .pth files in training_folder and return list of (path, epoch).
    Epoch from checkpoint['meta']['epoch'] if present, else from filename epoch_N.pth or iter_N.pth.
    Deduplicated by epoch (one checkpoint per epoch).
    """
    folder = os.path.abspath(training_folder)
    pth_files = glob.glob(os.path.join(folder, "*.pth"))
    # Optionally include last_checkpoint target
    last_ckpt = os.path.join(folder, "last_checkpoint")
    if os.path.isfile(last_ckpt):
        with open(last_ckpt, "r") as f:
            content = f.read().strip()
        # MMEngine may write "epoch_10" or a path
        target = os.path.join(folder, content) if not os.path.isabs(content) else content
        if os.path.isfile(target) and target not in pth_files:
            pth_files.append(target)

    epoch_from_filename = re.compile(r"epoch_(\d+)\.pth", re.IGNORECASE)
    iter_from_filename = re.compile(r"iter_(\d+)\.pth", re.IGNORECASE)

    result = []  # (path, epoch)
    for path in pth_files:
        epoch = None
        try:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
                epoch = ckpt["meta"].get("epoch")
        except Exception:
            pass
        if epoch is None:
            base = os.path.basename(path)
            m = epoch_from_filename.match(base)
            if m:
                epoch = int(m.group(1))
            else:
                m = iter_from_filename.match(base)
                if m:
                    # Treat iter as epoch for plotting (user can rename)
                    epoch = int(m.group(1))
        if epoch is not None:
            result.append((path, epoch))

    if not result:
        raise FileNotFoundError(
            f"No valid checkpoints (epoch_*.pth or iter_*.pth) found in {training_folder}. "
            "Each file must have epoch in meta or in filename."
        )

    # Deduplicate by epoch: keep one per epoch (prefer filename matching epoch)
    by_epoch = {}
    for path, epoch in result:
        if epoch not in by_epoch or path.endswith(f"epoch_{epoch}.pth"):
            by_epoch[epoch] = path
    return [(by_epoch[e], e) for e in sorted(by_epoch.keys())]


def _extract_locality_per_checkpoint(
    checkpoint_path: str,
    depth: int,
    local_up_to_layer: int,
    epoch_fallback: Optional[int] = None,
):
    """
    Load checkpoint and return (epoch, list of locality values for layer 0..depth-1).
    Locality = mean(sigmoid(gating_param)) for GPSA blocks; 0 for MHSA.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    epoch = None
    if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
        epoch = ckpt["meta"].get("epoch")
    if epoch is None:
        base = os.path.basename(checkpoint_path)
        m = re.match(r"epoch_(\d+)\.pth", base, re.IGNORECASE)
        if m:
            epoch = int(m.group(1))
        else:
            m = re.match(r"iter_(\d+)\.pth", base, re.IGNORECASE)
            if m:
                epoch = int(m.group(1))
    if epoch is None:
        epoch = epoch_fallback

    values = []
    for layer_idx in range(depth):
        key = f"neck.blocks.{layer_idx}.attn.gating_param"
        if key in state_dict:
            g = state_dict[key]
            if isinstance(g, torch.Tensor):
                val = torch.sigmoid(g).mean().item()
            else:
                val = float(torch.sigmoid(torch.tensor(g)).mean())
            values.append(val)
        else:
            # MHSA block or missing
            values.append(0.0)
    return epoch, values


def _plot_locality(
    data_by_layer,
    depth: int,
    output_path: str,
    title: str = "ConViT",
):
    """Build Figure-5-style plot: one line per layer, x=epochs, y=locality."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    # Color gradient: cooler (Layer 1) to warmer (Layer 12)
    cmap = plt.get_cmap("viridis")
    colors = [cmap((i + 0.5) / depth) for i in range(depth)]

    for layer_idx in range(depth):
        points = data_by_layer[layer_idx]  # list of (epoch, value)
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        epochs = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        ax.plot(
            epochs,
            values,
            color=colors[layer_idx],
            label=f"Layer {layer_idx + 1}",
        )

    ax.set_xlabel("Epochs")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, format="jpeg", bbox_inches="tight")
    plt.close()


def plot_convit_locality(
    training_folder: str,
    output_path: str = "plot.jpeg",
    title: str = "ConViT",
):
    """
    Load config and checkpoints from training_folder, extract locality per layer
    per epoch, and save a Figure-5-style plot to output_path.

    Returns:
        str: Absolute path to the saved figure.
    """
    depth, local_up_to_layer = _load_config_and_neck_layout(training_folder)
    checkpoints = _discover_checkpoints(training_folder)

    # data_by_layer[layer_idx] = [(epoch, value), ...]
    data_by_layer = {i: [] for i in range(depth)}
    for ckpt_path, epoch_from_list in checkpoints:
        epoch, values = _extract_locality_per_checkpoint(
            ckpt_path, depth, local_up_to_layer, epoch_fallback=epoch_from_list
        )
        if epoch is None:
            continue
        for layer_idx, val in enumerate(values):
            data_by_layer[layer_idx].append((epoch, val))

    output_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _plot_locality(data_by_layer, depth, output_abs, title=title)
    return output_abs


def main():
    parser = argparse.ArgumentParser(
        description="Plot ConViT locality (mean σ(gating)) per layer over epochs."
    )
    parser.add_argument(
        "training_folder",
        type=str,
        help="Path to training folder containing config .py and checkpoint .pth files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="plot.jpeg",
        help="Output path for the plot (default: plot.jpeg)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="ConViT 3D",
        help="Plot title (default: ConViT 3D)",
    )
    args = parser.parse_args()

    try:
        out = plot_convit_locality(
            args.training_folder,
            output_path=args.output,
            title=args.title,
        )
        print(f"Saved locality plot to {out}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
