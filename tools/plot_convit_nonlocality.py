# Copyright (c) OpenMMLab. All rights reserved.
"""
Plot ConViT Section 4 (Figure 5) nonlocality metric per layer over training epochs.

This aligns with the ConViT paper "Investigating the role of locality" (Section 4, Eq. 8):
  D_loc^(l,h) = (1/L) * sum_ij A_ij^(l,h) * ||delta_ij||
Higher D_loc = more non-local (attention spread over larger distances).

Usage:
  python tools/plot_convit_nonlocality.py <training_folder> [--output nonlocality_plot.jpeg]
  python tools/plot_convit_nonlocality.py <training_folder> --use-fake-data  # no dataset needed

Expects training_folder to contain:
  - A single .py config file (with model.neck.depth, model.neck.local_up_to_layer)
  - Checkpoints epoch_1.pth, epoch_2.pth, ...

Without --use-fake-data, the config must define test_dataloader (or train_dataloader)
so we can run the model on real data to compute attention. Run from repo root so
config _base_ paths resolve.
"""

import argparse
import math
import os
import re
import sys

import torch

# Ensure this script can import sibling modules when executed as:
#   python tools/plot_convit_nonlocality.py ...
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# PyTorch 2.6+ defaults to weights_only=True; MMEngine checkpoints can contain
# numpy scalars. Patch torch.load so checkpoint loading works for trusted checkpoints.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Reuse config/checkpoint discovery from the gating-based locality script
from plot_convit_locality import (
    _discover_checkpoints,
    _find_config,
    _load_config_and_neck_layout,
)


def _compute_nonlocality_from_captures(
    captures_per_layer, depth, local_up_to_layer, *, distance_unit: str = "meters"
):
    """
    Compute D_loc (Eq. 8) per layer from captured (attn, dist) lists.

    Paper Eq. 8 (per head): D_loc = (1/N) * sum_i sum_j A_ij * ||delta_ij||,
    then averaged over heads and batch.

    Also computes D_loc for uniform attention (A_ij = 1/N) using the same dist
    as a sanity baseline (analysis 5.4): D_loc_uniform = (1/N^2) * sum_ij d_ij.

    Returns (values, uniform_values): each is a list of one scalar per layer;
    layers without captures get float('nan'). uniform_values are averaged over
    layers in the caller to get one "Uniform (baseline)" curve per epoch.

    distance_unit:
      - 'meters': return D_loc in the coordinate units of the point cloud (typically meters)
      - 'token': return D_loc normalized by a characteristic token spacing (mean nearest-neighbor distance)
    """
    values = []
    uniform_values = []
    for layer_idx in range(depth):
        # captures_per_layer[layer_idx] is a list of (attn, dist) from one or more forwards
        # attn: (B, num_heads, N, N), dist: (B, N, N)
        layer_captures = captures_per_layer.get(layer_idx, [])
        if not layer_captures:
            values.append(float("nan"))
            uniform_values.append(float("nan"))
            continue
        total = 0.0
        uniform_total = 0.0
        count = 0
        for attn, dist in layer_captures:
            # Eq. 8 reduction:
            # per_query = sum_j A_ij * d_ij, then mean over queries i, then mean over heads/batch
            # attn (B, H, N, N), dist (B, N, N) -> broadcast dist to (B, 1, N, N)
            per_query = (attn * dist.unsqueeze(1)).sum(dim=-1)  # (B, H, N)
            per_head = per_query.mean(dim=-1)  # (B, H)
            d = per_head.mean()  # scalar tensor

            # Uniform baseline: A_ij = 1/N => D_loc_uniform = (1/N^2) * sum_ij d_ij
            B, _, N, _ = attn.shape
            d_uniform = dist.mean()  # (1/N^2)*sum_ij d_ij since dist has N*N elements

            if distance_unit == "token":
                # Normalize by mean nearest-neighbor distance to get a scale-free “token-step” unit.
                # dist is (B, N, N) with dist[i,i]=0; exclude diagonal when taking min.
                B, _, N, _ = attn.shape
                diag = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
                masked = dist.masked_fill(diag, float("inf"))
                nn_dist = masked.min(dim=-1).values  # (B, N)
                scale = nn_dist.mean().clamp_min(1e-12)
                d = d / scale
                d_uniform = d_uniform / scale

            d = float(d.item())
            d_uniform = float(d_uniform.item())
            total += d
            uniform_total += d_uniform
            count += 1
        values.append(total / count if count else float("nan"))
        uniform_values.append(uniform_total / count if count else float("nan"))
    return values, uniform_values


def _extract_nonlocality_per_checkpoint(
    checkpoint_path: str,
    cfg,
    depth: int,
    local_up_to_layer: int,
    use_fake_data: bool,
    num_batches: int,
    num_fake_points: int,
    distance_unit: str,
    device: torch.device,
    epoch_fallback: int = None,
):
    """
    Load model from config, load checkpoint, run forward(s), capture attention and
    distances from GPSA blocks, compute D_loc per layer. Returns (epoch, list of D_loc).
    """
    from mmdet3d.registry import MODELS
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model)
    try:
        load_checkpoint(model, checkpoint_path, map_location="cpu", strict=True)
    except Exception as e:
        load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()

    neck = model.neck
    if not hasattr(neck, "blocks"):
        raise ValueError("Model neck has no .blocks (expected ConViT-style VisionTransformer).")

    # Attach capture lists to ALL attention blocks so we can plot 1..depth like ConViT Fig. 5.
    # For MHSA blocks, we only capture if the attention module supports it.
    captures_per_layer = {i: [] for i in range(depth)}
    for i in range(depth):
        block = neck.blocks[i]
        block.attn._capture_nonlocality = []

    def _collect_captures():
        for i in range(depth):
            block = neck.blocks[i]
            captures_per_layer[i].extend(getattr(block.attn, "_capture_nonlocality", []))
            if getattr(block.attn, "_capture_nonlocality", None) is not None:
                block.attn._capture_nonlocality.clear()

    if use_fake_data:
        # Fake batch: list of (N, 4) tensors (xyz + intensity)
        for _ in range(num_batches):
            batch_inputs = {
                "points": [torch.randn(num_fake_points, 4, device=device, dtype=torch.float32)]
            }
            with torch.no_grad():
                model.extract_feat(batch_inputs)
            _collect_captures()
    else:
        # Real data: build dataloader from config
        from mmengine.runner import Runner

        try:
            dataloader_cfg = getattr(cfg, "test_dataloader", None) or getattr(
                cfg, "train_dataloader", None
            )
        except Exception:
            dataloader_cfg = None
        if dataloader_cfg is None:
            raise ValueError(
                "Config has no test_dataloader or train_dataloader. "
                "Use --use-fake-data to run without dataset."
            )
        dataloader = Runner.build_dataloader(dataloader_cfg)
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            # MMEngine batch: dict with 'inputs' and 'data_samples'
            inputs = batch.get("inputs", batch)
            if not isinstance(inputs, dict):
                inputs = {"points": inputs}
            # Move to device
            batch_inputs = {}
            for k, v in inputs.items():
                if k == "points" and isinstance(v, (list, tuple)):
                    batch_inputs[k] = [
                        p.to(device) if isinstance(p, torch.Tensor) else p for p in v
                    ]
                elif isinstance(v, torch.Tensor):
                    batch_inputs[k] = v.to(device)
                else:
                    batch_inputs[k] = v
            with torch.no_grad():
                model.extract_feat(batch_inputs)
            _collect_captures()
            batch_count += 1

    # Get epoch from checkpoint
    epoch = epoch_fallback
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
        epoch = ckpt["meta"].get("epoch", epoch)
    if epoch is None:
        base = os.path.basename(checkpoint_path)
        m = re.match(r"epoch_(\d+)\.pth", base, re.IGNORECASE)
        if m:
            epoch = int(m.group(1))
        else:
            m = re.match(r"iter_(\d+)\.pth", base, re.IGNORECASE)
            if m:
                epoch = int(m.group(1))

    values, uniform_values = _compute_nonlocality_from_captures(
        captures_per_layer,
        depth,
        local_up_to_layer,
        distance_unit=distance_unit,
    )
    return epoch, values, uniform_values


def _plot_nonlocality(
    data_by_layer,
    depth: int,
    output_path: str,
    title: str = "ConViT (nonlocality)",
    ylabel: str = "Nonlocality (avg attention distance)",
    data_uniform_baseline=None,
):
    """Figure-5-style plot: one line per layer, x=epochs, y=nonlocality (D_loc).
    If data_uniform_baseline is provided (list of (epoch, value)), plot one dashed
    "Uniform (baseline)" curve (analysis 5.4).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap((i + 0.5) / depth) for i in range(depth)]

    for layer_idx in range(depth):
        points = data_by_layer[layer_idx]
        if not points:
            continue
        points = [(e, v) for e, v in points if not (isinstance(v, float) and math.isnan(v))]
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

    if data_uniform_baseline:
        pts = [(e, v) for e, v in data_uniform_baseline if not (isinstance(v, float) and math.isnan(v))]
        if pts:
            pts.sort(key=lambda x: x[0])
            ep = np.array([p[0] for p in pts])
            uv = np.array([p[1] for p in pts])
            ax.plot(ep, uv, color="gray", linestyle="--", linewidth=1.5, label="Uniform (baseline)")

    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, format="jpeg", bbox_inches="tight")
    plt.close()


def get_nonlocality_data(
    training_folder: str,
    use_fake_data: bool = False,
    num_batches: int = 1,
    num_fake_points: int = 2048,
    distance_unit: str = "meters",
    device: str = None,
):
    """
    Compute ConViT nonlocality (Eq. 8) per layer per checkpoint without plotting.
    Returns (data_by_layer, data_uniform_baseline, depth) for use in combined plots
    or correlation (e.g. analysis 5.2).
    """
    from mmengine import Config

    depth, local_up_to_layer = _load_config_and_neck_layout(training_folder)
    config_path = _find_config(training_folder)
    cfg = Config.fromfile(config_path)
    checkpoints = _discover_checkpoints(training_folder)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mmdet3d.registry import MODELS
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    init_default_scope("mmdet3d")
    model = MODELS.build(cfg.model).to(device)
    model.eval()

    neck = model.neck
    if not hasattr(neck, "blocks"):
        raise ValueError("Model neck has no .blocks (expected ConViT-style VisionTransformer).")

    batches = []
    if use_fake_data:
        for _ in range(num_batches):
            batches.append(
                {"points": [torch.randn(num_fake_points, 4, device=device, dtype=torch.float32)]}
            )
    else:
        from mmengine.runner import Runner

        try:
            dataloader_cfg = getattr(cfg, "test_dataloader", None) or getattr(
                cfg, "train_dataloader", None
            )
        except Exception:
            dataloader_cfg = None
        if dataloader_cfg is None:
            raise ValueError(
                "Config has no test_dataloader or train_dataloader. Use --use-fake-data."
            )
        dataloader = Runner.build_dataloader(dataloader_cfg)
        for batch in dataloader:
            if len(batches) >= num_batches:
                break
            inputs = batch.get("inputs", batch)
            if not isinstance(inputs, dict):
                inputs = {"points": inputs}
            batch_inputs = {}
            for k, v in inputs.items():
                if k == "points" and isinstance(v, (list, tuple)):
                    batch_inputs[k] = [
                        p.to(device) if isinstance(p, torch.Tensor) else p for p in v
                    ]
                elif isinstance(v, torch.Tensor):
                    batch_inputs[k] = v.to(device)
                else:
                    batch_inputs[k] = v
            batches.append(batch_inputs)

    if not batches:
        raise ValueError("No batches available. Check dataloader and num_batches.")

    data_by_layer = {i: [] for i in range(depth)}
    data_uniform_baseline = []
    for idx, (ckpt_path, epoch_from_list) in enumerate(checkpoints, start=1):
        print(f"[{idx}/{len(checkpoints)}] Loading {os.path.basename(ckpt_path)} (epoch={epoch_from_list})")
        try:
            load_checkpoint(model, ckpt_path, map_location="cpu", strict=True)
        except Exception:
            load_checkpoint(model, ckpt_path, map_location="cpu")

        captures_per_layer = {i: [] for i in range(depth)}
        for i in range(depth):
            neck.blocks[i].attn._capture_nonlocality = []

        def _collect_captures():
            for i in range(depth):
                block = neck.blocks[i]
                captures_per_layer[i].extend(getattr(block.attn, "_capture_nonlocality", []))
                if getattr(block.attn, "_capture_nonlocality", None) is not None:
                    block.attn._capture_nonlocality.clear()

        with torch.no_grad():
            for batch_inputs in batches:
                model.extract_feat(batch_inputs)
                _collect_captures()

        values, uniform_per_layer = _compute_nonlocality_from_captures(
            captures_per_layer,
            depth,
            local_up_to_layer,
            distance_unit=distance_unit,
        )
        epoch = epoch_from_list
        for layer_idx, val in enumerate(values):
            data_by_layer[layer_idx].append((epoch, val))
        u_valid = [v for v in uniform_per_layer if not (isinstance(v, float) and math.isnan(v))]
        uniform_baseline = (sum(u_valid) / len(u_valid)) if u_valid else float("nan")
        data_uniform_baseline.append((epoch, uniform_baseline))

    return data_by_layer, data_uniform_baseline, depth


def plot_convit_nonlocality(
    training_folder: str,
    output_path: str = "nonlocality_plot.jpeg",
    title: str = "ConViT 3D (nonlocality)",
    use_fake_data: bool = False,
    num_batches: int = 1,
    num_fake_points: int = 2048,
    distance_unit: str = "meters",
    device: str = None,
):
    """
    Compute ConViT Section 4 nonlocality (Eq. 8) per layer per checkpoint and save
    a Figure-5-style plot. Uses forward hooks on GPSA to capture attention and
    pairwise distances.

    Returns:
        str: Absolute path to the saved figure.
    """
    data_by_layer, data_uniform_baseline, depth = get_nonlocality_data(
        training_folder,
        use_fake_data=use_fake_data,
        num_batches=num_batches,
        num_fake_points=num_fake_points,
        distance_unit=distance_unit,
        device=device,
    )

    output_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ylabel = (
        "Nonlocality (avg attention distance)"
        if distance_unit == "meters"
        else "Nonlocality (avg attention distance / nn_spacing)"
    )
    _plot_nonlocality(
        data_by_layer,
        depth,
        output_abs,
        title=title,
        ylabel=ylabel,
        data_uniform_baseline=data_uniform_baseline,
    )
    return output_abs


def main():
    parser = argparse.ArgumentParser(
        description="Plot ConViT nonlocality (Eq. 8, Section 4) per layer over epochs."
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
        default="nonlocality_plot.jpeg",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="ConViT 3D (nonlocality)",
        help="Plot title",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use random points instead of dataset (no data required)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to average per checkpoint (default 1)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points per sample for --use-fake-data (default 2048)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Default: cuda if available",
    )
    parser.add_argument(
        "--distance-unit",
        choices=["meters", "token"],
        default="meters",
        help="Y-axis units: raw coordinate units ('meters') or normalized by mean nearest-neighbor spacing ('token')",
    )
    args = parser.parse_args()

    try:
        out = plot_convit_nonlocality(
            args.training_folder,
            output_path=args.output,
            title=args.title,
            use_fake_data=args.use_fake_data,
            num_batches=args.num_batches,
            num_fake_points=args.num_points,
            distance_unit=args.distance_unit,
            device=torch.device(args.device) if args.device else None,
        )
        print(f"Saved nonlocality plot to {out}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
