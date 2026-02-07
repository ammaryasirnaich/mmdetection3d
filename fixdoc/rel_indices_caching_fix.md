# PointCont3D: rel_indices Caching Fix

## Overview

This document describes the fix applied to the PointCont3D neck (`mmdet3d/models/necks/pointcont3d.py`) to correct incorrect caching of `rel_indices` (relative position encodings) for 3D point clouds. The fix ensures correct behavior without increasing compute or memory during the forward pass.

## Problem

**Location:** `pointcont3d.py` (pre-fix), GPSA forward and `get_attention`.

**Original logic:** `rel_indices` was recomputed only when:
- `not hasattr(self, 'rel_indices')`, or
- `self.rel_indices.shape[0] != B` (batch size changed).

**Issue:** In 3D, relative position encodings depend on **coordinates** (`voxel_coord`), not just batch size. Different batches with the same `B` were reusing the previous batch’s relative positions, which is wrong and can hurt detection performance.

## Solution

Compute `rel_indices` **once per forward** at the neck and pass the same tensor into all blocks. No caching across forwards; no per-block recomputation.

## Code Changes

### 1. Shared helper: `_compute_rel_indices_3d(point_clouds)`

- **Where:** Module level in `pointcont3d.py` (before class `GPSA`).
- **Purpose:** Single implementation of relative position encoding for 3D point clouds.
- **Input:** `point_clouds` of shape `(batch_size, num_points, 3)` (x, y, z).
- **Output:** Tensor of shape `(batch_size, num_points, num_points, 4)` with (dx, dy, dz, Euclidean distance).
- **Usage:** Called once per forward from the neck; optionally from GPSA when `rel_indices` is not passed (e.g. backward compatibility).

### 2. Neck: `VisionTransformer.forward_features`

- **Change:** Compute `rel_indices` once at the start of the block loop, then pass it into every block.
- **Code pattern:**
  - `rel_indices = _compute_rel_indices_3d(voxel_coors)`
  - `for blk in self.blocks: x = blk(x, voxel_coors, rel_indices)`
- **Effect:** One `_compute_rel_indices_3d` call per forward; same tensor reused by all blocks.

### 3. Block: `Block.forward`

- **Change:** Accept optional `rel_indices` and pass it to the attention module.
- **Signature:** `forward(self, x, voxel_coords, rel_indices=None)`.
- **Code:** `self.attn(self.norm1(x), voxel_coords, rel_indices)`.

### 4. GPSA: `GPSA.forward` and `GPSA.get_attention`

- **`GPSA.forward(self, x, voxel_coord, rel_indices=None)`:**
  - Removed all caching logic (no more `self.rel_indices`).
  - If `rel_indices is None`, compute once with `rel_indices = _compute_rel_indices_3d(voxel_coord)`.
  - Call `get_attention(x, rel_indices)`.
- **`GPSA.get_attention(self, x, rel_indices)`:**
  - Now takes `rel_indices` as an argument.
  - Uses the passed-in `rel_indices` instead of `self.rel_indices` for the position score.

### 5. MHSA: `MHSA.forward`

- **Change:** Signature updated so it can be called like GPSA from `Block`.
- **Signature:** `forward(self, x, voxel_coords=None, rel_indices=None)`.
- **Behavior:** `voxel_coords` and `rel_indices` are ignored (MHSA does not use relative positions).

## Data Flow (After Fix)

```
VisionTransformer.forward_features(x, voxel_coors)
  rel_indices = _compute_rel_indices_3d(voxel_coors)   # once per forward
  for blk in self.blocks:
    x = blk(x, voxel_coors, rel_indices)

Block.forward(x, voxel_coords, rel_indices)
  attn_out = self.attn(norm1(x), voxel_coords, rel_indices)  # GPSA or MHSA
  ...

GPSA.forward(x, voxel_coord, rel_indices)
  # rel_indices provided by neck (or computed here if None)
  attn = self.get_attention(x, rel_indices)
  ...
```

## Impact

| Aspect | Before (buggy) | After (fix) |
|--------|----------------|-------------|
| **Correctness** | Wrong reuse across batches (same B, different coords) | Correct: one computation per forward from current `voxel_coors` |
| **Compute** | 1× on cache miss, 0× on reuse (incorrect reuse) | 1× per forward (single call at neck) |
| **Peak memory** | One (B, N, N, 4) per GPSA when used | One (B, N, N, 4) shared for block loop |
| **Persistent memory** | Cached on each GPSA module | No caching; slightly lower |

## Reference

- **Review doc:** `PointCont3D_ConViT_Review.md` (section “2. rel_indices caching is incorrect for 3D”).
- **Modified file:** `mmdet3d/models/necks/pointcont3d.py`.
