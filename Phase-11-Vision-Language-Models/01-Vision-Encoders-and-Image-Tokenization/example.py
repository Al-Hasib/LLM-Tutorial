"""
Vision encoders and image tokenization -- a ViT patch tokenizer built from scratch.

No downloads, no pretrained weights, no internet. Everything here runs on
synthetic tensors on CPU in a few seconds, but every number printed is really
computed, not asserted.

What this script demonstrates, in order:

  1. Patch embedding: turning an (C, H, W) image tensor into a sequence of
     (N_patches, d_model) vectors with a single strided Conv2d -- the exact
     operation every ViT-family vision tower starts with.
  2. The token-count explosion: N_patches grows with the SQUARE of resolution,
     and attention cost grows with the square of N_patches, so vision-tower
     cost grows with the FOURTH power of image side length. Measured here as
     real token counts and real attention-matrix element counts.
  3. 2D positional embedding interpolation: a ViT pretrained at 224x224 has a
     fixed-size position grid; running it at 448x448 requires bicubically
     resizing that grid. We do it and measure the round-trip error, showing
     why it works at all.
  4. Dynamic tiling ("AnyRes"): the production alternative to interpolation --
     cut a big image into 224x224 tiles, encode each at native scale, and
     concatenate. We compute the token/attention budget of both strategies.
  5. Pooling strategies: CLS token vs mean pooling over patch tokens, and why
     a VLM keeps ALL patch tokens instead of pooling to one vector.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Patch embedding: the entire "image tokenizer"
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Cut an image into non-overlapping patches and linearly project each one.

    The trick every ViT implementation uses: a Conv2d whose kernel size EQUALS
    its stride is exactly "flatten each patch, then apply one shared Linear".
    There is no convolutional inductive bias left -- patches never overlap.
    """

    def __init__(self, image_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size          # patches per side
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # One learned position vector per grid cell (no CLS position here).
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, images):
        # images: (B, C, H, W)
        x = self.proj(images)                          # (B, d_model, grid, grid)
        x = x.flatten(2).transpose(1, 2)               # (B, num_patches, d_model)
        x = x + self.pos_embed                         # position is ADDED, as in Phase 02
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)              # (B, 1 + num_patches, d_model)


print("=" * 74)
print("1. PATCH EMBEDDING: an image becomes a sequence of vectors")
print("=" * 74)

D_MODEL = 384
embed = PatchEmbed(image_size=224, patch_size=16, d_model=D_MODEL)
images = torch.randn(2, 3, 224, 224)                   # a batch of 2 fake images
tokens = embed(images)

print(f"input  images shape : {tuple(images.shape)}  (B, C, H, W) -- raw pixels")
print(f"output tokens shape : {tuple(tokens.shape)}  (B, 1+N_patches, d_model)")
print(f"patch grid          : {embed.grid} x {embed.grid} = {embed.num_patches} patch tokens (+1 CLS)")
print(f"pixels per patch    : 16x16x3 = {16 * 16 * 3} numbers -> {D_MODEL}-dim vector")
print()
print("Sanity check: the Conv2d really is 'flatten patch, then one shared Linear'.")
patch_00 = images[0, :, 0:16, 0:16].reshape(-1)                       # first patch, flattened
w = embed.proj.weight.reshape(D_MODEL, -1)                            # (d_model, C*p*p)
manual = w @ patch_00 + embed.proj.bias
conv_out = embed.proj(images)[0, :, 0, 0]
print(f"  max |manual_linear - conv2d| = {(manual - conv_out).abs().max().item():.3e}  (~0 => identical op)")
print()


# ---------------------------------------------------------------------------
# 2. The cost curve: tokens grow as resolution^2, attention as resolution^4
# ---------------------------------------------------------------------------

print("=" * 74)
print("2. WHY RESOLUTION IS EXPENSIVE (real counts, patch=16)")
print("=" * 74)
print(f"{'resolution':>12} {'patch tokens':>14} {'attn matrix cells':>20} {'rel. attn cost':>16}")
base_attn = None
for res in [224, 336, 448, 672, 896, 1344]:
    grid = res // 16
    n = grid * grid
    attn_cells = n * n                        # one head, one layer: N x N scores
    if base_attn is None:
        base_attn = attn_cells
    print(f"{res:>9}px {n:>14,} {attn_cells:>20,} {attn_cells / base_attn:>15.1f}x")
print()
print("Token count scales with res^2; self-attention with (res^2)^2 = res^4.")
print("Going 224 -> 896 (4x the side) costs 256x the attention work in the")
print("vision tower AND puts 16x more tokens into the LLM's context window.")
print("That second cost is usually the one that hurts: see Lessons 4 and 10.")
print()


# ---------------------------------------------------------------------------
# 3. Positional-embedding interpolation for a new resolution
# ---------------------------------------------------------------------------

def interpolate_pos_embed(pos_embed, old_grid, new_grid):
    """Resize a (1, old_grid^2, d) position grid to (1, new_grid^2, d) bicubically."""
    d = pos_embed.shape[-1]
    grid = pos_embed.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)   # (1, d, g, g)
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d)


print("=" * 74)
print("3. RUNNING A 224px-PRETRAINED TOWER AT 448px: position interpolation")
print("=" * 74)

old_grid, new_grid = 14, 28
pos_224 = embed.pos_embed.detach()
pos_448 = interpolate_pos_embed(pos_224, old_grid, new_grid)
print(f"pretrained position grid : {tuple(pos_224.shape)}  ({old_grid}x{old_grid} = {old_grid ** 2} positions)")
print(f"interpolated to          : {tuple(pos_448.shape)}  ({new_grid}x{new_grid} = {new_grid ** 2} positions)")

# Round trip: shrink back and compare. A *random* position grid is high-frequency
# noise, the worst possible case for interpolation; a trained one is far smoother.
# We show both to make the point that smoothness -- not the resizing itself -- is
# what makes interpolation viable.
round_trip = interpolate_pos_embed(pos_448, new_grid, old_grid)
err = (round_trip - pos_224).abs().mean().item()
scale = pos_224.abs().mean().item()
print(f"random (untrained) grid, round-trip 14->28->14 relative error : {100 * err / scale:5.2f}%")

# Now a smooth grid, standing in for a trained one: real learned 2D position
# grids are strongly spatially correlated (neighbouring positions get similar
# vectors), which is exactly the property interpolation needs.
smooth = F.avg_pool2d(
    F.pad(pos_224.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate"),
    kernel_size=3, stride=1,
).permute(0, 2, 3, 1).reshape(1, old_grid ** 2, -1)
smooth_rt = interpolate_pos_embed(interpolate_pos_embed(smooth, old_grid, new_grid), new_grid, old_grid)
serr = (smooth_rt - smooth).abs().mean().item() / smooth.abs().mean().item()
print(f"smooth  (trained-like) grid, same round trip relative error   : {100 * serr:5.2f}%")
print("Smoothness is the empirical reason interpolation works at all; it is still")
print("an approximation, which is why models are usually fine-tuned briefly at the")
print("new resolution afterwards.")
print()


# ---------------------------------------------------------------------------
# 4. Dynamic tiling (AnyRes) vs. one big interpolated forward pass
# ---------------------------------------------------------------------------

print("=" * 74)
print("4. TWO WAYS TO HANDLE A 896x896 IMAGE (patch=16, base tower 224px)")
print("=" * 74)

res, tile = 896, 224
n_tiles = (res // tile) ** 2
tokens_per_tile = (tile // 16) ** 2

single_pass_tokens = (res // 16) ** 2
single_pass_attn = single_pass_tokens ** 2
tiled_tokens = n_tiles * tokens_per_tile + tokens_per_tile   # + a downscaled global "thumbnail" tile
tiled_attn = (n_tiles + 1) * (tokens_per_tile ** 2)

print("A) one forward pass at 896px (interpolated positions)")
print(f"   vision tokens   : {single_pass_tokens:,}")
print(f"   attention cells : {single_pass_attn:,}")
print("   every patch attends to every other patch (full global context)")
print()
print(f"B) tiled: {n_tiles} x {tile}px tiles + 1 downscaled global thumbnail")
print(f"   vision tokens   : {tiled_tokens:,}  (same order -- tokens are NOT saved)")
print(f"   attention cells : {tiled_attn:,}  ({single_pass_attn / tiled_attn:.1f}x cheaper)")
print("   each tile is encoded at the tower's NATIVE resolution -- no position")
print("   interpolation, no train/test mismatch -- but patches in different tiles")
print("   never attend to each other inside the tower, so the LLM has to stitch")
print("   the tiles together itself. The thumbnail tile is what gives it the")
print("   global layout.")
print()


# ---------------------------------------------------------------------------
# 5. Pooling: what a classifier keeps vs. what a VLM keeps
# ---------------------------------------------------------------------------

print("=" * 74)
print("5. POOLING: one vector (CLIP) vs. all patch tokens (VLM)")
print("=" * 74)

cls_vec = tokens[:, 0, :]                 # (B, d) -- the CLS summary
mean_vec = tokens[:, 1:, :].mean(dim=1)   # (B, d) -- mean-pooled patches
all_patches = tokens[:, 1:, :]            # (B, N, d) -- everything

print(f"CLS pooled       : {tuple(cls_vec.shape)}   -> {cls_vec.numel():>7,} numbers/batch")
print(f"mean pooled      : {tuple(mean_vec.shape)}   -> {mean_vec.numel():>7,} numbers/batch")
print(f"all patch tokens : {tuple(all_patches.shape)} -> {all_patches.numel():>7,} numbers/batch")
print(f"ratio            : a VLM carries {all_patches.numel() / cls_vec.numel():.0f}x more information forward")
print()

# A concrete demonstration that pooling destroys location information:
# two images containing the SAME content in DIFFERENT places.
img_a = torch.zeros(1, 3, 224, 224)
img_a[:, :, :112, :112] = 1.0                 # bright square, top-left
img_b = torch.zeros(1, 3, 224, 224)
img_b[:, :, 112:, 112:] = 1.0                 # same square, bottom-right

with torch.no_grad():
    ta, tb = embed(img_a), embed(img_b)
pe = embed.pos_embed.detach()
seq_a, seq_b = ta[:, 1:], tb[:, 1:]
# Remove the position term so we isolate what POOLING alone loses.
mean_a_np = (seq_a - pe).mean(1)
mean_b_np = (seq_b - pe).mean(1)

print("Two images: an identical bright square, top-left vs bottom-right.")
print(f"  cosine(mean-pooled, position term removed) : {F.cosine_similarity(mean_a_np, mean_b_np).item():.4f}  <- indistinguishable")
print(f"  fraction of patch tokens that differ       : {(seq_a - seq_b).abs().sum(-1).gt(1e-6).float().mean().item():.0%}")
print()
print("Mean pooling collapses 'where' almost entirely; the token sequence keeps it.")
print("CLIP's contrastive loss (Lesson 2) only ever needs ONE vector per image, so")
print("its training signal never has to preserve location. A VLM asked 'what is")
print("written on the left-hand sign?' needs per-location detail, so it feeds the")
print("whole patch sequence to the LLM -- which is precisely why vision tokens")
print("dominate a VLM's context budget, and why Lesson 4 is about compressing them")
print("without throwing that detail away.")
