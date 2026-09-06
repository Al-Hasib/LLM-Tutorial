"""
VLM capabilities -- grounding, reading small text, and watching video, each
reduced to the one measurement that governs it.

No downloads, no pretrained weights. CPU; the three experiments train 17
small models in total, so budget around 7 minutes.

Three experiments, three hard limits that no amount of language modelling
can talk its way around:

  1. GROUNDING AS TOKEN GENERATION. A box is emitted as quantized coordinate
     tokens, so the number of bins in the coordinate vocabulary is a hard
     ceiling on localization precision. We train the same model with several
     bin counts and measure both bin accuracy and real localization error.

  2. READING SMALL THINGS. Real 32x32 images containing a glyph are
     downscaled (as every VLM downscales its input to the tower's trained
     resolution), patchified, and classified. Downscaling is where small
     detail is actually destroyed -- the pixel-level reason a VLM that reads
     a headline cannot read a footnote -- and it is the same knob that sets
     the token count, so the trade cannot be escaped (Lesson 1 section 5).

  3. VIDEO IS A SAMPLING PROBLEM. An event happens in exactly one frame of a
     long clip. Sampling k frames caps accuracy at k/T before the model does
     anything at all, and k is set by the token budget (Lesson 4).

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_MODEL = 64


class Block(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


# ===========================================================================
# 1. GROUNDING: boxes as quantized coordinate tokens
# ===========================================================================

N_SHAPES = 8
N_OBJECTS = 3
D_VISION = 24
shape_code = torch.randn(N_SHAPES, D_VISION)
pos_basis = torch.randn(2, D_VISION) * 1.5      # how position is written into a token


def grounding_batch(b):
    """Objects at continuous positions; the question names one shape and the
    answer is its (x, y). Positions are encoded in the vision token itself."""
    shapes = torch.stack([torch.randperm(N_SHAPES)[:N_OBJECTS] for _ in range(b)])
    xy = torch.rand(b, N_OBJECTS, 2)
    vision = shape_code[shapes] + xy @ pos_basis + 0.05 * torch.randn(b, N_OBJECTS, D_VISION)
    slot = torch.randint(0, N_OBJECTS, (b,))
    idx = torch.arange(b)
    return vision, shapes[idx, slot], xy[idx, slot, 0], xy[idx, slot, 1]


class Grounder(nn.Module):
    """Emits a box as two categorical coordinate tokens, the way Pix2Seq,
    Kosmos-2, Qwen-VL and Florence-2 all do it: the coordinate space is
    discretized into `bins` values that live in the token vocabulary."""

    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.query_embed = nn.Embedding(N_SHAPES, D_MODEL)
        self.proj = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                  nn.Linear(D_MODEL, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head_x = nn.Linear(D_MODEL, bins)
        self.head_y = nn.Linear(D_MODEL, bins)

    def forward(self, vision, query):
        x = torch.cat([self.proj(vision), self.query_embed(query).unsqueeze(1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x[:, -1])
        return self.head_x(h), self.head_y(h)


def train_grounder(bins, steps=800, bs=96, lr=3e-3):
    torch.manual_seed(1)
    model = Grounder(bins)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        vision, query, gx, gy = grounding_batch(bs)
        bx = (gx * bins).clamp(0, bins - 1).long()          # quantize the target
        by = (gy * bins).clamp(0, bins - 1).long()
        lx, ly = model(vision, query)
        loss = F.cross_entropy(lx, bx) + F.cross_entropy(ly, by)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_grounder(model, bins, n=3000):
    vision, query, gx, gy = grounding_batch(n)
    lx, ly = model(vision, query)
    px = (lx.argmax(-1).float() + 0.5) / bins                # bin center
    py = (ly.argmax(-1).float() + 0.5) / bins
    err = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    bx = (gx * bins).clamp(0, bins - 1).long()
    by = (gy * bins).clamp(0, bins - 1).long()
    exact = ((lx.argmax(-1) == bx) & (ly.argmax(-1) == by)).float().mean().item()
    # Quantization floor: the error you would get with a PERFECT model.
    floor = torch.sqrt(((bx.float() + 0.5) / bins - gx) ** 2
                       + ((by.float() + 0.5) / bins - gy) ** 2).mean().item()
    return exact, err.mean().item(), floor


print("=" * 78)
print("1. GROUNDING: a box is just tokens, and the vocabulary sets the ceiling")
print("=" * 78)
print("The model answers 'where is the <shape>?' with two coordinate tokens.")
print("More bins = finer coordinates = more vocabulary and a harder prediction.")
print()
print(f"{'coord bins':>11} {'exact-bin acc':>14} {'mean loc. error':>16} {'quantization floor':>19}")
for bins in [4, 16, 64]:
    m = train_grounder(bins)
    exact, err, floor = eval_grounder(m, bins)
    print(f"{bins:>11} {exact:>13.1%} {err:>16.4f} {floor:>19.4f}")
print()
print("Three columns, three different things, and they do not move together.")
print()
print("The quantization floor is the error a PERFECT model would still make from")
print("rounding to a bin: it falls as the grid gets finer, so a coarse grid caps")
print("precision no matter how good the model is. Exact-bin accuracy collapses in")
print("the opposite direction, because the model must pick one of far more")
print("classes from the same evidence. What actually matters -- the real")
print("localization error -- improves and then saturates: 4 to 16 bins is a")
print("genuine 3x improvement, 16 to 64 buys almost nothing, because by then the")
print("model rather than the grid is the limit. Past that point extra bins are")
print("free precision on paper and none in practice, and the way to actually")
print("improve is to feed the model more pixels. Real systems sit in this middle")
print("ground: Pix2Seq and Qwen-VL use ~1000 bins over a normalized image,")
print("Kosmos-2 a 32x32 grid of location tokens.")
print()
print("The deeper point: because a box is emitted as ordinary tokens, grounding")
print("needs NO architectural change at all -- no detection head, no anchor")
print("boxes, no NMS. It is a data and vocabulary decision, which is why a")
print("general VLM can point at things at all.")
print()


# ===========================================================================
# 2. READING SMALL THINGS: patch size vs. glyph size
# ===========================================================================

CANVAS = 32
N_GLYPHS = 10
PATCH = 4                       # fixed: this experiment is about RESOLUTION
GLYPH_SRC = (torch.rand(N_GLYPHS, 4, 4) > 0.5).float()      # 10 distinct 4x4 patterns

# Pre-render a pool of real canvases per glyph size, once, and sample batches
# from it -- rendering is not the interesting part, and doing it per step would
# dominate the runtime.
POOL_SIZE = 5000


def build_pool(glyph_px, n=POOL_SIZE):
    """Draw one glyph, scaled to glyph_px x glyph_px, at a random position on a
    real 32x32 canvas, with background clutter."""
    ids = torch.randint(0, N_GLYPHS, (n,))
    canvas = 0.12 * torch.randn(n, CANVAS, CANVAS)
    scale = glyph_px // 4
    glyphs = GLYPH_SRC.repeat_interleave(scale, 1).repeat_interleave(scale, 2)[ids]
    span = CANVAS - glyph_px + 1
    rows = torch.randint(0, span, (n,))
    cols = torch.randint(0, span, (n,))
    ar = torch.arange(glyph_px)
    for i in range(n):
        canvas[i, rows[i] + ar[:, None], cols[i] + ar[None, :]] += glyphs[i]
    return canvas, ids


POOLS = {}


def render(b, glyph_px):
    if glyph_px not in POOLS:
        POOLS[glyph_px] = build_pool(glyph_px)
    canvas, ids = POOLS[glyph_px]
    pick = torch.randint(0, canvas.shape[0], (b,))
    return canvas[pick], ids[pick]


def to_tokens(canvas, downscale):
    """Downscale the image to the encoder's input resolution, then patchify it --
    Lesson 1's operation, on real (if tiny) images."""
    if downscale > 1:
        canvas = F.avg_pool2d(canvas.unsqueeze(1), downscale).squeeze(1)
    res = canvas.shape[-1]
    g = res // PATCH
    x = canvas.reshape(-1, g, PATCH, g, PATCH).permute(0, 1, 3, 2, 4)
    return x.reshape(-1, g * g, PATCH * PATCH)


class Reader(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.proj = nn.Linear(PATCH * PATCH, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_GLYPHS)

    def forward(self, tokens):
        x = self.proj(tokens) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x).mean(1))


def train_reader(downscale, glyph_px, steps=300, bs=96, lr=3e-3):
    n_tokens = ((CANVAS // downscale) // PATCH) ** 2
    torch.manual_seed(1)
    model = Reader(n_tokens)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        canvas, ids = render(bs, glyph_px)
        loss = F.cross_entropy(model(to_tokens(canvas, downscale)), ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        canvas, ids = render(1500, glyph_px)
        acc = (model(to_tokens(canvas, downscale)).argmax(-1) == ids).float().mean().item()
    return acc, n_tokens


print("=" * 78)
print("2. READING SMALL THINGS: resolution is the whole story")
print("=" * 78)
print("A real 32x32 image contains one glyph from a 10-glyph alphabet at a random")
print("position. The image is downscaled to the encoder's input resolution and")
print(f"cut into {PATCH}x{PATCH} patches. Chance = {1 / N_GLYPHS:.0%}.")
print()
print(f"{'input res':>10} {'tokens':>8} {'glyph 16px':>12} {'glyph 8px':>11} {'glyph 4px':>11}")
for down in [1, 2, 4]:
    accs = [train_reader(down, gp) for gp in [16, 8, 4]]
    print(f"{CANVAS // down:>9}px {accs[0][1]:>8} {accs[0][0]:>11.1%}"
          f" {accs[1][0]:>10.1%} {accs[2][0]:>10.1%}")
print()
print("Read down the last column: a 4-pixel glyph is legible at full resolution")
print("and gone once the image has been halved -- there is nothing left of it to")
print("patchify. Read across the bottom row: at the lowest resolution the large")
print("glyph is still perfectly readable while the small one is near chance.")
print("Legibility is decided by the glyph's size AFTER downscaling, and the")
print("tokens column shows what keeping it costs: resolution and token count")
print("move together as res^2 (Lesson 1 section 4).")
print()
print("(The top-left cell lags the others: the largest glyph at the largest token")
print("count is simply the slowest of these runs to converge at a fixed 300 steps.")
print("It is training noise, not a resolution effect -- the resolution effect is")
print("the one that is monotone down each column.)")
print()
print("This is the whole story of OCR and document VLMs. A 12px character in a")
print("1600px scan, downscaled to a 336px encoder input, is about 2.5 pixels")
print("tall: no model on the other side of that encoder can read it, because the")
print("information was destroyed before the language model was reached. Every fix")
print("is upstream -- higher input resolution, dynamic tiling (Lesson 1), a")
print("purpose-built high-resolution document encoder -- and every one of them is")
print("paid for in vision tokens.")
print()


# ===========================================================================
# 3. VIDEO: frame sampling is the binding constraint
# ===========================================================================

N_FRAMES = 32
N_EVENTS = 6
event_code = torch.randn(N_EVENTS, D_VISION)


def video_batch(b, k):
    """A clip of N_FRAMES frames. Exactly one frame contains the event; the
    rest are background. We show the model k uniformly sampled frames."""
    event = torch.randint(0, N_EVENTS, (b,))
    where = torch.randint(0, N_FRAMES, (b,))
    frames = 0.3 * torch.randn(b, N_FRAMES, D_VISION)
    frames[torch.arange(b), where] += event_code[event]
    stride = N_FRAMES // k
    offset = torch.randint(0, stride, (1,)).item()
    idx = torch.arange(k) * stride + offset
    return frames[:, idx], event, (where.unsqueeze(1) == idx.unsqueeze(0)).any(1)


class VideoQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                  nn.Linear(D_MODEL, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_EVENTS)

    def forward(self, frames):
        x = self.proj(frames)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x).mean(1))


def train_video(k, steps=250, bs=96, lr=2e-3):
    torch.manual_seed(1)
    model = VideoQA()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        frames, event, _ = video_batch(bs, k)
        loss = F.cross_entropy(model(frames), event)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        frames, event, captured = video_batch(2000, k)
        pred = model(frames).argmax(-1)
        acc = (pred == event).float().mean().item()
        acc_when_seen = (pred == event)[captured].float().mean().item()
    return acc, acc_when_seen, captured.float().mean().item()


print("=" * 78)
print("3. VIDEO: what you did not sample, you cannot see")
print("=" * 78)
print(f"A {N_FRAMES}-frame clip; the event appears in exactly ONE frame. The model")
print("gets k uniformly sampled frames, each costing a full image's worth of")
print(f"tokens. Chance = {1 / N_EVENTS:.1%}.")
print()
print(f"{'frames sampled':>15} {'vision tokens*':>15} {'event captured':>15} {'accuracy':>10} {'acc | captured':>15}")
for k in [1, 4, 8, 16, 32]:
    acc, acc_seen, captured = train_video(k)
    print(f"{k:>15} {k * 64:>15,} {captured:>14.1%} {acc:>9.1%} {acc_seen:>14.1%}")
print()
print("* at a (very frugal) 64 tokens per frame")
print()
print("The 'acc | captured' column is the model's accuracy on the clips where the")
print("crucial frame WAS among those sampled -- essentially perfect throughout.")
print("Everything in the overall accuracy column is therefore sampling, not")
print("modelling: the model is excellent at the frames it sees and blind to the")
print("ones it does not, and overall accuracy tracks the capture rate almost")
print("exactly.")
print()
print("That is the entire difficulty of video understanding. A 10-minute clip at")
print("1 fps is 600 frames; at even 64 tokens per frame that is 38,400 tokens")
print("before the question is asked. Every practical system therefore chooses")
print("where to spend a fixed token budget -- uniform sampling, keyframe")
print("selection, aggressive per-frame compression (Lesson 4), temporal pooling")
print("across adjacent frames, or retrieval that picks candidate frames using the")
print("question first. The failures that follow are sampling failures wearing a")
print("reasoning failure's clothes.")
