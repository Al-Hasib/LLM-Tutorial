"""
Training a VLM -- the staged pipeline, catastrophic forgetting, and loss
masking, all measured on a real (tiny) autoregressive model.

No downloads, no pretrained weights. CPU, a few minutes.

The setup mirrors the real problem as closely as a toy can:

  * A small causal Transformer LM is PRETRAINED on a text-only task: a
    sequence of key/value pairs followed by a query key, answer = that key's
    value. This stands in for "a language model that already works", and it
    gives us a text ability we can MEASURE before and after adding vision.

  * Vision is then added: the same answer, but the key/value pairs arrive as
    vision tokens (one per object) instead of text tokens. This is the VQA
    side of the task.

  * Both tasks share the model, the vocabulary and the LM head, so anything
    that damages the language model shows up immediately as lost text
    accuracy -- exactly how catastrophic forgetting is detected in practice.

Experiments:
  1. Six training schedules (projector-only, staged, joint, replay-mixed,
     high-LR) scored on BOTH tasks. The forgetting column is the point.
  2. Loss masking: LM loss over every token vs. only over the answer.
  3. Freezing the vision tower vs. training it.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# --- vocabulary -------------------------------------------------------------
N_VALUES = 8            # answer tokens: ids 0..7
N_KEYS = 12             # text keys:     ids 8..19
N_SHAPES = 12           # image shapes:  ids 20..31
Q_TEXT, Q_IMG = 32, 33  # "answer the question" markers
VOCAB = 34

KEY0, SHAPE0 = N_VALUES, N_VALUES + N_KEYS

N_PAIRS = 4             # key/value pairs per text sample
N_OBJECTS = 4           # objects (vision tokens) per image
N_FILLER = 6            # "instruction phrasing" tokens in the image prompt --
                        # random, unpredictable, and therefore untrainable
D_VISION = 32
D_TOWER_OUT = 4         # the tower's output width: a hard information bottleneck
D_MODEL = 64

shape_code = torch.randn(N_SHAPES, D_VISION)
value_code = torch.randn(N_VALUES, D_VISION)


# ---------------------------------------------------------------------------
# 0. Data: the same lookup task, once in text, once in pixels
# ---------------------------------------------------------------------------

def text_batch(b):
    """[k1 v1 k2 v2 ... Q_TEXT kq a] -- pure text, no image."""
    keys = torch.stack([torch.randperm(N_KEYS)[:N_PAIRS] for _ in range(b)]) + KEY0
    vals = torch.randint(0, N_VALUES, (b, N_PAIRS))
    slot = torch.randint(0, N_PAIRS, (b,))
    qk = keys[torch.arange(b), slot]
    ans = vals[torch.arange(b), slot]
    seq = torch.stack([torch.stack([keys[:, i], vals[:, i]]) for i in range(N_PAIRS)])
    seq = seq.permute(2, 0, 1).reshape(b, 2 * N_PAIRS)          # interleave k,v,k,v...
    seq = torch.cat([seq,
                     torch.full((b, 1), Q_TEXT),
                     qk.unsqueeze(1),
                     ans.unsqueeze(1)], dim=1)
    return seq, ans


def image_batch(b):
    """vision tokens (one per object) + [Q_IMG shape_q a]."""
    shapes = torch.stack([torch.randperm(N_SHAPES)[:N_OBJECTS] for _ in range(b)])
    vals = torch.randint(0, N_VALUES, (b, N_OBJECTS))
    vision = shape_code[shapes] + value_code[vals]
    vision = vision + 0.2 * torch.randn_like(vision)
    slot = torch.randint(0, N_OBJECTS, (b,))
    ans = vals[torch.arange(b), slot]
    qs = shapes[torch.arange(b), slot] + SHAPE0
    # A randomly phrased instruction ("please describe...", "tell me...") --
    # here just random tokens. It is context, and it is not predictable.
    filler = torch.randint(KEY0, KEY0 + N_KEYS, (b, N_FILLER))
    seq = torch.cat([torch.full((b, 1), Q_IMG), filler, qs.unsqueeze(1), ans.unsqueeze(1)], dim=1)
    return vision, seq, ans


# ---------------------------------------------------------------------------
# 1. A minimal causal Transformer + a projector
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


def causal_mask(n):
    return torch.triu(torch.full((n, n), float("-inf")), diagonal=1)


class VLM(nn.Module):
    """One model, two input paths. `lm` is the part a real pipeline would be
    starting from (pretrained); `projector` and `vision_tower` are new."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)
        # "new" modules, absent from the text-only pretraining.
        # The tower has a narrow output (D_TOWER_OUT << D_VISION): like a real
        # vision encoder, it cannot forward everything in the image, and WHICH
        # information it keeps was decided by its own pretraining -- here, by
        # its frozen random initialization.
        self.vision_tower = nn.Linear(D_VISION, D_TOWER_OUT, bias=False)
        self.projector = nn.Sequential(nn.Linear(D_TOWER_OUT, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))

    def lm_params(self):
        mods = [self.tok_embed, *self.blocks, self.ln_f, self.head]
        ps = [p for m in mods for p in m.parameters()] + [self.pos_embed]
        return ps

    def forward(self, seq, vision=None):
        x = self.tok_embed(seq)
        if vision is not None:
            vis = self.projector(self.vision_tower(vision))
            x = torch.cat([vis, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        mask = causal_mask(x.shape[1])
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))                      # (B, L, VOCAB)


def lm_loss(model, seq, vision=None, answer_only=True):
    """Next-token cross-entropy. `answer_only` masks out every position except
    the answer -- i.e. the prompt is context, not a training target (SFT)."""
    logits = model(seq, vision)
    n_vis = 0 if vision is None else vision.shape[1]
    # Predict token t+1 from position t. Text positions start after the prefix.
    pred = logits[:, n_vis: -1]                             # (B, L_text-1, V)
    target = seq[:, 1:]                                     # (B, L_text-1)
    if answer_only:
        return F.cross_entropy(pred[:, -1], target[:, -1])
    return F.cross_entropy(pred.reshape(-1, VOCAB), target.reshape(-1))


@torch.no_grad()
def accuracy(model, kind, n=1500):
    if kind == "text":
        seq, ans = text_batch(n)
        logits = model(seq)
        pred = logits[:, -2].argmax(-1)                     # position before the answer
    else:
        vision, seq, ans = image_batch(n)
        logits = model(seq, vision)
        pred = logits[:, -2].argmax(-1)
    return (pred == ans).float().mean().item()


# ---------------------------------------------------------------------------
# 2. Stage 0: pretrain the language model on text only
# ---------------------------------------------------------------------------

def pretrain_lm(steps=1200, bs=64, lr=2e-3):
    torch.manual_seed(1)
    model = VLM()
    opt = torch.optim.Adam(model.lm_params(), lr=lr)
    for _ in range(steps):
        seq, _ = text_batch(bs)
        loss = lm_loss(model, seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


print("=" * 78)
print("STAGE 0: pretrain the LM on the TEXT-ONLY lookup task")
print("=" * 78)
base = pretrain_lm()
base_text_acc = accuracy(base, "text")
print(f"text-task accuracy after text-only pretraining : {base_text_acc:.1%}")
print(f"image-task accuracy (vision is untrained noise): {accuracy(base, 'image'):.1%}")
print(f"chance on both tasks                           : {1 / N_VALUES:.1%}")
print()
print("This is our 'pretrained LLM'. Everything below starts from these exact")
print("weights, so every difference in the tables is caused by the training")
print("schedule and nothing else.")
print()


# ---------------------------------------------------------------------------
# 3. Multimodal training schedules
# ---------------------------------------------------------------------------

import copy


def train_schedule(name, stage1_steps, stage2_steps, unfreeze_lm, lr_stage2,
                   text_replay=0.0, bs=64, lr_stage1=2e-3, freeze_tower=True):
    torch.manual_seed(2)
    model = copy.deepcopy(base)

    # --- Stage 1: alignment. Only the connector trains; the LM cannot move.
    if stage1_steps:
        params = list(model.projector.parameters())
        if not freeze_tower:
            params += list(model.vision_tower.parameters())
        opt = torch.optim.Adam(params, lr=lr_stage1)
        for _ in range(stage1_steps):
            vision, seq, _ = image_batch(bs)
            loss = lm_loss(model, seq, vision)
            opt.zero_grad()
            loss.backward()
            opt.step()
    acc_after_s1 = accuracy(model, "image") if stage1_steps else float("nan")

    # --- Stage 2: instruction tuning. Optionally unfreeze the LM.
    if stage2_steps:
        params = list(model.projector.parameters())
        if not freeze_tower:
            params += list(model.vision_tower.parameters())
        if unfreeze_lm:
            params += model.lm_params()
        opt = torch.optim.Adam(params, lr=lr_stage2)
        for _ in range(stage2_steps):
            vision, seq, _ = image_batch(bs)
            loss = lm_loss(model, seq, vision)
            if text_replay > 0:                     # mix text-only data back in
                tseq, _ = text_batch(max(1, int(bs * text_replay)))
                loss = loss + lm_loss(model, tseq)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return {
        "name": name,
        "img": accuracy(model, "image"),
        "txt": accuracy(model, "text"),
        "s1": acc_after_s1,
    }


print("=" * 78)
print("1. SIX SCHEDULES, SCORED ON BOTH TASKS")
print("=" * 78)
print("'text' is the ORIGINAL text-only ability, re-measured after multimodal")
print(f"training. It started at {base_text_acc:.1%}. Any drop is forgetting.")
print()

schedules = [
    # name,                             s1,   s2,  unfreeze, lr_s2, replay
    ("stage 1 only (projector, frozen LM)", 700, 0, False, 0.0, 0.0),
    ("stage 1 -> stage 2, LM frozen", 700, 700, False, 1e-3, 0.0),
    ("stage 1 -> stage 2, LM unfrozen", 700, 700, True, 2e-4, 0.0),
    ("stage 1 -> stage 2, LM unfrozen, HIGH lr", 700, 700, True, 2e-3, 0.0),
    ("stage 1 -> stage 2 + 50% text replay", 700, 700, True, 2e-4, 0.5),
    ("NO stage 1: unfreeze everything at once", 0, 1400, True, 2e-3, 0.0),
]

print(f"{'schedule':<44} {'image':>7} {'text':>7} {'forgetting':>11}")
rows = []
for name, s1, s2, unf, lr2, rep in schedules:
    r = train_schedule(name, s1, s2, unf, lr2, rep)
    rows.append(r)
    forget = base_text_acc - r["txt"]
    print(f"{name:<44} {r['img']:>6.1%} {r['txt']:>6.1%} {forget:>10.1%}")

print()
print("What the rows say:")
print()
print("* Freezing the LM makes forgetting IMPOSSIBLE by construction -- the text")
print("  weights never move, so the text score cannot change. That is the whole")
print("  appeal of stage 1: it is a safe way to teach the connector where to put")
print("  visual features, using cheap caption-style data. Note how far short of")
print("  the task it lands on its own, though: a projector alone has to make")
print("  visual features legible to an LM that has never been asked to read")
print("  them, and that ceiling is why stage 2 exists.")
print()
print("* Unfreezing the LM at a SMALL learning rate buys image accuracy at a")
print("  modest cost in text ability. Unfreezing at the same learning rate used")
print("  for the connector wrecks the text task: the multimodal data contains no")
print("  text-only examples, so nothing is holding those weights in place.")
print()
print("* Text replay -- mixing the original text-only data back into the")
print("  multimodal stage -- is the standard fix, and it works here for the same")
print("  reason it works in production: the objective now includes the ability")
print("  you are trying not to lose.")
print()
print("* Skipping stage 1 and unfreezing everything at once trains a randomly")
print("  initialized connector and a working LM with one learning rate. The")
print("  connector's early output is noise, and the LM adapts to noise before")
print("  the connector becomes useful -- which is exactly the instability the")
print("  two-stage recipe was invented to avoid.")
print()


# ---------------------------------------------------------------------------
# 4. Loss masking: prompt tokens as targets vs. as context
# ---------------------------------------------------------------------------

def train_masking(answer_only, steps=500, bs=64, lr=1e-3):
    torch.manual_seed(2)
    model = copy.deepcopy(base)
    opt = torch.optim.Adam(list(model.projector.parameters()) + model.lm_params(), lr=lr)
    for _ in range(steps):
        vision, seq, _ = image_batch(bs)
        loss = lm_loss(model, seq, vision, answer_only=answer_only)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return accuracy(model, "image"), accuracy(model, "text")


print("=" * 78)
print("2. LOSS MASKING: is the prompt a training target or just context?")
print("=" * 78)
img_masked, txt_masked = train_masking(answer_only=True)
img_all, txt_all = train_masking(answer_only=False)
print(f"{'loss computed over':<42} {'image acc':>10} {'text acc':>9}")
print(f"{'the answer only (standard SFT)':<42} {img_masked:>9.1%} {txt_masked:>8.1%}")
print(f"{'every token in the sequence':<42} {img_all:>9.1%} {txt_all:>8.1%}")
print()
print("The prompt here contains a randomly chosen query shape, which is")
print("fundamentally unpredictable -- no amount of capacity can lower that part")
print("of the loss. Training on it spends gradient on noise and dilutes the")
print("signal from the one token that matters. Masking the prompt (Phase 05")
print("Lesson 4's recipe, unchanged for images) is not a small detail; on a")
print("real VLM whose prompts include hundreds of vision tokens, the answer is a")
print("tiny fraction of the sequence and the dilution is proportionally worse.")
print()


# ---------------------------------------------------------------------------
# 5. Should the vision tower train too?
# ---------------------------------------------------------------------------

print("=" * 78)
print("3. FREEZING THE VISION TOWER vs. TRAINING IT")
print("=" * 78)
frozen = train_schedule("frozen tower", 700, 700, True, 2e-4, 0.5, freeze_tower=True)
unfrozen = train_schedule("trained tower", 700, 700, True, 2e-4, 0.5, freeze_tower=False)
print(f"{'vision tower':<30} {'image':>7} {'text':>7}")
print(f"{'frozen':<30} {frozen['img']:>6.1%} {frozen['txt']:>6.1%}")
print(f"{'trained (unfrozen)':<30} {unfrozen['img']:>6.1%} {unfrozen['txt']:>6.1%}")
print()
print("Unfreezing the tower gives the pipeline a way to fix what the tower's own")
print("pretraining objective threw away (Lesson 2 section 4) -- but on a real")
print("model it also risks destroying features that took hundreds of GPU-years")
print("of contrastive training to acquire, on a fine-tuning set orders of")
print("magnitude smaller. Current practice: freeze during alignment, then")
print("unfreeze late at a much lower learning rate than the rest of the model,")
print("if at all.")
