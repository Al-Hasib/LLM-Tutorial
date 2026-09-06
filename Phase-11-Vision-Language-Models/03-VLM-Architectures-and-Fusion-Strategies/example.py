"""
VLM fusion strategies -- three ways to get vision into a language model, all
implemented from scratch and trained on the same toy grounding task.

No downloads, no pretrained weights. CPU, ~1 minute.

The task (a deliberately small but genuinely non-trivial grounding problem):
  A "scene" is NUM_OBJECTS objects. Each object has a shape and a color, and
  is handed to the model as one vision token (a noisy vector encoding both).
  The question names one shape -- "what color is the <shape>?" -- and the
  model must output that object's color. A model that ignores the image is
  stuck at chance, and a model that pools the image into one vector can only
  narrow the guess to "colors present somewhere in the scene": solving the
  task requires attending to the ONE token holding the queried shape and
  reading off the color bound to it. That is grounding, in miniature.

The three architectures compared:
  A. PREFIX / PROJECTOR fusion (LLaVA, most open VLMs)
       vision tokens are projected into the LLM's embedding space and
       CONCATENATED in front of the text tokens; self-attention does the rest.
  B. CROSS-ATTENTION fusion (Flamingo, Llama-3-V, Idefics-style)
       the text stream keeps its own length; extra gated cross-attention
       layers let text tokens read from the vision tokens.
  C. EARLY / NATIVE fusion (Chameleon, Fuyu-style)
       no separate pretrained tower interface -- raw patch vectors go
       straight into the shared Transformer as tokens from step one.

Measured for each: held-out accuracy, parameter count, text sequence length,
attention cost, and -- for the frozen-LLM setting -- how much of the language
model has to be touched at all.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

NUM_SHAPES = 8
NUM_COLORS = 6
NUM_OBJECTS = 6           # vision tokens per scene
D_VISION = 32             # "vision tower" output width
D_MODEL = 64              # the LLM's hidden width
N_TEXT_TOKENS = 4         # the question, as text tokens


# ---------------------------------------------------------------------------
# 0. Synthetic scenes and questions
# ---------------------------------------------------------------------------

# Fixed random "appearance" codes: how a shape and a color look in raw
# vision-feature space. The two are ADDED, so a single token carries both and
# the model must disentangle them.
shape_code = torch.randn(NUM_SHAPES, D_VISION)
color_code = torch.randn(NUM_COLORS, D_VISION)


def make_batch(batch_size):
    """Returns vision tokens, question token ids, and the answer color id."""
    # Each scene has NUM_OBJECTS distinct shapes, so the question is unambiguous.
    shapes = torch.stack([torch.randperm(NUM_SHAPES)[:NUM_OBJECTS] for _ in range(batch_size)])
    colors = torch.randint(0, NUM_COLORS, (batch_size, NUM_OBJECTS))
    vision = shape_code[shapes] + color_code[colors]                    # (B, N_obj, D_VISION)
    vision = vision + 0.25 * torch.randn_like(vision)

    # Question: which of the objects present is being asked about?
    target_slot = torch.randint(0, NUM_OBJECTS, (batch_size,))
    queried_shape = shapes[torch.arange(batch_size), target_slot]       # (B,)
    answer = colors[torch.arange(batch_size), target_slot]              # (B,)

    # Question token ids: [WHAT] [COLOR] [IS] [<shape>]  -- the shape token id
    # is offset past the 3 fixed vocabulary words.
    fixed = torch.tensor([0, 1, 2]).expand(batch_size, 3)
    question = torch.cat([fixed, (queried_shape + 3).unsqueeze(1)], dim=1)   # (B, 4)
    return vision, question, answer


TEXT_VOCAB = 3 + NUM_SHAPES


# ---------------------------------------------------------------------------
# 1. Shared building blocks (a minimal Transformer, as in Phase 02)
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo's insertion: cross-attention from text into vision, wrapped in
    a TANH GATE initialized at zero so the block is a no-op at initialization.

    That detail is the whole reason a frozen pretrained LLM survives having
    these layers inserted: at step 0 the modified model computes bit-for-bit
    what the original model computed, and the gate opens only as far as
    training makes worthwhile.
    """

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))          # <- zero-init gate
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, text, vision):
        a = self.attn(self.ln_q(text), self.ln_kv(vision), self.ln_kv(vision),
                      need_weights=False)[0]
        text = text + torch.tanh(self.gate) * a
        return text + torch.tanh(self.ff_gate) * self.ff(self.ln_ff(text))


class TextBackbone(nn.Module):
    """Stands in for 'the pretrained LLM'. Same body in all three models."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.tok_embed = nn.Embedding(TEXT_VOCAB, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([SelfAttentionBlock(D_MODEL) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, NUM_COLORS)


# ---------------------------------------------------------------------------
# 2. The three architectures
# ---------------------------------------------------------------------------

class PrefixVLM(nn.Module):
    """A. LLaVA-style: project vision tokens into the LLM's space, prepend."""

    def __init__(self):
        super().__init__()
        self.lm = TextBackbone()
        self.projector = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))

    def forward(self, vision, question):
        vis = self.projector(vision)                                   # (B, N_obj, D_MODEL)
        txt = self.lm.tok_embed(question)                              # (B, N_text, D_MODEL)
        x = torch.cat([vis, txt], dim=1)                               # ONE sequence
        x = x + self.lm.pos_embed[:, : x.shape[1]]
        for blk in self.lm.blocks:
            x = blk(x)
        return self.lm.head(self.lm.ln_f(x[:, -1]))                    # read the last token

    def seq_len(self):
        return NUM_OBJECTS + N_TEXT_TOKENS


class CrossAttnVLM(nn.Module):
    """B. Flamingo-style: text stream keeps its length; gated cross-attention
    layers are interleaved so text tokens can read from the vision tokens."""

    def __init__(self):
        super().__init__()
        self.lm = TextBackbone()
        self.vision_proj = nn.Linear(D_VISION, D_MODEL)
        self.xattn = nn.ModuleList([GatedCrossAttentionBlock(D_MODEL)
                                    for _ in range(len(self.lm.blocks))])

    def forward(self, vision, question):
        vis = self.vision_proj(vision)
        x = self.lm.tok_embed(question) + self.lm.pos_embed[:, :question.shape[1]]
        for blk, xa in zip(self.lm.blocks, self.xattn):
            x = xa(x, vis)                                             # read the image
            x = blk(x)                                                 # then think in text
        return self.lm.head(self.lm.ln_f(x[:, -1]))

    def seq_len(self):
        return N_TEXT_TOKENS


class EarlyFusionVLM(nn.Module):
    """C. Native/early fusion: one Transformer, raw patch vectors as tokens,
    no separately pretrained tower interface and no frozen anything."""

    def __init__(self):
        super().__init__()
        self.lm = TextBackbone()
        self.patch_in = nn.Linear(D_VISION, D_MODEL)     # a single linear "tokenizer"
        self.modality_embed = nn.Parameter(torch.randn(2, D_MODEL) * 0.02)

    def forward(self, vision, question):
        vis = self.patch_in(vision) + self.modality_embed[0]
        txt = self.lm.tok_embed(question) + self.modality_embed[1]
        x = torch.cat([vis, txt], dim=1)
        x = x + self.lm.pos_embed[:, : x.shape[1]]
        for blk in self.lm.blocks:
            x = blk(x)
        return self.lm.head(self.lm.ln_f(x[:, -1]))

    def seq_len(self):
        return NUM_OBJECTS + N_TEXT_TOKENS


class BlindBaseline(nn.Module):
    """Text only -- never sees the image. The chance-level control."""

    def __init__(self):
        super().__init__()
        self.lm = TextBackbone()

    def forward(self, vision, question):
        x = self.lm.tok_embed(question) + self.lm.pos_embed[:, :question.shape[1]]
        for blk in self.lm.blocks:
            x = blk(x)
        return self.lm.head(self.lm.ln_f(x[:, -1]))

    def seq_len(self):
        return N_TEXT_TOKENS


class PooledVLM(nn.Module):
    """A control for Lesson 1's pooling point: the image is squashed to ONE
    vector before the LLM sees it, exactly like a CLIP embedding."""

    def __init__(self):
        super().__init__()
        self.lm = TextBackbone()
        self.projector = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))

    def forward(self, vision, question):
        vis = self.projector(vision).mean(dim=1, keepdim=True)         # (B, 1, D_MODEL)
        txt = self.lm.tok_embed(question)
        x = torch.cat([vis, txt], dim=1)
        x = x + self.lm.pos_embed[:, : x.shape[1]]
        for blk in self.lm.blocks:
            x = blk(x)
        return self.lm.head(self.lm.ln_f(x[:, -1]))

    def seq_len(self):
        return 1 + N_TEXT_TOKENS


# ---------------------------------------------------------------------------
# 3. Training and evaluation
# ---------------------------------------------------------------------------

def train_and_eval(model_cls, steps=700, batch_size=64, lr=2e-3, freeze_lm=False):
    torch.manual_seed(1)
    model = model_cls()
    if freeze_lm:
        for p in model.lm.parameters():
            p.requires_grad = False
        # The head still has to learn to emit colors; in a real VLM the LLM's
        # existing output head already knows the color words.
        for p in model.lm.head.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    for _ in range(steps):
        vision, question, answer = make_batch(batch_size)
        loss = F.cross_entropy(model(vision, question), answer)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        vision, question, answer = make_batch(2000)
        acc = (model(vision, question).argmax(1) == answer).float().mean().item()
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    return acc, n_total, n_train, model


print("=" * 78)
print("THE TASK: 6 objects, each one vision token carrying shape+color.")
print("Question: 'what color is the <shape>?'  Chance accuracy = 1/6 = 16.7%.")
print("=" * 78)
print()

print("=" * 78)
print("1. ALL LAYERS TRAINABLE")
print("=" * 78)
print(f"{'architecture':<34} {'accuracy':>9} {'params':>10} {'text seq len':>13} {'attn cells':>11}")

configs = [
    ("blind (text only, no image)", BlindBaseline),
    ("pooled image -> 1 token", PooledVLM),
    ("A. prefix / projector (LLaVA)", PrefixVLM),
    ("B. cross-attention (Flamingo)", CrossAttnVLM),
    ("C. early / native fusion", EarlyFusionVLM),
]

trained = {}
for name, cls in configs:
    acc, n_total, n_train, model = train_and_eval(cls)
    trained[name] = (acc, model)
    L = model.seq_len()
    print(f"{name:<34} {acc:>8.1%} {n_total:>10,} {L:>13} {L * L:>11}")

print()
print("The blind model is pinned at chance -- with no image there is nothing to")
print("read the colour off. The pooled model lands well above chance but far")
print("below solving the task: a single averaged vector still carries WHICH")
print("colours are in the scene, so guessing among them beats guessing among")
print("all six, but it has lost which colour is BOUND to which shape. That")
print("binding is what per-token vision features preserve, and it is Lesson 1's")
print("pooling argument restated as task accuracy. All three real fusion")
print("strategies solve the task outright.")
print()
print("Note the cost asymmetry in the last two columns: prefix fusion makes the")
print("LLM's sequence longer (here 6 image tokens + 4 text tokens), and self-")
print("attention is quadratic in that length. Cross-attention leaves the text")
print("stream at its original length and pays for the image separately, in")
print("cross-attention layers whose cost is (text_len x vision_len), i.e. LINEAR")
print("in the number of image tokens. With 576 or 3,136 real vision tokens that")
print("difference is the whole ballgame -- see Lesson 10.")
print()


# ---------------------------------------------------------------------------
# 4. The frozen-LLM setting: what each strategy costs you
# ---------------------------------------------------------------------------

print("=" * 78)
print("2. WITH THE LANGUAGE MODEL FROZEN (the realistic starting point)")
print("=" * 78)
print("A real VLM starts from an LLM that already works. Freezing it protects")
print("its text ability, and makes 'how much NEW machinery does this fusion")
print("strategy need?' the deciding question.")
print()
print(f"{'architecture':<34} {'accuracy':>9} {'trainable':>11} {'% of model':>11}")
for name, cls in [("A. prefix / projector (LLaVA)", PrefixVLM),
                  ("B. cross-attention (Flamingo)", CrossAttnVLM),
                  ("C. early / native fusion", EarlyFusionVLM)]:
    acc, n_total, n_train, _ = train_and_eval(cls, freeze_lm=True)
    print(f"{name:<34} {acc:>8.1%} {n_train:>11,} {n_train / n_total:>10.1%}")

print()
print("Prefix fusion trains a tiny projector and nothing else -- that is why it")
print("is the cheapest way to build a VLM and why LLaVA-style recipes dominate")
print("open-source work. Cross-attention adds real parameters (a whole gated")
print("block per layer, ~50% of the model here) but touches no existing weight,")
print("which is why Flamingo could bolt vision onto a frozen 70B LM.")
print()
print("Early fusion is in the table only for completeness: this toy task is easy")
print("enough that a frozen backbone still solves it, but the frozen column does")
print("not really apply to it. Its premise is that the text and image weights")
print("were never separate in the first place -- there is no pretrained language")
print("model sitting there to freeze, and buying its higher ceiling means paying")
print("for a full multimodal pretraining run (Lesson 5).")
print()


# ---------------------------------------------------------------------------
# 5. The zero-init gate: why a frozen LLM survives cross-attention insertion
# ---------------------------------------------------------------------------

print("=" * 78)
print("3. THE ZERO-INITIALIZED GATE (Flamingo's key trick)")
print("=" * 78)

torch.manual_seed(1)
xvlm = CrossAttnVLM()
blind = BlindBaseline()
blind.lm.load_state_dict(xvlm.lm.state_dict())      # identical language model

vision, question, _ = make_batch(8)
with torch.no_grad():
    out_with_xattn = xvlm(vision, question)
    out_text_only = blind(vision, question)
print(f"max |cross-attn model - text-only model| at init : "
      f"{(out_with_xattn - out_text_only).abs().max().item():.2e}")
print("Exactly zero: tanh(0) = 0, so every inserted block is an identity at")
print("initialization and the frozen LLM's behaviour is preserved bit-for-bit.")
print("Training then opens the gates only as far as the vision signal earns.")

_, gated_model = trained["B. cross-attention (Flamingo)"]
gates = [torch.tanh(b.gate).item() for b in gated_model.xattn]
print(f"gate values after training, layer by layer      : "
      f"{', '.join(f'{g:+.3f}' for g in gates)}")
print("Non-zero after training = the model chose to let the image in, and how")
print("much it let in per layer is a directly readable number.")
