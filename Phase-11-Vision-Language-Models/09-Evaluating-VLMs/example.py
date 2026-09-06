"""
Evaluating VLMs -- a small multiple-choice benchmark harness that reproduces
the three pathologies that make published VLM numbers hard to trust.

No downloads, no pretrained weights. CPU, a few minutes.

The benchmark: a scene of 4 objects (shape + colour), one question, four
lettered options, one correct. Two question families, mixed 50/50:

  VISION-NECESSARY -- "what colour is the <shape>?", with four colours as
      options. The image is the only place the answer exists.
  SHORTCUT -- the same question, but three of the four options are shapes
      rather than colours, so exactly one option is even the right KIND of
      answer. A blind model gets it right every time. Real benchmarks are
      full of these: implausible distractors, only one grammatically
      coherent option, or a question whose answer is the world's default.

Three experiments:
  1. The blind baseline -- how much of a benchmark score needs no image.
  2. Option-position bias, and circular evaluation as the fix.
  3. Contamination -- what a few hundred leaked test items do to a score.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

N_SHAPES = 10
N_COLORS = 6
N_OBJECTS = 4
N_OPTIONS = 4
D_VISION = 24
D_MODEL = 64

COLOR0 = 0
SHAPE0 = N_COLORS
VOCAB = SHAPE0 + N_SHAPES

shape_code = torch.randn(N_SHAPES, D_VISION)
color_code = torch.randn(N_COLORS, D_VISION)


# ---------------------------------------------------------------------------
# 0. Building benchmark items (fixed, so they can be leaked and memorized)
# ---------------------------------------------------------------------------

def make_items(n, shortcut_fraction=0.5, answer_pos=None):
    """Returns a dict of tensors describing n benchmark items.

    `answer_pos`: None = uniform random position for the correct option;
    an int = always that position; a tensor = per-item positions.
    """
    shapes = torch.stack([torch.randperm(N_SHAPES)[:N_OBJECTS] for _ in range(n)])
    colors = torch.randint(0, N_COLORS, (n, N_OBJECTS))
    vision = shape_code[shapes] + color_code[colors]
    # Partial, noisy evidence -- see Lesson 7. Without this the toy task is
    # perfectly solvable, every model scores 100%, and none of the benchmark
    # pathologies below can show up. A model that is never uncertain never
    # falls back on a prior.
    keep = (torch.rand(n, N_OBJECTS, 1) < 0.7).float()
    vision = vision * keep + 0.45 * torch.randn_like(vision)

    slot = torch.randint(0, N_OBJECTS, (n,))
    idx = torch.arange(n)
    q_shape = shapes[idx, slot]
    answer_color = colors[idx, slot]
    is_shortcut = torch.rand(n) < shortcut_fraction

    if answer_pos is None:
        pos = torch.randint(0, N_OPTIONS, (n,))
    elif isinstance(answer_pos, int):
        pos = torch.full((n,), answer_pos)
    else:
        pos = answer_pos

    options = torch.zeros(n, N_OPTIONS, dtype=torch.long)
    for i in range(n):
        if is_shortcut[i]:
            # Distractors are SHAPES: only one option is a colour at all.
            distractors = torch.randperm(N_SHAPES)[: N_OPTIONS - 1] + SHAPE0
        else:
            # Distractors are other colours: the image is required.
            pool = [c for c in range(N_COLORS) if c != answer_color[i]]
            perm = torch.randperm(len(pool))[: N_OPTIONS - 1]
            distractors = torch.tensor([pool[j] for j in perm]) + COLOR0
        opts = list(distractors)
        opts.insert(pos[i].item(), answer_color[i] + COLOR0)
        options[i] = torch.tensor(opts[:N_OPTIONS])

    return {"vision": vision, "q_shape": q_shape, "options": options,
            "label": pos, "shortcut": is_shortcut}


def rotate(items, k):
    """Rotate every item's options by k positions -- the circular-evaluation
    transform. The content is identical; only the letters move."""
    opts = torch.roll(items["options"], shifts=k, dims=1)
    label = (items["label"] + k) % N_OPTIONS
    out = dict(items)
    out["options"], out["label"] = opts, label
    return out


def subset(items, mask):
    return {k: v[mask] for k, v in items.items()}


# ---------------------------------------------------------------------------
# 1. The model: reads the scene, the question, and the four options
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class MCQModel(nn.Module):
    """Scores the four options and picks a LETTER -- exactly how VLMs are
    scored on MMMU/MMBench-style benchmarks, and the reason option position
    can matter at all."""

    def __init__(self, blind=False, n_layers=2):
        super().__init__()
        self.blind = blind
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.opt_pos = nn.Parameter(torch.randn(1, N_OPTIONS, D_MODEL) * 0.02)
        self.proj = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                  nn.Linear(D_MODEL, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.score = nn.Linear(D_MODEL, 1)

    def forward(self, items):
        q = self.tok(items["q_shape"] + SHAPE0).unsqueeze(1)
        opts = self.tok(items["options"]) + self.opt_pos
        parts = [q, opts] if self.blind else [self.proj(items["vision"]), q, opts]
        x = torch.cat(parts, dim=1)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x[:, -N_OPTIONS:])                   # the option positions
        return self.score(h).squeeze(-1)                   # (B, N_OPTIONS) letter logits


def train(items, blind=False, steps=600, bs=96, lr=2e-3, extra=None, extra_weight=1):
    """`extra` is a small set of items mixed into every batch -- the mechanism
    of benchmark contamination."""
    torch.manual_seed(1)
    model = MCQModel(blind=blind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = items["vision"].shape[0]
    for _ in range(steps):
        pick = torch.randint(0, n, (bs,))
        batch = subset(items, pick)
        loss = F.cross_entropy(model(batch), batch["label"])
        if extra is not None:
            m = extra["vision"].shape[0]
            epick = torch.randint(0, m, (min(bs, m),))
            ebatch = subset(extra, epick)
            loss = loss + extra_weight * F.cross_entropy(model(ebatch), ebatch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, items):
    if items["vision"].shape[0] == 0:
        return float("nan")
    return (model(items).argmax(-1) == items["label"]).float().mean().item()


@torch.no_grad()
def letter_distribution(model, items):
    counts = torch.bincount(model(items).argmax(-1), minlength=N_OPTIONS).float()
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Experiment 1: the blind baseline
# ---------------------------------------------------------------------------

train_items = make_items(6000)
test_items = make_items(4000)

print("=" * 78)
print("1. THE BLIND BASELINE: how much of the benchmark needs no image?")
print("=" * 78)
print("Half the items are vision-necessary; half have distractors that give the")
print("answer away. Chance = 25%.")
print()

sighted = train(train_items)
blind = train(train_items, blind=True)

vn = subset(test_items, ~test_items["shortcut"])
sc = subset(test_items, test_items["shortcut"])

print(f"{'model':<22} {'FULL benchmark':>15} {'vision-necessary':>18} {'shortcut items':>16}")
for name, m in [("VLM (sees the image)", sighted), ("BLIND (text only)", blind)]:
    print(f"{name:<22} {accuracy(m, test_items):>14.1%} {accuracy(m, vn):>17.1%}"
          f" {accuracy(m, sc):>15.1%}")
print()
overall_b = accuracy(blind, test_items)
overall_s = accuracy(sighted, test_items)
print(f"The headline number for this benchmark is {overall_s:.1%}. The blind model")
print(f"scores {overall_b:.1%} on the same items without any visual input at all, so")
print(f"only {overall_s - overall_b:.1%} of the headline is attributable to vision, and the")
print("vision-necessary column is the only one that measures what the benchmark")
print("claims to measure.")
print()
print("This is not a hypothetical. Published analyses of MMMU, ScienceQA and")
print("MMBench found large fractions of items answerable by a text-only model,")
print("and MMStar was built specifically to filter them out. The rule follows")
print("directly: a VLM benchmark result without a blind baseline is")
print("uninterpretable, and the cheap version of that baseline is to run the same")
print("harness with the images removed.")
print()


# ---------------------------------------------------------------------------
# Experiment 2: option-position bias and circular evaluation
# ---------------------------------------------------------------------------

print("=" * 78)
print("2. OPTION-POSITION BIAS, AND CIRCULAR EVALUATION")
print("=" * 78)
print("Real training corpora are not position-balanced. We tune one model on")
print("data where the correct option sits at position A 55% of the time, and")
print("compare it with a model tuned on balanced data.")
print()

skewed_pos = torch.where(torch.rand(6000) < 0.55, 0,
                         torch.randint(1, N_OPTIONS, (6000,)))
skewed_items = make_items(6000, answer_pos=skewed_pos)
skewed_model = train(skewed_items)

print(f"{'':<26} {'A':>7} {'B':>7} {'C':>7} {'D':>7}")
for name, m in [("balanced-data model", sighted), ("position-skewed model", skewed_model)]:
    d = letter_distribution(m, test_items)
    print(f"{name:<26} " + " ".join(f"{v:>6.1%}" for v in d))
print()
print("How the letter it picks is distributed, on a test set whose correct")
print("answers ARE uniformly placed. A biased model answers 'A' far more often")
print("than any image would justify.")
print()
print(f"{'':<26} {'A-heavy test set':>18} {'balanced test set':>19} {'circular eval':>15}")
for name, m in [("balanced-data model", sighted), ("position-skewed model", skewed_model)]:
    a_heavy = make_items(2000, answer_pos=0)
    balanced = test_items
    circ = sum(accuracy(m, rotate(test_items, k)) for k in range(N_OPTIONS)) / N_OPTIONS
    print(f"{name:<26} {accuracy(m, a_heavy):>17.1%} {accuracy(m, balanced):>18.1%}"
          f" {circ:>14.1%}")
print()
print("The same model scores very differently depending only on where the")
print("benchmark happens to put its correct answers -- and a benchmark built by")
print("people has no reason to be balanced. Circular evaluation (MMBench's fix)")
print("scores every item four times, once per rotation of the options, and")
print("averages: identical content, so any difference between rotations is pure")
print("position bias. It costs 4x the inference and removes the artifact")
print("entirely, which is why it is worth it.")
print()


# ---------------------------------------------------------------------------
# Experiment 3: contamination
# ---------------------------------------------------------------------------

print("=" * 78)
print("3. CONTAMINATION: a few hundred leaked test items")
print("=" * 78)

leaked = subset(test_items, torch.arange(400))              # 10% of the test set
clean = subset(test_items, torch.arange(400, 4000))
contaminated_model = train(train_items, extra=leaked)

print(f"{'model':<34} {'leaked items':>14} {'clean items':>13} {'FULL test':>11}")
for name, m in [("trained on train set only", sighted),
                ("train set + 400 leaked test items", contaminated_model)]:
    print(f"{name:<34} {accuracy(m, leaked):>13.1%} {accuracy(m, clean):>12.1%}"
          f" {accuracy(m, test_items):>10.1%}")
print()
print("The leaked items go to 100% -- memorized, not solved. On clean items the")
print("contaminated model is barely ahead, and that small margin is just the")
print("extra training data doing ordinary work, not a capability gain. So the")
print("headline number rises while the model's actual ability is unchanged, and")
print("nothing in the score itself tells you which of those two things you are")
print("looking at. Only 10% of the test set leaked here; a benchmark whose")
print("images are widely reused can leak far more.")
print()
print("VLM benchmarks are unusually exposed to this. Instruction data is")
print("routinely synthesized FROM public datasets (Lesson 6), and those same")
print("datasets supply the benchmark images -- so contamination happens by")
print("default rather than by accident, and deduplicating on the image is the")
print("only thing that catches it. The countermeasures are the ones from")
print("Phase 08: hold out private test sets, prefer benchmarks with hidden")
print("splits, decontaminate by image hash and near-duplicate search, and")
print("distrust any single headline number.")
print()

print("=" * 78)
print("A MINIMUM REPORTING CHECKLIST FOR VLM RESULTS")
print("=" * 78)
for line in [
    "the blind (text-only) baseline on the same harness",
    "circular / permuted-option evaluation, or a stated position-bias check",
    "how answers were extracted (generation + parse, or option log-likelihood)",
    "the input resolution and vision-token count actually used",
    "for video: the frame count and sampling strategy",
    "decontamination procedure, by image and not only by question text",
    "per-subtask numbers, not just an aggregate average",
]:
    print(f"  - {line}")
