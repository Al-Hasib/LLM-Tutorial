"""
Visual instruction tuning -- what the DATA mixture does, measured.

No downloads, no pretrained weights. CPU, a few minutes.

Everything downstream of the data is held constant: the same architecture,
the same initialization, the same number of training examples, the same
optimizer. Only the instruction-tuning MIXTURE changes. The question this
script answers is the one that decides real VLM quality: given a fixed
budget of training examples, what should be in them?

The world: a scene of 5 objects, left to right, each with a shape and a
color, handed to the model as one vision token per object.

Five instruction types over that scene:
    COLOR_OF <shape>   -> the color of that object
    COUNT <color>      -> how many objects have that color
    EXISTS <shape>     -> yes / no
    LEFT_OF <shape>    -> the shape immediately to its left
    COLOR_LEFT <shape> -> the COLOR of the object to its left   (held out!)

COLOR_LEFT is never trained on by anything. It is a composition of two
trained skills, and it is here to test whether "the model learned the parts"
implies "the model can do the combination".

Each instruction is wrapped in one of several PHRASINGS -- different filler
token patterns standing in for "What color is the...", "Tell me the color
of...", "Could you say what color..." -- so we can also measure what
happens when a model is tuned on one phrasing and evaluated on another.

Run:
    python example.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

N_OBJECTS = 5
N_SHAPES = 10
N_COLORS = 5

# --- answer/argument vocabulary ---------------------------------------------
COLOR0 = 0                       # 0..4      color tokens (also answers)
COUNT0 = COLOR0 + N_COLORS       # 5..10     count answers 0..5
YES, NO = COUNT0 + N_OBJECTS + 1, COUNT0 + N_OBJECTS + 2
SHAPE0 = NO + 1                  # shape tokens (also answers)
TASK0 = SHAPE0 + N_SHAPES        # the "content word" of each task
N_TASKS = 5
FILLER0 = TASK0 + N_TASKS        # phrasing filler tokens
N_PHRASINGS = 4
FILLER_PER_PHRASING = 3
VOCAB = FILLER0 + N_PHRASINGS * FILLER_PER_PHRASING

TASKS = ["COLOR_OF", "COUNT", "EXISTS", "LEFT_OF", "COLOR_LEFT"]

# A prompt is [filler, CONTENT_WORD, filler, argument, answer].
# The content word identifies the task and appears in every phrasing of it --
# like the word "color" appearing in every way of asking about a color. The
# filler tokens are the phrasing: each phrasing owns its own filler tokens, so
# an unseen phrasing brings tokens the model has never encountered, exactly as
# a rephrased instruction brings words the tuning set never contained.
PHRASING_FILLERS = [
    [FILLER0 + FILLER_PER_PHRASING * p + i for i in range(FILLER_PER_PHRASING)]
    for p in range(N_PHRASINGS)
]

D_VISION = 24
D_MODEL = 64

shape_code = torch.randn(N_SHAPES, D_VISION)
color_code = torch.randn(N_COLORS, D_VISION)


def make_examples(b, task, phrasings=(0,)):
    """Build a batch of (vision tokens, prompt+answer sequence, answer)."""
    shapes = torch.stack([torch.randperm(N_SHAPES)[:N_OBJECTS] for _ in range(b)])
    colors = torch.randint(0, N_COLORS, (b, N_OBJECTS))
    vision = shape_code[shapes] + color_code[colors] + 0.15 * torch.randn(b, N_OBJECTS, D_VISION)

    idx = torch.arange(b)
    task_id = TASKS.index(task)

    if task == "COLOR_OF":
        slot = torch.randint(0, N_OBJECTS, (b,))
        arg = shapes[idx, slot] + SHAPE0
        ans = colors[idx, slot] + COLOR0
    elif task == "COUNT":
        c = torch.randint(0, N_COLORS, (b,))
        arg = c + COLOR0
        ans = (colors == c.unsqueeze(1)).sum(1) + COUNT0
    elif task == "EXISTS":
        # Half the time ask about a shape that is present, half about one that isn't.
        present = torch.rand(b) < 0.5
        slot = torch.randint(0, N_OBJECTS, (b,))
        in_scene = shapes[idx, slot]
        # A shape guaranteed absent: the last of the permutation is unused.
        absent = torch.stack([torch.randperm(N_SHAPES)[:1] for _ in range(b)]).squeeze(1)
        is_absent = (shapes == absent.unsqueeze(1)).sum(1) == 0
        chosen = torch.where(present | ~is_absent, in_scene, absent)
        arg = chosen + SHAPE0
        hit = (shapes == chosen.unsqueeze(1)).sum(1) > 0
        ans = torch.where(hit, torch.full((b,), YES), torch.full((b,), NO))
    elif task == "LEFT_OF":
        slot = torch.randint(1, N_OBJECTS, (b,))          # never slot 0
        arg = shapes[idx, slot] + SHAPE0
        ans = shapes[idx, slot - 1] + SHAPE0
    elif task == "COLOR_LEFT":
        slot = torch.randint(1, N_OBJECTS, (b,))
        arg = shapes[idx, slot] + SHAPE0
        ans = colors[idx, slot - 1] + COLOR0
    else:
        raise ValueError(task)

    # prompt = [filler] [content word] [filler] [argument] [answer]
    choice = torch.tensor(phrasings)[torch.randint(0, len(phrasings), (b,))]
    fillers = torch.tensor(PHRASING_FILLERS)[choice]              # (b, FILLER_PER_PHRASING)
    seq = torch.stack([
        fillers[:, 0],
        torch.full((b,), TASK0 + task_id),
        fillers[:, 1],
        arg,
        ans,
    ], dim=1)
    return vision, seq, ans


CHANCE = {"COLOR_OF": 1 / N_COLORS, "COUNT": 1 / (N_OBJECTS + 1), "EXISTS": 1 / 2,
          "LEFT_OF": 1 / N_SHAPES, "COLOR_LEFT": 1 / N_COLORS}


# ---------------------------------------------------------------------------
# The model (same minimal causal Transformer used throughout this phase)
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


class VLM(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, D_MODEL) * 0.02)
        self.projector = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, vision, seq):
        x = torch.cat([self.projector(vision), self.tok_embed(seq)], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def answer_loss(model, vision, seq):
    """SFT loss: only the answer token is a target (Lesson 5 section 4)."""
    logits = model(vision, seq)
    return F.cross_entropy(logits[:, -2], seq[:, -1])


@torch.no_grad()
def evaluate(model, task, phrasings, n=1200, bs=400):
    correct = 0
    for _ in range(n // bs):
        vision, seq, ans = make_examples(bs, task, phrasings)
        pred = model(vision, seq)[:, -2].argmax(-1)
        correct += (pred == ans).sum().item()
    return correct / (bs * (n // bs))


TOTAL_STEPS = 1500
BATCH = 64


def train(mixture, phrasings=(0, 1, 2)):
    """`mixture` is the list of task names sampled uniformly during training."""
    torch.manual_seed(1)
    model = VLM()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(TOTAL_STEPS):
        task = mixture[step % len(mixture)]
        vision, seq, _ = make_examples(BATCH, task, phrasings)
        loss = answer_loss(model, vision, seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# 1. Task diversity at a fixed example budget
# ---------------------------------------------------------------------------

print("=" * 82)
print("1. TASK DIVERSITY AT A FIXED BUDGET")
print("=" * 82)
print(f"Every run sees exactly {TOTAL_STEPS * BATCH:,} training examples and the same")
print("architecture; only WHICH tasks those examples cover changes.")
print("COLOR_LEFT is never trained by any run -- it is a composition of two")
print("trained skills, included to test whether composition comes for free.")
print()

mixtures = {
    "COLOR_OF only": ["COLOR_OF"],
    "COLOR_OF + COUNT": ["COLOR_OF", "COUNT"],
    "all 4 trained tasks": ["COLOR_OF", "COUNT", "EXISTS", "LEFT_OF"],
}

header = f"{'trained on':<22}" + "".join(f"{t:>12}" for t in TASKS)
print(header)
print("-" * len(header))
models = {}
for name, mix in mixtures.items():
    model = train(mix)
    models[name] = model
    cells = "".join(f"{evaluate(model, t, (0, 1, 2)):>11.1%}" for t in TASKS)
    print(f"{name:<22}{cells}")
print(f"{'chance':<22}" + "".join(f"{CHANCE[t]:>11.1%}" for t in TASKS))
print()
print("Reading across the rows:")
print()
print("* A model tuned on ONE task scores ZERO on the others -- not chance,")
print("  zero. It answers every instruction with the kind of token its training")
print("  data used, so a COLOR_OF-tuned model replies to 'how many?' with a")
print("  color. It can see the image perfectly well; it has no idea what the")
print("  other instructions mean. This is the most common way a VLM disappoints")
print("  in practice: it does what its instruction data covered, and for")
print("  everything else it confidently produces the wrong SHAPE of answer.")
print()
print("* Splitting the SAME budget across four tasks makes the model useful on")
print("  all of them for a modest per-task cost (COUNT gives up some accuracy to")
print("  a quarter of the data). That trade is why real visual-instruction sets")
print("  are mixtures of dozens of task types rather than one large VQA corpus.")
print()
print("* COLOR_LEFT stays near chance for every mixture. The model knows how to")
print("  report a color, and knows how to find the object to the left, and cannot")
print("  put the two together, because nothing ever asked it to. Composition is")
print("  not free -- if you want a capability at inference time, something in the")
print("  instruction mixture has to look like it.")
print()


# ---------------------------------------------------------------------------
# 2. Phrasing diversity
# ---------------------------------------------------------------------------

print("=" * 82)
print("2. PHRASING DIVERSITY: TEMPLATE OVERFITTING")
print("=" * 82)
print("Same tasks, same budget. The only difference is how many ways each")
print("instruction is worded during training. Evaluation uses a phrasing that")
print("NO run trained on.")
print()

all_four = ["COLOR_OF", "COUNT", "EXISTS", "LEFT_OF"]
single_phrase = train(all_four, phrasings=(0,))
multi_phrase = train(all_four, phrasings=(0, 1, 2))

print(f"{'trained with':<26} {'seen phrasing':>15} {'UNSEEN phrasing':>17} {'drop':>7}")
for label, m, train_ph in [("1 phrasing", single_phrase, (0,)),
                           ("3 phrasings", multi_phrase, (0, 1, 2))]:
    seen = sum(evaluate(m, t, train_ph) for t in all_four) / 4
    unseen = sum(evaluate(m, t, (3,)) for t in all_four) / 4
    print(f"{label:<26} {seen:>14.1%} {unseen:>16.1%} {seen - unseen:>6.1%}")
print()
print("Two separate things happen here, and both favour phrasing diversity.")
print()
print("The drop column is template overfitting: the single-phrasing model loses")
print("several points when the wording changes, because it was never given a")
print("reason to treat the phrasing as irrelevant. The multi-phrasing model")
print("barely moves -- having seen the same task worded three ways, it has")
print("learned to key on what the phrasings share.")
print()
print("The larger effect is in the LEVEL, not the drop: the multi-phrasing model")
print("is more accurate even on wording it was trained on. Phrasing variety acts")
print("like augmentation, forcing the model to find the actual task signal")
print("instead of a surface shortcut. This is why instruction datasets are")
print("generated with many templates per task, and why 'scores well on the")
print("benchmark, brittle for users' usually traces back to a template-poor")
print("tuning set rather than to the model.")
print()


# ---------------------------------------------------------------------------
# 3. Teaching the composition explicitly
# ---------------------------------------------------------------------------

print("=" * 82)
print("3. WHAT IT TAKES TO GET THE HELD-OUT SKILL")
print("=" * 82)
mixed_with_target = train(all_four + ["COLOR_LEFT"])
before = evaluate(models["all 4 trained tasks"], "COLOR_LEFT", (0, 1, 2))
after = evaluate(mixed_with_target, "COLOR_LEFT", (0, 1, 2))
others_before = sum(evaluate(models["all 4 trained tasks"], t, (0, 1, 2)) for t in all_four) / 4
others_after = sum(evaluate(mixed_with_target, t, (0, 1, 2)) for t in all_four) / 4
print(f"{'':<34} {'COLOR_LEFT':>12} {'other 4 (avg)':>15}")
print(f"{'4-task mixture (no COLOR_LEFT)':<34} {before:>11.1%} {others_before:>14.1%}")
print(f"{'5-task mixture (with COLOR_LEFT)':<34} {after:>11.1%} {others_after:>14.1%}")
print()
print("Adding the task to the mixture -- at the same total budget, so every other")
print("task lost a fifth of its data -- takes COLOR_LEFT from chance to far above")
print("it, and the other four pay a visible but much smaller price. That is the")
print("economics of visual instruction tuning: coverage of a missing capability")
print("buys more than extra examples of a capability you already have, and the")
print("cost of adding it is real but bounded. It is also why so much effort goes")
print("into synthesizing instruction data for skills that no naturally occurring")
print("image-text corpus contains (OCR reasoning, chart reading, grounding with")
print("coordinates, multi-image comparison, GUI actions -- see Lesson 8).")
