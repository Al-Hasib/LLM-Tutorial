"""
Connectors and visual token compression -- four ways to turn N vision tokens
into K < N tokens for the LLM, trained from scratch and compared at equal K.

No downloads, no pretrained weights. CPU, a few minutes.

The task is Lesson 3's grounding task, scaled up: a scene of 16 objects, one
vision token each (shape + color), and a question naming one shape. The model
must answer with that object's color. Chance = 1/6 = 16.7%.

The connectors compared, all mapping (16 vision tokens) -> (K tokens):

  1. MLP projector (LLaVA)          K = 16, no compression at all
  2. Average pooling                K groups of adjacent tokens, averaged
  3. Learned-query resampler        K learned queries cross-attend to the
     (Perceiver / Q-Former, BLIP-2)  vision tokens; the queries do NOT see
                                     the question
  4. Instruction-aware resampler    the same, except the queries are
     (InstructBLIP-style)            conditioned on the question first

The measurement that matters: accuracy as K shrinks. A query-agnostic
compressor has to guess what will be asked; a query-aware one can keep the
one object the question is about. That difference is invisible at K = 16 and
decisive at K = 1.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

NUM_SHAPES = 20
NUM_COLORS = 6
NUM_OBJECTS = 16          # vision tokens produced by the "tower"
D_VISION = 32
D_MODEL = 64
N_TEXT = 4

shape_code = torch.randn(NUM_SHAPES, D_VISION)
color_code = torch.randn(NUM_COLORS, D_VISION)
TEXT_VOCAB = 3 + NUM_SHAPES


def make_batch(batch_size):
    shapes = torch.stack([torch.randperm(NUM_SHAPES)[:NUM_OBJECTS] for _ in range(batch_size)])
    colors = torch.randint(0, NUM_COLORS, (batch_size, NUM_OBJECTS))
    vision = shape_code[shapes] + color_code[colors]
    vision = vision + 0.2 * torch.randn_like(vision)
    slot = torch.randint(0, NUM_OBJECTS, (batch_size,))
    queried = shapes[torch.arange(batch_size), slot]
    answer = colors[torch.arange(batch_size), slot]
    fixed = torch.tensor([0, 1, 2]).expand(batch_size, 3)
    question = torch.cat([fixed, (queried + 3).unsqueeze(1)], dim=1)
    return vision, question, answer


# ---------------------------------------------------------------------------
# 1. The four connectors
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """LLaVA's connector: one MLP per token, K = N. Nothing is compressed."""

    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                 nn.Linear(D_MODEL, D_MODEL))

    def forward(self, vision, question_embed):
        return self.net(vision)                                   # (B, N, D_MODEL)


class PoolProjector(nn.Module):
    """Average adjacent tokens into K groups, then project.

    This is the cheap, parameter-free compressor -- the family that includes
    pixel-shuffle / patch-merge tricks. It is spatially local: group g always
    summarizes the same region regardless of the image or the question.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                 nn.Linear(D_MODEL, D_MODEL))

    def forward(self, vision, question_embed):
        b, n, d = vision.shape
        pooled = vision.reshape(b, self.k, n // self.k, d).mean(dim=2)
        return self.net(pooled)


class QueryResampler(nn.Module):
    """Perceiver Resampler / Q-Former: K LEARNED query vectors cross-attend to
    the vision tokens; their outputs are the compressed sequence.

    The queries are parameters -- the same K questions are asked of every
    image, and crucially they are chosen before anyone knows what the user
    will ask.
    """

    def __init__(self, k, instruction_aware=False):
        super().__init__()
        self.k = k
        self.instruction_aware = instruction_aware
        self.queries = nn.Parameter(torch.randn(1, k, D_MODEL) * 0.02)
        self.kv_proj = nn.Linear(D_VISION, D_MODEL)
        self.ln_q = nn.LayerNorm(D_MODEL)
        self.ln_kv = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 2 * D_MODEL),
                                nn.GELU(), nn.Linear(2 * D_MODEL, D_MODEL))

    def forward(self, vision, question_embed):
        b = vision.shape[0]
        q = self.queries.expand(b, -1, -1)
        if self.instruction_aware:
            # InstructBLIP's change: let the queries see the instruction before
            # they decide what to extract from the image.
            q = q + question_embed.mean(dim=1, keepdim=True)
        kv = self.ln_kv(self.kv_proj(vision))
        out = self.attn(self.ln_q(q), kv, kv, need_weights=False)[0]
        out = q + out
        return out + self.ff(out)


CONNECTORS = {
    "MLP projector (no compression)": lambda k: MLPProjector(k),
    "average pooling": lambda k: PoolProjector(k),
    "learned-query resampler": lambda k: QueryResampler(k, instruction_aware=False),
    "instruction-aware resampler": lambda k: QueryResampler(k, instruction_aware=True),
}


# ---------------------------------------------------------------------------
# 2. The (tiny) LLM the connector feeds
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


class VLM(nn.Module):
    def __init__(self, connector_fn, k, n_layers=2):
        super().__init__()
        self.tok_embed = nn.Embedding(TEXT_VOCAB, D_MODEL)
        self.connector = connector_fn(k)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, NUM_COLORS)

    def forward(self, vision, question):
        txt = self.tok_embed(question)
        vis = self.connector(vision, txt)
        x = torch.cat([vis, txt], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, -1]))


def train_and_eval(connector_fn, k, steps=700, batch_size=64, lr=2e-3):
    torch.manual_seed(1)
    model = VLM(connector_fn, k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        vision, question, answer = make_batch(batch_size)
        loss = F.cross_entropy(model(vision, question), answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        vision, question, answer = make_batch(2000)
        acc = (model(vision, question).argmax(1) == answer).float().mean().item()
    n_conn = sum(p.numel() for p in model.connector.parameters())
    return acc, n_conn


print("=" * 78)
print("TASK: 16 objects (1 vision token each), 'what color is the <shape>?'")
print(f"Chance = {1 / NUM_COLORS:.1%}. Every connector feeds the same 2-layer LLM.")
print("=" * 78)
print()
print("Accuracy as the connector is forced to emit fewer tokens (K):")
print()
header = f"{'connector':<32}" + "".join(f"{'K=' + str(k):>10}" for k in [16, 8, 4, 1])
print(header)
print("-" * len(header))

table = {}
for name, fn in CONNECTORS.items():
    row = []
    for k in [16, 8, 4, 1]:
        if name.startswith("MLP") and k != NUM_OBJECTS:
            row.append(None)                    # an MLP cannot change the token count
            continue
        acc, n_conn = train_and_eval(fn, k)
        row.append(acc)
        table[(name, k)] = (acc, n_conn)
    cells = "".join(f"{'  --':>10}" if a is None else f"{a:>9.1%}" for a in row)
    print(f"{name:<32}{cells}")

print()
print("Reading the table:")
print()
print("* At K=16 nothing is being compressed and everything works. All the")
print("  interesting behaviour is in the columns to the right.")
print()
print("* Average pooling degrades steadily. It is spatially local and content-")
print("  blind: group g always averages the same slots, so as groups get bigger")
print("  the objects inside one group blur together and the shape->color binding")
print("  is lost -- Lesson 3's pooled baseline, arriving gradually.")
print()
print("* The learned-query resampler compresses adaptively (its queries can")
print("  learn to attend to whichever slots are informative) but its queries are")
print("  FIXED PARAMETERS: the same K questions are asked of every image, chosen")
print("  before the user's question is known. At K=1 that is a hard information")
print("  bottleneck -- one vector cannot carry 16 shape->color bindings.")
print()
print("* The instruction-aware resampler sees the question before it compresses,")
print("  so at K=1 it only has to keep the ONE binding that was asked about. Same")
print("  parameter count, same K, radically different accuracy.")
print()

print("=" * 78)
print("WHAT THE COMPRESSION ACTUALLY BUYS (real numbers from Lesson 1)")
print("=" * 78)
print()
print(f"{'setting':<40} {'vis tokens':>11} {'LLM attn cells':>16} {'saving':>9}")
prompt_len = 100
base = None
for label, n_vis in [("CLIP ViT-L/14 @336, no compression", 576),
                     ("2x2 pixel-shuffle (4x fewer)", 144),
                     ("resampler to 64 queries", 64),
                     ("resampler to 32 queries", 32),
                     ("AnyRes 896px, no compression", 3136)]:
    total = n_vis + prompt_len
    cells = total * total
    if base is None:
        base = cells
    print(f"{label:<40} {n_vis:>11,} {cells:>16,} {base / cells:>8.1f}x")
print()
print(f"(one layer of self-attention over vision tokens + a {prompt_len}-token prompt)")
print()
print("Those savings are why every production VLM compresses somewhere. But the")
print("table above is the warning that comes with them: a compressor chosen")
print("without knowing the question can only keep what is USUALLY useful, and")
print("the queries it drops are gone before the LLM ever sees the image. That is")
print("the whole trade -- context budget against the risk that the one detail the")
print("user asked about was in the discarded 90%.")
print()

print("=" * 78)
print("CONNECTOR PARAMETER COUNTS (K=4)")
print("=" * 78)
for name in CONNECTORS:
    if (name, 4) in table:
        acc, n_conn = table[(name, 4)]
        print(f"{name:<32} {n_conn:>8,} params   acc {acc:>6.1%}")
print()
print("Connectors are cheap in parameters no matter which you pick -- the choice")
print("is about what information survives them, not about model size.")
