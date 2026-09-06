"""
Beyond vision -- three properties of genuinely multimodal models, each built
and measured from scratch.

No downloads, no pretrained weights. CPU, a few minutes.

  1. EMERGENT CROSS-MODAL ALIGNMENT (the ImageBind result). Train
     (text, image) pairs and (text, audio) pairs -- and NEVER a single
     (image, audio) pair -- then measure image->audio retrieval. Because both
     modalities were pulled toward the same text anchor, they end up aligned
     with each other for free. This is why an omni-model does not need
     O(n^2) paired datasets for n modalities.

  2. GENERATION IS NEXT-TOKEN PREDICTION (the Chameleon result). Quantize
     images into discrete tokens, put them in the SAME vocabulary as text,
     and train one causal Transformer on interleaved sequences with one
     objective. The same weights then do understanding (image -> text) and
     generation (text -> image), and we measure both.

  3. WHERE THE MODALITY GAP COMES FROM. Before any training, each randomly
     initialized encoder squeezes everything it sees into a narrow cone, and
     different encoders' cones point in different directions -- so the two
     modalities start out in separate regions of the sphere for reasons that
     have nothing to do with their content. We measure that gap at
     initialization and again after training.

Run:
    python example.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

LATENT = 10
EMBED = 16
RAW = {"text": 20, "image": 32, "audio": 28, "depth": 24}


# ===========================================================================
# 1. Emergent alignment: bind everything to text, get the rest for free
# ===========================================================================

renders = {m: torch.randn(LATENT, d) for m, d in RAW.items()}


def sample_pair(n, mod_a, mod_b):
    """Two views of the same underlying concept, in two different modalities."""
    z = torch.randn(n, LATENT)
    a = z @ renders[mod_a] + 0.6 * torch.randn(n, RAW[mod_a])
    b = z @ renders[mod_b] + 0.6 * torch.randn(n, RAW[mod_b])
    return a, b


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(), nn.Linear(64, EMBED))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def infonce(a, b, scale):
    logits = scale.exp() * a @ b.t()
    labels = torch.arange(a.shape[0])
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_bindings(pairs, steps=1500, bs=128, lr=3e-3):
    """`pairs` is the list of modality pairs we have paired data for."""
    torch.manual_seed(1)
    enc = nn.ModuleDict({m: Encoder(d) for m, d in RAW.items()})
    scale = nn.Parameter(torch.tensor(math.log(10.0)))
    opt = torch.optim.Adam(list(enc.parameters()) + [scale], lr=lr)
    for step in range(steps):
        ma, mb = pairs[step % len(pairs)]
        xa, xb = sample_pair(bs, ma, mb)
        loss = infonce(enc[ma](xa), enc[mb](xb), scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            scale.clamp_(0, math.log(100.0))
    return enc


@torch.no_grad()
def retrieval(enc, mod_a, mod_b, gallery=128, trials=16):
    hits = 0
    for _ in range(trials):
        xa, xb = sample_pair(gallery, mod_a, mod_b)
        sims = enc[mod_a](xa) @ enc[mod_b](xb).t()
        hits += (sims.argmax(1) == torch.arange(gallery)).sum().item()
    return hits / (gallery * trials)


print("=" * 78)
print("1. EMERGENT CROSS-MODAL ALIGNMENT: bind to text, get the rest free")
print("=" * 78)
print("Four modalities. Retrieval R@1 in 128-way galleries; chance = 0.8%.")
print("Every run trains the SAME number of steps.")
print()

TEXT_ANCHORED = [("text", "image"), ("text", "audio"), ("text", "depth")]
ALL_PAIRS = TEXT_ANCHORED + [("image", "audio"), ("image", "depth"), ("audio", "depth")]

untrained = train_bindings(TEXT_ANCHORED, steps=0)
anchored = train_bindings(TEXT_ANCHORED)
all_pairs = train_bindings(ALL_PAIRS)
image_only = train_bindings([("text", "image")])

rows = [
    ("untrained encoders", untrained),
    ("trained: text-image only", image_only),
    ("trained: text-X pairs only", anchored),
    ("trained: ALL pairs (upper bound)", all_pairs),
]
print(f"{'training data':<34} {'text->image':>12} {'text->audio':>12} {'image->audio':>13} {'audio->depth':>13}")
for name, enc in rows:
    print(f"{name:<34} {retrieval(enc, 'text', 'image'):>11.1%}"
          f" {retrieval(enc, 'text', 'audio'):>11.1%}"
          f" {retrieval(enc, 'image', 'audio'):>12.1%}"
          f" {retrieval(enc, 'audio', 'depth'):>12.1%}")
print()
print("The third row is the ImageBind result. No (image, audio) pair was ever")
print("shown to it, and no (audio, depth) pair either, yet both retrieve far")
print("above chance -- because each modality was pulled toward the same text")
print("embeddings, and being close to the same thing makes them close to each")
print("other. Training every pair explicitly (row 4) is better, as it should be,")
print("but it needs O(n^2) paired datasets instead of O(n).")
print()
print("Row 2 is the control: an encoder that was never bound to the anchor at all")
print("stays at chance for anything involving it. Alignment is not magic -- it")
print("propagates through the anchor, and only to modalities that were bound to")
print("it. That is exactly why text is the anchor in practice: paired")
print("(text, anything) data exists on the web, while (audio, depth) does not.")
print()


# ===========================================================================
# 2. One vocabulary, one objective: understanding AND generation
# ===========================================================================

N_CONCEPTS = 12
IMG_CODEBOOK = 16          # size of the discrete "image" vocabulary
IMG_TOKENS = 4             # tokens per image
TXT_TOKENS = 2             # tokens per description

# Every concept has a canonical image-token pattern and a canonical name.
concept_img = torch.stack([torch.randint(0, IMG_CODEBOOK, (IMG_TOKENS,))
                           for _ in range(N_CONCEPTS)])
concept_txt = torch.stack([torch.randint(0, N_CONCEPTS, (TXT_TOKENS,))
                           for _ in range(N_CONCEPTS)])

# One shared vocabulary: image codes, then text words, then two control tokens.
IMG0 = 0
TXT0 = IMG0 + IMG_CODEBOOK
BOI = TXT0 + N_CONCEPTS            # "begin image"
BOT = BOI + 1                      # "begin text"
VOCAB = BOT + 1
D_MODEL = 96
SEQ = 1 + IMG_TOKENS + 1 + TXT_TOKENS


def make_sequences(b, order):
    """order='caption'  ->  [BOI] img tokens [BOT] text tokens   (understanding)
       order='generate' ->  [BOT] text tokens [BOI] img tokens    (generation)"""
    c = torch.randint(0, N_CONCEPTS, (b,))
    img = concept_img[c].clone()
    # Realistic noise: one image token in five is corrupted, so the model must
    # generalize rather than memorize an exact pattern.
    mask = torch.rand(b, IMG_TOKENS) < 0.2
    img[mask] = torch.randint(0, IMG_CODEBOOK, (int(mask.sum()),))
    img = img + IMG0
    txt = concept_txt[c] + TXT0
    boi = torch.full((b, 1), BOI)
    bot = torch.full((b, 1), BOT)
    if order == "caption":
        return torch.cat([boi, img, bot, txt], dim=1), c
    return torch.cat([bot, txt, boi, img], dim=1), c


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


class AnyToAny(nn.Module):
    """A plain decoder-only Transformer over ONE mixed-modal vocabulary. There
    is no vision tower, no projector, and no image decoder -- an image is a
    sequence of vocabulary entries like any other."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, SEQ, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, seq):
        x = self.tok(seq) + self.pos[:, : seq.shape[1]]
        mask = torch.triu(torch.full((seq.shape[1],) * 2, float("-inf")), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def train_any(orders, steps=1200, bs=128, lr=2e-3):
    torch.manual_seed(1)
    model = AnyToAny()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        order = orders[step % len(orders)]
        seq, _ = make_sequences(bs, order)
        logits = model(seq)
        # Plain next-token prediction over the whole sequence -- the SAME loss
        # from Phase 01, with image tokens simply present in the vocabulary.
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_understanding(model, n=2000):
    """Image tokens in, text tokens out: does it name the concept?"""
    seq, c = make_sequences(n, "caption")
    logits = model(seq)
    pred = logits[:, -(TXT_TOKENS + 1):-1].argmax(-1)         # the text positions
    return (pred == concept_txt[c] + TXT0).all(1).float().mean().item()


@torch.no_grad()
def eval_generation(model, n=2000):
    """Text tokens in, image tokens out: greedily generate, then check whether
    the generated tokens match the concept's canonical image."""
    seq, c = make_sequences(n, "generate")
    prompt = seq[:, : 1 + TXT_TOKENS + 1]                     # [BOT] text [BOI]
    out = prompt
    for _ in range(IMG_TOKENS):
        nxt = model(out)[:, -1:].argmax(-1)
        out = torch.cat([out, nxt], dim=1)
    gen = out[:, -IMG_TOKENS:] - IMG0
    target = concept_img[c]
    per_token = (gen == target).float().mean().item()
    exact = (gen == target).all(1).float().mean().item()
    return per_token, exact


print("=" * 78)
print("2. ONE VOCABULARY, ONE OBJECTIVE: understanding AND generation")
print("=" * 78)
print("Images are quantized to 4 discrete tokens drawn from a 16-entry codebook")
print("and placed in the same vocabulary as text. One decoder-only Transformer,")
print("plain next-token prediction, trained on both orderings.")
print()

both = train_any(["caption", "generate"])
caption_only = train_any(["caption"])

print(f"{'trained on':<28} {'understanding':>14} {'generation (per-token)':>24} {'exact image':>13}")
for name, m in [("captioning order only", caption_only), ("both orders", both)]:
    u = eval_understanding(m)
    g, e = eval_generation(m)
    print(f"{name:<28} {u:>13.1%} {g:>23.1%} {e:>12.1%}")
print(f"{'chance':<28} {1 / N_CONCEPTS ** TXT_TOKENS:>13.1%}"
      f" {1 / IMG_CODEBOOK:>23.1%} {1 / IMG_CODEBOOK ** IMG_TOKENS:>12.1%}")
print()
print("The same weights that describe an image can draw one, and nothing in the")
print("architecture or the loss distinguishes the two directions -- 'generate an")
print("image' is 'predict the next token' where the next tokens happen to be")
print("image codes. That is the whole idea behind Chameleon-style early-fusion")
print("models, and behind image generation in a chat model.")
print()
print("The first row is the cost of leaving a direction out of the training mix:")
print("a model trained only to caption cannot generate, exactly as in Lesson 6 --")
print("capability follows the data, and 'it has all the parts' is not enough.")
print()
print("Two honest caveats this toy hides. Real image tokenizers (VQ-VAE,")
print("VQ-GAN) lose real detail in quantization, which is why token-based")
print("generation has historically trailed diffusion on image quality; and")
print("mixed-modal training at scale is unstable in ways this small model never")
print("encounters (Chameleon's paper is largely about the norm-growth and")
print("divergence fixes it needed).")
print()


# ===========================================================================
# 3. Where the modality gap comes from
# ===========================================================================

@torch.no_grad()
def geometry(enc, mod_a, mod_b, n=2000):
    """Measure WHERE each modality's embeddings sit, not just their ordering."""
    xa, xb = sample_pair(n, mod_a, mod_b)
    ea, eb = enc[mod_a](xa), enc[mod_b](xb)
    matched = (ea * eb).sum(-1).mean().item()          # cosine: same concept, across modalities
    within_a = (ea @ ea.t()).mean().item()             # cosine: random pairs, same modality
    within_b = (eb @ eb.t()).mean().item()
    centroid_dist = (ea.mean(0) - eb.mean(0)).norm().item()
    return matched, within_a, within_b, centroid_dist


print("=" * 78)
print("3. WHERE THE MODALITY GAP COMES FROM")
print("=" * 78)
print("'cos matched' compares the same concept across two modalities. 'cos within'")
print("compares two RANDOM concepts inside one modality -- it should be ~0 if the")
print("embeddings use the whole sphere, and large if they are crammed into a cone.")
print()
print(f"{'encoders':<20} {'pair':<15} {'cos matched':>12} {'cos within':>11} {'centroid dist':>14}")
for label, enc in [("UNTRAINED", untrained), ("text-anchored", anchored), ("all pairs", all_pairs)]:
    for a, b in [("text", "image"), ("image", "audio")]:
        mt, wa, wb, cd = geometry(enc, a, b)
        print(f"{label:<20} {a + '-' + b:<15} {mt:>+12.3f} {(wa + wb) / 2:>+11.3f} {cd:>14.3f}")
print()
print("Read the UNTRAINED rows first. Two randomly chosen concepts inside one")
print("modality have cosine ~0.5 -- these encoders map everything into a narrow")
print("cone -- while the SAME concept across two modalities has cosine ~0, and the")
print("two modality centroids sit almost a full unit apart. None of that is about")
print("content. It is the geometry a random deep network starts with: each")
print("encoder has its own cone, and the cones point in different directions.")
print()
print("That is the origin of the 'modality gap' reported for CLIP-style models.")
print("The gap is not created by contrastive learning; it is what contrastive")
print("learning inherits and then has to work against. In the trained rows above")
print("the cones have dissolved (within-modality cosine falls to ~0.01, matched")
print("pairs rise above 0.9) because this toy task is fully learnable with enough")
print("steps. On real data, where alignment is never that complete and the")
print("temperature is clamped, a measurable gap survives training -- which is why")
print("the effect has a name.")
print()
print("Practical consequences worth carrying:")
print()
print("  - absolute cross-modal similarity scores are not comparable to")
print("    within-modality ones, so a cosine threshold tuned on one is wrong on")
print("    the other;")
print("  - arithmetic that mixes modality embeddings (averaging an image and a")
print("    text vector) can land in neither region and behave oddly;")
print("  - and it is one reason a VLM trains a projector (Lesson 3) instead of")
print("    feeding CLIP embeddings straight into an LLM: 'aligned' never meant")
print("    'in the same place'.")
