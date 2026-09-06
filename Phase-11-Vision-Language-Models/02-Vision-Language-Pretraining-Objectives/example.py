"""
Vision-language pretraining objectives -- CLIP's softmax InfoNCE, SigLIP's
pairwise sigmoid loss, and a generative captioning loss, all trained from
scratch on synthetic data and compared with real measured numbers.

No downloads, no pretrained weights. CPU only; the whole script takes a few
minutes because experiment 1 trains eight encoders to convergence.

Two experiments, two synthetic worlds:

  Experiment 1 & 2 (contrastive losses). Each pair shares a continuous latent
  "concept" vector; the image view and the text view are different noisy
  linear renderings of it. Retrieval is therefore well posed: exactly one
  caption in the gallery belongs to each image.

  Experiment 3 (what an objective preserves). Each image shows an object of
  one CATEGORY with one ATTRIBUTE, and both are visible in the image -- but
  the caption names the category ONLY, exactly like real web alt-text, which
  says "a dog" and not "a small brown dog facing left". We then ask which
  training objective leaves the attribute recoverable from the frozen
  features.

Run:
    python example.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

LATENT = 10
IMG_RAW = 32
TXT_RAW = 24
EMBED = 16


# ---------------------------------------------------------------------------
# 0. The two synthetic data generators
# ---------------------------------------------------------------------------

img_render = torch.randn(LATENT, IMG_RAW)
txt_render = torch.randn(LATENT, TXT_RAW)


def sample_pairs(n):
    """Continuous shared latent -> one image view and one text view."""
    z = torch.randn(n, LATENT)
    img = z @ img_render + 0.9 * torch.randn(n, IMG_RAW)
    txt = z @ txt_render + 0.9 * torch.randn(n, TXT_RAW)
    return img, txt


NUM_CATEGORIES = 12
NUM_ATTRIBUTES = 4
cat_latents = torch.randn(NUM_CATEGORIES, LATENT)
attr_latents = torch.randn(NUM_ATTRIBUTES, LATENT)


def sample_scenes(n):
    """Image shows category AND attribute; caption carries the category only."""
    cats = torch.randint(0, NUM_CATEGORIES, (n,))
    attrs = torch.randint(0, NUM_ATTRIBUTES, (n,))
    img = (cat_latents[cats] + attr_latents[attrs]) @ img_render + 0.3 * torch.randn(n, IMG_RAW)
    txt = cat_latents[cats] @ txt_render + 0.05 * torch.randn(n, TXT_RAW)
    return img, txt, cats, attrs


class Tower(nn.Module):
    """A tiny stand-in for a vision or text encoder."""

    def __init__(self, in_dim, out_dim=EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def normalized(x):
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# 1. The two contrastive losses
# ---------------------------------------------------------------------------

def infonce_loss(img_e, txt_e, logit_scale):
    """CLIP: softmax cross-entropy over the batch, both directions, averaged.

    Every other item in the batch is a negative and the softmax normalizes
    ACROSS the batch, so the difficulty of the task -- and the amount of
    information a single gradient step can carry, at most log(N) nats -- is
    tied directly to batch size.
    """
    logits = logit_scale.exp() * img_e @ txt_e.t()          # (N, N)
    labels = torch.arange(img_e.shape[0])
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def siglip_loss(img_e, txt_e, logit_scale, logit_bias):
    """SigLIP: independent binary classification on every pair.

    Each of the N*N pairs gets its own sigmoid -- matched pairs positive,
    everything else negative. Nothing is normalized across the batch, so the
    loss stays well posed at small N. It needs a learned BIAS because the
    pair matrix is overwhelmingly negative (N positives vs N^2 - N negatives).
    """
    n = img_e.shape[0]
    logits = logit_scale.exp() * img_e @ txt_e.t() + logit_bias
    targets = 2 * torch.eye(n) - 1                          # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(targets * logits).sum() / n


@torch.no_grad()
def retrieval_r1(img_tower, txt_tower, gallery=256, trials=12):
    """Held-out image->text retrieval R@1 over `gallery`-way galleries."""
    hits = 0
    for _ in range(trials):
        img, txt = sample_pairs(gallery)
        ie, te = normalized(img_tower(img)), normalized(txt_tower(txt))
        hits += ((ie @ te.t()).argmax(dim=1) == torch.arange(gallery)).sum().item()
    return hits / (trials * gallery)


TOTAL_EXAMPLES = 32_000        # held FIXED across batch sizes -- see below


def train_contrastive(objective, batch_size, lr=3e-3):
    torch.manual_seed(1)                                    # identical init everywhere
    img_tower, txt_tower = Tower(IMG_RAW), Tower(TXT_RAW)
    logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
    logit_bias = nn.Parameter(torch.tensor(-10.0))
    params = list(img_tower.parameters()) + list(txt_tower.parameters()) + [logit_scale, logit_bias]
    opt = torch.optim.Adam(params, lr=lr)

    steps = TOTAL_EXAMPLES // batch_size                    # equal DATA, not equal steps
    for _ in range(steps):
        img, txt = sample_pairs(batch_size)
        ie, te = normalized(img_tower(img)), normalized(txt_tower(txt))
        if objective == "infonce":
            loss = infonce_loss(ie, te, logit_scale)
        else:
            loss = siglip_loss(ie, te, logit_scale, logit_bias)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            logit_scale.clamp_(0, math.log(100.0))          # CLIP clamps the temperature
    return img_tower, txt_tower, logit_scale, logit_bias, steps


print("=" * 78)
print("1. InfoNCE (CLIP) vs SIGMOID (SigLIP) ACROSS BATCH SIZES")
print("=" * 78)
print(f"Every run sees the SAME {TOTAL_EXAMPLES:,} training examples (so a small batch")
print("simply takes more steps) and starts from the same initialization.")
print("Score: held-out image->text retrieval R@1 in 256-way galleries (chance 0.4%).")
print()
print(f"{'batch':>7} {'steps':>7} {'negatives/anchor':>18} {'InfoNCE R@1':>13} {'Sigmoid R@1':>13}")

results = {}
for bs in [4, 16, 64, 256]:
    nce = train_contrastive("infonce", bs)
    sig = train_contrastive("sigmoid", bs)
    r_nce, r_sig = retrieval_r1(nce[0], nce[1]), retrieval_r1(sig[0], sig[1])
    results[bs] = (nce, sig, r_nce, r_sig)
    print(f"{bs:>7} {nce[4]:>7} {bs - 1:>18} {r_nce:>12.1%} {r_sig:>12.1%}")

print()
print("Both losses pull matched pairs together and push mismatched pairs apart;")
print("they differ in HOW negatives enter. InfoNCE's softmax is normalized across")
print("the batch, so with batch 4 the model only ever has to beat 3 distractors --")
print("it solves that easy task and stops improving, then transfers poorly to a")
print("256-way gallery. Give it more negatives and it keeps getting better. The")
print("sigmoid loss scores each pair independently against an absolute threshold,")
print("so it degrades far more gently as the batch shrinks.")
print()
print("This is why CLIP-style training needs enormous batches -- 32,768 in the")
print("original paper, sharded across hundreds of devices with an all-gather so")
print("that every device's softmax sees the global batch -- and why SigLIP was")
print("proposed: no all-gather, no global softmax, competitive quality at batch")
print("sizes that fit on far less hardware.")
print()
print(f"the logits matrix at CLIP's real batch size: {32768 ** 2:,} cells")
print(f"                                   in fp16: {32768 ** 2 * 2 / 1e9:.1f} GB for ONE matrix,")
print("which is why large-batch InfoNCE is implemented in chunks rather than")
print("materialized in full.")
print()


# ---------------------------------------------------------------------------
# 2. The learned temperature and bias
# ---------------------------------------------------------------------------

print("=" * 78)
print("2. WHAT THE LEARNED TEMPERATURE AND BIAS CONVERGE TO")
print("=" * 78)
for bs in [4, 64, 256]:
    nce, sig, _, _ = results[bs]
    print(f"batch {bs:>4}:  InfoNCE exp(logit_scale) = {nce[2].exp().item():6.1f}"
          f"   |   SigLIP exp(logit_scale) = {sig[2].exp().item():6.1f}, bias = {sig[3].item():7.2f}")
print()
print("Cosine similarities live in [-1, 1]. Feed those straight into a softmax and")
print("the distribution is nearly uniform no matter how good the encoders are, so")
print("gradients stay tiny; the temperature is therefore a LEARNED parameter that")
print("grows until the similarity distribution is sharp enough to be informative.")
print("(CLIP clamps it at 100 to keep training stable -- so does this script.)")
print("SigLIP's extra learned bias sits strongly negative because N^2 - N of the")
print("N^2 pairs are negatives: without it, 'no match anywhere' is an excellent")
print("local minimum for the first few thousand steps.")
print()


# ---------------------------------------------------------------------------
# 3. What each objective forces the encoder to KEEP
# ---------------------------------------------------------------------------

def train_contrastive_scenes(steps=3000, bs=64, lr=3e-3):
    """Contrastive training where the caption names the category only."""
    torch.manual_seed(1)
    img_tower, txt_tower = Tower(IMG_RAW), Tower(TXT_RAW)
    logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
    opt = torch.optim.Adam(list(img_tower.parameters()) + list(txt_tower.parameters())
                           + [logit_scale], lr=lr)
    for _ in range(steps):
        img, txt, _, _ = sample_scenes(bs)
        ie, te = normalized(img_tower(img)), normalized(txt_tower(txt))
        loss = infonce_loss(ie, te, logit_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            logit_scale.clamp_(0, math.log(100.0))
    return img_tower


def train_captioning(steps=3000, bs=64, lr=3e-3, caption_has_attribute=False):
    """Generative supervision: predict the caption's content from the image.

    A real captioner is autoregressive next-token prediction over the caption
    (the Phase 01 / Phase 04 LM loss). What matters here is only that the
    supervision is per-content-token rather than one similarity score per pair.
    """
    torch.manual_seed(1)
    img_tower = Tower(IMG_RAW)
    cat_head = nn.Linear(EMBED, NUM_CATEGORIES)
    attr_head = nn.Linear(EMBED, NUM_ATTRIBUTES)
    opt = torch.optim.Adam(list(img_tower.parameters()) + list(cat_head.parameters())
                           + list(attr_head.parameters()), lr=lr)
    for _ in range(steps):
        img, _, cats, attrs = sample_scenes(bs)
        feats = img_tower(img)
        loss = F.cross_entropy(cat_head(feats), cats)
        if caption_has_attribute:
            loss = loss + F.cross_entropy(attr_head(feats), attrs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return img_tower


@torch.no_grad()
def frozen_feats(tower, n):
    img, _, cats, attrs = sample_scenes(n)
    return normalized(tower(img)), cats, attrs


def linear_probe(tower, target, train_n=4000, test_n=2000, steps=800):
    """Freeze the encoder, fit a linear classifier on its features, measure test
    accuracy: is the information still linearly present in the representation?"""
    n_classes = NUM_ATTRIBUTES if target == "attr" else NUM_CATEGORIES
    Xtr, ctr, atr = frozen_feats(tower, train_n)
    Xte, cte, ate = frozen_feats(tower, test_n)
    ytr, yte = (atr, ate) if target == "attr" else (ctr, cte)
    probe = nn.Linear(EMBED, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(probe(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (probe(Xte).argmax(1) == yte).float().mean().item()


@torch.no_grad()
def collapse_ratio(tower, n=4000):
    """Within-category spread divided by between-category spread.

    Small = every image sharing a caption has been squashed onto the same
    point, so whatever distinguished those images is gone.
    """
    X, cats, _ = frozen_feats(tower, n)
    within = torch.stack([X[cats == k].var(0).sum() for k in range(NUM_CATEGORIES)
                          if (cats == k).sum() > 1]).mean()
    between = torch.stack([X[cats == k].mean(0) for k in range(NUM_CATEGORIES)]).var(0).sum()
    return (within / between).item()


print("=" * 78)
print("3. WHAT EACH OBJECTIVE FORCES THE ENCODER TO KEEP")
print("=" * 78)
print("Every image shows a CATEGORY and an ATTRIBUTE. The captions name the")
print("category only -- the attribute is visible in the pixels and absent from")
print("the text, exactly like real web alt-text. We freeze each trained image")
print("tower and linear-probe its features for both facts.")
print()

rows = [
    ("contrastive (InfoNCE)", train_contrastive_scenes()),
    ("captioning, category only", train_captioning(caption_has_attribute=False)),
    ("captioning, category + attribute", train_captioning(caption_has_attribute=True)),
]
print(f"{'image tower trained with':<34} {'category':>9} {'attribute':>10} {'within/between':>15}")
for name, tower in rows:
    print(f"{name:<34} {linear_probe(tower, 'cat'):>8.1%} {linear_probe(tower, 'attr'):>9.1%}"
          f" {collapse_ratio(tower):>15.4f}")
print(f"{'chance':<34} {1 / NUM_CATEGORIES:>8.1%} {1 / NUM_ATTRIBUTES:>9.1%}")
print()
print("Every tower nails the category -- that is what all three objectives are")
print("supervised on. They differ on the attribute, which no caption mentions.")
print()
print("The last column explains why. Contrastive training's target for an image")
print("is its caption embedding, and here all images of a category share one")
print("caption, so the loss actively squashes them onto the same point: its")
print("within-category spread is several times smaller than the captioner's, and")
print("the attribute -- the thing that distinguished those images -- degrades")
print("with it. The captioner is only ever asked to predict the category and")
print("still preserves the attribute far better, because a generative head has")
print("no incentive to actively DESTROY information it doesn't use.")
print()
print("This is the practical argument against building a VLM on a contrastive")
print("tower alone: a representation optimized to match short web captions is")
print("optimized to discard everything those captions omit, which is exactly the")
print("fine detail -- counts, small text, spatial relations, attributes -- a VLM")
print("gets asked about later. Real pipelines respond by mixing objectives")
print("(CoCa-style contrastive + captioning), by training on dense re-captions")
print("rather than raw alt-text, or by fusing a contrastive tower with a")
print("self-supervised one such as DINOv2 (Lesson 1 section 2).")
