"""
Interpretability and Mechanistic Interpretability

Two self-contained, honest demonstrations:

PART 1 -- Probing classifiers.
    A small MLP is trained on a PRIMARY task (classify an 8-bit binary number
    into one of 4 magnitude buckets, a task that is exactly determined by the
    two most-significant bits). The least-significant bit -- parity, i.e.
    even/odd -- is statistically independent of the bucket label and is never
    part of the training signal at all. We then freeze the trained model,
    pull out its internal hidden activations, and train a linear probe to
    recover parity from those frozen activations. We compare the trained
    model's probe accuracy against two baselines: a probe trained directly
    on the raw input bits, and a probe trained on the activations of an
    UNTRAINED (random-weight) copy of the same architecture. This is meant
    to make the "probing proves representation, not causal use" caveat from
    the README concrete: we can directly check whether the trained model's
    own OUTPUT actually depends on parity (it does not, by construction),
    even where a probe finds parity linearly decodable in its activations.

PART 2 -- Sparse Autoencoders (SAEs) and superposition.
    Synthetic "activation-like" vectors are built by packing more true
    underlying features (30) into fewer raw dimensions (10) than there are
    features -- a standard toy superposition setup -- via sparse random
    combinations of fixed feature directions. A small overcomplete SAE
    (hidden dimension >> input dimension) is trained on these vectors at
    several L1 sparsity penalty strengths, and we report, with real
    measured numbers, the reconstruction-quality vs. sparsity/dead-unit
    trade-off this produces.

Runtime: well under a minute on a CPU.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# =============================================================================
# PART 1 -- Probing classifiers
# =============================================================================

print("=" * 78)
print("PART 1 -- PROBING CLASSIFIERS")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1.1 Toy data: every integer 0..255 as an 8-bit binary vector, MSB first.
#     bit index 0 (leftmost, most significant) and bit index 1 together
#     exactly determine the "bucket" label (value // 64, so 4 buckets).
#     bit index 7 (rightmost, least significant) is exactly the parity bit
#     (even/odd). Because all 8 bits of a uniform 0..255 integer are
#     independent, the bucket label and the parity label are STATISTICALLY
#     INDEPENDENT of each other by construction -- parity carries no
#     information the primary task needs.
# ---------------------------------------------------------------------------

def to_bits(n, num_bits=8):
    return [float((n >> (num_bits - 1 - i)) & 1) for i in range(num_bits)]


all_values = list(range(256))
inputs = torch.tensor([to_bits(v) for v in all_values], dtype=torch.float32)   # (256, 8)
buckets = torch.tensor([v // 64 for v in all_values], dtype=torch.long)        # (256,) in {0,1,2,3}
parity = torch.tensor([v % 2 for v in all_values], dtype=torch.long)           # (256,) in {0,1}, 0=even 1=odd

# Shuffle once, then split into train/test so both tasks share the same split.
perm = torch.randperm(256)
train_idx, test_idx = perm[:200], perm[200:]

x_train, x_test = inputs[train_idx], inputs[test_idx]
bucket_train, bucket_test = buckets[train_idx], buckets[test_idx]
parity_train, parity_test = parity[train_idx], parity[test_idx]

print(f"Dataset: all 256 8-bit integers -> {len(train_idx)} train / {len(test_idx)} test")
print("Primary task label (bucket) depends only on the top 2 bits.")
print("Probed concept (parity) depends only on the bottom 1 bit.")
print("These two labels are statistically independent by construction.\n")


# ---------------------------------------------------------------------------
# 1.2 The model trained on the PRIMARY task only (bucket classification).
#     We will probe the activations coming out of `hidden2` (16-dim).
# ---------------------------------------------------------------------------

class BucketClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 4)

    def forward(self, x):
        h1 = F.relu(self.hidden1(x))
        h2 = F.relu(self.hidden2(h1))     # <-- this is the activation we probe
        logits = self.output(h2)
        return logits, h2


trained_model = BucketClassifier()
optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.02)

for epoch in range(300):
    logits, _ = trained_model(x_train)
    loss = F.cross_entropy(logits, bucket_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    test_logits, _ = trained_model(x_test)
    bucket_acc = (test_logits.argmax(dim=-1) == bucket_test).float().mean().item()
print(f"Primary task (bucket classification) test accuracy: {bucket_acc * 100:.1f}%")

# Sanity/causal check: does the trained model's OWN prediction actually
# depend on parity? Flip the least-significant bit of every test input and
# see whether the predicted bucket changes. If the model is doing what the
# task requires, flipping parity should change the prediction ~0% of the
# time -- this is the ground truth against which we'll compare probe results.
with torch.no_grad():
    x_test_flipped = x_test.clone()
    x_test_flipped[:, -1] = 1.0 - x_test_flipped[:, -1]   # flip parity bit only
    logits_orig, _ = trained_model(x_test)
    logits_flipped, _ = trained_model(x_test_flipped)
    pred_changed = (logits_orig.argmax(-1) != logits_flipped.argmax(-1)).float().mean().item()
print(f"Fraction of predictions that change when we flip parity (causal check): {pred_changed * 100:.1f}%")
print("-> The model's own OUTPUT does not use parity at all (as expected -- it")
print("   was never part of the training signal). Any probe that nonetheless")
print("   decodes parity from an internal LAYER is finding a REPRESENTATION,")
print("   not evidence the model's computation actually USES it downstream.")


# ---------------------------------------------------------------------------
# 1.3 Extract FROZEN activations (no gradients into the classifier) from
#     three sources, then train a linear probe on each to predict parity:
#       (a) the TRAINED model's hidden2 activations
#       (b) the raw input bits themselves (baseline: is the concept even
#           linearly present in the input at all?)
#       (c) an UNTRAINED (freshly initialized, never trained) copy of the
#           exact same architecture's hidden2 activations (baseline: do
#           random projections alone preserve linear decodability?)
# ---------------------------------------------------------------------------

def get_frozen_activations(model, x):
    model.eval()
    with torch.no_grad():
        _, h2 = model(x)
    return h2


trained_model.eval()
h_train_trained = get_frozen_activations(trained_model, x_train)
h_test_trained = get_frozen_activations(trained_model, x_test)

untrained_model = BucketClassifier()   # freshly initialized, NEVER trained
h_train_untrained = get_frozen_activations(untrained_model, x_train)
h_test_untrained = get_frozen_activations(untrained_model, x_test)


def train_linear_probe(feat_train, label_train, feat_test, label_test, num_classes=2, epochs=400, lr=0.05):
    """A probe is deliberately just ONE linear layer -- logistic regression --
    so that success means the concept is LINEARLY decodable, not decodable by
    an arbitrarily powerful nonlinear function (which could recover almost
    anything from almost any representation and would tell us little)."""
    probe = nn.Linear(feat_train.shape[1], num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    feat_train = feat_train.detach()
    feat_test = feat_test.detach()
    for _ in range(epochs):
        logits = probe(feat_train)
        loss = F.cross_entropy(logits, label_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        test_acc = (probe(feat_test).argmax(-1) == label_test).float().mean().item()
    return test_acc


acc_raw_input = train_linear_probe(x_train, parity_train, x_test, parity_test)
acc_trained_activations = train_linear_probe(h_train_trained, parity_train, h_test_trained, parity_test)
acc_untrained_activations = train_linear_probe(h_train_untrained, parity_train, h_test_untrained, parity_test)

print("\nLinear probe test accuracy for the PARITY concept, by feature source:")
print(f"  (a) probe on TRAINED model's hidden activations   : {acc_trained_activations * 100:5.1f}%")
print(f"  (b) probe on RAW INPUT bits (baseline)             : {acc_raw_input * 100:5.1f}%")
print(f"  (c) probe on UNTRAINED (random) model's activations: {acc_untrained_activations * 100:5.1f}%")


def describe(acc, name):
    if acc > 0.95:
        return f"near-perfect -- {name} still linearly encodes parity almost exactly"
    elif acc > 0.65:
        return f"clearly above chance (50%) -- {name} partially preserves parity"
    else:
        return f"close to chance (50%) -- {name} has largely destroyed parity"


print("\nInterpretation of these actual numbers:")
print(f"  - Raw input probe is {describe(acc_raw_input, 'the raw input')}: parity IS")
print("    exactly one input coordinate, so a linear probe can read it off directly.")
print(f"  - Trained-model probe is {describe(acc_trained_activations, 'the trained activations')}.")
print(f"  - Untrained-model probe is {describe(acc_untrained_activations, 'the untrained activations')}.")
if acc_untrained_activations > 0.65:
    print("  - Honest note: even an UNTRAINED network's random projection keeps parity")
    print("    substantially linearly readable here -- a real, known phenomenon (random")
    print("    features are often somewhat linearly informative). So high probe accuracy")
    print("    on the trained model, by itself, is only weak evidence that TRAINING")
    print("    specifically is what makes the concept decodable -- an untrained network")
    print("    can look similarly decodable. Combined with the causal check above (the")
    print("    model's own predictions do not change when parity is flipped), this is a")
    print("    concrete illustration of the lesson's central caveat: a probe finding a")
    print("    linearly-readable concept is evidence about REPRESENTATION, not proof the")
    print("    model's downstream computation CAUSALLY USES that concept.")
else:
    print("  - The untrained network's activations are much less decodable than the")
    print("    trained model's, which is consistent with a story where training itself")
    print("    reshapes the representation. Even so, remember the causal check above:")
    print("    the trained model's OWN predictions never actually use parity -- so this")
    print("    probe result is still only evidence of REPRESENTATION, not causal use.")


# =============================================================================
# PART 2 -- Sparse Autoencoders and superposition
# =============================================================================

print("\n" + "=" * 78)
print("PART 2 -- SPARSE AUTOENCODERS (SAE) AND SUPERPOSITION")
print("=" * 78)

# ---------------------------------------------------------------------------
# 2.1 Build synthetic "activation-like" data with superposition baked in:
#     N_TRUE_FEATURES (30) fixed, random, unit-norm feature directions living
#     in an AMBIENT space of only AMBIENT_DIM (10) dimensions -- more true
#     features than raw dimensions, exactly the toy-superposition setup from
#     Elhage et al. (2022). Each sample activates only a SPARSE random subset
#     of the 30 features (on average ~3), then sums their directions,
#     scaled by random nonnegative coefficients, into one 10-dim vector.
#     A real model's activations are believed to look like this: many more
#     "true concepts" than raw neurons, superimposed via sparsity.
# ---------------------------------------------------------------------------

N_TRUE_FEATURES = 30
AMBIENT_DIM = 10
SPARSITY_PROB = 0.1     # probability each true feature is "active" per sample (~3 of 30 on average)
N_SAMPLES = 4000

feature_directions = F.normalize(torch.randn(AMBIENT_DIM, N_TRUE_FEATURES), dim=0)   # unit vectors, columns

active_mask = (torch.rand(N_SAMPLES, N_TRUE_FEATURES) < SPARSITY_PROB).float()
magnitudes = torch.rand(N_SAMPLES, N_TRUE_FEATURES) * 0.5 + 0.5   # magnitude in [0.5, 1.0] when active
coefficients = active_mask * magnitudes                            # (N_SAMPLES, N_TRUE_FEATURES), mostly zero

activation_data = coefficients @ feature_directions.T               # (N_SAMPLES, AMBIENT_DIM)

avg_active_true_features = active_mask.sum(dim=1).mean().item()
print(f"Synthetic data: {N_TRUE_FEATURES} true features superimposed into {AMBIENT_DIM} raw dimensions")
print(f"Average number of true features active per sample: {avg_active_true_features:.2f} (of {N_TRUE_FEATURES})")

data_train, data_test = activation_data[:3200], activation_data[3200:]


# ---------------------------------------------------------------------------
# 2.2 The SAE itself: linear encoder -> ReLU -> linear decoder.
#     OVERCOMPLETE: hidden dim (60) is larger than the input dim (10), so the
#     model has room to unpack superimposed features into separate directions.
# ---------------------------------------------------------------------------

SAE_HIDDEN_DIM = 60   # overcomplete: 6x the ambient input dimension
ACTIVE_THRESHOLD = 0.05   # a hidden unit counts as "active" on a sample if its value exceeds this


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = F.relu(self.encoder(x))     # sparse code (after training pushes most units to 0)
        x_hat = self.decoder(h)
        return x_hat, h


def train_sae(l1_strength, epochs=300, lr=0.02):
    sae = SparseAutoencoder(AMBIENT_DIM, SAE_HIDDEN_DIM)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(epochs):
        x_hat, h = sae(data_train)
        recon_loss = F.mse_loss(x_hat, data_train)
        sparsity_loss = h.abs().mean()          # mean L1 over the batch and hidden units
        loss = recon_loss + l1_strength * sparsity_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    sae.eval()
    with torch.no_grad():
        x_hat_test, h_test = sae(data_test)
        final_recon_loss = F.mse_loss(x_hat_test, data_test).item()
        active_per_sample = (h_test > ACTIVE_THRESHOLD).float()          # (N_test, SAE_HIDDEN_DIM)
        avg_sparsity = active_per_sample.mean().item()                    # mean fraction of units active per input
        # a unit is "alive" if it fires on at least 1% of test samples, "dead" otherwise
        fraction_of_samples_active = active_per_sample.mean(dim=0)        # per-unit firing rate
        num_alive_units = (fraction_of_samples_active > 0.01).sum().item()
        num_dead_units = SAE_HIDDEN_DIM - num_alive_units
    return final_recon_loss, avg_sparsity, num_alive_units, num_dead_units


L1_STRENGTHS = [0.0005, 0.005, 0.02, 0.08]   # weak -> strong

print(f"\nTraining a {AMBIENT_DIM} -> {SAE_HIDDEN_DIM} (overcomplete) SAE at {len(L1_STRENGTHS)} L1 strengths...\n")
print(f"{'L1 strength':>12} | {'recon. MSE':>10} | {'avg sparsity':>12} | {'alive units':>12} | {'dead units':>10}")
print("-" * 68)

results = []
for l1 in L1_STRENGTHS:
    recon_loss, avg_sparsity, num_alive, num_dead = train_sae(l1)
    results.append((l1, recon_loss, avg_sparsity, num_alive, num_dead))
    print(f"{l1:>12} | {recon_loss:>10.4f} | {avg_sparsity:>11.1%} | {num_alive:>12d} | {num_dead:>10d}")

print("\nInterpretation of these actual numbers:")
weakest = results[0]
strongest = results[-1]
print(f"  - Weakest L1 ({weakest[0]}): reconstruction MSE = {weakest[1]:.4f}, "
      f"avg sparsity = {weakest[2]:.1%}, {weakest[4]} dead units out of {SAE_HIDDEN_DIM}.")
print(f"  - Strongest L1 ({strongest[0]}): reconstruction MSE = {strongest[1]:.4f}, "
      f"avg sparsity = {strongest[2]:.1%}, {strongest[4]} dead units out of {SAE_HIDDEN_DIM}.")

if strongest[2] < weakest[2] and strongest[4] >= weakest[4]:
    print("  - As predicted: increasing the L1 strength drives the code SPARSER (lower")
    print("    average fraction of active units per input) and pushes MORE units to")
    print("    become permanently dead, while weaker L1 keeps more units alive but with")
    print("    denser, less individually-selective activations.")
if strongest[1] > weakest[1]:
    print("  - And reconstruction quality trades off against that sparsity: the strongest")
    print("    L1 setting has the WORST (highest) reconstruction error of the settings")
    print("    tested here -- squeezing the code sparser makes it a lossier bottleneck.")
else:
    print("  - Reconstruction error did not strictly worsen with stronger L1 in this run --")
    print("    with this much data and this small a feature set the network still found a")
    print("    reasonably faithful sparse code even under the strongest penalty tested.")

print("\n-> This is the real reconstruction/sparsity/dead-unit trade-off SAE research")
print("   has to navigate at every scale: enough sparsity pressure to pull apart")
print("   superimposed, polysemantic directions into individually monosemantic")
print("   features, without so much pressure that reconstruction (and therefore the")
print("   features actually being preserved at all) falls apart or units simply die.")
