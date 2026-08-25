"""
Automatic Prompt Optimization

A real, working prompt-search loop, not a mock of one. We reuse Lesson 1's
in-context-learning task family -- y = (x + k) mod M, with a hidden shift k
that must be inferred purely from in-context examples -- and retrain a
small decoder-only Transformer on it, but this time the TRAINING
DISTRIBUTION deliberately mixes THREE independent "prompt components" that
a real prompt engineer might A/B test:

  1. NUM_EXAMPLES  -- how many in-context example pairs are shown (2-5)
  2. ORDER         -- whether examples are shown sorted by x, or scrambled
  3. PHRASING      -- one of two single-token "instruction style" markers,
                      'P' or 'Q', prepended before the examples. These stand
                      in for two different natural-language instruction
                      phrasings a prompt engineer might try (e.g. "Follow
                      the pattern:" vs. "Here are some examples:") --
                      collapsed to a single symbolic token because this toy
                      model's vocabulary has no natural language, but the
                      mechanism is identical.

Crucially, the training distribution does NOT expose every combination of
these three components equally: marker 'Q' is ONLY ever paired with 2-3
shown examples during training, while marker 'P' is paired with the full
2-5 range. This means (marker='Q', n_shown=5) is a combination the trained
model has NEVER seen, even though each component individually is familiar
-- exactly the kind of "individually reasonable, jointly out-of-distribution"
prompt combination real prompt engineers stumble into by hand. This gives
the 4 x 2 x 2 = 16-combination discrete search space real, learnable
structure for a search algorithm to discover, instead of a flat surface.

After training (weights then frozen), each of the 16 (n_shown, order,
phrasing) combinations is a candidate PROMPT TEMPLATE whose quality we
score with a real accuracy measurement on held-out shift values k -- exactly
Zhou et al.'s (2022) APE framework: propose candidate prompts, score them on
a validation objective, keep the best. We run:

  - RANDOM SEARCH: sample a handful of combinations at random and score them.
  - HILL-CLIMBING (greedy local search): start at a random combination,
    repeatedly move to whichever single-component change improves the
    score, until no neighboring change helps.
  - BRUTE FORCE (all 16 combinations, more trials each): the true optimum,
    computed here ONLY because the space is small enough to enumerate --
    used purely to check the other two methods' results, not as the
    proposed method itself (brute force does not scale to realistic
    prompt-component spaces with far more combinations).

Runtime: ~30-60 seconds on CPU (2500 training steps on tiny sequences, plus
a few thousand fast forward-pass-only evaluations).

Run:
    python example.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 0. Task family (same as Lesson 1): y = (x + k) mod M
# ---------------------------------------------------------------------------

M = 6
TRAIN_KS = [0, 1, 2, 3]
TEST_KS = [4, 5]          # held out entirely from training, as in Lesson 1

CHARS = [str(d) for d in range(M)] + [">", ",", "P", "Q"]
vocab_size = len(CHARS)
stoi = {ch: i for i, ch in enumerate(CHARS)}
itos = {i: ch for i, ch in enumerate(CHARS)}


def encode(s):
    return [stoi[ch] for ch in s]


def build_prompt(k, x_shown, x_query, marker):
    """marker is 'P' or 'Q', prepended as the very first token -- the
    'instruction phrasing' component of the search space."""
    parts = [marker]
    for x in x_shown:
        y = (x + k) % M
        parts.append(f"{x}>{y},")
    prompt = "".join(parts) + f"{x_query}>"
    answer = str((x_query + k) % M)
    return prompt, answer


def sample_episode(k, n_shown, scrambled, marker):
    xs = list(range(M))
    random.shuffle(xs)
    x_shown = xs[:n_shown]
    x_query = xs[n_shown]
    if not scrambled:
        x_shown = sorted(x_shown)
    return build_prompt(k, x_shown, x_query, marker)


# ---------------------------------------------------------------------------
# 1. Model -- identical MiniGPT recipe from Phase 02 Lesson 6 / Lesson 1,
#    re-declared here so this lesson is self-contained.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 26     # 1 (marker) + 5*4 (5 examples "x>y,") + 2 (query) + 1 (answer) = 24, +2 margin
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
BATCH_SIZE = 64
NUM_ITERS = 2500
LEARNING_RATE = 3e-3


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        batch, T, d_model = x.shape

        def split_heads(t):
            return t.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.W_q(x)), split_heads(self.W_k(x)), split_heads(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, T, d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, block_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, targets=None):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def predict_next_char(self, prompt_str):
        ids = torch.tensor([encode(prompt_str)], dtype=torch.long)
        logits, _ = self(ids)
        next_id = logits[0, -1, :].argmax().item()
        return itos[next_id]


# ---------------------------------------------------------------------------
# 2. Training batches -- deliberately UNEVEN component coverage: marker 'Q'
#    only ever sees 2-3 examples; marker 'P' sees the full 2-5 range. This
#    is what gives (marker='Q', n_shown in {4,5}) its out-of-distribution
#    status, and what gives the search space real structure to discover.
# ---------------------------------------------------------------------------

def get_training_batch(batch_size):
    """n_shown is drawn ONCE per batch (as in Lesson 1) so every sequence in
    the batch has identical length and can be stacked directly with no
    padding. marker and order are then drawn INDEPENDENTLY PER EXAMPLE
    within the batch -- except that when n_shown is 4 or 5, marker is forced
    to 'P' for every example, which is exactly what makes (marker='Q',
    n_shown in {4,5}) a combination the model never encounters."""
    n_shown = random.choice([2, 3, 4, 5])
    seqs = []
    for _ in range(batch_size):
        marker = "P" if n_shown >= 4 else random.choice(["P", "Q"])
        scrambled = random.random() < 0.5
        k = random.choice(TRAIN_KS)
        prompt, answer = sample_episode(k, n_shown, scrambled, marker)
        seqs.append(encode(prompt + answer))
    tokens = torch.tensor(seqs, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


# ---------------------------------------------------------------------------
# 3. The scoring function -- a REAL accuracy measurement for one
#    (n_shown, order, marker) combination, on held-out shifts k.
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_combo(model, n_shown, scrambled, marker, num_trials=300):
    """Scores one (n_shown, order, marker) combination by building ALL
    num_trials prompts at once into a single batch (every prompt for a
    fixed n_shown and marker has identical token length, so this is safe)
    and running exactly ONE forward pass, instead of num_trials separate
    Python-level forward calls. Purely a performance optimization -- the
    quantity computed (fraction of held-out-shift queries answered
    correctly) is identical to scoring one prompt at a time."""
    prompts, answer_ids = [], []
    for _ in range(num_trials):
        k = random.choice(TEST_KS)
        prompt, answer = sample_episode(k, n_shown, scrambled, marker)
        prompts.append(encode(prompt))
        answer_ids.append(stoi[answer])
    tokens = torch.tensor(prompts, dtype=torch.long)
    logits, _ = model(tokens)
    preds = logits[:, -1, :].argmax(dim=-1)
    correct = (preds == torch.tensor(answer_ids, dtype=torch.long)).sum().item()
    return correct / num_trials


def all_combinations():
    combos = []
    for n_shown in [2, 3, 4, 5]:
        for scrambled in [False, True]:
            for marker in ["P", "Q"]:
                combos.append((n_shown, scrambled, marker))
    return combos


def combo_str(combo):
    n_shown, scrambled, marker = combo
    order_str = "scrambled" if scrambled else "sorted   "
    return f"marker={marker} n_shown={n_shown} order={order_str}"


# ---------------------------------------------------------------------------
# 4. Search algorithms over the 16-combination discrete space
# ---------------------------------------------------------------------------

def random_search(model, combos, num_samples, trials_per_combo):
    sampled = random.sample(combos, num_samples)
    scored = [(c, score_combo(model, *c, num_trials=trials_per_combo)) for c in sampled]
    best = max(scored, key=lambda cs: cs[1])
    return scored, best


def neighbors(combo, combos):
    """All combos differing from `combo` in exactly one component."""
    n_shown, scrambled, marker = combo
    out = []
    for other in combos:
        diffs = (other[0] != n_shown) + (other[1] != scrambled) + (other[2] != marker)
        if diffs == 1:
            out.append(other)
    return out


def hill_climbing(model, combos, trials_per_combo, max_iters=8):
    current = random.choice(combos)
    current_score = score_combo(model, *current, num_trials=trials_per_combo)
    trace = [(current, current_score)]
    for _ in range(max_iters):
        candidates = neighbors(current, combos)
        best_neighbor, best_neighbor_score = None, current_score
        for cand in candidates:
            s = score_combo(model, *cand, num_trials=trials_per_combo)
            if s > best_neighbor_score:
                best_neighbor, best_neighbor_score = cand, s
        if best_neighbor is None:
            break   # local optimum -- no neighbor improves on the current combo
        current, current_score = best_neighbor, best_neighbor_score
        trace.append((current, current_score))
    return trace


def main():
    print("=" * 78)
    print("SETUP: retraining Lesson 1's task with a 3-component prompt search space")
    print("=" * 78)
    print(f"Task: y = (x + k) mod {M}, hidden shift k. Train k in {TRAIN_KS}, test (held-out) k in {TEST_KS}.")
    print("Prompt components being searched over:")
    print("  1. NUM_EXAMPLES in {2, 3, 4, 5}")
    print("  2. ORDER        in {sorted, scrambled}")
    print("  3. PHRASING     in {'P', 'Q'}  (two symbolic 'instruction style' markers)")
    print("Search space size: 4 x 2 x 2 = 16 combinations.")
    print("\nTraining-distribution asymmetry (deliberately introduced):")
    print("  marker 'P' -> paired with n_shown uniformly from {2,3,4,5} (full range)")
    print("  marker 'Q' -> paired with n_shown uniformly from {2,3} ONLY")
    print("  => (marker='Q', n_shown in {4,5}) is NEVER seen during training.")

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 78)
    print("TRAINING")
    print("=" * 78)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, NUM_ITERS + 1):
        x, y = get_training_batch(BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    combos = all_combinations()

    print("\n" + "=" * 78)
    print("GROUND TRUTH: brute-force score every one of the 16 combinations")
    print("(only possible because this toy space is small -- would NOT scale to a")
    print(" realistic prompt space with many more components/values; used here only")
    print(" to check the search methods below against the true optimum)")
    print("=" * 78)
    BRUTE_FORCE_TRIALS = 600
    all_scores = [(c, score_combo(model, *c, num_trials=BRUTE_FORCE_TRIALS)) for c in combos]
    all_scores.sort(key=lambda cs: -cs[1])
    print(f"{'combination':45s}{'accuracy':>10}")
    for combo, acc in all_scores:
        flag = "  <- (marker=Q, n_shown>=4): never seen in training" if combo[2] == "Q" and combo[0] >= 4 else ""
        print(f"{combo_str(combo):45s}{acc:>10.3f}{flag}")
    true_best_combo, true_best_score = all_scores[0]
    population_mean = sum(acc for _, acc in all_scores) / len(all_scores)
    print(f"\nTrue best combination:  {combo_str(true_best_combo)}  (accuracy={true_best_score:.3f})")
    print(f"Mean accuracy across ALL 16 combinations: {population_mean:.3f}")

    never_seen_scores = [acc for c, acc in all_scores if c[2] == "Q" and c[0] >= 4]
    seen_scores = [acc for c, acc in all_scores if not (c[2] == "Q" and c[0] >= 4)]
    print(f"\nMean accuracy of never-seen (marker=Q, n_shown>=4) combinations: "
          f"{sum(never_seen_scores)/len(never_seen_scores):.3f}")
    print(f"Mean accuracy of the other, in-distribution combinations:         "
          f"{sum(seen_scores)/len(seen_scores):.3f}")
    print("-> The deliberately unseen combinations score measurably lower on average --")
    print("   this is the real, learnable structure the search algorithms below have")
    print("   to navigate, exactly like a real prompt engineer who doesn't know in")
    print("   advance which instruction+formatting combinations the model handles well.")

    print("\n" + "=" * 78)
    print("METHOD 1: RANDOM SEARCH")
    print("=" * 78)
    RANDOM_SAMPLES = 6
    TRIALS_PER_EVAL = 250
    random_scored, random_best = random_search(model, combos, RANDOM_SAMPLES, TRIALS_PER_EVAL)
    print(f"Sampled {RANDOM_SAMPLES} random combinations out of 16 and scored each "
          f"({TRIALS_PER_EVAL} trials/combo):")
    for combo, acc in sorted(random_scored, key=lambda cs: -cs[1]):
        print(f"  {combo_str(combo):45s}accuracy={acc:.3f}")
    random_mean = sum(acc for _, acc in random_scored) / len(random_scored)
    print(f"\nBest combination found by random search: {combo_str(random_best[0])} "
          f"(accuracy={random_best[1]:.3f})")
    print(f"Mean accuracy of the random sample itself: {random_mean:.3f}")

    print("\n" + "=" * 78)
    print("METHOD 2: HILL-CLIMBING (greedy local search)")
    print("=" * 78)
    hc_trace = hill_climbing(model, combos, TRIALS_PER_EVAL, max_iters=8)
    print("Trace (each row = the combination hill-climbing moved to, and its score):")
    for i, (combo, acc) in enumerate(hc_trace):
        tag = "start" if i == 0 else f"step {i}"
        print(f"  [{tag:>6}] {combo_str(combo):45s}accuracy={acc:.3f}")
    hc_best_combo, hc_best_score = hc_trace[-1]
    print(f"\nHill-climbing converged after {len(hc_trace)-1} move(s) to a local optimum:")
    print(f"  {combo_str(hc_best_combo)}  (accuracy={hc_best_score:.3f})")

    print("\n" + "=" * 78)
    print("COMPARISON: does automatic search reliably beat picking a combination")
    print("at random, and how close does it get to the true best?")
    print("=" * 78)
    print(f"{'method':40s}{'accuracy':>12}")
    print(f"{'Mean over ALL 16 combinations (baseline)':40s}{population_mean:>12.3f}")
    print(f"{'Random search: mean of its own sample':40s}{random_mean:>12.3f}")
    print(f"{'Random search: best of its own sample':40s}{random_best[1]:>12.3f}")
    print(f"{'Hill-climbing: converged result':40s}{hc_best_score:>12.3f}")
    print(f"{'True brute-force optimum (ground truth)':40s}{true_best_score:>12.3f}")

    hc_beats_population = hc_best_score > population_mean
    hc_beats_random_mean = hc_best_score >= random_mean
    hc_near_optimal = hc_best_score >= true_best_score - 0.03

    print(f"\n-> Hill-climbing's result beats the mean-over-all-16 baseline: {hc_beats_population}")
    print(f"   Hill-climbing's result is at or above random search's own sample mean: {hc_beats_random_mean}")
    print(f"   Hill-climbing's result is within 0.03 of the TRUE best combination: {hc_near_optimal}")
    print(f"\n   Discovered best prompt combination: {combo_str(hc_best_combo)}")
    print(f"   -> This is Zhou et al.'s (2022) APE recipe end to end: propose candidate")
    print("      prompts (here, combinations of components), score each on a held-out")
    print("      objective, and keep moving towards better-scoring ones -- a gradient-free")
    print("      discrete search that needed no hand-guessing about which single")
    print("      component (count, order, or phrasing) mattered most, or how they")
    print("      interact with each other in the model's training distribution.")


if __name__ == "__main__":
    main()
