"""
KV Cache and Speculative Decoding

Three demos:
  1. A tiny decoder-only Transformer (same recipe as Phase 02 Lesson 6),
     briefly trained, then generated from TWO ways: a naive loop that
     recomputes attention over the whole sequence at every step, and a
     real KV-CACHED loop that caches and reuses K/V tensors. We verify
     both produce byte-for-byte IDENTICAL token sequences (the cache is
     a pure speed optimization, not an approximation), and we count the
     actual number of attended query-key pairs each approach computes,
     showing the O(T) vs O(T^2)-per-step growth pattern directly in the
     numbers (not just wall-clock, which is noisy at this toy scale).
  2. A toy speculative-decoding simulation: a "draft" stand-in that
     agrees with a "target" ground-truth sequence with some per-token
     probability p, measuring the real reduction in expensive
     target-model calls needed to generate a fixed-length sequence.

Runtime: well under a minute on CPU (a ~200-step training run on a tiny
character corpus, plus a few small simulations).

Run:
    python example.py
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 0. A tiny toy corpus and character-level tokenizer (same recipe as
#    Phase 02 Lesson 6, kept small so training finishes in seconds).
# ---------------------------------------------------------------------------

CORPUS = """
the quick brown fox jumps over the lazy dog.
the lazy dog barks at the quick brown fox.
the fox runs into the dark forest at night.
""".strip().lower()
CORPUS = (CORPUS + "\n") * 12

chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(CORPUS), dtype=torch.long)

D_MODEL = 32
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 2
BLOCK_SIZE = 48
BATCH_SIZE = 16
NUM_ITERS = 250
LEARNING_RATE = 3e-3


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


# ---------------------------------------------------------------------------
# 1. A decoder-only Transformer whose attention layer can OPTIONALLY use a
#    KV cache. Passing cache=None reproduces Phase 02 Lesson 6's ordinary
#    full-sequence causal self-attention exactly. Passing a cache makes it
#    process only the newly appended token(s), attending against cached K/V.
# ---------------------------------------------------------------------------

class CausalSelfAttentionCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, t, batch, T):
        return t.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, counter, cache=None):
        """x: (batch, T_new, d_model) -- the NEW tokens only when a cache is
        supplied (T_new is usually 1 during cached generation), or the FULL
        sequence so far when cache is None (the naive, recompute-everything
        path). Returns (output, updated_cache)."""
        batch, T_new, _ = x.shape
        Q = self._split_heads(self.W_q(x), batch, T_new)
        K_new = self._split_heads(self.W_k(x), batch, T_new)
        V_new = self._split_heads(self.W_v(x), batch, T_new)

        if cache is not None:
            K = torch.cat([cache["k"], K_new], dim=2)
            V = torch.cat([cache["v"], V_new], dim=2)
        else:
            K, V = K_new, V_new
        T_k = K.shape[2]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)   # (batch, heads, T_new, T_k)
        if cache is None:
            # No cache means x holds the WHOLE sequence so far -> need the usual
            # causal mask so position i only attends to keys 0..i.
            mask = torch.tril(torch.ones(T_new, T_k)).bool()
            scores = scores.masked_fill(~mask, float("-inf"))
            valid_pairs = T_new * (T_new + 1) // 2   # sum_{i=0}^{T_new-1} (i+1)
        else:
            # With a cache, x holds ONLY the brand-new token(s): every key already
            # in the cache (plus the new one) is causally valid, no mask needed.
            valid_pairs = T_new * T_k

        counter["pairs"] += batch * self.num_heads * valid_pairs

        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, T_new, self.num_heads * self.d_k)
        return self.W_o(out), {"k": K, "v": V}


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionCache(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, counter, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), counter, cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward_full(self, token_ids, counter):
        """Naive path: recompute causal self-attention over the ENTIRE sequence,
        exactly Phase 02 Lesson 6's forward pass, no cache at all."""
        batch, T = token_ids.shape
        positions = torch.arange(T)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x, _ = block(x, counter, cache=None)
        x = self.final_norm(x)
        return self.output_head(x)   # (batch, T, vocab_size)

    def forward_step(self, new_token_id, position, cache_list, counter):
        """Cached path: process ONLY the single new token, reusing and
        extending cache_list (one {'k','v'} dict per layer)."""
        pos_tensor = torch.tensor([position])
        x = self.token_embedding(new_token_id) + self.position_embedding(pos_tensor)
        new_cache_list = []
        for block, cache in zip(self.blocks, cache_list):
            x, new_cache = block(x, counter, cache=cache)
            new_cache_list.append(new_cache)
        x = self.final_norm(x)
        return self.output_head(x), new_cache_list   # logits: (batch, 1, vocab_size)

    def train_loss(self, token_ids, targets):
        counter = {"pairs": 0}   # not part of the inference-cost measurement
        logits = self.forward_full(token_ids, counter)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


@torch.no_grad()
def generate_naive(model, start_ids, max_new_tokens, counter):
    """Recompute the WHOLE sequence's attention from scratch at every step --
    the naive loop from Phase 02 Lesson 6, instrumented to count attention work."""
    token_ids = start_ids.clone()
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.block_size:]
        logits = model.forward_full(context, counter)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy: deterministic
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids


@torch.no_grad()
def generate_cached(model, start_ids, max_new_tokens, counter):
    """Build the KV cache by feeding the prompt in one token at a time (this
    keeps causal masking trivially correct: each token only ever sees
    already-cached, earlier tokens), then generate by extending the cache
    one new token per step -- no recomputation of earlier positions."""
    batch, T0 = start_ids.shape
    cache_list = [None] * model.num_layers
    token_ids = start_ids.clone()
    logits = None
    for pos in range(T0):
        tok = start_ids[:, pos:pos + 1]
        logits, cache_list = model.forward_step(tok, pos, cache_list, counter)
    for _ in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_token], dim=1)
        pos = token_ids.shape[1] - 1
        logits, cache_list = model.forward_step(next_token, pos, cache_list, counter)
    return token_ids


def kv_cache_demo():
    print("=" * 70)
    print("1. KV CACHE: IDENTICAL OUTPUT, LESS REDUNDANT ATTENTION COMPUTE")
    print("=" * 70)

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    print(f"Tiny decoder-only model: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{NUM_LAYERS} layers, {NUM_HEADS} heads")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"\nTraining briefly ({NUM_ITERS} steps) so generation isn't pure noise...")
    for step in range(1, NUM_ITERS + 1):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        loss = model.train_loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")
    model.eval()

    start_ids = torch.tensor([encode("the ")], dtype=torch.long)
    max_new_tokens = 40

    counter_naive = {"pairs": 0}
    tokens_naive = generate_naive(model, start_ids, max_new_tokens, counter_naive)

    counter_cached = {"pairs": 0}
    tokens_cached = generate_cached(model, start_ids, max_new_tokens, counter_cached)

    identical = torch.equal(tokens_naive, tokens_cached)
    print(f"\nNaive generation output:  {decode(tokens_naive[0].tolist())!r}")
    print(f"Cached generation output: {decode(tokens_cached[0].tolist())!r}")
    print(f"\nOutputs are byte-for-byte IDENTICAL: {identical}")
    assert identical, "KV cache changed the output -- this would be a bug in the cache logic!"
    print("-> Confirms the KV cache is a pure speed optimization: same weights, same greedy")
    print("   decoding rule, same tokens out -- caching never changes WHAT the model computes,")
    print("   only how much redundant work it takes to compute it.")

    print(f"\nTotal attended query-key pairs (summed over all layers, heads, and generation steps):")
    print(f"  naive (recompute every step):  {counter_naive['pairs']:>10,}")
    print(f"  KV-cached:                     {counter_cached['pairs']:>10,}")
    print(f"  ratio (naive / cached):        {counter_naive['pairs'] / counter_cached['pairs']:>10.1f}x")

    # Show the growth pattern directly: per-step attended-pair counts for a
    # longer generation, isolated to ONE layer/head's worth of work so the
    # O(T) vs O(T^2)-per-step shapes are visible without the constant
    # layer/head multiplier obscuring them.
    print("\nPer-step attended-pair counts for a single layer+head (context length T -> pairs")
    print("computed AT THAT STEP), showing the growth pattern as generation proceeds:")
    print(f"{'context length T':>18}{'naive: T*(T+1)/2':>22}{'cached: T':>14}")
    for T in [5, 10, 20, 40, 80]:
        naive_pairs_this_step = T * (T + 1) // 2
        cached_pairs_this_step = T
        print(f"{T:>18}{naive_pairs_this_step:>22,}{cached_pairs_this_step:>14,}")
    print("\n-> Naive per-step cost grows QUADRATICALLY with context length (it re-scores the")
    print("   entire prefix every time); cached per-step cost grows only LINEARLY (one new")
    print("   query against the existing keys). That per-step gap is exactly why the KV")
    print("   cache is the single most important systems optimization for long generations.")


# ---------------------------------------------------------------------------
# 2. Speculative decoding: draft proposes, target verifies in one parallel
#    call, longest correct prefix is accepted. Measured as a Monte Carlo
#    simulation over a fixed-length target ground-truth sequence.
# ---------------------------------------------------------------------------

def simulate_speculative_decoding(seq_len, draft_agreement_p, draft_tokens_per_round, num_trials=500):
    """Returns the AVERAGE number of expensive target-model calls needed to
    generate a sequence of length seq_len, using speculative decoding.

    Each round: the draft model proposes `draft_tokens_per_round` candidate
    tokens; each candidate independently matches the target's true next token
    with probability draft_agreement_p (a stand-in for how well-aligned the
    small draft model is with the large target model). The target verifies
    the whole batch of candidates in ONE forward pass (one target-model call
    per round, regardless of how many candidates it verifies at once):
    accept the longest correct prefix, then the target itself supplies the
    next token (either the correction after the first wrong guess, or, if
    every candidate was right, one bonus token for free) before the next round."""
    total_calls = 0
    for _ in range(num_trials):
        pos = 0
        calls = 0
        while pos < seq_len:
            k = min(draft_tokens_per_round, seq_len - pos)
            # How many of the k draft guesses are correct, read left to right,
            # until the first wrong one (longest correct PREFIX only).
            accepted = 0
            for _ in range(k):
                if random.random() < draft_agreement_p:
                    accepted += 1
                else:
                    break
            calls += 1   # exactly one target verification pass per round
            # accepted tokens are confirmed correct; the target also supplies
            # one more token this round (a correction if a guess was wrong, or
            # a free bonus token if every guess in the round was accepted).
            pos += accepted + 1
        total_calls += calls
    return total_calls / num_trials


def speculative_decoding_demo():
    print("\n" + "=" * 70)
    print("2. SPECULATIVE DECODING: FEWER EXPENSIVE TARGET-MODEL CALLS")
    print("=" * 70)

    seq_len = 60
    draft_tokens_per_round = 4
    print(f"Generating a fixed sequence of length {seq_len}.")
    print(f"WITHOUT speculative decoding, the target model is called once per token:")
    print(f"  target-model calls = {seq_len} (always, by definition)\n")

    print(f"WITH speculative decoding (draft proposes {draft_tokens_per_round} tokens/round, "
          f"target verifies all {draft_tokens_per_round} in ONE call):")
    print(f"{'draft/target agreement p':>26}{'avg target calls':>20}{'reduction vs no-spec':>24}")
    for p in [0.3, 0.5, 0.7, 0.9]:
        avg_calls = simulate_speculative_decoding(seq_len, p, draft_tokens_per_round)
        reduction = (1 - avg_calls / seq_len) * 100
        print(f"{p:>26.1f}{avg_calls:>20.1f}{reduction:>23.1f}%")

    print("\n-> Every row generates the SAME sequence length, using the SAME target model")
    print("   verification rule, so output quality is unaffected -- only the low-agreement")
    print("   draft (p=0.3) barely beats plain one-token-at-a-time decoding, while a")
    print("   well-aligned draft (p=0.9) lets the target confirm several tokens per call,")
    print("   cutting the number of expensive target-model forward passes substantially.")
    print("   This is exactly why speculative decoding's speedup depends entirely on how")
    print("   well the cheap draft model's guesses track the expensive target model.")


def main():
    kv_cache_demo()
    speculative_decoding_demo()


if __name__ == "__main__":
    main()
