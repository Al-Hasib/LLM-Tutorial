"""
Sequence-to-Sequence and Attention

Two from-scratch NumPy demos:
  1. Attention as a "soft, differentiable lookup" -- given a query, retrieve
     a weighted blend of values based on how well each key matches.
  2. A direct comparison of the Seq2Seq bottleneck (relying only on the
     final encoder hidden state) vs. attention (retrieving from every
     encoder hidden state) as sequence length grows.

Run:
    python example.py
"""

import numpy as np

rng = np.random.default_rng(0)


def softmax(scores):
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    return exps / exps.sum()


def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def attention(query, keys, values):
    """Dot-product attention: query (D,), keys (N, D), values (N, D_v).
    Returns (context vector, attention weights)."""
    scores = keys @ query                      # (N,) -- one score per key
    weights = softmax(scores)                   # (N,) -- sums to 1
    context = weights @ values                   # weighted sum of values
    return context, weights


# ---------------------------------------------------------------------------
# 1. Attention as a soft lookup / retrieval mechanism
# ---------------------------------------------------------------------------

def soft_lookup_demo():
    print("=" * 70)
    print("1. ATTENTION AS A SOFT, DIFFERENTIABLE LOOKUP")
    print("=" * 70)

    num_items = 6
    dim = 16
    keys = rng.normal(size=(num_items, dim))
    # Values are a separate, clearly-labeled payload per key so we can tell
    # exactly which value got retrieved.
    values = np.eye(num_items, dim)  # value i is (mostly) the one-hot-ish row i

    target_idx = 3
    query = keys[target_idx] + rng.normal(scale=0.1, size=dim)  # a noisy copy of key[3]

    context, weights = attention(query, keys, values)

    print(f"Query = a noisy copy of key[{target_idx}]")
    print("Attention weights over the 6 keys:")
    for i, w in enumerate(weights):
        marker = "  <-- target" if i == target_idx else ""
        print(f"  key[{i}]: {w:.4f}{marker}")

    sim_to_target_value = cosine_similarity(context, values[target_idx])
    print(f"\ncosine(context, value[{target_idx}]) = {sim_to_target_value:.4f}")
    print("-> Even though the query was never exactly equal to any key, attention")
    print("   correctly retrieves (almost) the right value by content similarity --")
    print("   this 'retrieve by matching content, not by fixed position' behavior")
    print("   is exactly why attention generalizes so much better than a fixed slot.")


# ---------------------------------------------------------------------------
# 2. The Seq2Seq bottleneck vs. attention, as sequence length grows
# ---------------------------------------------------------------------------

def encode_all_hidden_states(x_seq, Wxh, Whh, bh, hidden_dim):
    """A minimal vanilla-RNN encoder that returns EVERY hidden state,
    not just the final one -- this is what makes attention possible."""
    h = np.zeros(hidden_dim)
    hidden_states = []
    for x in x_seq:
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        hidden_states.append(h)
    return np.array(hidden_states)  # shape (T, hidden_dim)


def bottleneck_vs_attention_demo():
    print("\n" + "=" * 70)
    print("2. FIXED-CONTEXT BOTTLENECK vs. ATTENTION, AS SEQUENCE LENGTH GROWS")
    print("=" * 70)
    print("Setup: an important 'signal' occurs at position 0 of the source")
    print("sequence. We ask two different ways of building a 'context vector'")
    print("to recover information about that first position, as the sentence")
    print("(sequence length T) gets longer.\n")

    input_dim, hidden_dim = 8, 16
    scale = 1.0 / np.sqrt(hidden_dim)
    Wxh = rng.normal(0, scale, size=(hidden_dim, input_dim))
    Whh = rng.normal(0, scale, size=(hidden_dim, hidden_dim))
    bh = np.zeros(hidden_dim)

    max_len = 40
    signal = rng.normal(size=input_dim)
    noise_tokens = [rng.normal(size=input_dim) for _ in range(max_len - 1)]
    full_sequence = [signal] + noise_tokens

    print(f"{'seq_len':>8}  {'fixed-context sim':>19}  {'attention sim':>15}")
    for T in [2, 5, 10, 20, 30, 40]:
        x_seq = full_sequence[:T]
        hidden_states = encode_all_hidden_states(x_seq, Wxh, Whh, bh, hidden_dim)
        h_first, h_last = hidden_states[0], hidden_states[-1]

        # Fixed-context (classic Seq2Seq): only h_last survives to the decoder.
        fixed_context_sim = cosine_similarity(h_last, h_first)

        # Attention: the decoder can query ALL hidden states, including h_first
        # itself. Using h_first as the query is a stand-in for "the decoder
        # asks a question whose answer lives at position 0" -- dot-product
        # attention naturally scores a vector highest against itself.
        context, _ = attention(query=h_first, keys=hidden_states, values=hidden_states)
        attention_sim = cosine_similarity(context, h_first)

        print(f"{T:>8}  {fixed_context_sim:>19.4f}  {attention_sim:>15.4f}")

    print("\n-> Fixed-context similarity is low and noisy even for short sequences,")
    print("   and never recovers: with an untrained (random-weight) RNN, each new")
    print("   timestep's tanh recurrence mixes in enough new information that the")
    print("   final hidden state stops resembling h_first almost immediately --")
    print("   exactly the bottleneck problem described in the README, just visible")
    print("   even faster here than it would be in a fully trained network.")
    print("-> Attention similarity stays pinned near 1.0 regardless of T, because")
    print("   the decoder can always look directly back at h_first instead of")
    print("   relying on whatever survived being compressed into one final vector.")


def main():
    soft_lookup_demo()
    bottleneck_vs_attention_demo()


if __name__ == "__main__":
    main()
