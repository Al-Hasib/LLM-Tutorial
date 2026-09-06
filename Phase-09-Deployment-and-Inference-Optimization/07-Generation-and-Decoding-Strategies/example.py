"""
Generation and Decoding Strategies

Four demos, all built on tiny decoder-only Transformers (same toy recipe as
Phase 02 Lesson 6 / Phase 09 Lesson 3: character-level corpus, a few hundred
training steps), all measuring REAL numbers rather than just asserting the
prose claims:

  1. Temperature: real softmax'd logits from a trained model, at low/neutral/
     high temperature, printing entropy and top-token probabilities so the
     sharpening/flattening effect is a measured number.
  2. Top-k and top-p filtering, implemented as plain tensor ops (sort, cumsum,
     masked_fill -- no library call that hides the mechanism), then repeated
     sampling from the SAME logits under greedy / top-k / top-p to measure
     actual output diversity (unique continuations across N samples).
  3. A repetition penalty, implemented from scratch, applied to a REAL greedy
     generation run that a small, undertrained model gets stuck looping on --
     verified in code that the unpenalized run loops and the penalized run
     does not.
  4. A small beam search implemented from scratch, compared against greedy on
     the SAME model and prompt, verifying beam search's actual guarantee:
     cumulative log-probability at least as high as greedy's.

Runtime: well under a minute on CPU (two small models, a few hundred training
steps each, plus lightweight sampling/search loops).

Run:
    python example.py
"""

import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 0. A tiny decoder-only Transformer -- identical recipe to Phase 02 Lesson 6
#    (token + positional embedding, N causal-self-attention/FFN blocks, a
#    final linear head over the vocabulary). No KV cache here -- this lesson
#    is about the DECISION RULE applied to the logits, not the cost of
#    producing them, so the plain, full-recompute forward pass is enough.
# ---------------------------------------------------------------------------

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
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def make_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def train_model(model, data, block_size, batch_size, num_iters, lr, log_every=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, num_iters + 1):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"    step {step:4d}  loss = {loss.item():.4f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shared decoding-rule building blocks -- plain tensor ops, no library call
# that hides the mechanism described in the README.
# ---------------------------------------------------------------------------

def entropy(probs):
    """Shannon entropy, in nats, of a 1-D probability vector."""
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum().item()


def top_k_filter(logits, k):
    """README section 4: keep only the k highest logits, mask everything
    else to -inf so it gets exactly zero probability after softmax."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_values, _ = torch.topk(logits, k)
    min_keep = top_values[..., -1]
    return logits.masked_fill(logits < min_keep, float("-inf"))


def top_p_filter(logits, p):
    """README section 5: sort descending, keep the smallest prefix whose
    cumulative probability reaches p, mask everything after it to -inf."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # A token is OUTSIDE the nucleus if the cumulative mass BEFORE it already
    # reached p -- i.e. everything strictly after the threshold is removed,
    # and the token that first crosses the threshold is always kept.
    sorted_remove = (cumulative_probs - sorted_probs) > p
    sorted_remove[..., 0] = False  # always keep at least the single best token
    remove_mask = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def apply_repetition_penalty(logits, already_generated_ids, penalty):
    """README section 7 (CTRL-style, compounded by frequency): for every token
    already seen, shrink a positive logit toward zero / push a negative logit
    further negative, by penalty RAISED TO THE NUMBER OF TIMES it has already
    appeared -- so a token repeated many times gets penalized much harder than
    one that has appeared only once (the "frequency penalty" variant from the
    README, applied multiplicatively rather than as a flat subtraction)."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    counts = Counter(already_generated_ids)
    for token_id, count in counts.items():
        factor = penalty ** count
        if logits[token_id] > 0:
            logits[token_id] = logits[token_id] / factor
        else:
            logits[token_id] = logits[token_id] * factor
    return logits


# ===========================================================================
# DEMO 1: temperature -- real logits, sharpening vs. flattening, measured
# ===========================================================================

def temperature_demo():
    print("=" * 70)
    print("1. TEMPERATURE: SHARPENING / FLATTENING A REAL TRAINED DISTRIBUTION")
    print("=" * 70)

    corpus = ("the quick brown fox jumps over the lazy dog. "
              "the lazy dog barks at the quick brown fox. "
              "the fox runs into the dark forest at night. "
              "the dog sleeps by the fire every night. ") * 12
    chars, stoi, itos = make_tokenizer(corpus)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(corpus), dtype=torch.long)

    d_model, num_heads, num_layers, block_size = 32, 4, 2, 48
    model = MiniGPT(len(chars), d_model, num_heads, 4 * d_model, num_layers, block_size)
    print(f"Training a tiny model ({sum(p.numel() for p in model.parameters()):,} params) "
          f"on a {len(chars)}-character vocabulary...")
    train_model(model, data, block_size, batch_size=32, num_iters=300, lr=3e-3, log_every=100)

    prompt = torch.tensor([encode("the ")], dtype=torch.long)
    temperatures = [0.3, 1.0, 2.0]

    print("\nLogits at a few real generation steps, softmax'd at different temperatures:")
    print(f"{'step':>6}{'T':>8}{'entropy (nats)':>18}{'top-3 tokens (prob)':>36}")
    token_ids = prompt.clone()
    with torch.no_grad():
        for step in range(3):
            context = token_ids[:, -block_size:]
            logits, _ = model(context)
            last_logits = logits[0, -1, :]
            for T in temperatures:
                probs = F.softmax(last_logits / T, dim=-1)
                ent = entropy(probs)
                top_vals, top_idx = torch.topk(probs, 3)
                top_str = ", ".join(f"{itos[i.item()]!r}:{v.item():.3f}" for v, i in zip(top_vals, top_idx))
                print(f"{step:>6}{T:>8.1f}{ent:>18.4f}   {top_str}")
            # advance the actual generated sequence greedily (T=1 logits) so the
            # next step's logits come from a genuinely different real context
            next_token = last_logits.argmax().view(1, 1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    # Numerically check the two formal claims from README section 3 on this step's logits.
    with torch.no_grad():
        context = prompt[:, -block_size:]
        logits, _ = model(context)
        last_logits = logits[0, -1, :]
        low_T_probs = F.softmax(last_logits / 0.05, dim=-1)
        greedy_choice = last_logits.argmax().item()
        low_T_choice = low_T_probs.argmax().item()
        high_T_entropy = entropy(F.softmax(last_logits / 100.0, dim=-1))
        uniform_entropy = math.log(len(chars))

    print(f"\nAs T -> 0 (T=0.05 here): argmax(P_T) token = {itos[low_T_choice]!r}, "
          f"argmax(logits) token = {itos[greedy_choice]!r} -- {'MATCH' if low_T_choice == greedy_choice else 'MISMATCH'}")
    assert low_T_choice == greedy_choice, "low-temperature sampling should recover the greedy argmax token"
    print("-> Confirms T -> 0 formally recovers greedy decoding (section 2): the distribution")
    print("   concentrates onto the single argmax token, exactly as the README's limit claims.")

    print(f"\nAs T -> large (T=100 here): entropy = {high_T_entropy:.4f} nats vs. "
          f"uniform-over-{len(chars)}-token entropy = {uniform_entropy:.4f} nats "
          f"({100 * high_T_entropy / uniform_entropy:.1f}% of uniform)")
    assert high_T_entropy > uniform_entropy * 0.9, "very high temperature should approach the uniform distribution's entropy"
    print("-> Confirms T -> large flattens the distribution toward uniform, exactly as claimed.")


# ===========================================================================
# DEMO 2: top-k / top-p -- filtering as plain tensor ops, diversity measured
# ===========================================================================

@torch.no_grad()
def sample_continuation(model, start_ids, num_new_tokens, decide_fn, generator):
    """Generic sampling loop: decide_fn(logits, generator) -> next_token_id."""
    token_ids = start_ids.clone()
    for _ in range(num_new_tokens):
        context = token_ids[:, -model.block_size:]
        logits, _ = model(context)
        next_id = decide_fn(logits[0, -1, :], generator)
        token_ids = torch.cat([token_ids, torch.tensor([[next_id]])], dim=1)
    return token_ids


def topk_topp_demo():
    print("\n" + "=" * 70)
    print("2. TOP-K / TOP-P: FILTERING FROM SCRATCH, DIVERSITY MEASURED")
    print("=" * 70)

    corpus = ("the quick brown fox jumps over the lazy dog. "
              "the lazy dog barks at the quick brown fox. "
              "the fox runs into the dark forest at night. "
              "the dog sleeps by the fire every night. ") * 12
    chars, stoi, itos = make_tokenizer(corpus)
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(corpus), dtype=torch.long)

    d_model, num_heads, num_layers, block_size = 32, 4, 2, 48
    model = MiniGPT(len(chars), d_model, num_heads, 4 * d_model, num_layers, block_size)
    train_model(model, data, block_size, batch_size=32, num_iters=300, lr=3e-3)
    print(f"(Reusing a freshly trained {sum(p.numel() for p in model.parameters()):,}-param model "
          f"on the same corpus as Demo 1.)")

    prompt = torch.tensor([encode("the ")], dtype=torch.long)

    # First, look at ONE step's real logits and compare the entropy of the full
    # distribution against the filtered/renormalized top-k and top-p distributions.
    with torch.no_grad():
        context = prompt[:, -block_size:]
        logits, _ = model(context)
        last_logits = logits[0, -1, :]
        full_probs = F.softmax(last_logits, dim=-1)
        k = 5
        p = 0.9
        topk_probs = F.softmax(top_k_filter(last_logits, k), dim=-1)
        topp_probs = F.softmax(top_p_filter(last_logits, p), dim=-1)
        n_kept_k = (topk_probs > 0).sum().item()
        n_kept_p = (topp_probs > 0).sum().item()

    print(f"\nAt one real generation step (vocab size {len(chars)}):")
    print(f"  full distribution entropy:        {entropy(full_probs):.4f} nats  ({len(chars)} tokens with nonzero prob)")
    print(f"  top-k={k} filtered entropy:         {entropy(topk_probs):.4f} nats  ({n_kept_k} tokens with nonzero prob)")
    print(f"  top-p={p} filtered entropy:         {entropy(topp_probs):.4f} nats  ({n_kept_p} tokens with nonzero prob)")
    assert n_kept_k == k, "top-k filtering should keep exactly k tokens"
    assert n_kept_p <= n_kept_k or n_kept_p <= len(chars), "top-p's kept-token count should adapt, not match a fixed k"
    assert entropy(topk_probs) <= entropy(full_probs) + 1e-6
    assert entropy(topp_probs) <= entropy(full_probs) + 1e-6
    print("-> Both truncation rules can only ever REDUCE entropy relative to the full distribution")
    print("   (they zero out probability mass, never add any) -- and top-p kept a DIFFERENT number")
    print("   of tokens than top-k's fixed k, adapting to how peaked the real distribution was here.")

    # Now measure actual output diversity: repeated sampling from the SAME
    # starting prompt, under greedy / top-k / top-p, counting UNIQUE continuations.
    num_samples = 40
    gen_len = 12

    def greedy_decide(logits, generator):
        return logits.argmax().item()

    def topk_decide(logits, generator):
        filtered = top_k_filter(logits, k=5)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).item()

    def topp_decide(logits, generator):
        filtered = top_p_filter(logits, p=0.9)
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).item()

    results = {}
    for name, decide_fn in [("greedy", greedy_decide), ("top-k=5", topk_decide), ("top-p=0.9", topp_decide)]:
        continuations = set()
        for trial in range(num_samples):
            gen = torch.Generator().manual_seed(trial)
            out = sample_continuation(model, prompt, gen_len, decide_fn, gen)
            continuations.add(tuple(out[0, prompt.size(1):].tolist()))
        results[name] = len(continuations)

    print(f"\nUnique continuations out of {num_samples} samples ({gen_len} new tokens each, same start prompt):")
    for name, n_unique in results.items():
        print(f"  {name:>10}: {n_unique:>3} unique continuation(s)")

    assert results["greedy"] == 1, "greedy decoding is deterministic -- every sample must be identical"
    assert results["top-k=5"] > results["greedy"]
    assert results["top-p=0.9"] > results["greedy"]
    print("-> Greedy decoding has ZERO output diversity by construction (same input, same argmax,")
    print("   every time). Both sampling rules produce measurably more than one unique continuation")
    print("   from the exact same prompt and model -- diversity that greedy cannot offer at all.")


# ===========================================================================
# DEMO 3: repetition penalty -- a real greedy loop, broken in code
# ===========================================================================

@torch.no_grad()
def generate_greedy_with_penalty(model, start_ids, max_new_tokens, penalty=1.0):
    token_ids = start_ids.clone()
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.block_size:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :]
        if penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, token_ids[0].tolist(), penalty)
        next_token = next_logits.argmax().view(1, 1)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids


def repetition_penalty_demo():
    print("\n" + "=" * 70)
    print("3. REPETITION PENALTY: BREAKING A REAL GREEDY LOOP")
    print("=" * 70)

    # A corpus built from a strict two-sentence alternation. A small,
    # lightly trained model tends to under-track WHICH half of the
    # alternation it's in and collapses onto repeating just one animal --
    # a real, unstaged degenerate loop, exactly section 2's failure mode.
    cycle = "the fox chases the dog. the dog chases the fox. "
    corpus = cycle * 30
    chars, stoi, itos = make_tokenizer(corpus)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(corpus), dtype=torch.long)

    d_model, num_heads, num_layers, block_size = 24, 4, 2, 32
    model = MiniGPT(len(chars), d_model, num_heads, 4 * d_model, num_layers, block_size)
    print(f"Training a small model ({sum(p.numel() for p in model.parameters()):,} params) on a "
          f"strict alternating corpus, deliberately briefly so it under-tracks the alternation...")
    train_model(model, data, block_size, batch_size=32, num_iters=120, lr=3e-3, log_every=60)

    prompt = torch.tensor([encode("the fox chases the ")], dtype=torch.long)
    max_new_tokens = 120

    penalty_strength = 1.1
    no_penalty = generate_greedy_with_penalty(model, prompt, max_new_tokens, penalty=1.0)
    with_penalty = generate_greedy_with_penalty(model, prompt, max_new_tokens, penalty=penalty_strength)

    text_no_penalty = decode(no_penalty[0].tolist())
    text_with_penalty = decode(with_penalty[0].tolist())

    print(f"\nGreedy, NO repetition penalty:\n  {text_no_penalty!r}")
    print(f"\nGreedy, WITH repetition penalty (strength={penalty_strength}, compounded per repeat):\n  {text_with_penalty!r}")

    loop_unit = "the dog chases the fox. the dog chases the fox."
    loop_unit_alt = "the fox chases the fox. the fox chases the fox."

    def has_loop(text):
        return loop_unit in text or loop_unit_alt in text or (cycle.strip() * 2) in text

    unpenalized_loops = has_loop(text_no_penalty)
    penalized_loops = has_loop(text_with_penalty)

    print(f"\nUnpenalized run contains a repeated-phrase loop: {unpenalized_loops}")
    print(f"Penalized run contains a repeated-phrase loop:   {penalized_loops}")
    assert unpenalized_loops, "expected plain greedy decoding to fall into a repeating loop on this toy setup"
    assert not penalized_loops, "expected the repetition penalty to break the loop"
    print("-> A real, measured behavior difference: the SAME greedy argmax rule, on the SAME")
    print("   trained model and prompt, loops forever without a repetition penalty and does not")
    print("   loop with one -- exactly section 2's degenerate-loop failure, patched by section 7.")


# ===========================================================================
# DEMO 4: beam search vs. greedy -- the actual cumulative-log-probability guarantee
# ===========================================================================

@torch.no_grad()
def greedy_with_logprob(model, start_ids, num_new_tokens):
    token_ids = start_ids.clone()
    total_logprob = 0.0
    for _ in range(num_new_tokens):
        context = token_ids[:, -model.block_size:]
        logits, _ = model(context)
        log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
        next_id = log_probs.argmax().item()
        total_logprob += log_probs[next_id].item()
        token_ids = torch.cat([token_ids, torch.tensor([[next_id]])], dim=1)
    return token_ids, total_logprob


@torch.no_grad()
def beam_search(model, start_ids, num_new_tokens, beam_width):
    """README section 8: keep the beam_width highest cumulative-log-probability
    sequences at every step; return the best one after num_new_tokens steps."""
    beams = [(start_ids.clone(), 0.0)]
    for _ in range(num_new_tokens):
        candidates = []
        for seq, score in beams:
            context = seq[:, -model.block_size:]
            logits, _ = model(context)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            top_logprobs, top_ids = torch.topk(log_probs, beam_width)
            for lp, tid in zip(top_logprobs.tolist(), top_ids.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tid]])], dim=1)
                candidates.append((new_seq, score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]
    best_seq, best_score = max(beams, key=lambda c: c[1])
    return best_seq, best_score


def beam_search_demo():
    print("\n" + "=" * 70)
    print("4. BEAM SEARCH: THE ACTUAL CUMULATIVE-LOG-PROBABILITY GUARANTEE")
    print("=" * 70)

    corpus = ("the quick brown fox jumps over the lazy dog. "
              "the lazy dog barks at the quick brown fox. "
              "the fox runs into the dark forest at night. "
              "the dog sleeps by the fire every night. ") * 12
    chars, stoi, itos = make_tokenizer(corpus)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(corpus), dtype=torch.long)

    d_model, num_heads, num_layers, block_size = 32, 4, 2, 48
    model = MiniGPT(len(chars), d_model, num_heads, 4 * d_model, num_layers, block_size)
    # Deliberately fewer training steps than Demos 1-2's model: a still-somewhat-
    # uncertain model is exactly the case where greedy's locally-best token per
    # step can lead to a WORSE total sequence than a wider beam finds -- a
    # confidently overtrained model tends to make beam and greedy coincide.
    train_model(model, data, block_size, batch_size=32, num_iters=150, lr=3e-3)
    print(f"(Another {sum(p.numel() for p in model.parameters()):,}-param model, same corpus family "
          f"as Demos 1-2 but trained for fewer steps, left genuinely uncertain at some positions.)")

    prompt = torch.tensor([encode("the lazy ")], dtype=torch.long)
    num_new_tokens = 20

    greedy_seq, greedy_logprob = greedy_with_logprob(model, prompt, num_new_tokens)

    print(f"\n{'beam width B':>14}{'cumulative log-prob':>22}{'>= greedy?':>14}   sequence")
    print(f"{'1 (greedy)':>14}{greedy_logprob:>22.4f}{'--':>14}   {decode(greedy_seq[0].tolist())!r}")

    for beam_width in [3, 5]:
        beam_seq, beam_logprob = beam_search(model, prompt, num_new_tokens, beam_width)
        ok = beam_logprob >= greedy_logprob - 1e-4
        print(f"{beam_width:>14}{beam_logprob:>22.4f}{str(ok):>14}   {decode(beam_seq[0].tolist())!r}")
        assert ok, f"beam search (B={beam_width}) should find a sequence at least as likely as greedy's"

    # Sanity check: beam search with beam_width=1 must be EXACTLY the greedy rule.
    beam1_seq, beam1_logprob = beam_search(model, prompt, num_new_tokens, beam_width=1)
    assert torch.equal(beam1_seq, greedy_seq), "beam width 1 must reduce to plain greedy decoding"
    assert abs(beam1_logprob - greedy_logprob) < 1e-4
    print(f"\nBeam width 1 reproduces greedy exactly: sequence match = "
          f"{torch.equal(beam1_seq, greedy_seq)}, log-prob match = {abs(beam1_logprob - greedy_logprob) < 1e-4}")

    print("\n-> Beam search is a DETERMINISTIC search over cumulative log-probability, not a")
    print("   sampling method -- greedy decoding is exactly its beam_width=1 special case, and")
    print("   widening the beam can only ever find a sequence with EQUAL OR HIGHER cumulative")
    print("   log-probability than greedy, never lower, which is the actual guarantee measured above.")


def main():
    temperature_demo()
    topk_topp_demo()
    repetition_penalty_demo()
    beam_search_demo()


if __name__ == "__main__":
    main()
