"""
Pretraining Objectives

Takes ONE toy tokenized sentence and applies all four pretraining
masking recipes to it -- causal LM, masked LM, span corruption, and
prefix LM -- printing exactly which input each objective feeds the
model, what target it asks for at each position, and which positions
actually contribute to the loss.

To make the "same architecture, different loss-masking recipe" idea
completely concrete (and not just a printed table), each objective's
loss is computed for real with torch.nn.functional.cross_entropy over
RANDOM logits (there's no trained model here -- the point is the
masking mechanics, not the predictions) using ignore_index=-100.
Positions outside each recipe's loss mask come back with exactly 0.0
loss, which is the numerical proof that "no loss here" really means
no gradient signal from that position, not just an unused column in a
printed table.

Run:
    python example.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Toy sentence and vocabulary. Word-level tokens (not sub-word/BPE) purely
# so every position in the printed tables reads as an actual English word.
# ---------------------------------------------------------------------------

SENTENCE = "the cat sat on the mat while the dog slept".split()
#            0    1   2   3   4    5    6    7    8    9

EXTRA_TOKENS = ["[MASK]", "<X>", "<Y>", "<Z>", "<BOS>", "forest"]
vocab = sorted(set(SENTENCE)) + EXTRA_TOKENS
stoi = {tok: i for i, tok in enumerate(vocab)}
VOCAB_SIZE = len(vocab)


def ids(tokens):
    return torch.tensor([stoi[t] for t in tokens], dtype=torch.long)


def print_table(title, rows):
    """rows: list of (position, input_tok, target_tok_or_dash, loss_yesno, loss_value_or_dash)"""
    print(f"\n--- {title} ---")
    print(f"{'pos':>4}{'input':>10}{'predicts (target)':>20}{'in loss?':>14}{'loss value':>12}")
    for pos, inp, tgt, in_loss, loss_val in rows:
        print(f"{pos:>4}{inp:>10}{tgt:>20}{in_loss:>14}{loss_val:>12}")


def compute_losses(logits, target_ids):
    """logits: (T, vocab_size) random 'model outputs'. target_ids: (T,) with
    IGNORE_INDEX at every non-loss position. Returns per-position loss tensor
    -- PyTorch's cross_entropy returns exactly 0.0 at ignored positions."""
    return F.cross_entropy(logits, target_ids, ignore_index=IGNORE_INDEX, reduction="none")


# ---------------------------------------------------------------------------
# 1. Causal LM (Phase 02 Lesson 6): every position predicts the next token,
# strict causal mask, loss on every position.
# ---------------------------------------------------------------------------

def causal_lm_demo():
    input_tokens = SENTENCE[:-1]          # tokens 0..8
    target_tokens = SENTENCE[1:]          # tokens 1..9 (shifted by one)
    T = len(input_tokens)

    target_ids = ids(target_tokens)       # every position IS supervised
    logits = torch.randn(T, VOCAB_SIZE)
    losses = compute_losses(logits, target_ids)

    rows = [
        (i, input_tokens[i], target_tokens[i], "yes", f"{losses[i].item():.3f}")
        for i in range(T)
    ]
    print_table("1. CAUSAL LM  (decoder-only, e.g. GPT)", rows)
    print(f"    -> attention: causal mask (each position sees only positions <= itself)")
    print(f"    -> loss positions: {T}/{T}  (every position supervised)")
    return losses


# ---------------------------------------------------------------------------
# 2. Masked LM (BERT): corrupt ~some positions, predict their ORIGINAL
# identity from bidirectional context, loss ONLY on corrupted positions.
# Uses 3 masked positions here (higher than BERT's real ~15%) purely so a
# 10-word toy sentence can visibly demonstrate all three of BERT's masking
# sub-cases: [MASK] replacement, random-token replacement, and unchanged.
# ---------------------------------------------------------------------------

def masked_lm_demo():
    input_tokens = list(SENTENCE)         # full 10-token sequence, bidirectional
    target_tokens = [None] * len(input_tokens)

    masked_positions = {
        2: "[MASK]",   # 80%-case: replace with [MASK], predict original "sat"
        5: SENTENCE[5],  # 10%-case: leave unchanged ("mat"), STILL a loss position
        8: "forest",    # 10%-case: replace with a random other token, predict original "dog"
    }
    for pos, replacement in masked_positions.items():
        target_tokens[pos] = SENTENCE[pos]   # target = the ORIGINAL token, always
        input_tokens[pos] = replacement       # input = the (possibly corrupted) token

    T = len(input_tokens)
    target_ids = torch.tensor(
        [stoi[target_tokens[i]] if target_tokens[i] is not None else IGNORE_INDEX for i in range(T)]
    )
    logits = torch.randn(T, VOCAB_SIZE)
    losses = compute_losses(logits, target_ids)

    rows = [
        (
            i, input_tokens[i],
            target_tokens[i] if target_tokens[i] is not None else "-",
            "yes" if i in masked_positions else "no",
            f"{losses[i].item():.3f}",
        )
        for i in range(T)
    ]
    print_table("2. MASKED LM  (encoder-only, e.g. BERT)", rows)
    print(f"    -> attention: fully bidirectional (no mask at all)")
    print(f"    -> loss positions: {len(masked_positions)}/{T}  (only the corrupted positions)")
    return losses


# ---------------------------------------------------------------------------
# 3. Span corruption (T5): replace CONTIGUOUS spans with sentinels on the
# encoder side (no loss there); the decoder generates just the missing
# content, tagged by sentinel, as a short target sequence, with loss over
# every position of that (short) target.
# ---------------------------------------------------------------------------

def span_corruption_demo():
    # Encoder side: two contiguous spans removed and replaced by sentinels.
    # span A = positions 1-2 ("cat sat") -> <X>; span B = positions 7-8 ("the dog") -> <Y>
    encoder_input = ["the", "<X>", "on", "the", "mat", "while", "<Y>", "slept"]

    # Decoder side: teacher-forced input starts with <BOS>; target reconstructs
    # ONLY the missing spans, each tagged by the sentinel that replaced it,
    # terminated by a final sentinel <Z> marking "no more spans".
    decoder_input = ["<BOS>", "<X>", "cat", "sat", "<Y>", "the", "dog"]
    decoder_target = ["<X>", "cat", "sat", "<Y>", "the", "dog", "<Z>"]
    T = len(decoder_input)

    target_ids = ids(decoder_target)   # every decoder position IS supervised
    logits = torch.randn(T, VOCAB_SIZE)
    losses = compute_losses(logits, target_ids)

    print(f"\n--- 3. SPAN CORRUPTION  (encoder-decoder, e.g. T5) ---")
    print(f"    encoder input (bidirectional, NO loss): {' '.join(encoder_input)}")
    rows = [
        (i, decoder_input[i], decoder_target[i], "yes", f"{losses[i].item():.3f}")
        for i in range(T)
    ]
    print_table("decoder side (causal, cross-attends to encoder)", rows)
    print(f"    -> loss positions: {T}/{T} decoder positions "
          f"(but target is only {T} tokens vs. the original 10-token sentence --")
    print(f"       reconstructing just the missing spans is cheaper than reconstructing everything)")
    return losses


# ---------------------------------------------------------------------------
# 4. Prefix LM (UniLM / PaLM-style): bidirectional over a PREFIX, causal over
# the continuation. Loss only on the continuation positions.
# ---------------------------------------------------------------------------

def prefix_lm_demo():
    k = 4   # prefix length: "the cat sat on" gets bidirectional attention, no loss
    input_tokens = SENTENCE[:-1]     # same shape as causal LM: tokens 0..8
    target_tokens = SENTENCE[1:]     # tokens 1..9
    T = len(input_tokens)

    target_ids = torch.tensor(
        [stoi[target_tokens[i]] if i >= k else IGNORE_INDEX for i in range(T)]
    )
    logits = torch.randn(T, VOCAB_SIZE)
    losses = compute_losses(logits, target_ids)

    rows = [
        (
            i, input_tokens[i], target_tokens[i] if i >= k else "-",
            "no (prefix)" if i < k else "yes", f"{losses[i].item():.3f}",
        )
        for i in range(T)
    ]
    print_table("4. PREFIX LM  (hybrid, e.g. UniLM / PaLM variant)", rows)
    print(f"    -> attention: positions 0..{k - 1} (prefix) see each other bidirectionally;")
    print(f"       positions {k}..{T - 1} attend causally (to the whole prefix + earlier continuation)")
    print(f"    -> loss positions: {T - k}/{T}  (continuation only; prefix contributes no loss)")
    return losses


def main():
    print("=" * 78)
    print("SAME SENTENCE, FOUR PRETRAINING OBJECTIVES")
    print("=" * 78)
    print(f"Sentence: {' '.join(SENTENCE)}")
    print(f"Vocabulary ({VOCAB_SIZE} tokens): {vocab}")

    causal_losses = causal_lm_demo()
    mlm_losses = masked_lm_demo()
    span_losses = span_corruption_demo()
    prefix_losses = prefix_lm_demo()

    print("\n" + "=" * 78)
    print("SUMMARY: LOSS POSITIONS PER OBJECTIVE (same underlying Transformer math)")
    print("=" * 78)
    print(f"{'objective':22s}{'total positions':>18}{'loss positions':>16}{'fraction':>12}")
    for name, losses in [
        ("Causal LM", causal_losses), ("Masked LM", mlm_losses),
        ("Span corruption", span_losses), ("Prefix LM", prefix_losses),
    ]:
        total = losses.numel()
        active = (losses != 0.0).sum().item()
        print(f"{name:22s}{total:>18}{active:>16}{active / total:>12.0%}")

    print("\n-> Every objective above ran through the exact same mechanism: random")
    print("   logits, torch.nn.functional.cross_entropy with ignore_index=-100.")
    print("   The ONLY thing that changed between objectives was which target")
    print("   value (real token id, or -100) sat at each position -- confirmed")
    print("   numerically above, since every ignored position's loss came back")
    print("   as exactly 0.000. Causal LM and span corruption supervise every")
    print("   position of their (different-length) sequences; masked LM and")
    print("   prefix LM each supervise only a subset. Swapping which subset is")
    print("   supervised, and whether the attention mask is causal or bidirectional,")
    print("   IS the difference between a GPT-style, BERT-style, T5-style, and")
    print("   UniLM/PaLM-style pretraining run -- not a different Transformer.")


if __name__ == "__main__":
    main()
