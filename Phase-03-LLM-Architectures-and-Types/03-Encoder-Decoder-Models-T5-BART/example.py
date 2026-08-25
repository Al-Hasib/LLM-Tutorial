"""
Encoder-Decoder Models: T5 and BART

From-scratch implementations of the two pretraining-data-construction
recipes described in the README:
  1. T5-style span corruption -- replace contiguous spans with sentinel
     tokens, target is only the missing pieces.
  2. BART-style denoising -- four corruption strategies (token masking,
     token deletion, sentence permutation, document rotation), target
     is always the full original text.

Both are architecture-agnostic data-prep steps: feeding the resulting
(corrupted_input, target) pairs into the encoder-decoder Transformer
already built in Phase 02 Lesson 4 is the entire remaining training
recipe -- nothing about the model itself changes.

Run:
    python example.py
"""

import random

random.seed(0)


# ---------------------------------------------------------------------------
# 1. T5-style span corruption
# ---------------------------------------------------------------------------

def t5_span_corruption(tokens, noise_density=0.15, max_span_length=3):
    n = len(tokens)
    target_noise = max(1, round(n * noise_density))
    corrupted = [False] * n
    noise_so_far = 0

    attempts = 0
    while noise_so_far < target_noise and attempts < 100:
        attempts += 1
        start = random.randrange(n)
        span_len = random.randint(1, max_span_length)
        span_len = min(span_len, n - start, target_noise - noise_so_far)
        if span_len <= 0 or any(corrupted[start:start + span_len]):
            continue   # overlaps an already-chosen span, or doesn't fit -- try again
        for j in range(start, start + span_len):
            corrupted[j] = True
        noise_so_far += span_len

    input_tokens, target_tokens = [], []
    sentinel_id = 0
    i = 0
    while i < n:
        if corrupted[i]:
            sentinel = f"<extra_id_{sentinel_id}>"
            input_tokens.append(sentinel)
            target_tokens.append(sentinel)
            while i < n and corrupted[i]:
                target_tokens.append(tokens[i])
                i += 1
            sentinel_id += 1
        else:
            input_tokens.append(tokens[i])
            i += 1
    target_tokens.append(f"<extra_id_{sentinel_id}>")  # T5 terminates with one final sentinel
    return input_tokens, target_tokens


def t5_demo():
    print("=" * 70)
    print("1. T5-STYLE SPAN CORRUPTION")
    print("=" * 70)
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the sun rises over the dark forest every morning",
    ]
    for sentence in sentences:
        tokens = sentence.split()
        corrupted_input, target = t5_span_corruption(tokens)
        print(f"  original: {' '.join(tokens)}")
        print(f"  input:    {' '.join(corrupted_input)}")
        print(f"  target:   {' '.join(target)}\n")
    print("-> The target is much SHORTER than the input -- it only reconstructs")
    print("   the missing spans, tagged by sentinel, not the whole sentence.")


# ---------------------------------------------------------------------------
# 2. BART-style denoising corruptions
# ---------------------------------------------------------------------------

def bart_token_masking(tokens, mask_prob=0.3):
    corrupted = [("[MASK]" if random.random() < mask_prob else t) for t in tokens]
    return corrupted, list(tokens)   # target = the FULL original sequence


def bart_token_deletion(tokens, delete_prob=0.3):
    corrupted = [t for t in tokens if random.random() >= delete_prob]
    return corrupted, list(tokens)


def bart_sentence_permutation(sentences):
    shuffled = sentences[:]
    # Reshuffle until it's actually different -- for a document with only a
    # handful of sentences, random.shuffle CAN legitimately land back on the
    # original order; we retry purely so this demo always shows a real change.
    for _ in range(20):
        random.shuffle(shuffled)
        if shuffled != sentences or len(sentences) < 2:
            break
    return shuffled, sentences   # target = the correct original order


def bart_document_rotation(tokens):
    if len(tokens) < 2:
        return list(tokens), list(tokens)
    pivot = random.randint(1, len(tokens) - 1)
    rotated = tokens[pivot:] + tokens[:pivot]
    return rotated, list(tokens)   # target = the true original order/start


def bart_demo():
    print("\n" + "=" * 70)
    print("2. BART-STYLE DENOISING (target is always the FULL original text)")
    print("=" * 70)

    tokens = "the quick brown fox jumps over the lazy dog".split()

    corrupted, target = bart_token_masking(tokens)
    print("Token masking:")
    print(f"  input:  {' '.join(corrupted)}")
    print(f"  target: {' '.join(target)}\n")

    corrupted, target = bart_token_deletion(tokens)
    print("Token deletion (model must infer WHERE tokens are missing, not just what):")
    print(f"  input:  {' '.join(corrupted)}")
    print(f"  target: {' '.join(target)}\n")

    sentences = [
        "the fox saw the hen.",
        "the hen ran into the barn.",
        "the fox followed close behind.",
    ]
    shuffled, original_order = bart_sentence_permutation(sentences)
    print("Sentence permutation:")
    print(f"  input:  {' '.join(shuffled)}")
    print(f"  target: {' '.join(original_order)}\n")

    corrupted, target = bart_document_rotation(tokens)
    print("Document rotation (model must identify the TRUE starting point):")
    print(f"  input:  {' '.join(corrupted)}")
    print(f"  target: {' '.join(target)}")

    print("\n-> Every BART target is the complete, correctly-ordered original text --")
    print("   a strictly harder reconstruction target than T5's 'just the missing")
    print("   pieces', which is exactly why BART tends to shine on generation-heavy")
    print("   tasks like summarization: it was always trained to produce full,")
    print("   fluent text, never just isolated spans.")


def main():
    t5_demo()
    bart_demo()


if __name__ == "__main__":
    main()
