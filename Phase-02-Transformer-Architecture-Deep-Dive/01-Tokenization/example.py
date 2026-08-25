"""
Tokenization

A from-scratch Byte-Pair Encoding (BPE) tokenizer: learn merges from a
toy corpus, encode/decode text with them, and show BPE handling a
never-before-seen word gracefully -- something word-level tokenization
structurally cannot do.

Run:
    python example.py
"""

from collections import Counter

END_OF_WORD = "</w>"   # marks word boundaries so merges never cross words

CORPUS = [
    "low", "lower", "lowest", "low", "lower",
    "new", "newer", "newest", "new",
    "wide", "wider", "widest",
]


def word_to_symbols(word):
    """'low' -> ['l', 'o', 'w', '</w>']"""
    return list(word) + [END_OF_WORD]


def get_pair_counts(word_freqs):
    """Count every adjacent symbol pair, weighted by how often the word occurs."""
    pair_counts = Counter()
    for symbols, freq in word_freqs:
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def merge_pair(pair, word_freqs):
    """Replace every occurrence of `pair` with a single merged symbol."""
    merged_symbol = "".join(pair)
    new_word_freqs = []
    for symbols, freq in word_freqs:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_freqs.append((new_symbols, freq))
    return new_word_freqs


def train_bpe(corpus, num_merges):
    word_counts = Counter(corpus)
    word_freqs = [(word_to_symbols(word), freq) for word, freq in word_counts.items()]

    merges = []  # ordered list of merges, order matters for encoding new text
    for step in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        word_freqs = merge_pair(best_pair, word_freqs)
        print(f"  merge {step + 1:2d}: {best_pair} -> {''.join(best_pair)!r} "
              f"(appeared {pair_counts[best_pair]} times)")

    vocab = sorted({sym for symbols, _ in word_freqs for sym in symbols})
    return merges, vocab


def bpe_encode(word, merges):
    """Apply learned merges, IN LEARNED ORDER, to a new word."""
    symbols = word_to_symbols(word)
    for pair in merges:
        i = 0
        new_symbols = []
        merged_symbol = "".join(pair)
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def word_level_lookup(word, known_vocab):
    """A word-level tokenizer: either the exact word is known, or it's <unk>."""
    return [word] if word in known_vocab else ["<unk>"]


def main():
    print("Training corpus (word frequencies):")
    print(f"  {dict(Counter(CORPUS))}\n")

    print("Learning BPE merges:")
    merges, vocab = train_bpe(CORPUS, num_merges=12)

    print(f"\nFinal learned vocabulary ({len(vocab)} symbols): {vocab}")

    print("\n" + "=" * 70)
    print("ENCODING SEEN WORDS")
    print("=" * 70)
    for word in ["low", "newest", "wider"]:
        tokens = bpe_encode(word, merges)
        print(f"  {word!r:10s} -> {tokens}")

    print("\n" + "=" * 70)
    print("ENCODING A WORD NEVER SEEN DURING TRAINING")
    print("=" * 70)
    unseen_word = "widening"  # shares "widen"->"wide"+"..." structure, never seen whole
    bpe_tokens = bpe_encode(unseen_word, merges)
    word_level_tokens = word_level_lookup(unseen_word, set(CORPUS))
    print(f"  BPE tokenizer:        {unseen_word!r} -> {bpe_tokens}")
    print(f"  word-level tokenizer: {unseen_word!r} -> {word_level_tokens}")
    print("\n  -> BPE falls back to smaller, still-meaningful known pieces (down to")
    print("     individual characters in the worst case) instead of giving up.")
    print("     A word-level tokenizer has no such fallback -- it's simply unknown.")


if __name__ == "__main__":
    main()
