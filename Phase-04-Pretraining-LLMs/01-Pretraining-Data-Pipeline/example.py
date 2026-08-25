"""
Pretraining Data Pipeline

Three demos, all pure Python/standard-library (this lesson is about
document-level algorithms, not neural nets, so there's no PyTorch here):

  1. A heuristic quality filter (too short / too symbol-heavy / too
     repetitive) applied to a toy pool of "web documents".
  2. Exact deduplication via hashing -- fast, precise, but blind to
     anything that isn't byte-identical.
  3. A from-scratch MinHash near-duplicate detector -- estimates Jaccard
     similarity from short signatures and correctly flags near-duplicate
     documents (differing by only a few words) that exact-hash dedup
     completely misses.

Run:
    python example.py
"""

import hashlib
import random
from collections import Counter

random.seed(0)


# ---------------------------------------------------------------------------
# Toy document pool. Meant to stand in for "documents that survived HTML
# extraction" -- i.e. already plain text, not raw HTML. Includes: normal
# documents, exact duplicates, NEAR-duplicates (a few words changed), and
# clearly low-quality documents (too short, spammy/symbol-heavy, repetitive).
# ---------------------------------------------------------------------------

DOCS = {
    "wiki_forest_1": (
        "The boreal forest, also known as taiga, is the world's largest land "
        "biome. It is characterized by coniferous trees such as pine, spruce, "
        "and fir, and experiences long, cold winters with short, mild summers. "
        "Many species of birds migrate to the taiga only during the summer "
        "breeding season."
    ),
    # Exact duplicate of the document above -- e.g. a syndicated mirror copy.
    "mirror_forest_1": (
        "The boreal forest, also known as taiga, is the world's largest land "
        "biome. It is characterized by coniferous trees such as pine, spruce, "
        "and fir, and experiences long, cold winters with short, mild summers. "
        "Many species of birds migrate to the taiga only during the summer "
        "breeding season."
    ),
    # NEAR-duplicate: same document, but a few words changed/inserted (as if
    # a second site republished it with light editing). Exact hash dedup will
    # NOT catch this -- the bytes differ -- but MinHash should.
    "reprint_forest_1": (
        "The boreal forest, also called taiga, is the world's largest "
        "terrestrial biome. It is characterized by coniferous trees such as "
        "pine, spruce, and fir, and experiences long, harsh winters with "
        "short, cool summers. Many species of birds migrate to the taiga "
        "only during the brief summer breeding season."
    ),
    "wiki_ocean_1": (
        "The Pacific Ocean is the largest and deepest of Earth's five ocean "
        "basins. It extends from the Arctic Ocean in the north to the "
        "Southern Ocean in the south, and is bounded by Asia and Australia "
        "in the west and the Americas in the east."
    ),
    "blog_cooking_1": (
        "To make a simple tomato sauce, start by sauteing chopped garlic and "
        "onion in olive oil until fragrant. Add crushed tomatoes, a pinch of "
        "salt, and a few basil leaves, then let it simmer for twenty minutes "
        "until the sauce thickens and the flavors meld together."
    ),
    "news_article_1": (
        "City officials announced today that the downtown bridge renovation "
        "project will begin next month, with construction expected to take "
        "approximately eight months to complete. Traffic will be rerouted "
        "through the adjacent avenue during construction hours."
    ),
    # Low quality: far too short to carry useful training signal.
    "spam_short_1": "Click here now!!!",
    # Low quality: extremely high symbol-to-word ratio -- likely spam/markup
    # residue that survived HTML extraction.
    "spam_symbols_1": "!!! BUY NOW $$$ >>> #1 DEAL *** LIMITED TIME <<< 50% OFF !!! $$$ ###",
    # Low quality: highly repetitive templated/boilerplate text.
    "spam_repeat_1": ("buy now buy now buy now " * 8).strip(),
    "academic_abstract_1": (
        "This paper presents a comprehensive empirical study of gradient "
        "descent optimization methods for large-scale neural network "
        "training, evaluating convergence behavior across a range of "
        "learning rate schedules and batch sizes."
    ),
}


# ---------------------------------------------------------------------------
# 1. Heuristic quality filter
# ---------------------------------------------------------------------------

def quality_check(text, min_words=8, max_symbol_ratio=0.28, max_repeat_frac=0.4):
    """Returns (passed: bool, reason: str). Mirrors the cheap, rule-based
    checks real pipelines (CCNet, Gopher/MassiveText, RefinedWeb) run over
    every document before any expensive classifier gets involved."""
    words = text.split()
    num_words = len(words)

    if num_words < min_words:
        return False, f"too short ({num_words} words < {min_words})"

    # Symbol-to-word ratio: count characters that are neither alphanumeric
    # nor whitespace, relative to word count. Spammy/markup-residue text
    # tends to be dominated by punctuation and symbols.
    num_symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    symbol_ratio = num_symbols / max(num_words, 1)
    if symbol_ratio > max_symbol_ratio:
        return False, f"symbol-to-word ratio too high ({symbol_ratio:.2f} > {max_symbol_ratio})"

    # Repetition: fraction of all word occurrences taken up by the single
    # most common word. Templated/auto-generated spam repeats the same
    # phrase over and over; normal prose does not.
    counts = Counter(words)
    most_common_frac = counts.most_common(1)[0][1] / num_words
    if most_common_frac > max_repeat_frac:
        return False, f"too repetitive (top word = {most_common_frac:.0%} of all words)"

    return True, "passed"


def quality_filter_demo():
    print("=" * 70)
    print("1. HEURISTIC QUALITY FILTER")
    print("=" * 70)
    kept, rejected = [], []
    for name, text in DOCS.items():
        passed, reason = quality_check(text)
        (kept if passed else rejected).append(name)
        status = "KEEP  " if passed else "REJECT"
        print(f"  [{status}] {name:22s} {reason}")

    print(f"\n-> Kept {len(kept)}/{len(DOCS)} documents. Rejected: {rejected}")
    print("   Note the three 'spam_*' documents were all caught by simple,")
    print("   cheap rules -- no neural network needed for this pass. This is")
    print("   exactly why heuristic filtering runs FIRST, over the entire raw")
    print("   crawl, before any more expensive classifier-based filtering.")
    return kept


# ---------------------------------------------------------------------------
# 2. Exact deduplication via hashing
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase + collapse whitespace, so trivial formatting differences
    (extra spaces, capitalization) don't defeat exact-hash matching."""
    return " ".join(text.lower().split())


def exact_dedup(docs):
    """Returns (unique_docs, duplicate_pairs) using SHA-1 over normalized text."""
    seen_hashes = {}
    unique_docs = {}
    duplicate_pairs = []
    for name, text in docs.items():
        h = hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()
        if h in seen_hashes:
            duplicate_pairs.append((seen_hashes[h], name))
        else:
            seen_hashes[h] = name
            unique_docs[name] = text
    return unique_docs, duplicate_pairs


def exact_dedup_demo(kept_docs):
    print("\n" + "=" * 70)
    print("2. EXACT DEDUPLICATION (SHA-1 HASHING)")
    print("=" * 70)
    docs = {name: DOCS[name] for name in kept_docs}
    unique_docs, dup_pairs = exact_dedup(docs)

    print(f"Documents going in:  {len(docs)}")
    print(f"Exact duplicates found: {dup_pairs if dup_pairs else 'none'}")
    print(f"Documents remaining after exact dedup: {len(unique_docs)}")
    print("\n-> 'mirror_forest_1' is a byte-identical copy of 'wiki_forest_1' and")
    print("   was correctly caught. But 'reprint_forest_1' -- the SAME article")
    print("   with a handful of words changed -- has a different hash entirely,")
    print("   so exact dedup lets it straight through. That's exactly the gap")
    print("   near-duplicate detection (part 3) exists to close.")
    return unique_docs


# ---------------------------------------------------------------------------
# 3. MinHash near-duplicate detection
# ---------------------------------------------------------------------------

def shingles(text, n=3):
    """Word n-gram shingles -- the standard unit of comparison for MinHash.
    Character shingles also work; word shingles are more readable to inspect."""
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def true_jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)


def minhash_signature(shingle_set, num_hashes, salts):
    """One MinHash signature = num_hashes independent 'minimum hash value
    over the shingle set' computations. Each salt simulates an independent
    random hash function by mixing a distinct salt into MD5 before hashing --
    a standard cheap trick when you don't have num_hashes literal distinct
    hash algorithms lying around."""
    signature = []
    for salt in salts:
        min_val = min(
            int(hashlib.md5(f"{salt}:{shingle}".encode("utf-8")).hexdigest(), 16)
            for shingle in shingle_set
        )
        signature.append(min_val)
    return signature


def estimated_jaccard_from_signatures(sig_a, sig_b):
    """Fraction of matching signature positions = unbiased estimator of the
    true Jaccard similarity (this is the key MinHash property from the
    README: P(minhash_h(A) == minhash_h(B)) == Jaccard(A, B) for one hash)."""
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def minhash_dedup_demo(unique_docs):
    print("\n" + "=" * 70)
    print("3. NEAR-DUPLICATE DETECTION VIA MINHASH")
    print("=" * 70)

    num_hashes = 128
    salts = list(range(num_hashes))

    doc_shingles = {name: shingles(text) for name, text in unique_docs.items()}
    doc_signatures = {
        name: minhash_signature(s, num_hashes, salts) for name, s in doc_shingles.items()
    }

    print(f"Computing {num_hashes}-element MinHash signatures over 3-word shingles "
          f"for {len(unique_docs)} documents (post exact-dedup).\n")

    names = list(unique_docs.keys())
    threshold = 0.5   # flag as near-duplicate if estimated Jaccard exceeds this
    print(f"{'doc A':22s}{'doc B':22s}{'true Jaccard':>14}{'MinHash est.':>14}{'flagged?':>10}")
    flagged_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            tj = true_jaccard(doc_shingles[a], doc_shingles[b])
            ej = estimated_jaccard_from_signatures(doc_signatures[a], doc_signatures[b])
            is_flagged = ej > threshold
            if is_flagged:
                flagged_pairs.append((a, b))
            # Only print pairs with any meaningful similarity, to keep the
            # table readable -- most cross-topic pairs are near 0 on both.
            if tj > 0.05 or ej > 0.05:
                print(f"{a:22s}{b:22s}{tj:>14.3f}{ej:>14.3f}{'YES' if is_flagged else 'no':>10}")

    print(f"\nFlagged as near-duplicates (estimated Jaccard > {threshold}): {flagged_pairs}")
    print("\n-> 'wiki_forest_1' and 'reprint_forest_1' share no identical bytes and")
    print("   were completely invisible to exact-hash dedup in part 2, yet MinHash's")
    print("   signature-based ESTIMATE of their Jaccard similarity lands close to the")
    print("   true value computed directly from the shingle sets, and correctly")
    print("   clears the near-duplicate threshold. This is exactly how billion-document")
    print("   pipelines catch republished/lightly-edited content: comparing a handful")
    print("   of small integers per document instead of the full text of every pair.")


def main():
    kept = quality_filter_demo()
    unique_docs = exact_dedup_demo(kept)
    minhash_dedup_demo(unique_docs)


if __name__ == "__main__":
    main()
