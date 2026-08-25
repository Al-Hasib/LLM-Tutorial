"""
Introduction to NLP

From-scratch (no sklearn/gensim) implementations of the classic text
representation methods: tokenization, Bag-of-Words, TF-IDF, one-hot
encoding's similarity problem, cosine similarity, and bigrams.

Run:
    python example.py
"""

import math
from collections import Counter


CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
]


# ---------------------------------------------------------------------------
# 1. Tokenization + vocabulary
# ---------------------------------------------------------------------------

def tokenize(text):
    """Very simple whitespace tokenizer with lowercasing."""
    return text.lower().split()


def build_vocab(corpus):
    vocab = sorted({token for doc in corpus for token in tokenize(doc)})
    return vocab


# ---------------------------------------------------------------------------
# 2. Bag-of-Words
# ---------------------------------------------------------------------------

def bag_of_words(doc, vocab):
    counts = Counter(tokenize(doc))
    return [counts.get(word, 0) for word in vocab]


# ---------------------------------------------------------------------------
# 3. TF-IDF (manual, matching the formula in the README)
# ---------------------------------------------------------------------------

def term_frequency(doc, vocab):
    counts = Counter(tokenize(doc))
    return [counts.get(word, 0) for word in vocab]


def inverse_document_frequency(corpus, vocab):
    n_docs = len(corpus)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in corpus if word in tokenize(doc))
        idf.append(math.log(n_docs / (1 + doc_count)))
    return idf


def tfidf_vector(doc, corpus, vocab, idf):
    tf = term_frequency(doc, vocab)
    return [tf_i * idf_i for tf_i, idf_i in zip(tf, idf)]


# ---------------------------------------------------------------------------
# 4. Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 5. One-hot encoding's blind spot
# ---------------------------------------------------------------------------

def one_hot(word, vocab):
    return [1 if w == word else 0 for w in vocab]


# ---------------------------------------------------------------------------
# 6. Bigrams
# ---------------------------------------------------------------------------

def bigrams(doc):
    tokens = tokenize(doc)
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def main():
    vocab = build_vocab(CORPUS)
    print("Corpus:")
    for i, doc in enumerate(CORPUS):
        print(f"  [{i}] {doc!r}")
    print(f"\nVocabulary ({len(vocab)} words): {vocab}\n")

    print("=" * 70)
    print("BAG-OF-WORDS")
    print("=" * 70)
    bow_vectors = [bag_of_words(doc, vocab) for doc in CORPUS]
    for i, vec in enumerate(bow_vectors):
        print(f"  doc[{i}] BoW = {vec}")

    print("\n" + "=" * 70)
    print("TF-IDF")
    print("=" * 70)
    idf = inverse_document_frequency(CORPUS, vocab)
    print("  IDF per vocab word (low = appears everywhere, uninformative):")
    for word, idf_val in zip(vocab, idf):
        print(f"    {word:8s} idf={idf_val:.3f}")
    tfidf_vectors = [tfidf_vector(doc, CORPUS, vocab, idf) for doc in CORPUS]
    for i, vec in enumerate(tfidf_vectors):
        print(f"  doc[{i}] TF-IDF = {[round(v, 3) for v in vec]}")

    print("\n" + "=" * 70)
    print("COSINE SIMILARITY: BoW vs. TF-IDF")
    print("=" * 70)
    print("  Using raw Bag-of-Words vectors (counts only):")
    for i in range(len(CORPUS)):
        for j in range(i + 1, len(CORPUS)):
            sim = cosine_similarity(bow_vectors[i], bow_vectors[j])
            print(f"    sim(doc[{i}], doc[{j}]) = {sim:.4f}")
    print("  -> doc[0] and doc[1] share 'the', 'sat', 'on' -> highest BoW similarity.")

    print("\n  Using TF-IDF vectors (common-word weight down to ~0):")
    for i in range(len(CORPUS)):
        for j in range(i + 1, len(CORPUS)):
            sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            print(f"    sim(doc[{i}], doc[{j}]) = {sim:.4f}")
    print("  -> All ~0.0: the shared words ('the', 'sat', 'on') get IDF ~ 0 "
          "because they\n     appear in most documents, so TF-IDF zeroes out exactly "
          "the overlap BoW found.\n     This is TF-IDF working as designed - it only "
          "considers overlap in *distinctive*\n     words, and here doc[0]/doc[1] don't "
          "share any distinctive ones ('cat'/'mat' vs 'dog'/'log').")

    print("\n" + "=" * 70)
    print("ONE-HOT ENCODING'S BLIND SPOT")
    print("=" * 70)
    cat_vec = one_hot("cats", vocab)
    dog_vec = one_hot("dogs", vocab)
    mat_vec = one_hot("mat", vocab)
    print(f"  cosine('cats', 'dogs') = {cosine_similarity(cat_vec, dog_vec):.4f}  "
          f"(semantically related words)")
    print(f"  cosine('cats', 'mat')  = {cosine_similarity(cat_vec, mat_vec):.4f}  "
          f"(semantically unrelated words)")
    print("  -> Identical (0.0) either way: one-hot vectors carry zero notion "
          "of meaning.\n     This is exactly what learned embeddings (next phase) fix.")

    print("\n" + "=" * 70)
    print("BIGRAMS")
    print("=" * 70)
    for i, doc in enumerate(CORPUS):
        print(f"  doc[{i}] bigrams = {bigrams(doc)}")


if __name__ == "__main__":
    main()
