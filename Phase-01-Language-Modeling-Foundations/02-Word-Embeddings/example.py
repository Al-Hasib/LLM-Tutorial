"""
Word Embeddings

A from-scratch Skip-gram word2vec implementation with negative sampling,
trained with plain NumPy + manual gradients (no autograd) on a tiny toy
corpus designed to have clear semantic clusters (royalty / people / pets).
After training we inspect nearest neighbors by cosine similarity and try
the classic "king - man + woman ~ queen" analogy.

Run:
    python example.py
"""

import numpy as np

rng = np.random.default_rng(42)

CORPUS = [
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the king wears a crown",
    "the queen wears a crown",
    "the king is a wise man",
    "the queen is a wise woman",
    "the prince is a young king",
    "the princess is a young queen",
    "the man walks the dog",
    "the woman walks the dog",
    "the man feeds the dog",
    "the woman feeds the cat",
    "the dog chases the cat",
    "the cat chases the mouse",
]

WINDOW = 2
EMBED_DIM = 16
NUM_NEGATIVES = 5
LEARNING_RATE = 0.05
EPOCHS = 300


def build_vocab(corpus):
    tokens = [tok for sent in corpus for tok in sent.lower().split()]
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    unigram_counts = np.array([tokens.count(w) for w in vocab], dtype=float)
    return vocab, word2idx, unigram_counts


def build_skipgram_pairs(corpus, word2idx, window):
    pairs = []
    for sent in corpus:
        tokens = sent.lower().split()
        ids = [word2idx[t] for t in tokens]
        for i, center in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, ids[j]))
    return pairs


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNegSampling:
    def __init__(self, vocab_size, embed_dim, unigram_counts):
        # Two separate embedding tables, standard word2vec setup:
        # W_center: a word's representation when it's the input/center word
        # W_context: a word's representation when it's the predicted/output word
        self.W_center = (rng.random((vocab_size, embed_dim)) - 0.5) / embed_dim
        self.W_context = (rng.random((vocab_size, embed_dim)) - 0.5) / embed_dim

        # Negative sampling distribution: unigram frequency raised to 0.75,
        # the standard word2vec smoothing that slightly boosts rare words.
        smoothed = unigram_counts ** 0.75
        self.neg_sample_probs = smoothed / smoothed.sum()
        self.vocab_size = vocab_size

    def sample_negatives(self, true_context, k):
        negatives = []
        while len(negatives) < k:
            candidate = rng.choice(self.vocab_size, p=self.neg_sample_probs)
            if candidate != true_context:
                negatives.append(candidate)
        return negatives

    def train_step(self, center, context, lr, k):
        v_c = self.W_center[center]                       # (D,)
        negatives = self.sample_negatives(context, k)
        words = [context] + negatives
        labels = np.array([1.0] + [0.0] * k)               # positive then negatives

        u_words = self.W_context[words]                    # (k+1, D)
        scores = sigmoid(u_words @ v_c)                     # (k+1,)
        error = scores - labels                             # (k+1,)  grad wrt the dot products

        grad_v_c = error @ u_words                           # (D,)
        grad_u_words = np.outer(error, v_c)                  # (k+1, D)

        # Gradient descent updates.
        self.W_context[words] -= lr * grad_u_words
        self.W_center[center] -= lr * grad_v_c

        # Binary cross-entropy loss for this pair + its negative samples.
        eps = 1e-10
        loss = -np.sum(labels * np.log(scores + eps) + (1 - labels) * np.log(1 - scores + eps))
        return loss

    def embedding(self, idx):
        # Common convention: the "true" word embedding is the sum (or average)
        # of its center and context vectors.
        return self.W_center[idx] + self.W_context[idx]


def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def nearest_neighbors(model, word2idx, vocab, query_word, top_k=3):
    query_vec = model.embedding(word2idx[query_word])
    sims = []
    for word, idx in word2idx.items():
        if word == query_word:
            continue
        sims.append((word, cosine_similarity(query_vec, model.embedding(idx))))
    sims.sort(key=lambda pair: pair[1], reverse=True)
    return sims[:top_k]


def main():
    vocab, word2idx, unigram_counts = build_vocab(CORPUS)
    pairs = build_skipgram_pairs(CORPUS, word2idx, WINDOW)
    print(f"Vocabulary ({len(vocab)} words): {vocab}")
    print(f"Generated {len(pairs)} skip-gram (center, context) training pairs\n")

    model = SkipGramNegSampling(len(vocab), EMBED_DIM, unigram_counts)

    print("Training skip-gram with negative sampling...")
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(pairs)
        total_loss = 0.0
        for center, context in pairs:
            total_loss += model.train_step(center, context, LEARNING_RATE, NUM_NEGATIVES)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  avg loss = {total_loss / len(pairs):.4f}")

    print("\n" + "=" * 70)
    print("NEAREST NEIGHBORS (cosine similarity on learned embeddings)")
    print("=" * 70)
    for query in ["king", "queen", "man", "woman", "dog", "cat"]:
        neighbors = nearest_neighbors(model, word2idx, vocab, query)
        formatted = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
        print(f"  {query:8s} -> {formatted}")

    print("\n" + "=" * 70)
    print("VECTOR ARITHMETIC: king - man + woman =~ ?")
    print("=" * 70)
    result_vec = (model.embedding(word2idx["king"])
                  - model.embedding(word2idx["man"])
                  + model.embedding(word2idx["woman"]))
    sims = []
    for word, idx in word2idx.items():
        if word in ("king", "man", "woman"):
            continue
        sims.append((word, cosine_similarity(result_vec, model.embedding(idx))))
    sims.sort(key=lambda pair: pair[1], reverse=True)
    print("  Top matches:", ", ".join(f"{w} ({s:.3f})" for w, s in sims[:5]))
    print("  NOTE: this analogy is famous on embeddings trained from billions of")
    print("  words. On a ~14-sentence toy corpus it's a fun demo, not a guarantee --")
    print("  treat whatever word comes out on top as illustrative, not definitive.")


if __name__ == "__main__":
    main()
