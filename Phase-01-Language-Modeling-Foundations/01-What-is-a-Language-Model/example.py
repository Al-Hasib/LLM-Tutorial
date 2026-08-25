"""
What is a Language Model

A from-scratch bigram language model: counting-based probability
estimation, add-one smoothing, perplexity, and text generation by
sampling -- the simplest possible working language model.

Run:
    python example.py
"""

import math
import random
from collections import defaultdict, Counter

random.seed(42)

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "the dog chased the cat",
    "a cat and a dog are friends",
]

START = "<s>"
END = "</s>"


def tokenize(sentence):
    return [START] + sentence.lower().split() + [END]


class BigramLanguageModel:
    def __init__(self, corpus):
        self.vocab = set()
        self.bigram_counts = defaultdict(Counter)   # bigram_counts[w_{t-1}][w_t] = count
        self.unigram_counts = Counter()              # count of each word appearing as a "previous word"

        for sentence in corpus:
            tokens = tokenize(sentence)
            self.vocab.update(tokens)
            for prev_word, word in zip(tokens[:-1], tokens[1:]):
                self.bigram_counts[prev_word][word] += 1
                self.unigram_counts[prev_word] += 1

        self.vocab_size = len(self.vocab)

    def prob(self, word, prev_word):
        """Add-one (Laplace) smoothed P(word | prev_word)."""
        count_pair = self.bigram_counts[prev_word][word]
        count_prev = self.unigram_counts[prev_word]
        return (count_pair + 1) / (count_prev + self.vocab_size)

    def sentence_log_prob(self, sentence):
        tokens = tokenize(sentence)
        log_prob = 0.0
        for prev_word, word in zip(tokens[:-1], tokens[1:]):
            p = self.prob(word, prev_word)
            log_prob += math.log(p)
        return log_prob

    def perplexity(self, sentence):
        tokens = tokenize(sentence)
        n_transitions = len(tokens) - 1
        log_prob = self.sentence_log_prob(sentence)
        return math.exp(-log_prob / n_transitions)

    def generate(self, max_words=15):
        prev_word = START
        result = []
        # <s> can only ever be a *previous* word, never a generated one.
        candidates = [w for w in self.vocab if w != START]
        for _ in range(max_words):
            weights = [self.prob(w, prev_word) for w in candidates]
            next_word = random.choices(candidates, weights=weights, k=1)[0]
            if next_word == END:
                break
            result.append(next_word)
            prev_word = next_word
        return " ".join(result)


def main():
    print("Training corpus:")
    for s in CORPUS:
        print(f"  {s!r}")

    lm = BigramLanguageModel(CORPUS)
    print(f"\nVocabulary size (incl. <s>/</s>): {lm.vocab_size}")

    print("\n" + "=" * 70)
    print("BIGRAM PROBABILITIES (a few examples)")
    print("=" * 70)
    for prev_word, word in [("the", "cat"), ("the", "dog"), ("cat", "sat"), ("the", "zebra")]:
        p = lm.prob(word, prev_word)
        print(f"  P({word!r} | {prev_word!r}) = {p:.4f}")
    print("  -> 'the zebra' never appeared in training, but smoothing still gives it")
    print("     a small non-zero probability instead of killing the whole sentence.")

    print("\n" + "=" * 70)
    print("SENTENCE PROBABILITY AND PERPLEXITY")
    print("=" * 70)
    test_sentences = [
        "the cat sat on the mat",     # seen during training verbatim
        "the dog sat on the mat",     # a plausible recombination, unseen verbatim
        "a mat chased a log",         # grammatically odd / unlikely recombination
    ]
    for s in test_sentences:
        log_p = lm.sentence_log_prob(s)
        ppl = lm.perplexity(s)
        print(f"  {s!r}")
        print(f"    log P(sentence) = {log_p:.4f}   perplexity = {ppl:.2f}")
    print("\n  -> Lower perplexity = the model finds the sentence less 'surprising'.")
    print("     The first two tie: 'cat' and 'dog' occur in symmetric contexts in this")
    print("     toy corpus, so the bigram counts can't tell them apart. The nonsense")
    print("     recombination scores clearly worse -- higher perplexity.")

    print("\n" + "=" * 70)
    print("TEXT GENERATION (sampling next word repeatedly)")
    print("=" * 70)
    for i in range(3):
        print(f"  sample {i + 1}: {lm.generate()}")


if __name__ == "__main__":
    main()
