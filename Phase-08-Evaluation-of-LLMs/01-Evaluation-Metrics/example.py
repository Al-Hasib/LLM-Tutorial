"""
Evaluation Metrics

Implements, from scratch, the classic automatic text-evaluation metrics:
  1. BLEU (n-gram precision + brevity penalty)          -- machine translation
  2. ROUGE-N and ROUGE-L (n-gram / LCS recall)           -- summarization
  3. Exact Match and token-level F1                      -- extractive QA
  4. Perplexity of a trained toy bigram language model   -- recap from Phase 01

The centerpiece demo (section 5) is honest and a little uncomfortable: a
CORRECT paraphrase is scored against a reference and comes out with a much
lower BLEU/ROUGE score than a mediocre answer that happens to reuse the
reference's exact words. That is not a bug in this implementation -- it is
the real, well-documented behavior of these metrics, and precisely why the
course later introduces LLM-as-a-Judge for open-ended generation.

Run:
    python example.py
"""

import math
from collections import Counter

# ---------------------------------------------------------------------------
# Shared tokenizer -- lowercase, split on whitespace, strip trailing punctuation.
# Deliberately simple; real systems use the same subword tokenizer the model
# uses (Phase 02 Lesson 1), but whitespace tokens keep every formula legible.
# ---------------------------------------------------------------------------

def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# 1. BLEU: modified n-gram precision (clipped) + brevity penalty
# ---------------------------------------------------------------------------

def modified_precision(candidate_tokens, reference_tokens_list, n):
    """Clipped n-gram precision: candidate n-gram counts are capped at the
    MAX count seen in any single reference, so repeating one correct n-gram
    forever cannot inflate the score."""
    candidate_counts = Counter(ngrams(candidate_tokens, n))
    if not candidate_counts:
        return 0.0

    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_counts = Counter(ngrams(ref_tokens, n))
        for ngram, count in ref_counts.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

    clipped = sum(min(count, max_ref_counts[ngram]) for ngram, count in candidate_counts.items())
    total = sum(candidate_counts.values())
    return clipped / total


def brevity_penalty(candidate_tokens, reference_tokens_list):
    c = len(candidate_tokens)
    # "closest" reference length, as in the original BLEU paper
    r = min(reference_tokens_list, key=lambda ref: abs(len(ref) - c))
    r = len(r)
    if c > r:
        return 1.0
    if c == 0:
        return 0.0
    return math.exp(1 - r / c)


def bleu(candidate, references, max_n=4):
    """candidate: a string. references: a list of reference strings."""
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(r) for r in references]

    precisions = []
    for n in range(1, max_n + 1):
        p_n = modified_precision(candidate_tokens, reference_tokens_list, n)
        precisions.append(p_n)

    if min(precisions) == 0.0:
        # standard convention: BLEU is 0 if any order's precision is exactly 0
        # (log(0) is undefined) -- this happens a lot with short toy sentences.
        geo_mean = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / max_n
        geo_mean = math.exp(log_avg)

    bp = brevity_penalty(candidate_tokens, reference_tokens_list)
    return bp * geo_mean, precisions, bp


# ---------------------------------------------------------------------------
# 2. ROUGE-N (n-gram recall) and ROUGE-L (LCS-based F-measure)
# ---------------------------------------------------------------------------

def rouge_n(candidate, reference, n):
    candidate_tokens, reference_tokens = tokenize(candidate), tokenize(reference)
    reference_counts = Counter(ngrams(reference_tokens, n))
    candidate_counts = Counter(ngrams(candidate_tokens, n))
    if not reference_counts:
        return 0.0
    overlap = sum(min(count, candidate_counts[ngram]) for ngram, count in reference_counts.items())
    return overlap / sum(reference_counts.values())


def lcs_length(a, b):
    """Standard O(len(a)*len(b)) dynamic-programming longest-common-subsequence length."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate, reference, beta=1.2):
    candidate_tokens, reference_tokens = tokenize(candidate), tokenize(reference)
    lcs = lcs_length(candidate_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    r_lcs = lcs / len(reference_tokens)
    p_lcs = lcs / len(candidate_tokens)
    return (1 + beta ** 2) * r_lcs * p_lcs / (r_lcs + beta ** 2 * p_lcs)


# ---------------------------------------------------------------------------
# 3. Exact Match and token-level F1 (extractive QA, SQuAD-style)
# ---------------------------------------------------------------------------

def exact_match(prediction, reference):
    return int(tokenize(prediction) == tokenize(reference))


def token_f1(prediction, reference):
    pred_tokens, ref_tokens = tokenize(prediction), tokenize(reference)
    pred_counts, ref_counts = Counter(pred_tokens), Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())   # multiset intersection
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 4. A trained toy bigram language model, for the perplexity recap
#    (same recipe as Phase 01 Lesson 1, condensed)
# ---------------------------------------------------------------------------

from collections import defaultdict

START, END = "<s>", "</s>"

PPL_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "the dog chased the cat",
    "a cat and a dog are friends",
]


class BigramLM:
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()
        for sentence in corpus:
            tokens = [START] + sentence.lower().split() + [END]
            self.vocab.update(tokens)
            for prev, cur in zip(tokens[:-1], tokens[1:]):
                self.bigram_counts[prev][cur] += 1
                self.unigram_counts[prev] += 1
        self.vocab_size = len(self.vocab)

    def prob(self, cur, prev):
        return (self.bigram_counts[prev][cur] + 1) / (self.unigram_counts[prev] + self.vocab_size)

    def perplexity(self, sentence):
        tokens = [START] + sentence.lower().split() + [END]
        log_prob = sum(math.log(self.prob(cur, prev)) for prev, cur in zip(tokens[:-1], tokens[1:]))
        return math.exp(-log_prob / (len(tokens) - 1))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def bleu_rouge_demo():
    print("=" * 78)
    print("1-2. BLEU AND ROUGE ON TOY (CANDIDATE, REFERENCE) PAIRS")
    print("=" * 78)

    reference = "the cat sat quietly on the warm windowsill"
    pairs = [
        ("near-verbatim match",
         "the cat sat quietly on the warm windowsill"),
        ("good paraphrase (correct meaning, different words)",
         "a feline rested calmly upon the sunny window ledge"),
        ("partial overlap, wrong detail",
         "the cat sat quietly on the cold floor"),
        ("word-salad using reference vocabulary",
         "windowsill warm the on quietly sat the cat"),
    ]

    print(f"Reference: {reference!r}\n")
    header = f"{'candidate':52} {'BLEU':>7} {'R-1':>7} {'R-2':>7} {'R-L':>7}"
    print(header)
    print("-" * len(header))
    results = {}
    for label, candidate in pairs:
        score, precisions, bp = bleu(candidate, [reference])
        r1 = rouge_n(candidate, reference, 1)
        r2 = rouge_n(candidate, reference, 2)
        rl = rouge_l(candidate, reference)
        results[label] = (score, r1, r2, rl)
        print(f"{label:52} {score:>7.3f} {r1:>7.3f} {r2:>7.3f} {rl:>7.3f}")

    print("\n-> The word-salad row reuses every single word from the reference (so its")
    print("   unigram overlap -- ROUGE-1 -- is high) but scrambles their order; BLEU's")
    print("   4-gram requirement and ROUGE-L's contiguity-respecting LCS both collapse")
    print("   toward 0, correctly rejecting it as not a fluent match.")

    paraphrase_bleu = results["good paraphrase (correct meaning, different words)"][0]
    wrong_detail_bleu = results["partial overlap, wrong detail"][0]
    print(f"\n-> THE HONEST DEMONSTRATION: the good paraphrase scores BLEU={paraphrase_bleu:.3f},")
    print(f"   LOWER than the factually-wrong-detail candidate's BLEU={wrong_detail_bleu:.3f},")
    print("   even though a human reader would call the paraphrase fully correct and the")
    print("   'cold floor' candidate factually wrong. Both metrics only count shared")
    print("   n-grams/subsequences -- they have no notion of meaning, so a candidate that")
    print("   reuses the reference's words (even to say something false) is scored higher")
    print("   than one that says the same true thing in different words. This is not a")
    print("   quirk of this toy implementation -- it is the documented, well-known failure")
    print("   mode of every surface-overlap metric, and the reason Lesson 3 (LLM-as-a-")
    print("   Judge) exists for evaluating open-ended generation.")


def qa_metrics_demo():
    print("\n" + "=" * 78)
    print("3. EXACT MATCH AND TOKEN F1 (EXTRACTIVE QA)")
    print("=" * 78)

    reference_answer = "Eiffel Tower"
    predictions = [
        "Eiffel Tower",
        "the Eiffel Tower",
        "Eiffel Tower in Paris",
        "Statue of Liberty",
    ]
    print(f"Reference answer: {reference_answer!r}\n")
    print(f"{'prediction':30} {'EM':>6} {'F1':>7}")
    print("-" * 45)
    for pred in predictions:
        em = exact_match(pred, reference_answer)
        f1 = token_f1(pred, reference_answer)
        print(f"{pred:30} {em:>6} {f1:>7.3f}")

    print("\n-> 'the Eiffel Tower' and 'Eiffel Tower in Paris' both fail Exact Match")
    print("   (EM=0) despite containing the fully correct answer, because EM demands a")
    print("   character-for-character match. Token F1 gives graded partial credit instead")
    print("   -- this is exactly why SQuAD-style QA leaderboards report BOTH metrics side")
    print("   by side rather than relying on EM alone.")


def perplexity_recap_demo():
    print("\n" + "=" * 78)
    print("4. PERPLEXITY RECAP (Phase 01 Lesson 1) -- SCORING THE MODEL, NOT THE OUTPUT")
    print("=" * 78)
    print("Note the difference in what's being measured: BLEU/ROUGE/F1 above compare a")
    print("GENERATED candidate against a REFERENCE. Perplexity instead scores how well a")
    print("model's own probability distribution predicts naturally-occurring text -- no")
    print("generated candidate or reference pair is involved at all.\n")

    lm = BigramLM(PPL_CORPUS)
    test_sentences = [
        "the cat sat on the mat",   # seen verbatim in training
        "the dog sat on the mat",   # plausible unseen recombination
        "a mat chased a log",       # unlikely / ungrammatical recombination
    ]
    for s in test_sentences:
        ppl = lm.perplexity(s)
        print(f"  {s!r:32} perplexity = {ppl:6.2f}")

    print("\n-> Lower perplexity = the model found the sentence less surprising. The toy")
    print("   bigram model, trained only on this tiny corpus, still ranks the grammatical")
    print("   recombination below the nonsense one -- but it CANNOT be used to score the")
    print("   candidates from the BLEU/ROUGE demo above, because perplexity needs a")
    print("   probability model, not a pair of text strings. Perplexity and overlap")
    print("   metrics answer two different questions: 'is the model's distribution good?'")
    print("   vs. 'does this specific generated text match a reference?' -- both are")
    print("   needed, neither substitutes for the other.")


def main():
    bleu_rouge_demo()
    qa_metrics_demo()
    perplexity_recap_demo()


if __name__ == "__main__":
    main()
