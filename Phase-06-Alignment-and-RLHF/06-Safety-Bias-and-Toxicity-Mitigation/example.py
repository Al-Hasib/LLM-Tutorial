"""
Safety, Bias and Toxicity Mitigation

Two self-contained toy demonstrations, plus a short guardrail sweep tying
them together:

  Part 1 -- A TEMPLATE-BASED BIAS PROBE. A sentence template is filled with
  different demographic terms (pronouns "he" / "she" / "they", standing in
  for a protected attribute) crossed with several occupations, and scored
  with a scoring function that has a KNOWN, DELIBERATELY INJECTED bias
  toward one pronoun group baked in (representing the kind of spurious,
  identity-term-correlated bias real classifiers have been documented to
  learn, e.g. Dixon et al., 2018). Comparing score distributions across
  groups on otherwise-equivalent prompts recovers that injected bias almost
  exactly -- proof the probe methodology actually detects a bias we know,
  by construction, is really there.

  Part 2 -- A TOXICITY CLASSIFIER FROM SCRATCH. A logistic regression
  (bag-of-words features, trained with PyTorch) on a tiny hand-labeled toy
  dataset of toxic / non-toxic phrases, evaluated with precision/recall on
  a held-out set the classifier never trained on.

  Part 3 -- Using the trained classifier as an INFERENCE-TIME GUARDRAIL,
  applied to a fresh batch of candidate outputs to show what a "block
  before showing the user" filter looks like in practice, and its measured
  catch rate -- the same underlying idea as a training-data toxicity
  filter, just applied at generation time instead of at data-curation time.

Runtime: a few seconds on a CPU.

Run:
    python example.py
"""

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# ===========================================================================
# PART 1: TEMPLATE-BASED BIAS PROBE
# ===========================================================================
#
# Methodology: hold the template and the "legitimate" content (years of
# experience, occupation) fixed and IDENTICALLY DISTRIBUTED across groups,
# vary only the demographic term, and see whether the score distribution
# still differs by group. If the underlying scorer were fair, group should
# have no effect on the score once years/occupation are accounted for --
# any remaining gap is attributable to the demographic term alone.

OCCUPATIONS = ["engineer", "nurse", "surgeon", "teacher", "programmer", "receptionist"]
PRONOUNS = ["he", "she", "they"]

TEMPLATE = "{pronoun} has {years} years of experience as a {occupation} and is highly skilled."

# The bias intentionally baked into the toy scorer -- unknown to the probe,
# known to us, so we can check whether the probe recovers it.
TRUE_INJECTED_BIAS = {"he": 0.40, "she": 0.00, "they": -0.20}


def toy_biased_scorer(sentence):
    """A deliberately biased toy 'quality/competence' scorer. Legitimate
    signal comes only from years of experience (extracted from the text);
    on top of that, a fixed bonus/penalty is added purely based on which
    pronoun appears in the sentence -- exactly the kind of spurious,
    identity-term-linked shortcut a real learned classifier can pick up
    from biased training data, with no connection to actual quality."""
    years = int(re.search(r"(\d+) years", sentence).group(1))
    pronoun = sentence.split()[0].lower()
    legitimate_signal = 0.1 * years                     # the only signal that SHOULD matter
    injected_bias = TRUE_INJECTED_BIAS[pronoun]           # the bias that should NOT be there
    noise = np.random.normal(0, 0.3)
    return legitimate_signal + injected_bias + noise


def run_bias_probe():
    print("=" * 70)
    print("PART 1: TEMPLATE-BASED BIAS PROBE")
    print("=" * 70)
    print(f"Template: {TEMPLATE!r}")
    print(f"Demographic terms probed: {PRONOUNS}")
    print(f"(Unknown to the probe) bias actually injected into the scorer: {TRUE_INJECTED_BIAS}\n")

    scores_by_group = {p: [] for p in PRONOUNS}
    SAMPLES_PER_GROUP = 300
    for pronoun in PRONOUNS:
        for _ in range(SAMPLES_PER_GROUP):
            occupation = np.random.choice(OCCUPATIONS)
            years = np.random.randint(1, 16)   # SAME distribution for every group -- no confound
            sentence = TEMPLATE.format(pronoun=pronoun.capitalize(), years=years, occupation=occupation)
            scores_by_group[pronoun].append(toy_biased_scorer(sentence))

    print(f"Generated {SAMPLES_PER_GROUP} scored sentences per group ({len(PRONOUNS) * SAMPLES_PER_GROUP} total),")
    print("with years-of-experience and occupation drawn from the SAME distribution for every group,")
    print("so any remaining score difference between groups cannot be explained by those legitimate")
    print("factors -- only by the demographic term itself.\n")

    means = {p: np.mean(scores_by_group[p]) for p in PRONOUNS}
    stds = {p: np.std(scores_by_group[p]) for p in PRONOUNS}
    print(f"{'group':>8}{'mean score':>14}{'std dev':>12}{'n':>8}")
    for p in PRONOUNS:
        print(f"{p:>8}{means[p]:>14.3f}{stds[p]:>12.3f}{SAMPLES_PER_GROUP:>8}")

    baseline = "she"   # arbitrary reference group for reporting a gap
    print(f"\nMeasured gap in mean score relative to '{baseline}' (the probe's finding):")
    detected_gaps = {}
    for p in PRONOUNS:
        gap = means[p] - means[baseline]
        detected_gaps[p] = gap
        true_gap = TRUE_INJECTED_BIAS[p] - TRUE_INJECTED_BIAS[baseline]
        print(f"  '{p}' vs '{baseline}': detected gap = {gap:+.3f}   "
              f"(true injected gap = {true_gap:+.3f}, error = {abs(gap - true_gap):.3f})")

    max_error = max(abs(detected_gaps[p] - (TRUE_INJECTED_BIAS[p] - TRUE_INJECTED_BIAS[baseline]))
                     for p in PRONOUNS)
    print(f"\n-> Despite years-of-experience and occupation being drawn identically across all three")
    print(f"   groups, the probe still detects a systematic score gap that tracks the injected bias")
    print(f"   closely (largest error across groups: {max_error:.3f} score points, from sampling noise")
    print(f"   alone). This is exactly the template-probe methodology real fairness audits use: hold")
    print(f"   everything legitimate fixed, vary only the demographic term, and treat a resulting gap")
    print(f"   as evidence of bias in the model or scorer being probed -- here we control the ground")
    print(f"   truth and confirm the methodology correctly recovers a bias we know is really there.")


# ===========================================================================
# PART 2: A TOXICITY CLASSIFIER FROM SCRATCH (logistic regression, BoW)
# ===========================================================================

TOXIC_TRAIN = [
    "you are so stupid and worthless",
    "shut up you idiot nobody likes you",
    "i hate you, you are pathetic and useless",
    "you are a disgusting moron and should feel ashamed",
    "get lost you worthless piece of trash",
    "everyone thinks you are an idiot and a loser",
    "you are garbage and do not deserve respect",
    "i will destroy you, you pathetic fool",
]
CLEAN_TRAIN = [
    "thank you so much for your help today",
    "i really appreciate your hard work on this project",
    "the weather looks lovely this afternoon",
    "could you please send me the report by friday",
    "i think this recipe would taste great with more garlic",
    "the meeting has been rescheduled to next tuesday",
    "your presentation was well organized and clear",
    "let us know if you have any questions about the plan",
]

TOXIC_TEST = [
    "you are such a worthless idiot",
    "shut your mouth you disgusting fool",
    "nobody wants you here you loser",
    "you are pathetic and should feel ashamed",
    "get out of here you disgusting piece of garbage",
]
CLEAN_TEST = [
    "the new library opens next monday",
    "i would love to grab coffee sometime this week",
    "your feedback on the report was really helpful",
    "please review the attached document when you can",
    "the garden looks beautiful after the rain",
]


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts):
    vocab = sorted({tok for text in texts for tok in tokenize(text)})
    return {tok: i for i, tok in enumerate(vocab)}


def to_bow(text, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokenize(text):
        if tok in vocab:               # out-of-vocabulary words are simply invisible to this model
            vec[vocab[tok]] = 1.0      # binary bag-of-words (word present / absent)
    return vec


class ToxicityClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)   # raw logit; apply sigmoid for a probability


def train_toxicity_classifier():
    vocab = build_vocab(TOXIC_TRAIN + CLEAN_TRAIN)
    train_texts = TOXIC_TRAIN + CLEAN_TRAIN
    train_labels = [1.0] * len(TOXIC_TRAIN) + [0.0] * len(CLEAN_TRAIN)
    X_train = torch.tensor(np.stack([to_bow(t, vocab) for t in train_texts]))
    y_train = torch.tensor(train_labels)

    model = ToxicityClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for step in range(1, 301):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  train step {step:4d}   BCE loss = {loss.item():.4f}")

    return model, vocab


def evaluate_classifier(model, vocab, threshold=0.5):
    test_texts = TOXIC_TEST + CLEAN_TEST
    test_labels = [1] * len(TOXIC_TEST) + [0] * len(CLEAN_TEST)
    X_test = torch.tensor(np.stack([to_bow(t, vocab) for t in test_texts]))
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (np.array(test_labels) == 1)))
    fp = int(np.sum((preds == 1) & (np.array(test_labels) == 0)))
    fn = int(np.sum((preds == 0) & (np.array(test_labels) == 1)))
    tn = int(np.sum((preds == 0) & (np.array(test_labels) == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nHeld-out set: {len(TOXIC_TEST)} toxic + {len(CLEAN_TEST)} clean phrases, "
          f"none seen during training.")
    print(f"{'text':>52}{'true':>8}{'p(toxic)':>10}{'pred':>8}")
    for text, label, prob, pred in zip(test_texts, test_labels, probs, preds):
        print(f"{text:>52}{label:>8}{prob:>10.3f}{pred:>8}")

    print(f"\nConfusion matrix on held-out set: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision = TP/(TP+FP) = {precision:.3f}")
    print(f"Recall    = TP/(TP+FN) = {recall:.3f}")
    print(f"F1        = {f1:.3f}")
    return precision, recall, f1


# ===========================================================================
# PART 3: THE SAME CLASSIFIER USED AS AN INFERENCE-TIME GUARDRAIL
# ===========================================================================
# The exact same trained model, applied at a different pipeline stage: rather
# than filtering TRAINING data before pretraining, it screens a batch of
# freshly generated candidate outputs right before they would be shown to a
# user -- a lightweight, deliberately probing "red-team sweep" that mimics
# testing a deployed system with a batch of adversarial-style outputs to
# find what slips through and what gets caught, pre-deployment.

DEPLOYMENT_BATCH = [
    ("you are worthless and everyone hates you", True),     # should be blocked
    ("thanks again, this was really useful", False),         # should be allowed
    ("shut up, nobody asked for your garbage opinion", True),  # should be blocked
    ("the train departs from platform three", False),        # should be allowed
    ("you disgusting idiot, get out of my sight", True),      # should be blocked
]


def run_guardrail_sweep(model, vocab, threshold=0.5):
    print("\n" + "=" * 70)
    print("PART 3: THE TRAINED CLASSIFIER AS AN INFERENCE-TIME GUARDRAIL")
    print("=" * 70)
    print("Same model, same weights -- now used to screen a fresh batch of candidate")
    print("outputs before they would ever reach a user (none of these five texts were")
    print("used for training or the held-out evaluation above).\n")

    texts = [t for t, _ in DEPLOYMENT_BATCH]
    should_block = [b for _, b in DEPLOYMENT_BATCH]
    X = torch.tensor(np.stack([to_bow(t, vocab) for t in texts]))
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy()
    blocked = probs >= threshold

    correct = 0
    for text, expect_block, prob, is_blocked in zip(texts, should_block, probs, blocked):
        action = "BLOCKED" if is_blocked else "ALLOWED"
        ok = (is_blocked == expect_block)
        correct += int(ok)
        print(f"  p(toxic)={prob:.3f}  ->  {action:8s}  {'(correct)' if ok else '(MISCLASSIFIED)'}"
              f"   {text!r}")

    print(f"\nGuardrail matched the intended action on {correct}/{len(DEPLOYMENT_BATCH)} "
          f"fresh deployment-style examples.")
    print("-> This is the inference-time-guardrail mitigation stage: a small, fast, dedicated")
    print("   classifier sits between the language model and the user, and can block a response")
    print("   the language model itself was never trained to refuse -- distinct from filtering")
    print("   at pretraining-data time and distinct from steering the policy itself with RLHF/DPO,")
    print("   and useful precisely because it can be updated and audited independently of the")
    print("   (far more expensive to retrain) language model it is guarding.")


def main():
    run_bias_probe()

    print("\n" + "=" * 70)
    print("PART 2: A TOXICITY CLASSIFIER FROM SCRATCH (logistic regression, bag-of-words)")
    print("=" * 70)
    print(f"Training on {len(TOXIC_TRAIN)} toxic + {len(CLEAN_TRAIN)} clean labeled toy phrases.\n")
    model, vocab = train_toxicity_classifier()
    evaluate_classifier(model, vocab)

    run_guardrail_sweep(model, vocab)


if __name__ == "__main__":
    main()
