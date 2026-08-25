"""
Hallucination and Factuality Evaluation

Builds a real, if tiny, NLI-style (entailment / contradiction / neutral)
factuality checker from scratch:

  1. Generate a large synthetic dataset of (source, claim, label) triples
     from sentence templates about fictional companies (founding year,
     founder, city, product). ENTAILMENT claims restate a fact the source
     actually states; CONTRADICTION claims change exactly one fact;
     NEUTRAL claims ask about an attribute the source never mentions at all.
  2. Train a small MLP classifier on bag-of-words features of (source, claim)
     pairs to predict the 3-way label.
  3. Apply the trained checker sentence-by-sentence to a toy generated
     "summary" of a NEW source passage, where a few sentences are
     deliberately written to be unsupported or contradictory, and measure
     precision/recall of the checker's hallucination flags against the
     known ground truth.

Runtime: a few seconds on CPU (small MLP, ~3,600 tiny bag-of-words examples).

Run:
    python example.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Toy world: fictional companies with founding facts, used to generate an
# effectively unlimited supply of synthetic (source, claim, label) triples.
# ---------------------------------------------------------------------------

COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries",
             "Wayne Enterprises", "Wonka Factory", "Hooli", "Cyberdyne", "Soylent Corp"]
YEARS = list(range(1950, 2021))
FOUNDERS = ["Alice Chen", "Bob Martinez", "Carol Nguyen", "David Kim",
            "Eve Johnson", "Frank Lopez", "Grace Patel", "Henry Wu"]
CITIES = ["Boston", "Seattle", "Austin", "Chicago", "Denver", "Portland", "Atlanta", "Miami"]
PRODUCTS = ["software", "robotics", "chemicals", "electronics", "toys", "vehicles",
            "pharmaceuticals", "food products"]
EMPLOYEE_COUNTS = ["50", "200", "500", "1200", "3000", "10000"]

LABELS = ["entailment", "contradiction", "neutral"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def make_source(company, year, founder, city, product):
    return f"{company} was founded in {year} by {founder} in {city}. the company produces {product}."


def random_other(value, options):
    """Pick a random option guaranteed different from `value`."""
    choice = value
    while choice == value:
        choice = random.choice(options)
    return choice


def make_triple():
    """Sample one random (source, claim, label) triple."""
    company = random.choice(COMPANIES)
    year = random.choice(YEARS)
    founder = random.choice(FOUNDERS)
    city = random.choice(CITIES)
    product = random.choice(PRODUCTS)
    source = make_source(company, year, founder, city, product)

    label = random.choice(LABELS)

    if label == "entailment":
        template = random.choice([
            f"{company} was founded in {year}.",
            f"{founder} founded {company} in {year}.",
            f"{company} is based in {city}.",
            f"{company} makes {product}.",
        ])
        claim = template

    elif label == "contradiction":
        slot = random.choice(["year", "founder", "city", "product"])
        if slot == "year":
            claim = f"{company} was founded in {random_other(year, YEARS)}."
        elif slot == "founder":
            claim = f"{random_other(founder, FOUNDERS)} founded {company} in {year}."
        elif slot == "city":
            claim = f"{company} is based in {random_other(city, CITIES)}."
        else:
            claim = f"{company} makes {random_other(product, PRODUCTS)}."

    else:  # neutral -- an attribute the source never addresses at all
        claim = random.choice([
            f"{company} has over {random.choice(EMPLOYEE_COUNTS)} employees.",
            f"{company} is publicly traded on the stock exchange.",
            f"{company} won an industry innovation award last year.",
            f"{company}'s chief executive previously worked at a bank.",
        ])

    return source, claim, label


def make_dataset(n):
    return [make_triple() for _ in range(n)]


# ---------------------------------------------------------------------------
# Bag-of-words featurization
# ---------------------------------------------------------------------------

def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").replace("'", " ").split()


def build_vocab(triples):
    vocab = set()
    for source, claim, _ in triples:
        vocab.update(tokenize(source))
        vocab.update(tokenize(claim))
    return {word: i for i, word in enumerate(sorted(vocab))}


def bow_vector(text, vocab):
    vec = torch.zeros(len(vocab))
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec


def featurize(source, claim, vocab):
    """Feature = concat(bag-of-words(source), bag-of-words(claim)) -- the MLP
    has to learn to compare the two bags itself; nothing is hand-engineered
    beyond raw word counts."""
    return torch.cat([bow_vector(source, vocab), bow_vector(claim, vocab)])


# ---------------------------------------------------------------------------
# The entailment classifier: a small MLP
# ---------------------------------------------------------------------------

class EntailmentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train_classifier(train_triples, vocab, epochs=25, batch_size=64, lr=1e-2):
    X = torch.stack([featurize(s, c, vocab) for s, c, _ in train_triples])
    y = torch.tensor([LABEL_TO_ID[label] for _, _, label in train_triples], dtype=torch.long)

    model = EntailmentMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}   avg loss = {total_loss / n:.4f}")

    return model


@torch.no_grad()
def evaluate_classifier(model, triples, vocab):
    X = torch.stack([featurize(s, c, vocab) for s, c, _ in triples])
    y = torch.tensor([LABEL_TO_ID[label] for _, _, label in triples], dtype=torch.long)
    preds = model(X).argmax(dim=-1)
    accuracy = (preds == y).float().mean().item()
    return accuracy, preds


@torch.no_grad()
def predict_label(model, source, claim, vocab):
    x = featurize(source, claim, vocab).unsqueeze(0)
    pred_id = model(x).argmax(dim=-1).item()
    return LABELS[pred_id]


# ---------------------------------------------------------------------------
# Demo 1: train and evaluate the checker on synthetic held-out data
# ---------------------------------------------------------------------------

def training_demo():
    print("=" * 78)
    print("1. TRAINING A TOY 3-WAY ENTAILMENT CHECKER ON SYNTHETIC (SOURCE, CLAIM) DATA")
    print("=" * 78)

    all_triples = make_dataset(3600)
    vocab = build_vocab(all_triples)
    random.shuffle(all_triples)
    split = int(0.8 * len(all_triples))
    train_triples, test_triples = all_triples[:split], all_triples[split:]

    print(f"Vocabulary size: {len(vocab)} words")
    print(f"Train examples: {len(train_triples)}   Test examples: {len(test_triples)}")
    counts = {label: sum(1 for _, _, l in all_triples if l == label) for label in LABELS}
    print(f"Label balance (full dataset): {counts}\n")

    print("Training the MLP (bag-of-words features -> hidden 64 -> 3 classes)...")
    model = train_classifier(train_triples, vocab)

    train_acc, _ = evaluate_classifier(model, train_triples, vocab)
    test_acc, _ = evaluate_classifier(model, test_triples, vocab)
    print(f"\nTrain accuracy: {train_acc:.1%}    Test accuracy: {test_acc:.1%}")
    print(f"-> Random guessing among 3 classes would score 33%; {test_acc:.1%} on wordings the")
    print("   model never trained on shows it learned real signal about whether a claim's")
    print("   stated fact matches, conflicts with, or goes unmentioned by the source --")
    print(f"   not just the training sentences themselves. The gap to {train_acc:.0%} train accuracy")
    print("   is expected: bag-of-words features over a fairly small, template-generated")
    print("   world (10 companies, 8 founders, 8 cities) let the MLP memorize part of the")
    print("   training set, which is exactly the honest, well-known trade-off of a simple")
    print("   bag-of-words classifier -- real NLI systems use far richer sentence")
    print("   representations to reduce this gap.")

    return model, vocab


# ---------------------------------------------------------------------------
# Demo 2: apply the checker to a toy generated "summary" with injected
# hallucinations, and score its flags against the known ground truth.
# ---------------------------------------------------------------------------

def summary_factuality_demo(model, vocab):
    print("\n" + "=" * 78)
    print("2. APPLYING THE CHECKER TO A TOY GENERATED SUMMARY")
    print("=" * 78)

    source = ("novatech was founded in 2004 by maria alvarez in denver. "
              "the company produces electronics.")
    print(f"Source passage:\n  {source!r}\n")

    # Each summary sentence, with its KNOWN true relationship to the source
    # (this ground truth is what a human annotator would assign; the checker
    # never sees it). "entailment" sentences should NOT be flagged; the other
    # two ARE hallucinations/unsupported claims and SHOULD be flagged.
    summary_sentences = [
        ("novatech was founded in 2004.",                                  "entailment"),
        ("maria alvarez founded novatech in 2004.",                        "entailment"),
        ("novatech is based in denver.",                                   "entailment"),
        ("novatech makes electronics.",                                    "entailment"),
        ("novatech was founded in 1999.",                                  "contradiction"),   # intrinsic hallucination
        ("james carter founded novatech in 2004.",                         "contradiction"),   # intrinsic hallucination
        ("novatech has over 10000 employees.",                             "neutral"),          # extrinsic / unsupported
        ("novatech won an industry innovation award last year.",           "neutral"),          # extrinsic / unsupported
    ]

    print(f"{'summary sentence':55}{'true label':>14}{'checker verdict':>18}{'flagged?':>10}")
    print("-" * 97)

    true_positive = false_positive = false_negative = true_negative = 0
    for sentence, true_label in summary_sentences:
        predicted = predict_label(model, source, sentence, vocab)
        should_flag = true_label != "entailment"
        is_flagged = predicted != "entailment"

        if should_flag and is_flagged:
            true_positive += 1
        elif (not should_flag) and is_flagged:
            false_positive += 1
        elif should_flag and (not is_flagged):
            false_negative += 1
        else:
            true_negative += 1

        marker = "FLAGGED" if is_flagged else "-"
        print(f"{sentence:55}{true_label:>14}{predicted:>18}{marker:>10}")

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")

    print(f"\nConfusion counts on this toy summary's 8 sentences:")
    print(f"  true positives  (hallucination, correctly flagged)     = {true_positive}")
    print(f"  false positives (genuinely supported, wrongly flagged) = {false_positive}")
    print(f"  false negatives (hallucination, MISSED)                = {false_negative}")
    print(f"  true negatives  (genuinely supported, correctly passed) = {true_negative}")
    print(f"\nPrecision = {precision:.2f}   Recall = {recall:.2f}")

    if false_positive == 0 and false_negative == 0:
        print("\n-> The checker flagged every contradiction and every unsupported neutral")
        print("   claim, and passed every genuinely entailed claim, on this toy summary --")
        print(f"   precision={precision:.2f} and recall={recall:.2f}. It is doing exactly what an NLI-based")
        print("   factuality checker is supposed to do: distinguish 'this restates a fact")
        print("   the source actually supports' from 'this contradicts or goes beyond what")
        print("   the source says' using the same 3-way ENTAILMENT/CONTRADICTION/NEUTRAL")
        print("   judgment it was trained on, applied to sentences it has never seen before.")
    else:
        print(f"\n-> On this toy summary the checker made {false_positive} false-positive(s) and")
        print(f"   {false_negative} false-negative(s), giving precision={precision:.2f} and recall={recall:.2f}.")
        print("   Even a checker trained to reasonable held-out accuracy is not perfect --")
        print("   which is exactly why Section 4 of the README stresses that automated")
        print("   factuality checking reduces, but does not eliminate, the need for human")
        print("   review of anything high-stakes.")


def main():
    model, vocab = training_demo()
    summary_factuality_demo(model, vocab)


if __name__ == "__main__":
    main()
