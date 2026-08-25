"""
Reward Modeling

Implements the Bradley-Terry pairwise preference loss from scratch in
PyTorch and uses it to train a small reward model (RM) on SYNTHETIC
preference data generated from a KNOWN ground-truth scoring function --
which lets us directly verify correctness, something impossible with real
human labels. "Responses" here are toy feature vectors (standing in for
attributes like factual accuracy, relevance, politeness, appropriate
length); a fixed, hidden linear function of those features is the ground
truth "true quality." Human preference labels are simulated by SAMPLING
from the Bradley-Terry distribution over the true scores (not just always
picking the higher-scoring one) to mimic realistic, sometimes-noisy human
judgments. After training the RM on nothing but these pairwise comparisons
(it never sees a true score directly), we check that its learned scores
correlate strongly with the true, hidden ground truth on held-out items.

Runtime: a few seconds on a CPU.

Run:
    python example.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

FEATURE_DIM = 8   # a toy "response" is an 8-dim feature vector

# The ground-truth reward function: a fixed, hidden linear weighting of the
# features (e.g. w[0] might be "how factually accurate," w[1] "how polite,"
# etc. -- the RM will never see these weights, only pairwise comparisons of
# the resulting scores).
TRUE_WEIGHTS = torch.tensor([1.8, -1.2, 0.9, 0.5, -0.7, 1.1, -0.3, 0.6])


def true_reward(features):
    """The hidden ground-truth scorer. Never seen by the reward model --
    only used here to (a) generate preference labels and (b) grade the
    RM's learned ranking afterward."""
    return features @ TRUE_WEIGHTS


def make_items(n):
    return torch.rand(n, FEATURE_DIM) * 2 - 1   # features in [-1, 1]


def make_preference_pairs(items, num_pairs, label_noise=True):
    """Sample random pairs of items and generate a preference label for each.

    If label_noise=True, the "human" choice is SAMPLED from the Bradley-Terry
    probability P(i preferred over j) = sigmoid(true_r_i - true_r_j), exactly
    like real human labelers who are not perfectly consistent -- rather than
    deterministically always picking the higher-true-score item. This is a
    more honest simulation of real preference data.
    """
    n = items.shape[0]
    idx_a = torch.randint(0, n, (num_pairs,))
    idx_b = torch.randint(0, n, (num_pairs,))
    true_scores = true_reward(items)

    r_a, r_b = true_scores[idx_a], true_scores[idx_b]
    if label_noise:
        prob_a_wins = torch.sigmoid(r_a - r_b)
        a_wins = torch.bernoulli(prob_a_wins).bool()
    else:
        a_wins = (r_a > r_b)

    # Arrange as (chosen_idx, rejected_idx)
    chosen = torch.where(a_wins, idx_a, idx_b)
    rejected = torch.where(a_wins, idx_b, idx_a)
    return chosen, rejected


# ---------------------------------------------------------------------------
# 1. The reward model: a small MLP scalar head, standing in for "pretrained
# backbone + scalar head" (Lesson README section 4) -- the Bradley-Terry
# mechanics are identical regardless of what produces the pooled features.
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,) scalar reward per item


def bradley_terry_loss(reward_model, items, chosen_idx, rejected_idx):
    """loss = -log( sigmoid( r(chosen) - r(rejected) ) ), averaged over the batch.
    This is exactly the negative log-likelihood of the observed human choices
    under the Bradley-Terry model (Lesson README section 3)."""
    r_chosen = reward_model(items[chosen_idx])
    r_rejected = reward_model(items[rejected_idx])
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# ---------------------------------------------------------------------------
# 2. Evaluation utilities: verify the learned RM against the KNOWN ground
# truth, something real RLHF pipelines can never directly do.
# ---------------------------------------------------------------------------

def spearman_correlation(a, b):
    """Manual Spearman rank correlation (no scipy dependency): correlate the
    RANKS of two score vectors rather than the raw values, which is exactly
    what we want since the RM's raw scale is meaningless (README section 2) --
    only the ORDERING it induces should match the ground truth."""
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks

    ra, rb = rank(a), rank(b)
    return np.corrcoef(ra, rb)[0, 1]


def pairwise_accuracy(reward_model, items, true_scores, num_test_pairs=2000):
    """Sample random held-out pairs and check how often the RM's PREFERENCE
    (which of the two it scores higher) agrees with the ground truth's
    preference -- the most direct measure of "did it learn the right thing
    to prefer," independent of any raw score scale or offset."""
    n = items.shape[0]
    idx_a = torch.randint(0, n, (num_test_pairs,))
    idx_b = torch.randint(0, n, (num_test_pairs,))
    with torch.no_grad():
        rm_scores = reward_model(items)
    rm_a, rm_b = rm_scores[idx_a], rm_scores[idx_b]
    true_a, true_b = true_scores[idx_a], true_scores[idx_b]

    rm_prefers_a = rm_a > rm_b
    true_prefers_a = true_a > true_b
    agree = (rm_prefers_a == true_prefers_a).float().mean().item()
    return agree


def main():
    print("=" * 70)
    print("1. SYNTHETIC PREFERENCE DATA FROM A KNOWN GROUND-TRUTH SCORER")
    print("=" * 70)
    print(f"Ground truth is a hidden linear function of {FEATURE_DIM} toy features")
    print(f"(true weights, NEVER shown to the reward model): "
          f"{[round(w, 2) for w in TRUE_WEIGHTS.tolist()]}")

    train_items = make_items(400)
    test_items = make_items(300)
    test_true_scores = true_reward(test_items)

    NUM_PAIRS = 3000
    chosen_idx, rejected_idx = make_preference_pairs(train_items, NUM_PAIRS, label_noise=True)

    # Sanity check: how often does the "noisy human" labeler actually agree
    # with the ground truth's own ranking? Real human labelers are not
    # perfectly consistent either -- this shows our simulation isn't trivial.
    true_scores_train = true_reward(train_items)
    label_matches_ground_truth = (
        true_scores_train[chosen_idx] > true_scores_train[rejected_idx]
    ).float().mean().item()
    print(f"\nGenerated {NUM_PAIRS} pairwise comparisons with simulated (noisy) human labels.")
    print(f"Fraction of labels that agree with the ground-truth ranking: "
          f"{label_matches_ground_truth:.3f}")
    print("-> Not 1.000 -- some comparisons are close calls where the simulated")
    print("   'human' picks the slightly-worse item, exactly like real labelers")
    print("   disagreeing on ambiguous cases (README section 5).")

    print("\n" + "=" * 70)
    print("2. TRAINING THE REWARD MODEL WITH THE BRADLEY-TERRY LOSS")
    print("=" * 70)
    print("loss = -log( sigmoid( r(chosen) - r(rejected) ) )")
    print("The RM NEVER sees a true score directly -- only which item won each pair.\n")

    reward_model = RewardModel(FEATURE_DIM)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)

    BATCH_SIZE = 64
    NUM_STEPS = 800
    for step in range(1, NUM_STEPS + 1):
        batch = torch.randint(0, NUM_PAIRS, (BATCH_SIZE,))
        loss = bradley_terry_loss(
            reward_model, train_items, chosen_idx[batch], rejected_idx[batch]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}   bradley-terry loss = {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("3. VERIFYING THE LEARNED REWARD MODEL AGAINST THE HIDDEN GROUND TRUTH")
    print("=" * 70)
    with torch.no_grad():
        learned_scores = reward_model(test_items).numpy()
    true_scores_np = test_true_scores.numpy()

    corr = spearman_correlation(learned_scores, true_scores_np)
    acc = pairwise_accuracy(reward_model, test_items, test_true_scores)

    print(f"Spearman rank correlation between learned RM scores and the true")
    print(f"hidden ground-truth scores, on {test_items.shape[0]} held-out items: {corr:.3f}")
    print(f"Pairwise agreement between the RM's preferences and the ground")
    print(f"truth's preferences, over 2000 random held-out pairs: {acc:.3f}")

    print(f"\nFor comparison, the noisy simulated 'human' labels themselves only")
    print(f"agreed with ground truth {label_matches_ground_truth:.3f} of the time on")
    print(f"TRAINING pairs -- yet the trained RM's pairwise agreement with ground")
    print(f"truth on held-out data is {acc:.3f}. Averaging the Bradley-Terry loss over")
    print(f"thousands of noisy comparisons recovers a scorer that is MORE accurate")
    print(f"than any single noisy label it was trained on -- exactly the effect that")
    print(f"makes reward modeling from imperfect human preferences work in practice.")

    print("\nSample of learned vs. true scores on 5 held-out items (raw values --")
    print("remember from README section 2 that only relative order is meaningful,")
    print("not the absolute scale, so the numbers themselves needn't match):")
    print(f"{'true score':>14}{'learned score':>16}")
    for i in range(5):
        print(f"{true_scores_np[i]:>14.3f}{learned_scores[i]:>16.3f}")


if __name__ == "__main__":
    main()
