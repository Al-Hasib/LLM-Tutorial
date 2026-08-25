"""
Chain-of-Thought and Reasoning Prompts: Self-Consistency, Rigorously

Chain-of-Thought (Wei et al., 2022) and zero-shot CoT (Kojima et al., 2022,
"Let's think step by step") are about *eliciting* an intermediate reasoning
trace from a model via prompting alone. Whether spelling out intermediate
steps actually helps depends on the specific model and task in a way that
cannot be honestly reduced to a small offline simulation without either
faking an API call or reinventing an LLM -- so this script does not attempt
to simulate "does writing out steps help." Instead it focuses on the part of
the CoT toolkit that IS a precise, provable, fully self-contained statistical
claim: Wang et al.'s (2022) SELF-CONSISTENCY method, which samples several
independent reasoning paths for the SAME question (e.g. via temperature > 0)
and takes a majority vote over their final answers instead of trusting one
greedy decode.

We model one reasoning sample as a Bernoulli trial: it lands on the correct
final answer with probability p, independent of every other sample (this is
exactly the assumption self-consistency relies on -- sampled reasoning paths
must be independent enough that their errors don't all point the same way).
We then prove -- both with the exact binomial formula AND with a Monte Carlo
simulation that must agree with it -- a Condorcet-Jury-Theorem-style result:

    * If p > 0.5, majority voting across k independent samples has HIGHER
      accuracy than a single sample, strictly increasing towards 1.0 as
      k grows.
    * If p < 0.5, majority voting makes things WORSE, strictly decreasing
      towards 0.0 as k grows -- voting amplifies whichever way the bias
      already leans, for better or worse.
    * If p == 0.5 exactly, voting changes nothing -- there is no signal to
      amplify.

Finally we go one step closer to how self-consistency behaves on real CoT
tasks: instead of one single "wrong" answer competing with the correct one,
wrong reasoning paths land on several DIFFERENT wrong final answers (a model
rarely makes the exact same arithmetic slip twice). We simulate plurality
voting over such a fragmented answer space and show it outperforms the
binary case at the same p, for the same intuitive reason political theorists
know as vote-splitting: a fragmented opposition is easier for the plurality
to beat than a unified one.

Runtime: a few seconds on CPU (pure Python + Monte Carlo, no model training).

Run:
    python example.py
"""

import math
import random
from collections import Counter

random.seed(0)


# ---------------------------------------------------------------------------
# 1. Exact binomial formula for binary majority voting
# ---------------------------------------------------------------------------

def majority_accuracy_exact(p, k):
    """Probability that a majority of k independent Bernoulli(p) trials are
    'correct', for ODD k (so there are no ties to break). This is exactly
    the Condorcet Jury Theorem's quantity: k independent voters, each right
    with probability p, majority rule."""
    assert k % 2 == 1, "use odd k to avoid ties"
    half = k // 2
    total = 0.0
    for i in range(half + 1, k + 1):
        total += math.comb(k, i) * (p ** i) * ((1 - p) ** (k - i))
    return total


# ---------------------------------------------------------------------------
# 2. Monte Carlo simulation of the same quantity, to validate the simulator
#    methodology before trusting it on the harder multi-way case in step 5.
# ---------------------------------------------------------------------------

def simulate_binary_majority(p, k, num_trials):
    correct = 0
    for _ in range(num_trials):
        votes = [1 if random.random() < p else 0 for _ in range(k)]
        majority = 1 if sum(votes) > k / 2 else 0
        correct += majority
    return correct / num_trials


# ---------------------------------------------------------------------------
# 3. Multi-way plurality voting: wrong answers are spread across several
#    distinct wrong options instead of consolidating into one "other" bucket
#    -- a closer analogue of real self-consistency over free-form answers
#    (numbers, expressions) where two wrong reasoning paths rarely agree.
# ---------------------------------------------------------------------------

def simulate_plurality_vote(p, k, num_wrong_options, num_trials):
    """Each sample is 'CORRECT' w.p. p, else lands uniformly at random on one
    of num_wrong_options distinct wrong labels. Final answer = the label with
    the most votes among the k samples (ties broken uniformly at random)."""
    correct = 0
    wrong_labels = [f"wrong_{i}" for i in range(num_wrong_options)]
    for _ in range(num_trials):
        votes = []
        for _ in range(k):
            if random.random() < p:
                votes.append("CORRECT")
            else:
                votes.append(random.choice(wrong_labels))
        counts = Counter(votes)
        top_count = max(counts.values())
        winners = [label for label, c in counts.items() if c == top_count]
        winner = random.choice(winners)   # random tie-break
        correct += int(winner == "CORRECT")
    return correct / num_trials


def main():
    print("=" * 78)
    print("SELF-CONSISTENCY (Wang et al., 2022): MAJORITY VOTING OVER")
    print("INDEPENDENT REASONING SAMPLES, AS A CONDORCET-JURY-THEOREM RESULT")
    print("=" * 78)
    print("Model: each independently sampled reasoning path lands on the correct")
    print("final answer with probability p, and on a wrong answer otherwise.")
    print("Self-consistency draws k such samples and majority-votes the final")
    print("answer, instead of trusting a single greedy decode (k=1).")

    print("\n" + "=" * 78)
    print("1. EXACT FORMULA vs. MONTE CARLO SIMULATION (sanity check)")
    print("=" * 78)
    print("Before trusting simulation for the harder multi-way case below, we")
    print("confirm the Monte Carlo simulator agrees with the exact binomial")
    print("majority formula for the plain binary case.\n")
    NUM_TRIALS = 200_000
    print(f"{'p':>6}{'k':>6}{'exact':>12}{'simulated':>14}{'abs diff':>12}")
    for p_check, k_check in [(0.6, 5), (0.6, 15), (0.3, 5), (0.9, 7)]:
        exact = majority_accuracy_exact(p_check, k_check)
        sim = simulate_binary_majority(p_check, k_check, NUM_TRIALS)
        print(f"{p_check:>6.2f}{k_check:>6d}{exact:>12.4f}{sim:>14.4f}{abs(exact - sim):>12.4f}")
    print("\n-> Exact formula and simulation agree to within Monte Carlo noise")
    print("   (a few thousandths at 200,000 trials). The simulator is trustworthy.")

    print("\n" + "=" * 78)
    print("2. THE HELPFUL CASE: p > 0.5 -- voting drives accuracy UP towards 1.0")
    print("=" * 78)
    k_values = [1, 3, 5, 7, 15, 31, 51]
    good_ps = [0.55, 0.6, 0.7, 0.9]
    header = f"{'k':>6}  " + "".join(f"p={p:<10.2f}" for p in good_ps)
    print(header)
    up_trend_holds = True
    prev_row = None
    for k in k_values:
        row_vals = [majority_accuracy_exact(p, k) for p in good_ps]
        print(f"{k:>6}" + "".join(f"{v:>12.4f}" for v in row_vals))
        if prev_row is not None:
            for v_prev, v_now in zip(prev_row, row_vals):
                if v_now < v_prev - 1e-12:
                    up_trend_holds = False
        prev_row = row_vals
    print(f"\n-> For every p > 0.5 tested, accuracy at k=1 equals p itself, and rises")
    print(f"   MONOTONICALLY as k grows (confirmed: {up_trend_holds}), approaching 1.0.")
    print("   Even a weak reasoner (p=0.55, barely better than a coin flip per")
    print(f"   sample) reaches {majority_accuracy_exact(0.55, 51):.3f} accuracy at k=51 samples --")
    print("   this is the entire mechanism behind self-consistency's reported gains.")

    print("\n" + "=" * 78)
    print("3. THE BOUNDARY CASE: p == 0.5 exactly -- voting changes NOTHING")
    print("=" * 78)
    for k in [1, 5, 15, 51]:
        acc = majority_accuracy_exact(0.5, k)
        print(f"  k={k:<4d} accuracy = {acc:.4f}")
    print("\n-> At p=0.5 there is no signal to amplify -- each sample is a fair coin")
    print("   flip between right and wrong, and the majority of fair coin flips is")
    print("   itself just another fair coin flip. Accuracy stays pinned at 0.500")
    print("   for every k. Voting can only amplify a bias that already exists; it")
    print("   cannot manufacture one out of pure noise.")

    print("\n" + "=" * 78)
    print("4. THE HARMFUL CASE: p < 0.5 -- voting drives accuracy DOWN towards 0.0")
    print("=" * 78)
    bad_ps = [0.45, 0.4, 0.3, 0.1]
    header = f"{'k':>6}  " + "".join(f"p={p:<10.2f}" for p in bad_ps)
    print(header)
    down_trend_holds = True
    prev_row = None
    for k in k_values:
        row_vals = [majority_accuracy_exact(p, k) for p in bad_ps]
        print(f"{k:>6}" + "".join(f"{v:>12.4f}" for v in row_vals))
        if prev_row is not None:
            for v_prev, v_now in zip(prev_row, row_vals):
                if v_now > v_prev + 1e-12:
                    down_trend_holds = False
        prev_row = row_vals
    print(f"\n-> For every p < 0.5 tested, accuracy DECREASES monotonically as k grows")
    print(f"   (confirmed: {down_trend_holds}), approaching 0.0. This is the honest boundary")
    print("   condition of self-consistency: majority voting is not a free lunch --")
    print("   it amplifies whatever the per-sample accuracy already leans towards.")
    print("   If a reasoning strategy is WORSE than a coin flip per sample (a genuinely")
    print("   confusing or adversarial prompt can do this), sampling more and voting")
    print("   makes the final answer reliably WORSE, not better. Self-consistency is")
    print("   only worth applying when there is reason to believe p > 0.5 to begin")
    print("   with -- e.g. the base reasoning method already beats chance on the task.")

    print("\n" + "=" * 78)
    print("5. CLOSER TO REAL COT: WRONG ANSWERS SPLIT ACROSS MANY WRONG OPTIONS")
    print("=" * 78)
    print("Real free-form answers (a number, an expression) rarely repeat a wrong")
    print("value exactly -- different flawed reasoning paths tend to land on")
    print("DIFFERENT wrong answers, not one single competing wrong answer. We")
    print("simulate this by spreading the (1-p) wrong mass uniformly across")
    print("several distinct wrong labels, then take a PLURALITY vote.\n")
    p_demo = 0.4
    k_demo = 9
    NUM_TRIALS_PLURALITY = 100_000
    binary_acc = majority_accuracy_exact(p_demo, k_demo)
    print(f"p = {p_demo} (a MINORITY of samples are correct), k = {k_demo} samples")
    print(f"{'num_wrong_options':>20}{'plurality accuracy':>22}")
    print(f"{'1 (binary case)':>20}{binary_acc:>22.4f}   <- exact formula, for reference")
    plurality_accs = {}
    for num_wrong in [1, 2, 4, 8]:
        acc = simulate_plurality_vote(p_demo, k_demo, num_wrong, NUM_TRIALS_PLURALITY)
        plurality_accs[num_wrong] = acc
        print(f"{num_wrong:>20d}{acc:>22.4f}")

    fragmentation_helps = plurality_accs[8] > plurality_accs[1] + 0.02
    print(f"\n-> Even though p={p_demo} < 0.5 (a minority of samples are individually")
    print("   correct), spreading the wrong answers across more distinct options lets")
    print(f"   the single correct answer win the plurality more often: accuracy rises from")
    print(f"   {plurality_accs[1]:.3f} (1 wrong option, i.e. the binary case) to {plurality_accs[8]:.3f} (8 wrong")
    print(f"   options) at the same k and p (fragmentation helps: {fragmentation_helps}).")
    print("   This is vote-splitting: the correct answer is the single largest bloc")
    print("   because the WRONG mass is divided against itself. It is one honest")
    print("   reason self-consistency can outperform this script's own binary-case")
    print("   math on real tasks with open-ended (not binary) final answers --")
    print("   though note it does NOT rescue the p<0.5 case in general: with enough")
    print("   wrong options and low enough p, the wrong votes are still more numerous")
    print("   in total than the correct ones, just less concentrated on any one label.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("Wei et al. (2022) show that PROMPTING a model to produce intermediate")
    print("reasoning steps (chain-of-thought) can raise per-sample accuracy p on")
    print("multi-step problems in the first place. Kojima et al. (2022) show a")
    print("surprisingly simple zero-shot version of this ('Let's think step by")
    print("step') recovers much of the benefit without any worked examples at all.")
    print("Wang et al. (2022) then layer SELF-CONSISTENCY on top: given any")
    print("reasoning method with per-sample accuracy p > 0.5, independently")
    print("resampling and majority-voting provably pushes accuracy higher still,")
    print("exactly as proven and measured above -- with the honest caveat that the")
    print("same mechanism backfires if p is not actually above 0.5 to begin with.")


if __name__ == "__main__":
    main()
