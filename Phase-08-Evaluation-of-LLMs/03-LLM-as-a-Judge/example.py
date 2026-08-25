"""
LLM-as-a-Judge

A from-scratch, fully synthetic simulation of a "judge" scoring function with
THREE deliberate, quantifiable biases bolted on top of a hidden true-quality
signal:

  1. POSITION bias    -- a flat bonus added to whichever response is shown
                          in the first slot of the prompt.
  2. VERBOSITY bias   -- a bonus proportional to response length, regardless
                          of actual quality.
  3. SELF-PREFERENCE  -- a bonus a judge gives to a response that shares its
     bias              own "model family", regardless of actual quality.

Each demo below runs thousands of simulated pairwise comparisons and reports
the REAL measured effect of the bias on win rate, then applies the standard
mitigation from the README and reports the REAL recovered win rate. No LLM
API calls are made anywhere -- "judge" here means a small scoring function
with injected biases, standing in for a real biased LLM judge.

Run:
    python example.py
"""

import random

random.seed(0)

# ---------------------------------------------------------------------------
# The judge's raw scoring function: hidden true quality + injected biases + noise
# ---------------------------------------------------------------------------

def judge_score(quality, length, shown_first, judge_family, response_family,
                 position_bonus=0.0, verbosity_coef=0.0, self_pref_bonus=0.0,
                 noise_std=1.0):
    """A single response's score, as a biased judge would produce it.
    `quality` is the HIDDEN ground-truth goodness of the response -- something
    only this simulation knows, never the judge itself."""
    score = quality
    if shown_first:
        score += position_bonus
    score += verbosity_coef * length
    if response_family == judge_family:
        score += self_pref_bonus
    score += random.gauss(0.0, noise_std)
    return score


# ---------------------------------------------------------------------------
# DEMO 1: POSITION BIAS
# A harness that always shows the "New" model's response in slot 1 and the
# "Baseline" model's response in slot 2 -- a very natural (and very common)
# way to write an eval script. Both models are drawn from an IDENTICAL true
# quality distribution, so a fair judge should find New wins ~50% of the time.
# ---------------------------------------------------------------------------

def position_bias_demo():
    print("=" * 78)
    print("1. POSITION BIAS: same quality distribution, fixed slot order")
    print("=" * 78)

    n_trials = 5000
    position_bonus = 2.0
    noise_std = 1.5

    print(f"New and Baseline responses both drawn from the SAME true-quality")
    print(f"distribution (Normal(mean=6, std=1)) -- a fair judge should find New")
    print(f"wins essentially 50% of the time. position_bonus={position_bonus}, noise_std={noise_std}\n")

    naive_new_wins = 0
    mitigated_new_wins = 0
    mitigated_ties = 0

    for _ in range(n_trials):
        quality_new = random.gauss(6.0, 1.0)
        quality_base = random.gauss(6.0, 1.0)
        length_new = random.gauss(100, 15)
        length_base = random.gauss(100, 15)

        # --- naive harness: New always shown first ---
        score_new_naive = judge_score(quality_new, length_new, shown_first=True,
                                       judge_family="J", response_family="J",
                                       position_bonus=position_bonus, noise_std=noise_std)
        score_base_naive = judge_score(quality_base, length_base, shown_first=False,
                                        judge_family="J", response_family="J",
                                        position_bonus=position_bonus, noise_std=noise_std)
        if score_new_naive > score_base_naive:
            naive_new_wins += 1

        # --- mitigated: evaluate BOTH orderings, average the score difference ---
        # ordering 1: New first, Base second
        s_new_1 = judge_score(quality_new, length_new, shown_first=True,
                               judge_family="J", response_family="J",
                               position_bonus=position_bonus, noise_std=noise_std)
        s_base_1 = judge_score(quality_base, length_base, shown_first=False,
                                judge_family="J", response_family="J",
                                position_bonus=position_bonus, noise_std=noise_std)
        diff_1 = s_new_1 - s_base_1

        # ordering 2: Base first, New second
        s_base_2 = judge_score(quality_base, length_base, shown_first=True,
                                judge_family="J", response_family="J",
                                position_bonus=position_bonus, noise_std=noise_std)
        s_new_2 = judge_score(quality_new, length_new, shown_first=False,
                               judge_family="J", response_family="J",
                               position_bonus=position_bonus, noise_std=noise_std)
        diff_2 = s_new_2 - s_base_2

        avg_diff = (diff_1 + diff_2) / 2.0
        if avg_diff > 0:
            mitigated_new_wins += 1
        elif avg_diff == 0:
            mitigated_ties += 1

    naive_rate = naive_new_wins / n_trials
    mitigated_rate = mitigated_new_wins / n_trials

    print(f"{'protocol':45}{'New win rate':>18}")
    print("-" * 63)
    print(f"{'naive (New always shown first)':45}{naive_rate:>18.1%}")
    print(f"{'mitigated (swap orderings, average diff)':45}{mitigated_rate:>18.1%}")

    print(f"\n-> New and Baseline have IDENTICAL true-quality distributions, so the")
    print(f"   correct win rate is 50%. The naive fixed-order harness measured")
    print(f"   {naive_rate:.1%} purely from the +{position_bonus} position bonus New always")
    print(f"   receives for sitting in the first slot -- New looks meaningfully better")
    print(f"   than Baseline even though it isn't. Swapping the order and averaging the")
    print(f"   two score differences brought the measured rate to {mitigated_rate:.1%}, within")
    print(f"   simulation noise of the true 50%, because the position bonus is added")
    print(f"   with opposite sign to New in the two orderings and cancels out exactly.")


# ---------------------------------------------------------------------------
# DEMO 2: VERBOSITY BIAS
# Model "Concise" is truly better but writes short responses; model "Verbose"
# is truly worse but writes long ones. A judge with a length-reward term can
# be fooled into preferring the worse-but-longer model.
# ---------------------------------------------------------------------------

def linear_regression(xs, ys):
    """Closed-form OLS slope and intercept for y ~ alpha + beta*x."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    beta = cov / var
    alpha = mean_y - beta * mean_x
    return alpha, beta


def verbosity_bias_demo():
    print("\n" + "=" * 78)
    print("2. VERBOSITY BIAS: shorter-but-better vs. longer-but-worse")
    print("=" * 78)

    n_trials = 5000
    verbosity_coef = 0.018
    noise_std = 1.5

    print("Concise responses: true quality ~ Normal(7, 1), length ~ Normal(80, 10)")
    print("Verbose responses: true quality ~ Normal(5, 1), length ~ Normal(220, 20)")
    print(f"(Concise is genuinely 2 points better on average, but Verbose is far longer)")
    print(f"verbosity_coef={verbosity_coef}, noise_std={noise_std}, no position bias in this demo\n")

    oracle_wins = 0        # decision using TRUE QUALITY ONLY (+ same noise) -- the honest baseline
    naive_wins = 0         # decision using the verbosity-biased judge score
    corrected_wins = 0     # decision after regressing out the length effect

    records = []  # (score_concise, len_concise, score_verbose, len_verbose, oracle_win)

    for _ in range(n_trials):
        q_concise = random.gauss(7.0, 1.0)
        q_verbose = random.gauss(5.0, 1.0)
        len_concise = max(1.0, random.gauss(80, 10))
        len_verbose = max(1.0, random.gauss(220, 20))
        noise_c = random.gauss(0.0, noise_std)
        noise_v = random.gauss(0.0, noise_std)

        # Oracle: quality + noise only, NO verbosity term -- "what a fair judge would say"
        oracle_score_c = q_concise + noise_c
        oracle_score_v = q_verbose + noise_v
        oracle_win = oracle_score_c > oracle_score_v
        oracle_wins += int(oracle_win)

        # Naive judge: quality + verbosity bonus + SAME noise draws (isolates the bias effect)
        score_c = q_concise + verbosity_coef * len_concise + noise_c
        score_v = q_verbose + verbosity_coef * len_verbose + noise_v
        naive_win = score_c > score_v
        naive_wins += int(naive_win)

        records.append((score_c, len_concise, score_v, len_verbose))

    # Fit the judge's own revealed length preference across ALL graded responses
    # (both Concise's and Verbose's scores and lengths pooled together), then
    # subtract the fitted length effect back out of every score.
    all_scores = [r[0] for r in records] + [r[2] for r in records]
    all_lengths = [r[1] for r in records] + [r[3] for r in records]
    alpha, beta = linear_regression(all_lengths, all_scores)

    for score_c, len_c, score_v, len_v in records:
        residual_c = score_c - (alpha + beta * len_c)
        residual_v = score_v - (alpha + beta * len_v)
        corrected_wins += int(residual_c > residual_v)

    oracle_rate = oracle_wins / n_trials
    naive_rate = naive_wins / n_trials
    corrected_rate = corrected_wins / n_trials

    print(f"Fitted judge length preference (pooled OLS): score ~ {alpha:.3f} + {beta:.5f} * length")
    print(f"(the TRUE injected coefficient was {verbosity_coef} -- the fit undershoots it, because")
    print(f"Concise responses are simultaneously SHORTER and HIGHER quality, so some of the")
    print(f"genuine quality signal looks, to a pooled regression, like 'a negative length")
    print(f"relationship' and partly masks the true verbosity effect being estimated)\n")

    print(f"{'protocol':45}{'Concise (truly better) win rate':>34}")
    print("-" * 79)
    print(f"{'oracle (true quality only, no bias)':45}{oracle_rate:>34.1%}")
    print(f"{'naive (verbosity-biased judge)':45}{naive_rate:>34.1%}")
    print(f"{'corrected (length regressed out)':45}{corrected_rate:>34.1%}")

    print(f"\n-> The oracle rate ({oracle_rate:.1%}) is what a judge with no verbosity bias")
    print(f"   would measure, using nothing but the true quality gap. The naive judge's")
    print(f"   length bonus for Verbose's much longer responses drags its measured rate")
    print(f"   for the ACTUALLY better model down to {naive_rate:.1%} -- a real, measurable")
    print(f"   distortion that makes the WORSE model win most comparisons. Regressing the")
    print(f"   judge's own scores on length and using the residual moves the rate back up")
    print(f"   to {corrected_rate:.1%} -- no longer favoring the worse model -- but it does NOT fully")
    print(f"   recover the {oracle_rate:.1%} oracle rate, because the pooled fit above underestimates")
    print(f"   the true bias coefficient. This is an honest, known limitation of naive length")
    print(f"   regression: when length and true quality are themselves correlated across the")
    print(f"   compared systems, a simple pooled correction only partially separates them --")
    print(f"   which is exactly why production length-controlled evaluators (e.g. AlpacaEval's")
    print(f"   length-controlled win rate) use more careful statistical controls than this.")


# ---------------------------------------------------------------------------
# DEMO 3: SELF-PREFERENCE BIAS
# A single judge from family "F1" favors responses that share its family,
# even at equal true quality. A panel of judges from different families,
# averaged, suppresses this idiosyncratic per-judge bias.
# ---------------------------------------------------------------------------

def self_preference_bias_demo():
    print("\n" + "=" * 78)
    print("3. SELF-PREFERENCE BIAS: one judge vs. a diverse panel")
    print("=" * 78)

    n_trials = 5000
    self_pref_bonus = 1.8
    noise_std = 1.5
    judge_families = ["F1", "F2", "F3"]

    print("Response from family 'F1' vs. response from family 'F2', both drawn from")
    print("the SAME true-quality distribution -- a fair verdict should be ~50% either way.")
    print(f"Each judge gives a +{self_pref_bonus} bonus to a response sharing its OWN family.\n")

    single_judge_f1_wins = 0   # a lone F1-family judge grading F1 vs F2
    panel_f1_wins = 0          # a 3-judge panel (F1, F2, F3), majority vote

    for _ in range(n_trials):
        quality_f1_response = random.gauss(6.0, 1.0)
        quality_f2_response = random.gauss(6.0, 1.0)
        length_f1 = random.gauss(100, 15)
        length_f2 = random.gauss(100, 15)

        # --- a single judge belonging to family F1 ---
        s_f1 = judge_score(quality_f1_response, length_f1, shown_first=True,
                            judge_family="F1", response_family="F1",
                            self_pref_bonus=self_pref_bonus, noise_std=noise_std)
        s_f2 = judge_score(quality_f2_response, length_f2, shown_first=False,
                            judge_family="F1", response_family="F2",
                            self_pref_bonus=self_pref_bonus, noise_std=noise_std)
        if s_f1 > s_f2:
            single_judge_f1_wins += 1

        # --- a panel of 3 judges (F1, F2, F3), each with its own self-preference,
        #     majority vote decides the winner ---
        votes_for_f1 = 0
        for judge_fam in judge_families:
            js_f1 = judge_score(quality_f1_response, length_f1, shown_first=True,
                                 judge_family=judge_fam, response_family="F1",
                                 self_pref_bonus=self_pref_bonus, noise_std=noise_std)
            js_f2 = judge_score(quality_f2_response, length_f2, shown_first=False,
                                 judge_family=judge_fam, response_family="F2",
                                 self_pref_bonus=self_pref_bonus, noise_std=noise_std)
            if js_f1 > js_f2:
                votes_for_f1 += 1
        if votes_for_f1 >= 2:   # majority of 3
            panel_f1_wins += 1

    single_rate = single_judge_f1_wins / n_trials
    panel_rate = panel_f1_wins / n_trials

    print(f"{'protocol':45}{'F1-response win rate':>24}")
    print("-" * 69)
    print(f"{'single judge (family F1)':45}{single_rate:>24.1%}")
    print(f"{'panel of 3 judges (F1, F2, F3), majority vote':45}{panel_rate:>24.1%}")

    print(f"\n-> True quality is identical between the F1 and F2 responses, so 50% is the")
    print(f"   correct rate. The lone F1 judge, biased toward its own family, measured")
    print(f"   {single_rate:.1%}. Spreading the self-preference bias across a 3-judge panel")
    print(f"   from different families -- each pulling toward a DIFFERENT family -- brought")
    print(f"   the majority-vote rate to {panel_rate:.1%}, closer to the true 50%, because no")
    print(f"   single family's bias can dominate a majority vote across a diverse panel.")


def main():
    position_bias_demo()
    verbosity_bias_demo()
    self_preference_bias_demo()


if __name__ == "__main__":
    main()
