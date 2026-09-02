"""
Reasoning Models and GRPO

PART A -- GRPO's mechanism, isolated and measured directly, then run end to
end on a toy VERIFIABLE-reward task. Each "prompt" is a target digit-sum
(e.g. "find 3 digits, 0-5, that sum to 7"); the "policy" samples a 3-digit
sequence per prompt; the reward is DETERMINISTIC and genuinely verifiable:
1.0 if the digits actually sum to the target, else 0.0 -- a real
correctness check, standing in for "did the extracted final answer match
the ground truth" (README section 1), unlike Lesson 3's deliberately-
exploitable "hack token" reward.

Three things are measured, honestly, in increasing order of how much they
depend on this specific toy task's difficulty:
  1. A property that is ALWAYS true regardless of task: GRPO's
     group-relative advantage gives a real negative ("push away from this")
     signal to failed completions; raw-reward-only ("no baseline") never
     does.
  2. A property that is ALWAYS true, provable directly from the definition
     of z-scoring: GRPO's advantage keeps a roughly CONSTANT signal scale
     regardless of how easy or hard a prompt is, while a raw reward's
     signal strength collapses toward zero for very easy or very hard
     prompts (variance = p(1-p), which vanishes as p -> 0 or 1).
  3. An end-to-end training run showing the full GRPO update (group-relative
     advantage + clipped surrogate + KL penalty against a frozen reference)
     actually solves the toy task, alongside an honest note about why this
     SPECIFIC small, low-dimensional toy search space doesn't need property
     1-2's benefits to be solved eventually by cruder methods too -- those
     properties are what matter as the action space and reasoning length
     grow far beyond what fits in one lesson's runtime.

PART B -- Test-time scaling: using a DELIBERATELY partially-trained (i.e.
still imperfect) policy, sample N completions per prompt, extract each
one's resulting sum, and take a MAJORITY VOTE over those N extracted
answers -- exactly Phase 07 Lesson 2's self-consistency mechanism, applied
here as the deliberate way to spend extra inference compute on a fixed,
already-trained model. We measure real accuracy as N grows and see the
diminishing-returns curve directly.

Runtime: ~15-30 seconds on a CPU (no real neural network -- like Lesson 3,
this optimizes a small logits tensor directly, to keep every RL mechanic
fully transparent).

Run:
    python example.py
"""

import random
from collections import Counter

import torch
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# The toy verifiable-reward task: NUM_PROMPTS distinct "problems," each
# asking for SEQ_LEN digits (0..VOCAB_SIZE-1) that sum to a specific target.
# ---------------------------------------------------------------------------

VOCAB_SIZE = 6                      # digits 0-5
SEQ_LEN = 3                         # 3 digits per response
TARGET_SUMS = [4, 6, 9, 11]         # one target per prompt (max possible sum = 15)
NUM_PROMPTS = len(TARGET_SUMS)

GROUP_SIZE = 16          # G completions sampled per prompt, per rollout (GRPO's "group")
CLIP_EPS = 0.2           # identical PPO trust-region clip range to Lesson 3
KL_BETA = 0.03           # KL penalty weight against the frozen reference policy
PPO_EPOCHS = 4           # update epochs reused per rollout batch, same as Lesson 3
LR = 0.04
NUM_ITERS = 120
PARTIAL_TRAIN_ITERS = 20  # deliberately stopped early -- see Part B


def make_reference_logits():
    """The frozen reference policy pi_ref (README section 2/4) -- mildly
    non-uniform per prompt, standing in for an SFT starting point before any
    reasoning-RL fine-tuning."""
    torch.manual_seed(7)
    return torch.randn(NUM_PROMPTS, SEQ_LEN, VOCAB_SIZE) * 0.3


REFERENCE_LOGITS = make_reference_logits()


def verifiable_reward(actions):
    """The verifiable reward (README section 1): 1.0 if the sampled digits
    for prompt p ACTUALLY sum to TARGET_SUMS[p], else 0.0. A deterministic
    checker, not a learned model -- genuinely un-gameable in the way Lesson
    3's reward-model stand-in was, because there is no proxy to exploit."""
    targets = torch.tensor(TARGET_SUMS).view(1, NUM_PROMPTS)
    sums = actions.sum(dim=-1)                      # (batch, NUM_PROMPTS)
    return (sums == targets).float()


def rollout(policy_logits, group_size=GROUP_SIZE):
    """Sample `group_size` completions for EVERY prompt at once. Returns
    actions (group, NUM_PROMPTS, SEQ_LEN), old/ref log-probs (group,
    NUM_PROMPTS, SEQ_LEN), and the raw verifiable reward (group, NUM_PROMPTS)."""
    with torch.no_grad():
        probs = F.softmax(policy_logits, dim=-1)                     # (NUM_PROMPTS, SEQ_LEN, V)
        dist = torch.distributions.Categorical(
            probs=probs.unsqueeze(0).expand(group_size, -1, -1, -1)
        )
        actions = dist.sample()                                      # (group, NUM_PROMPTS, SEQ_LEN)
        old_log_probs = dist.log_prob(actions)                       # (group, NUM_PROMPTS, SEQ_LEN)

        ref_probs = F.softmax(REFERENCE_LOGITS, dim=-1)
        ref_dist = torch.distributions.Categorical(
            probs=ref_probs.unsqueeze(0).expand(group_size, -1, -1, -1)
        )
        ref_log_probs = ref_dist.log_prob(actions)

        raw_reward = verifiable_reward(actions)                      # (group, NUM_PROMPTS)
    return actions, old_log_probs, ref_log_probs, raw_reward


def group_relative_advantage(raw_reward):
    """GRPO's entire trick (README section 2-3): normalize each prompt's G
    rewards by THAT PROMPT's own group mean/std -- no critic network, no
    global baseline. Broadcast to every token position of that completion."""
    mean = raw_reward.mean(dim=0, keepdim=True)               # (1, NUM_PROMPTS)
    std = raw_reward.std(dim=0, keepdim=True)
    advantage = (raw_reward - mean) / (std + 1e-4)             # (group, NUM_PROMPTS)
    return advantage.unsqueeze(-1).expand(-1, -1, SEQ_LEN)     # broadcast over SEQ_LEN


def no_baseline_advantage(raw_reward):
    """The ablation: the SAME raw reward, with NO baseline subtracted at
    all -- every failed completion (reward=0) contributes exactly zero
    advantage, so it can never be actively pushed down, only successes get
    pushed up (README section 4's ablation)."""
    return raw_reward.unsqueeze(-1).expand(-1, -1, SEQ_LEN)


def clipped_update(policy_logits, actions, old_log_probs, ref_log_probs, advantage, optimizer,
                    group_size=GROUP_SIZE):
    """Identical clipped-surrogate-plus-KL-penalty update to Lesson 3's
    ppo_update, generalized over the extra NUM_PROMPTS dimension. Only the
    `advantage` tensor passed in differs between GRPO and the ablation."""
    for _ in range(PPO_EPOCHS):
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(
            probs=probs.unsqueeze(0).expand(group_size, -1, -1, -1)
        )
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        kl_term = (new_log_probs - ref_log_probs.detach()).mean()   # sampled KL(pi||pi_ref) estimate
        loss = policy_loss + KL_BETA * kl_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(use_group_baseline, label, num_iters=NUM_ITERS, seed=0, verbose=True):
    torch.manual_seed(seed)
    policy_logits = REFERENCE_LOGITS.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([policy_logits], lr=LR)

    history = []
    for it in range(1, num_iters + 1):
        actions, old_log_probs, ref_log_probs, raw_reward = rollout(policy_logits.detach())

        advantage = group_relative_advantage(raw_reward) if use_group_baseline \
            else no_baseline_advantage(raw_reward)

        clipped_update(policy_logits, actions, old_log_probs, ref_log_probs, advantage, optimizer)

        success_rate = raw_reward.mean().item()
        history.append(success_rate)
        if verbose and (it % 20 == 0 or it == 1):
            print(f"  [{label}] iter {it:4d}   success rate (avg over {GROUP_SIZE * NUM_PROMPTS} "
                  f"rollouts) = {success_rate:.3f}")

    return policy_logits.detach(), history


# ---------------------------------------------------------------------------
# DEMO 1: the negative-advantage mechanism, made concrete
# ---------------------------------------------------------------------------

def measure_negative_advantage_fraction():
    actions, _, _, raw_reward = rollout(REFERENCE_LOGITS.clone())
    grpo_adv = group_relative_advantage(raw_reward)[:, :, 0]
    noB_adv = no_baseline_advantage(raw_reward)[:, :, 0]
    grpo_neg_frac = (grpo_adv < 0).float().mean().item()
    noB_neg_frac = (noB_adv < 0).float().mean().item()
    return grpo_neg_frac, noB_neg_frac


# ---------------------------------------------------------------------------
# DEMO 2: signal-scale consistency across easy/medium/hard prompts. Pure
# statistics -- simulated Bernoulli(p) groups, independent of any actual
# training run, so this always reproduces identically regardless of how the
# policy above happens to train.
# ---------------------------------------------------------------------------

def measure_signal_consistency():
    torch.manual_seed(0)
    LARGE_GROUP = 64
    NUM_SIMULATED_GROUPS = 4000
    ps = [0.03, 0.125, 0.5, 0.875, 0.97]   # very hard -> medium -> very easy
    results = []
    for p in ps:
        rewards = torch.bernoulli(torch.full((NUM_SIMULATED_GROUPS, LARGE_GROUP), p))
        raw_std = rewards.std().item()
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        grpo_adv = (rewards - mean) / (std + 1e-4)
        grpo_std = grpo_adv.std().item()
        results.append((p, raw_std, grpo_std))
    return results


def part_a_demo():
    print("=" * 78)
    print("PART A: GRPO'S MECHANISM, MEASURED DIRECTLY, THEN RUN END TO END")
    print("=" * 78)
    print(f"{NUM_PROMPTS} prompts, each: find {SEQ_LEN} digits in 0..{VOCAB_SIZE - 1} summing to a target.")
    print(f"Targets: {TARGET_SUMS}   Group size (completions sampled per prompt): {GROUP_SIZE}")
    print("Reward is a DETERMINISTIC correctness check -- 1.0 if digits sum to the")
    print("target, 0.0 otherwise. No learned reward model, no human labels at RL time.\n")

    print("-" * 78)
    print("1. NEGATIVE-ADVANTAGE MECHANISM (one rollout from the near-random start)")
    print("-" * 78)
    grpo_neg_frac, noB_neg_frac = measure_negative_advantage_fraction()
    print(f"Fraction of sampled completions with a STRICTLY NEGATIVE advantage:")
    print(f"  GRPO (group-relative baseline):  {grpo_neg_frac:.1%}")
    print(f"  No baseline (raw reward only):   {noB_neg_frac:.1%}")
    print("-> With no baseline, a failed completion (reward=0) gets advantage exactly")
    print("   0 -- it is never actively pushed down, only successes get pushed up.")
    print("   GRPO's group-relative advantage is NEGATIVE for below-group-average")
    print("   completions (which includes every failure whenever ANY group member")
    print("   succeeded), giving the policy a real 'move away from this' signal that")
    print("   plain REINFORCE-without-a-baseline structurally cannot provide.")

    print("\n" + "-" * 78)
    print("2. SIGNAL-SCALE CONSISTENCY ACROSS EASY/MEDIUM/HARD PROMPTS")
    print("-" * 78)
    print("(Pure statistics on simulated Bernoulli(p) groups -- independent of any")
    print("actual training run below, so this always reproduces identically.)\n")
    results = measure_signal_consistency()
    print(f"{'per-sample success prob p':28}{'raw-reward advantage std':>28}{'GRPO advantage std':>22}")
    for p, raw_std, grpo_std in results:
        print(f"{p:<28}{raw_std:>28.3f}{grpo_std:>22.3f}")
    raw_ratio = max(r[1] for r in results) / min(r[1] for r in results)
    grpo_ratio = max(r[2] for r in results) / min(r[2] for r in results)
    print(f"\n-> Raw-reward advantage's scale varies {raw_ratio:.1f}x between the hardest/easiest")
    print(f"   prompts and the p=0.5 prompt (it literally IS sqrt(p*(1-p)), which vanishes")
    print(f"   toward both extremes) -- meaning a very easy or very hard prompt gives the")
    print(f"   policy almost no gradient signal at all, even though there is still plenty")
    print(f"   left to learn (e.g. going from 95% to 99.9% success). GRPO's group-relative")
    print(f"   z-scoring keeps the advantage's scale within {grpo_ratio:.1f}x across the SAME")
    print(f"   sweep -- a consistently strong learning signal regardless of how easy or")
    print(f"   hard any individual prompt happens to be.")

    print("\n" + "-" * 78)
    print("3. END-TO-END: DOES THE FULL GRPO UPDATE ACTUALLY SOLVE THE TASK?")
    print("-" * 78)
    trained_policy_grpo, history_grpo = train(use_group_baseline=True, label="GRPO")

    print("\n" + "(for comparison -- same reward, same clipping, same KL penalty, only")
    print(" the advantage computation differs: no group baseline at all)")
    _, history_noB = train(use_group_baseline=False, label="no-baseline")

    final_grpo = sum(history_grpo[-10:]) / 10
    final_noB = sum(history_noB[-10:]) / 10
    print(f"\nFinal success rate (last 10 rollouts): GRPO={final_grpo:.1%}   no-baseline={final_noB:.1%}")
    print("-> Both methods solve this SPECIFIC toy task -- it is a tiny, low-dimensional")
    print("   search space (216 possible digit sequences per prompt), and Adam's own")
    print("   adaptive step sizing partly compensates for raw reward's inconsistent scale")
    print("   here. That is an honest property of this toy example, not a claim that")
    print("   baselines don't matter: demos 1-2 above measure the ACTUAL mechanism GRPO")
    print("   provides directly, and that mechanism is what separates 'learns from sparse,")
    print("   verifiable rewards at all' from 'doesn't' once the search space is a real")
    print("   space of token sequences many steps long, not 216 three-digit combinations.")

    return trained_policy_grpo


# ===========================================================================
# PART B: TEST-TIME SCALING -- MAJORITY VOTE OVER N SAMPLES, VS. N
# ===========================================================================

def sample_completions(policy_logits, prompt_idx, n):
    probs = F.softmax(policy_logits[prompt_idx], dim=-1)          # (SEQ_LEN, V)
    dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(n, -1, -1))
    actions = dist.sample()                                        # (n, SEQ_LEN)
    return actions.sum(dim=-1).tolist()                            # extracted "answers" (sums)


def majority_vote_accuracy(policy_logits, n, num_trials=3000):
    correct = 0
    for _ in range(num_trials):
        prompt_idx = random.randrange(NUM_PROMPTS)
        extracted_sums = sample_completions(policy_logits, prompt_idx, n)
        majority_answer = Counter(extracted_sums).most_common(1)[0][0]
        if majority_answer == TARGET_SUMS[prompt_idx]:
            correct += 1
    return correct / num_trials


def part_b_demo():
    print("\n" + "=" * 78)
    print("PART B: TEST-TIME SCALING -- MAJORITY VOTE OVER N SAMPLES")
    print("=" * 78)
    print(f"Training a FRESH policy but stopping deliberately early, after only")
    print(f"{PARTIAL_TRAIN_ITERS} GRPO iterations (Part A trains to {NUM_ITERS}) -- a genuinely")
    print("IMPERFECT, still-learning policy, exactly the regime where spending extra")
    print("INFERENCE compute is worth it (a policy already at ~100% one-shot accuracy")
    print("has nothing left for majority voting to improve).\n")
    partial_policy, partial_history = train(
        use_group_baseline=True, label="partial-GRPO", num_iters=PARTIAL_TRAIN_ITERS, verbose=False
    )
    one_shot_acc = sum(partial_history[-5:]) / 5
    print(f"One-shot (N=1) success rate after {PARTIAL_TRAIN_ITERS} iterations: {one_shot_acc:.1%}\n")

    print("For each trial: sample N completions for a random prompt, extract each one's")
    print("digit-sum as its 'answer,' and take the MAJORITY-VOTE answer across all N --")
    print("exactly Phase 07 Lesson 2 section 3's self-consistency mechanism, now spent as")
    print("the deliberate way a reasoning model uses extra INFERENCE compute on a fixed,")
    print("already-trained policy (as opposed to Phase 03 Lesson 5's scaling laws, which")
    print("spend more compute at TRAINING time instead).\n")

    ns = [1, 3, 5, 9, 15, 25]
    print(f"{'N (samples per vote)':25}{'majority-vote accuracy':>26}")
    accuracies = []
    for n in ns:
        acc = majority_vote_accuracy(partial_policy, n)
        accuracies.append(acc)
        print(f"{n:<25}{acc:>26.1%}")

    print(f"\n-> Accuracy rises from {accuracies[0]:.1%} at N=1 (a single sample) toward "
          f"{accuracies[-1]:.1%}")
    print(f"   at N={ns[-1]}, with shrinking marginal gains per extra sample -- the same")
    print(f"   diminishing-returns shape Phase 07 Lesson 2 section 4's Condorcet Jury")
    print(f"   Theorem predicts for any per-sample accuracy above chance. Because each")
    print(f"   WRONG digit-sequence can land on several different incorrect sums (not")
    print(f"   one single 'the' wrong answer), this is really the multi-candidate")
    print(f"   plurality-voting regime from Lesson 2 section 5, not the plain binary")
    print(f"   case -- wrong votes split against each other across several wrong sums,")
    print(f"   which is part of why the correct answer wins the vote more easily than")
    print(f"   the raw per-sample accuracy alone would suggest.")


def main():
    part_a_demo()
    part_b_demo()


if __name__ == "__main__":
    main()
