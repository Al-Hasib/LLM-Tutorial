"""
RLHF with PPO

Implements a simplified PPO-style policy-gradient training loop from
scratch, applied to a toy "sequential token generation" task standing in
for the RL stage of the full RLHF pipeline (README section 1). To keep the
mechanics fully transparent, the "policy" independently chooses one token
out of a small vocabulary at each of T sequence positions (a drastic
simplification of a real autoregressive language model -- there is no
conditioning on previously generated tokens -- but it preserves exactly
the RL machinery this lesson is about: per-token reward, a per-token KL
penalty against a frozen reference policy, advantage estimation, and PPO's
clipped surrogate objective).

The toy reward function scores +1 for every occurrence of one specific
"hack token" in the generated sequence, standing in for a reward model
with a blind spot (README section 3): maximizing this reward with no
constraint just means spamming that one token, which is exactly the kind
of degenerate, non-language-like behavior a real reward model's quirks
can be exploited into producing.

We run the SAME training loop twice: once with the KL penalty against the
frozen reference policy (beta > 0), and once with it switched off
(beta = 0), to honestly reproduce reward hacking side by side with the
mechanism that prevents it.

Runtime: a few seconds on a CPU (no neural network training in the usual
sense -- this optimizes a small logits matrix directly).

Run:
    python example.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB_SIZE = 6
SEQ_LEN = 8
HACK_TOKEN = VOCAB_SIZE - 1     # the token the (toy, hackable) reward function rewards
BATCH_SIZE = 256                # episodes ("generated sequences") per rollout
NUM_ITERS = 60                  # outer rollout -> update cycles
PPO_EPOCHS = 4                  # PPO update epochs per rollout batch
CLIP_EPS = 0.2                  # PPO's trust-region clip range


def make_reference_logits():
    """The frozen reference policy pi_ref -- stands in for the SFT model
    (README section 4). Initialized to a mildly non-uniform distribution
    over the vocabulary at each position, representing "natural,
    already-reasonable" behavior before any RL fine-tuning."""
    torch.manual_seed(42)
    return torch.randn(SEQ_LEN, VOCAB_SIZE) * 0.5


REFERENCE_LOGITS = make_reference_logits()


def analytic_kl(policy_logits, reference_logits):
    """Exact KL( pi_theta(.|t) || pi_ref(.|t) ), summed over all T positions.
    Used here only for MONITORING how far the policy has drifted -- the
    actual per-token penalty applied during training uses the standard
    sampled log-ratio estimator computed in rollout(), matching what real
    RLHF implementations use (README section 4)."""
    log_p = F.log_softmax(policy_logits, dim=-1)
    log_ref = F.log_softmax(reference_logits, dim=-1)
    p = log_p.exp()
    kl_per_position = (p * (log_p - log_ref)).sum(dim=-1)
    return kl_per_position.sum().item()


def rollout(policy_logits, beta):
    """Sample a batch of toy 'sequences' from the CURRENT policy (this
    becomes the PPO 'old' policy for the following update epochs), and
    compute the per-token reward-minus-KL-penalty and a simple batch-mean
    baseline advantage.

    Returns: sampled actions, old log-probs (detached), advantages (detached).
    """
    with torch.no_grad():
        probs = F.softmax(policy_logits, dim=-1)                     # (T, V)
        dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(BATCH_SIZE, -1, -1))
        actions = dist.sample()                                       # (batch, T)

        old_log_probs = dist.log_prob(actions)                        # (batch, T)

        ref_probs = F.softmax(REFERENCE_LOGITS, dim=-1)
        ref_dist = torch.distributions.Categorical(probs=ref_probs.unsqueeze(0).expand(BATCH_SIZE, -1, -1))
        ref_log_probs = ref_dist.log_prob(actions)                    # (batch, T)

        raw_reward = (actions == HACK_TOKEN).float()                  # (batch, T) 1 per hack-token hit
        kl_estimate = old_log_probs - ref_log_probs                   # per-token sampled KL estimate
        total_reward = raw_reward - beta * kl_estimate                # README section 4's formula, per token

        baseline = total_reward.mean(dim=0, keepdim=True)             # simple variance-reduction baseline
        advantages = total_reward - baseline

    return actions, old_log_probs, advantages, raw_reward.sum(dim=1).mean().item()


def ppo_update(policy_logits, actions, old_log_probs, advantages, optimizer):
    """Several epochs of the PPO clipped surrogate objective (README section 5)
    on the SAME rollout batch, exactly PPO's key efficiency trick: squeeze
    several gradient updates out of one (expensive, in a real LLM) rollout."""
    for _ in range(PPO_EPOCHS):
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(BATCH_SIZE, -1, -1))
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        loss = -torch.min(surrogate1, surrogate2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(beta, label):
    policy_logits = REFERENCE_LOGITS.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([policy_logits], lr=0.1)

    history = []
    for it in range(1, NUM_ITERS + 1):
        actions, old_log_probs, advantages, avg_raw_reward = rollout(policy_logits.detach(), beta)
        ppo_update(policy_logits, actions, old_log_probs, advantages, optimizer)

        kl = analytic_kl(policy_logits.detach(), REFERENCE_LOGITS)
        history.append((it, avg_raw_reward, kl))

        if it % 10 == 0 or it == 1:
            print(f"  [{label}] iter {it:3d}   avg reward/episode = {avg_raw_reward:.3f} "
                  f"(max possible = {SEQ_LEN})   KL(policy || reference) = {kl:.3f}")

    return policy_logits.detach(), history


def describe_final_policy(policy_logits, label):
    probs = F.softmax(policy_logits, dim=-1)
    hack_prob = probs[:, HACK_TOKEN].mean().item()
    print(f"\n[{label}] average probability mass the policy places on the hack token "
          f"across all {SEQ_LEN} positions: {hack_prob:.3f}")
    print(f"[{label}] full probability distribution at position 0: "
          f"{[round(p, 3) for p in probs[0].tolist()]}")


def main():
    print("=" * 70)
    print("SETUP")
    print("=" * 70)
    print(f"Vocabulary size: {VOCAB_SIZE}, sequence length: {SEQ_LEN}, "
          f"'hack token' index: {HACK_TOKEN}")
    print("Toy reward = +1 for every occurrence of the hack token in the sequence --")
    print("a stand-in for a reward model with an exploitable blind spot (README section 3).")
    ref_probs_pos0 = F.softmax(REFERENCE_LOGITS[0], dim=-1)
    print(f"Reference (frozen SFT-like) policy's distribution at position 0: "
          f"{[round(p, 3) for p in ref_probs_pos0.tolist()]}")

    print("\n" + "=" * 70)
    print("1. TRAINING WITH THE KL PENALTY (beta = 0.5) -- the correct recipe")
    print("=" * 70)
    policy_with_kl, history_with_kl = train(beta=0.5, label="beta=0.5")
    describe_final_policy(policy_with_kl, "beta=0.5")

    print("\n" + "=" * 70)
    print("2. TRAINING WITH NO KL PENALTY (beta = 0.0) -- reward hacking, on purpose")
    print("=" * 70)
    policy_no_kl, history_no_kl = train(beta=0.0, label="beta=0.0")
    describe_final_policy(policy_no_kl, "beta=0.0")

    print("\n" + "=" * 70)
    print("3. COMPARISON")
    print("=" * 70)
    final_reward_kl = history_with_kl[-1][1]
    final_kl_kl = history_with_kl[-1][2]
    final_reward_nokl = history_no_kl[-1][1]
    final_kl_nokl = history_no_kl[-1][2]

    print(f"{'setting':>14}{'final avg reward':>20}{'final KL':>14}")
    print(f"{'beta=0.5':>14}{final_reward_kl:>20.3f}{final_kl_kl:>14.3f}")
    print(f"{'beta=0.0':>14}{final_reward_nokl:>20.3f}{final_kl_nokl:>14.3f}")

    print(f"\n-> With the KL penalty active, average reward rose from a near-random")
    print(f"   starting point to {final_reward_kl:.2f} out of a max of {SEQ_LEN}, while KL")
    print(f"   divergence from the reference policy stayed bounded at {final_kl_kl:.3f} --")
    print(f"   reward improved WITHOUT the policy drifting arbitrarily far from")
    print(f"   sensible (reference-like) behavior.")
    print(f"\n-> With NO KL penalty, reward reached {final_reward_nokl:.2f} -- HIGHER than the")
    print(f"   KL-constrained run -- but KL divergence exploded to {final_kl_nokl:.3f}, "
          f"{final_kl_nokl / max(final_kl_kl, 1e-6):.0f}x")
    print(f"   larger. The policy achieves this extra reward by collapsing almost all")
    print(f"   probability mass onto the single hack token at every position (see the")
    print(f"   'average probability mass on the hack token' lines above) -- exactly")
    print(f"   the degenerate, non-language-like reward hacking behavior the KL")
    print(f"   penalty exists to prevent. Higher reward here is NOT a better policy;")
    print(f"   it is Goodhart's law made concrete: a proxy (the reward model / toy")
    print(f"   reward function) stops tracking what it was meant to measure once")
    print(f"   optimization pressure against it is unconstrained.")


if __name__ == "__main__":
    main()
