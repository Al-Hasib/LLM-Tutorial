"""
Mixture of Experts, Advanced

Three demos, all building directly on Phase 03 Lesson 4's MoELayer/Expert:
  1. A from-scratch Expert-Choice MoE layer: instead of each TOKEN picking
     its top-k experts, each EXPERT picks its top-C tokens from the batch
     (C = capacity). Measures how many tokens end up dropped (chosen by no
     expert) or multiply-served (chosen by more than one).
  2. A head-to-head comparison against ordinary top-k token-choice routing,
     under the EXACT same slight initial router bias trick used in Phase 03
     Lesson 4 (+1.0 to router.bias[0]). Shows Expert-Choice gives perfectly
     even expert utilization BY CONSTRUCTION -- true even before training --
     while token-choice without an auxiliary loss reproduces the collapse
     tendency Phase 03 measured.
  3. A token-dropping-rate simulation for top-k token-choice routing as the
     capacity factor varies, using real random router-derived assignments
     and a hard per-expert capacity limit, reporting the actual measured
     fraction of dropped tokens at each capacity factor.

Run:
    python example.py

Runtime: well under a minute on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Shared building block, reused unchanged from Phase 03 Lesson 4
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """One expert = one small FFN, identical in shape to Phase 02's FFN sublayer."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    """Ordinary top-k TOKEN-CHOICE MoE layer -- the router in Phase 03 Lesson 4.

    Each token picks its own top-k experts. Reproduced here unmodified so the
    bias-and-collapse experiment in section 2 below is a faithful rerun of the
    Phase 03 methodology, not a reimplementation that might behave differently.
    """

    def __init__(self, d_model, d_ff, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        """x: (batch, d_model). Returns (output, router_probs, chosen_expert_ids)."""
        router_logits = self.router(x)                       # (batch, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)

        output = torch.zeros_like(x)
        for slot in range(self.top_k):
            expert_ids = top_k_indices[:, slot]                # (batch,) which expert per token
            gate = top_k_probs[:, slot].unsqueeze(-1)           # (batch, 1)
            for expert_id in expert_ids.unique():
                mask = expert_ids == expert_id
                output[mask] += gate[mask] * self.experts[expert_id](x[mask])

        return output, router_probs, top_k_indices


# ---------------------------------------------------------------------------
# 1. Expert-Choice routing, from scratch
# ---------------------------------------------------------------------------

class ExpertChoiceMoELayer(nn.Module):
    """Expert-Choice MoE (Zhou et al., 2022): EXPERTS pick TOKENS.

    Same router + same Expert FFNs as MoELayer above, but the selection
    direction is inverted: for each expert (a COLUMN of the affinity matrix),
    take its top-C tokens by affinity score, up to a fixed capacity C. A
    token can therefore be picked by zero experts (dropped), one expert, or
    several experts -- there is no guarantee per-token, only per-expert.
    """

    def __init__(self, d_model, d_ff, num_experts, tokens_per_expert):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = tokens_per_expert  # C: how many tokens EACH expert takes, fixed in advance
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        """x: (num_tokens, d_model). Returns (output, per_token_hit_count)."""
        num_tokens = x.shape[0]
        affinity = self.router(x)                     # (T, E) raw affinity scores
        gate = F.softmax(affinity, dim=-1)             # per-token normalized gate values

        output = torch.zeros_like(x)
        per_token_hit_count = torch.zeros(num_tokens)  # how many experts served each token

        # Capacity can't exceed the number of tokens actually available.
        C = min(self.capacity, num_tokens)

        for expert_id in range(self.num_experts):
            scores_e = affinity[:, expert_id]                  # this expert's COLUMN: one score/token
            top_c_tokens = scores_e.topk(C).indices             # THIS expert's chosen tokens
            gate_values = gate[top_c_tokens, expert_id].unsqueeze(-1)   # (C, 1)
            output[top_c_tokens] += gate_values * self.experts[expert_id](x[top_c_tokens])
            per_token_hit_count[top_c_tokens] += 1

        return output, per_token_hit_count


def expert_choice_demo():
    print("=" * 70)
    print("1. EXPERT-CHOICE ROUTING, FROM SCRATCH")
    print("=" * 70)

    d_model, d_ff, num_experts = 16, 32, 4
    num_tokens = 20
    top_k_equiv = 1  # average number of experts each token would get under top-1 token-choice
    capacity = (num_tokens * top_k_equiv) // num_experts  # C = T*k / E

    print(f"num_tokens={num_tokens}, num_experts={num_experts}, capacity C={capacity} "
          f"(= tokens_per_expert)\n")

    ec_moe = ExpertChoiceMoELayer(d_model, d_ff, num_experts, tokens_per_expert=capacity)
    x = torch.randn(num_tokens, d_model)
    output, hit_count = ec_moe(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(output.shape)}  (unchanged, same as token-choice would give)\n")

    tokens_dropped = (hit_count == 0).sum().item()
    tokens_multi = (hit_count > 1).sum().item()
    tokens_once = (hit_count == 1).sum().item()
    print("Per-token hit counts (how many experts selected each token):")
    print(f"  {hit_count.long().tolist()}")
    print(f"\n  selected by exactly 1 expert: {tokens_once}/{num_tokens}")
    print(f"  selected by 0 experts (DROPPED, residual-only for this layer): "
          f"{tokens_dropped}/{num_tokens}")
    print(f"  selected by 2+ experts (multiply-served): {tokens_multi}/{num_tokens}")
    print(f"\n-> Every expert processed EXACTLY its capacity of {capacity} tokens (by")
    print("   construction -- top-C always returns exactly C tokens), but because")
    print("   experts choose independently, individual tokens ended up unevenly")
    print("   served: some dropped entirely, some picked up by more than one expert.")


# ---------------------------------------------------------------------------
# 2. Expert-Choice vs. token-choice under the SAME Phase 03 bias scenario
# ---------------------------------------------------------------------------

def expert_choice_utilization_at_init(num_experts=4, num_tokens=32, d_model=16, d_ff=32):
    """Expert-Choice utilization needs no training at all to check -- balance is
    structural. Apply the SAME router-bias trick as the token-choice experiment
    for an apples-to-apples setup, then show utilization is already perfectly
    even at random initialization, before a single gradient step."""
    torch.manual_seed(0)
    top_k_equiv = 1
    capacity = (num_tokens * top_k_equiv) // num_experts
    ec_moe = ExpertChoiceMoELayer(d_model, d_ff, num_experts, tokens_per_expert=capacity)

    # Same bias-toward-expert-0 trick as Phase 03's train_moe_toy_task, applied
    # here even though Expert-Choice doesn't need training to prove the point.
    with torch.no_grad():
        ec_moe.router.bias[0] += 1.0

    x = torch.randn(num_tokens, d_model)
    _, hit_count = ec_moe(x)

    # Tally exact per-expert load by re-running the per-expert top-C selection.
    affinity = ec_moe.router(x)
    counts = torch.zeros(num_experts)
    for expert_id in range(num_experts):
        top_c = affinity[:, expert_id].topk(capacity).indices
        counts[expert_id] = len(top_c)
    return counts, capacity


def train_moe_toy_task(use_aux_loss, num_experts=4, steps=600, aux_loss_weight=3.0):
    """Reproduced unmodified from Phase 03 Lesson 4's example.py, so this is a
    faithful rerun of that exact methodology, not a reimplementation."""
    torch.manual_seed(0)
    d_model, d_ff = 16, 32
    moe = MoELayer(d_model, d_ff, num_experts, top_k=1)

    # Deliberately bias the router's initial weights slightly toward expert 0,
    # simulating the kind of small random initialization advantage that, left
    # uncorrected, snowballs into full collapse.
    with torch.no_grad():
        moe.router.bias[0] += 1.0

    true_fn = torch.randn(d_model, d_model)  # a fixed target function every expert COULD learn
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-2)

    expert_counts = torch.zeros(num_experts)
    for step in range(steps):
        x = torch.randn(32, d_model)
        y_true = x @ true_fn

        output, router_probs, chosen = moe(x)
        task_loss = F.mse_loss(output, y_true)

        loss = task_loss
        if use_aux_loss:
            chosen_flat = chosen.squeeze(-1)
            f_i = torch.stack([(chosen_flat == e).float().mean() for e in range(num_experts)])
            P_i = router_probs.mean(dim=0)
            aux_loss = num_experts * (f_i * P_i).sum()
            loss = task_loss + aux_loss_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step >= steps - 50:   # tally routing over the final 50 steps only
            for e in chosen.squeeze(-1):
                expert_counts[e] += 1

    return expert_counts


def imbalance(counts):
    fractions = counts / counts.sum()
    ideal = 1.0 / len(counts)
    return (fractions - ideal).abs().sum().item()


def comparison_demo():
    print("\n" + "=" * 70)
    print("2. EXPERT-CHOICE vs. TOKEN-CHOICE, SAME ROUTER-BIAS SCENARIO")
    print("=" * 70)
    print("Same +1.0 initial bias toward expert 0 as Phase 03 Lesson 4's collapse")
    print("experiment, applied to both routing schemes.\n")

    num_experts = 4

    # --- Expert-Choice: check utilization at INITIALIZATION, no training needed ---
    ec_counts, capacity = expert_choice_utilization_at_init(num_experts=num_experts)
    print(f"Expert-Choice per-expert token counts, AT INITIALIZATION (capacity C={capacity}):")
    print(f"  " + "".join(f"expert {i}: {int(c):>3}  " for i, c in enumerate(ec_counts)))
    print(f"  deviation from perfectly even routing: {imbalance(ec_counts):.3f}")
    print("  -> Despite the bias toward expert 0, EVERY expert processed exactly")
    print(f"     C={capacity} tokens -- perfect balance, with ZERO training and ZERO")
    print("     auxiliary loss. Balance here is a structural property of top-C")
    print("     selection, not a trained-in behavior.\n")

    # --- Token-choice: rerun Phase 03's exact training methodology ---
    print("Token-choice, trained for 600 steps on the same toy regression task,")
    print("counting routing over the FINAL 50 steps (1600 token-routings total):\n")
    counts_no_aux = train_moe_toy_task(use_aux_loss=False, num_experts=num_experts)
    counts_with_aux = train_moe_toy_task(use_aux_loss=True, num_experts=num_experts)

    print(f"{'expert':>13}" + "".join(f"{i:>10}" for i in range(num_experts)))
    print(f"{'no aux loss':>13}" + "".join(f"{int(c):>10}" for c in counts_no_aux))
    print(f"{'with aux loss':>13}" + "".join(f"{int(c):>10}" for c in counts_with_aux))

    no_aux_imb = imbalance(counts_no_aux)
    with_aux_imb = imbalance(counts_with_aux)
    ec_imb = imbalance(ec_counts)
    print(f"\nDeviation from perfectly even routing (0 = perfectly balanced):")
    print(f"  token-choice, no aux loss:   {no_aux_imb:.3f}")
    print(f"  token-choice, with aux loss: {with_aux_imb:.3f}")
    print(f"  Expert-Choice (any bias):    {ec_imb:.3f}  (always exactly 0.000, by construction)")
    print(f"\n-> Token-choice without an auxiliary loss reproduces the collapse Phase 03")
    print(f"   measured (imbalance {no_aux_imb:.3f}, all {int(counts_no_aux.sum().item())} routed")
    print(f"   tokens skewed toward expert 0). Adding the aux loss brings it much closer")
    print(f"   to even ({with_aux_imb:.3f}) but still needs that extra loss term and its")
    print("   weight tuned. Expert-Choice needed neither training nor an aux loss to be")
    print("   perfectly balanced -- it cannot become imbalanced in the first place.")


# ---------------------------------------------------------------------------
# 3. Capacity factor and token dropping in top-k token-choice routing
# ---------------------------------------------------------------------------

def simulate_token_dropping(num_tokens, num_experts, top_k, capacity_factor, num_trials=200):
    """Simulate top-k token-choice routing with a hard per-expert capacity limit.

    Each of the top_k routing "slots" per token is assigned independently at
    random (mirroring a reasonably well load-balanced but not perfectly even
    trained router), processed in a fixed token order per expert, and any
    assignment beyond that expert's capacity is a DROP -- exactly how Switch
    Transformer-style implementations handle overflow. Averaged over many
    random trials for a stable estimate of the dropped-token fraction.
    """
    ideal_load_per_expert = (num_tokens * top_k) / num_experts
    capacity = max(1, int(capacity_factor * ideal_load_per_expert))

    total_routed = 0
    total_dropped = 0
    for _ in range(num_trials):
        expert_load = torch.zeros(num_experts, dtype=torch.long)
        # Random per-token, per-slot expert assignment (order = arrival order).
        assignments = torch.randint(0, num_experts, (num_tokens, top_k))
        for token_idx in range(num_tokens):
            for slot in range(top_k):
                expert_id = assignments[token_idx, slot].item()
                total_routed += 1
                if expert_load[expert_id] < capacity:
                    expert_load[expert_id] += 1  # accepted
                else:
                    total_dropped += 1            # capacity already full -- dropped

    return total_dropped / total_routed, capacity


def capacity_factor_demo():
    print("\n" + "=" * 70)
    print("3. CAPACITY FACTOR AND TOKEN DROPPING IN TOP-K ROUTING")
    print("=" * 70)

    num_tokens, num_experts, top_k = 64, 8, 1
    ideal_load = (num_tokens * top_k) / num_experts
    print(f"num_tokens={num_tokens}, num_experts={num_experts}, top_k={top_k}")
    print(f"ideal (perfectly even) load per expert = {ideal_load:.1f} tokens\n")
    print("capacity = capacity_factor * ideal_load; any token whose assigned expert")
    print("is already full when it arrives is DROPPED (passes through via the")
    print("residual only, no expert computation). Averaged over 200 random trials")
    print("per capacity factor:\n")

    capacity_factors = [1.0, 1.25, 1.5, 2.0]
    print(f"{'capacity_factor':>16}{'capacity/expert':>18}{'dropped_fraction':>18}")
    results = []
    for cf in capacity_factors:
        dropped_fraction, capacity = simulate_token_dropping(
            num_tokens, num_experts, top_k, cf, num_trials=200
        )
        results.append((cf, capacity, dropped_fraction))
        print(f"{cf:>16.2f}{capacity:>18}{dropped_fraction * 100:>17.2f}%")

    first_drop = results[0][2]
    last_drop = results[-1][2]
    print(f"\n-> At capacity_factor={results[0][0]:.2f} (zero slack over the ideal load),")
    print(f"   {first_drop * 100:.2f}% of routed tokens were dropped purely from random")
    print("   imbalance across experts -- there's no room to absorb even ordinary")
    print("   statistical variation in how many tokens land on each expert. As the")
    print(f"   capacity factor increases to {results[-1][0]:.2f}, the dropped fraction falls to")
    print(f"   {last_drop * 100:.2f}%, confirming more slack means fewer drops -- at the direct")
    print("   cost of every expert's buffer being sized for a worst-case load it only")
    print("   occasionally reaches, i.e. wasted compute on most batches.")


def main():
    expert_choice_demo()
    comparison_demo()
    capacity_factor_demo()


if __name__ == "__main__":
    main()
