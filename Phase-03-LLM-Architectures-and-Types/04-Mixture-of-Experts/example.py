"""
Mixture of Experts

Three demos:
  1. A working top-k Mixture-of-Experts layer in PyTorch (router + N
     expert FFNs), contrasted with an equivalent dense FFN.
  2. A compute-cost comparison: MoE gives you many more total
     parameters for roughly the SAME per-token FLOPs as a much smaller
     dense FFN.
  3. The load-balancing problem, reproduced directly: train a router
     with a slight initial bias, with vs. without an auxiliary
     load-balancing loss, and measure how evenly tokens get routed.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. A top-k Mixture-of-Experts layer
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


def moe_forward_demo():
    print("=" * 70)
    print("1. A WORKING TOP-K MOE LAYER")
    print("=" * 70)

    d_model, d_ff, num_experts, top_k = 16, 32, 8, 2
    moe = MoELayer(d_model, d_ff, num_experts, top_k)

    batch_size = 6
    x = torch.randn(batch_size, d_model)
    output, router_probs, chosen = moe(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(output.shape)}  (unchanged, same as a dense FFN would give)")
    print(f"\nEach token's top-{top_k} chosen experts:")
    for i in range(batch_size):
        print(f"  token {i}: experts {chosen[i].tolist()}  "
              f"(weights {router_probs[i, chosen[i]].detach().numpy().round(3)})")


# ---------------------------------------------------------------------------
# 2. Compute-cost comparison: MoE vs. an equivalent dense FFN
# ---------------------------------------------------------------------------

def ffn_flops_per_token(d_model, d_ff):
    """Rough multiply-add FLOPs for one token through one FFN (2 matmuls)."""
    return 2 * (d_model * d_ff) * 2   # x2 for the two Linear layers, x2 for mult+add


def compute_cost_demo():
    print("\n" + "=" * 70)
    print("2. TOTAL PARAMETERS vs. COMPUTE-PER-TOKEN: DENSE FFN vs. MOE")
    print("=" * 70)

    d_model = 768
    d_ff = 4 * d_model
    num_experts = 8
    top_k = 2

    dense_params = 2 * d_model * d_ff             # one FFN's parameters (ignoring biases)
    moe_total_params = num_experts * dense_params  # must STORE every expert
    moe_active_params_per_token = top_k * dense_params  # only these get COMPUTED per token

    dense_flops = ffn_flops_per_token(d_model, d_ff)
    moe_flops = top_k * dense_flops   # router adds a negligible d_model*num_experts on top

    print(f"d_model={d_model}, d_ff={d_ff}, num_experts={num_experts}, top_k={top_k}\n")
    print(f"{'':30s}{'total params':>16}{'active params/token':>22}{'FLOPs/token':>16}")
    print(f"{'Dense FFN':30s}{dense_params:>16,}{dense_params:>22,}{dense_flops:>16,}")
    print(f"{'MoE layer (' + str(num_experts) + ' experts)':30s}"
          f"{moe_total_params:>16,}{moe_active_params_per_token:>22,}{moe_flops:>16,}")

    print(f"\n-> The MoE layer stores {moe_total_params / dense_params:.0f}x more total")
    print(f"   parameters than one dense FFN, but each token only activates "
          f"{moe_active_params_per_token / dense_params:.0f}x")
    print("   as much compute as a single dense FFN -- NOT 8x. This is the entire")
    print("   value proposition: much more model capacity, without a proportional")
    print("   increase in compute cost per token.")


# ---------------------------------------------------------------------------
# 3. The load-balancing problem, reproduced directly
# ---------------------------------------------------------------------------

def train_moe_toy_task(use_aux_loss, num_experts=4, steps=600, aux_loss_weight=3.0):
    torch.manual_seed(0)
    d_model, d_ff = 16, 32
    moe = MoELayer(d_model, d_ff, num_experts, top_k=1)

    # Deliberately bias the router's initial weights slightly toward expert 0,
    # simulating the kind of small random initialization advantage that, left
    # uncorrected, snowballs into full collapse (the README's "rich get richer").
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
            # Switch-Transformer-style load-balancing loss: encourages both the
            # fraction of tokens routed to each expert (f_i) and the average
            # router probability mass on each expert (P_i) to be uniform.
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


def load_balancing_demo():
    print("\n" + "=" * 70)
    print("3. THE LOAD-BALANCING PROBLEM, MEASURED DIRECTLY")
    print("=" * 70)
    print("Same toy regression task, same slight initial bias toward expert 0.")
    print("Counting which expert each token gets routed to, over the FINAL 50")
    print("training steps (1600 token-routings total):\n")

    num_experts = 4
    counts_no_aux = train_moe_toy_task(use_aux_loss=False, num_experts=num_experts)
    counts_with_aux = train_moe_toy_task(use_aux_loss=True, num_experts=num_experts)

    print(f"{'expert':>8}" + "".join(f"{i:>10}" for i in range(num_experts)))
    print(f"{'no aux loss':>8}" + "".join(f"{int(c):>10}" for c in counts_no_aux))
    print(f"{'with aux':>8}" + "".join(f"{int(c):>10}" for c in counts_with_aux))

    def imbalance(counts):
        fractions = counts / counts.sum()
        ideal = 1.0 / len(counts)
        return (fractions - ideal).abs().sum().item()

    print(f"\nTotal deviation from perfectly even routing (0 = perfectly balanced):")
    print(f"  without aux loss: {imbalance(counts_no_aux):.3f}")
    print(f"  with aux loss:    {imbalance(counts_with_aux):.3f}")
    print("\n-> Without correction, the small initial bias toward expert 0 tends to")
    print("   snowball: more tokens routed there means more gradient updates,")
    print("   which reinforces the router's preference further. The auxiliary")
    print("   load-balancing loss counteracts this directly, keeping routing")
    print("   much closer to evenly spread across all experts.")


def main():
    moe_forward_demo()
    compute_cost_demo()
    load_balancing_demo()


if __name__ == "__main__":
    main()
