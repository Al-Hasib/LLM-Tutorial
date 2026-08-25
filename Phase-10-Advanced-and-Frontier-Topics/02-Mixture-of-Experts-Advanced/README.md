# Mixture of Experts, Advanced

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `02-Mixture-of-Experts-Advanced`

## Why this matters

[Phase 03 Lesson 4](../../Phase-03-LLM-Architectures-and-Types/04-Mixture-of-Experts/README.md) introduced the core MoE idea: replace one dense FFN with `E` experts, route each token to its top-`k` experts with a small router, and get far more total parameters without a proportional increase in compute per token. It also introduced the central failure mode — router collapse, where a small early advantage for one expert snowballs into that expert absorbing most of the tokens — and the standard fix, an auxiliary load-balancing loss that nudges routing back toward uniform.

That basic picture is enough to understand *why* MoE works, but it glosses over how routing is actually done in production systems that serve millions of requests with strict latency and memory budgets. Following the previous lesson on [Multimodal LLMs](../01-Multimodal-LLMs/README.md), this lesson goes one level deeper on the mechanics of routing itself: an entirely different routing paradigm that sidesteps load balancing by construction, the token-dropping behavior that top-k routing has in real implementations (which the toy example in Phase 03 didn't need to model), and the fine-grained "many small experts + shared experts" design that modern open MoE models converged on. The next lesson, [State Space Models (Mamba)](../03-State-Space-Models-Mamba/README.md), leaves MoE behind entirely and looks at a non-Transformer alternative to attention — a reminder that sparsity-via-routing and sequence-mixing-without-attention are two independent axes along which the standard Transformer recipe is being pushed.

## What this lesson covers

- A quick recap of top-k token-choice routing and why it needs an auxiliary loss (pointer back to Phase 03, not re-derived)
- Expert-Choice routing (Zhou et al., 2022): flipping who picks whom, so experts pick tokens instead of tokens picking experts
- Why Expert-Choice guarantees perfect load balance *by construction*, and what it gives up to get that (token dropping, multiple selection)
- Capacity factor and token dropping in ordinary top-k routing: the practical inefficiency real implementations have to handle that the basic lesson's toy example never triggers
- Fine-grained expert segmentation and shared experts (DeepSeekMoE-style): trading a few large experts for many small ones plus always-on shared experts
- A from-scratch PyTorch implementation of Expert-Choice routing, a head-to-head comparison against token-choice under the exact same router-bias scenario from Phase 03, and a capacity-factor/token-dropping simulation

## 1. Recap: top-k token-choice routing and its collapse problem

In the routing scheme from Phase 03, each **token** is the active party: it computes a score against every expert, picks its top-`k`, and gets processed by those experts' FFNs.

```
router_logits = x @ W_router                            # (E,) one score per expert, per token
router_probs  = softmax(router_logits)
top_k_experts, top_k_weights = top_k(router_probs, k)
output = Σ_{i in top_k_experts} top_k_weights[i] * Expert_i(x)
```

Nothing in this formulation constrains how many tokens any given expert ends up with — it's an emergent property of a training process that has every incentive to make it *uneven* (better-trained experts look more attractive to the router, so they get chosen more, so they get more gradient updates, so they get even better). [Phase 03 Lesson 4](../../Phase-03-LLM-Architectures-and-Types/04-Mixture-of-Experts/README.md#3-the-load-balancing-problem) covers this collapse dynamic and its auxiliary-loss fix in detail — that material isn't repeated here. What matters for this lesson is the *shape* of the fix: it's a soft, statistical nudge added to the loss function, tuned by a hyperparameter (the aux-loss weight), that only encourages balance on average over a batch. It doesn't guarantee balance for any individual batch, and getting the aux-loss weight wrong is itself a real tuning headache — too weak and collapse creeps back in, too strong and it distorts the router away from routing decisions that would otherwise help the task loss.

Expert-Choice routing asks a different question: what if load balance were enforced structurally, so no auxiliary loss is needed at all?

## 2. Expert-Choice routing: experts pick tokens

Zhou et al. (2022) invert the routing decision. Instead of each token choosing its top-`k` experts, **each expert chooses its top-`C` tokens** from the batch, where `C` (the expert's *capacity*) is a fixed number computed in advance from the batch size.

Formally, for a batch of `T` tokens and `E` experts, compute the same kind of affinity matrix as before — one score per (token, expert) pair — but read it out along the *other* axis:

```
A = X @ W_router                     # (T, E) affinity matrix, one row per token, one column per expert
G = softmax(A, dim=-1)               # per-token normalization, used only as the gate value later

for expert e in 1..E:
    scores_e = A[:, e]                       # column e: every token's affinity to expert e
    top_C_tokens = top_k(scores_e, C)         # THIS expert's top-C tokens, by score
    output[top_C_tokens] += G[top_C_tokens, e] * Expert_e(x[top_C_tokens])
```

The capacity `C` is fixed ahead of time as a target average load, typically

```
C = (T * k) / E
```

where `k` is the average number of experts each token should end up matched with (analogous to top-`k` in token-choice; `k=1` or `k=2` are common choices). Every expert then reads off exactly its top-`C` tokens *by affinity score* from its own column of `A` — a per-column top-`C`, not a per-row top-`k`.

**Why this guarantees perfect load balance by construction:** every expert always selects exactly `C` tokens, full stop. There is no dependency on training dynamics, no rich-get-richer feedback loop to correct, because there's no way for one expert to end up with more tokens than another — the top-`C` operation caps it at exactly `C` by definition. This is true even at random initialization, before a single gradient step: balance isn't a training *outcome* here, it's a structural *invariant* of the routing operation itself. That's why Expert-Choice needs no auxiliary load-balancing loss whatsoever.

**What it costs to get that guarantee:**

- **Token dropping.** Because experts pick tokens independently of each other, a token can simply fail to appear in *any* expert's top-`C` list — every expert preferred `C` other tokens over it. That token gets no expert computation at all for this layer (it passes through via the residual connection alone, if the surrounding architecture provides one).
- **Multiple selection.** Symmetrically, a token can appear in *several* experts' top-`C` lists at once (nothing prevents expert 3 and expert 7 from both wanting the same popular token), so it receives contributions from more than one expert — the per-token "how many experts did I get" count is no longer a fixed `k`, only an *average* of `k` across the batch.
- **Batch dependence.** Because the top-`C` selection is computed relative to the *other tokens currently in the batch*, the same token can be routed differently depending on what else happens to be in its batch — a property token-choice routing doesn't have (there, a token's routing depends only on itself). This complicates autoregressive generation, where you'd like a token's routing to be stable regardless of what else is being decoded alongside it, and is one reason Expert-Choice is more commonly discussed for encoder/training-time settings than for causal decoding.

Expert-Choice trades a *soft, tunable, imperfect* fix for load balancing (the aux loss) for a *hard, exact, structural* one — at the cost of a new failure mode (dropped and multiply-served tokens) that top-k routing's aux loss doesn't create.

## 3. Capacity factor and token dropping in top-k routing

Token-choice routing has its own version of a capacity limit, and it's easy to miss if you only ever look at a toy implementation (like Phase 03's `MoELayer`, which lets every chosen expert process every token routed to it, with no cap). Real distributed MoE implementations cannot do this: each expert typically lives on a specific accelerator with a fixed-size buffer allocated for how many tokens it can process in one forward pass, because that buffer size has to be decided *before* the batch is seen (you can't dynamically resize a matmul's shape mid-training-step without stalling the whole pipeline).

So every expert is given a hard **capacity** — the maximum number of tokens it will accept in a given batch — computed as:

```
capacity = capacity_factor * (num_tokens / num_experts)          # top-1 routing
capacity = capacity_factor * (num_tokens / num_experts) * top_k  # general top-k routing
```

`num_tokens / num_experts` is the load each expert would receive under *perfectly even* routing; the **capacity factor** (e.g. `1.0`, `1.25`, `2.0`) is slack added on top of that ideal, to absorb the routing imbalance that inevitably exists even with an aux loss pulling toward uniform.

When a token is routed to an expert that has *already filled its capacity* for this batch (processing tokens in order, whichever arrived first), standard implementations (Switch Transformer among them) simply **drop** that token for that expert: it receives no computation from that expert at all and is passed through the layer via the residual stream alone, exactly as if it hadn't been routed anywhere. This is a real, silent efficiency loss — dropped tokens get less model capacity applied to them than tokens that made it under capacity, purely as an artifact of what else happened to be in their batch, not because of anything about the token itself.

- **`capacity_factor = 1.0`** gives no slack at all: even a small amount of routing imbalance (which always exists in practice, aux loss or not) forces drops.
- **Raising the capacity factor** (`1.25`, `2.0`, ...) gives experts room to absorb imbalance without dropping tokens, at the direct cost of wasted compute and memory — every expert's buffer is sized for its *worst-case* load, so most experts sit partially empty most of the time, computing on padding.

This is a genuine three-way trade-off: bigger capacity factor means fewer dropped tokens but more wasted compute; smaller capacity factor means less waste but more dropped tokens; and the aux-loss weight interacts with both, since better-balanced routing needs less capacity slack to avoid drops in the first place. `example.py` §3 simulates this directly and reports the actual dropped-token fraction at several capacity factors.

## 4. Fine-grained expert segmentation and shared experts

DeepSeekMoE (Dai et al., 2024) pushes on a different axis: instead of a handful of *large* experts (e.g. 8 experts, each the size of a standard FFN), it uses **many more, smaller** experts — carving each would-be expert's parameters into several finer-grained pieces (e.g. subdividing 8 large experts into 64 small ones) and routing top-`k` over the finer-grained set with a proportionally larger `k`. The intuition: a single large expert routed to as a monolithic unit is forced to be a "jack of all trades" for every token that reaches it, whereas many small experts let the router compose a more precise, more specialized combination for each token — closer to picking exactly the right blend of narrow specialists rather than the closest-fit generalist.

On top of the fine-grained routed experts, DeepSeekMoE adds a small number of **shared experts**: experts that are *not* routed at all — every single token passes through all of them unconditionally, in addition to whichever routed experts it's matched with. The idea is to let the shared experts absorb the common, generic computation that essentially every token needs (general-purpose transformations that would otherwise be redundantly re-learned by many different routed experts), freeing the routed experts to specialize on what's actually token-specific rather than spending capacity re-deriving the same generic transformation as their neighbors. This is a big part of how DeepSeek-V2/V3 achieve strong quality with a low fraction of *active* parameters per token relative to their enormous *total* parameter count.

## Video Script Outline

1. Recap in one breath: token-choice top-k routing and the collapse problem it needs an aux loss to fight (pointer to Phase 03)
2. The inversion — "what if experts picked tokens instead?" — introduce Expert-Choice routing
3. Walk through the affinity-matrix formulation: same matrix as before, read along the other axis
4. Why perfect load balance falls out for free, even at initialization, with no aux loss anywhere
5. The price of that guarantee: dropped tokens and multiply-served tokens, and why Expert-Choice suits training/encoders more than causal decoding
6. Capacity factor in ordinary top-k routing: a hard buffer size decided before the batch is even seen, and the token-dropping that results
7. Fine-grained segmentation and shared experts, DeepSeekMoE-style — many small specialists plus always-on generalists
8. Walkthrough of `example.py` — a from-scratch Expert-Choice layer, a head-to-head vs. token-choice under the same Phase 03 router-bias scenario, and real measured token-dropping-rate numbers across capacity factors

## Further Reading

- Zhou et al. (2022), *Mixture-of-Experts with Expert Choice Routing*
- Fedus, Zoph, Shazeer (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (capacity factor and token dropping in a real, widely deployed implementation)
- Dai et al. (2024), *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (fine-grained expert segmentation and shared experts)
