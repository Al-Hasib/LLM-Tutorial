# Mixture of Experts

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `04-Mixture-of-Experts`

## Why this matters

Lessons 1-3 covered *how* attention and generation are organized (decoder-only vs. encoder-only vs. encoder-decoder). Mixture of Experts (MoE) is an orthogonal idea entirely: it changes *how many of a model's parameters actually get used* for any given token. It's the key technique behind why some of today's largest models (Mixtral, DeepSeek-V2/V3, GPT-4 by some public reporting) can have enormous total parameter counts while keeping the actual compute cost per token far lower than a dense model of the same total size.

## Architecture at a glance

```
                token hidden state x
                        │
              ┌─────────▼─────────┐
              │       Router        │  Linear(d_model → E) + softmax
              └─────────┬─────────┘
                        │ top-k expert ids + gate weights
     ┌──────────┬───────┼────────┬──────────┐
     ▼          ▼       ▼        ▼          ▼
 Expert 0   Expert 1  Expert 2  ...   Expert E-1     ← only the top-k
     │          │       │        │          │           SELECTED experts
     └────┬─────┘       └───┬────┘          │           actually run
          │  (not selected) │  (not selected)│
          ▼                 ▼                ▼
   weighted sum of the top-k experts' outputs (gate-weighted)
                        │
                     output
```

This *replaces only the FFN sublayer* inside a decoder block ([Lesson 1](../01-Decoder-Only-Models-GPT-Family/README.md#architecture-at-a-glance)) or encoder block ([Lesson 2](../02-Encoder-Only-Models-BERT-Family/README.md#architecture-at-a-glance)) — attention, residuals, and LayerNorms are all untouched. It is an orthogonal axis of variation, not a fifth architecture family: a decoder-only model *or* an encoder-decoder model can each be built with dense FFNs or MoE FFNs. `example.py` builds this router + experts layer as real, trainable PyTorch code.

## What this lesson covers

- The core idea: decouple total parameters from compute-per-token
- The gating/router network and top-k routing
- Replacing the FFN sublayer with multiple expert FFNs
- The load-balancing problem, and the auxiliary loss that fixes it
- Trade-offs: memory vs. compute, and distributed-training overhead

## 1. The core idea

Recall from [Phase 02 Lesson 5 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#5-the-position-wise-feed-forward-network-ffn): the feed-forward sublayer processes every token independently, and typically holds roughly two-thirds of a dense Transformer layer's parameters. In a dense model, **every single parameter is used for every single token** — doubling the FFN's size doubles the compute cost for every token that passes through it.

MoE breaks that coupling: instead of one FFN, a layer holds `E` separate "expert" FFNs (say, `E=8`), and a lightweight **router** decides, *per token*, which small subset of experts (typically the top 1 or 2) actually process that token. The other `E-2` experts' weights simply aren't touched for that token at all. Total parameter count scales with `E` (you must still *store* every expert), but compute-per-token stays roughly constant, regardless of how large `E` gets — this is called **sparse activation**.

## 2. The router and top-k routing

A router is just a small linear layer plus softmax, mapping each token's hidden vector to a probability distribution over the `E` experts:

```
router_logits = x @ W_router                 # (E,) -- one score per expert
router_probs  = softmax(router_logits)        # a distribution over experts
top_k_experts, top_k_weights = top_k(router_probs, k)   # e.g. k=2
output = Σ_{i in top_k_experts} top_k_weights[i] * Expert_i(x)
```

Only the selected top-`k` experts' FFNs are actually evaluated for that token — the rest contribute nothing to that token's forward (or backward) pass, which is exactly where the compute savings come from.

## 3. The load-balancing problem

Left alone, a router trained purely to minimize the task loss tends to **collapse**: a few experts get slightly better initial routing scores, receive more tokens, get more gradient updates, get even better at the task, and receive even more tokens — a rich-get-richer feedback loop that leaves most experts undertrained and effectively wasted. This defeats the entire purpose of having many experts.

The standard fix is an **auxiliary load-balancing loss**, added on top of the normal task loss, that explicitly penalizes uneven routing — for instance, encouraging the fraction of tokens routed to each expert to stay close to `1/E`. `example.py` measures routing imbalance directly and shows the effect of adding this incentive.

## 4. Trade-offs

- **Memory**: every expert's weights must be stored (and typically kept in fast accelerator memory), so total memory footprint scales with total parameters, same as a dense model of that size — MoE's savings are in *compute*, not memory.
- **Distributed training/inference overhead**: since different tokens in a batch route to different experts, and experts are often spread across different accelerators, tokens must be shuffled ("all-to-all" communication) to reach whichever device holds their assigned expert — a real systems-engineering cost that dense models don't have.
- **Training stability**: routing decisions are discrete (top-k is not smoothly differentiable), which historically made MoE training finicky — the load-balancing loss and careful initialization are both partly aimed at this.

## Video Script Outline

1. Motivation — "what if a model could have way more parameters without a way more expensive forward pass?"
2. Router + top-k selection, replacing the single FFN from Phase 02
3. The rich-get-richer collapse problem, made concrete
4. The load-balancing auxiliary loss
5. Memory vs. compute trade-off, and the distributed "all-to-all" cost
6. Walkthrough of `example.py` — a working top-k MoE layer, compute-cost comparison vs. an equivalent dense FFN, and the load-balancing effect measured directly
7. Recap + pointer to [Lesson 7's survey](../07-Survey-of-Popular-Open-LLMs/README.md), where MoE shows up in real deployed models like Mixtral

## Further Reading

- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (the modern MoE-for-deep-learning origin)
- Fedus, Zoph, Shazeer (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*
- Jiang et al. (2024), *Mixtral of Experts* (a widely-used open MoE model, revisited in [Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md))
