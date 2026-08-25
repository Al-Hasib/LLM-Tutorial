# Distributed Training Basics

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `03-Distributed-Training-Basics`

## Why this matters

[Lesson 2](../02-Pretraining-Objectives/README.md) fixed *what* signal to train on. This lesson confronts a much blunter problem: [Phase 03 Lesson 5's scaling laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) established that competitive LLMs need billions of parameters trained on trillions of tokens — and neither the *memory* nor the *wall-clock compute* for that fits on one accelerator. Every real pretraining run, including the frontier models this entire course has been building toward, happens across hundreds or thousands of GPUs/TPUs simultaneously. This lesson builds the mental model for how that actually works, so the training loop from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) — which you've now trained on a single CPU core several times over — stops being a toy and starts looking like the real thing, minus the "many machines" part (which `example.py` simulates rather than actually requiring).

## What this lesson covers

- Why a single accelerator can't hold or compute a real LLM's training step
- Data Parallelism (DP): replicate the model, split the batch, average gradients
- Tensor/Model Parallelism: split individual weight matrices across devices
- Pipeline Parallelism: split layers across devices, pipeline microbatches through them
- ZeRO/FSDP: shard optimizer states, gradients, and parameters instead of replicating them
- How these techniques combine in real large-scale training

## 1. Why single-device training doesn't scale

Two independent walls get hit as model size grows:

**Memory.** Training needs to hold, simultaneously, on-device:
- **Parameters** — the weights themselves
- **Gradients** — one value per parameter, produced by backprop
- **Optimizer states** — Adam/AdamW ([Lesson 4](../04-Mixed-Precision-and-Optimization/README.md#3-adamw-decoupled-weight-decay)) keeps a running momentum *and* variance estimate per parameter — twice the parameter count again
- **Activations** — every intermediate layer's output from the forward pass, kept around because backprop needs them

For a model with `N` parameters trained with Adam in mixed precision, the params+gradients+optimizer-states alone come to roughly `16N` bytes (worked out exactly, with real numbers, in `example.py` — 2 bytes/param fp16 weights + 2 bytes/param fp16 gradients + 4+4+4 bytes/param for the fp32 master weights, momentum, and variance in the optimizer). A 7-billion-parameter model needs **~112GB** for this alone, before a single activation is stored — already past what even a high-end single accelerator (typically 40-80GB) can hold.

**Compute (wall clock).** Even where memory isn't the bottleneck, a large model trained on trillions of tokens on one device would simply take too long — years, for frontier-scale runs. Splitting the *work* across many devices that run *in parallel* is the only way to bring wall-clock time back down to something practical.

Four complementary techniques address these two walls in different ways.

## 2. Data Parallelism (DP): the simplest form

Put a full copy of the model on each of `N` devices, split every training batch into `N` equal shards (one per device), and have each device run its own forward + backward pass on its shard **independently and in parallel**:

```
device 0: batch_shard_0 -> forward -> backward -> local_gradient_0
device 1: batch_shard_1 -> forward -> backward -> local_gradient_1
   ...
device N-1: batch_shard_{N-1} -> forward -> backward -> local_gradient_{N-1}
```

Each device now has a *different* gradient (computed from a different slice of data), but every device must apply the *same* update to stay in sync. The fix is an **all-reduce**: every device sends its local gradient to every other device and they all end up holding the *average* gradient across all `N` shards — mathematically identical to computing the gradient on the whole batch on one (infinitely large) device, since the loss is a mean over examples and the mean of shard-means (equal shard sizes) is the mean of the whole. `example.py` verifies exactly this equivalence numerically. DP scales compute (N devices work in parallel) but does *nothing* for memory — every device still needs a full copy of the model, gradients, and optimizer states.

## 3. Tensor / Model Parallelism: splitting individual layers

When a single weight matrix is too large to fit in one device's memory (or you simply want one layer's matmul to run faster by splitting it), **tensor parallelism** splits the matrix itself across devices. For a linear layer `Y = XW`, split `W` column-wise across 2 devices into `W = [W_1 | W_2]`:

```
device 0 computes: Y_1 = X @ W_1
device 1 computes: Y_2 = X @ W_2
Y = concat(Y_1, Y_2)     -- requires communication to assemble
```

Every device now only ever stores and computes with a *fraction* of that layer's weights — this is what makes it possible to train models whose individual layers wouldn't fit on one device at all (Megatron-LM's approach to giant Transformer FFN and attention projection matrices). The cost is that the two devices must communicate (an all-gather or similar) to reassemble the full result before the next layer can use it — much more communication-heavy, per layer, than data parallelism's once-per-step all-reduce.

## 4. Pipeline Parallelism: splitting layers across devices

Instead of splitting *within* a layer, pipeline parallelism splits the model *by layer*: device 0 holds layers 1-8, device 1 holds layers 9-16, and so on, and activations physically flow from device to device as data passes through the network — like an assembly line. Naively, this leaves most devices idle most of the time (device 1 has nothing to do until device 0 finishes layers 1-8 for the *current* batch). The standard fix is to split each batch into several **microbatches** and pipeline them through the stages, so while device 1 processes microbatch 1, device 0 is already starting microbatch 2 — keeping every stage busy simultaneously, at the cost of some unavoidable "bubble" idle time at the very start and end of each batch.

## 5. ZeRO / FSDP: shard, don't replicate

Data parallelism's core inefficiency is that every device stores a **full, redundant** copy of parameters, gradients, and optimizer states. **ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) and PyTorch's **FSDP** (Fully Sharded Data Parallel) fix this directly: instead of every one of the `N` data-parallel devices holding 100% of the optimizer states/gradients/parameters, each device holds only a `1/N` **shard**, and the full values are reconstructed on the fly (via communication) only when actually needed for a given layer's forward/backward computation, then discarded again:

```
ZeRO stage 1: shard optimizer states only        -> ~4x memory reduction (Adam-dominated cost)
ZeRO stage 2: shard optimizer states + gradients -> more reduction
ZeRO stage 3 / full FSDP: shard params too       -> memory scales down close to linearly with N
```

This is DP's compute pattern (each device still processes its own batch shard) combined with tensor-parallelism-style memory savings — you get *both* the parallel-compute benefit of data parallelism *and* the per-device memory reduction that used to require model/tensor parallelism. `example.py` computes exactly how per-device memory drops as `N` grows, for each ZeRO stage, using the standard `16Ψ`-style byte-accounting formula from the ZeRO paper.

## 6. How real training combines all of these

Frontier LLM pretraining runs essentially never use just one of these techniques — they compose them: tensor parallelism *within* a physical multi-GPU server (where interconnects are fastest), pipeline parallelism *across* groups of servers, and ZeRO/FSDP-style sharded data parallelism *across* the outermost, largest group of replicas — a scheme often called **3D parallelism**. Choosing the right combination and degree of each is itself a significant systems-engineering problem, tuned per cluster and per model size.

## Video Script Outline

1. Motivation — "why does training GPT-scale models require a data-center, not a GPU?"
2. The two walls: memory (params+grads+optimizer+activations) and wall-clock compute
3. Data parallelism and the all-reduce, illustrated with a tiny simulated example
4. Tensor parallelism: splitting one matrix across devices
5. Pipeline parallelism: splitting layers, and the microbatch trick that avoids idle bubbles
6. ZeRO/FSDP: sharding instead of replicating, and why it's the biggest practical memory win
7. Walkthrough of `example.py` — simulated multi-device gradient averaging, and a memory calculator across ZeRO stages
8. Recap: 3D parallelism as the real-world combination of all four + preview of Lesson 4 (mixed precision, which multiplies with everything in this lesson)

## Further Reading

- Rajbhandari, Rasley, Ruwase, He (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*
- Shoeybi et al. (2019), *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (tensor parallelism)
- Huang et al. (2019), *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*
- Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (3D parallelism in practice)
- PyTorch documentation, *Fully Sharded Data Parallel (FSDP)*
