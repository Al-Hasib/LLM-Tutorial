# Model Distillation and Pruning

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `05-Model-Distillation-and-Pruning`

## Why this matters

[Quantization](../02-Quantization/README.md) shrinks a model by representing the *same* weights with fewer bits. This lesson covers the two techniques that instead shrink a model by removing weights (or training a smaller model outright) — a complementary, not competing, set of levers on the exact same "make it smaller and cheaper" problem. You already met a real distilled model by name back in [Phase 03 Lesson 2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#6-roberta-and-todays-encoder-only-landscape) — DistilBERT — without yet seeing the mechanism that makes it work. This lesson builds that mechanism from scratch.

## What this lesson covers

- Knowledge distillation: training a small student to match a large teacher's soft outputs
- Why soft targets carry more signal than hard labels ("dark knowledge")
- Structured vs. unstructured pruning
- Magnitude-based pruning: the simplest practical algorithm
- The accuracy-vs-sparsity trade-off curve

## 1. Knowledge distillation

Hinton, Vinyals, Dean (2015) start from an observation: a trained "teacher" network's output distribution over classes carries more information than just "which class is correct." Consider a teacher classifying an image of a "2" that also assigns non-trivial probability to "7" (they share a similar top stroke) but almost none to "9": that relative confidence across the *wrong* classes reflects real, learned similarity structure — Hinton et al. call this **dark knowledge** — and a one-hot hard label ("the answer is 2") throws it away entirely.

Distillation trains a smaller **student** network to match the teacher's full output distribution, not just its top prediction, using a **temperature-scaled softmax** to soften the distribution and expose more of that relative structure:

```
soft_target = softmax(teacher_logits / T)      # T > 1 "softens" the distribution, revealing more of the runner-up classes
soft_prediction = softmax(student_logits / T)
distillation_loss = KL_divergence(soft_target, soft_prediction)
total_loss = alpha * distillation_loss * T^2 + (1 - alpha) * hard_label_cross_entropy
```

A higher temperature `T` flattens the distribution further (at `T -> infinity`, every class approaches uniform probability), exposing more of the teacher's relative-confidence structure at the cost of also amplifying noise in very-low-probability classes — `T` is a tunable hyperparameter, not a fixed constant. The `T^2` factor rescales the distillation loss's gradient magnitude to stay comparable to the hard-label term's, since scaling logits down by `T` also scales gradients down by roughly `T`.

## 2. Structured vs. unstructured pruning

Pruning removes parameters from an already-trained network rather than training a smaller one from the start:

- **Structured pruning** removes whole, architecturally-meaningful units — entire neurons, attention heads, or layers. The result is a smaller *dense* network that runs on ordinary hardware with no special support, but coarse-grained removal (a whole head or layer at a time) tends to cost more accuracy per parameter removed than finer-grained pruning.
- **Unstructured pruning** zeroes out individual weights anywhere in a weight matrix, regardless of position. This achieves much higher compression for a given accuracy loss, because it can remove exactly the least-useful individual connections — but a matrix that's mostly zeros scattered arbitrarily throughout still needs *sparse-matrix-aware* hardware/kernels to actually run faster or smaller in memory; naively storing it dense wastes the saving entirely.

## 3. Magnitude pruning

The simplest practical unstructured-pruning algorithm, and still a strong baseline: rank every weight by its absolute value, and zero out the smallest fraction (the **sparsity level**, e.g. 80% of all weights set to exactly zero). The intuition is that a weight close to zero already contributes little to the network's output, so removing it should disturb behavior the least — an intuition that holds up surprisingly well in practice, especially at moderate sparsity levels, though accuracy inevitably degrades as sparsity climbs high enough to start removing weights that *do* matter.

## 4. The accuracy-vs-sparsity trade-off

There is no free lunch: some sparsity is nearly free (redundant capacity gets removed with minimal accuracy cost), but past some point every additional percentage of sparsity costs measurably more accuracy, and the curve typically has a "knee" — a sparsity level beyond which accuracy degrades sharply rather than gradually. `example.py` §2 measures this curve directly on a real trained toy network, at several sparsity levels, with real numbers rather than an assumed shape.

## Video Script Outline

1. Motivation — "shrink by removing weights, or train small in the first place, instead of quantization's fewer-bits-per-weight approach"
2. Dark knowledge: why a wrong-class probability still carries signal a hard label throws away
3. The distillation loss, temperature scaling, and the `T^2` rescaling detail
4. Structured vs. unstructured pruning, and the sparse-hardware caveat
5. Magnitude pruning's algorithm, in one sentence
6. Walkthrough of `example.py` §1 — train a teacher, then a student two ways (hard labels alone vs. distillation), compare real held-out accuracy
7. Walkthrough of `example.py` §2 — magnitude-prune a trained network across several sparsity levels, plot the real accuracy-vs-sparsity curve
8. Recap + pointer to [Lesson 6](../06-Cost-and-Latency-Optimization/README.md), where a smaller distilled/pruned model becomes the "cheap" tier of a routing cascade

## Further Reading

- Hinton, Vinyals, Dean (2015), *Distilling the Knowledge in a Neural Network*
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
- Han, Mao, Dally (2015), *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* (magnitude pruning at scale, combined with quantization)
- Frankle & Carbin (2019), *The Lottery Ticket Hypothesis* (why some sparse subnetworks train just as well as the full dense network)
