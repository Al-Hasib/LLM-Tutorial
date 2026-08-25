# Scaling Laws

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `05-Scaling-Laws`

## Why this matters

Every lesson so far has been about *what* to build. This lesson is about a question that turned out to matter just as much: given a fixed compute budget, how big should the model be, and how much data should you train it on? The empirical answer — discovered by fitting power-law curves to hundreds of training runs — directly explains why LLMs got as big as they did, why some famous large models (like the original 175B GPT-3) were arguably *undertrained*, and why modern LLMs train on far more data per parameter than early ones did.

## What this lesson covers

- The empirical scaling-law relationship between loss, model size, dataset size, and compute
- Kaplan et al. (2020): the original scaling laws, and their "bigger is better" recommendation
- Chinchilla (Hoffmann et al., 2022): the compute-optimal correction
- The power-law loss formula, and where a model's performance ceiling actually comes from
- Practical implications for how real LLMs are trained today

## 1. The empirical relationship

Train models of many different sizes (`N` parameters) on many different amounts of data (`D` tokens), plot the final training loss against `N` and `D` on a log-log scale, and a strikingly clean pattern emerges: loss decreases as a **power law** in both — a straight line on a log-log plot, extending smoothly over many orders of magnitude. This was the core empirical finding of Kaplan et al. (2020): scaling behavior is remarkably *predictable*, which means you can extrapolate: train a handful of smaller, cheap models, fit the curve, and predict how a much larger, expensive model will perform before ever training it.

## 2. The power-law loss formula

A widely used functional form (Chinchilla's version) decomposes the achievable loss into three additive pieces:

```
L(N, D) = E + A / N^α + B / D^β
```

- `E` — the **irreducible loss**: the entropy of natural language itself; no amount of scale drives loss below this floor.
- `A / N^α` — the loss penalty from having a **finite number of parameters**; shrinks as `N` grows.
- `B / D^β` — the loss penalty from having a **finite amount of training data**; shrinks as `D` grows.

`example.py` fits and visualizes this exact formula (using the published Chinchilla-fitted constants) to make the trade-off concrete.

## 3. Kaplan et al. (2020): the original recommendation

The first major scaling-laws paper found that, for a fixed compute budget, loss improved faster by growing the model than by growing the dataset — leading to the practical recommendation: **train very large models, and don't worry about training them to full convergence on a comparatively modest amount of data.** GPT-3 (175B parameters, trained on ~300B tokens) was built largely following this guidance.

## 4. Chinchilla (2022): models were undertrained

Hoffmann et al. revisited the question with a wider, more carefully controlled sweep of training runs and reached a different conclusion: for a fixed compute budget, **model size and dataset size should scale roughly together** — very roughly, about 20 training tokens per parameter — rather than favoring model size so heavily. Their headline result: a **70B-parameter model** ("Chinchilla"), trained on **1.4 trillion tokens**, outperformed the much larger **280B-parameter Gopher** model, trained on only ~300B tokens, using **the same total training compute** for both. GPT-3-scale models, by this analysis, were significantly *undertrained* relative to their parameter count — a smaller model trained longer would have been a better use of the same compute budget.

## 5. Why compute, not just parameters, is the real currency

Both papers frame the question the same way: given a compute budget `C` (roughly `C ≈ 6ND` FLOPs, since each token requires roughly `2N` FLOPs for the forward pass and `4N` for the backward pass), how should you split it between `N` and `D`? `example.py` performs exactly this optimization directly: for a range of compute budgets, it searches over possible `(N, D)` splits satisfying `C = 6ND` and finds the one that minimizes `L(N, D)` under the power-law formula — recovering, from the formula alone, the same qualitative conclusion Chinchilla reached empirically.

## 6. Practical implications

This is why essentially every LLM released after Chinchilla (LLaMA, Mistral, and most others) trains on hundreds of billions to trillions of tokens *per model*, even at parameter counts of "only" 7B-70B — a stark departure from GPT-3-era training recipes. It's also why smaller, "over-trained beyond Chinchilla-optimal" models became popular for practical deployment: Chinchilla-optimal minimizes *training* compute for a given loss, but a smaller model that's cheaper to run at *inference* time is often worth training on even more data than the formula's training-compute-optimal point suggests, since inference cost (paid once per user request, forever) can dwarf training cost (paid once) at deployment scale.

## Video Script Outline

1. Motivation — "given a fixed budget, what's the smartest way to spend it?"
2. The power-law finding: loss vs. N and D, log-log straight lines
3. The additive loss formula: irreducible + finite-N + finite-D terms
4. Kaplan's "go bigger" era vs. Chinchilla's correction, told as a before/after story
5. The `C ≈ 6ND` compute-budget framing
6. Walkthrough of `example.py` — fit the formula, then solve for the compute-optimal `(N, D)` at several budgets
7. Recap: why post-Chinchilla LLMs train on so much more data than GPT-3 did, and the training-vs-inference-cost twist

## Further Reading

- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla)
- Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models* (an explicit real-world application of training well past Chinchilla-optimal, for cheaper inference)
