# Model Merging and Editing

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `04-Model-Merging-and-Editing`

## Why this matters

[Phase 05: Full Fine-tuning vs PEFT](../../Phase-05-Finetuning-LLMs/01-Full-Finetuning-vs-PEFT/README.md) and [LoRA and QLoRA](../../Phase-05-Finetuning-LLMs/02-LoRA-and-QLoRA/README.md) established a simple fact about how the field actually works: a huge number of people take the *same* open-weight base model and fine-tune it in different directions — one team specializes it for coding, another for a foreign language, another for a customer-support tone, another for math reasoning. Each of those fine-tuning runs produces a full set of weights that lives in the same parameter space as the base model it started from, differing from it only by whatever gradient updates the fine-tuning run applied.

This lesson asks the question that falls naturally out of that observation: if two fine-tuned models are just "the base model plus some weight changes," can those weight changes be treated as objects in their own right — added, subtracted, blended, or surgically overwritten — without ever running gradient descent again? The surprising empirical answer is yes, to a useful degree. **Task arithmetic** lets you combine several fine-tuned specializations into one model by literally adding their weight deltas together. **SLERP** lets you blend two fine-tuned models into one that inherits properties from both, without the naive averaging failure modes. **Model editing** (ROME) goes further still, showing that even a single, specific fact stored in a trained model can be located and overwritten directly in the weights, without retraining on anything. All three techniques treat trained weights not as an opaque, immutable end product, but as a manipulable artifact — a theme that will resurface, from the opposite direction, in the next and final lesson, [Interpretability and Mechanistic Interpretability](../05-Interpretability-and-Mechanistic-Interpretability/README.md), which asks not "how do we edit what a model knows" but "how do we even tell what it knows and how it computes it." It also closes a loop with [State Space Models (Mamba)](../03-State-Space-Models-Mamba/README.md): merging and editing are architecture-agnostic ideas — they operate purely on the fact that a model *is* a big vector of numbers, whether those numbers implement attention or a recurrent state-space scan.

## What this lesson covers

- **Task arithmetic** (Ilharco et al., 2022): defining a "task vector" as the difference between fine-tuned and base weights, and adding/subtracting those vectors to combine or remove capabilities
- **SLERP** (Spherical Linear intERPolation): why interpolating along the great-circle arc between two weight vectors preserves their norm, unlike naive linear averaging
- **Model editing with ROME** (Meng et al., 2022): locating a specific factual association inside a model's MLP layers via causal tracing, and overwriting it with a single targeted rank-one weight update — no retraining
- Honest limitations: why merging degrades as tasks conflict or scale grows, and why localized edits can have non-local side effects

## 1. Task Arithmetic: weight deltas as first-class objects

Suppose you start from one shared base model with parameters `theta_base` (e.g. a pretrained LLM before any instruction tuning). You fine-tune it on task `i` — say, a coding dataset — and get a new set of weights `theta_finetuned_i`. Ilharco et al. (2022) define the **task vector** for that fine-tune as simply the elementwise difference:

```
tau_i = theta_finetuned_i - theta_base
```

`tau_i` is a vector living in exactly the same space as the weights themselves (one number per parameter). It captures, additively, "whatever fine-tuning on task `i` changed about the base model." The paper's central empirical finding is that these vectors behave, to a surprisingly useful degree, like independent, composable edits:

```
theta_merged = theta_base + sum_i( lambda_i * tau_i )
```

- **Adding task vectors together** (`theta_base + tau_1 + tau_2`) tends to produce a single merged model that is competent at *both* task 1 and task 2 — without ever training on both simultaneously, and without access to either task's original training data at merge time (you only need the two checkpoints).
- **Negating a task vector** (`theta_base - tau_toxic`) tends to *suppress* whatever behavior that fine-tune induced — e.g. subtracting a task vector obtained by fine-tuning on toxic text measurably reduces toxic generations in the resulting model, without a separate detoxification training run.
- The scaling coefficients `lambda_i` (usually a single shared `lambda`, tuned on a small validation set) control how strongly each task vector's edit is applied; `lambda_i = 1` is the naive default, but it is frequently *not* the best choice in practice — as `example.py` demonstrates empirically, merged-model quality can be sensitive to this coefficient, with some intermediate value trading off "enough of the edit to help" against "not so much that it wrecks the model's other capabilities."

Why does this work at all? Intuitively, fine-tuning on a downstream task typically moves the weights only a modest distance from the base model, along a task-specific direction. If two tasks' directions are close to orthogonal, adding both updates barely interferes with either — each task's direction stays approximately unaffected by the other's presence. If the two tasks pull weights in *conflicting* directions (e.g. both fine-tunes want to repurpose the same attention head for different, incompatible purposes), the sum partially cancels out, and merge quality degrades — this is precisely the failure mode later merging techniques like **TIES-Merging** (Yadav et al., 2023) target directly, by explicitly resolving sign conflicts between task vectors before summing them.

## 2. SLERP: interpolating along the sphere, not the straight line

Task arithmetic adds vectors together. A different, equally common merging move is to *interpolate* between two models' weights — for example, to produce a model that sits "50% of the way between" two fine-tunes of the same base. The naive approach is **linear interpolation (LERP)**:

```
lerp(p0, p1, t) = (1 - t) * p0 + t * p1
```

LERP has a subtle problem. High-dimensional weight vectors from two different but related fine-tunes typically have *similar norm* (magnitude) but point in *different directions* — geometrically, they sit at roughly the same distance from the origin but at some nontrivial angle apart. Averaging two vectors that are similar in length but different in direction produces a vector that is **shorter** than either input — exactly the way the average of two unit vectors 90 degrees apart has length `~0.71`, not `1`. Applied to model weights, this means LERP systematically **shrinks the effective magnitude of the weights** as `t` moves away from the endpoints, with the shrinkage worst at `t = 0.5` — silently scaling down activations and degrading the merged model's behavior in a way that has nothing to do with the *content* of either fine-tune, purely an artifact of vector averaging.

**Spherical Linear interpolation (SLERP)** fixes this by interpolating along the great-circle arc connecting the two vectors on the hypersphere they both approximately lie on, rather than the straight chord between them:

```
omega = arccos( (p0 . p1) / (|p0| * |p1|) )      # angle between p0 and p1

slerp(p0, p1, t) = (sin((1-t) * omega) / sin(omega)) * p0
                  + (sin(t * omega)     / sin(omega)) * p1
```

`omega` is the angle between the two weight vectors, computed from their dot product exactly as in ordinary vector geometry. The two coefficients `sin((1-t)*omega)/sin(omega)` and `sin(t*omega)/sin(omega)` are not a simple `(1-t, t)` split — they are chosen so that the interpolated vector's magnitude varies smoothly between `|p0|` and `|p1|` along the arc, instead of dipping below both. When `t = 0`, the formula reduces to `p0`; when `t = 1`, it reduces to `p1`; at every `t` in between, the result stays on (or very near) the sphere connecting the two, rather than cutting through its interior. `example.py` computes both LERP and SLERP between two genuinely different weight vectors and prints the resulting norm at every `t` from 0 to 1, making the dip vs. no-dip contrast a directly observed number rather than an assertion.

In practice, SLERP is applied per-tensor (or per-layer) rather than to the entire flattened parameter vector at once, since different layers can have very different scales and "directions" — merging tools such as `mergekit` popularized layer-wise SLERP as a default strategy for blending two fine-tunes of the same base architecture, often outperforming plain weight averaging (the technique behind **Model Soups**, Wortsman et al., 2022) precisely because it does not suffer the norm-collapse problem.

## 3. Model Editing: ROME and surgical fact edits

Task arithmetic and SLERP both operate on *entire* fine-tuned checkpoints — the unit of editing is "everything this training run changed." **ROME** (Rank-One Model Editing; Meng et al., 2022) operates at the opposite extreme: it edits a **single fact** the model outputs, by changing a tiny, precisely identified slice of weights, with no gradient-descent training at all.

The method has two stages:

**1. Causal tracing — find *where* a fact lives.** Meng et al. run a prompt like `"The Eiffel Tower is located in the city of ___"` through the model, then systematically corrupt and restore individual hidden states (at different layers and token positions) while observing how much each restoration recovers the correct answer `"Paris"`. This produces a causal map showing that factual recall for this kind of associative fact is disproportionately mediated by a small number of **mid-layer MLP modules**, specifically at the token position of the fact's subject (`"Eiffel Tower"`). This localizes the edit target to one specific weight matrix in one specific layer — not "somewhere in the network," a concrete address.

**2. Rank-one update — overwrite the fact at that address.** ROME models the relevant MLP's down-projection weight matrix `W` as an associative key-value store: it maps a "key" vector `k` (roughly, an internal representation of the subject, `"Eiffel Tower"`) to a "value" vector `v` (roughly, an internal representation that decodes to the object, `"Paris"`). To insert a new association — say, editing the model to instead answer `"Rome"` — ROME solves for a new value vector `v_new` that would produce the desired output, then applies a minimal, targeted **rank-one update** to `W`:

```
W_new = W + (v_new - v_old) * k^T / (k^T * k)
```

This is a single outer-product update (`rank 1`, since it is the product of one column vector and one row vector) chosen to be the *smallest* change to `W`, in a specific least-squares sense, that redirects the key `k` to the new value `v_new` while disturbing other keys as little as possible. Crucially, there is no backpropagation, no loss function optimized over many steps, and no training data beyond the single fact being edited — it is closed-form linear algebra applied to one weight matrix. The edit takes effect immediately and, in the paper's evaluation, generalizes reasonably well to paraphrases of the edited fact ("Which city is the Eiffel Tower in?") while leaving mostly unrelated facts intact — though "mostly" is doing real work in that sentence: follow-up work (e.g. on sequential and mass editing) found that ROME-style edits can degrade other stored knowledge as more edits are stacked, and that a single localized edit can still have non-local ripple effects on facts that share machinery with the edited one. Model editing at this granularity remains an open, actively researched problem rather than a fully solved one — which is exactly the kind of question the next lesson's interpretability tools are built to help answer: *why* does a rank-one update to this one matrix change this one fact, and what does that tell us about how the fact was represented in the first place?

## Video Script Outline

1. Motivation — many people fine-tune the same base model differently; can we combine or edit the results without retraining?
2. Task vectors: `tau = theta_finetuned - theta_base`, and why they can be added and subtracted like ordinary vectors
3. Task arithmetic in action — merging two fine-tunes, and the honest caveat that the scaling coefficient `lambda` matters
4. The LERP norm-shrinkage problem, shown geometrically (averaging two vectors of similar length but different direction is shorter than either)
5. SLERP's fix: interpolating along the arc, with the exact formula, and the measured norm-preservation result from `example.py`
6. ROME: causal tracing to locate a fact in a specific MLP layer, then the rank-one closed-form update that overwrites it
7. Honest limitations across all three techniques — conflicting task vectors, per-layer merge decisions, edit ripple effects
8. Recap of the whole phase, and handoff to the final lesson: interpretability, or "how do we know what's really in these weights in the first place?"

## Further Reading

- Ilharco et al. (2022), *Editing Models with Task Arithmetic*
- Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME)
- Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging Models*
- Wortsman et al. (2022), *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time*
