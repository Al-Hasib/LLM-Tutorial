# Interpretability and Mechanistic Interpretability

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `05-Interpretability-and-Mechanistic-Interpretability`

## Why this matters

[Lesson 4: Model Merging and Editing](../04-Model-Merging-and-Editing/README.md) treated a trained model as a black box you can splice, average, or patch based on its external behavior. This lesson asks the harder question underneath all of that: **what is actually happening inside the network to produce that behavior in the first place?** Every prior lesson in this course has described what a Transformer computes — [scaled dot-product attention](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md), residual streams, feed-forward blocks — but almost nothing so far has asked *why* a specific trained set of weights produces a specific behavior, or *where* inside billions of parameters a given fact, skill, or bias actually lives. That is the subject of interpretability, and its most ambitious branch, **mechanistic interpretability**, which tries to reverse-engineer neural networks the way you'd reverse-engineer a compiled binary: not just observing that it works, but recovering the actual algorithm implemented in its weights.

This isn't just intellectual curiosity. [Phase 06 Lesson 1: The Alignment Problem](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md) established that we train models by shaping their *external* behavior (next-token prediction, then preference-based fine-tuning) without ever directly specifying or verifying their *internal* computation. A model can look aligned on every prompt you think to test and still be pursuing something different internally — deceptive behavior, a spurious correlation standing in for the real concept, a capability that only activates in distribution shift. Interpretability is the closest thing the field has to actually checking, rather than merely behaviorally testing, what a model is doing. It sits at the foundation of a serious long-term case for trusting increasingly capable systems.

And this is, fittingly, the last lesson of the entire course. Every phase before this one has been about building, training, aligning, prompting, evaluating, and deploying LLMs; this one is about the field's attempt to actually *understand* the artifact all of that produces — an appropriate place to end, because it is one of the most active, unsolved frontiers in the field today.

## What this lesson covers

- Probing classifiers: training a simple classifier on frozen internal activations to test whether a concept is linearly decodable at a given layer — and the honest limits of what that proves
- Attention pattern analysis as a direct, inspectable interpretability signal, and induction heads as a real, well-documented mechanistic circuit
- Sparse autoencoders (SAEs): decomposing superimposed, polysemantic activations into a larger set of sparse, more monosemantic features
- A hands-on demonstration of both a probing-classifier experiment and a toy sparse autoencoder trained on synthetic superimposed features
- Looking back across the entire 11-phase curriculum, and looking forward to where the field goes from here

## 1. Why "it works" isn't "we understand it"

A model trained with the recipe from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) — embeddings, attention, feed-forward blocks, next-token cross-entropy — ends up as a large matrix of numbers that nobody wrote by hand. We know the *architecture* precisely (every operation is explicit, differentiable, and fully specified in code), but we do not know, in general, the *algorithm* those particular trained weights implement. This is the central peculiarity of deep learning: the source code is simple and known; the resulting program is opaque and has to be discovered empirically, after the fact, by inspecting a huge number of learned floating-point parameters. Interpretability research is the attempt to close that gap, at various levels of ambition:

- **Behavioral/black-box interpretability** — probe the model only through inputs and outputs (this is most of [Phase 08: Evaluation](../../Phase-08-Evaluation-of-LLMs/README.md)). Useful, but tells you *what* the model does, not *how*.
- **Representational interpretability** — look at the model's internal activations (hidden states) without fully reverse-engineering the computation that produced them. Probing classifiers, discussed next, live here.
- **Mechanistic interpretability** — go further and try to recover the actual sub-computation (a "circuit": a small set of attention heads and/or neurons wired together) responsible for a specific capability, in enough detail that you could, in principle, re-derive it from the weights alone. Induction heads and sparse autoencoders, below, are the two most mature results at this level.

## 2. Probing classifiers

The simplest way to ask "does this layer's representation contain concept X?" is to just try to extract it: freeze the trained model entirely (no gradient updates to it), run inputs through it, pull out the hidden activation vector at some chosen layer, and train a small **new** classifier — usually a single linear layer — to predict a human-labeled concept from that activation alone:

```
frozen_model(input) -> hidden activations h  (no gradients flow into the frozen model)
probe = Linear(d_model -> num_classes)
loss  = CrossEntropy( probe(h.detach()), concept_label )
```

Only `probe`'s weights are trained. If a *linear* probe reaches high accuracy, that is evidence that the concept is represented as a **linearly decodable direction** in that layer's activation space — a much stronger and more specific claim than "some nonlinear function of the activations could in principle recover it" (a sufficiently powerful nonlinear probe can often decode almost anything from almost any layer, which is why the field converged on linear probes specifically: linear separability is a meaningful, non-trivial property of a representation, not just a statement about probe capacity).

**The important caveat — correlation is not causation, even here.** A high probing accuracy tells you the concept is *present and linearly readable* in that layer's activations. It does **not** by itself tell you the model's *own downstream computation actually uses* that information. It's entirely possible for a model to compute a feature as an incidental byproduct of computing something else, have that feature sit there linearly decodably, and never route it anywhere that affects the model's output — a probe would still find it with high accuracy. This is a widely-cited failure mode of naive probing (Belinkov, 2022; Hewitt & Liang, 2019, on probes learning to memorize rather than reveal): a good probe proves *representation*, not *causal use*. Firmer causal claims require interventions — e.g. activation patching (zeroing out or swapping the presumed feature and checking whether the model's output actually changes) — which is a step beyond correlational probing and part of why mechanistic interpretability treats circuit-level causal experiments as the gold standard, with probing as a useful but weaker first-pass signal.

`example.py` runs a real version of this experiment: a small model is trained on one task, and a linear probe is then trained on its frozen internal activations to recover a *different*, related concept the model was never directly told to predict — compared honestly against a probe trained on the raw input and a probe trained on an untrained network's activations.

## 3. Attention patterns as a direct interpretability signal

Unlike most of a network's internals, attention weights are unusually interpretable almost "for free," because [Lesson 2 of Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) already gave them an exact meaning: `weights = softmax(QKᵀ / √d_k)` is, row by row, a probability distribution over "which earlier positions does this position draw information from." You can literally plot that `(T, T)` matrix as a heatmap and read off, for a specific trained head, which tokens it attends to for a specific input — no reverse-engineering required, because the object itself is already a distribution over positions by construction.

Doing this systematically across many trained models and inputs, researchers have found that individual heads often specialize into recognizable, reusable patterns. The best-documented example is the **induction head** (Olsson et al., 2022): a small circuit, typically built from *two* attention heads in different layers working together, that implements a simple but powerful algorithm:

```
Given a sequence: ... [A][B] ... [A] <- current position
Induction head predicts: [B]   (i.e., "last time I saw A, it was followed by B; predict B again")
```

Mechanically, a "previous-token head" in an earlier layer first copies information about the token *before* each position into that position's residual stream; an "induction head" in a later layer then looks for the *current* token's past occurrences and attends to the position immediately after them, copying forward whatever followed last time. The net effect is pattern completion: if `"Doctor Smith ... Doctor"` appeared once, the model predicts `"Smith"` again the next time `"Doctor"` shows up — even for names or made-up tokens it was never trained on, because the circuit implements a general copying algorithm rather than memorizing specific words. This is a large part of the mechanistic explanation for **in-context learning**: a good fraction of a Transformer's ability to pick up a pattern from earlier in its own context window and repeat it later is executed, quite literally, by this one circuit, and induction heads have been found to emerge at a fairly predictable point in training across many different model architectures and sizes — one of mechanistic interpretability's cleanest, most reproducible findings to date.

## 4. Superposition and sparse autoencoders

Probing and attention analysis both run into the same wall as models get larger: **superposition**. A model has far more distinct concepts it might want to represent than it has neurons or residual-stream dimensions to represent them in "one concept per direction" fashion. Toy-model work (Elhage et al., 2022, *Toy Models of Superposition*) shows that when features are sparse (each one is only relevant on a small fraction of inputs) and the network is under-complete (fewer dimensions than features), the network learns to pack multiple features into overlapping, non-orthogonal directions in the *same* small set of dimensions, tolerating a small amount of interference between them in exchange for representing far more concepts than it has room for individually. The practical consequence: a single neuron, or a single direction in a hidden layer, often does **not** correspond to one clean human concept — it's "polysemantic," firing for several unrelated things that happen to share a direction. This is precisely why naive "what does neuron 417 mean" inspection of real models so often produces a confusing, seemingly-unrelated grab-bag of activating examples.

**Sparse autoencoders (SAEs)** are the current leading tool for undoing this packing. The idea (formalized for LLM internals in Anthropic's "Towards Monosemanticity" work, Bricken et al., 2023) is to train a small autoencoder on a model's activations, but with two deliberate twists relative to a standard autoencoder:

```
h = ReLU(W_enc x + b_enc)              # encoder: activation -> sparse code (h is OVERCOMPLETE: dim(h) >> dim(x))
x_hat = W_dec h + b_dec                # decoder: sparse code -> reconstructed activation

L = || x - x_hat ||^2  +  lambda * || h ||_1
    \_____reconstruction_____/    \___sparsity penalty___/
```

- **Overcomplete, not undercomplete.** A standard autoencoder compresses (hidden dim smaller than input, forcing a bottleneck). An SAE does the opposite: its hidden dimension is deliberately made *larger* than the input activation dimension — giving the model room to unpack `d` superimposed input dimensions into a much larger number of candidate feature directions, more than the raw activation width alone could linearly represent.
- **L1 penalty on the hidden code, not on the weights.** The `lambda * ||h||_1` term directly penalizes how many hidden units are active (non-zero) for any given input, not how large the weights are. This pushes the model toward representations where, for any single input, only a small handful of the (many) available hidden units fire — which is exactly the sparse-and-large-dictionary picture that superposition theory predicts underlies the original activations.

If this works, each of the resulting sparse hidden units tends to be far more **monosemantic**: activating for one relatively coherent, often human-interpretable concept, rather than the tangled mix a raw neuron exhibits. This doesn't eliminate the correlation-vs-causation caveat from Section 2 — an SAE feature is still a *representational* discovery, and showing it's causally load-bearing still requires an intervention (e.g., clamping the feature and checking whether the model's behavior changes in the predicted way) — but it gives interpretability researchers a much larger, cleaner set of candidate concepts to run those causal experiments on than raw neurons ever did.

`example.py` builds a small, honest version of this: synthetic activation vectors constructed as sparse combinations of a known, larger set of "true" underlying features packed (superimposed) into fewer raw dimensions, an SAE trained on them at several L1 strengths, and a report of the real reconstruction-vs-sparsity trade-off this produces — including which units end up "dead" (never activate) at each strength.

## Looking Back, Looking Forward

This is the last lesson of an 11-phase, 59-topic course, so it's worth walking back through the whole arc once, in order:

- **[Phase 00: Prerequisites](../../Phase-00-Prerequisites/README.md)** built the math, neural network, NLP, and PyTorch foundations everything after it assumes.
- **[Phase 01: Language Modeling Foundations](../../Phase-01-Language-Modeling-Foundations/README.md)** defined what a language model even is, introduced word embeddings, walked through RNNs/LSTMs/GRUs and *why* their sequential recurrence became the bottleneck, and previewed attention as the fix.
- **[Phase 02: Transformer Architecture Deep Dive](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md)** made that preview rigorous: tokenization, scaled dot-product and multi-head self-attention, positional encoding, the full encoder-decoder architecture, and residuals/LayerNorm/FFN — assembled into a real, trainable mini-GPT.
- **[Phase 03: LLM Architectures and Types](../../Phase-03-LLM-Architectures-and-Types/README.md)** took that one architecture and showed its family tree: decoder-only (GPT), encoder-only (BERT), encoder-decoder (T5/BART), Mixture of Experts, scaling laws, and long-context techniques.
- **[Phase 04: Pretraining LLMs](../../Phase-04-Pretraining-LLMs/README.md)** covered how a model that size actually gets trained on that much data: the data pipeline, pretraining objectives, distributed training, and mixed-precision optimization.
- **[Phase 05: Fine-tuning LLMs](../../Phase-05-Finetuning-LLMs/README.md)** turned a raw pretrained base model into something task-adaptable and instructable: full fine-tuning vs. PEFT, LoRA/QLoRA, prompt/prefix tuning and adapters, and instruction tuning (SFT).
- **[Phase 06: Alignment and RLHF](../../Phase-06-Alignment-and-RLHF/README.md)** closed the gap SFT leaves open: the alignment problem itself, reward modeling, RLHF with PPO, DPO, RLAIF/Constitutional AI, and safety/bias/toxicity mitigation.
- **[Phase 07: Prompt Engineering and In-Context Learning](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/README.md)** — with this lesson's induction-head circuit as one of the actual mechanisms underneath it — covered how to get the most out of a trained model purely through its input: zero/few-shot prompting, chain-of-thought, tree-of-thought/ReAct, automatic prompt optimization, and structured output/function calling.
- **[Phase 08: Evaluation of LLMs](../../Phase-08-Evaluation-of-LLMs/README.md)** asked how you'd actually know if any of the above worked: metrics, standard benchmarks, LLM-as-a-judge, hallucination/factuality evaluation, and human evaluation methodologies.
- **[Phase 09: Deployment and Inference Optimization](../../Phase-09-Deployment-and-Inference-Optimization/README.md)** covered getting a trained model to run efficiently in the real world: quantization, KV-cache and speculative decoding, serving frameworks, distillation/pruning, and cost/latency optimization.
- **[Phase 10: Advanced and Frontier Topics](../README.md)** (this phase) closed with what's actively being researched right now: multimodal LLMs, advanced Mixture of Experts, state space models (Mamba), model merging and editing, and — this lesson — interpretability.

Where does the field go from here? Honestly: nobody fully knows, and that's the point of calling this a frontier rather than a settled subject. A few of the open threads that this lesson only scratches the surface of:

- **Interpretability at real scale.** Everything in this lesson is demonstrated on toy models with a handful of dimensions. Scaling probing, circuit analysis, and SAEs to models with tens of billions of parameters and correspondingly enormous numbers of candidate features is an active, expensive, and only partially solved engineering and science problem — current frontier SAE work extracts millions of features from production-scale models, and still likely captures only a fraction of what's really going on.
- **More capable and efficient architectures.** [Phase 10 Lesson 3](../03-State-Space-Models-Mamba/README.md)'s state space models and [Lesson 2](../02-Mixture-of-Experts-Advanced/README.md)'s sparse MoE routing are two of several ongoing attempts to beat the Transformer's compute/memory scaling without giving up its quality — this is nowhere near a finished conversation.
- **Aligning increasingly capable systems.** [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md)'s techniques were largely developed on models far less capable than current frontier systems; whether preference-based fine-tuning alone remains sufficient as capabilities keep increasing — or whether it needs to be paired with the kind of internal, mechanistic verification this lesson describes — is an open and actively debated question, not a solved one.
- **Multimodality and beyond next-token text.** [Phase 10 Lesson 1](../01-Multimodal-LLMs/README.md) is an early chapter in models that reason over images, audio, and other modalities natively rather than as an add-on; how much of what this course covered (attention, scaling laws, alignment techniques, interpretability tools) transfers cleanly and how much needs to be substantially rethought is still being worked out in public, in real time.

That open-endedness is, honestly, the appropriate note to end an LLM course on in 2026: not "here is the finished picture," but "here is a real, working understanding of how we got here, sturdy enough to keep reading the field's next result and actually follow it."

## Video Script Outline

1. Motivation — from "splicing model weights" (Lesson 4) to "understanding what the weights compute" (this lesson), and why that matters for alignment
2. The three levels of interpretability: behavioral, representational, mechanistic
3. Probing classifiers — the setup, what high accuracy does and does not prove, the causal-use caveat
4. Attention patterns and induction heads — a real, well-documented circuit behind in-context learning
5. Superposition — why one neuron rarely means one concept
6. Sparse autoencoders — the overcomplete-plus-L1 trick, and Anthropic's monosemanticity results
7. Walkthrough of `example.py` — the probing experiment's real numbers, then the SAE's real reconstruction/sparsity trade-off across L1 strengths
8. Send-off: recap the entire 11-phase journey from prerequisites to frontier research, and where the field goes from here

## Further Reading

- Alain & Bengio (2016), *Understanding Intermediate Layers Using Linear Classifier Probes*
- Olsson et al. (2022), *In-context Learning and Induction Heads* (Anthropic)
- Elhage et al. (2022), *Toy Models of Superposition* (Anthropic) — the theoretical picture SAEs are built to undo
- Bricken et al. (2023), *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (Anthropic)
