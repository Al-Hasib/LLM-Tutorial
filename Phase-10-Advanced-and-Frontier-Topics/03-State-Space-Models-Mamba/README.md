# State Space Models (Mamba)

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `03-State-Space-Models-Mamba`

## Why this matters

By this point in the course you've seen both ends of the sequence-modeling trade-off up close. [Phase 01 Lesson 3: RNNs, LSTMs and GRUs](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md) gave you a model that costs `O(T)` per sequence but pays for it with a strictly sequential recurrence (you cannot compute `h_t` before `h_{t-1}`) and, even with LSTM/GRU gating, gradients and information still degrade over long distances. [Phase 01 Lesson 5: Introduction to Transformers](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md) gave you the opposite trade: self-attention has a path length of exactly 1 between any two tokens (no vanishing-signal-over-distance problem at all) and is fully parallel across the sequence during training, but it costs `O(T^2)` in both compute and memory, which is precisely why long-context inference is expensive.

State Space Models — and Mamba in particular — are a serious attempt to ask: **can we get an RNN's linear-time inference and a Transformer's parallel trainability, in the same model?** The surprising mathematical answer, discovered in the S4 line of work and sharpened by Mamba, is "mostly yes." This lesson builds that answer up from first principles: a classical linear state-space model, the convolution trick that makes it parallelizable, and the "selectivity" idea that turns it from a fixed linear filter into something that can actually reason about content — much like the gates you already met in LSTMs. From here the course moves to [Model Merging and Editing](../04-Model-Merging-and-Editing/README.md); the previous lesson was [Mixture of Experts, Advanced](../02-Mixture-of-Experts-Advanced/README.md), which attacked the *parameter-count-vs-compute* trade-off rather than the *sequence-length* trade-off tackled here — the two lines of research are complementary, and production models increasingly combine both.

## What this lesson covers

- Restating the trade-off: attention is `O(T^2)` but parallel, RNNs are `O(T)` but sequential and gradient-limited — what "best of both worlds" would even mean
- Classical linear state-space models: the same mathematical object as a Kalman filter, with fixed `A`, `B`, `C` matrices
- The key duality that makes S4-style models work: a linear recurrence can be computed as one global convolution against a fixed kernel — parallel to train, sequential to run
- Why that duality alone isn't enough: fixed matrices can't do content-based reasoning
- Mamba's selective state spaces: making `A`, `B`, `C` functions of the current input
- Why selectivity breaks the simple convolution trick, and how the hardware-aware parallel scan gets the parallelism back
- Where this leaves the landscape of sequence-mixing primitives

## 1. What would "best of both worlds" even look like?

Lay the two extremes side by side:

```
Self-attention:  O(T^2) compute/memory,  fully parallel across positions,  path length 1 (no decay with distance)
Vanilla RNN:     O(T)   compute/memory,  strictly sequential,               path length T (vanishing gradients)
```

A model that trained like attention (one shot, parallel across the whole sequence, GPU-friendly) but ran like an RNN (constant-size state, `O(1)` work per new token, `O(T)` total) would be enormously appealing for long sequences — the memory and compute of attention become the bottleneck precisely when `T` is large (long documents, audio, DNA, agent trajectories), which is exactly where you'd most want the RNN-style cost curve. State Space Models are built around one specific mathematical fact that makes this combination possible for a *linear* recurrence — something no nonlinear RNN (vanilla RNN, LSTM, GRU) can exploit, because their recurrences involve a nonlinearity (`tanh`, gates) applied at every step, which has no equivalent closed-form convolution.

## 2. Classical linear state space models

Strip a recurrent network down to its most basic linear form and you get a **state space model (SSM)** — the exact same mathematical object control theorists and statisticians have used for decades to describe systems like a Kalman filter tracking a moving object:

```
h_t = A h_{t-1} + B x_t      # state update
y_t = C h_t                  # output/readout
```

- `h_t` is a hidden state vector of fixed size `N` (the "state dimension") — the SSM's memory.
- `x_t` is the input at time `t`, `y_t` is the output at time `t`.
- `A` (`N x N`), `B` (`N x input_dim`), `C` (`output_dim x N`) are **fixed** matrices — the same three matrices at every timestep, and critically, in this classical form, they do **not** depend on the input at all. They're learned once (or hand-designed, as in classical signal processing) and then frozen for every input sequence the model ever sees.

This looks almost identical to the vanilla RNN's `h_t = tanh(W_xh x_t + W_hh h_{t-1})` from [Phase 01 Lesson 3](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#1-the-vanilla-rnn) — the difference that matters enormously is the **missing nonlinearity**. Because the recurrence is purely linear, we can unroll it symbolically instead of just numerically.

*(Real S4 models are actually defined as continuous-time differential equations, `h'(t) = A h(t) + B x(t)`, then converted to the discrete recurrence above via a "discretization" step — e.g. zero-order hold, which turns a continuous `A` into a discrete `Ā = exp(ΔA)` for some step size `Δ`. That detail matters for how S4 initializes `A` well (the HiPPO matrix) but isn't needed to see the trick in the next section, so we work directly with the discrete `A`, `B`, `C` form.)*

## 3. The core trick: a linear recurrence is also a global convolution

Unroll the recurrence by hand, starting from `h_0 = 0`:

```
h_1 = B x_1
h_2 = A B x_1 + B x_2
h_3 = A^2 B x_1 + A B x_2 + B x_3
...
h_t = sum_{k=1}^{t} A^{t-k} B x_k
```

Apply the readout `y_t = C h_t`:

```
y_t = sum_{k=1}^{t} (C A^{t-k} B) x_k
```

Look closely at that coefficient `C A^{t-k} B`: it depends **only on the lag `t - k`**, never on `t` and `k` individually. Define a kernel using exactly those lag-indexed coefficients:

```
K = ( C B,  C A B,  C A^2 B,  ...,  C A^{T-1} B )        # one scalar/vector per lag i = 0 .. T-1
```

Then the whole output sequence is a single **causal convolution** of the input against this fixed kernel:

```
y_t = sum_{k=1}^{t} K_{t-k} x_k             i.e.        y = x * K
```

This is the entire trick behind S4 (Gu, Goel, Re 2021). It says the *exact same numbers* (`y_1 ... y_T`) can be produced two completely different ways:

1. **Sequentially**, one step at a time: `h_t = A h_{t-1} + B x_t`, `O(T)` total work, but each step waits on the last.
2. **In parallel**, as one convolution: build the kernel `K` once (`T` matrix powers of `A`), then convolve the entire input against it in one shot — no step needs to wait for any other step, so it can be trained the way attention is trained, in parallel across the whole sequence, and even accelerated with FFT-based convolution.

`example.py` builds a small fixed-`A/B/C` system and computes its output *both* ways, then checks the two answers agree to floating-point precision — that agreement is the whole duality made concrete.

This really is "best of both worlds" in a way neither attention nor a vanilla RNN can claim: train via the convolution form (parallel, like attention, though at `O(T log T)` with FFT or `O(T^2)` naively rather than attention's `O(T^2)` with a much larger per-pair cost), then deploy via the recurrence form (`O(1)` memory and compute per new token, exactly like an RNN, with no `O(T^2)` attention cache to carry around). Attention is parallel but quadratic; a vanilla RNN is linear-time but sequential-only and gradient-limited; a linear SSM in this form is linear-time **and** parallelizable, because linearity is what lets the recurrence be re-expressed as a convolution in closed form.

## 4. The catch: fixed matrices can't be selective

A fixed convolution kernel is, structurally, a **linear time-invariant (LTI) filter** — the same filter shape gets applied no matter what the input actually contains. That's a real limitation: it means the model cannot decide, based on *what token it just saw*, to suddenly remember something important or discard something irrelevant. Compare this to an LSTM's forget gate `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` from [Phase 01 Lesson 3 §4](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#4-lstms-gates-and-a-protected-cell-state): it explicitly computes, from the current input, how much of the past to keep. A plain S4-style SSM has no equivalent — `A`, `B`, `C` are the same regardless of content, so it can't implement "copy this token verbatim," "ignore filler tokens," or "reset memory at a delimiter" the way a gated RNN can. In practice this made vanilla S4 excellent at raw long-range signal propagation (its claim to fame was the Long Range Arena benchmark) but noticeably weaker than attention on tasks that need content-based reasoning, like language modeling.

## 5. Mamba's key idea: make the SSM selective

Mamba (Gu & Dao, 2023) keeps the same linear-recurrence skeleton but makes `A`, `B`, and `C` — or in Mamba's specific parameterization, the discretization step size `Δ` together with `B` and `C` — **functions of the current input token**, computed via small learned linear projections:

```
B_t = Linear_B(x_t)          # input-dependent, recomputed at every timestep
C_t = Linear_C(x_t)          # input-dependent, recomputed at every timestep
Δ_t = softplus(Linear_Δ(x_t))  # input-dependent step size -> controls the effective A_t = exp(Δ_t A)

h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

This is exactly the same move that made LSTMs work: instead of a fixed multiplicative factor applied every step (vanilla RNN's `W_hh`, or a plain SSM's fixed `A`), the "how much to keep vs. overwrite" decision is now **computed from the current input**, so the model can choose, per token, what to remember or forget — a marker token can trigger "hold this in state for a long time" while a filler token triggers "let this fade quickly." A large `Δ_t` behaves like an LSTM's forget gate being close to 0 (aggressively overwrite state with new content); a small `Δ_t` behaves like a forget gate close to 1 (barely update the state, effectively skipping the token). `example.py` builds a small selective SSM using exactly this "gate computed from the input, tuned toward retaining by default" idea — the same trick as the LSTM's forget-bias initialization from Phase 01 — and shows it preserves an early "marker" input's influence on the final state far better than a fixed-decay SSM does, as sequence length grows.

## 6. The cost of selectivity, and the hardware-aware parallel scan

Selectivity isn't free. The moment `B_t`, `C_t` (and the effective `A_t`) depend on `t`, the coefficient multiplying `x_k` in the output sum is `C_t A_{k+1} A_{k+2} ... A_t B_k` — it depends on the absolute positions `t` and `k`, not just the lag `t - k`. That breaks the shift-invariance the Section 3 kernel relied on: there is no longer a single fixed kernel `K` you can convolve the whole sequence against, because the "filter" is now different at every position. Mamba loses the simple global-convolution training trick in exchange for content-based selection.

What Mamba adds back is a **hardware-aware parallel scan**. The recurrence `h_t = A_t h_{t-1} + B_t x_t` is still what's called an *associative* operation — the composition of two consecutive linear updates is itself a linear update of the same form, which is exactly the algebraic property that lets you compute a running sum or product in `O(log T)` parallel depth instead of `O(T)` sequential steps (a "parallel scan," the same primitive behind parallel prefix-sum). Mamba implements this scan as a custom GPU kernel that keeps the (small) per-step states in fast on-chip SRAM rather than repeatedly writing them out to slower GPU memory (HBM) — a systems-level optimization, not a change to the math. The result: Mamba trains with logarithmic-depth parallelism (not the single-matmul parallelism of an LTI convolution or of attention, but far better than a naive `O(T)` sequential loop) while still being able to run as an genuine `O(1)`-per-step, constant-memory recurrence at inference time — no KV cache that grows with sequence length, unlike attention.

## 7. Where this leaves the landscape

```
                      train-time parallelism        inference cost         content-based selection
Self-attention        full (one matmul)              O(T) per new token,    yes (softmax over all past
                                                       growing KV cache        keys/values, every step)
Vanilla RNN/LSTM/GRU   none (strictly sequential)     O(1) per new token,    yes (LSTM/GRU gates)
                                                       constant memory
Linear SSM (S4)        full (global convolution)      O(1) per new token,    no (fixed A, B, C)
                                                       constant memory
Selective SSM (Mamba)  parallel scan, O(log T) depth  O(1) per new token,    yes (input-dependent A, B, C)
                                                       constant memory
```

No option dominates on every axis — this is a genuine design space, not a solved problem. Mamba and its descendants (Mamba-2, hybrid Mamba/attention architectures like Jamba) are an active area of research precisely because getting selectivity, hardware efficiency, and quality all at once is hard, and different downstream tasks stress these trade-offs differently.

## Video Script Outline

1. Recap the trade-off: attention is `O(T^2)` but parallel, RNNs are `O(T)` but sequential and gradient-limited
2. Introduce the classical linear SSM (`h_t = Ah_{t-1} + Bx_t`, `y_t = Ch_t`) and note the Kalman-filter family resemblance
3. Derive the recurrence-equals-convolution duality by hand, unrolling `h_t` and spotting the lag-only dependence
4. Walk through `example.py` Part 1: sequential vs. convolution outputs matching to floating-point precision
5. Explain the catch: fixed `A/B/C` can't do content-based selection, unlike an LSTM's gates
6. Introduce Mamba's selectivity: `B`, `C`, `Δ` as functions of the input; connect explicitly to the LSTM forget gate
7. Explain why selectivity breaks the convolution trick, and how the hardware-aware parallel scan restores efficient training
8. Walk through `example.py` Part 2: selective vs. fixed SSM's gradient retention as sequence length grows; recap and preview Model Merging and Editing

## Further Reading

- Gu, Goel, Re (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4)
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2)
