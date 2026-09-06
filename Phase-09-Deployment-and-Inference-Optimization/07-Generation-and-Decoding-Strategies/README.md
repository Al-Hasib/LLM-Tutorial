# Generation and Decoding Strategies

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `07-Generation-and-Decoding-Strategies`

## Why this matters

Every lesson so far in this phase has quietly skipped over one step. [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) spends an entire lesson on caching Key/Value tensors so decode is cheap, and its speculative-decoding rejection-sampling rule (["Why this is exact, and why it's faster"](../03-KV-Cache-and-Speculative-Decoding/README.md#6-why-this-is-exact-and-why-its-faster)) is built on the assumption that the target model produces a real *sampling distribution* at each step, not a single deterministic token — but the lesson never says what that distribution is, how it's shaped, or how a token actually gets pulled out of it. [Lesson 4](../04-Serving-Frameworks/README.md) schedules and batches thousands of concurrent generation requests without ever asking what decision ends each one. [Phase 02 Lesson 6's mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#4-autoregressive-generation) — the model every one of these lessons is built on top of — already computes `softmax(logits / temperature)` and calls `torch.multinomial` once, at a fixed temperature of `0.8`, and moves on without ever asking *why* that value, what the alternatives are, or what happens at the boundary cases (`temperature -> 0`, or no temperature scaling at all). "Sample a token from the logits" has been treated as a solved, one-line implementation detail everywhere it appears. It isn't one line — it's a whole design space, with real, measurable failure modes on either extreme (deterministic degenerate loops at one end, incoherent noise at the other), and this lesson is the missing piece: the actual decision rule that turns a vector of logits into the next token, in every generation loop this course has used so far.

## What this lesson covers

- From logits to a probability distribution: the softmax recap, and what "sampling" actually means
- Greedy decoding: always take the argmax, and its real failure mode (repetitive loops)
- Temperature: rescaling logits before softmax, and how `T -> 0` formally recovers greedy decoding
- Top-k sampling: truncating to the k most likely tokens before sampling
- Top-p (nucleus) sampling: truncating to a cumulative-probability mass instead of a fixed count
- How real pipelines compose temperature, top-k, top-p, and sampling, and why the order matters
- Repetition penalty and frequency/presence penalties: patching greedy's degenerate-loop failure mode directly
- Beam search: a deterministic *search* over cumulative log-probability, not sampling at all, and why it's a poor fit for open-ended chat
- Stopping criteria: EOS, max-new-tokens, and stop strings, and how a finished sequence's KV-cache slot feeds back into serving

## 1. Recap: from logits to a probability distribution

Every model in this course ends a forward pass the same way: a final linear layer projects the last hidden state into a vector of raw scores, one per vocabulary entry, called **logits** ([Phase 02 Lesson 6 §2](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#2-the-full-model)). Logits are unnormalized — they can be any real number, positive or negative, and they don't sum to anything meaningful on their own. **Softmax** turns them into an actual probability distribution over the vocabulary:

```
P(token_i) = exp(logit_i) / sum_j( exp(logit_j) )
```

Every value in `P` is positive and the whole vector sums to `1` — this is a genuine probability distribution over the vocabulary, and it's what `example.py` prints entropy and top-token-probability numbers over throughout. Everything in this lesson — greedy, temperature, top-k, top-p, beam search — is a different rule for turning this one distribution into an actual next token. There is no other input: the model only ever hands the rest of the generation pipeline this one vector of numbers, at every single step.

## 2. Greedy decoding: always take the argmax

The simplest possible rule: at every step, take the single highest-probability token and move on.

```
next_token = argmax(P)     # equivalently, argmax(logits) -- softmax is monotonic, doesn't change WHICH is largest
```

Greedy decoding is fully **deterministic** — the same model and the same prompt always produce the exact same output, with no randomness anywhere in the loop. That determinism is exactly what makes it a convenient baseline (it's what [Lesson 3's `example.py`](../03-KV-Cache-and-Speculative-Decoding/example.py) uses to check that a naive and a KV-cached generation loop produce byte-for-byte identical tokens — a random sampling rule would have made that comparison meaningless run-to-run). But for open-ended generation, greedy decoding has a well-known, real failure mode: it gets stuck in **repetitive loops**. If the model ever assigns the highest probability to a token that leads back into a state it has already been in — a common outcome once a phrase has appeared once and the model's own attention now finds that phrase highly predictable — greedy decoding has no mechanism to escape: the same input always produces the same "most likely" next token, so the loop repeats forever, or until a max-length cutoff (§9) forces it to stop. `example.py`'s third demo produces and verifies exactly this loop on a real trained model, then shows a repetition penalty (§7) breaking it.

## 3. Temperature: rescaling the distribution before softmax

**Temperature** rescales the logits by `1/T` before applying softmax:

```
P_T(token_i) = exp(logit_i / T) / sum_j( exp(logit_j / T) )
```

Because softmax is exponential, dividing by a temperature `T < 1` *amplifies* the gaps between logits before exponentiating, pushing the distribution's mass further onto its already-highest-probability tokens — the distribution gets **sharper**, more confident, closer to a single spike. A temperature `T > 1` does the opposite: it *shrinks* the gaps between logits, pulling probability mass away from the top candidates and toward everything else — the distribution gets **flatter**, closer to uniform over the vocabulary. This is the exact knob [Phase 02 Lesson 6's `generate` method](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/example.py) applies (at a fixed `T=0.8`) without ever explaining it.

The two limits are worth stating precisely, because one of them ties directly back to §2:

```
T -> 0+ :  P_T concentrates ALL probability mass on argmax(logits)  -->  sampling from P_T becomes
           IDENTICAL to greedy decoding (§2) -- greedy is temperature-sampling's zero-temperature limit.
T -> infinity :  P_T -> uniform distribution over the entire vocabulary  -->  every token equally likely,
           regardless of what the model actually predicted.
```

`example.py`'s first demo computes this directly on real trained-model logits: the same logit vector, softmax'd at a low temperature, a temperature of `1`, and a high temperature, with entropy and top-token probability printed at each — a small, measured number showing the sharpening/flattening effect, not just the prose above.

## 4. Top-k sampling: truncate to a fixed count

Temperature (§3) reshapes the *whole* distribution but never removes any token from consideration — even at a very low temperature, a token with a tiny probability still has some nonzero chance of being sampled, and over a long enough generation, rare accidents compound. **Top-k sampling** (Fan, Lewis & Dauphin, 2018) fixes this by discarding the long tail outright:

```
1. Sort tokens by probability, descending.
2. Keep only the top k tokens; set every other token's probability to 0.
3. Renormalize the kept probabilities so they sum to 1 again.
4. Sample from this truncated, renormalized distribution.
```

A small `k` (say, `k=1`) makes top-k sampling identical to greedy decoding (§2) — there's only one token left to sample. A large `k` (at or beyond the vocabulary size) makes it identical to plain temperature sampling with no truncation at all (§3) — nothing gets discarded. Between those extremes, top-k guarantees the model can never sample one of its very-lowest-probability tokens, no matter how flat the distribution briefly gets at a given step, while still leaving room for genuine variety among the tokens that remain. `example.py`'s second demo implements this truncate-renormalize-sample rule as plain tensor operations (no library call that hides the mechanism) and measures the resulting output diversity directly.

## 5. Top-p (nucleus) sampling: truncate to a probability mass

Top-k's fixed cutoff has a real weakness: the *right* number of tokens to keep isn't constant — it depends on how peaked or flat the model's distribution happens to be at that particular step. When the model is very confident (one token dominates), even a small fixed `k` might needlessly include several near-zero-probability tokens; when the model is genuinely uncertain among many plausible continuations, the same fixed `k` might cut off real, reasonable candidates that happen to rank just below the cutoff. **Top-p (nucleus) sampling** (Holtzman, Buys, Du, Forbes & Choi, 2020) replaces the fixed count with a fixed cumulative-probability threshold:

```
1. Sort tokens by probability, descending.
2. Walk down the sorted list, accumulating probability mass, until the running
   total first reaches or exceeds p (e.g. p = 0.9).
3. Keep exactly that prefix of tokens (the "nucleus"); set every other token's
   probability to 0; renormalize the kept probabilities so they sum to 1.
4. Sample from this truncated, renormalized distribution.
```

The key contrast with top-k (§4): the *size* of the kept set is no longer fixed — it adapts, automatically, to the shape of the distribution at each individual step. When the model is sharply confident, the nucleus might contain only one or two tokens (small effective `k`, reached fast); when the model is genuinely spread across many plausible next tokens, the nucleus can widen to dozens of tokens before crossing the `p` threshold. Top-k cannot do this: a fixed `k` is either too large for confident steps or too small for uncertain ones, because it has no way to see how the probability mass is actually distributed at that step. Holtzman et al.'s paper is also the source of the core empirical finding this whole lesson leans on for §8 below: pure greedy/beam-style maximization of likelihood produces text that is measurably duller and more repetitive than human text, which is precisely why a *sampling*-based rule like top-p, not a maximization rule, is the standard for open-ended generation. `example.py`'s second demo implements top-p exactly as the three steps above, as plain tensor operations, alongside top-k, on the same logits.

## 6. Composing strategies: temperature, then top-k, then top-p, then sample

Real generation pipelines (Hugging Face `transformers`, vLLM, and most commercial chat APIs' `temperature`/`top_p`/`top_k` parameters) rarely apply just one of §3-5 in isolation — they compose all three, applied in a specific order:

```
1. logits <- logits / T                          # temperature: reshape the whole distribution first
2. logits <- top_k_filter(logits, k)              # truncate to the k highest logits, mask the rest to -inf
3. probs  <- top_p_filter(softmax(logits), p)     # softmax, then truncate to the smallest p-mass prefix
4. next_token <- sample(probs)                    # finally, sample one token from what's left
```

The order matters because each stage operates on the *output* of the previous one. Applying top-k before temperature would truncate based on the *un*-reshaped distribution — a large `k` might keep tokens that a low temperature would have made negligibly unlikely anyway, or a low `k` might discard tokens that a high temperature was specifically trying to keep in play. Running top-p after top-k (rather than instead of it) is deliberate too: top-k first bounds the worst case (never more than `k` candidates survive, capping how much work top-p's cumulative sum has to do and guaranteeing an upper bound on tail risk), and top-p then adaptively tightens that bounded set further based on the actual shape of what's left. Setting `k` to the full vocabulary size (or `0`, by convention in most libraries, meaning "disabled") and relying on `p` alone, or vice versa, are both common special cases of this same pipeline — not different mechanisms.

## 7. Repetition penalty: patching greedy's degenerate-loop failure directly

§2 showed greedy decoding's concrete failure mode: once a phrase repeats, the model's own attention often makes repeating it *again* the highest-probability move, and a deterministic rule has no way out. Sampling (§3-6) helps by occasionally choosing a non-top token, but doesn't specifically target the mechanism causing the loop — it's a generic fix for a specific problem. A **repetition penalty** (Keskar et al., 2019, `CTRL`) attacks the loop directly, by discouraging the model from re-selecting tokens it has already produced:

```
for token_id in set(already_generated_tokens):
    if logits[token_id] > 0:
        logits[token_id] /= penalty      # penalty > 1: shrink a positive logit toward 0
    else:
        logits[token_id] *= penalty      # penalty > 1: push a negative logit further negative
```

The asymmetric multiply/divide (rather than a flat subtraction) keeps the penalty's effect roughly proportional regardless of a logit's sign or scale. Two related, simpler variants set a fixed threshold instead of a multiplicative one: a **frequency penalty** subtracts an amount proportional to *how many times* a token has already appeared (repeated repeats get penalized harder each time), and a **presence penalty** subtracts a single flat amount for any token that has appeared *at all*, regardless of count (discourages re-use without extra punishment for reusing something a lot). All three are applied to the logits *before* temperature/top-k/top-p (§6), so the rest of the pipeline reshapes and truncates an already-penalized distribution. `example.py`'s third demo runs this exact multiplicative penalty on the same toy model and prompt that gets stuck in a greedy loop in §2, under the same deterministic argmax rule, and verifies in code that the loop breaks.

## 8. Beam search: a search, not a sample

Every strategy so far — greedy, temperature, top-k, top-p — makes one decision per step and never looks back. **Beam search** instead tracks the `B` highest cumulative-log-probability sequences ("beams") simultaneously, at every step:

```
1. Start with B copies of the prompt (or, at step 1, the single prompt expanded to
   its top-B next-token candidates).
2. At each step, for EACH of the B current beams, compute the model's next-token
   distribution and consider extending that beam by every candidate token.
3. Out of all B * vocab_size candidate extensions, keep only the B with the
   highest cumulative log-probability (sum of log P(token) over the whole
   sequence so far) -- discard the rest.
4. Repeat until every beam has produced an end-of-sequence token or hit the
   length budget; return the surviving beam with the highest cumulative
   log-probability.
```

This is important to say precisely: beam search is **not a sampling method** at all — given a fixed model, prompt, and beam width `B`, it always returns the exact same output, deterministically, just like greedy decoding (§2). It is better understood as a *search* over the same objective greedy decoding is (locally) optimizing — the highest-probability sequence under the model — just a wider, less myopic one: greedy is beam search with `B=1`, and a wider beam explores more of the space of high-probability sequences before committing, which is why beam search can find (and `example.py`'s fourth demo verifies, with real numbers) a sequence with cumulative log-probability *at least as high as* greedy's, on the exact same prompt and model.

That guarantee is also exactly why beam search is a poor fit for open-ended chat and instruction-following, despite finding "better" (higher-likelihood) sequences by its own objective: Holtzman et al.'s finding, already cited in §5, is that maximizing sequence likelihood — which is precisely what beam search does, more thoroughly than greedy — is the *wrong* objective for open-ended text. Human-written text is not the highest-likelihood continuation at each step; it's varied, sometimes locally surprising, and beam search's relentless pursuit of the single most probable sequence tends to produce noticeably bland, repetitive, generic text — the same qualitative failure mode as greedy's loops (§2), just harder to detect because it's spread across a whole sequence rather than one obvious repeated phrase. This is precisely the empirical motivation for top-p sampling (§5) as the standard choice for chat and instruction-following models. Beam search remains a good fit, and is still widely used, for tasks whose output space is much narrower and closer to having one or a few genuinely "correct" answers — machine translation and summarization being the classic examples — where a systematic search for the single highest-likelihood output is closer to what the task actually wants than exploring stylistic variety would be.

## 9. Stopping criteria: EOS, length budgets, and stop strings

Every generation loop needs a rule for *when to stop*, independent of which per-step decoding strategy (§2-8) is in use:

- **EOS (end-of-sequence) token**: the model itself was trained to predict a special end-of-sequence token when it considers the response complete; generation stops the moment that token is sampled or selected as the argmax, same as any other token.
- **Max-new-tokens budget**: a hard ceiling on how many tokens a single request may generate, regardless of whether EOS ever appears — the backstop that guarantees a request (and, per §2, a greedy loop that never escapes on its own) eventually terminates.
- **Stop sequences / stop strings**: a caller-supplied list of literal strings (e.g. `"\n\nUser:"` in a chat template) that, once they appear in the generated text, end the response early — useful when a model's output format has an application-level terminator that isn't the model's own EOS token.

This matters for serving, not just for a single request in isolation: the instant any one of these three conditions fires, that sequence's slot in a batch — and the KV cache memory reserved for it ([Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md#3-the-kv-cache-pay-for-each-tokens-kv-exactly-once)'s per-sequence cache) — is no longer doing useful work and must be freed. [Lesson 4's continuous batching](../04-Serving-Frameworks/README.md#4-hugging-face-tgi-continuous-in-flight-batching) is exactly the serving-side mechanism that immediately backfills a freed slot with the next waiting request rather than leaving it idle — the stopping decision made here, per sequence, is the event that triggers the slot-management behavior that whole lesson is about.

## Video Script Outline

1. Motivation — every lesson so far in this phase (and the mini-GPT from Phase 02) has been quietly assuming "pick a token from the logits" is a solved step; today it isn't assumed anymore
2. Softmax recap: logits to a real probability distribution, and what "sampling" means concretely
3. Greedy decoding: always argmax, fully deterministic, and its real repetitive-loop failure mode
4. Temperature: rescaling logits by `1/T`, sharpening vs. flattening, and `T -> 0` as greedy decoding's limit
5. Top-k sampling: truncate to a fixed count, renormalize, sample
6. Top-p (nucleus) sampling: truncate to a cumulative-probability mass instead, and why that adapts per step where top-k can't
7. How real pipelines compose temperature -> top-k -> top-p -> sample, and why the order isn't arbitrary
8. Repetition penalty and frequency/presence penalties: a direct patch for greedy's degenerate loops
9. Beam search: cumulative-log-probability search across B beams, not sampling, and why it's the wrong tool for open-ended chat but the right one for translation/summarization
10. Stopping criteria: EOS, max-new-tokens, stop strings, and how a freed slot feeds continuous batching
11. Walkthrough of `example.py` — measured entropy/temperature numbers, top-k/top-p diversity, a real greedy loop broken by a repetition penalty, and beam search's cumulative log-probability advantage over greedy, all verified in code
12. Recap + pointer to [Lesson 4: Serving Frameworks](../04-Serving-Frameworks/README.md), where a stopped sequence's freed slot is exactly what continuous batching backfills

## Further Reading

- Holtzman, Buys, Du, Forbes & Choi (2020), *The Curious Case of Neural Text Degeneration* (introduces top-p/nucleus sampling and the empirical case against likelihood-maximizing decoding for open-ended text)
- Fan, Lewis & Dauphin (2018), *Hierarchical Neural Story Generation* (introduces top-k sampling)
- Keskar, McCann, Varshney, Xiong & Socher (2019), *CTRL: A Conditional Transformer Language Model for Controllable Generation* (repetition penalty)
- Graves (2012), *Sequence Transduction with Recurrent Neural Networks*, and the many later encoder-decoder MT papers that popularized beam search as the standard decoding method for translation
