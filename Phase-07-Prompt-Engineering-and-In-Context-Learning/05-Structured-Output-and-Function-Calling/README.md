# Structured Output and Function Calling

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `05-Structured-Output-and-Function-Calling`

## Why this matters

Every technique so far in this phase — [few-shot examples](../01-Prompting-Basics-Zero-Few-Shot/README.md), [chain-of-thought](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md), [ReAct](../03-Tree-of-Thought-and-ReAct/README.md), even [automatically searching for a better prompt](../04-Automatic-Prompt-Optimization/README.md) — still only *asks* the model nicely to produce a certain kind of output. That's fine when the output is free-form text a human will read, but the moment downstream code needs to `json.loads()` the response, or call a specific function with specific arguments, "the model was instructed to output JSON" is not a guarantee — it's a hope. This lesson covers the two techniques that turn that hope into something closer to a guarantee: **constrained decoding**, which makes syntactic validity structurally impossible to violate, and **function calling**, the request → parse → execute → respond loop that lets a model's structured output actually *do* something in the real world — the exact mechanism [Lesson 3's ReAct loop](../03-Tree-of-Thought-and-ReAct/README.md#3-react-yao-et-al-2022-reason-act-observe-repeat) depends on every time it takes an "Act" step.

## What this lesson covers

- Why prompting alone can't *guarantee* valid structured output, no matter how good the model is
- Constrained decoding: masking illegal tokens to `-inf` before every sampling step
- A real JSON grammar, implemented as a character-level state machine
- Proof by experiment: a genuinely untrained, randomly-initialized model, forced to be 100% syntactically valid anyway
- Function calling: the `{"name": ..., "arguments": {...}}` request format
- The full request → parse → execute → respond loop, with real Python functions actually running
- Where this fits relative to ReAct: function calling is the mechanism, ReAct is one strategy for deciding *when* to invoke it

## 1. Why "just ask nicely" isn't enough

A well-trained, well-prompted model outputs valid JSON the overwhelming majority of the time — but "the overwhelming majority of the time" is not good enough for code that calls `json.loads()` on the response and crashes (or worse, silently misbehaves) on the failures. A single stray trailing comma, an unescaped quote inside a string, or a truncated response cut off mid-object breaks a naive parser. Two structurally different fixes exist: constrain the *generation process itself* so invalid tokens are never produced (Part A of `example.py`), or accept that the model's raw output is just a *request* and put a real, defensive execution layer on the other side of it (Part B). Production systems typically use both together.

## 2. Constrained decoding: guarantee validity at the token level

At every generation step, a language model produces a probability distribution (logits) over its entire vocabulary. Constrained decoding intervenes **before sampling**: given everything generated so far, compute the set of characters/tokens that are grammatically legal to come next, and set every other logit to `-infinity` (so its post-softmax probability becomes exactly zero) before sampling:

```
logits = model(generated_so_far)          # raw scores over the whole vocabulary
valid  = grammar.next_valid_tokens(generated_so_far)   # e.g. {'"', 'a', 'd', ...}
for token in vocabulary:
    if token not in valid:
        logits[token] = -inf
next_token = sample(softmax(logits))
```

Because sampling from a distribution with zero probability on every invalid token can *never* select one, this is a **hard guarantee**, not a statistical tendency — and critically, that guarantee holds **regardless of how good the underlying model is**. `example.py` Part A proves this in the most extreme way possible: it applies constrained decoding to a Transformer that has never been trained at all (pure random weights) and still gets perfectly valid output every time.

## 3. The grammar, as a state machine

`example.py` defines a small but real grammar for one JSON shape:

```
{"action": "add" | "sub" | "mul", "value": <1-to-3-digit integer, no leading zero>}
```

`next_valid_chars(prefix)` implements this as a character-level state machine: given only the string generated *so far*, it returns exactly the set of characters legal to emit next (an empty set means the structure is already complete — generation must stop). For example, once `rest` inside the quotes has matched `"add"` exactly, the only legal next character is the closing quote; after three digits have been emitted for `value`, the only legal next character is `}`. The script also implements a **second, independently-written** full-string validity checker and cross-checks thousands of random walks through the grammar against it — this is the same "verify the mechanism actually does what it claims" discipline used throughout this course, not just asserted.

## 4. The experiment: validity from masking, not from learning

With the grammar wired up, `example.py` Part A generates from a tiny, genuinely **untrained** causal Transformer (random initialization, no training loop anywhere in Part A) two ways: once with the grammar mask applied at every step, once with no mask at all (free sampling over the whole character vocabulary). The result: **100% of constrained generations are valid**, while unconstrained generations from the exact same random model are almost never valid. Since the model's weights are pure noise in both cases, none of that validity gap can be coming from anything the model "knows" — it comes entirely from the mask. This is precisely why production structured-output APIs (OpenAI's JSON mode / structured outputs, Anthropic's tool use, grammar libraries like Outlines or llama.cpp's GBNF grammars) implement constraints at the **decoding layer**, rather than relying on a well-trained model to comply on its own.

## 5. Function calling: request, parse, execute, respond

Constrained decoding guarantees a response *parses* — it says nothing about what the model should be allowed to actually *do*. Function calling is the convention that closes that gap. The model emits a structured request naming a function and its arguments:

```json
{"name": "convert_units", "arguments": {"value": 42, "from_unit": "km", "to_unit": "miles"}}
```

External code — never the model itself — parses that JSON, checks the requested function exists, dispatches to the **real** implementation with the **real** parsed arguments, executes it, and splices the real return value back into the final response. `example.py` Part B scripts exactly this loop end to end against two real Python functions (a calculator and a unit-converter), and independently recomputes both expected results to verify the real values came back correctly.

## 6. Why the model's job is deliberately small

The full loop only works because each side does the part it's actually good at: the model's entire job is to emit one syntactically valid request (which Section 2-4 showed can be *guaranteed* structurally, independent of model quality) — it never computes `128 * 37 + 6` itself, it asks a calculator to. The calling code's job is to parse, validate, dispatch, and execute against real systems (a database, an API, a calculator) that the model has no direct access to. This division is exactly what makes function calling trustworthy for anything with real consequences: a model can hallucinate fluently, but it cannot hallucinate a tool's return value once that tool has actually run.

## 7. Where this fits relative to ReAct

[Lesson 3's ReAct loop](../03-Tree-of-Thought-and-ReAct/README.md#3-react-yao-et-al-2022-reason-act-observe-repeat) interleaves reasoning ("Thought") with tool use ("Act") and feeds real tool results back in ("Observation") — but it never specified *how* an "Act" step turns into an actual function running. This lesson is that missing mechanism: every ReAct "Act" is, underneath, exactly the request → parse → execute → respond loop from Section 5, and a production ReAct agent typically also applies constrained decoding (Section 2-4) to guarantee each action request actually parses before dispatch is even attempted. Structured output and function calling aren't a separate topic from agentic prompting patterns — they're the load-bearing mechanism underneath them.

## Video Script Outline

1. Motivation — prompting alone can't *guarantee* valid output; downstream code needs more than a hope
2. Constrained decoding: mask illegal tokens to `-inf` before sampling, a hard guarantee independent of model quality
3. Walk through the grammar's state machine, `next_valid_chars`, and the independent validity-checker cross-check
4. Walkthrough of `example.py` Part A — an untrained random model, 100% valid with masking vs. almost never valid without it
5. Function calling: the `{"name", "arguments"}` format and the request → parse → execute → respond loop
6. Walkthrough of `example.py` Part B — real Python functions actually executing, real results spliced back in, independently verified
7. Why the model's job is deliberately narrow, and why that's what makes tool use trustworthy
8. Recap — tie back to Lesson 3's ReAct: this is the mechanism underneath every "Act" step, and this phase's final lesson: from here, [Phase 08](../../Phase-08-Evaluation-of-LLMs/README.md) covers how to actually measure whether any of these prompting strategies are working

## Further Reading

- OpenAI, *Function calling and other API updates* / *Structured Outputs* documentation
- Anthropic, *Tool use (function calling)* documentation
- Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* (the Outlines library — regex/CFG-constrained decoding in practice)
- Geng et al. (2023), *Grammar-Constrained Decoding for Structured NLP Tasks Without Finetuning*
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models* — revisited from [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md), the strategy layer built on top of this lesson's mechanism
