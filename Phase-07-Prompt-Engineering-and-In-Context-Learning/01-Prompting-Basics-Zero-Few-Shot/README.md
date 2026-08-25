# Prompting Basics: Zero-Shot and Few-Shot

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `01-Prompting-Basics-Zero-Few-Shot`

## Why this matters

Every lesson up to this point has been about *changing the model* — architecture ([Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md)), pretraining objective ([Phase 04](../../Phase-04-Pretraining-LLMs/README.md)), weights ([Phase 05](../../Phase-05-Finetuning-LLMs/README.md)), or its preferences ([Phase 06](../../Phase-06-Alignment-and-RLHF/README.md)). This phase is about the opposite: the weights are now completely **frozen**, and the only lever left is *what text you put in the prompt*. [Phase 03 Lesson 1 §3](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#3-gpt-3-in-context-learning-emerges) already introduced the headline result — GPT-3 can perform a brand-new task from a handful of examples placed directly in the prompt, with no gradient updates at all. This lesson makes that mechanism concrete and measurable on a task small enough to train and probe end to end, and sets up the vocabulary (zero-shot, few-shot, in-context examples, prompt sensitivity) that every later lesson in this phase — chain-of-thought ([Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md)), tree-of-thought and ReAct ([Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md)), automatic prompt optimization ([Lesson 4](../04-Automatic-Prompt-Optimization/README.md)), and structured output ([Lesson 5](../05-Structured-Output-and-Function-Calling/README.md)) — builds directly on top of.

## What this lesson covers

- Zero-shot vs. few-shot prompting: what "shot" actually refers to
- In-context learning (ICL) as a form of learning that updates no weights at all
- Why a model has to be trained on *varying* tasks/episodes before ICL can work
- `example.py`: training a tiny decoder-only Transformer on a family of hidden-parameter functions, then testing genuine in-context generalization to unseen parameters
- How accuracy scales with the number of in-context examples
- Prompt sensitivity to example order (Zhao et al., 2021, "Calibrate Before Use")

## 1. Zero-shot vs. few-shot: what "shot" means

A **shot** is one input/output example placed in the prompt before the real query. **Zero-shot** prompting gives the model only an instruction or the bare query — no worked examples. **Few-shot** prompting prepends `k` example pairs (`k` is usually 1-32 for real LLMs) directly into the context window before the actual question, e.g.:

```
2>4,3>5,0>2,x_query>
```

is a 3-shot prompt: three example pairs, followed by a query the model must answer by pattern-matching against them. Crucially, nothing about the *model* changes between zero-shot and few-shot — the only difference is the text in the context window. This is the entire premise of prompt engineering as a discipline: behavior is being steered through input alone.

## 2. In-context learning: learning with the weights frozen

GPT-2 showed zero-shot task transfer; GPT-3 showed that stacking a handful of examples into the prompt made this dramatically more reliable — a phenomenon Brown et al. (2020) named **in-context learning (ICL)**. What makes ICL strange, and worth dwelling on, is that it looks like learning (performance improves as more examples are supplied) but involves **zero gradient updates** — no backward pass, no optimizer step, nothing written to any weight. Whatever "learning" is happening is entirely a computation carried out in the forward pass, using the in-context examples as data the attention mechanism reads and conditions on.

This immediately raises the question `example.py` is built to answer directly: if the weights never change, where does the ability to solve a *genuinely new* task come from? The answer is that the weights were pretrained on a large **distribution of related tasks**, and the network learned a general-purpose *algorithm* — "read some examples, infer the rule that connects their inputs to their outputs, apply that rule to the new query" — rather than memorizing any single task's answer key. ICL at inference time is that pretrained algorithm running on a fresh instance of the same family.

## 3. `example.py`: a task family small enough to train and verify

Real GPT-3 in-context learning can't be inspected end-to-end — nobody can retrain GPT-3 to check the mechanism. `example.py` builds a **small, honest analogue** instead, following the recipe in section 2 exactly:

- The task family is `y = (x + k) mod M`, for `M = 6` symbols and a hidden shift `k`.
- Every training example is a fresh **episode**: a random `k` is drawn, a handful of `(x, y)` pairs generated under that `k` are shown as in-context examples, and the model must predict `y` for one more query `x` — trained purely with next-token prediction, the same objective as every GPT model in this course.
- Because `k` is re-sampled every single episode, the model can never memorize a fixed lookup table. The *only* way to answer correctly is to infer `k` from the examples given in that specific prompt and apply it — precisely the "read examples, infer the rule, apply it" algorithm from section 2, forced into existence by the training distribution itself.
- The architecture is the exact `MiniGPT` decoder-only stack from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) (causal self-attention + feed-forward blocks), re-declared locally so the lesson is self-contained.
- After training on shifts `k in {0,1,2,3}` only, the model is evaluated on shifts `k in {4,5}` — values it **never saw during training**. Any accuracy above the `1/M` chance baseline on these held-out shifts can only come from reading and using the in-context examples at inference time, with weights completely frozen. This is the measured, reproducible version of what section 2 described in the abstract.

## 4. Number of in-context examples vs. accuracy

`example.py` sweeps the number of shown examples (`n_shown = 1..5`) and measures held-out-shift accuracy at each point. Since training only ever showed between 2 and 5 examples per episode, 1-shot prompts fall outside the training distribution and accuracy there is measurably the worst. More examples give the model more redundant evidence to pin down the hidden `k` before it has to commit to an answer — the same qualitative curve reported for real LLMs, where accuracy on a new task typically rises (with diminishing returns) as more few-shot examples are added, up to the point where the context window or example diversity runs out.

## 5. Prompt sensitivity: example order matters (Zhao et al., 2021)

Holding the *set* of in-context examples fixed but changing their *order* should not matter if the model has learned a truly order-invariant rule-extraction algorithm. `example.py` tests this directly: it compares accuracy when examples are shown sorted by `x` (the order used throughout training) against a scrambled order the model never trained on, using the exact same examples and the exact same hidden `k`. Zhao et al. (2021), *"Calibrate Before Use: Improving Few-Shot Performance of Language Models,"* documented this same effect in real LLMs — logically equivalent few-shot prompts that differ only in example order (or minor formatting) can produce measurably different accuracy, because the model's training data never taught it perfect invariance to those surface details. This is a direct, practical consequence for anyone writing prompts: example order, formatting, and even which examples are chosen are not neutral choices, and should be controlled for (or explicitly randomized and averaged over) rather than assumed harmless — the same instinct that motivates the automatic prompt search in [Lesson 4](../04-Automatic-Prompt-Optimization/README.md).

## Video Script Outline

1. Motivation — the weights are frozen now; the only lever left is the prompt
2. Zero-shot vs. few-shot, defined precisely with a prompt example
3. In-context learning as "an algorithm running in the forward pass," not weight updates — recap GPT-3's headline result
4. The puzzle: how can a frozen model learn a *new* task? Answer: pretraining on a distribution of tasks, not one fixed task
5. Walkthrough of `example.py`'s episodic training setup — hidden shift `k`, held-out test shifts, why memorization is impossible
6. Run it: before/after training accuracy on never-seen shifts, purely via in-context examples
7. Two measured curves: examples-count vs. accuracy, and sorted-vs-scrambled order (Zhao et al. 2021)
8. Recap + preview: prompting the model to *show its work* is next (Chain-of-Thought, Lesson 2)

## Further Reading

- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3; the paper that named and popularized in-context learning)
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2; zero-shot task transfer, the precursor result)
- Zhao et al. (2021), *Calibrate Before Use: Improving Few-Shot Performance of Language Models*
- Liu et al. (2021), *What Makes Good In-Context Examples for GPT-3?* (example selection and ordering effects)
- Min et al. (2022), *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*
