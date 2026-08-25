# The Alignment Problem

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `01-The-Alignment-Problem`

## Why this matters

Every lesson up to this point has been about building a model that is good at exactly one thing: predicting the next token of text, as trained in [Phase 04: Pretraining LLMs](../../Phase-04-Pretraining-LLMs/README.md) using the objective first introduced all the way back in [Phase 01 Lesson 1](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md#1-what-a-language-model-actually-is) and trained end-to-end on a real architecture in [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md). That objective — "given the text so far, what token is statistically most likely to come next in *this kind of text*?" — is extraordinarily powerful for learning grammar, facts, reasoning patterns, and style from raw text. But it is quietly **not the same objective** as "be a helpful, honest assistant that answers the user's question." A model trained purely on that objective has no notion of "helpful" at all; it only knows "plausible continuation." This lesson makes that gap concrete, motivating why the entire rest of this phase (reward modeling, RLHF, DPO, RLAIF/Constitutional AI, and safety mitigation) exists as a **separate training stage** bolted on after pretraining and instruction tuning ([Phase 05 Lesson 4: Instruction Tuning (SFT)](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md)).

## What this lesson covers

- Why next-token prediction alone does not produce an assistant
- The helpful, honest, harmless (HHH) framing for what "aligned" means in practice
- Three distinct failure modes a raw pretrained ("base") model exhibits when treated as a chatbot
- Why SFT narrows but does not close the gap, and why a further preference-based stage is needed
- A hands-on demonstration: training a tiny base model on unlabeled, unstructured text and watching it fail to "answer" a question, for the honest reason that it was never shown what answering looks like

## 1. The pretraining objective, restated precisely

A pretrained language model is trained to maximize:

```
P(token_t | token_1, ..., token_{t-1})
```

over a huge corpus of raw text scraped from books, websites, forums, code, and so on ([Phase 04 Lesson 1: Pretraining Data Pipeline](../../Phase-04-Pretraining-LLMs/01-Pretraining-Data-Pipeline/README.md)). Nothing in that objective distinguishes "text that answers a question helpfully" from "text that continues a question with more questions," "text that trails off," or "text that continues a harmful request with harmful content" — the model is only ever rewarded for matching the *statistics of its training distribution*. If the training distribution contains FAQ lists where questions are followed by more questions, forum threads where a request is followed by a deflection, or harmful text followed by more harmful text, the model will just as happily learn to reproduce those patterns as it would learn to reproduce a helpful answer — because **from the loss function's point of view, they are exactly the same kind of prediction task.** A base model is a highly capable *next-token statistics engine*, not an agent with intentions like "help this person."

## 2. Three concrete failure modes of a raw base model

Prompting a raw pretrained model (no instruction tuning, no alignment) as though it were a chatbot tends to surface one of a few characteristic failures:

1. **Continuing the question instead of answering it.** If question-like text in the training data is usually followed by more question-like text (a FAQ page, a survey, a list of discussion prompts), the statistically most likely continuation of a question is *another question*, not an answer — because that is what the model actually saw.
2. **Trailing off or restating.** If the training data rarely contains the pattern "question immediately followed by a direct, concise answer," the model has nothing to imitate and may simply continue with generic, tangential text.
3. **Continuing harmful content harmfully.** If the statistically likely continuation of a harmful prompt (because such prompts appear in the training data followed by harmful text — forums, fiction, arguments) is more harmful text, a raw base model has no built-in mechanism to refuse — refusal is a *behavior* that must be explicitly taught, not a default.

All three failures share one root cause: **the model is doing exactly what it was trained to do (predict plausible continuations); "helpful," "honest," and "harmless" were simply never part of that training signal.**

## 3. The HHH framing

Anthropic's alignment framing (Askell et al., 2021, *A General Language Assistant as a Laboratory for Alignment*) organizes the target behavior into three properties, often abbreviated **HHH**:

- **Helpful** — actually try to satisfy the user's request, ask clarifying questions when needed, give a direct and useful answer.
- **Honest** — report the model's actual best estimate, express calibrated uncertainty, and avoid fabricating information (a version of the [hallucination problem](../../Phase-08-Evaluation-of-LLMs/04-Hallucination-and-Factuality-Evaluation/README.md), which is partly a pretraining-vs-alignment issue too).
- **Harmless** — refuse or safely deflect requests that would cause real-world harm, without being so overcautious that it stops being helpful.

None of HHH is implied by the pretraining loss. Each one is a *behavioral specification* that has to be trained in separately, and — crucially — the three properties are frequently **in tension** with each other (an overly cautious model that refuses everything is harmless but not helpful; a model that never expresses doubt is unhelpfully overconfident and, in a subtle sense, less honest). Balancing this tension is a large part of what the rest of this phase is actually about.

## 4. Why SFT narrows the gap but doesn't close it

Instruction tuning / SFT ([Phase 05 Lesson 4](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md)) fine-tunes the base model on curated `(instruction, good response)` pairs, which directly teaches the *format and existence* of the helpful-answer behavior — this alone is a huge improvement over the raw base model. But SFT only shows the model *examples of good behavior* it should imitate; it never tells the model *how much better one response is than another*, never punishes subtly-bad-but-plausible responses that weren't in the training set, and provides no mechanism to correct for the model's own idiosyncratic failure modes discovered after training. That requires comparative feedback on the model's *own* outputs — which is exactly what [Lesson 2: Reward Modeling](../02-Reward-Modeling/README.md) and [Lesson 3: RLHF with PPO](../03-RLHF-with-PPO/README.md) add on top.

## 5. What `example.py` demonstrates

To make this gap undeniable rather than asserted, `example.py` trains a genuinely tiny GPT-style model (the exact decoder-only architecture from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)) on a small, deliberately **unstructured, raw, internet-like** toy corpus: a mix of plain statements and questions, with **no instruction formatting and no example of a question being directly answered anywhere in the data**. After training, the model is prompted with a direct factual question. Because the training data never contained the pattern "question -> direct answer," the model cannot produce one — instead it does exactly what the failure modes above predict: it continues with another question-shaped line or an unrelated statement, never the factual answer. This isn't a "dumb model" problem that more scale would fix on its own; it is a direct, mechanical consequence of what the training objective and training data did and did not contain.

## Video Script Outline

1. Motivation — "pretraining teaches next-token prediction; it never teaches 'be helpful'"
2. The pretraining objective restated, and why it can't distinguish helpful text from any other statistically plausible text
3. Three failure modes: answering with more questions, trailing off, continuing harmful text
4. The HHH framing (helpful, honest, harmless) and the tensions between the three
5. Why SFT helps but isn't sufficient on its own
6. Live walkthrough of `example.py` — train the tiny base model, prompt it with a real question, watch it fail to answer for a mechanically clear reason
7. Preview: Lessons 2-6 build the machinery (reward models, RLHF, DPO, RLAIF, safety) that actually closes this gap

## Further Reading

- Askell et al. (2021), *A General Language Assistant as a Laboratory for Alignment* (the HHH framing)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — the paper that popularized the full pretraining -> SFT -> RLHF pipeline)
- Bai et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*
- Bender, Gebru et al. (2021), *On the Dangers of Stochastic Parrots* (a critical perspective on what pretrained LMs do and do not "understand")
