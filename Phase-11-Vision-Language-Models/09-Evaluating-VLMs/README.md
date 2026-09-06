# Evaluating VLMs

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `09-Evaluating-VLMs`

## Why this matters

[Phase 08](../../Phase-08-Evaluation-of-LLMs/README.md) established how hard it is to evaluate a text LLM honestly: metrics that don't measure what they claim, judges with biases, contamination, and benchmarks that saturate. Every one of those problems exists for VLMs, plus a new one that is specific and pervasive — **a VLM benchmark can be substantially solvable without the image**. A text-only model with good world knowledge can answer a surprising share of "visual" questions from the question text, the option list, and general plausibility, and any score that doesn't isolate that share is measuring the language model's priors rather than the vision system. This lesson builds a small benchmark harness that reproduces the three pathologies that most often invalidate published VLM numbers, and ends with the reporting checklist that makes a result interpretable.

## What this lesson covers

- The blind baseline: how much of a VLM benchmark needs no visual input, measured
- Why visual shortcuts exist in benchmarks at all, and how MMStar-style filtering removes them
- Option-position bias, and circular evaluation as the fix
- Contamination, which for VLMs happens by default rather than by accident
- Answer extraction: generation-and-parse vs. option log-likelihood
- The benchmark landscape: what MMMU, MMBench, DocVQA, ChartQA, POPE, MathVista and video benchmarks each actually measure
- A minimum reporting checklist

## 1. The blind baseline

`example.py` builds a four-option MCQ benchmark over a synthetic scene. Half its items are **vision-necessary** (options are four colours; only the image says which is right). Half are **shortcut** items whose distractors are the wrong *kind* of answer entirely — the sort of item that appears in real benchmarks as an implausible distractor, the only grammatically coherent option, or a question whose answer is the world's default.

```
model                   FULL benchmark   vision-necessary   shortcut items
VLM (sees the image)            86.3%             73.3%           100.0%
BLIND (text only)               61.2%             24.3%           100.0%
```

The headline is 86.3%. A model with **no visual input whatsoever** scores 61.2% on the same items. Only 25 points of that headline are attributable to vision, and the vision-necessary column — where the blind model sits at chance, as it must — is the only one measuring what the benchmark claims to measure.

This is not a synthetic curiosity. Analyses of MMMU, ScienceQA and MMBench have found large fractions of items answerable by text-only models, and **MMStar** was constructed specifically by filtering out items that strong LLMs could answer without the image. The operational rule is simple and cheap: **run your harness with the images removed.** Whatever it scores is the part of your benchmark that isn't about vision. This is the same move as [Phase 08 Lesson 6](../../Phase-08-Evaluation-of-LLMs/06-VLM-as-a-Judge/README.md)'s ungrounded-judge control, used here on the benchmark rather than on the judge.

Two related controls are worth running when a result matters: **shuffled-image** (pair each question with a random other image — a model that scores the same either way is not using the image) and **no-question** (options only).

## 2. Option-position bias and circular evaluation

Most VLM benchmarks are multiple-choice, and most VLMs are scored by which *letter* they emit. That makes the answer's position an input to the model, and models have position preferences — inherited from instruction data that is not position-balanced, exactly as [Phase 08 Lesson 3](../../Phase-08-Evaluation-of-LLMs/03-LLM-as-a-Judge/README.md) documented for judges.

`example.py` §2 tunes one model on data where the correct option sits at position A 55% of the time, and compares it with a balanced-data model. On a test set whose correct answers *are* uniformly placed:

```
                                 A       B       C       D
balanced-data model         25.4%   25.6%   25.9%   23.1%
position-skewed model       31.8%   23.0%   23.8%   21.4%
```

```
                             A-heavy test set   balanced test set   circular eval
balanced-data model                    86.2%              86.3%           86.4%
position-skewed model                  94.2%              88.3%           88.4%
```

The skewed model gains six points on a benchmark that happens to favour A — and a benchmark assembled by people has no particular reason to be balanced. **Circular evaluation** (MMBench's fix) scores every item `N_options` times, once per rotation of the options, and averages. The content is identical across rotations, so any variation is pure position bias, and averaging removes it: the skewed model's 94.2% falls to 88.4%, in line with its real ability. It costs 4× the inference, and it is worth it whenever the number will be quoted.

## 3. Contamination

```
model                                leaked items   clean items   FULL test
trained on train set only                  85.8%         86.4%      86.3%
train set + 400 leaked test items         100.0%         88.5%      89.6%
```

The 400 leaked items (10% of the test set) go to 100% — memorized, not solved. On clean items the contaminated model is barely ahead, and that small margin is just extra training data doing ordinary work. So the headline rises while capability is unchanged, and **nothing in the score distinguishes the two cases.**

VLMs are unusually exposed here, for a structural reason worth stating plainly: visual instruction data is routinely synthesized *from public datasets* ([Lesson 6 §1](../06-Visual-Instruction-Tuning-and-VLM-Data/README.md#1-what-visual-instruction-data-is-and-where-it-comes-from)), and those same datasets supply the benchmark images. COCO images appear in training mixtures and in benchmarks. Contamination is the default state unless someone actively prevents it, and preventing it means deduplicating on the **image** — hashes plus near-duplicate search on embeddings — because the question text is often freshly generated and will not match.

## 4. Answer extraction is part of the benchmark

Two ways to score the same MCQ item:

- **Generation + parse**: let the model generate, then extract the letter with a regex or a small parser. Measures the whole system, including instruction-following, and is fragile — "The answer is B" parses, "I think it's the blue one" often doesn't, and a model that phrases answers unusually gets penalized for formatting rather than for being wrong.
- **Option log-likelihood**: score each option under the model and take the highest. Robust and cheap, but it measures a discrimination ability the model never uses at inference time, and it can flatter a model that would have babbled.

Neither is wrong; they measure different systems, and results are not comparable across the two. Many published discrepancies between reproductions of the same benchmark are entirely this. **State which one you used** — and if you used generation, report the parse-failure rate, since a silently-zero-scored unparseable answer is indistinguishable from a wrong one.

## 5. The benchmark landscape

| Benchmark | Measures | Watch out for |
|---|---|---|
| **MMMU** | college-level multi-discipline reasoning with figures | heavy knowledge component; a large blind-solvable share |
| **MMStar** | vision-indispensable items, filtered by design | small; deliberately hard |
| **MMBench** | broad capability taxonomy, circular eval built in | Chinese/English splits differ |
| **MME / SEED-Bench** | broad perception + cognition, yes/no or MCQ | yes/no format invites the biases from [Lesson 7](../07-VLM-Hallucination-and-Alignment/README.md) |
| **POPE / CHAIR** | object hallucination | quote the negative-sampling regime (Lesson 7 §2) |
| **DocVQA / InfographicVQA** | reading real documents | resolution-bound; report the input resolution (Lesson 8 §2) |
| **ChartQA / PlotQA** | chart reasoning | numeric-tolerance rules change the score materially |
| **TextVQA / OCRBench** | scene and dense text | same resolution caveat |
| **MathVista / MathVerse** | visual mathematical reasoning | MathVerse specifically tests whether the diagram is used at all |
| **RefCOCO / grounding suites** | localization by IoU | IoU threshold and coordinate format matter (Lesson 8 §1) |
| **Video-MME / MVBench / EgoSchema** | video understanding | frame count and sampling dominate the result (Lesson 8 §3) |
| **VLM arenas / judge-based evals** | open-ended helpfulness | inherits every bias from [Phase 08 Lesson 3](../../Phase-08-Evaluation-of-LLMs/03-LLM-as-a-Judge/README.md) and [Lesson 6](../../Phase-08-Evaluation-of-LLMs/06-VLM-as-a-Judge/README.md) |

The pattern across the table: **the evaluation protocol is often a bigger lever on the number than the model is.** Resolution, frame count, answer extraction, option order, and decontamination each move scores by more than the gap between adjacent models on most leaderboards.

## 6. The checklist

For a VLM result to be interpretable, report:

- the **blind (text-only) baseline** on the same harness
- **circular / permuted-option** evaluation, or a stated position-bias check
- **how answers were extracted** (generation + parse, or option log-likelihood), and the parse-failure rate
- the **input resolution and vision-token count** actually used
- for video, the **frame count and sampling strategy**
- the **decontamination procedure**, by image and not only by question text
- **per-subtask numbers**, not just an aggregate average

## Video Script Outline

1. The problem specific to VLM evaluation: a "visual" benchmark that doesn't need the image
2. The harness, and the two item families
3. The blind baseline result: 86.3% headline, 61.2% with no image at all
4. Why shortcut items exist, and MMStar's filtering approach
5. Shuffled-image and no-question controls
6. Position bias: letter distributions, and a six-point swing from option placement alone
7. Circular evaluation: what it costs and what it removes
8. Contamination: 100% on leaked items, no capability gain, invisible in the score
9. Why VLM contamination is structural — instruction data synthesized from benchmark datasets
10. Answer extraction as part of the benchmark, and why reproductions disagree
11. The landscape table, and the claim that protocol beats model
12. The reporting checklist
13. Recap + preview of [Lesson 10](../10-VLM-Inference-and-Deployment/README.md)

## Further Reading

- Chen et al. (2024), *Are We on the Right Way for Evaluating Large Vision-Language Models?* (MMStar; the blind-solvability analysis this lesson's §1 is built on)
- Yue et al. (2023), *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark*
- Liu et al. (2023), *MMBench: Is Your Multi-modal Model an All-around Player?* (circular evaluation)
- Zhang et al. (2024), *MathVerse* (isolating whether the diagram is actually used)
- Fu et al. (2024), *Video-MME* (frame-count sensitivity in video evaluation)
- Duan et al. (2024), *VLMEvalKit* (a harness that standardizes extraction and protocol across benchmarks)
