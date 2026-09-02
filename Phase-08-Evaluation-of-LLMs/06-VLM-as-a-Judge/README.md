# VLM-as-a-Judge

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `06-VLM-as-a-Judge`

## Why this matters

[Lesson 3](../03-LLM-as-a-Judge/README.md) established LLM-as-a-Judge for pure text: a judge reads a prompt and a candidate response and grades it, and the whole lesson is about biases in that grading process — position, verbosity, self-preference — that distort a judge's verdict *even when it has full access to everything it needs to grade correctly*. The moment the thing being graded involves an image — a caption, a visual question answer, a described chart or screenshot — a text-only judge loses that "full access" assumption entirely: it can tell you whether a caption reads fluently, but it cannot tell you whether the caption actually describes what's in the picture, because it never saw the picture. A **VLM-as-a-judge** — a vision-language model (the same architecture family built in [Phase 10 Lesson 1](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#3-from-an-aligned-space-to-a-multimodal-llm-llava)) that receives the image alongside the text and rubric — closes that gap, but only if it actually uses the image to check claims, rather than just producing another fluent-sounding guess. This lesson is about that new failure mode — a judge's *grounding* — sitting on top of everything Lesson 3 already covered.

## What this lesson covers

- Why a text-only judge structurally cannot verify claims about an image
- New, vision-specific judge failure modes: object hallucination, counting errors, spatial/relational errors
- How these differ from Lesson 3's biases: those distort a judge's *behavior*, these are about whether it can even *check* the claim at all
- Rubric design for vision tasks: grading only groundable, checkable claims
- A real, from-scratch simulation: a judge that verifies captions against a known ground-truth scene, vs. one that can't
- Why an ungrounded "judge" degenerates back into exactly Lesson 3's length/fluency-driven biases

## 1. Why a text-only judge can't grade multimodal output

Lesson 3's two protocols — pairwise comparison and absolute rubric scoring — both quietly assume the judge can independently assess how well a response matches reality closely enough to grade it. For open-ended text with no reference answer, that's already hard, and Lesson 3's entire subject is how a judge's verdict gets distorted even so. For a captioning or visual-QA task, the assumption breaks completely rather than just imperfectly: a judge with no image access literally cannot distinguish "the sky is blue and there are three birds" from "the sky is green and there are five birds." It can only fall back to fluency, plausibility, and surface style — none of which correlate with whether the claim is actually true.

## 2. New failure modes specific to vision grounding

- **Object hallucination** — the caption mentions an object or attribute that isn't actually present in the image, the vision-language analogue of the text hallucination covered in [Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md#1-defining-hallucination); it's well-documented enough in real VLM captioning research to have its own dedicated benchmarks (CHAIR, POPE — see Further Reading).
- **Counting errors** — correctly identifying that an object is present, but getting how many wrong.
- **Spatial/relational errors** — left/right, above/below, in-front-of/behind stated incorrectly.

Lesson 3's three biases persist *regardless* of whether the judge could, in principle, tell the responses apart correctly — they're about how a judge distorts a comparison it's otherwise capable of making. These failure modes are one level more basic: they're about whether the judge can make a valid comparison **at all**.

## 3. Protocol and rubric design

The same pairwise/absolute distinction from [Lesson 3 §1](../03-LLM-as-a-Judge/README.md#1-two-judge-protocols) still applies — a VLM-judge just has the image in its context alongside the text and (for absolute scoring) a rubric. The one addition that actually matters: a usable vision rubric has to decompose into individually **groundable** claims — does the caption name every salient object that's actually present? get the count right? get the spatial relation right? — rather than a vague holistic criterion like "is this a good caption." A criterion nobody can check against the image is exactly as useless here as an uncontrolled-for confound was in Lesson 3 — it can't be turned into a real, measurable signal.

## 4. `example.py` — a real, grounded VLM-judge, built from a symbolic scene

This repo has no real vision model or actual pixels to work with, so `example.py` builds something smaller but genuinely real and checkable instead of a hand-wavy description: a **scene** is an explicit, structured ground-truth data structure — a small list of (color, shape, count) objects plus one spatial relation between two of them, a tiny CLEVR-style scene graph standing in for an image. From each scene, a fixed caption template generates the one fully correct caption, and four labeled error variants: an object-hallucination sentence appended for a color/shape combination that never appears in the scene, a count swapped to the wrong number, an attribute (color) swapped to one that doesn't match any object actually present, and a spatial relation flipped to its opposite.

Two judges then grade these captions:

- **The grounded judge** parses each caption's count and relation claims back out with a small, honest regex-based extractor (it doesn't need real NLP — the caption vocabulary is fixed by construction, exactly the same "small but real mechanism" this repo uses for other constrained-language tasks) and checks every claim against the scene's actual ground truth.
- **The ungrounded judge** sees only the caption text — no scene — and falls back to the one signal it actually has: response length, i.e. exactly [Lesson 3 §3](../03-LLM-as-a-Judge/README.md#3-verbosity-bias)'s verbosity bias, deployed here not as an add-on flaw but as the *entire* thing left for a judge with no way to check the substance of the claim.

`example.py` runs both judges, thousands of times, on random scene/caption pairs and reports real measured pairwise win rates (does the judge correctly prefer the fully correct caption over each error variant?) for every error type, plus a worked single-scene walkthrough showing exactly which claims the grounded judge marks true or false and why.

## 5. Recap: complementary to Lesson 3, not a replacement for it

Even a properly grounded VLM-judge — one that verifies every claim against the image correctly — can still exhibit position bias, verbosity bias, and self-preference bias in how it *weighs* two already-verified captions against each other. Grounding (this lesson) and the bias mitigations ([Lesson 3 §5](../03-LLM-as-a-Judge/README.md#5-mitigations): swap-and-average orderings, length control, diverse judge panels) solve different problems and both are needed in a production multimodal eval pipeline — one without the other still leaves a real, measurable failure mode on the table.

## Video Script Outline

1. Motivation — Lesson 3's judge could see everything it needed to grade; a multimodal judge with no image access can't
2. The new failure modes: object hallucination, counting errors, spatial/relational errors — and how they differ in kind from Lesson 3's biases
3. Rubric design: only groundable, checkable claims are gradeable at all
4. The scene graph: a tiny, structured, CLEVR-style ground truth standing in for a real image
5. The four labeled error injections, and the grounded judge's real claim-by-claim check against the scene
6. The ungrounded judge: no image access, falls back to Lesson 3's verbosity bias as its only remaining signal
7. Walkthrough of `example.py`'s aggregate win-rate table — grounded near-perfect, ungrounded at chance or actively backwards
8. Recap — grounding and Lesson 3's bias mitigations solve different problems; a real pipeline needs both

## Further Reading

- Rohrbach, Hendricks, Burns, Darrell, Saenko (2018), *Object Hallucination in Image Captioning* (introduces the CHAIR metric, the original object-hallucination measurement for image captioning)
- Li et al. (2023), *Evaluating Object Hallucination in Large Vision-Language Models* (introduces the POPE benchmark)
- Liu, Li, Wu, Lee (2023), *Visual Instruction Tuning* — the LLaVA paper, revisited from [Phase 10 Lesson 1](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#3-from-an-aligned-space-to-a-multimodal-llm-llava); the architecture a VLM-judge itself typically is
- Zheng, Chiang, Sheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — revisited from [Lesson 3](../03-LLM-as-a-Judge/README.md), the text-only judge biases this lesson builds on top of
