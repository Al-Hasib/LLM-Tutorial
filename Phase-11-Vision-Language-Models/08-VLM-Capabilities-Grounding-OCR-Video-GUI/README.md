# VLM Capabilities: Grounding, OCR, Documents, Video and GUIs

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `08-VLM-Capabilities-Grounding-OCR-Video-GUI`

## Why this matters

Everything so far has treated "answer a question about an image" as the task. The capabilities that make VLMs economically interesting are more specific than that: point at the defect in this photo, read this receipt, extract this table, find the moment in this video where the machine stops, click the Submit button on this screen. Each of these looks like a different research area, and each turns out to be governed by one hard constraint that no amount of language modelling can argue with — a vocabulary limit, a resolution limit, or a sampling limit. This lesson takes the four biggest capability families, shows how each is expressed in a plain autoregressive VLM without new architecture, and measures the constraint that actually binds. It is the practical payoff of Lessons 1–7 and the input to [Lesson 9](../09-Evaluating-VLMs/README.md)'s evaluation questions.

## What this lesson covers

- Grounding: boxes and points as ordinary tokens, and how coordinate quantization sets a precision ceiling
- Measured: bin count vs. localization error vs. exact-match accuracy — three curves that move differently
- OCR and documents: why resolution, not model quality, decides whether text is readable
- Measured: legibility as a function of downscaling, and the token bill for keeping it
- Charts, tables and structured extraction: getting machine-readable output out of a language model
- Video: why every video result is really a frame-sampling result
- GUI and screen understanding: what agentic VLMs need that photo VLMs don't
- The referring/segmentation family, and where a plain VLM stops being enough

## 1. Grounding: coordinates are just tokens

The key realization, from Pix2Seq and adopted by Kosmos-2, Qwen-VL, Florence-2, PaliGemma and others: **you do not need a detection head**. Discretize the image into a grid of coordinate bins, add one token per bin to the vocabulary, and grounding becomes a sequence-prediction problem indistinguishable from any other:

```
"Where is the cat?"  ->  "<box>(x=412, y=178)(x=690, y=520)</box>"
```

No anchors, no non-maximum suppression, no architectural change — a data-and-vocabulary decision. This is why a general-purpose VLM can point at things at all, and why the same model can emit boxes, points, polygons, or box-interleaved captions ("a <box>cat</box> sitting on a <box>mat</box>") just by being trained on data formatted that way.

The design choice that matters is **how many bins**. `example.py` §1 trains the same model with three grid resolutions and reports three different things:

```
 coord bins  exact-bin acc  mean loc. error  quantization floor
          4         91.0%           0.0974              0.0954
         16         60.2%           0.0319              0.0239
         64          4.8%           0.0357              0.0060
```

- The **quantization floor** is the error a *perfect* model would still make from rounding to the nearest bin. It falls as the grid gets finer, so a coarse grid caps precision no matter how good the model is.
- **Exact-bin accuracy** collapses in the other direction, because the model must choose among far more classes from the same evidence.
- The number that actually matters — **mean localization error** — improves and then *saturates*. 4 → 16 bins is a genuine 3× improvement; 16 → 64 buys nothing, because by then the model, not the grid, is the limit.

The general lesson: extra bins past the model's own precision are free precision on paper and none in practice. To actually localize better you feed the model more pixels. Real systems sit in the middle — ~1000 bins over a normalized image (Pix2Seq, Qwen-VL), or a 32×32 grid of dedicated location tokens (Kosmos-2).

A second consequence worth noting: grounded output is **checkable**. A box can be compared against ground truth, or against a detector, in a way "the cat is on the left" cannot. That makes grounding a mitigation for hallucination as well as a capability ([Lesson 7 §4](../07-VLM-Hallucination-and-Alignment/README.md#4-mitigations-that-are-not-alignment)).

## 2. OCR and documents: resolution decides

The most common misdiagnosis in applied VLM work is treating unreadable text as a model failure. `example.py` §2 shows why it usually isn't. A real 32×32 image contains one glyph from a 10-glyph alphabet; the image is downscaled to the encoder's input resolution, patchified at a fixed 4×4 patch size, and classified:

```
 input res   tokens   glyph 16px   glyph 8px   glyph 4px
       32px       64       81.4%      95.6%      97.7%
       16px       16       97.2%      98.2%      32.0%
        8px        4       99.0%      47.7%      19.9%
```

Down the last column, a 4-pixel glyph is legible at full resolution and gone once the image is halved. Across the bottom row, the same encoder that reads a large glyph perfectly is at chance on a small one. What decides legibility is the glyph's size **after downscaling** — and the `tokens` column is the bill for keeping it, growing as `res²`.

Scale that to reality: a 12px character in a 1600px scan, downscaled to a 336px encoder input, is about 2.5 pixels tall. There is no language model on the other side of that encoder that can read it, because the information was destroyed before the LLM was reached. Every fix is upstream:

- **Higher input resolution** — direct, and quadratically expensive.
- **Dynamic tiling / AnyRes** ([Lesson 1 §5](../01-Vision-Encoders-and-Image-Tokenization/README.md#5-high-resolution-in-practice-dynamic-tiling)) — encode tiles at native resolution; the standard answer for documents.
- **Purpose-built high-resolution encoders** — Donut, Pix2Struct, and the document-specialist family, which forgo general vision quality for text legibility.
- **Native-resolution towers** — no square resize at all (Qwen2-VL, NaViT-style).

Two further document-specific issues sit on top of resolution. **Reading order** — a two-column PDF, a form, a table — is a layout problem the patch sequence doesn't solve on its own, which is why layout-aware training data matters. And **structured output**: extracting a table means emitting machine-readable structure (JSON, HTML, Markdown), which is [Phase 07 Lesson 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/05-Structured-Output-and-Function-Calling/README.md)'s territory with the added twist that the schema's *content* must be grounded in pixels. Charts are the same problem in a harder form: the answer key for a rendered chart is the data it was rendered from, which makes synthetic chart data unusually high quality (Lesson 6 §5) and chart benchmarks unusually clean.

## 3. Video: a sampling problem wearing a reasoning problem's clothes

`example.py` §3 makes this as stark as it can be made. A 32-frame clip contains the crucial event in exactly one frame; the model sees `k` uniformly sampled frames:

```
 frames sampled  vision tokens*  event captured   accuracy  acc | captured
              1              64           2.8%     19.1%         100.0%
              4             256          12.3%     26.5%         100.0%
              8             512          25.0%     37.1%         100.0%
             16           1,024          48.9%     56.6%         100.0%
             32           2,048         100.0%    100.0%         100.0%
```

The last column is the whole point: **given the right frame, the model is perfect**. Overall accuracy tracks the capture rate, not the model's ability. Every point of "video understanding" in that middle column is sampling, not modelling.

A 10-minute clip at 1 fps is 600 frames. At a frugal 64 tokens per frame that is 38,400 tokens before the question is even asked, so the token budget from [Lesson 4](../04-Connectors-and-Visual-Token-Compression/README.md) is what sets `k`. The strategies that follow are all ways to spend a fixed budget better:

- **Uniform sampling** — the default, and the one measured above.
- **Keyframe selection / scene-change detection** — spend frames where the video changes.
- **Aggressive per-frame compression** — a resampler down to 8–32 tokens per frame, trading spatial detail for temporal coverage. Often the right trade, and exactly the wrong one for reading text on screen.
- **Temporal pooling / merging** across adjacent frames (Qwen2-VL merges pairs of frames), exploiting the fact that consecutive frames are nearly identical.
- **Question-conditioned retrieval** — use the question to pick candidate frames first. The instruction-aware compression idea from [Lesson 4 §3](../04-Connectors-and-Visual-Token-Compression/README.md#3-measured-accuracy-vs-compression), applied along the time axis.

Everything genuinely temporal — ordering, causality, counting repetitions, "what happened just before" — sits on top of this and is only reachable once sampling is adequate. This is also why video benchmark numbers are so sensitive to evaluation protocol: change the frame count and you change the result without touching the model.

## 4. GUI and screen understanding

Screen understanding is where several of this lesson's constraints collide, and it is the fastest-moving capability area because it is what "computer-use" agents need.

- **Text-dense and high-resolution.** A 2560×1440 screenshot has small, sharp text everywhere; §2's argument applies at full force, which is why GUI models run at high resolution with tiling.
- **Precise grounding required.** "Click the third item in the dropdown" needs coordinates accurate enough to actually land on the element — §1's precision ceiling becomes a functional requirement rather than a metric.
- **The output is an action, not a description.** A GUI VLM emits `click(x, y)`, `type("...")`, `scroll(...)` — grounded function calls, which is [Phase 07 Lesson 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/05-Structured-Output-and-Function-Calling/README.md)'s structured output with pixel-grounded arguments.
- **Free high-quality training data.** Screenshots come with a DOM or accessibility tree that names every element and its exact box, so instruction data can be generated with exact answer keys and no annotation.

## 5. Referring and segmentation: where a plain VLM stops

Boxes are cheap to tokenize; pixel-accurate masks are not. Emitting a mask as coordinate tokens is possible (polygon vertices, as Florence-2 does) but coarse. The dominant approach instead hands the job off: LISA-style models emit a special `<SEG>` token whose hidden state is fed to a dedicated segmentation decoder (a SAM-family model), so the VLM does the language-and-reference reasoning and a specialist produces the pixels. This is a useful boundary marker for the whole lesson — the token-sequence trick covers an enormous amount of ground, and dense per-pixel output is where it runs out.

## Video Script Outline

1. The four capability families, and the claim that each has one binding constraint
2. Grounding without a detection head: coordinates as vocabulary
3. The three-curve table — floor, exact-match, and real error — and why they diverge
4. Grounded output as a hallucination mitigation
5. The resolution table: a 4px glyph legible at 32px and gone at 16px
6. Scaling that to a real document: 12px text in a 336px input is 2.5 pixels
7. The upstream fixes, all paid for in tokens
8. Charts and tables: structured output whose schema must be grounded in pixels
9. The video table, and the "accuracy given the frame was sampled" column
10. Budget strategies for video: keyframes, compression, merging, question-conditioned retrieval
11. GUIs: where high resolution, precise grounding, and action output all meet — with free training data
12. Segmentation: the boundary where token sequences stop being enough
13. Recap + preview of [Lesson 9](../09-Evaluating-VLMs/README.md)

## Further Reading

- Chen et al. (2022), *Pix2Seq: A Language Modeling Framework for Object Detection* (coordinates as tokens)
- Peng et al. (2023), *Kosmos-2: Grounding Multimodal Large Language Models to the World* (location tokens and grounded captioning)
- Kim et al. (2022), *OCR-free Document Understanding Transformer* (Donut) and Lee et al. (2022), *Pix2Struct* (high-resolution document encoders)
- Masry et al. (2022), *ChartQA* (chart reasoning, and rendered-data answer keys)
- Wang et al. (2024), *Qwen2-VL* (native dynamic resolution and frame merging for video)
- Cheng et al. (2024), *SeeClick* / Hong et al. (2023), *CogAgent* (GUI grounding and screen agents)
- Lai et al. (2023), *LISA: Reasoning Segmentation via Large Language Model* (handing pixels to a specialist decoder)
