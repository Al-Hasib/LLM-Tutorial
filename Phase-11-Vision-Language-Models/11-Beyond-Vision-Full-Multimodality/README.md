# Beyond Vision: Full Multimodality

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `11-Beyond-Vision-Full-Multimodality`

## Why this matters

Ten lessons of this phase have been about one extra modality, chosen because it is the best-studied and the one with the most data. This final lesson asks what changes when there are four, or seven, and when the model has to *produce* non-text output rather than only consume it — the shift from "vision-language model" to "omni model." Three things turn out to be true, and each is measurable in a toy: alignment between two modalities can appear **without any paired data between them**, generation of any modality is **the same next-token objective** already used for text, and "aligned" does **not** mean the modalities occupy the same region of embedding space. Those three facts, plus everything earlier in the phase, are most of what separates a VLM from a Gemini- or GPT-4o-class multimodal system. This is the closing lesson of the course, and it is also the widest — the recipe generalizes far past images, and the constraints generalize with it.

## What this lesson covers

- Modalities beyond vision: audio, video, 3D/depth, and their encoders
- Why text is the anchor modality, and the `O(n)` vs `O(n²)` paired-data argument
- Measured: emergent cross-modal alignment (the ImageBind result), and its limits
- Any-to-any generation: discrete tokenization, and one objective for understanding *and* generation
- Measured: the same weights captioning and generating, and what happens when a direction is left out of the data
- Measured: where the "modality gap" comes from, and what it means practically
- Interleaved I/O, speech-to-speech, and the latency argument for native multimodality
- What stays hard

## 1. The recipe generalizes

[Phase 10 Lesson 1](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md) claimed that a Transformer only needs vectors in a shared space, and Lessons 1–4 of this phase built exactly that pipeline for images. Nothing in it is image-specific:

| Modality | Encoder | Token structure | Distinctive problem |
|---|---|---|---|
| **Image** | ViT on patches | `res²/patch²` tokens | resolution vs. token budget (Lessons 1, 8) |
| **Audio / speech** | log-mel spectrogram → conv + Transformer (Whisper-style) | ~50 tokens/second | one long axis; speech and non-speech audio want different things |
| **Video** | per-frame image encoder + temporal merging | frames × tokens/frame | sampling dominates everything (Lesson 8 §3) |
| **3D / depth / point cloud** | point or voxel encoders | irregular, no natural grid | very little paired data with text |
| **Time series / sensors** | patch or conv encoders | one long axis | almost no paired text at all |

Audio deserves one note because it is the most-deployed non-visual modality: **speech recognition and general audio understanding are different tasks**, and a token budget that suits one suits the other badly. Whisper-style encoders exist for transcription; music and environmental audio need tokens that preserve texture rather than phonemes. The same understanding-versus-detail tension as Lesson 2's contrastive-vs-captioning argument, in a new modality.

## 2. Text is the anchor: `O(n)` instead of `O(n²)`

With `n` modalities there are `n(n−1)/2` possible pairings, and paired training data for most of them does not exist — there is no web-scale corpus of (audio, depth) pairs. ImageBind's observation is that you do not need one. Bind every modality to a **single anchor** for which paired data *does* exist, and the rest follows.

`example.py` §1 tests this directly. It trains only `(text, image)`, `(text, audio)` and `(text, depth)` pairs, and **never** a single `(image, audio)` pair:

```
training data                       text->image  text->audio  image->audio  audio->depth
untrained encoders                        0.3%         0.5%          1.1%          0.5%
trained: text-image only                100.0%         1.3%          1.5%          0.6%
trained: text-X pairs only               99.5%        99.4%         99.6%         99.5%
trained: ALL pairs (upper bound)         99.7%        99.6%        100.0%         99.5%
```

Row 3 is the result: image→audio retrieval at 99.6% from a model that never saw the two modalities together. The mechanism is unglamorous — both were pulled toward the same text embeddings, and two things close to the same third thing are close to each other — but the consequence is large: paired-data cost scales as `O(n)` in the number of modalities rather than `O(n²)`.

Row 2 is the control that keeps the claim honest. A modality bound to nothing stays at chance for everything. Alignment propagates *through* the anchor and only to modalities actually bound to it, which is precisely why text is the anchor in practice: (text, anything) pairs exist on the web.

The real-world caveats: emergent alignment is weaker than direct training (row 4 is better, and by a larger margin at real scale), and it is bounded by the anchor's own expressiveness — anything a caption never describes cannot be transported through it, which is [Lesson 2 §4](../02-Vision-Language-Pretraining-Objectives/README.md#4-what-contrastive-pretraining-destroys)'s ceiling reappearing at the level of a whole modality.

## 3. Generation is next-token prediction

Every model in this phase so far consumes images and emits text. The change that makes a system genuinely "any-to-any" is startlingly small: **quantize the other modality into discrete tokens and put them in the same vocabulary.** A VQ-VAE/VQ-GAN tokenizer turns an image into a grid of codebook indices; those indices become vocabulary entries; the sequence `[BOI] <img tokens> [BOT] <text tokens>` and the reverse are both just sequences.

`example.py` §2 implements a miniature Chameleon: one decoder-only Transformer, one shared vocabulary, plain next-token cross-entropy, trained on both orderings.

```
trained on                    understanding   generation (per-token)   exact image
captioning order only                96.2%                     3.8%          0.0%
both orders                          96.2%                   100.0%        100.0%
chance                                0.7%                     6.2%          0.0%
```

The same weights that describe an image can draw one, and nothing in the architecture or the loss distinguishes the directions — "generate an image" is "predict the next token" where the next tokens happen to be image codes. Autoregressive image generation, interleaved image-and-text output, and speech output all follow from this one move.

The first row is the familiar warning in a new setting: a model trained only in the captioning direction cannot generate at all (3.8%, at chance). It has every part it needs and has never been asked to run them backwards. [Lesson 6](../06-Visual-Instruction-Tuning-and-VLM-Data/README.md)'s "capability follows the data" applies to output directions exactly as it applies to task types.

Two caveats the toy hides, both important in practice:

- **Quantization loses real detail.** Real image tokenizers discard high-frequency information, which is why token-based generation has historically trailed diffusion on image quality. Several current systems therefore keep a diffusion decoder for output while using tokens for understanding, or generate tokens that condition a diffusion model.
- **Mixed-modal training at scale is unstable.** Chameleon's paper is substantially about the norm-growth and divergence problems that appear when text and image tokens share a softmax; QK-norm and careful normalization placement were required, not optional.

## 4. The modality gap: aligned ≠ co-located

```
encoders             pair             cos matched  cos within  centroid dist
UNTRAINED            text-image            +0.009      +0.553          0.990
UNTRAINED            image-audio           -0.134      +0.449          1.114
text-anchored        text-image            +0.931      +0.010          0.132
text-anchored        image-audio           +0.924      +0.009          0.136
```

Read the untrained rows first. Two *random* concepts inside one modality have cosine ~0.5 — a randomly initialized deep encoder crams everything into a narrow cone — while the *same* concept across two modalities sits at ~0, and the modality centroids are nearly a full unit apart. None of that is about content; it is the geometry a random network starts with, and each encoder's cone points somewhere different.

That is the origin of the **modality gap** documented for CLIP-style models: contrastive learning does not create it, it inherits it and has to work against it. In this toy the training fully dissolves the cones, because the task is completely learnable; on real data, where alignment is never that complete and the temperature is clamped, a measurable gap survives — which is why the effect has a name.

Three practical consequences worth carrying out of the course:

- Absolute cross-modal similarity scores are not comparable to within-modality ones, so a cosine threshold tuned on one will be wrong on the other.
- Arithmetic that mixes modality embeddings (averaging an image vector and a text vector) can land in neither region and behave strangely.
- It is one reason a VLM trains a projector ([Lesson 3](../03-VLM-Architectures-and-Fusion-Strategies/README.md)) rather than feeding CLIP embeddings straight into an LLM. "Aligned" never meant "in the same place."

## 5. What native multimodality buys, beyond capability

Two arguments for building one model over all modalities rather than a pipeline of specialists, and neither is about benchmark scores:

**Latency.** A cascade of speech-to-text → LLM → text-to-speech pays three sequential model latencies and cannot start responding until transcription finishes. A model that consumes audio tokens and emits audio tokens can begin responding mid-utterance. For conversational speech, that difference is the product.

**Information that the intermediate representation drops.** Transcribing speech to text discards tone, emphasis, hesitation, accent, overlapping speakers, and background sound — everything that made it speech rather than a transcript. The same argument as [Lesson 4](../04-Connectors-and-Visual-Token-Compression/README.md)'s: whatever the intermediate representation omits is unrecoverable downstream, and a text bottleneck between two models is the most aggressive compression in the whole system.

Interleaved input and output is the other capability that only a unified model gets cleanly: a document containing text and figures in, a response containing text and generated diagrams out, in one sequence, in the right order.

## 6. What stays hard

- **Data.** Instruction data for non-visual modalities is far scarcer than for images, and the synthesis tricks from [Lesson 6](../06-Visual-Instruction-Tuning-and-VLM-Data/README.md) need a source of structured truth that often does not exist for audio or 3D.
- **Token budgets multiply.** Every constraint from [Lessons 4](../04-Connectors-and-Visual-Token-Compression/README.md) and [10](../10-VLM-Inference-and-Deployment/README.md) applies per modality and adds up. A minute of audio plus a minute of video plus a document is not a marginal prompt.
- **Modality competition.** At fixed capacity and a fixed data budget, adding a modality costs the others something; balancing a mixture across modalities is harder than balancing across tasks.
- **Evaluation.** Every pathology in [Lesson 9](../09-Evaluating-VLMs/README.md) has a multimodal analogue, and the blind-baseline problem gets worse: an audio benchmark can be solvable from a transcript, a video benchmark from a single frame. The control is the same — run the harness with the modality removed.
- **Generation quality vs. unification.** Whether one autoregressive model can match specialist diffusion generators is genuinely unsettled, and the current answer in production systems is usually "not yet, so keep the specialist decoder."

## Video Script Outline

1. From VLM to omni model: what actually changes and what doesn't
2. The encoder table — audio, video, 3D — and the recipe surviving intact
3. The `O(n²)` paired-data problem, and the anchor idea
4. The measured ImageBind result: image→audio retrieval with no image-audio data
5. The control row that keeps it honest — alignment propagates only through the anchor
6. Discrete tokenization: images as vocabulary entries
7. The mini-Chameleon result: one loss, understanding and generation, and the missing-direction row
8. Quantization loss and training instability — the caveats that keep diffusion decoders in production
9. The modality gap at initialization, and what survives training
10. Latency and lost information: why native multimodality wins outside benchmarks
11. What stays hard — data, budgets, competition, evaluation
12. Course wrap-up: the whole phase in one arc, from patch embeddings to omni models

## Further Reading

- Girdhar et al. (2023), *ImageBind: One Embedding Space To Bind Them All* (the emergent-alignment result measured in §2)
- Team Chameleon (2024), *Chameleon: Mixed-Modal Early-Fusion Foundation Models* (one vocabulary, one objective, plus the stability fixes)
- Radford et al. (2022), *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper; the standard audio encoder)
- Liang et al. (2022), *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning* (the gap's origin at initialization)
- Défossez et al. (2024), *Moshi: a speech-text foundation model for real-time dialogue* (the latency argument, built)
- Wu et al. (2023), *NExT-GPT: Any-to-Any Multimodal LLM* (any-to-any with specialist decoders per output modality)
