# Vision-Language Pretraining Objectives

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `02-Vision-Language-Pretraining-Objectives`

## Why this matters

[Lesson 1](../01-Vision-Encoders-and-Image-Tokenization/README.md) built the machinery that turns pixels into a sequence of vectors, and ended on the observation that the *architecture* of the tower matters much less than what it was trained to do. This lesson is that "what." A vision tower's pretraining objective decides which facts about an image survive into its features and which are thrown away, and — because a VLM is usually built on a frozen or near-frozen tower — anything discarded here is gone for good, no matter how good the language model bolted on later is. [Phase 04 Lesson 2](../../Phase-04-Pretraining-LLMs/02-Pretraining-Objectives/README.md) made the same argument for text: the objective, not the architecture, is what determines what a pretrained model knows. Here the stakes are higher, because the objectives on offer differ far more from each other than causal LM differs from masked LM, and the most popular one — CLIP's contrastive loss — is provably lossy in a specific, measurable way.

## What this lesson covers

- Contrastive alignment: InfoNCE (CLIP) in detail, and why it is so batch-size hungry
- SigLIP's pairwise sigmoid loss, and what removing the global softmax buys
- The learned temperature and bias, and why both exist
- Generative objectives: captioning (CapPa, CoCa), and image-text matching with hard negatives (BLIP)
- Masked and self-supervised objectives (BEiT-3, FLAVA, DINOv2) as the no-language alternative
- Measured: what a contrastive tower *destroys* that a captioning tower keeps
- Data, not loss: alt-text, filtering, and the re-captioning trick that fixed more than any objective change

## 1. Contrastive alignment: InfoNCE

[Phase 10 Lesson 1 §2](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#2-clip-contrastive-language-image-pretraining) introduced CLIP's symmetric InfoNCE loss; this section is about its consequences. For a batch of `N` (image, text) pairs, both encoders emit unit-norm vectors, an `N × N` similarity matrix is formed, and cross-entropy is applied along both axes with the diagonal as the correct labels:

```
logits = exp(t) * image_embeds @ text_embeds.T          # (N, N)
L = ½ · [ CE(logits, arange(N)) + CE(logitsᵀ, arange(N)) ]
```

The crucial structural property is that the softmax is normalized **across the batch**. Every other item in the batch is a negative, so a single training example's task is "pick the right caption out of `N`" — and an `N`-way classification carries at most `log N` nats of information. At `N = 4` the task is trivial; the encoders solve it with a coarse representation and then have nothing left to learn. At `N = 32,768` (CLIP's actual batch size) the task is genuinely hard, and the pressure to build a fine-grained representation persists all the way through training.

`example.py` §1 measures exactly this. Eight encoders are trained on the *same* number of examples (only the batch size, and therefore the step count, differs) and scored by held-out 256-way retrieval:

```
  batch   steps   negatives/anchor   InfoNCE R@1   Sigmoid R@1
      4    8000                  3        63.5%        81.4%
     16    2000                 15        76.8%        96.7%
     64     500                 63        97.0%        97.9%
    256     125                255        98.1%        97.2%
```

InfoNCE climbs from 63.5% to 98.1% purely as a function of how many negatives each step sees, with the data budget held fixed. That single fact drives an enormous amount of real engineering: contrastive training runs shard the batch across hundreds of accelerators and **all-gather** every embedding, so each device's softmax sees the global batch. The `32,768²` logits matrix is 2.1 GB in fp16 all by itself, so production implementations compute it in chunks rather than materializing it.

## 2. SigLIP: dropping the softmax

SigLIP (Zhai et al., 2023) replaces the batch-wise softmax with an independent **sigmoid** on every pair — a binary "do these two go together?" classification on all `N²` cells:

```
logits  = exp(t) * image_embeds @ text_embeds.T + b
targets = +1 on the diagonal, −1 everywhere else
L = − Σ log σ(targets ⊙ logits) / N
```

Nothing is normalized across the batch, so nothing has to be gathered across devices, and the loss stays well-posed at small `N`. In `example.py`'s table, the sigmoid loss reaches 81.4% at batch 4 where InfoNCE manages 63.5%, and the two converge as the batch grows. That matches the paper's headline result at real scale: SigLIP matches or beats CLIP while being far less batch-hungry, which is why it has become the default tower in most new open VLMs.

The `+ b` term is not decoration. `N² − N` of the `N²` pairs are negatives, so at initialization "predict *no match* everywhere" is an excellent strategy; a learned, strongly negative bias absorbs that prior so the gradient signal on the `N` positives isn't swamped. `example.py` §2 shows it settling around −10 to −11 in every run.

The learned **temperature** `exp(t)` is common to both losses and equally load-bearing. Cosine similarities are confined to `[−1, 1]`; pushed through a softmax directly they give a nearly uniform distribution and vanishing gradients regardless of encoder quality. So the scale is a trained parameter, growing until similarities are sharp enough to be informative — CLIP clamps it at 100 for stability, and so does the example script.

## 3. Generative and matching objectives

Contrastive alignment is one family. The others:

| Objective | Signal per example | Used by | What it's good at |
|---|---|---|---|
| **Contrastive (ITC)** | one similarity score per pair | CLIP, SigLIP, ALIGN | retrieval, zero-shot classification, cheap to scale |
| **Captioning (LM loss)** | one cross-entropy per caption token | CoCa, CapPa, PaLI, GIT | fine-grained detail, directly reusable as a VLM head |
| **Image-Text Matching (ITM)** | binary match/no-match with *hard* negatives, using cross-attention | BLIP, ALBEF | fine distinctions contrastive scoring blurs |
| **Masked modeling** | per-masked-token reconstruction | BEiT-3, FLAVA, MAE | dense spatial features; no captions needed |
| **Self-distillation** | agreement between augmented views | DINOv2 | strongest dense/spatial features, zero language |

The information-per-example difference is stark: contrastive supervision gives one number per pair, while captioning gives a full cross-entropy per token. That is the mechanism behind CoCa's design (one tower, both losses) and behind the finding that captioning-pretrained towers transfer better to VQA-style tasks even when they retrieve worse.

BLIP's ITM head is worth a note because it fixes a specific contrastive failure. Contrastive scoring compares two pooled vectors with a dot product — it can never model *interaction* between specific words and specific regions. ITM feeds both modalities through cross-attention and asks a binary question, using the contrastive scores to mine the hardest negatives in the batch (the mismatched pairs the contrastive head already thinks are similar). That is where "a man riding a horse" vs "a horse riding a man" starts to become distinguishable.

## 4. What contrastive pretraining destroys

The most important practical fact about CLIP-style pretraining is that its lossiness is not incidental — it is what the objective asks for. `example.py` §3 makes it measurable. Every synthetic image carries a **category** and an **attribute**; every caption names the category only, exactly like real web alt-text ("a dog", not "a small brown dog facing left"). Three image towers are trained, frozen, and linear-probed:

```
image tower trained with            category  attribute  within/between
contrastive (InfoNCE)                100.0%     55.4%          0.0060
captioning, category only            100.0%     87.5%          0.0310
captioning, category + attribute     100.0%    100.0%          0.5694
chance                                 8.3%     25.0%
```

All three learn the category perfectly. They differ on the attribute *no caption ever mentions*, and the third column says why: contrastive training's target for an image is its caption's embedding, and every image of a category shares one caption, so the loss actively **collapses** them onto a single point. Its within-category spread is 5× smaller than the captioner's, and the attribute degrades with it. The captioner is supervised on exactly the same category label, yet retains far more of the attribute — a generative head has no incentive to destroy information it happens not to use.

Scale this up and it is the reason CLIP towers are weak on counting, small text, precise spatial relations, and fine attributes: those are precisely the things a one-line web caption leaves out, so a loss trained to reproduce web captions is trained to discard them. The field's responses are all visible in current models — mixing objectives (CoCa), fusing a contrastive tower with a self-supervised one such as DINOv2, or attacking the data instead of the loss.

## 5. Data beats objective

The single largest quality jump in open VLM pretraining came not from a new loss but from **re-captioning**: running an existing captioner over the pretraining corpus to replace short, noisy alt-text with dense synthetic descriptions (as in LLaVA's and DALL·E 3's data pipelines, and the "recaption everything" result reported for many later models). If the objective's ceiling is set by what the captions mention, richer captions raise the ceiling without touching the loss at all.

The rest of the data story is filtering. LAION-400M/5B were built by keeping pairs whose CLIP similarity exceeded a threshold — which quietly bakes an older CLIP model's biases into the next one's training set. DataComp reframed the whole thing as a benchmark where the *dataset* is the submission and the training recipe is fixed, and its results are unambiguous: at fixed compute, filtering strategy moves final accuracy more than most architectural or objective changes do. Deduplication against downstream eval sets matters too, for the same reason [Phase 04 Lesson 1](../../Phase-04-Pretraining-LLMs/01-Pretraining-Data-Pipeline/README.md) gives for text — and it is routinely done badly, which [Lesson 9](../09-Evaluating-VLMs/README.md) revisits as a benchmark-contamination problem.

## Video Script Outline

1. Motivation — the tower's objective decides what survives into the features; the LLM cannot recover what was thrown away
2. InfoNCE recap, and the `log N` information ceiling of a batch-wise softmax
3. The measured batch-size curve at fixed data budget: 63.5% → 98.1%
4. What that costs in engineering: all-gather, sharded softmax, a 2.1 GB logits matrix
5. SigLIP: independent sigmoids, no all-gather, and why the bias term is necessary
6. The learned temperature — cosine similarities are too flat for a softmax without it
7. The objective zoo: contrastive vs captioning vs ITM vs masked vs self-distillation, and information per example
8. The collapse experiment: contrastive destroys what captions omit, measured three ways
9. Data beats objective — re-captioning, CLIP-score filtering, DataComp
10. Recap + preview of [Lesson 3](../03-VLM-Architectures-and-Fusion-Strategies/README.md): how these features get into a language model

## Further Reading

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Zhai, Mustafa, Kolesnikov, Beyer (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP)
- Yu et al. (2022), *CoCa: Contrastive Captioners are Image-Text Foundation Models* (both objectives in one tower)
- Li, Li, Xiong, Hoi (2022), *BLIP: Bootstrapping Language-Image Pre-training* (ITC + ITM with hard negatives + captioning)
- Tschannen et al. (2023), *Image Captioners Are Scalable Vision Learners Too* (CapPa; captioning as a competitive pretraining objective)
- Gadre et al. (2023), *DataComp: In search of the next generation of multimodal datasets* (the dataset as the variable)
