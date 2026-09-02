# Multimodal LLMs

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `01-Multimodal-LLMs`

## Why this matters

Everything so far in this course — from [self-attention and multi-head attention](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) through [decoder-only architectures](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md) to [instruction tuning](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md) — has treated the input as a sequence of *text* tokens. But a decoder-only Transformer's core mechanism doesn't actually know or care what a "token" represents semantically; it only needs a sequence of vectors in a shared embedding space, plus positions. That single observation is the entire trick behind modern multimodal LLMs (GPT-4V/GPT-4o, Gemini, Claude's vision capability, LLaVA, Flamingo): if you can turn an image into a short sequence of vectors that live in the *same* space as text-token embeddings, the exact same causal self-attention machinery and the exact same next-token-prediction training objective from earlier phases can process "see this image and answer a question about it" without inventing any new core architecture. This lesson covers the two ideas that make that possible — CLIP-style contrastive alignment, and LLaVA-style visual instruction tuning — as the opening lesson of the course's final phase. From here, the phase continues into [Mixture of Experts, Advanced](../02-Mixture-of-Experts-Advanced/README.md).

## What this lesson covers

- Why vision-language models need a *shared* embedding space before anything else works
- CLIP: contrastive pretraining that aligns images and text via a symmetric InfoNCE loss
- What CLIP's aligned space enables: zero-shot classification and cross-modal retrieval
- LLaVA-style architecture: turning image patches into "visual tokens" an LLM can read
- Visual instruction tuning: training the model to actually follow instructions about images
- Where these ideas fall short, and what full end-to-end multimodal training adds on top

## 1. The core problem: images and text don't start out comparable

A piece of text, once tokenized, becomes a sequence of integers that a learned embedding table turns into vectors. An image is a grid of pixels — a fundamentally different kind of object, with no obvious "vocabulary." Before an LLM's attention layers can do anything useful with an image, two problems have to be solved:

1. **Turn the image into a sequence of vectors at all.** The standard tool, familiar from Vision Transformers (ViT), is to cut the image into fixed-size patches (e.g. 14x14 pixels), flatten and linearly project each patch, and treat the resulting sequence of patch vectors the way a Transformer treats a sequence of token embeddings — with a learned positional encoding added, exactly analogous to [Phase 02&#39;s positional encoding](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md) for text.
2. **Get those vectors to actually *mean* something in relation to text.** A randomly initialized patch projection produces vectors that live in some arbitrary vector space with no relationship to the LLM's text-token embedding space at all. Just feeding them into a language model's attention layers as-is would be like handing the model noise — the two modalities have to be *aligned* first.

CLIP solves problem 2 directly, by training an image encoder and a text encoder from scratch so that their output spaces are already aligned before either one ever meets a language model.

## 2. CLIP: contrastive language-image pretraining

CLIP (Radford et al., 2021) trains two separate encoders — a vision encoder `f_img` (a ViT or a CNN) and a text encoder `f_text` (a Transformer, in the same causal or bidirectional style as earlier phases) — with **no manually labeled classes at all**. The only supervision is naturally occurring (image, caption) pairs scraped from the web: 400 million of them, in the original paper.

The training signal is a **contrastive loss**. For a batch of `N` (image, text) pairs:

```
image_embeds = normalize( f_img(images) )     # (N, d), unit-length rows
text_embeds  = normalize( f_text(texts) )     # (N, d), unit-length rows

logits = (image_embeds @ text_embeds.T) * exp(temperature)   # (N, N) similarity matrix
```

Row `i`, column `j` of `logits` is the cosine similarity between image `i` and text `j`, scaled by a learned temperature. The **only** correct match for image `i` is text `i` (the caption it was actually paired with) — every other text in the batch is treated as a negative. This turns the problem into an `N`-way classification, solved with ordinary cross-entropy, in *both* directions and averaged (this is the "symmetric" part of the loss):

```
L_image_to_text = CrossEntropy( logits,   labels = [0, 1, ..., N-1] )   # rows: pick the right column
L_text_to_image = CrossEntropy( logits.T, labels = [0, 1, ..., N-1] )   # columns: pick the right row
L_CLIP = (L_image_to_text + L_text_to_image) / 2
```

This is exactly the InfoNCE contrastive loss, applied over the batch as its own set of negatives — no separate negative-sampling scheme is needed, since every non-matching pair in the batch is automatically a negative for both encoders simultaneously. Minimizing it pushes each image's embedding to be close (high cosine similarity) to its own caption's embedding and far from every other caption's embedding in the batch, and symmetrically for text. Note that gradients flow into *both* encoders from the same loss — the two towers are trained jointly, not one frozen against the other.

The direct payoff of this training scheme is that the two encoders end up sharing one embedding space, which enables two things without any further training:

- **Zero-shot classification**: turn class names into text prompts ("a photo of a {class}"), embed them all with `f_text`, embed a new image with `f_img`, and pick the class whose text embedding has the highest cosine similarity to the image — no classification head, no fine-tuning on that dataset's labels.
- **Cross-modal retrieval**: given a query in one modality, rank candidates in the other modality by cosine similarity in the shared space. `example.py` implements exactly this retrieval task on toy data, before and after contrastive training, to make the effect of the loss directly measurable.

## 3. From an aligned space to a multimodal LLM: LLaVA

CLIP gives you an aligned embedding space, but it is fundamentally a pair of *encoders* — it can score how well an image and text match, but it cannot hold a conversation, follow an instruction, or generate free-form text about an image. LLaVA (Liu et al., 2023, *Visual Instruction Tuning*) shows how to bolt a CLIP-style vision encoder onto a decoder-only LLM with very little new machinery:

```
image -> [frozen CLIP vision encoder] -> patch embeddings (N_patches, d_vision)
       -> [small trainable projection layer, e.g. one or two Linear layers]
       -> "visual tokens" (N_patches, d_model)     # now in the LLM's own embedding space

sequence fed to the LLM =
   [visual tokens] + [text token embeddings for the instruction/question]

output = DecoderOnlyLLM(sequence)   -- exactly the architecture from
                                        Phase 03 Lesson 1, unchanged
```

The three pieces:

- **A pretrained (often frozen) vision encoder** — typically CLIP's own vision tower, since it was already trained to produce vectors that are meaningfully aligned with text semantics. It turns the image into a fixed-length sequence of patch embeddings.
- **A small trainable projection layer** — the only genuinely new component. Its entire job is to map CLIP's vision-embedding space into the LLM's token-embedding space, so that a "visual token" and a normal text-token embedding are dimensionally and semantically compatible enough for the *same* attention layers to process both.
- **The decoder-only LLM itself**, usually pretrained and frozen or lightly fine-tuned, exactly as covered in [Phase 03 Lesson 1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md). It receives one interleaved sequence — visual tokens standing in for "what the image contains," followed by (or interspersed with) ordinary text-token embeddings for the user's question — and processes the whole thing autoregressively with ordinary causal self-attention. Nothing about the attention mechanism changes: a text token can attend back to a visual token exactly as it would attend to any earlier text token, because by construction they now live in the same space.

Training this system is a two-stage recipe, and the second stage is a direct application of [instruction tuning / SFT](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md):

1. **Feature alignment pretraining**: with the vision encoder and LLM both frozen, train *only* the small projection layer on (image, caption) pairs, so it learns to place visual tokens somewhere the frozen LLM can already interpret sensibly.
2. **Visual instruction tuning**: fine-tune the projection layer (and usually the LLM too) on a dataset of (image, instruction, response) triples — "here is an image, here is a question or command about it, here is the ideal answer" — using the same next-token-prediction cross-entropy loss over the response tokens that ordinary text-only SFT uses. This is precisely the instruction-tuning idea from Phase 05, just with some of the "instruction" now expressed as an image instead of purely as text.

The result is a model that can be shown an image it has never seen, along with a free-form instruction ("describe this," "what's wrong with this chart?," "read the text in this sign"), and generate a fluent, grounded, autoregressive text response — using the exact same generation loop from [Phase 02&#39;s mini-Transformer lesson](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#4-autoregressive-generation), just with a richer input sequence.

## 4. What this glosses over

- **Resolution and patch count**: real vision encoders produce hundreds of patch tokens per image, which is expensive to feed through a full LLM context window; production systems use tricks like patch merging, resampling (as in Flamingo's Perceiver Resampler, below), or dynamic tiling for high-resolution images.
- **Video and audio**: the same "encode to a sequence of vectors, project into the LLM's space" recipe generalizes to other modalities, though video adds a temporal dimension and audio has its own encoder families (e.g. Whisper-style encoders); the core idea does not change.
- **Interleaved, multi-image, and generation-of-images cases**: this lesson covers understanding images as input; some frontier systems also *generate* images or video as output, which requires an entirely different decoding mechanism (diffusion, or discrete image-token autoregression) not covered here.

## Video Script Outline

1. Motivation — attention only needs vectors in a shared space; multimodality is "just" building that space
2. The two sub-problems: turning pixels into a sequence, and making that sequence mean something relative to text
3. CLIP's contrastive setup: two encoders, one batch, a similarity matrix, a symmetric cross-entropy loss
4. What an aligned space buys you for free: zero-shot classification and retrieval
5. LLaVA's three pieces: frozen vision encoder, trainable projection, unchanged decoder-only LLM
6. Visual instruction tuning as SFT with images — connecting back to Phase 05
7. Walkthrough of `example.py` — toy CLIP-style training from scratch, measuring retrieval accuracy before vs. after
8. Recap + what's glossed over + preview of [Mixture of Experts, Advanced](../02-Mixture-of-Experts-Advanced/README.md)

## Further Reading

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (the CLIP paper)
- Liu, Li, Wu, Lee (2023), *Visual Instruction Tuning* (the LLaVA paper)
- Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning* (the Perceiver Resampler approach to bridging vision and language)
- Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (the Vision Transformer / patch-embedding scheme CLIP's vision tower and LLaVA both build on)
