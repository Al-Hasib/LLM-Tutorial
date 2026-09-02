# Transformer Encoder-Decoder Architecture

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `04-Transformer-Encoder-Decoder`

## Why this matters

You now have every individual piece: [tokenization](../01-Tokenization/README.md), [multi-head attention](../02-Self-Attention-and-Multi-Head-Attention/README.md), and [positional encoding](../03-Positional-Encoding/README.md). This lesson is where they get assembled into the complete architecture from Vaswani et al. (2017) — the original Transformer, built for machine translation, structured exactly like the [Seq2Seq model from Phase 01](../../Phase-01-Language-Modeling-Foundations/04-Seq2Seq-and-Attention/README.md#1-sequence-to-sequence-seq2seq) but with every recurrent layer replaced by attention.

## What this lesson covers

- The full encoder stack
- The full decoder stack, including **cross-attention**
- How cross-attention connects the two stacks
- The complete forward-pass data flow, end to end
- Where encoder-only, decoder-only, and encoder-decoder models diverge from this base (preview of Phase 03)

## 1. The encoder stack

The encoder's job: take the source sequence and produce a rich, contextualized representation of every token, where each token's representation has already "seen" every other token in the sequence (since encoder self-attention has no causal mask — it's bidirectional). One encoder layer is:

```
x = x + MultiHeadAttention(x, x, x)     # self-attention sublayer, residual added
x = LayerNorm(x)
x = x + FeedForward(x)                   # position-wise FFN sublayer, residual added
x = LayerNorm(x)
```

(The exact placement of `LayerNorm` — before vs. after each sublayer — is exactly what [Lesson 5](../05-LayerNorm-Residuals-FFN/README.md) covers in depth; the sketch above is the original paper's "Post-LN" version.) Stack `N` of these identical layers (the original paper used `N=6`) to get the full encoder.

## 2. The decoder stack

The decoder generates the target sequence one token at a time, and each decoder layer has **three** sublayers instead of two:

```
x = x + MaskedMultiHeadAttention(x, x, x)              # 1. causal self-attention over target-so-far
x = LayerNorm(x)
x = x + MultiHeadAttention(query=x, key=enc_out, value=enc_out)   # 2. cross-attention over the SOURCE
x = LayerNorm(x)
x = x + FeedForward(x)                                    # 3. position-wise FFN
x = LayerNorm(x)
```

Sublayer 1 uses the [causal mask from Lesson 2](../02-Self-Attention-and-Multi-Head-Attention/README.md#4-causal-masking) so the decoder can't peek at target tokens it hasn't generated yet. Sublayer 3 is identical in form to the encoder's FFN.

## 3. Cross-attention: where the two stacks meet

Sublayer 2 is the one genuinely new piece, and it's a direct generalization of [Bahdanau/Luong attention from Phase 01](../../Phase-01-Language-Modeling-Foundations/04-Seq2Seq-and-Attention/README.md#3-the-fix-attention):

```
CrossAttention: queries come from the DECODER, keys and values come from the ENCODER's final output
```

This is exactly the mechanism that lets a translation model's decoder, while generating each target word, look back at the *entire* source sentence and decide which source words are relevant *right now* — no more single fixed context vector, no bottleneck. It's the same attention math as self-attention; the only difference is *where* `Q` vs. `K`/`V` come from.

## 4. The full forward pass

```
source tokens -> [token embedding + positional encoding] -> Encoder (N layers) -> encoder_output
                                                                                        |
target tokens (shifted right) -> [token embedding + positional encoding] -> Decoder (N layers, cross-attends to encoder_output)
                                                                                        |
                                                                              Linear -> Softmax -> next-token probabilities
```

"Shifted right" means: at training time, the decoder's input at position `t` is the *true* target token at position `t-1` (this is called **teacher forcing**), and its causal mask ensures position `t`'s output only ever depended on positions `< t` of the target — so the whole target sequence can be trained in one parallel forward pass instead of one token at a time.

## 5. Where architectures diverge from here

This full encoder-decoder shape is one of three major patterns you'll see formalized in [Phase 03: LLM Architectures and Types](../../Phase-03-LLM-Architectures-and-Types/README.md):

- **Encoder-only** (BERT-style): keep only the encoder stack, train with masked-token prediction instead of next-token prediction — good for understanding tasks, not generation.
- **Decoder-only** (GPT-style): keep only the decoder stack, and simply *drop the cross-attention sublayer* (there's no separate source sequence — every position attends causally to earlier positions in the same sequence). This is the architecture nearly every modern general-purpose LLM uses, and exactly what [Lesson 6&#39;s mini-GPT](../06-Mini-Transformer-From-Scratch/README.md) builds.
- **Encoder-decoder** (T5/BART-style): keep the full architecture described in this lesson — still the natural fit for tasks with a clear input/output distinction, like translation or summarization.

## Video Script Outline

1. Motivation — "every piece exists, this is where they snap together"
2. Encoder stack: self-attention + FFN, residuals, N layers
3. Decoder stack: masked self-attention + cross-attention + FFN
4. Cross-attention explained as generalized Bahdanau/Luong attention
5. Full data flow diagram, teacher forcing explained
6. Walkthrough of `example.py` — a working encoder + decoder stack in PyTorch, toy forward pass
7. Recap + preview: encoder-only / decoder-only / encoder-decoder split in Phase 03

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.1 and Figure 1 (the canonical architecture diagram)
- Jay Alammar, *The Illustrated Transformer* — the full encoder-decoder walkthrough section
- Sasha Rush et al., *The Annotated Transformer* — the entire architecture implemented and explained line by line
