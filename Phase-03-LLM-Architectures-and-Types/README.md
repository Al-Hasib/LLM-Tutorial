# LLM Architectures and Types

[← Back to curriculum index](../README.md)

Survey the major families of Transformer-based models and the design choices that define them.

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [Decoder-Only Models: the GPT Family](01-Decoder-Only-Models-GPT-Family/README.md) |
| 02 | [Encoder-Only Models: the BERT Family](02-Encoder-Only-Models-BERT-Family/README.md) |
| 03 | [Encoder-Decoder Models: T5 and BART](03-Encoder-Decoder-Models-T5-BART/README.md) |
| 04 | [Mixture of Experts](04-Mixture-of-Experts/README.md) |
| 05 | [Scaling Laws](05-Scaling-Laws/README.md) |
| 06 | [Long-Context Techniques](06-Long-Context-Techniques/README.md) |
| 07 | [Survey of Popular Open LLMs](07-Survey-of-Popular-Open-LLMs/README.md) |

## The three architecture families, side by side

Lessons 1-3 cover the three ways to arrange encoder and decoder stacks; Lesson 4 (MoE) is an orthogonal axis that can apply *inside* any of the three. Each lesson's README has a full ASCII diagram of its block under "Architecture at a glance" — this table is the quick comparison:

| | Decoder-only ([Lesson 1](01-Decoder-Only-Models-GPT-Family/README.md#architecture-at-a-glance)) | Encoder-only ([Lesson 2](02-Encoder-Only-Models-BERT-Family/README.md#architecture-at-a-glance)) | Encoder-decoder ([Lesson 3](03-Encoder-Decoder-Models-T5-BART/README.md#architecture-at-a-glance)) |
|---|---|---|---|
| Self-attention | Causal (left-to-right only) | Bidirectional (every position sees every other) | Bidirectional in encoder, causal in decoder |
| Cross-attention | None | None | Yes — decoder queries, encoder keys/values |
| Pretraining objective | Next-token prediction | Masked Language Modeling | Span corruption (T5) / denoising (BART) |
| Can generate open-ended text? | Yes | No | Yes, conditioned on an encoded input |
| Best suited for | Open-ended generation, chat, one-model-for-everything | Classification, embeddings, extractive QA | Translation, summarization — tasks with a clean input→output split |
| Examples | GPT-1/2/3, LLaMA, Mistral | BERT, RoBERTa | T5, BART |

`example.py` in each of Lessons 1-3 now builds and trains its architecture's actual model class from scratch in PyTorch, not just a description — see each lesson for the code and a training/generation run.

## Where model compression fits

Shrinking a trained model (distillation, pruning, quantization) is a *deployment* concern, not a fourth architecture family, so it's covered later, in **[Phase 09 Lesson 4: Model Distillation and Pruning](../Phase-09-Deployment-and-Inference-Optimization/04-Model-Distillation-and-Pruning/README.md)** — the mechanism behind DistilBERT (mentioned in [Lesson 2](02-Encoder-Only-Models-BERT-Family/README.md#6-roberta-and-todays-encoder-only-landscape)) and DistilGPT2. It applies to every architecture on this page: you can distill a decoder-only, encoder-only, or encoder-decoder teacher into a smaller student of the same family.
