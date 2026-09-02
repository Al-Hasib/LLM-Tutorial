# LLM Tutorial — Beginner to Advanced

A full YouTube playlist curriculum for teaching Large Language Models from the ground up: math/DL prerequisites → classic NLP → Transformer internals → LLM architectures & types → pretraining → fine-tuning → alignment/RLHF → prompting → evaluation → deployment/inference optimization → frontier research.

This course is scoped to **the LLM itself** — its math, architecture, training, alignment, evaluation, and inference internals. It deliberately excludes building applications on top of an LLM (RAG pipelines, agents, chatbots, etc.) so every lesson stays focused on the model, not on software built around it.

## How this repo is organized

- Each **`Phase-XX-*`** folder is one stage of the course, taught in order.
- Each phase contains numbered **topic folders** (`01-...`, `02-...`, …), taught in order within the phase.
- Every topic folder = **one video**, and contains:
  - `README.md` — the lesson doc / video script: theory, explanations, and a script outline.
  - `example.py` — a runnable, commented code demo for that lesson.
- Early architecture/foundation topics implement things **from scratch** in raw PyTorch (attention, transformer blocks, a mini-GPT, a small pretraining run) so the internals are never a black box. Later applied phases (fine-tuning, deployment) switch to **industry-standard libraries** (Hugging Face `transformers`/`peft`/`trl`, `vLLM`, etc.) to reflect real-world practice.
- Status: the full folder/file structure is scaffolded now; lesson content is being written in **phase-by-phase** afterward. A topic whose `README.md` still says "Content for this lesson is not yet written" hasn't been filled in yet.

## Curriculum Index

### [Phase 00 — Prerequisites](Phase-00-Prerequisites/README.md)
| # | Topic |
|---|-------|
| 01 | [Python and Math Refresher](Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md) |
| 02 | [Neural Networks Basics](Phase-00-Prerequisites/02-Neural-Networks-Basics/README.md) |
| 03 | [Introduction to NLP](Phase-00-Prerequisites/03-Intro-to-NLP/README.md) |
| 04 | [PyTorch Fundamentals](Phase-00-Prerequisites/04-PyTorch-Fundamentals/README.md) |

### [Phase 01 — Language Modeling Foundations](Phase-01-Language-Modeling-Foundations/README.md)
| # | Topic |
|---|-------|
| 01 | [What is a Language Model](Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md) |
| 02 | [Word Embeddings](Phase-01-Language-Modeling-Foundations/02-Word-Embeddings/README.md) |
| 03 | [RNNs, LSTMs and GRUs](Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md) |
| 04 | [Sequence-to-Sequence and Attention](Phase-01-Language-Modeling-Foundations/04-Seq2Seq-and-Attention/README.md) |
| 05 | [Introduction to Transformers](Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md) |

### [Phase 02 — Transformer Architecture Deep Dive](Phase-02-Transformer-Architecture-Deep-Dive/README.md)
| # | Topic |
|---|-------|
| 01 | [Tokenization](Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md) |
| 02 | [Self-Attention and Multi-Head Attention](Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) |
| 03 | [Positional Encoding](Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md) |
| 04 | [Transformer Encoder-Decoder Architecture](Phase-02-Transformer-Architecture-Deep-Dive/04-Transformer-Encoder-Decoder/README.md) |
| 05 | [Layer Norm, Residuals and Feed-Forward Sublayers](Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md) |
| 06 | [Building a Mini-Transformer / Mini-GPT From Scratch](Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) |
| 07 | [Efficient Attention: FlashAttention, Sparse and Linear Attention](Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md) |

### [Phase 03 — LLM Architectures and Types](Phase-03-LLM-Architectures-and-Types/README.md)
| # | Topic |
|---|-------|
| 01 | [Decoder-Only Models: the GPT Family](Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md) |
| 02 | [Encoder-Only Models: the BERT Family](Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md) |
| 03 | [Encoder-Decoder Models: T5 and BART](Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md) |
| 04 | [Mixture of Experts](Phase-03-LLM-Architectures-and-Types/04-Mixture-of-Experts/README.md) |
| 05 | [Scaling Laws](Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) |
| 06 | [Long-Context Techniques](Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md) |
| 07 | [Survey of Popular Open LLMs](Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md) |

### [Phase 04 — Pretraining LLMs](Phase-04-Pretraining-LLMs/README.md)
| # | Topic |
|---|-------|
| 01 | [Pretraining Data Pipeline](Phase-04-Pretraining-LLMs/01-Pretraining-Data-Pipeline/README.md) |
| 02 | [Pretraining Objectives](Phase-04-Pretraining-LLMs/02-Pretraining-Objectives/README.md) |
| 03 | [Distributed Training Basics](Phase-04-Pretraining-LLMs/03-Distributed-Training-Basics/README.md) |
| 04 | [Mixed Precision and Optimization](Phase-04-Pretraining-LLMs/04-Mixed-Precision-and-Optimization/README.md) |
| 05 | [Pretraining a Small LLM From Scratch](Phase-04-Pretraining-LLMs/05-Pretraining-a-Small-LLM-From-Scratch/README.md) |

### [Phase 05 — Fine-tuning LLMs](Phase-05-Finetuning-LLMs/README.md)
| # | Topic |
|---|-------|
| 01 | [Full Fine-tuning vs Parameter-Efficient Fine-tuning](Phase-05-Finetuning-LLMs/01-Full-Finetuning-vs-PEFT/README.md) |
| 02 | [LoRA and QLoRA](Phase-05-Finetuning-LLMs/02-LoRA-and-QLoRA/README.md) |
| 03 | [Prompt Tuning, Prefix Tuning and Adapters](Phase-05-Finetuning-LLMs/03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md) |
| 04 | [Instruction Tuning (SFT)](Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md) |
| 05 | [Fine-tuning with Hugging Face (PEFT + TRL)](Phase-05-Finetuning-LLMs/05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) |
| 06 | [Domain-Specific Fine-tuning Case Study](Phase-05-Finetuning-LLMs/06-Domain-Specific-Finetuning-Case-Study/README.md) |

### [Phase 06 — Alignment and RLHF](Phase-06-Alignment-and-RLHF/README.md)
| # | Topic |
|---|-------|
| 01 | [The Alignment Problem](Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md) |
| 02 | [Reward Modeling](Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md) |
| 03 | [RLHF with PPO](Phase-06-Alignment-and-RLHF/03-RLHF-with-PPO/README.md) |
| 04 | [Direct Preference Optimization (DPO)](Phase-06-Alignment-and-RLHF/04-Direct-Preference-Optimization-DPO/README.md) |
| 05 | [RLAIF and Constitutional AI](Phase-06-Alignment-and-RLHF/05-RLAIF-and-Constitutional-AI/README.md) |
| 06 | [Safety, Bias and Toxicity Mitigation](Phase-06-Alignment-and-RLHF/06-Safety-Bias-and-Toxicity-Mitigation/README.md) |
| 07 | [Reasoning Models and GRPO](Phase-06-Alignment-and-RLHF/07-Reasoning-Models-and-GRPO/README.md) |

### [Phase 07 — Prompt Engineering and In-Context Learning](Phase-07-Prompt-Engineering-and-In-Context-Learning/README.md)
| # | Topic |
|---|-------|
| 01 | [Prompting Basics: Zero-Shot and Few-Shot](Phase-07-Prompt-Engineering-and-In-Context-Learning/01-Prompting-Basics-Zero-Few-Shot/README.md) |
| 02 | [Chain-of-Thought and Reasoning Prompts](Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md) |
| 03 | [Tree-of-Thought and ReAct](Phase-07-Prompt-Engineering-and-In-Context-Learning/03-Tree-of-Thought-and-ReAct/README.md) |
| 04 | [Automatic Prompt Optimization](Phase-07-Prompt-Engineering-and-In-Context-Learning/04-Automatic-Prompt-Optimization/README.md) |
| 05 | [Structured Output and Function Calling](Phase-07-Prompt-Engineering-and-In-Context-Learning/05-Structured-Output-and-Function-Calling/README.md) |

### [Phase 08 — Evaluation of LLMs](Phase-08-Evaluation-of-LLMs/README.md)
| # | Topic |
|---|-------|
| 01 | [Evaluation Metrics](Phase-08-Evaluation-of-LLMs/01-Evaluation-Metrics/README.md) |
| 02 | [Standard Benchmarks](Phase-08-Evaluation-of-LLMs/02-Standard-Benchmarks/README.md) |
| 03 | [LLM-as-a-Judge](Phase-08-Evaluation-of-LLMs/03-LLM-as-a-Judge/README.md) |
| 04 | [Hallucination and Factuality Evaluation](Phase-08-Evaluation-of-LLMs/04-Hallucination-and-Factuality-Evaluation/README.md) |
| 05 | [Human Evaluation Methodologies](Phase-08-Evaluation-of-LLMs/05-Human-Evaluation-Methodologies/README.md) |
| 06 | [VLM-as-a-Judge](Phase-08-Evaluation-of-LLMs/06-VLM-as-a-Judge/README.md) |

### [Phase 09 — Deployment and Inference Optimization](Phase-09-Deployment-and-Inference-Optimization/README.md)
| # | Topic |
|---|-------|
| 01 | [GPU and Hardware Fundamentals](Phase-09-Deployment-and-Inference-Optimization/01-GPU-and-Hardware-Fundamentals/README.md) |
| 02 | [Quantization](Phase-09-Deployment-and-Inference-Optimization/02-Quantization/README.md) |
| 03 | [KV Cache and Speculative Decoding](Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md) |
| 04 | [Serving Frameworks](Phase-09-Deployment-and-Inference-Optimization/04-Serving-Frameworks/README.md) |
| 05 | [Model Distillation and Pruning](Phase-09-Deployment-and-Inference-Optimization/05-Model-Distillation-and-Pruning/README.md) |
| 06 | [Cost and Latency Optimization](Phase-09-Deployment-and-Inference-Optimization/06-Cost-and-Latency-Optimization/README.md) |

### [Phase 10 — Advanced and Frontier Topics](Phase-10-Advanced-and-Frontier-Topics/README.md)
| # | Topic |
|---|-------|
| 01 | [Multimodal LLMs](Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md) |
| 02 | [Mixture of Experts, Advanced](Phase-10-Advanced-and-Frontier-Topics/02-Mixture-of-Experts-Advanced/README.md) |
| 03 | [State Space Models (Mamba)](Phase-10-Advanced-and-Frontier-Topics/03-State-Space-Models-Mamba/README.md) |
| 04 | [Model Merging and Editing](Phase-10-Advanced-and-Frontier-Topics/04-Model-Merging-and-Editing/README.md) |
| 05 | [Interpretability and Mechanistic Interpretability](Phase-10-Advanced-and-Frontier-Topics/05-Interpretability-and-Mechanistic-Interpretability/README.md) |

---

**Totals:** 11 phases · 59 topics/videos.
