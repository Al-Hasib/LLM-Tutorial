# Fine-tuning LLMs

[← Back to curriculum index](../README.md)

Adapt a pretrained base model to specific tasks and instructions efficiently.

## The path through this phase

A pretrained base model predicts plausible continuations; it does not answer questions. This phase is about closing that gap cheaply — first the methods, then the tooling, then a worked case study.

```mermaid
flowchart LR
    C["01 · full fine-tuning vs PEFT<br/>the memory arithmetic that<br/>rules out the obvious approach"] --> L["02 · LoRA and QLoRA<br/>low-rank adapters"]
    C --> P["03 · prompt · prefix · adapter tuning<br/>the other PEFT families"]
    L --> S["04 · instruction tuning (SFT)<br/>teaching a base model to follow instructions"]
    P --> S
    S --> H["05 · doing it for real<br/>Hugging Face PEFT + TRL"]
    H --> CS["06 · a domain-specific case study"]
```

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [Full Fine-tuning vs Parameter-Efficient Fine-tuning](01-Full-Finetuning-vs-PEFT/README.md) |
| 02 | [LoRA and QLoRA](02-LoRA-and-QLoRA/README.md) |
| 03 | [Prompt Tuning, Prefix Tuning and Adapters](03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md) |
| 04 | [Instruction Tuning (SFT)](04-Instruction-Tuning-SFT/README.md) |
| 05 | [Fine-tuning with Hugging Face (PEFT + TRL)](05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) |
| 06 | [Domain-Specific Fine-tuning Case Study](06-Domain-Specific-Finetuning-Case-Study/README.md) |
