# Advanced and Frontier Topics

[← Back to curriculum index](../README.md)

Explore where LLM research is headed beyond the now-standard Transformer recipe.

## The path through this phase

Phases 02–09 built one specific recipe: a dense, decoder-only, text-only Transformer. Each lesson here breaks a different assumption in that sentence, and the lessons are independent — read them in any order.

```mermaid
flowchart TD
    C["the standard recipe:<br/>dense · decoder-only · text-only"] --> M["01 · multimodal LLMs<br/>drops “text-only”<br/>→ expanded into all of Phase 11"]
    C --> E["02 · Mixture of Experts, advanced<br/>drops “dense”"]
    C --> S["03 · state space models (Mamba)<br/>drops attention itself"]
    C --> ME["04 · model merging and editing<br/>drops “one training run,<br/>one model”"]
    C --> I["05 · interpretability<br/>drops treating the<br/>model as a black box"]
```

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [Multimodal LLMs](01-Multimodal-LLMs/README.md) |
| 02 | [Mixture of Experts, Advanced](02-Mixture-of-Experts-Advanced/README.md) |
| 03 | [State Space Models (Mamba)](03-State-Space-Models-Mamba/README.md) |
| 04 | [Model Merging and Editing](04-Model-Merging-and-Editing/README.md) |
| 05 | [Interpretability and Mechanistic Interpretability](05-Interpretability-and-Mechanistic-Interpretability/README.md) |
