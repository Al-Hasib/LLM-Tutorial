# Deployment and Inference Optimization

[← Back to curriculum index](../README.md)

Make a trained model fast, cheap, and reliable to actually serve in production.

## The path through this phase

Training is a one-off cost; inference is the bill that arrives every day. This phase starts from the hardware, because almost every optimization here is a response to one of two ceilings — arithmetic throughput or memory bandwidth — and knowing which one you are hitting decides which technique will help.

```mermaid
flowchart LR
    G["01 · GPU & hardware fundamentals<br/>compute-bound vs bandwidth-bound"] --> Q["02 · quantization"]
    G --> K["03 · KV cache and<br/>speculative decoding"]
    Q --> S["04 · serving frameworks<br/>vLLM · continuous batching · paged KV"]
    K --> S
    S --> D["05 · distillation and pruning"]
    S --> C["06 · cost and latency optimization"]
    S --> DE["07 · generation and decoding strategies"]
    C --> KC["08 · kernel and compiler optimization"]
    KC --> DI["09 · distributed inference at scale"]
    DI --> PB["10 · production serving<br/>and benchmarking"]
    PB --> F["11 · frontier inference systems"]
```

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [GPU and Hardware Fundamentals](01-GPU-and-Hardware-Fundamentals/README.md) |
| 02 | [Quantization](02-Quantization/README.md) |
| 03 | [KV Cache and Speculative Decoding](03-KV-Cache-and-Speculative-Decoding/README.md) |
| 04 | [Serving Frameworks](04-Serving-Frameworks/README.md) |
| 05 | [Model Distillation and Pruning](05-Model-Distillation-and-Pruning/README.md) |
| 06 | [Cost and Latency Optimization](06-Cost-and-Latency-Optimization/README.md) |
| 07 | [Generation and Decoding Strategies](07-Generation-and-Decoding-Strategies/README.md) |
| 08 | [Kernel and Compiler Optimization](08-Kernel-and-Compiler-Optimization/README.md) |
| 09 | [Distributed Inference at Scale](09-Distributed-Inference-at-Scale/README.md) |
| 10 | [Production Serving and Benchmarking](10-Production-Serving-and-Benchmarking/README.md) |
| 11 | [Frontier Inference Systems](11-Frontier-Inference-Systems/README.md) |
