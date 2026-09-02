"""
GPU and Hardware Fundamentals

Two parts:

  A. A real, computable arithmetic-intensity calculator (README section 5),
     applied to Transformer-shaped matmuls to show, with real numbers, why
     PREFILL lands compute-bound and DECODE lands memory-bound (README
     section 6) against an honestly-approximate real GPU's ridge point.
     Also sweeps decode's arithmetic intensity across batch size, showing
     concretely how many concurrent decode requests it takes to cross back
     over the ridge point -- the real justification for continuous batching
     (Phase 09 Lesson 4).

  B. A CPU-measurable ANALOGY (explicitly labeled as an analogy, not a GPU
     reproduction) for the same qualitative effect: one big batched matmul
     vs. many separate single-row matmuls, timed for real on whatever CPU
     this script runs on. This machine has no discrete GPU to measure HBM/
     SRAM traffic on directly -- CPU cache levels (L1/L2/L3) play a loosely
     analogous role to a GPU's SRAM/HBM split, at a much smaller scale and
     with different numbers, but the SAME qualitative shape: reusing a
     weight matrix across many rows of work is dramatically more efficient,
     per row, than paying to touch it once per row.

Run:
    python example.py
"""

import time
import torch

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# PART A: ARITHMETIC INTENSITY, COMPUTED FOR REAL, FOR PREFILL VS. DECODE
# ---------------------------------------------------------------------------

BYTES_PER_ELEM = 2  # bf16/fp16 -- the standard training/inference dtype (Phase 04 Lesson 4)

# An honestly-approximate, clearly-labeled reference ridge point for a modern
# data-center GPU's bf16 tensor cores: peak FLOPs / peak HBM bandwidth. Real
# chips vary; this is a round, order-of-magnitude stand-in (roughly what an
# NVIDIA A100 works out to: ~312 TFLOPs bf16 / ~2 TB/s HBM bandwidth), not a
# precise spec for any one product.
PEAK_TFLOPS = 300.0          # approx., bf16 tensor-core peak, TFLOPs/s
PEAK_BANDWIDTH_TB_S = 2.0    # approx., HBM bandwidth, TB/s
RIDGE_POINT = (PEAK_TFLOPS * 1e12) / (PEAK_BANDWIDTH_TB_S * 1e12)  # FLOPs/byte


def matmul_flops(m, k, n):
    """FLOPs for an (m, k) @ (k, n) matmul (README section 4)."""
    return 2 * m * k * n


def matmul_bytes(m, k, n, bytes_per_elem=BYTES_PER_ELEM):
    """Bytes moved for an (m, k) @ (k, n) matmul: read the activations, read
    the weights, write the output. This is a naive DRAM-traffic model (every
    matrix is assumed to make one full trip to/from HBM) -- exactly the kind
    of coarse accounting a roofline analysis uses, and exactly what a
    non-tiled, unfused kernel actually does. A tiled kernel (like Phase 02
    Lesson 7's FlashAttention) can do better than this for SOME operations;
    ordinary dense matmuls on most inference stacks do not."""
    activations = m * k
    weights = k * n
    output = m * n
    return (activations + weights + output) * bytes_per_elem


def arithmetic_intensity(m, k, n, bytes_per_elem=BYTES_PER_ELEM):
    flops = matmul_flops(m, k, n)
    bts = matmul_bytes(m, k, n, bytes_per_elem)
    return flops / bts, flops, bts


def classify(ai):
    return "COMPUTE-BOUND" if ai > RIDGE_POINT else "MEMORY-BOUND"


def prefill_vs_decode_demo():
    print("=" * 90)
    print("PART A.1: ARITHMETIC INTENSITY -- PREFILL vs. DECODE, ON A REAL FFN-SHAPED MATMUL")
    print("=" * 90)
    d_model = 4096
    d_ff = 4 * d_model   # the FFN up-projection shape (Phase 02 Lesson 5 section 5)
    prefill_seq_len = 2048

    print(f"Weight matrix: ({d_model}, {d_ff})  [one FFN up-projection, bf16]")
    print(f"Reference ridge point: {PEAK_TFLOPS:.0f} TFLOPs / {PEAK_BANDWIDTH_TB_S:.1f} TB/s "
          f"= {RIDGE_POINT:.1f} FLOPs/byte (approx., a modern data-center GPU)\n")

    ai_prefill, flops_prefill, bytes_prefill = arithmetic_intensity(prefill_seq_len, d_model, d_ff)
    ai_decode, flops_decode, bytes_decode = arithmetic_intensity(1, d_model, d_ff)

    header = f"{'phase':10}{'m (tokens)':>12}{'FLOPs':>16}{'bytes moved':>16}{'AI (FLOPs/B)':>16}{'classification':>18}"
    print(header)
    print("-" * len(header))
    print(f"{'prefill':10}{prefill_seq_len:>12,}{flops_prefill:>16,.3g}{bytes_prefill:>16,}"
          f"{ai_prefill:>16,.1f}{classify(ai_prefill):>18}")
    print(f"{'decode':10}{1:>12,}{flops_decode:>16,.3g}{bytes_decode:>16,}"
          f"{ai_decode:>16,.3f}{classify(ai_decode):>18}")

    print(f"\n-> Prefill's arithmetic intensity ({ai_prefill:,.0f} FLOPs/byte) sits ~"
          f"{ai_prefill / RIDGE_POINT:.0f}x ABOVE the ridge point -- comfortably compute-bound.")
    print(f"   Decode's ({ai_decode:.3f} FLOPs/byte) sits ~{RIDGE_POINT / ai_decode:.0f}x BELOW it --")
    print(f"   deeply memory-bound. Note decode's AI lands almost exactly at "
          f"2/{BYTES_PER_ELEM} = {2 / BYTES_PER_ELEM:.2f}:")
    print(f"   with only ONE token's worth of activations, the weight matrix (which dominates the")
    print(f"   byte count) contributes almost exactly one multiply-add (2 FLOPs) per")
    print(f"   {BYTES_PER_ELEM}-byte element it holds -- there is structurally no way for a single")
    print(f"   token's decode step to do more arithmetic per byte than that, no matter the model size.")


def decode_batching_sweep_demo():
    print("\n" + "=" * 90)
    print("PART A.2: HOW MANY CONCURRENT DECODE REQUESTS TO CROSS BACK OVER THE RIDGE POINT?")
    print("=" * 90)
    d_model = 4096
    d_ff = 4 * d_model
    print("Same FFN weight matrix, but now m = number of CONCURRENT decode requests batched")
    print("together in one matmul (each contributing exactly one token) -- exactly what")
    print("continuous batching (Lesson 4) does.\n")

    batch_sizes = [1, 8, 32, 64, 128, 192, 256, 512]
    header = f"{'batch size (m)':>16}{'AI (FLOPs/byte)':>20}{'classification':>18}"
    print(header)
    print("-" * len(header))
    crossed = None
    for b in batch_sizes:
        ai, _, _ = arithmetic_intensity(b, d_model, d_ff)
        label = classify(ai)
        print(f"{b:>16}{ai:>20,.1f}{label:>18}")
        if crossed is None and ai > RIDGE_POINT:
            crossed = b

    # Binary-search the approximate exact crossover batch size for a concrete number.
    lo, hi = 1, 4096
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ai_mid, _, _ = arithmetic_intensity(mid, d_model, d_ff)
        if ai_mid > RIDGE_POINT:
            hi = mid
        else:
            lo = mid
    print(f"\n-> Crosses from memory-bound to compute-bound somewhere around batch size ~{hi} "
          f"concurrent decode requests")
    print(f"   (at this FFN shape and reference ridge point). This is exactly why real LLM serving")
    print(f"   systems need surprisingly LARGE numbers of concurrent requests batched together")
    print(f"   before decode stops being memory-bound -- and exactly the gap continuous batching")
    print(f"   (Lesson 4: Serving Frameworks) is built to keep filled with real, waiting traffic.")


# ---------------------------------------------------------------------------
# PART B: A CPU-MEASURABLE ANALOGY (explicitly not a GPU reproduction)
# ---------------------------------------------------------------------------

def cpu_batching_analogy_demo():
    print("\n" + "=" * 90)
    print("PART B: A CPU-MEASURABLE ANALOGY -- BATCHED vs. ONE-ROW-AT-A-TIME MATMULS")
    print("=" * 90)
    print("HONEST CAVEAT: this machine has no discrete GPU to measure HBM/SRAM traffic on.")
    print("What follows is an ANALOGY, not a reproduction: a CPU's cache hierarchy (L1/L2/L3)")
    print("plays a loosely similar role to a GPU's SRAM-vs-HBM split -- much smaller gap, very")
    print("different absolute numbers -- but the same QUALITATIVE effect shows up: reusing a")
    print("weight matrix across many rows of work in one call is far more efficient per row than")
    print("paying to touch that same weight matrix once per row, one row at a time.\n")

    d = 2048
    n_tokens = 4096
    weight = torch.randn(d, d)

    # "Prefill-shaped": all n_tokens rows multiplied against the weight in ONE matmul call.
    x_batched = torch.randn(n_tokens, d)
    torch.mm(x_batched[:8], weight)  # warm-up, avoid first-call overhead skewing the timing
    start = time.perf_counter()
    out_batched = torch.mm(x_batched, weight)
    batched_time = time.perf_counter() - start

    # "Decode-shaped": the SAME total number of rows, but ONE row at a time, n_tokens separate calls.
    x_rows = [torch.randn(1, d) for _ in range(n_tokens)]
    torch.mm(x_rows[0], weight)  # warm-up
    start = time.perf_counter()
    for row in x_rows:
        _ = torch.mm(row, weight)
    sequential_time = time.perf_counter() - start

    batched_tokens_per_sec = n_tokens / batched_time
    sequential_tokens_per_sec = n_tokens / sequential_time

    print(f"Weight matrix: ({d}, {d}), fp32. {n_tokens} total rows of work, either way.\n")
    header = f"{'mode':28}{'wall-clock':>14}{'tokens/sec':>16}"
    print(header)
    print("-" * len(header))
    print(f"{'one batched matmul':28}{batched_time * 1000:>11.2f} ms{batched_tokens_per_sec:>16,.0f}")
    print(f"{'N separate 1-row matmuls':28}{sequential_time * 1000:>11.2f} ms{sequential_tokens_per_sec:>16,.0f}")

    speedup = sequential_time / batched_time
    print(f"\n-> The single batched call reached {speedup:.1f}x the tokens/sec of doing the exact")
    print(f"   same total arithmetic one row at a time. Two effects are bundled together here")
    print(f"   (both real, on this CPU, right now): better reuse of the weight matrix while it's")
    print(f"   resident in a fast cache level, and less per-call dispatch overhead. A real GPU's")
    print(f"   HBM/SRAM gap and warp-level parallelism (README sections 1-2) produce the same")
    print(f"   qualitative shape far more dramatically -- this is the same shape, at a much")
    print(f"   smaller scale, on completely different hardware.")


def main():
    prefill_vs_decode_demo()
    decode_batching_sweep_demo()
    cpu_batching_analogy_demo()


if __name__ == "__main__":
    main()
