"""
Kernel and Compiler Optimization

This machine has no GPU, so there is no way to literally measure a "kernel
launch" or literally capture a "CUDA Graph" here. Both demos below are
honest about that limitation and say explicitly, in their own output, what
is a REAL measurement and what is an ILLUSTRATIVE numeric model:

  1. Kernel-fusion analogue (REAL measurement, but of a stand-in): we time
     a chain of many SEPARATE, individually-dispatched torch operations
     against the exact same chain compiled once with `torch.jit.script`
     and called as a SINGLE dispatch from Python. This really does measure
     a real reduction in per-call CPU dispatch overhead, using PyTorch's
     own JIT compiler -- but it is a CPU dispatch-overhead analogue standing
     in for GPU kernel-LAUNCH overhead, not a literal GPU measurement. Said
     again in the printed output so no reader is misled.

  2. CUDA-Graph-replay analogue (ILLUSTRATIVE numeric model, NOT measured):
     a simple arithmetic model of "pay a fixed per-step dispatch overhead
     every step" vs. "pay it once, then replay," using an illustrative,
     explicitly-labeled microsecond-scale overhead figure -- not measured
     on any real GPU, since none is available in this environment.

Runtime: well under a minute on CPU (small tensors, a few thousand
repetitions, and pure arithmetic -- no GPU or heavy training involved).

Run:
    python example.py
"""

import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Kernel-fusion analogue: the tanh-approximation GELU activation, computed
#    two mathematically IDENTICAL ways:
#      - "unfused": as ~7 separate, individually Python-dispatched torch ops
#        (pow, mul, add, mul, tanh, add, mul) -- the analogue of a chain of
#        separate GPU kernel launches, each paying its own fixed dispatch
#        overhead (Lesson README section 1).
#      - "fused": one call to `torch.nn.functional.gelu(..., approximate=
#        "tanh")`, PyTorch's own built-in, single-dispatch, natively fused
#        implementation of the SAME formula -- the analogue of kernel fusion
#        collapsing that whole chain into one kernel (Lesson README section 2).
#    Both are REAL torch ops actually executing on this CPU; nothing here is
#    simulated. What is an analogue, not a literal measurement, is the STAND-IN:
#    Python-level call-dispatch overhead on a CPU standing in for GPU kernel-
#    LAUNCH overhead, since no GPU is available in this environment.
# ---------------------------------------------------------------------------

TENSOR_SIZE = 8       # deliberately tiny: dispatch overhead should dominate
NUM_CALLS = 20000     # number of times each version is called, for stable timing
WARMUP_CALLS = 1000

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_unfused(x):
    """Tanh-approximation GELU, computed as 7 separate, individually
    Python-dispatched torch calls -- one 'kernel launch' analogue per op."""
    x_cubed = torch.pow(x, 3)
    inner = torch.add(x, torch.mul(x_cubed, 0.044715))
    scaled = torch.mul(inner, SQRT_2_OVER_PI)
    tanhed = torch.tanh(scaled)
    one_plus_tanh = torch.add(tanhed, 1.0)
    result = torch.mul(x, one_plus_tanh)
    return torch.mul(result, 0.5)


def gelu_fused(x):
    """The IDENTICAL formula, but as ONE call into PyTorch's built-in,
    natively fused GELU kernel -- one dispatch instead of seven."""
    return F.gelu(x, approximate="tanh")


def time_calls(fn, x, num_calls, warmup_calls):
    # Warmup lets any one-time setup (caches, allocators) settle before the
    # timed region, so the measurement reflects steady-state per-call cost,
    # not one-off startup cost.
    for _ in range(warmup_calls):
        fn(x)
    start = time.perf_counter()
    for _ in range(num_calls):
        fn(x)
    elapsed = time.perf_counter() - start
    return elapsed


def kernel_fusion_analogue_demo():
    print("=" * 70)
    print("1. KERNEL-FUSION ANALOGUE (REAL MEASUREMENT OF A CPU STAND-IN)")
    print("=" * 70)
    print(f"Tanh-approximation GELU on a tensor of size {TENSOR_SIZE}, timed over "
          f"{NUM_CALLS} calls after {WARMUP_CALLS} warmup calls, computed two ways:")
    print("  (a) unfused -- 7 separate, individually Python-dispatched torch ops")
    print("  (b) fused   -- 1 call to torch.nn.functional.gelu (PyTorch's own")
    print("                  natively fused implementation of the same formula)")
    print("\nHONEST FRAMING: both versions really execute on this CPU right now --")
    print("this is a REAL measured wall-clock timing, not a simulation. What it is NOT")
    print("is a literal GPU kernel-launch measurement: there is no GPU here, so this")
    print("stands in for 'many GPU kernel launches vs. one fused kernel launch' using")
    print("'many Python-dispatched CPU ops vs. one fused CPU op' instead.")

    x = torch.randn(TENSOR_SIZE)

    unfused_time = time_calls(gelu_unfused, x, NUM_CALLS, WARMUP_CALLS)
    fused_time = time_calls(gelu_fused, x, NUM_CALLS, WARMUP_CALLS)

    # Sanity check: both paths must compute the SAME result -- fusion changes
    # how the work is dispatched, never what gets computed.
    out_unfused = gelu_unfused(x)
    out_fused = gelu_fused(x)
    outputs_match = torch.allclose(out_unfused, out_fused, atol=1e-6)
    assert outputs_match, "Fused and unfused GELU diverged -- this would be a correctness bug."

    per_call_unfused_us = unfused_time / NUM_CALLS * 1e6
    per_call_fused_us = fused_time / NUM_CALLS * 1e6
    speedup = unfused_time / fused_time

    print(f"\nOutputs identical (fusion never changes the math): {outputs_match}")
    print(f"\nTotal time, {NUM_CALLS} calls:")
    print(f"  unfused (7 separate torch calls): {unfused_time:.4f} s  ({per_call_unfused_us:.2f} us/call)")
    print(f"  fused (1 torch.nn.functional.gelu call):  {fused_time:.4f} s  "
          f"({per_call_fused_us:.2f} us/call)")
    print(f"  measured speedup (unfused / fused): {speedup:.2f}x")
    print("\n-> Real, measured number, on this CPU, for this stand-in workload. The")
    print("   mechanism it stands in for -- collapsing many per-op GPU kernel launches")
    print("   into one fused kernel dispatch -- is exactly Lesson README sections 1-2's")
    print("   point, just demonstrated here through Python/CPU call-dispatch overhead")
    print("   instead of literal CUDA kernel-launch overhead, since no GPU is present.")
    return speedup


# ---------------------------------------------------------------------------
# 2. CUDA-Graph-replay analogue: an ILLUSTRATIVE numeric model (not measured)
#    of paying a fixed per-step dispatch overhead every step vs. once.
# ---------------------------------------------------------------------------

# Illustrative, explicitly-labeled figures -- NOT measured on any real GPU
# (none is available in this environment). Real per-kernel-launch overhead on
# real hardware is commonly cited in the low single-digit-to-tens of
# microseconds; a decode step dispatches dozens of kernels (attention,
# projections, norms, activations, per layer), so a few hundred microseconds
# of TOTAL per-step launch overhead is a realistic order of magnitude to
# illustrate with -- it is still a modeled estimate, not a measurement.
ILLUSTRATIVE_PER_STEP_OVERHEAD_US = 300.0   # fixed dispatch overhead paid per step
ILLUSTRATIVE_PER_STEP_COMPUTE_US = 700.0    # actual GPU compute time per step


def total_time_us_without_graph(num_steps, overhead_us, compute_us):
    """Every step re-pays the fixed dispatch overhead -- the ordinary,
    un-graphed way of running N sequential decode steps."""
    return num_steps * (overhead_us + compute_us)


def total_time_us_with_graph(num_steps, overhead_us, compute_us):
    """The overhead is paid exactly ONCE, at capture time; every subsequent
    step replays the whole captured graph with a single dispatch, paying
    only the compute cost from then on."""
    return overhead_us + num_steps * compute_us


def cuda_graph_replay_analogue_demo():
    print("\n" + "=" * 70)
    print("2. CUDA-GRAPH-REPLAY ANALOGUE (ILLUSTRATIVE NUMERIC MODEL, NOT MEASURED)")
    print("=" * 70)
    print("HONEST FRAMING: no GPU is available in this environment, so nothing below")
    print("is measured. This is a simple arithmetic MODEL of the capture-once/")
    print("replay-many-times idea, using illustrative, explicitly-labeled microsecond")
    print(f"figures: {ILLUSTRATIVE_PER_STEP_OVERHEAD_US:.0f} us/step of fixed dispatch overhead, "
          f"{ILLUSTRATIVE_PER_STEP_COMPUTE_US:.0f} us/step of actual compute.")

    print(f"\n{'decode steps (N)':>18}{'no graph (us)':>18}{'with graph (us)':>18}"
          f"{'speedup':>12}{'overhead saved (us)':>22}")
    for num_steps in [1, 10, 50, 100, 500, 1000, 5000]:
        without = total_time_us_without_graph(
            num_steps, ILLUSTRATIVE_PER_STEP_OVERHEAD_US, ILLUSTRATIVE_PER_STEP_COMPUTE_US
        )
        withg = total_time_us_with_graph(
            num_steps, ILLUSTRATIVE_PER_STEP_OVERHEAD_US, ILLUSTRATIVE_PER_STEP_COMPUTE_US
        )
        speedup = without / withg
        overhead_saved = without - withg
        print(f"{num_steps:>18}{without:>18,.0f}{withg:>18,.0f}{speedup:>11.2f}x"
              f"{overhead_saved:>22,.0f}")

    print("\n-> Modeled, not measured: as N (the number of decode steps, i.e. output")
    print("   tokens generated) grows, replaying one captured graph keeps paying the")
    print("   per-step overhead exactly once no matter how large N gets, while the")
    print("   un-graphed path keeps re-paying it every single step -- so the benefit")
    print("   grows with N instead of staying fixed. This is precisely the shape of a")
    print("   real decode loop: the same fixed op sequence, repeated once per output")
    print("   token, is exactly the workload CUDA Graphs are built to amortize.")


def main():
    kernel_fusion_analogue_demo()
    cuda_graph_replay_analogue_demo()


if __name__ == "__main__":
    main()
