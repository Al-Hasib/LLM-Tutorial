"""
Distributed Inference at Scale

No multi-GPU hardware is required (or used) here -- both demos are honest,
formula/Monte-Carlo-based simulations of real distributed-inference effects,
clearly labeled as such.

Two demos:
  1. Pipeline-parallelism bubble sweep: computes the real
     bubble_fraction = (num_stages - 1) / num_micro_batches formula across a
     range of num_micro_batches values and a couple of num_stages values,
     showing concretely that num_micro_batches=1 (a single request's decode
     step -- one token, nothing to pipeline) leaves a huge bubble, while many
     concurrent in-flight micro-batches (many concurrent decode requests
     batched together, i.e. continuous batching) drives it toward zero.
  2. Expert-parallelism idle-GPU Monte Carlo simulation: for a range of
     decode-time batch sizes, actually sample random top-k expert routing
     decisions for each token in the batch (real randomness, not a
     hardcoded number) and measure the real fraction of experts (and thus
     GPUs, assuming one expert per GPU) that receive at least one token
     that step -- showing small decode batches leave many experts/GPUs idle
     while large batches saturate nearly all of them.

Runtime: well under a minute on CPU (pure arithmetic plus a few thousand
cheap random-sampling trials).

Run:
    python example.py
"""

import random

random.seed(0)


# ---------------------------------------------------------------------------
# 1. Pipeline parallelism: the bubble formula, computed for real, swept
#    across num_micro_batches -- exactly Lesson README section 3's formula.
# ---------------------------------------------------------------------------

def bubble_fraction(num_stages, num_micro_batches):
    """Real formula: bubble_fraction = (num_stages - 1) / num_micro_batches.
    num_micro_batches=1 means there is exactly ONE unit of work moving
    through the pipeline (a single request's single decode-step token) --
    the worst case. Larger num_micro_batches means more independent units
    of work (e.g. more concurrent in-flight requests' decode steps) keeping
    every stage busy simultaneously."""
    return (num_stages - 1) / num_micro_batches


def pipeline_bubble_demo():
    print("=" * 70)
    print("1. PIPELINE PARALLELISM: THE BUBBLE FRACTION, SWEPT FOR REAL")
    print("=" * 70)

    stage_counts = [4, 8]
    micro_batch_counts = [1, 2, 4, 8, 16, 64]

    print("\nbubble_fraction = (num_stages - 1) / num_micro_batches")
    print("(fraction of every stage's time spent idle, waiting on the pipeline)\n")

    header = f"{'num_micro_batches':>18}" + "".join(f"{'stages=' + str(s):>16}" for s in stage_counts)
    print(header)
    for m in micro_batch_counts:
        row = f"{m:>18}"
        for s in stage_counts:
            frac = bubble_fraction(s, m)
            row += f"{frac:>15.3f} "
        print(row)

    single_token_bubble = bubble_fraction(8, 1)
    many_concurrent_bubble = bubble_fraction(8, 64)
    print(f"\nAt num_stages=8:")
    print(f"  num_micro_batches=1  (one request, one decode-step token): "
          f"bubble_fraction = {single_token_bubble:.3f}  -> {single_token_bubble*100:.1f}% of every stage's time IDLE")
    print(f"  num_micro_batches=64 (64 concurrent in-flight decode steps): "
          f"bubble_fraction = {many_concurrent_bubble:.3f}  -> {many_concurrent_bubble*100:.1f}% idle")
    print("\n-> A single request's decode step has exactly ONE token to pipeline -- there is no")
    print("   'more micro-batches' available within that one request to fill the bubble with.")
    print("   The bubble only shrinks toward zero when the SERVER supplies many concurrent")
    print("   requests' decode steps as additional micro-batches flowing through the same")
    print("   pipeline stages simultaneously -- exactly why pipeline-parallel inference")
    print("   throughput depends on continuous batching keeping enough requests in flight.")


# ---------------------------------------------------------------------------
# 2. Expert parallelism: Monte Carlo simulation of decode-time idle-GPU risk.
#    Each token in a batch independently samples a random top-k subset of
#    experts (uniform random routing -- a reasonable stand-in for a
#    load-balanced router averaged over many draws); we measure the real
#    fraction of experts that get touched by at least one token that step.
# ---------------------------------------------------------------------------

def simulate_expert_utilization(num_experts, experts_per_token, batch_size, num_trials=2000):
    """Returns the AVERAGE fraction of experts that receive at least one
    token, over num_trials independent random batches of size batch_size.
    Each token samples experts_per_token distinct experts uniformly at
    random out of num_experts (a real random draw every trial, every
    token -- not a hardcoded/asserted number)."""
    total_fraction = 0.0
    for _ in range(num_trials):
        touched = set()
        for _ in range(batch_size):
            chosen = random.sample(range(num_experts), experts_per_token)
            touched.update(chosen)
        total_fraction += len(touched) / num_experts
    return total_fraction / num_trials


def expert_parallelism_demo():
    print("\n" + "=" * 70)
    print("2. EXPERT PARALLELISM: MONTE CARLO EXPERT/GPU UTILIZATION PER STEP")
    print("=" * 70)

    num_experts = 64
    experts_per_token = 2   # top-k routing, k=2 (Mixtral-style)
    batch_sizes = [1, 4, 16, 64, 256]

    print(f"\n{num_experts} experts total (one expert per GPU), top-{experts_per_token} routing per token.")
    print("Each trial: sample REAL random top-k expert choices for every token in the batch,")
    print("then measure what fraction of the 64 experts (GPUs) got at least one token.\n")

    print(f"{'batch_size':>12}{'avg fraction of experts touched':>34}{'avg experts IDLE that step':>30}")
    for bs in batch_sizes:
        frac = simulate_expert_utilization(num_experts, experts_per_token, bs)
        idle_experts = num_experts * (1 - frac)
        print(f"{bs:>12}{frac:>34.3f}{idle_experts:>30.1f}")

    frac_1 = simulate_expert_utilization(num_experts, experts_per_token, 1)
    frac_256 = simulate_expert_utilization(num_experts, experts_per_token, 256)
    print(f"\nAt batch_size=1   (a single decode-step token): only {frac_1*100:.1f}% of experts touched "
          f"-> {num_experts*(1-frac_1):.1f} of {num_experts} GPUs sit idle that step.")
    print(f"At batch_size=256 (many concurrent decode requests batched together): "
          f"{frac_256*100:.1f}% of experts touched -> only {num_experts*(1-frac_256):.1f} GPUs idle.")
    print("\n-> With a small decode batch, the handful of tokens in flight route to only a few")
    print("   of the 64 experts -- every GPU holding an unrouted-to expert does no useful work")
    print("   that step, a real distinct failure mode from the pipeline bubble above (whole")
    print("   devices going unused for a step, not stages waiting on data). Large batches")
    print("   route enough independent tokens that, statistically, nearly every expert gets")
    print("   touched -- which is exactly why MoE inference serving needs large enough batches")
    print("   (or clever expert placement/replication) to keep its expert-holding GPUs busy.")


def main():
    pipeline_bubble_demo()
    expert_parallelism_demo()


if __name__ == "__main__":
    main()
