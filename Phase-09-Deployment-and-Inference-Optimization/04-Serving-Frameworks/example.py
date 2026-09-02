"""
Serving Frameworks

A discrete-event simulation comparing two ways a serving framework can
group generation requests into batches on a fixed-capacity accelerator:

  1. STATIC batching (the naive scheme): form a batch of N requests, run
     every slot until the SLOWEST member of the batch finishes, only then
     admit the next N waiting requests. Slots whose request finished early
     sit idle for the rest of that batch's lifetime.
  2. CONTINUOUS ("in-flight") batching, as popularized by Orca and used by
     Hugging Face TGI: maintain N concurrent slots; the instant any slot's
     request finishes, immediately backfill it with the next waiting
     request. No slot ever waits for the rest of the batch.

Both strategies process the IDENTICAL workload (same requests, same
generation lengths, same arrival order), so the comparison isolates the
effect of the batching policy itself. We measure total simulated time
steps to clear the whole workload and per-scheme slot utilization
(fraction of slot-ticks that did real work vs. sat idle).

This is a pure discrete-event simulation over Python's standard library
only -- it models the SCHEDULING problem that vLLM/TGI-style serving
frameworks solve, not a real transformer forward pass.

Runtime: well under a second.

Run:
    python example.py
"""

import random

random.seed(0)

NUM_SLOTS = 8          # how many sequences the accelerator can process concurrently
NUM_REQUESTS = 48       # total requests to serve across the whole simulated run


def make_workload(num_requests):
    """Draw a variable, skewed distribution of generation lengths -- most
    requests are short (a quick answer), a minority are much longer (a
    detailed explanation). This mirrors real chat/completion traffic and is
    exactly the shape of workload where static batching suffers most."""
    lengths = []
    for _ in range(num_requests):
        if random.random() < 0.75:
            lengths.append(random.randint(2, 8))     # short request
        else:
            lengths.append(random.randint(30, 50))   # long request
    return lengths


# ---------------------------------------------------------------------------
# 1. Static batching simulation
# ---------------------------------------------------------------------------

def simulate_static_batching(gen_lengths, num_slots):
    """Chop the request queue into fixed-size batches of `num_slots`. Each
    batch runs for as many steps as its SLOWEST member needs; every other
    slot in that batch sits idle once its own request finishes."""
    total_time = 0
    total_slot_ticks = 0
    busy_slot_ticks = 0
    num_batches = 0

    queue = list(gen_lengths)
    while queue:
        batch = queue[:num_slots]
        queue = queue[num_slots:]
        num_batches += 1

        batch_duration = max(batch)          # the whole batch waits for the slowest member
        total_time += batch_duration
        total_slot_ticks += len(batch) * batch_duration
        busy_slot_ticks += sum(batch)        # real work done: each request's own length

    return {
        "total_time": total_time,
        "total_slot_ticks": total_slot_ticks,
        "busy_slot_ticks": busy_slot_ticks,
        "num_batches": num_batches,
    }


# ---------------------------------------------------------------------------
# 2. Continuous (in-flight) batching simulation
# ---------------------------------------------------------------------------

def simulate_continuous_batching(gen_lengths, num_slots):
    """Maintain `num_slots` concurrent slots. Every simulated tick, every
    slot that holds a request does one step of work; a slot whose request
    just finished is immediately backfilled from the waiting queue before
    the next tick, so it never sits idle while work is available."""
    queue = list(gen_lengths)
    slots = [None] * num_slots   # each entry: remaining steps for that slot's request, or None

    def refill_empty_slots():
        for i in range(num_slots):
            if slots[i] is None and queue:
                slots[i] = queue.pop(0)

    refill_empty_slots()

    total_time = 0
    total_slot_ticks = 0
    busy_slot_ticks = 0

    while any(s is not None for s in slots) or queue:
        total_time += 1
        for i in range(num_slots):
            total_slot_ticks += 1          # every slot is "available capacity" every tick
            if slots[i] is not None:
                busy_slot_ticks += 1       # this slot did real work this tick
                slots[i] -= 1
                if slots[i] == 0:
                    slots[i] = None
        refill_empty_slots()               # backfill any slot that just freed up, immediately

    return {
        "total_time": total_time,
        "total_slot_ticks": total_slot_ticks,
        "busy_slot_ticks": busy_slot_ticks,
    }


# ---------------------------------------------------------------------------
# 3. Small, hand-traceable example first
# ---------------------------------------------------------------------------

def small_worked_example():
    print("=" * 70)
    print("1. A SMALL, HAND-TRACEABLE EXAMPLE")
    print("=" * 70)
    tiny_lengths = [10, 2, 15, 3]
    num_slots = 2
    print(f"4 requests needing {tiny_lengths} generation steps, {num_slots} slots\n")

    static_result = simulate_static_batching(tiny_lengths, num_slots)
    cont_result = simulate_continuous_batching(tiny_lengths, num_slots)

    print("Static batching: batch 1 = requests needing [10, 2] steps -> both slots")
    print("  occupied for 10 steps (slot 2 idle for 8 of them, since its request")
    print("  finished after step 2 but the batch can't turn over yet).")
    print("  batch 2 = requests needing [15, 3] steps -> both slots occupied for 15")
    print("  steps (slot 2 idle for 12 of them).")
    print(f"  -> total time = {static_result['total_time']} steps "
          f"(10 + 15), num_batches = {static_result['num_batches']}")

    print("\nContinuous batching: slot 2 finishes its 2-step request at time 2 and is")
    print("  immediately handed the 3rd request (15 steps) without waiting for slot 1's")
    print("  10-step request to finish; slot 1 then picks up the last request (3 steps)")
    print("  the moment it frees up at time 10.")
    print(f"  -> total time = {cont_result['total_time']} steps")

    print(f"\n-> Same 4 requests, same total work (30 generation steps), but static")
    print(f"   batching takes {static_result['total_time']} steps end-to-end while")
    print(f"   continuous batching takes only {cont_result['total_time']} steps --")
    print(f"   purely from never leaving a slot idle while work is waiting.")


# ---------------------------------------------------------------------------
# 4. The full comparison at a realistic-shaped workload
# ---------------------------------------------------------------------------

def full_comparison():
    print("\n" + "=" * 70)
    print("2. FULL WORKLOAD: STATIC vs. CONTINUOUS BATCHING")
    print("=" * 70)

    gen_lengths = make_workload(NUM_REQUESTS)
    total_work = sum(gen_lengths)
    print(f"{NUM_REQUESTS} requests, {NUM_SLOTS} concurrent slots, generation lengths drawn")
    print("from a skewed distribution (75% short: 2-8 steps, 25% long: 30-50 steps).")
    print(f"Total real work across all requests: {total_work} generation-steps.\n")

    static_result = simulate_static_batching(gen_lengths, NUM_SLOTS)
    cont_result = simulate_continuous_batching(gen_lengths, NUM_SLOTS)

    static_util = static_result["busy_slot_ticks"] / static_result["total_slot_ticks"]
    cont_util = cont_result["busy_slot_ticks"] / cont_result["total_slot_ticks"]

    print(f"{'strategy':<22}{'total time (steps)':>20}{'slot-ticks (busy/total)':>26}{'utilization':>14}")
    print(f"{'Static batching':<22}{static_result['total_time']:>20}"
          f"{str(static_result['busy_slot_ticks']) + '/' + str(static_result['total_slot_ticks']):>26}"
          f"{static_util:>13.1%}")
    print(f"{'Continuous batching':<22}{cont_result['total_time']:>20}"
          f"{str(cont_result['busy_slot_ticks']) + '/' + str(cont_result['total_slot_ticks']):>26}"
          f"{cont_util:>13.1%}")

    speedup = static_result["total_time"] / cont_result["total_time"]
    util_gain = cont_util / static_util

    print(f"\n-> Both strategies perform the exact same {total_work} generation-steps")
    print(f"   of real work, in the same arrival order, on the same {NUM_SLOTS} slots.")
    print(f"   Static batching needed {static_result['total_time']} total time steps and formed")
    print(f"   {static_result['num_batches']} batches, achieving only {static_util:.1%} slot utilization --")
    print(f"   most of the waste comes from short requests sharing a batch with a long")
    print(f"   one and then sitting idle until the whole batch turns over.")
    print(f"   Continuous batching cleared the identical workload in {cont_result['total_time']} time")
    print(f"   steps ({speedup:.2f}x faster) at {cont_util:.1%} utilization ({util_gain:.2f}x higher) by")
    print(f"   backfilling a freed slot immediately instead of waiting for the batch.")
    print(f"   This is exactly the throughput gain Orca/Hugging Face TGI's continuous")
    print(f"   batching delivers over naive static batching on real, variable-length")
    print(f"   chat traffic -- no change to the model, only to the scheduling policy.")


def main():
    small_worked_example()
    full_comparison()


if __name__ == "__main__":
    main()
