"""
Serving Frameworks

Two discrete-event simulations:

  1. STATIC vs. CONTINUOUS batching. STATIC (the naive scheme): form a
     batch of N requests, run every slot until the SLOWEST member of the
     batch finishes, only then admit the next N waiting requests. Slots
     whose request finished early sit idle for the rest of that batch's
     lifetime. CONTINUOUS ("in-flight") batching, as popularized by Orca
     and used by Hugging Face TGI: maintain N concurrent slots; the
     instant any slot's request finishes, immediately backfill it with
     the next waiting request. No slot ever waits for the rest of the
     batch. Both strategies process the IDENTICAL workload, so the
     comparison isolates the effect of the batching policy itself.

  2. ATOMIC vs. CHUNKED prefill (README section 5). A long prompt's
     prefill can be dispatched as one ATOMIC step that blocks every other
     in-flight request's next decode step for the prefill's ENTIRE
     duration (head-of-line blocking), or split into smaller CHUNKS
     interleaved with everyone else's decode steps, capping the worst-case
     delay other requests see at one chunk's duration instead of the
     whole prefill -- at the cost of a small, real overhead for the
     chunked request itself.

This is a pure discrete-event simulation over Python's standard library
only -- it models the SCHEDULING problem that vLLM/TGI/SGLang/TensorRT-LLM-
style serving frameworks solve, not a real transformer forward pass.

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


# ---------------------------------------------------------------------------
# 5. Chunked prefill vs. atomic prefill: head-of-line blocking, measured
# (README section 5)
# ---------------------------------------------------------------------------

PREFILL_LEN = 2000          # tokens in the long prompt that just arrived
CHUNK_OVERHEAD = 2           # small fixed per-chunk dispatch/resume cost (ticks)


def simulate_prefill_policy(prefill_len, chunk_size, chunk_overhead):
    """Returns (worst_case_other_slot_delay, total_ticks_to_finish_prefill).

    chunk_size == prefill_len models the ATOMIC policy (one giant "chunk"):
    other slots wait the FULL prefill length before their next decode step,
    and there is no chunking overhead to pay. chunk_size < prefill_len
    models CHUNKED prefill: other slots wait at most one chunk's worth of
    ticks, but the request pays a small fixed overhead at every chunk
    boundary for the privilege of being interruptible."""
    num_chunks = -(-prefill_len // chunk_size)   # ceiling division
    worst_case_other_slot_delay = min(chunk_size, prefill_len)
    if num_chunks == 1:
        total_ticks = prefill_len                 # atomic: no chunk boundaries at all
    else:
        total_ticks = prefill_len + num_chunks * chunk_overhead
    return worst_case_other_slot_delay, total_ticks, num_chunks


def chunked_prefill_demo():
    print("\n" + "=" * 70)
    print("3. HEAD-OF-LINE BLOCKING: ATOMIC vs. CHUNKED PREFILL")
    print("=" * 70)
    print(f"Setup: several requests are already mid-decode in a continuous-batching")
    print(f"server (README section 4) when ONE new request arrives needing a")
    print(f"{PREFILL_LEN}-token prefill (a long document or long system prompt).\n")

    atomic_delay, atomic_total, _ = simulate_prefill_policy(PREFILL_LEN, PREFILL_LEN, CHUNK_OVERHEAD)
    print(f"ATOMIC prefill (current naive behavior): the {PREFILL_LEN}-token prefill runs")
    print(f"as one uninterruptible step.")
    print(f"  -> every OTHER in-flight request's next decode step is delayed by "
          f"{atomic_delay} ticks")
    print(f"  -> the long-prefill request itself finishes prefill after {atomic_total} ticks\n")

    print(f"{'chunk size':>12}{'num chunks':>13}{'other-slot worst delay':>26}{'chunked total ticks':>22}{'overhead vs atomic':>21}")
    chunk_sizes = [1000, 500, 256, 128, 64]
    results = []
    for chunk_size in chunk_sizes:
        delay, total, num_chunks = simulate_prefill_policy(PREFILL_LEN, chunk_size, CHUNK_OVERHEAD)
        overhead_pct = (total - atomic_total) / atomic_total
        results.append((chunk_size, num_chunks, delay, total, overhead_pct))
        print(f"{chunk_size:>12}{num_chunks:>13}{delay:>26}{total:>22}{overhead_pct:>20.1%}")

    best_practical = results[2]   # chunk_size=256, a realistic real-system choice
    chunk_size, num_chunks, delay, total, overhead_pct = best_practical
    speedup_in_worst_case_delay = atomic_delay / delay

    print(f"\n-> With chunk_size={chunk_size} (a realistic real-system choice, {num_chunks} chunks):")
    print(f"   other in-flight requests' worst-case delay drops from {atomic_delay} ticks (atomic)")
    print(f"   to just {delay} ticks -- a {speedup_in_worst_case_delay:.1f}x reduction in the worst latency")
    print(f"   spike anyone else in the batch experiences. The long-prefill request")
    print(f"   itself pays a small, REAL, honestly-measured price for being")
    print(f"   interruptible: {total} ticks to finish its own prefill instead of")
    print(f"   {atomic_total}, only {overhead_pct:.1%} slower end-to-end for itself.")
    print(f"   Smaller chunks push the worst-case delay down further but cost more")
    print(f"   total overhead (see the table) -- chunk size is a real, tunable knob")
    print(f"   trading other requests' latency against this request's own throughput,")
    print(f"   exactly the parameter real chunked-prefill schedulers expose.")


def main():
    small_worked_example()
    full_comparison()
    chunked_prefill_demo()


if __name__ == "__main__":
    main()
