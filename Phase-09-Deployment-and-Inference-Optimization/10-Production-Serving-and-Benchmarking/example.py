"""
Production Serving and Benchmarking

Three demos, each measuring REAL computed numbers from a simulation --
no hardcoded results:

  1. BOUNDED vs. UNBOUNDED request queues (README section 2). A bursty
     Poisson-ish arrival process (a quiet baseline rate, then an injected
     burst) is fed once through an UNBOUNDED queue in front of a
     fixed-capacity server, and once through a BOUNDED queue that sheds
     (rejects) new arrivals once it's full. We measure p50/p99 queueing
     latency for ACCEPTED requests in both cases, plus the number and
     fraction of requests rejected in the bounded case -- showing directly
     that bounding the queue keeps accepted-request latency under control
     at the cost of shedding some load, while the unbounded queue lets
     latency blow up for everyone during the burst.

  2. ROUND-ROBIN vs. LEAST-OUTSTANDING-REQUESTS load balancing (README
     section 3). A stream of requests with randomly varying service times
     (real generation lengths vary a lot) is routed across a small fixed
     number of replicas under each policy. We measure the real resulting
     per-replica busy-time imbalance (max busy time minus min busy time
     across replicas) and overall makespan under each policy, showing
     least-outstanding-requests balances load better than round-robin
     when service times are uneven.

  3. METRICS FROM DATA (README section 5). A small synthetic set of
     per-token completion timestamps for a handful of concurrently-served
     requests (built from demo 2's own simulated service times) is used to
     compute REAL TTFT, TPOT, per-request throughput, and aggregate fleet
     throughput directly from those timestamps, using the formulas from
     README section 5 -- so the metric definitions are demonstrated as
     actual code computing actual numbers, not just prose formulas.

This is a pure discrete-event / time-stepped simulation over Python's
standard library only -- it models the QUEUEING, ROUTING, and MEASUREMENT
problems a production serving layer solves on top of the single-process
batching Lesson 4 covers, not a real transformer forward pass.

Runtime: well under a second.

Run:
    python example.py
"""

import random
import statistics

random.seed(0)


# ---------------------------------------------------------------------------
# 1. Bounded vs. unbounded queues with admission control (README section 2)
# ---------------------------------------------------------------------------

SERVER_CAPACITY = 4        # how many requests the server can process concurrently
SERVICE_TIME = 1.0         # ticks to fully process one request (fixed, for this demo)
QUEUE_BOUND = 6            # max requests allowed to wait (bounded case only)
SIM_TICKS = 400            # total simulated ticks
BASELINE_ARRIVAL_RATE = 0.5   # requests/tick outside the burst
BURST_ARRIVAL_RATE = 8.0     # requests/tick DURING the burst -- a real overload spike (2x capacity)
BURST_START, BURST_END = 150, 200


def make_bursty_arrivals(num_ticks, baseline_rate, burst_rate, burst_start, burst_end):
    """Poisson-ish arrivals: at each tick, draw a Poisson-distributed number of
    new requests using the tick's rate (baseline, or the much higher burst
    rate during [burst_start, burst_end)). Returns a list, one entry per
    tick, of how many requests arrive at that tick."""
    arrivals = []
    for t in range(num_ticks):
        rate = burst_rate if burst_start <= t < burst_end else baseline_rate
        # Simple Poisson draw via Knuth's algorithm -- fine at these small rates.
        l = pow(2.718281828, -rate)
        k, p = 0, 1.0
        while True:
            k += 1
            p *= random.random()
            if p <= l:
                break
        arrivals.append(k - 1)
    return arrivals


def simulate_queue(arrivals_per_tick, capacity, service_time, queue_bound=None):
    """Time-stepped simulation of a FIFO queue feeding a fixed-capacity
    server. queue_bound=None means UNBOUNDED (never rejects); an integer
    means a request is REJECTED the instant it would make the waiting
    queue exceed that bound. Returns queueing latencies (ticks waited
    before service starts) for every ACCEPTED request, plus a reject count.

    Server model: `capacity` independent slots, each busy for exactly
    `service_time` ticks once it picks up a request from the head of the
    queue. This isolates the QUEUEING policy's effect from any batching
    policy (Lesson 4's concern, not this one)."""
    queue = []          # list of arrival ticks still waiting
    slot_free_at = [0.0] * capacity   # tick at which each slot becomes free
    queueing_latencies = []
    num_rejected = 0
    num_arrived = 0

    for t in range(len(arrivals_per_tick)):
        n_new = arrivals_per_tick[t]
        for _ in range(n_new):
            num_arrived += 1
            if queue_bound is not None and len(queue) >= queue_bound:
                num_rejected += 1
                continue
            queue.append(t)

        # Assign waiting requests to any slot that is free by now, in FIFO order.
        for i in range(capacity):
            while queue and slot_free_at[i] <= t:
                arrival_t = queue.pop(0)
                queueing_latencies.append(t - arrival_t)
                slot_free_at[i] = t + service_time

    # Drain: let any remaining queued requests finish being admitted after
    # the arrival window ends, so nothing is silently dropped from the count.
    t = len(arrivals_per_tick)
    while queue:
        # advance to the next tick any slot frees up
        t = min(max(t, min(slot_free_at)), t + 10_000)
        for i in range(capacity):
            while queue and slot_free_at[i] <= t:
                arrival_t = queue.pop(0)
                queueing_latencies.append(t - arrival_t)
                slot_free_at[i] = t + service_time
        t += 1

    return {
        "queueing_latencies": queueing_latencies,
        "num_rejected": num_rejected,
        "num_arrived": num_arrived,
    }


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


def queueing_demo():
    print("=" * 70)
    print("1. BOUNDED vs. UNBOUNDED QUEUES: ADMISSION CONTROL UNDER A BURST")
    print("=" * 70)

    arrivals = make_bursty_arrivals(
        SIM_TICKS, BASELINE_ARRIVAL_RATE, BURST_ARRIVAL_RATE, BURST_START, BURST_END
    )
    total_arrived = sum(arrivals)
    print(f"Simulated {SIM_TICKS} ticks, server capacity = {SERVER_CAPACITY} concurrent slots, "
          f"service time = {SERVICE_TIME:.0f} tick/request.")
    print(f"Baseline arrival rate = {BASELINE_ARRIVAL_RATE}/tick; burst rate = {BURST_ARRIVAL_RATE}/tick "
          f"during ticks [{BURST_START}, {BURST_END}) -- a real overload spike.")
    print(f"Total requests generated across the run: {total_arrived}\n")

    unbounded = simulate_queue(arrivals, SERVER_CAPACITY, SERVICE_TIME, queue_bound=None)
    bounded = simulate_queue(arrivals, SERVER_CAPACITY, SERVICE_TIME, queue_bound=QUEUE_BOUND)

    def summarize(label, result):
        lat = result["queueing_latencies"]
        p50 = percentile(lat, 50)
        p99 = percentile(lat, 99)
        worst = max(lat) if lat else float("nan")
        rejected = result["num_rejected"]
        arrived = result["num_arrived"]
        pct_rejected = rejected / arrived * 100 if arrived else 0.0
        print(f"{label}")
        print(f"  accepted requests:        {len(lat)}")
        print(f"  rejected requests:        {rejected} ({pct_rejected:.1f}% of {arrived} arrivals)")
        print(f"  queueing latency p50:     {p50:.1f} ticks")
        print(f"  queueing latency p99:     {p99:.1f} ticks")
        print(f"  queueing latency worst:   {worst:.1f} ticks")
        return p50, p99, worst, rejected

    unb_p50, unb_p99, unb_worst, unb_rej = summarize("UNBOUNDED queue (never rejects):", unbounded)
    print()
    bnd_p50, bnd_p99, bnd_worst, bnd_rej = summarize(
        f"BOUNDED queue (max {QUEUE_BOUND} waiting, sheds load beyond that):", bounded
    )

    print(f"\n-> The unbounded queue rejects NOTHING ({unb_rej} rejected) but its p99 queueing")
    print(f"   latency reaches {unb_p99:.1f} ticks (worst case {unb_worst:.1f} ticks) -- every request")
    print(f"   admitted during the burst gets stuck behind an ever-growing backlog, and that")
    print(f"   backlog doesn't fully drain until long after the burst itself ends.")
    print(f"   The bounded queue rejects {bnd_rej} requests ({bnd_rej / bounded['num_arrived'] * 100:.1f}% of arrivals) during")
    print(f"   the overload, but every ACCEPTED request sees p99 latency of only {bnd_p99:.1f} ticks")
    print(f"   (worst case {bnd_worst:.1f} ticks) -- a real, bounded ceiling instead of an open-ended one.")
    print(f"   This is the load-shedding trade-off from README section 2 in real numbers: bounding")
    print(f"   the queue trades a controlled amount of outright rejection for a hard latency")
    print(f"   guarantee on everything that IS accepted.")


# ---------------------------------------------------------------------------
# 2. Round-robin vs. least-outstanding-requests load balancing
#    (README section 3)
# ---------------------------------------------------------------------------

NUM_REPLICAS = 4
NUM_ROUTED_REQUESTS = 200


def make_variable_service_times(num_requests):
    """Real generation lengths vary a lot -- mostly short responses, a
    meaningful minority much longer (same skewed shape Lesson 4's example.py
    uses for its own workload, regenerated independently here so this file
    stays self-contained)."""
    lengths = []
    for _ in range(num_requests):
        if random.random() < 0.8:
            lengths.append(random.randint(1, 5))     # short response
        else:
            lengths.append(random.randint(20, 40))   # long response
    return lengths


def simulate_round_robin(service_times, num_replicas):
    """Assign request i to replica (i mod num_replicas), completely ignoring
    how busy each replica currently is. Returns per-replica total busy time
    and each request's completion (queueing + service) time on its replica."""
    replica_free_at = [0.0] * num_replicas
    replica_busy_time = [0.0] * num_replicas
    completion_times = []
    for i, service_time in enumerate(service_times):
        r = i % num_replicas
        start = replica_free_at[r]
        finish = start + service_time
        replica_free_at[r] = finish
        replica_busy_time[r] += service_time
        completion_times.append(finish)
    return replica_busy_time, completion_times


def simulate_least_outstanding(service_times, num_replicas):
    """Assign each request to whichever replica is CURRENTLY free soonest
    (equivalently, has the least outstanding/remaining work queued on it --
    with a single FIFO queue per replica this is the same thing). Returns
    per-replica total busy time and completion times, for direct comparison
    with round-robin under the IDENTICAL sequence of service times."""
    replica_free_at = [0.0] * num_replicas
    replica_busy_time = [0.0] * num_replicas
    completion_times = []
    for service_time in service_times:
        r = min(range(num_replicas), key=lambda i: replica_free_at[i])
        start = replica_free_at[r]
        finish = start + service_time
        replica_free_at[r] = finish
        replica_busy_time[r] += service_time
        completion_times.append(finish)
    return replica_busy_time, completion_times


def load_balancing_demo():
    print("\n" + "=" * 70)
    print("2. ROUND-ROBIN vs. LEAST-OUTSTANDING-REQUESTS LOAD BALANCING")
    print("=" * 70)

    service_times = make_variable_service_times(NUM_ROUTED_REQUESTS)
    total_work = sum(service_times)
    print(f"{NUM_ROUTED_REQUESTS} requests routed across {NUM_REPLICAS} replicas, service times drawn from")
    print("a skewed distribution (80% short: 1-5 units, 20% long: 20-40 units).")
    print(f"Total real work across all requests: {total_work} service-time units.\n")

    rr_busy, rr_completions = simulate_round_robin(service_times, NUM_REPLICAS)
    lor_busy, lor_completions = simulate_least_outstanding(service_times, NUM_REPLICAS)

    def summarize(label, busy_times, completions):
        imbalance = max(busy_times) - min(busy_times)
        makespan = max(completions)
        p99_latency = percentile(completions, 99)
        print(f"{label}")
        print(f"  per-replica busy time:    {[round(b, 1) for b in busy_times]}")
        print(f"  max-min imbalance:        {imbalance:.1f} units")
        print(f"  makespan (last finish):   {makespan:.1f} units")
        print(f"  p99 completion time:      {p99_latency:.1f} units")
        return imbalance, makespan

    rr_imbalance, rr_makespan = summarize("ROUND-ROBIN:", rr_busy, rr_completions)
    print()
    lor_imbalance, lor_makespan = summarize("LEAST-OUTSTANDING-REQUESTS:", lor_busy, lor_completions)

    print(f"\n-> Both policies route the exact SAME {NUM_ROUTED_REQUESTS} requests, in the same arrival")
    print(f"   order, with the same service times, across the same {NUM_REPLICAS} replicas -- only the")
    print(f"   routing DECISION differs. Round-robin's blind cycling leaves a max-min busy-time")
    print(f"   imbalance of {rr_imbalance:.1f} units across replicas (some replicas luck into a run of")
    print(f"   long requests, others don't) and a makespan of {rr_makespan:.1f} units. Least-outstanding-")
    print(f"   requests, which only ever sends a new request to whichever replica is free soonest,")
    print(f"   cuts that imbalance to {lor_imbalance:.1f} units and finishes the whole batch in {lor_makespan:.1f}")
    print(f"   units -- exactly the README section 3 claim that least-outstanding-requests adapts")
    print(f"   to uneven request costs where round-robin, blind to load, cannot.")

    return service_times, lor_completions


# ---------------------------------------------------------------------------
# 3. Computing TTFT, TPOT, and throughput from real per-token timestamps
#    (README section 5)
# ---------------------------------------------------------------------------

TOKEN_TICK = 0.05   # seconds per one simulated decode step, for turning the
                    # abstract "service time units" above into a concrete
                    # per-token timeline


def build_token_timestamps(service_times, ttft_seconds_per_request, seed_offset=0):
    """For each request, synthesize a full per-token completion timestamp
    trace: the first token lands after that request's own TTFT (prefill
    time, varies per request since prompt lengths vary), and every
    subsequent token lands one (slightly noisy) decode step later. Returns
    a list of per-request dicts: {token_timestamps, ttft}."""
    rng = random.Random(1000 + seed_offset)
    traces = []
    for i, num_output_tokens in enumerate(service_times):
        num_output_tokens = max(2, num_output_tokens)  # need >= 2 tokens for a TPOT to exist
        ttft = ttft_seconds_per_request[i % len(ttft_seconds_per_request)]
        timestamps = [ttft]
        t = ttft
        for _ in range(num_output_tokens - 1):
            step = TOKEN_TICK * rng.uniform(0.85, 1.15)   # real decode steps jitter a little
            t += step
            timestamps.append(t)
        traces.append({"token_timestamps": timestamps, "ttft": ttft})
    return traces


def compute_request_metrics(trace):
    """Compute TTFT, TPOT, and per-request throughput directly from one
    request's token completion timestamps, using the README section 5
    formulas verbatim."""
    timestamps = trace["token_timestamps"]
    ttft = trace["ttft"]
    num_output_tokens = len(timestamps)
    total_latency = timestamps[-1]   # time of the LAST token, from request arrival at t=0
    tpot = (total_latency - ttft) / (num_output_tokens - 1)
    throughput = num_output_tokens / total_latency
    # Average ITL, computed independently from consecutive-token gaps, to show
    # it lands on the same number as TPOT (README section 5's explicit claim).
    itls = [timestamps[j] - timestamps[j - 1] for j in range(1, len(timestamps))]
    avg_itl = statistics.mean(itls)
    return {
        "ttft": ttft,
        "tpot": tpot,
        "avg_itl": avg_itl,
        "total_latency": total_latency,
        "num_output_tokens": num_output_tokens,
        "throughput": throughput,
    }


def metrics_from_data_demo(service_times):
    print("\n" + "=" * 70)
    print("3. TTFT, TPOT, AND THROUGHPUT: COMPUTED FROM REAL TIMESTAMPS")
    print("=" * 70)

    # Take a handful of requests from demo 2's own workload so this demo
    # builds on real, already-generated service times rather than inventing
    # a disconnected dataset.
    sample = service_times[:6]
    ttft_pool = [0.08, 0.12, 0.20, 0.35]   # a few representative prefill latencies (seconds)
    traces = build_token_timestamps(sample, ttft_pool)

    print(f"{len(traces)} requests, per-token completion timestamps synthesized from each")
    print("request's own output length (from demo 2) plus a per-request TTFT and small")
    print("per-step decode jitter -- a concrete stand-in for what a real server logs.\n")

    print(f"{'req':>4}{'tokens':>8}{'TTFT (s)':>11}{'TPOT (s)':>11}{'avg ITL (s)':>13}"
          f"{'total lat (s)':>15}{'throughput (tok/s)':>21}")
    per_request_metrics = []
    for i, trace in enumerate(traces):
        m = compute_request_metrics(trace)
        per_request_metrics.append(m)
        print(f"{i:>4}{m['num_output_tokens']:>8}{m['ttft']:>11.3f}{m['tpot']:>11.4f}"
              f"{m['avg_itl']:>13.4f}{m['total_latency']:>15.3f}{m['throughput']:>21.2f}")

    max_tpot_itl_gap = max(abs(m["tpot"] - m["avg_itl"]) for m in per_request_metrics)

    # Aggregate (fleet-level) throughput: sum of tokens produced by every
    # request divided by the wall-clock span over which they were all being
    # served CONCURRENTLY (from the earliest arrival to the latest finish) --
    # not the sum of each request's own total_latency, which would double
    # count the fact that they overlap in time.
    total_tokens = sum(m["num_output_tokens"] for m in per_request_metrics)
    fleet_span = max(m["total_latency"] for m in per_request_metrics)
    aggregate_throughput = total_tokens / fleet_span

    avg_per_request_throughput = statistics.mean(m["throughput"] for m in per_request_metrics)

    print(f"\nPer-request throughput averaged across these {len(traces)} requests: "
          f"{avg_per_request_throughput:.2f} tok/s")
    print(f"Aggregate (fleet-level) throughput -- {total_tokens} total output tokens produced")
    print(f"across all {len(traces)} requests, served concurrently over a {fleet_span:.3f}s span:")
    print(f"  {aggregate_throughput:.2f} tok/s")

    print(f"\n-> TPOT and average ITL agree to within {max_tpot_itl_gap:.4f}s per request (both measure")
    print(f"   the same per-step decode cost, just aggregated differently -- README section 5's")
    print(f"   explicit claim, confirmed here in actual computed numbers, not asserted in prose).")
    print(f"   Aggregate fleet throughput ({aggregate_throughput:.2f} tok/s) is far higher than any single")
    print(f"   request's own throughput (avg {avg_per_request_throughput:.2f} tok/s) precisely because {len(traces)} requests'")
    print(f"   token streams overlap in time -- this is the number that actually answers how")
    print(f"   much traffic a deployment can serve, not any individual request's own speed.")


def main():
    queueing_demo()
    service_times, _ = load_balancing_demo()
    metrics_from_data_demo(service_times)


if __name__ == "__main__":
    main()
