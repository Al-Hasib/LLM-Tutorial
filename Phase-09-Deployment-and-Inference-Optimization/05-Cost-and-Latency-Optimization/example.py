"""
Cost and Latency Optimization

Two real simulations:
  1. The batching trade-off: a stream of randomly-arriving requests is
     grouped into batches of size B on a single simulated server; we
     measure BOTH throughput and average per-request latency across a
     sweep of B, including the case where a batch size is too small to
     even keep up with arrival rate (queue backlog).
  2. A toy model-routing cascade: a cheap upfront "difficulty" classifier
     sends easy queries to a cheap/fast model and hard queries to an
     expensive/accurate model, compared against always-cheap and
     always-expensive baselines on real measured cost and accuracy.

Run:
    python example.py
"""

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. The batching trade-off: throughput vs. latency
# ---------------------------------------------------------------------------

def simulate_batching(batch_size, num_requests=600, mean_interarrival_ms=10.0,
                       overhead_ms=50.0, marginal_ms_per_request=2.0):
    """A single-server batching simulation. Requests arrive one at a time;
    a batch is formed as soon as `batch_size` requests have queued, then
    processed as one unit (fixed overhead + a small per-request addition,
    matching real hardware where a batch is far cheaper per-item than
    running items one at a time, but not entirely free of scaling cost).
    The server processes one batch at a time (a single simulated GPU)."""
    interarrival_times = rng.exponential(mean_interarrival_ms, size=num_requests)
    arrival_times = np.cumsum(interarrival_times)

    processing_time = overhead_ms + marginal_ms_per_request * batch_size

    latencies = []
    server_free_at = 0.0
    for start in range(0, num_requests - num_requests % batch_size, batch_size):
        chunk = arrival_times[start:start + batch_size]
        fill_time = chunk[-1]                                  # when the batch becomes full
        batch_start = max(fill_time, server_free_at)             # server may still be busy
        batch_end = batch_start + processing_time
        server_free_at = batch_end
        latencies.extend(batch_end - chunk)                      # every request in the batch finishes together

    latencies = np.array(latencies)
    total_time_ms = server_free_at - arrival_times[0]
    throughput_per_sec = (len(latencies) / total_time_ms) * 1000.0
    return throughput_per_sec, latencies.mean(), latencies.max()


def batching_tradeoff_demo():
    print("=" * 70)
    print("1. THE BATCHING TRADE-OFF: THROUGHPUT vs. LATENCY")
    print("=" * 70)
    mean_interarrival_ms = 10.0
    arrival_rate_per_sec = 1000.0 / mean_interarrival_ms
    print(f"Requests arrive with mean interarrival {mean_interarrival_ms:.0f}ms "
          f"(~{arrival_rate_per_sec:.0f} requests/sec on average).")
    print("Batch cost model: 50ms fixed overhead + 2ms per additional request in the batch.\n")

    print(f"{'batch size':>10}{'throughput/sec':>16}{'avg latency (ms)':>20}{'max latency (ms)':>20}")
    for B in [1, 2, 4, 8, 16, 32, 64]:
        throughput, avg_latency, max_latency = simulate_batching(B)
        print(f"{B:>10}{throughput:>16.1f}{avg_latency:>20.1f}{max_latency:>20.1f}")

    print("\n-> At B=1, each request pays the full 52ms processing cost alone, but")
    print("   requests arrive every ~10ms on average -- the server cannot keep up,")
    print("   a backlog builds across the whole run, and both average AND max")
    print("   latency blow up far beyond what a single request's own processing")
    print("   time would suggest. Growing B raises the throughput ceiling (more")
    print("   requests completed per unit of server-busy-time) enough to escape")
    print("   that backlog, but every request in a bigger batch waits longer for")
    print("   the batch to FILL before processing even starts -- so latency keeps")
    print("   climbing even once throughput is no longer the bottleneck. There is")
    print("   no single batch size that minimizes both at once.")


# ---------------------------------------------------------------------------
# 2. Model routing / cascades
# ---------------------------------------------------------------------------

COST_CHEAP = 1.0
COST_EXPENSIVE = 20.0


def cheap_model_is_correct(difficulty):
    """The cheap model's accuracy degrades as queries get harder."""
    p_correct = np.clip(0.97 - 0.7 * difficulty, 0.05, 0.97)
    return rng.random(size=difficulty.shape) < p_correct


def expensive_model_is_correct(difficulty):
    """The expensive model stays highly accurate across the whole difficulty range."""
    p_correct = np.clip(0.97 - 0.08 * difficulty, 0.05, 0.97)
    return rng.random(size=difficulty.shape) < p_correct


def train_difficulty_router(threshold_candidates, n_train=2000):
    """Pick the routing threshold (on a noisy, CHEAP-to-compute proxy feature
    for difficulty) that maximizes (accuracy achieved) - (cost paid / scale)
    on a held-out validation set -- a simple, honest stand-in for "a small
    classifier trained to predict whether escalation is worth it." """
    true_difficulty = rng.uniform(0, 1, size=n_train)
    observed_feature = np.clip(true_difficulty + rng.normal(0, 0.15, size=n_train), 0, 1)

    best_threshold, best_score = None, -np.inf
    for t in threshold_candidates:
        route_expensive = observed_feature > t
        correct = np.where(
            route_expensive,
            expensive_model_is_correct(true_difficulty),
            cheap_model_is_correct(true_difficulty),
        )
        cost = np.where(route_expensive, COST_EXPENSIVE, COST_CHEAP)
        # Reward accuracy, penalize cost -- the trade-off a real deployment has to pick.
        score = correct.mean() - 0.02 * cost.mean()
        if score > best_score:
            best_score, best_threshold = score, t
    return best_threshold


def cascade_demo():
    print("\n" + "=" * 70)
    print("2. MODEL ROUTING / CASCADES: CHEAP-BY-DEFAULT, ESCALATE WHEN NEEDED")
    print("=" * 70)

    threshold = train_difficulty_router(np.linspace(0.05, 0.95, 19))
    print(f"Learned routing threshold on the (noisy) difficulty proxy: {threshold:.2f}")
    print("(observed_feature > threshold -> escalate to the expensive model)\n")

    n_test = 5000
    true_difficulty = rng.uniform(0, 1, size=n_test)
    observed_feature = np.clip(true_difficulty + rng.normal(0, 0.15, size=n_test), 0, 1)

    always_cheap_correct = cheap_model_is_correct(true_difficulty)
    always_expensive_correct = expensive_model_is_correct(true_difficulty)

    route_expensive = observed_feature > threshold
    cascade_correct = np.where(
        route_expensive,
        expensive_model_is_correct(true_difficulty),
        cheap_model_is_correct(true_difficulty),
    )
    cascade_cost = np.where(route_expensive, COST_EXPENSIVE, COST_CHEAP)

    print(f"{'strategy':>18}{'avg cost/query':>18}{'accuracy':>12}")
    print(f"{'always cheap':>18}{COST_CHEAP:>18.2f}{always_cheap_correct.mean():>12.3f}")
    print(f"{'always expensive':>18}{COST_EXPENSIVE:>18.2f}{always_expensive_correct.mean():>12.3f}")
    print(f"{'cascade':>18}{cascade_cost.mean():>18.2f}{cascade_correct.mean():>12.3f}")

    escalated_fraction = route_expensive.mean()
    cost_saved_vs_expensive = 1 - cascade_cost.mean() / COST_EXPENSIVE
    accuracy_gap_vs_expensive = always_expensive_correct.mean() - cascade_correct.mean()

    midpoint_cost = (COST_CHEAP + COST_EXPENSIVE) / 2
    midpoint_accuracy = (always_cheap_correct.mean() + always_expensive_correct.mean()) / 2
    print(f"\nEscalated to the expensive model: {escalated_fraction:.1%} of queries")
    print(f"Cost saved vs. always using the expensive model: {cost_saved_vs_expensive:.1%}")
    print(f"Accuracy gap vs. always using the expensive model: {accuracy_gap_vs_expensive:.3f}")
    print(f"\nFor reference, simply averaging the two baselines 50/50 would cost "
          f"{midpoint_cost:.2f} at {midpoint_accuracy:.3f} accuracy.")
    print(f"The cascade's cost ({cascade_cost.mean():.2f}) landed close to that same")
    print(f"midpoint, but its accuracy ({cascade_correct.mean():.3f}) came in ABOVE the")
    print(f"midpoint -- because it isn't routing randomly, it's deliberately sending the")
    print(f"queries where the cheap model is weakest up to the expensive model, so the")
    print(f"same escalation budget buys more accuracy than a coin flip would.")
    print("   The real risk (per the README) is the escalation decision being wrong on")
    print("   genuinely hard queries the cheap proxy feature underestimates -- that")
    print("   shows up ONLY in the accuracy gap above, never in the cost numbers, which")
    print("   is exactly why a cascade needs both metrics evaluated before shipping,")
    print("   not cost alone.")


def main():
    batching_tradeoff_demo()
    cascade_demo()


if __name__ == "__main__":
    main()
