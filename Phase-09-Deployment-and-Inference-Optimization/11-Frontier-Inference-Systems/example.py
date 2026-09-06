"""
Frontier Inference Systems

Two demos:
  1. KV-cache memory for MHA vs. GQA vs. MQA vs. MLA, using the real
     cache-size formula from Lesson 3 Sec 4 (`2 * batch * seq_len *
     num_kv_heads * d_k * num_layers * bytes_per_value`), with MLA's
     `num_kv_heads * d_k` term replaced by a single small `d_latent`.
     Computed for one concrete, clearly-labeled ILLUSTRATIVE model
     configuration (NOT a specific real published model's exact
     config) across a sweep of context lengths, with real computed
     ratios between the four attention variants.
  2. A Monte Carlo simulation of a stream of requests whose prompts
     share a small, Zipfian-skewed set of "prefixes" (a few very
     common ones, many rare ones), routed across a fixed number of
     replicas each with a limited-capacity LRU prefix cache. We
     measure the REAL simulated cache-hit rate under (a) round-robin
     routing, which ignores cache state entirely, vs. (b) a simple
     cache-aware routing rule that sends a request to whichever
     replica most recently served its exact prefix, falling back to
     round-robin only on a cache miss.

Runtime: well under a second on any machine (pure arithmetic and a
few thousand-iteration Monte Carlo loops, no model training).

Run:
    python example.py
"""

import random
from collections import OrderedDict

random.seed(0)


# ---------------------------------------------------------------------------
# 1. KV-cache memory: MHA vs. GQA vs. MQA vs. MLA
#
#    Formula from Lesson 3 Sec 4:
#        KV cache bytes = 2 * batch * seq_len * num_kv_heads * d_k
#                          * num_layers * bytes_per_value
#    MLA replaces the `num_kv_heads * d_k` term with a single latent
#    dimension `d_latent`, reconstructing per-head K/V from that one
#    latent vector via learned up-projections at attention time
#    instead of caching separate K/V per (group of) heads at all.
# ---------------------------------------------------------------------------

# One concrete, ILLUSTRATIVE model configuration -- NOT a specific real
# published model's exact numbers, just realistic orders of magnitude for
# a large dense decoder-only model, stated explicitly so the arithmetic
# below is fully reproducible.
NUM_LAYERS = 60
NUM_HEADS = 96
D_K = 128                       # per-head Key/Value dimension
GQA_NUM_KV_HEADS = NUM_HEADS // 8   # one KV pair shared per group of 8 heads
MQA_NUM_KV_HEADS = 1                # one KV pair shared across ALL heads
D_LATENT = 64                       # MLA's single cached latent dim per token/layer
                                     # (deliberately smaller than one GQA group's
                                     # own d_k=128, per Lesson 3 Sec 4's formula)
BYTES_PER_VALUE = 2                  # bf16/fp16 storage
BATCH = 1                            # cache size for a single sequence


def kv_cache_bytes(seq_len, num_kv_heads_or_latent, num_layers=NUM_LAYERS,
                    d_k=None, batch=BATCH, bytes_per_value=BYTES_PER_VALUE):
    """Lesson 3 Sec 4's formula. For MHA/GQA/MQA, pass num_kv_heads_or_latent
    = num_kv_heads and d_k = D_K (the term is num_kv_heads * d_k). For MLA,
    pass num_kv_heads_or_latent = d_latent and d_k = 1 (the term is just
    d_latent -- a single cached vector per token per layer, not per head)."""
    per_token_per_layer_dim = num_kv_heads_or_latent * (d_k if d_k is not None else 1)
    return 2 * batch * seq_len * per_token_per_layer_dim * num_layers * bytes_per_value


def gb(num_bytes):
    return num_bytes / (1024 ** 3)


def kv_cache_memory_demo():
    print("=" * 78)
    print("1. KV CACHE MEMORY: MHA vs. GQA vs. MQA vs. MLA")
    print("=" * 78)
    print("Illustrative model config (NOT a specific real published model):")
    print(f"  num_layers={NUM_LAYERS}, num_heads={NUM_HEADS}, d_k={D_K}, "
          f"bytes_per_value={BYTES_PER_VALUE} (bf16), batch={BATCH}")
    print(f"  GQA num_kv_heads = num_heads/8 = {GQA_NUM_KV_HEADS}")
    print(f"  MQA num_kv_heads = {MQA_NUM_KV_HEADS}")
    print(f"  MLA d_latent = {D_LATENT}  (< one GQA group's own d_k={D_K})\n")

    context_lengths = [4_096, 32_768, 131_072]
    variants = [
        ("MHA (num_kv_heads=num_heads)", NUM_HEADS, D_K),
        ("GQA (num_kv_heads=num_heads/8)", GQA_NUM_KV_HEADS, D_K),
        ("MQA (num_kv_heads=1)", MQA_NUM_KV_HEADS, D_K),
        ("MLA (single d_latent per token)", D_LATENT, 1),
    ]

    header = f"{'context length':>16}" + "".join(f"{name.split(' ')[0]:>14}" for name, _, _ in variants)
    print(header + "   (GB)")
    results = {}
    for seq_len in context_lengths:
        row = f"{seq_len:>16,}"
        results[seq_len] = {}
        for name, num_kv_or_latent, d_k in variants:
            b = kv_cache_bytes(seq_len, num_kv_or_latent, d_k=d_k)
            results[seq_len][name] = b
            row += f"{gb(b):>14.3f}"
        print(row)

    print("\nReal computed ratios at the longest context (131,072 tokens):")
    longest = context_lengths[-1]
    mha_bytes = results[longest]["MHA (num_kv_heads=num_heads)"]
    for name, _, _ in variants:
        ratio = mha_bytes / results[longest][name]
        print(f"  MHA uses {ratio:6.1f}x the cache memory of {name}"
              if name != "MHA (num_kv_heads=num_heads)"
              else f"  MHA is the {ratio:.1f}x baseline")

    mla_bytes = results[longest]["MLA (single d_latent per token)"]
    gqa_bytes = results[longest]["GQA (num_kv_heads=num_heads/8)"]
    print(f"\n-> At {longest:,} tokens of context, MLA uses "
          f"{mha_bytes / mla_bytes:.1f}x LESS cache memory than plain MHA, and "
          f"{gqa_bytes / mla_bytes:.1f}x less than GQA -- GQA/MQA shrink the cache by sharing")
    print("   identical K/V across a group of heads (fewer distinct num_kv_heads); MLA shrinks it")
    print("   further by caching a single small latent vector per token instead of per-head K/V at")
    print("   all, reconstructing full multi-head K/V from that latent via up-projection at attention")
    print("   time -- more aggressive memory savings, at the cost of that extra per-step up-projection")
    print("   compute that GQA/MQA's directly-cached, reused K/V never has to pay.")


# ---------------------------------------------------------------------------
# 2. Cache-aware fleet routing vs. round-robin: real simulated hit rates.
#
#    A small pool of "prefixes" is drawn with Zipfian-skewed popularity
#    (a few prefixes are very common, most are rare) -- a stand-in for
#    a real workload's shared system prompts / few-shot templates / long
#    documents. Each replica has a limited-capacity LRU cache of prefixes
#    it currently holds. We compare:
#      (a) round-robin routing: request i goes to replica (i % num_replicas),
#          completely ignoring which replica has this prefix cached.
#      (b) cache-aware routing: route to whichever replica most recently
#          served this EXACT prefix, if that replica still has it cached;
#          otherwise fall back to round-robin.
# ---------------------------------------------------------------------------

NUM_PREFIXES = 24
ZIPF_SKEW = 1.3          # higher -> more skewed toward a few popular prefixes
NUM_REPLICAS = 4
CACHE_CAPACITY = 6       # distinct prefixes each replica's cache can hold
NUM_REQUESTS = 4_000


def zipfian_weights(num_items, skew):
    return [1.0 / ((rank + 1) ** skew) for rank in range(num_items)]


def make_prefix_stream(num_requests, num_prefixes, skew):
    prefixes = [f"prefix_{i}" for i in range(num_prefixes)]
    weights = zipfian_weights(num_prefixes, skew)
    # random.choices does real weighted random sampling, one prefix per request
    # (the request's unique per-user suffix is irrelevant to cache routing/hit
    # logic, so it is omitted -- only the shared prefix identity matters here).
    return random.choices(prefixes, weights=weights, k=num_requests)


class ReplicaCache:
    """A tiny LRU cache of prefixes a replica currently holds."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._store = OrderedDict()

    def has(self, key):
        return key in self._store

    def touch(self, key):
        """Record that `key` was just served by this replica (whether it was
        already cached or not), marking it most-recently-used and evicting
        the least-recently-used entry if the cache is now over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        else:
            self._store[key] = True
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)


def simulate_round_robin(prefix_stream, num_replicas, capacity):
    caches = [ReplicaCache(capacity) for _ in range(num_replicas)]
    hits = 0
    for i, prefix in enumerate(prefix_stream):
        replica = i % num_replicas
        if caches[replica].has(prefix):
            hits += 1
        caches[replica].touch(prefix)
    return hits / len(prefix_stream)


def simulate_cache_aware(prefix_stream, num_replicas, capacity):
    caches = [ReplicaCache(capacity) for _ in range(num_replicas)]
    last_served_by = {}   # prefix -> replica id that most recently served it
    rr_counter = 0
    hits = 0
    for prefix in prefix_stream:
        preferred = last_served_by.get(prefix)
        if preferred is not None and caches[preferred].has(prefix):
            replica = preferred
            hits += 1
        else:
            # cache miss (or never seen before): fall back to round-robin
            # among replicas, exactly like the naive policy would.
            replica = rr_counter % num_replicas
            rr_counter += 1
        caches[replica].touch(prefix)
        last_served_by[prefix] = replica
    return hits / len(prefix_stream)


def cache_aware_routing_demo():
    print("\n" + "=" * 78)
    print("2. CACHE-AWARE FLEET ROUTING vs. ROUND-ROBIN: SIMULATED HIT RATES")
    print("=" * 78)
    print(f"{NUM_REQUESTS:,} requests, {NUM_PREFIXES} distinct shared prefixes "
          f"(Zipfian skew={ZIPF_SKEW}), {NUM_REPLICAS} replicas, "
          f"each replica's prefix cache holds {CACHE_CAPACITY} prefixes.\n")

    weights = zipfian_weights(NUM_PREFIXES, ZIPF_SKEW)
    total_w = sum(weights)
    print("Prefix popularity (share of traffic), most popular first:")
    for rank in range(5):
        print(f"  prefix_{rank}: {weights[rank] / total_w * 100:5.1f}% of requests")
    print(f"  ... plus {NUM_PREFIXES - 5} rarer prefixes making up the rest\n")

    prefix_stream = make_prefix_stream(NUM_REQUESTS, NUM_PREFIXES, ZIPF_SKEW)

    rr_hit_rate = simulate_round_robin(prefix_stream, NUM_REPLICAS, CACHE_CAPACITY)
    ca_hit_rate = simulate_cache_aware(prefix_stream, NUM_REPLICAS, CACHE_CAPACITY)

    print(f"{'routing policy':>28}{'cache hit rate':>18}")
    print(f"{'round-robin (cache-blind)':>28}{rr_hit_rate * 100:>17.1f}%")
    print(f"{'cache-aware':>28}{ca_hit_rate * 100:>17.1f}%")

    improvement = ca_hit_rate / rr_hit_rate if rr_hit_rate > 0 else float("inf")
    print(f"\n-> Cache-aware routing achieves a {improvement:.1f}x higher hit rate than round-robin on "
          f"the SAME simulated request stream, replica count, and cache capacity -- the only")
    print("   difference is whether the router uses cache-topology information (which replica last")
    print("   served this exact prefix) when making its routing decision. Round-robin scatters even")
    print("   a very popular prefix's requests evenly across all replicas, so any one replica sees")
    print("   it only a fraction of the time and its LRU cache entry is frequently evicted between")
    print("   visits; cache-aware routing keeps a popular prefix's requests concentrated on whichever")
    print("   replica already has it hot, avoiding redundant prefill exactly as Lesson 6 Sec 2's")
    print("   single-server prefix caching does -- just applied at the level of the whole fleet.")


def main():
    kv_cache_memory_demo()
    cache_aware_routing_demo()


if __name__ == "__main__":
    main()
