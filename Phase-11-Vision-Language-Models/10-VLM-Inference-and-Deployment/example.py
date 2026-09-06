"""
VLM inference -- where the time and the memory actually go.

Part 1 MEASURES a real (small) vision tower + language model on this machine:
the vision forward pass, the LLM prefill over vision + text tokens, and a
real KV-cached decode loop. Parts 2-4 then do the arithmetic for a
production-size VLM, using the same structure.

No downloads, no pretrained weights, no GPU required. Under a minute.

The one fact this lesson is built around: a VLM request is prefill-dominated
in a way a text request almost never is. 576 vision tokens for one image is
longer than most user questions, and 3,136 (Lesson 1's AnyRes numbers) is
longer than most whole conversations -- so time-to-first-token, KV-cache
memory, and throughput are all governed by the image, not by the prompt.

Run:
    python example.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)


# ===========================================================================
# Part 1: measured -- a real vision tower, prefill, and KV-cached decode
# ===========================================================================

D = 256
N_HEADS = 4
N_LAYERS = 4
HEAD_DIM = D // N_HEADS


class CausalSelfAttention(nn.Module):
    """Manual attention with an explicit KV cache, so decode really reuses the
    keys and values from prefill instead of recomputing them."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)

    def forward(self, x, cache=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        shape = (b, t, N_HEADS, HEAD_DIM)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if cache is not None and cache["k"] is not None:
            k = torch.cat([cache["k"], k], dim=2)          # reuse past keys
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        causal = t > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).reshape(b, t, D)
        return self.out(y)


class Layer(nn.Module):
    def __init__(self, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.ff(self.ln2(x))


class VisionTower(nn.Module):
    """Patch embed + a few bidirectional blocks. Runs ONCE per image."""

    def __init__(self, patch=16, d_vision=D):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(3, d_vision, patch, patch)
        self.blocks = nn.ModuleList([Layer() for _ in range(N_LAYERS)])

    def forward(self, image):
        x = self.embed(image).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)                                     # no cache: not autoregressive
        return x


class MiniVLM(nn.Module):
    def __init__(self, vocab=1000):
        super().__init__()
        self.tower = VisionTower()
        self.projector = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.tok = nn.Embedding(vocab, D)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab)

    def new_cache(self):
        return [{"k": None, "v": None} for _ in self.layers]

    def encode_image(self, image):
        return self.projector(self.tower(image))

    def forward(self, embeds, cache=None):
        x = embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[i])
        return self.head(self.ln_f(x))


def timeit(fn, repeats=3):
    fn()                                                    # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000       # ms


print("=" * 78)
print("1. MEASURED ON THIS MACHINE (small model, CPU) -- where the time goes")
print("=" * 78)

model = MiniVLM().eval()
image = torch.randn(1, 3, 224, 224)              # 14x14 = 196 patches at patch=16
prompt = torch.randint(0, 1000, (1, 32))
N_DECODE = 32

with torch.no_grad():
    vis = model.encode_image(image)
    n_vis = vis.shape[1]

    t_vision = timeit(lambda: model.encode_image(image))

    def prefill_with_image():
        cache = model.new_cache()
        embeds = torch.cat([vis, model.tok(prompt)], dim=1)
        return model(embeds, cache), cache

    def prefill_text_only():
        cache = model.new_cache()
        return model(model.tok(prompt), cache), cache

    t_prefill_img = timeit(prefill_with_image)
    t_prefill_txt = timeit(prefill_text_only)

    def decode_loop():
        _, cache = prefill_with_image()
        nxt = torch.randint(0, 1000, (1, 1))
        for _ in range(N_DECODE):
            logits = model(model.tok(nxt), cache)
            nxt = logits[:, -1:].argmax(-1)

    t_full = timeit(decode_loop)
    t_decode = t_full - t_prefill_img

print(f"vision tokens produced           : {n_vis}")
print(f"text prompt tokens               : {prompt.shape[1]}")
print()
print(f"vision tower forward             : {t_vision:8.1f} ms")
print(f"LLM prefill, image + text        : {t_prefill_img:8.1f} ms")
print(f"LLM prefill, text only           : {t_prefill_txt:8.1f} ms   <- the same prompt, no image")
print(f"decode of {N_DECODE} tokens (KV-cached)  : {t_decode:8.1f} ms"
      f"  ({t_decode / N_DECODE:.2f} ms/token)")
print()
ttft = t_vision + t_prefill_img
print(f"TIME TO FIRST TOKEN              : {ttft:8.1f} ms"
      f"   (vision {100 * t_vision / ttft:.0f}%, prefill {100 * t_prefill_img / ttft:.0f}%)")
print(f"total for {N_DECODE} output tokens        : {ttft + t_decode:8.1f} ms"
      f"   (TTFT is {100 * ttft / (ttft + t_decode):.0f}% of it)")
print()
print(f"Adding one image made prefill {t_prefill_img / t_prefill_txt:.1f}x more expensive, and the vision")
print("tower has to run before any of it. The important structural point is WHERE")
print("that cost lands: entirely in time-to-first-token. Per-token decode speed --")
print("the tokens/second figure a text-LLM benchmark reports -- is unaffected by")
print("the image, because once the KV cache is built each new token costs the same")
print("as it would have in a text-only model.")
print()
print("So an image never slows the stream down; it delays the start of it, and it")
print("consumes cache and compute that would otherwise have served other")
print("requests. Whether a user notices depends on how long the answer is: for a")
print("300-word description the TTFT is lost in the stream, and for a one-word")
print("classification or a small JSON extraction the TTFT IS the response time.")
print("Part 2 puts real numbers on both cases.")
print()


# ===========================================================================
# Part 2: the arithmetic for a production-size VLM
# ===========================================================================

print("=" * 78)
print("2. THE SAME STRUCTURE AT PRODUCTION SCALE (7B-class VLM, one A100)")
print("=" * 78)

PARAMS = 7e9
LAYERS, N_KV_HEADS, HEAD_D = 32, 32, 128
BYTES = 2                                        # fp16
FLOPS = 150e12                                   # realistic achieved dense FLOP/s
BANDWIDTH = 1.5e12                               # achieved HBM bytes/s
PROMPT = 100
OUTPUT = 200


def kv_bytes_per_token():
    return 2 * LAYERS * N_KV_HEADS * HEAD_D * BYTES


def report(label, n_vis, n_images=1):
    total_vis = n_vis * n_images
    prefill_tokens = total_vis + PROMPT
    prefill_flops = 2 * PARAMS * prefill_tokens
    ttft_ms = prefill_flops / FLOPS * 1000
    # Decode is memory-bandwidth bound: every weight is read per token.
    decode_ms_per_token = PARAMS * BYTES / BANDWIDTH * 1000
    kv_mb = prefill_tokens * kv_bytes_per_token() / 1e6
    total_s = (ttft_ms + decode_ms_per_token * OUTPUT) / 1000
    print(f"{label:<34} {total_vis:>8,} {ttft_ms:>9.0f} {kv_mb:>10.0f} {total_s:>9.2f}")


print(f"{'configuration':<34} {'vis tok':>8} {'TTFT ms':>9} {'KV cache MB':>10} {'total s':>9}")
report("text only, no image", 0, 0)
report("1 image, 32 resampler queries", 32)
report("1 image, 144 tok (pixel-shuffle)", 144)
report("1 image, CLIP@336 = 576 tok", 576)
report("1 image, AnyRes 896px = 3136 tok", 3136)
report("4 images @ 576 tok", 576, 4)
report("16 video frames @ 64 tok", 64, 16)
report("64 video frames @ 64 tok", 64, 64)
print()
print(f"(prefill at {FLOPS / 1e12:.0f} TFLOP/s achieved, decode bandwidth-bound at")
print(f"{BANDWIDTH / 1e12:.1f} TB/s, {OUTPUT} output tokens, {PROMPT}-token text prompt)")
print()
print("Read the TTFT column against the text-only row: the vision tokens ARE the")
print("prompt as far as cost is concerned. And read the KV column as a capacity")
print("limit rather than a latency one -- at 0.5 MB per 1,000 tokens per")
print("request, an AnyRes image costs about half a gigabyte of KV cache, which")
print("is what caps how many concurrent VLM requests a GPU can hold. Fewer")
print("concurrent requests means less batching, which means worse throughput per")
print("GPU, which is why token compression (Lesson 4) shows up in the cost per")
print("request twice over.")
print()
print("How much of the response time is the image depends on the output length:")
print()
print(f"{'output tokens':>14} {'text only (s)':>14} {'AnyRes image (s)':>18} {'image share':>12}")
for out_tokens in [1, 20, 200, 1000]:
    dec = PARAMS * BYTES / BANDWIDTH * 1000 * out_tokens
    txt = (2 * PARAMS * PROMPT / FLOPS * 1000 + dec) / 1000
    img = (2 * PARAMS * (3136 + PROMPT) / FLOPS * 1000 + dec) / 1000
    print(f"{out_tokens:>14} {txt:>14.2f} {img:>18.2f} {(img - txt) / img:>11.0%}")
print()
print("For a one-token answer -- a yes/no, a class label, a routing decision --")
print("the image is essentially the entire request. For a long description it is a")
print("rounding error on latency, though it still occupies the same KV cache and")
print("the same prefill FLOPs, so it never stops mattering for capacity.")
print()


# ===========================================================================
# Part 3: prefix caching -- the biggest single win in VLM serving
# ===========================================================================

print("=" * 78)
print("3. PREFIX CACHING: many questions about the same image")
print("=" * 78)
print("A multi-turn conversation about one image, or a batch of questions about")
print("one document page, re-sends the same vision tokens every time. If the")
print("prefix is unchanged, its KV entries can be reused instead of recomputed.")
print()

n_vis = 576
prefill_flops_vis = 2 * PARAMS * n_vis
prefill_flops_txt = 2 * PARAMS * PROMPT
print(f"{'turns about one image':>22} {'no caching (ms)':>17} {'prefix cached (ms)':>20} {'saving':>8}")
for turns in [1, 2, 4, 8, 16]:
    cold = turns * (prefill_flops_vis + prefill_flops_txt) / FLOPS * 1000
    warm = (prefill_flops_vis + turns * prefill_flops_txt) / FLOPS * 1000
    print(f"{turns:>22} {cold:>17.0f} {warm:>20.0f} {cold / warm:>7.1f}x")
print()
print("The vision prefill is paid once instead of once per turn. Two things have")
print("to be true for this to work, and they are exactly the reasons some designs")
print("give it up:")
print()
print("  - the image tokens must be a PREFIX, unchanged across turns -- which")
print("    holds for prefix/projector fusion (Lesson 3) with image-first")
print("    prompts, and fails if anything before or inside them varies;")
print("  - the vision tokens must not depend on the question. Instruction-aware")
print("    compression (Lesson 4 section 3) buys accuracy per token and gives")
print("    this saving up, because its tokens change with every new question.")
print()
print("Beyond the KV cache there is a second, cruder cache worth having: the")
print("vision tower's own output for an image hash. In an application that asks")
print("many questions about the same catalogue of images, that turns the tower")
print("into a one-time cost per image rather than a per-request one.")
print()


# ===========================================================================
# Part 4: batching, and why images make scheduling harder
# ===========================================================================

print("=" * 78)
print("4. BATCHING WITH IMAGES")
print("=" * 78)
print("Continuous batching (Phase 09 Lesson 4) interleaves many requests'")
print("decode steps. Vision tokens disrupt it in two specific ways.")
print()
CHUNK = 2048
print(f"{'request':<30} {'prefill tokens':>15} {'chunks of ' + str(CHUNK):>16}")
for label, n in [("text chat turn", 100 + 0),
                 ("1 image @ 576 + prompt", 576 + 100),
                 ("1 AnyRes image @ 3136", 3136 + 100),
                 ("8-image comparison @ 576", 8 * 576 + 100),
                 ("64-frame video @ 64", 64 * 64 + 100)]:
    print(f"{label:<30} {n:>15,} {-(-n // CHUNK):>16}")
print()
print("1) A prefill that long BLOCKS the batch. While the scheduler is chewing")
print("   through 4,000+ vision tokens for one request, every other request in")
print("   the batch stops producing tokens, so one user's image is felt as a")
print("   stutter by everyone else. The fix, and the reason chunked prefill")
print("   exists, is to split the prefill into fixed-size pieces and interleave")
print("   decode steps between them.")
print()
print("2) Request cost becomes wildly variable. A text turn and a video request")
print("   differ by ~40x in prefill work here, so a scheduler that treats")
print("   requests as interchangeable will badly mis-predict both latency and")
print("   memory. Production VLM serving therefore admits requests against a")
print("   TOKEN budget (vision tokens included) rather than a request count.")
print()
print("Everything else from Phase 09 transfers unchanged: quantization applies")
print("to the language model exactly as before (the vision tower is a small")
print("fraction of the weights and is often left in higher precision),")
print("speculative decoding accelerates the decode phase and does nothing for")
print("the vision-dominated prefill, and paged KV cache is what makes the")
print("variable-length sequences above manageable at all.")
