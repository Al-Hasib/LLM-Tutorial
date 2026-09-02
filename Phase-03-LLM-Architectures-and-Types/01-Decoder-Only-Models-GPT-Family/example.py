"""
Decoder-Only Models: the GPT Family

Two parts:
  1. Computes real parameter counts for the published GPT-1/GPT-2-family
     configurations, using the exact same architecture formula as Phase 02's
     mini-GPT (token embedding + positional embedding + N x (attention + FFN)
     blocks + tied output head), and compares the exact count against the
     "quick estimate" formula (~12 * n_layer * d_model^2) commonly used in
     scaling-laws literature (Lesson 5 uses this same shorthand).
  2. Builds and trains the actual decoder-only architecture from the
     README's diagram -- causal self-attention + feed-forward blocks,
     stacked N times -- on a toy corpus, and generates text from it
     before and after training. This is the exact same block shape
     Part 1 was counting parameters for, just as real, runnable code.

Runtime: ~1-2 minutes on a CPU (Part 2 trains for 1500 steps).

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


def count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True):
    """Exact parameter count for a GPT-2-style decoder-only Transformer:
    - token embedding:      vocab_size * d_model
    - positional embedding: n_ctx * d_model            (learned, GPT-2 style)
    - per layer:
        attention (4 Linear(d_model, d_model), with bias): 4*(d_model^2 + d_model)
        2 LayerNorms (gamma + beta each):                   4*d_model
        FFN (d_model -> 4*d_model -> d_model, with bias):
            (d_model*d_ff + d_ff) + (d_ff*d_model + d_model), d_ff = 4*d_model
    - final LayerNorm: 2*d_model
    - output head: tied to token embedding in GPT-2 (0 extra params) or
      an untied vocab_size * d_model matrix otherwise.
    """
    d_ff = 4 * d_model

    token_embed = vocab_size * d_model
    pos_embed = n_ctx * d_model

    attn_params = 4 * (d_model * d_model + d_model)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    layernorm_params = 2 * (2 * d_model)   # 2 LayerNorms per layer, gamma+beta each
    per_layer = attn_params + ffn_params + layernorm_params

    final_layernorm = 2 * d_model
    output_head = 0 if weight_tying else vocab_size * d_model

    total = token_embed + pos_embed + n_layer * per_layer + final_layernorm + output_head
    return total


def quick_estimate(n_layer, d_model):
    """The ~12*n_layer*d_model^2 shorthand used throughout scaling-laws papers
    (see Lesson 5) -- it ignores embedding/vocab terms entirely, which is a
    fine approximation once d_model is large relative to n_ctx and vocab_size
    is a small multiple of d_model, but noticeably off for smaller configs."""
    return 12 * n_layer * d_model * d_model


GPT_CONFIGS = [
    # name,           n_layer, d_model, n_head, n_ctx, vocab_size, published_params
    ("GPT-1",         12,      768,     12,     512,   40000,      "~117M"),
    ("GPT-2 small",   12,      768,     12,     1024,  50257,      "~117M"),
    ("GPT-2 medium",  24,      1024,    16,     1024,  50257,      "~345M"),
    ("GPT-2 large",   36,      1280,    20,     1024,  50257,      "~774M"),
    ("GPT-2 XL",      48,      1600,    25,     1024,  50257,      "~1.5B"),
]


def param_count_demo():
    print("=" * 90)
    print("PART 1: PARAMETER COUNTS: PUBLISHED GPT CONFIGS vs. EXACT FORMULA vs. QUICK ESTIMATE")
    print("=" * 90)
    header = f"{'model':<14}{'layers':>8}{'d_model':>9}{'heads':>7}{'published':>12}" \
             f"{'exact count':>16}{'~12*L*d^2':>14}"
    print(header)
    for name, n_layer, d_model, n_head, n_ctx, vocab_size, published in GPT_CONFIGS:
        exact = count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True)
        estimate = quick_estimate(n_layer, d_model)
        print(f"{name:<14}{n_layer:>8}{d_model:>9}{n_head:>7}{published:>12}"
              f"{exact:>16,}{estimate:>14,}")

    print("\n-> The exact formula lands close to each model's published parameter")
    print("   count (small deviations come from GPT's real vocab size / embedding")
    print("   details, which vary slightly by exact release). The '~12*L*d^2'")
    print("   shorthand ignores the embedding table entirely -- notice it UNDERSHOOTS")
    print("   noticeably for GPT-2 small (where the ~38.6M-parameter embedding table")
    print("   is a large fraction of the model) but gets proportionally much closer")
    print("   for GPT-2 XL, where 48 huge transformer layers dwarf the embedding")
    print("   table. This is exactly why scaling-laws papers (Lesson 5) can get away")
    print("   with the simpler formula when studying large-scale trends.")

    print("\n" + "=" * 90)
    print("HOW MUCH OF EACH MODEL IS 'JUST' THE EMBEDDING TABLE?")
    print("=" * 90)
    for name, n_layer, d_model, n_head, n_ctx, vocab_size, published in GPT_CONFIGS:
        exact = count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True)
        embed_params = vocab_size * d_model + n_ctx * d_model
        print(f"  {name:<14} embedding share = {embed_params / exact:.1%}")

    print("\n-> This share shrinks steadily as models get bigger -- exactly why the")
    print("   embedding-free shorthand estimate gets more accurate at larger scale.")


# ---------------------------------------------------------------------------
# PART 2: the actual decoder-only architecture from the README's diagram --
# causal self-attention + feed-forward blocks, stacked N times, trained
# with real next-token prediction. Every GPT_CONFIGS row above is just this
# block with bigger numbers.
# ---------------------------------------------------------------------------

CORPUS = """
the transformer reads the whole sentence before it answers.
gpt only ever looks at the words that came before it.
the decoder predicts the next token, one token at a time.
attention lets every word look at every earlier word.
scale turned out to matter more than clever architecture tricks.
in-context learning needs no gradient update at all.
the model generates text by sampling one token after another.
pretraining teaches the model the statistics of ordinary text.
""".strip().lower()
CORPUS = (CORPUS + "\n") * 10  # repeat so there is enough data to learn from

CHARS = sorted(set(CORPUS))
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}


def encode(text):
    return [STOI[ch] for ch in text]


def decode(ids):
    return "".join(ITOS[i] for i in ids)


BLOCK_SIZE = 32    # context window
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3


class CausalSelfAttention(nn.Module):
    """The 'Causal Self-Attention' box in the README diagram: identical
    mechanism to bidirectional self-attention, plus one addition -- an
    upper-triangular mask so position i can only attend to positions <= i.
    That single mask is the entire architectural difference from Lesson 2's
    encoder-only attention."""

    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        batch, T, d_model = x.shape

        def split_heads(t):
            return t.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.W_q(x)), split_heads(self.W_k(x)), split_heads(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, T, d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderOnlyBlock(nn.Module):
    """The 'Decoder Block x N' box: Pre-LN residual attention, then Pre-LN
    residual feed-forward. No cross-attention anywhere -- that's what makes
    this decoder-ONLY, unlike Lesson 3's encoder-decoder block."""

    def __init__(self, d_model, num_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyModel(nn.Module):
    """The full GPT-family architecture: token+positional embedding -> N
    decoder blocks -> final LayerNorm -> linear head over the vocabulary.
    This is exactly the shape GPT_CONFIGS above scales up to GPT-1/2/3."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [DecoderOnlyBlock(d_model, num_heads, d_ff, block_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, targets=None):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)   # (batch, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]     # crop to context window
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def architecture_demo():
    print("\n" + "=" * 90)
    print("PART 2: THE ACTUAL DECODER-ONLY ARCHITECTURE, BUILT AND TRAINED")
    print("=" * 90)
    print(f"Vocabulary ({VOCAB_SIZE} unique characters): {CHARS}")

    data = torch.tensor(encode(CORPUS), dtype=torch.long)
    model = DecoderOnlyModel(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}  "
          f"(a tiny instance of the exact same architecture as the GPT_CONFIGS rows above)")

    start_ids = torch.tensor([encode("the ")], dtype=torch.long)

    print("\nGeneration BEFORE training (random weights):")
    generated = model.generate(start_ids, max_new_tokens=60)
    print(f"  {decode(generated[0].tolist())!r}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("\nTraining (next-token prediction, cross-entropy loss)...")
    for step in range(1, 1501):
        x, y = get_batch(data, BLOCK_SIZE, batch_size=32)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\nGeneration AFTER training:")
    generated = model.generate(start_ids, max_new_tokens=100)
    print(f"  {decode(generated[0].tolist())!r}")

    print("\n-> This is the exact block from the README diagram -- causal self-attention")
    print("   + feed-forward, stacked N times, no cross-attention, no bidirectional")
    print("   mask anywhere. Scale THIS UP (more layers, bigger d_model, more data,")
    print("   more compute) and, per Lessons 5 and Part 1's parameter counts above,")
    print("   you get GPT-2 and then GPT-3 -- nothing about the architecture changes.")


def main():
    param_count_demo()
    architecture_demo()


if __name__ == "__main__":
    main()
