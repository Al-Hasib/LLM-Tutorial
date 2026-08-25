"""
Building a Mini-Transformer / Mini-GPT From Scratch

A complete, trainable, decoder-only Transformer (a tiny GPT), assembled
from every building block in this phase: token + positional embeddings,
causal multi-head self-attention, a feed-forward sublayer, Pre-LN
residual connections, and a final linear head over the vocabulary.
Trained character-by-character on a small toy corpus with the exact
next-token-prediction objective every real LLM uses.

Runtime: ~1-2 minutes on a CPU (2000 training steps).

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# 0. Toy corpus and character-level tokenizer
# ---------------------------------------------------------------------------

CORPUS = """
the quick brown fox jumps over the lazy dog.
the lazy dog barks at the quick brown fox.
the fox runs away into the dark forest.
the dog chases the fox through the forest.
the forest is full of tall trees and green leaves.
birds sing songs in the tall trees every morning.
the sun rises over the forest every morning.
the moon shines over the forest every night.
the quick brown fox is quicker than the lazy dog.
the lazy dog is lazier than the quick brown fox.
""".strip().lower()
CORPUS = (CORPUS + "\n") * 8   # repeat so there is enough data for the model to learn from

chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(CORPUS), dtype=torch.long)

# ---------------------------------------------------------------------------
# Hyperparameters -- deliberately tiny so this trains in well under a
# minute on a CPU while still clearly demonstrating the full recipe.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32     # context window (max tokens attended over)
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
BATCH_SIZE = 32
NUM_ITERS = 2000
LEARNING_RATE = 3e-3


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


# ---------------------------------------------------------------------------
# 1. The model: token+positional embedding -> N decoder blocks -> head
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        batch, T, d_model = x.shape

        def split_heads(t):
            return t.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.W_q(x)), split_heads(self.W_k(x)), split_heads(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
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


class DecoderBlock(nn.Module):
    """Pre-LN decoder-only block (Lesson 5 section 4): normalize BEFORE each
    sublayer, add the residual around the whole thing. No cross-attention --
    this is what makes it decoder-ONLY (Lesson 4 section 5)."""

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


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Learned positional embeddings this time (Lesson 3 section 5's alternative
        # to the sinusoidal encoding used in Lessons 3-4) -- simplest to wire up
        # for a fixed, known block_size like this one.
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, block_size) for _ in range(num_layers)]
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


def main():
    print(f"Corpus length: {len(CORPUS)} characters")
    print(f"Vocabulary ({vocab_size} unique characters): {chars}")

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params:,}")

    print("\n" + "=" * 70)
    print("GENERATION BEFORE TRAINING (random weights)")
    print("=" * 70)
    start_ids = torch.tensor([encode("the ")], dtype=torch.long)
    generated = model.generate(start_ids, max_new_tokens=80)
    print(repr(decode(generated[0].tolist())))

    print("\n" + "=" * 70)
    print("TRAINING (next-token prediction, cross-entropy loss)")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, NUM_ITERS + 1):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("GENERATION AFTER TRAINING")
    print("=" * 70)
    generated = model.generate(start_ids, max_new_tokens=120)
    print(repr(decode(generated[0].tolist())))
    print("\n-> Note this model has only ever seen the toy corpus above, so it")
    print("   isn't 'intelligent' -- it has just learned the character-level")
    print("   statistics of THIS text (common words, spacing, punctuation).")
    print("   That is nonetheless the exact same training signal, loss function,")
    print("   and architecture family used to pretrain every real LLM in this")
    print("   course -- just at a scale that fits inside one lesson.")


if __name__ == "__main__":
    main()
