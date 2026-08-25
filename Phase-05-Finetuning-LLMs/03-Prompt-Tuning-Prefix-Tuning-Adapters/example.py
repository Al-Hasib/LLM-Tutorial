"""
Prompt Tuning, Prefix Tuning and Adapters

Three demos, all built on one small frozen pretrained decoder-only
Transformer (the same architecture family as Phase 02's mini-GPT):

  1. Soft Prompt Tuning: prepend trainable "virtual token" embeddings to
     the input, freeze the entire base model, and verify DIRECTLY that
     only the soft prompt receives gradients while every base parameter's
     .grad stays None.
  2. Bottleneck Adapters: insert small trainable MLP modules after every
     frozen block, and verify the same gradient-isolation property.
  3. A trainable-parameter-count comparison across Prompt Tuning, Prefix
     Tuning, Adapters, and LoRA (Lesson 2), at a realistic 7B-model-class
     configuration.

Prefix Tuning itself is not separately trained here (it is architecturally
prefix tuning's cousin, prepending to every layer's K/V instead of just
the input embeddings) -- its parameter count is still computed exactly in
demo 3 from the same formula used in the README.

Runtime: ~20-30 seconds on a CPU.

Run:
    python example.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 0. Toy corpus and character-level tokenizer (same style as Phase 02 L6)
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
""".strip().lower()
CORPUS = (CORPUS + "\n") * 10

chars = sorted(set(CORPUS + "says hi"))  # make sure the target phrase's characters exist
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(CORPUS), dtype=torch.long)

D_MODEL = 48
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 2
BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# 1. A small decoder-only Transformer -- identical family to Phase 02 L6,
#    used here as "the pretrained base model" that everything else freezes.
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
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


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, block_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch, T = token_ids.shape
        positions = torch.arange(T)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def pretrain_base_model():
    """Stand-in for 'a pretrained base model' -- trained briefly on generic
    corpus text with ordinary next-token prediction, then frozen."""
    base = TinyGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-3)
    for step in range(600):
        x, y = get_batch(data, BLOCK_SIZE, 32)
        logits = base(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return base, loss.item()


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# 2. Soft Prompt Tuning
# ---------------------------------------------------------------------------

NUM_VIRTUAL_TOKENS = 4
PROMPT_TEXT = "the fox"
TARGET_CONTINUATION = " says hi"
FULL_TEXT = PROMPT_TEXT + TARGET_CONTINUATION


def soft_prompt_forward(base, soft_prompt, token_ids):
    """Manually replicate TinyGPT.forward, but with virtual token embeddings
    prepended before the real token embeddings -- the base model's own
    embedding table, blocks, and output head are used AS-IS (frozen)."""
    batch, T = token_ids.shape
    K = soft_prompt.shape[0]
    tok_emb = base.token_embedding(token_ids)
    prefix = soft_prompt.unsqueeze(0).expand(batch, -1, -1)
    x = torch.cat([prefix, tok_emb], dim=1)              # (batch, K+T, d_model)
    positions = torch.arange(K + T)
    x = x + base.position_embedding(positions)
    for block in base.blocks:
        x = block(x)
    x = base.final_norm(x)
    return base.output_head(x)                            # (batch, K+T, vocab)


@torch.no_grad()
def generate_with_soft_prompt(base, soft_prompt, start_ids, max_new_tokens, temperature=0.8):
    ids = start_ids.clone()
    max_real_len = base.block_size - soft_prompt.shape[0]
    for _ in range(max_new_tokens):
        context = ids[:, -max_real_len:]
        logits = soft_prompt_forward(base, soft_prompt, context)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def build_sft_style_example(full_text, prompt_text):
    """The exact -100/ignore_index masking pattern used again, more fully,
    in Lesson 4: predict every character of `full_text` from the one before
    it, but only count loss on positions predicting the part AFTER the
    prompt (labels = -100 elsewhere)."""
    ids = encode(full_text)
    prompt_len = len(encode(prompt_text))
    input_ids = ids[:-1]
    labels = [ids[i + 1] if (i + 1) >= prompt_len else -100 for i in range(len(ids) - 1)]
    return torch.tensor([input_ids]), torch.tensor([labels])


def soft_prompt_tuning_demo(base):
    print("=" * 78)
    print("1. SOFT PROMPT TUNING: TRAINABLE INPUT-PREPENDED EMBEDDINGS")
    print("=" * 78)

    input_ids, labels = build_sft_style_example(FULL_TEXT, PROMPT_TEXT)
    soft_prompt = nn.Parameter(torch.randn(NUM_VIRTUAL_TOKENS, D_MODEL) * 0.02)

    print(f"Task: prepend {NUM_VIRTUAL_TOKENS} trainable virtual tokens ahead of {PROMPT_TEXT!r}, ")
    print(f"train them (ONLY them) so the frozen base continues with {TARGET_CONTINUATION!r}")
    print(f"instead of whatever it would naturally continue with.\n")

    print("BEFORE training -- frozen base model, no soft prompt, generating from "
          f"{PROMPT_TEXT!r}:")
    baseline = base.generate(torch.tensor([encode(PROMPT_TEXT)]), max_new_tokens=20)
    print(f"  {decode(baseline[0].tolist())!r}  (ordinary corpus-style continuation)")

    optimizer = torch.optim.Adam([soft_prompt], lr=0.05)
    for step in range(1, 301):
        logits = soft_prompt_forward(base, soft_prompt, input_ids)
        # Only the last len(labels) positions correspond to real-token predictions.
        pred_logits = logits[:, NUM_VIRTUAL_TOKENS:, :]
        loss = F.cross_entropy(pred_logits.reshape(-1, vocab_size), labels.reshape(-1),
                                ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")

    # The core claim, checked directly: base model NEVER received a gradient.
    base_grads_all_none = all(p.grad is None for p in base.parameters())
    soft_prompt_has_grad = soft_prompt.grad is not None and soft_prompt.grad.abs().sum().item() > 0
    print(f"\nEvery base-model parameter's .grad is None (never updated): {base_grads_all_none}")
    print(f"Soft prompt's .grad is populated and nonzero: {soft_prompt_has_grad}")

    print(f"\nAFTER training -- SAME frozen base, soft prompt now prepended, "
          f"generating from {PROMPT_TEXT!r}:")
    steered = generate_with_soft_prompt(base, soft_prompt, torch.tensor([encode(PROMPT_TEXT)]),
                                         max_new_tokens=20, temperature=0.3)
    print(f"  {decode(steered[0].tolist())!r}")
    print(f"\n-> The base model's weights never moved (verified above) -- everything about")
    print("   this new behavior comes from 4 x 48 = 192 trainable numbers steering the")
    print("   SAME frozen network toward a completely different continuation.")

    return soft_prompt


# ---------------------------------------------------------------------------
# 3. Bottleneck Adapters
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """Down-project -> nonlinearity -> up-project, with a residual connection.
    W_up is zero-initialized so the adapter starts as a no-op (output 0),
    exactly like LoRA's zero-initialized B matrix in Lesson 2."""

    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


def adapter_forward(base, adapters, token_ids):
    """Same frozen embedding/blocks/head as the base model, but with one
    trainable Adapter's output added (residually) after every block.
    (Houlsby et al.'s original places one adapter after EACH sublayer,
    i.e. two per block; this uses one per block for simplicity -- the
    gradient-isolation mechanics are identical either way.)"""
    batch, T = token_ids.shape
    positions = torch.arange(T)
    x = base.token_embedding(token_ids) + base.position_embedding(positions)
    for block, adapter in zip(base.blocks, adapters):
        x = block(x)
        x = x + adapter(x)
    x = base.final_norm(x)
    return base.output_head(x)


@torch.no_grad()
def generate_with_adapters(base, adapters, start_ids, max_new_tokens, temperature=0.3):
    ids = start_ids.clone()
    for _ in range(max_new_tokens):
        context = ids[:, -base.block_size:]
        logits = adapter_forward(base, adapters, context)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def adapter_demo(base):
    print("\n" + "=" * 78)
    print("2. ADAPTERS: TRAINABLE BOTTLENECK MLPS INSERTED INSIDE EVERY BLOCK")
    print("=" * 78)

    bottleneck = 8
    adapters = nn.ModuleList([Adapter(D_MODEL, bottleneck) for _ in range(NUM_LAYERS)])
    adapter_params = sum(p.numel() for p in adapters.parameters())
    print(f"Same frozen base model as demo 1. Inserting {NUM_LAYERS} adapters "
          f"(bottleneck={bottleneck}), {adapter_params} trainable parameters total.")
    print(f"Same task: force the continuation of {PROMPT_TEXT!r} to become "
          f"{TARGET_CONTINUATION!r}.\n")

    input_ids, labels = build_sft_style_example(FULL_TEXT, PROMPT_TEXT)
    optimizer = torch.optim.Adam(adapters.parameters(), lr=0.05)
    for step in range(1, 301):
        logits = adapter_forward(base, adapters, input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1),
                                ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")

    base_grads_all_none = all(p.grad is None for p in base.parameters())
    adapter_has_grad = all(p.grad is not None and p.grad.abs().sum().item() >= 0
                            for p in adapters.parameters())
    adapter_grad_nonzero = sum(p.grad.abs().sum().item() for p in adapters.parameters()) > 0
    print(f"\nEvery base-model parameter's .grad is still None: {base_grads_all_none}")
    print(f"Every adapter parameter received a gradient, total |grad| > 0: "
          f"{adapter_has_grad and adapter_grad_nonzero}")

    print(f"\nAFTER training -- frozen base + trained adapters, generating from "
          f"{PROMPT_TEXT!r}:")
    steered = generate_with_adapters(base, adapters, torch.tensor([encode(PROMPT_TEXT)]),
                                      max_new_tokens=20)
    print(f"  {decode(steered[0].tolist())!r}")
    print(f"\n-> Same frozen-base guarantee as soft prompt tuning, via a structurally")
    print(f"   different mechanism: {adapter_params} trainable numbers living INSIDE the")
    print("   network (not prepended to the input) steer the same frozen weights.")


# ---------------------------------------------------------------------------
# 4. Trainable-parameter comparison, at a realistic 7B-model-class config
# ---------------------------------------------------------------------------

def parameter_comparison_demo():
    print("\n" + "=" * 78)
    print("3. TRAINABLE PARAMETERS AT A REALISTIC SCALE: ALL FOUR METHODS")
    print("=" * 78)

    d_model = 4096
    num_layers = 32
    num_virtual_tokens = 20
    bottleneck = 64
    lora_r = 16
    lora_num_matrices = 2  # e.g. LoRA applied to the Q and V attention projections

    prompt_tuning = num_virtual_tokens * d_model
    prefix_tuning = num_virtual_tokens * d_model * 2 * num_layers   # x2 for K and V, every layer
    adapters = 2 * d_model * bottleneck * num_layers                # down+up, every layer
    lora = lora_num_matrices * lora_r * (d_model + d_model) * num_layers

    print(f"Config: d_model={d_model}, layers={num_layers}, virtual_tokens={num_virtual_tokens}, "
          f"adapter_bottleneck={bottleneck}, LoRA r={lora_r} on {lora_num_matrices} matrices/layer\n")

    rows = [
        ("Prompt Tuning", prompt_tuning),
        ("Prefix Tuning", prefix_tuning),
        ("Adapters", adapters),
        ("LoRA", lora),
    ]
    print(f"{'method':<16}{'trainable params':>18}{'as % of a 7B model':>22}")
    for name, count in rows:
        pct = 100 * count / 7e9
        print(f"{name:<16}{count:>18,}{pct:>21.4f}%")

    cheapest = min(rows, key=lambda r: r[1])
    priciest = max(rows, key=lambda r: r[1])
    print(f"\n-> {cheapest[0]} is the cheapest here ({cheapest[1]:,} params) because it only ever")
    print(f"   adds parameters ONCE, at the input. {priciest[0]} is the priciest ({priciest[1]:,}")
    print(f"   params) because its parameter count scales with EVERY layer. All four are")
    print("   still a tiny fraction of a 7B-parameter model's own size -- the entire point")
    print("   of every method in this lesson and Lesson 2.")


def main():
    base, final_pretrain_loss = pretrain_base_model()
    print(f"Base model pretrained on generic corpus text. Final pretraining loss: "
          f"{final_pretrain_loss:.4f}\n")
    freeze(base)

    soft_prompt_tuning_demo(base)
    adapter_demo(base)
    parameter_comparison_demo()


if __name__ == "__main__":
    main()
