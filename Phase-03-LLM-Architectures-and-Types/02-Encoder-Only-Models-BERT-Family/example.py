"""
Encoder-Only Models: the BERT Family

A tiny BERT-style encoder-only Transformer trained from scratch with the
real Masked Language Modeling recipe (80/10/10 masking, loss only on
masked positions) -- a direct contrast with Phase 02's causal
next-token-prediction training of the exact same architecture family,
minus the causal mask.

Runtime: ~30-60 seconds on a CPU.

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
# 0. Toy corpus, word-level vocabulary, and special tokens
# ---------------------------------------------------------------------------

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog barks at the quick brown fox",
    "the small dog barks at the tall cat",
    "the tall cat runs away from the small dog",
    "the quick brown fox runs into the dark forest",
    "birds sing songs in the tall green trees",
    "the sun rises over the dark forest every morning",
    "the moon shines over the dark forest every night",
]
SENTENCES = SENTENCES * 30  # repeat for enough training data

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
words = sorted({w for s in SENTENCES for w in s.split()})
vocab = SPECIALS + words
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
PAD, CLS, SEP, MASK = (stoi[t] for t in SPECIALS)

MAX_LEN = 12  # [CLS] + up to 10 words + [SEP]


def encode_sentence(sentence):
    tokens = [CLS] + [stoi[w] for w in sentence.split()] + [SEP]
    tokens = tokens[:MAX_LEN]
    tokens = tokens + [PAD] * (MAX_LEN - len(tokens))
    return tokens


# ---------------------------------------------------------------------------
# 1. BERT-style masking: 80% [MASK], 10% random token, 10% unchanged
# ---------------------------------------------------------------------------

def apply_mlm_masking(token_ids, mask_prob=0.15):
    """Returns (corrupted_ids, labels). labels[i] = -100 (ignored by the loss)
    everywhere EXCEPT the positions chosen for masking, where it holds the
    true original token id."""
    corrupted = list(token_ids)
    labels = [-100] * len(token_ids)

    maskable_positions = [i for i, t in enumerate(token_ids) if t not in (PAD, CLS, SEP)]
    num_to_mask = max(1, int(round(len(maskable_positions) * mask_prob)))
    chosen = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))

    for i in chosen:
        labels[i] = token_ids[i]
        roll = random.random()
        if roll < 0.8:
            corrupted[i] = MASK
        elif roll < 0.9:
            corrupted[i] = random.randrange(len(SPECIALS), vocab_size)  # a random real word
        # else: leave unchanged (10% of the time)
    return corrupted, labels


# ---------------------------------------------------------------------------
# 2. A minimal bidirectional (no causal mask!) encoder-only Transformer
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask):
        batch, T, d_model = x.shape

        def split_heads(t):
            return t.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.W_q(x)), split_heads(self.W_k(x)), split_heads(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # pad_mask: (batch, 1, 1, T) -- broadcast over heads and query positions.
        # NO causal mask here at all: every position may attend to every
        # other position, before AND after it. This is the entire
        # architectural difference from Phase 02's decoder.
        scores = scores.masked_fill(pad_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, T, d_model)
        return self.W_o(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyBERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        pad_mask = (token_ids != PAD).view(batch, 1, 1, T)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.final_norm(x)
        return self.mlm_head(x)   # (batch, T, vocab_size)


def get_batch(batch_size):
    chosen = random.sample(SENTENCES, batch_size) if batch_size <= len(SENTENCES) \
        else random.choices(SENTENCES, k=batch_size)
    inputs, labels = [], []
    for s in chosen:
        ids = encode_sentence(s)
        corrupted, lbl = apply_mlm_masking(ids)
        inputs.append(corrupted)
        labels.append(lbl)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main():
    print(f"Vocabulary size (incl. specials): {vocab_size}")

    model = TinyBERT(vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=3, max_len=MAX_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print("\n" + "=" * 70)
    print("TRAINING WITH MASKED LANGUAGE MODELING (loss only on masked positions)")
    print("=" * 70)
    for step in range(1, 1501):
        token_ids, labels = get_batch(batch_size=16)
        logits = model(token_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("FILL-IN-THE-MASK PREDICTIONS AFTER TRAINING")
    print("=" * 70)
    test_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the small dog barks at the tall cat",
    ]
    model.eval()
    with torch.no_grad():
        for sentence in test_sentences:
            words_list = sentence.split()
            mask_idx = random.randrange(len(words_list))
            true_word = words_list[mask_idx]
            words_list[mask_idx] = "[MASK]"

            ids = [CLS] + [stoi[w] for w in words_list] + [SEP]
            ids = ids[:MAX_LEN] + [PAD] * (MAX_LEN - len(ids))
            token_ids = torch.tensor([ids])

            logits = model(token_ids)
            mask_position = 1 + mask_idx  # offset by [CLS]
            predicted_id = logits[0, mask_position].argmax().item()
            predicted_word = itos[predicted_id]

            masked_sentence = " ".join(words_list)
            print(f"  input:     {masked_sentence!r}")
            print(f"  true word: {true_word!r}   predicted: {predicted_word!r}\n")

    print("-> Unlike Phase 02's causal mini-GPT, this model was free to use BOTH")
    print("   the words before AND after the masked position to make its guess --")
    print("   that bidirectional context is exactly what a decoder-only model,")
    print("   with its causal mask, structurally cannot do.")


if __name__ == "__main__":
    main()
