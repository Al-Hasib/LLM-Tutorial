"""
Pretraining a Small LLM From Scratch

The full recipe from this phase, assembled into one run:
  1. A tiny data-pipeline step (quality filter + exact dedup) over a
     deliberately messy raw corpus, before training ever starts.
  2. A train/validation split, with BOTH losses tracked throughout.
  3. AdamW + warmup/cosine learning-rate schedule + gradient clipping.
  4. Generation samples captured at several checkpoints across training,
     so improvement is something you watch happen.

Runtime: ~1-2 minutes on a CPU.

Run:
    python example.py
"""

import hashlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. A deliberately messy raw corpus, and a tiny pipeline step to clean it
#    (Lesson 1: quality filtering + exact deduplication)
# ---------------------------------------------------------------------------

CLEAN_DOCUMENTS = [
    "the old lighthouse stood on the rocky cliff, its light sweeping the dark sea.",
    "every night the keeper climbed the spiral stairs to light the lamp.",
    "ships far out at sea watched for that light to find their way home.",
    "storms battered the coast but the lighthouse always held its ground.",
    "the keeper kept a logbook of every ship that passed safely through the strait.",
    "children from the village loved to hear the keeper's stories of the sea.",
    "one winter the keeper spotted a small boat lost in the fog and rang the bell.",
    "the sailors rowed toward the sound of the bell until they reached the shore.",
    "the lighthouse still stands today, though its lamp has long since gone electric.",
    "gulls circled the tower each morning, calling out over the crashing waves.",
    "the keeper repainted the tower white every spring before the tourists arrived.",
    "on clear nights the light could be seen for miles across the open water.",
    "fishermen trusted the lighthouse more than any chart or compass they owned.",
    "a young apprentice came to learn the keeper's trade the following summer.",
]
# Held out ENTIRELY from training -- the model never sees these documents in any
# form, so validation loss measures genuine generalization, not a text fragment.
VALIDATION_DOCUMENTS = [
    "the museum in town now displays the lighthouse's original brass lamp.",
    "visitors climb the same worn stairs the keeper once climbed every night.",
]

RAW_DOCUMENTS = CLEAN_DOCUMENTS + [
    CLEAN_DOCUMENTS[0],   # exact duplicate
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",     # junk: no real content
    "###???!!!###???!!!###???!!!###???!!!###???!!!###???!!!###???!!!###???!!!",         # junk: symbol spam
    CLEAN_DOCUMENTS[1],   # exact duplicate
    CLEAN_DOCUMENTS[0],   # exact duplicate (again)
    "hi",                                                                              # junk: too short
]
RAW_DOCUMENTS = RAW_DOCUMENTS * 4   # repeat the (still-messy) raw set to give the pipeline demo enough volume


def passes_quality_filter(doc, min_length=15, max_symbol_ratio=0.3, max_repeat_ratio=0.4):
    """Three simple heuristic rules, matching Lesson 1 section 3: reject
    documents that are too short to carry real content, dominated by
    non-alphanumeric symbols rather than actual words, or dominated by a
    single repeated character (e.g. "aaaa...a") -- which is alphanumeric
    and so would slip past a symbol-ratio check alone."""
    if len(doc) < min_length:
        return False
    symbol_count = sum(1 for ch in doc if not (ch.isalnum() or ch.isspace()))
    if symbol_count / len(doc) > max_symbol_ratio:
        return False
    most_common_count = max(doc.count(ch) for ch in set(doc))
    if most_common_count / len(doc) > max_repeat_ratio:
        return False
    return True


def deduplicate_exact(docs):
    """Exact deduplication via hashing (Lesson 1 section 4) -- keeps the
    first occurrence of each distinct document, drops the rest."""
    seen_hashes = set()
    unique_docs = []
    for doc in docs:
        doc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)
    return unique_docs


def run_data_pipeline(raw_docs):
    filtered = [d for d in raw_docs if passes_quality_filter(d)]
    deduped = deduplicate_exact(filtered)
    return deduped


# ---------------------------------------------------------------------------
# 2. Tokenizer, train/validation split
# ---------------------------------------------------------------------------

def build_vocab(text):
    chars = sorted(set(text))
    return chars, {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}


BLOCK_SIZE = 32
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
BATCH_SIZE = 32
NUM_ITERS = 1200
MAX_LR = 3e-3
MIN_LR = 3e-4
WARMUP_STEPS = 150
GRAD_CLIP_NORM = 1.0


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def warmup_cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """The exact schedule from Lesson 4 section 4."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# 3. The decoder-only model (same recipe as Phase 02 Lesson 6)
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


class MiniGPT(nn.Module):
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

    def forward(self, token_ids, targets=None):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


def main():
    print("=" * 70)
    print("1. THE DATA PIPELINE: QUALITY FILTER + EXACT DEDUPLICATION")
    print("=" * 70)
    unique_raw_count = len(set(RAW_DOCUMENTS))
    cleaned_docs = run_data_pipeline(RAW_DOCUMENTS)
    print(f"Raw documents (with repeats of the messy set): {len(RAW_DOCUMENTS)}")
    print(f"Distinct raw documents before any cleaning:    {unique_raw_count}")
    print(f"Documents surviving filter + dedup:            {len(cleaned_docs)}")
    print("\nSurvivors:")
    for d in cleaned_docs:
        print(f"  - {d}")

    train_text = " ".join(cleaned_docs)
    val_text = " ".join(VALIDATION_DOCUMENTS)
    # Vocabulary is built from BOTH so validation text never hits an unknown
    # character, but the validation DOCUMENTS themselves are never trained on.
    chars, stoi, itos = build_vocab(train_text + val_text)
    vocab_size = len(chars)
    train_data = torch.tensor([stoi[ch] for ch in train_text], dtype=torch.long)
    val_data = torch.tensor([stoi[ch] for ch in val_text], dtype=torch.long)
    print(f"\nTraining corpus:   {len(train_text)} characters, {vocab_size} unique characters")
    print(f"Validation corpus: {len(val_text)} characters, from "
          f"{len(VALIDATION_DOCUMENTS)} documents NEVER seen during training")

    print("\n" + "=" * 70)
    print("2. MODEL AND TRAINING SETUP")
    print("=" * 70)
    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {num_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

    @torch.no_grad()
    def estimate_val_loss(num_batches=10):
        model.eval()
        losses = []
        for _ in range(num_batches):
            x, y = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    prompt_ids = torch.tensor([[stoi[ch] for ch in "the "]], dtype=torch.long)
    checkpoints = {}

    print("\n" + "=" * 70)
    print("3. TRAINING (AdamW + warmup/cosine schedule + gradient clipping)")
    print("=" * 70)
    for step in range(1, NUM_ITERS + 1):
        lr = warmup_cosine_lr(step, MAX_LR, MIN_LR, WARMUP_STEPS, NUM_ITERS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, train_loss = model(x, y)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if step % 300 == 0 or step == 1:
            val_loss = estimate_val_loss()
            print(f"  step {step:5d}  lr={lr:.5f}  train_loss={train_loss.item():.4f}  "
                  f"val_loss={val_loss:.4f}")

        if step in (1, NUM_ITERS // 2, NUM_ITERS):
            generated = model.generate(prompt_ids.clone(), max_new_tokens=60)
            checkpoints[step] = "".join(itos[i] for i in generated[0].tolist())

    print("\n" + "=" * 70)
    print("4. GENERATION QUALITY ACROSS CHECKPOINTS (same prompt: 'the ')")
    print("=" * 70)
    for step, sample in checkpoints.items():
        print(f"  step {step:5d}: {sample!r}")

    final_train_loss = train_loss.item()
    final_val_loss = estimate_val_loss(num_batches=20)
    print("\n" + "=" * 70)
    print("5. FINAL TRAIN vs. VALIDATION LOSS")
    print("=" * 70)
    print(f"  final train loss:      {final_train_loss:.4f}")
    print(f"  final validation loss: {final_val_loss:.4f}")
    gap = final_val_loss - final_train_loss
    print(f"  gap (val - train):     {gap:+.4f}")
    if gap > 0.3:
        print("\n-> A meaningfully positive gap: the model fits the training text")
        print("   noticeably better than the held-out validation text -- some genuine")
        print("   overfitting, expected with a corpus this small and this repetitive.")
    else:
        print("\n-> A small gap: the model's performance on held-out text tracks its")
        print("   training performance reasonably closely -- not memorizing much more")
        print("   than the tiny, repetitive corpus itself allows.")


if __name__ == "__main__":
    main()
