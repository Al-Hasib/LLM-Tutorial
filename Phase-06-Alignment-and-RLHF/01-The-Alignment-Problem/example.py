"""
The Alignment Problem

Demonstrates the pretraining-vs-alignment gap CONCRETELY rather than just
asserting it: trains a tiny decoder-only GPT (the exact mini-GPT architecture
from Phase 02 Lesson 6) on a small, deliberately raw and UNSTRUCTURED toy
corpus -- a mix of plain statements and questions, with no instruction
formatting and, crucially, no example ANYWHERE in the data of a question
being directly answered. We then prompt the trained model with a real
factual question and show it does NOT answer it -- it continues with
another question-shaped line or an unrelated statement, because that is
literally the only kind of continuation its training data ever contained.
This is not a "the model is too small/dumb" artifact; it is a direct,
mechanical illustration of the alignment gap: next-token prediction only
ever learns to match the statistics of what it was shown.

Runtime: ~1 minute on a CPU (1500 training steps on a small character-level
model).

Run:
    python example.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
random.seed(1337)

# ---------------------------------------------------------------------------
# 0. A raw, unstructured, "internet-like" toy corpus.
#
# This is the crux of the demo: QUESTION_LINES and STATEMENT_LINES are mixed
# together in random order, over and over. A question is therefore followed,
# across the many repeats, sometimes by another question and sometimes by an
# unrelated statement -- but NEVER by a direct answer to that specific
# question, because no such answer exists anywhere in this corpus. The model
# can only ever learn "what kind of line tends to follow a question" (another
# question-shaped line, or a statement), never "what is the factual answer."
# ---------------------------------------------------------------------------

QUESTION_LINES = [
    "what is the capital of france?",
    "what is the best programming language?",
    "how do i learn to code faster?",
    "why is the sky blue during the day?",
    "what is the meaning of life?",
    "how does gravity actually work?",
    "what should i eat for dinner tonight?",
    "why do cats sleep so much every day?",
    "how do airplanes stay up in the air?",
    "what is the tallest mountain on earth?",
]

STATEMENT_LINES = [
    "i think it might rain again today.",
    "the weather has been nice this week.",
    "my cat slept on the couch all afternoon.",
    "i went to the store yesterday morning.",
    "the movie last night was pretty good.",
    "the coffee shop downtown closes early on sundays.",
    "she walked the dog around the block twice.",
    "the garden needs more water this summer.",
    "he forgot his umbrella again this morning.",
    "the train was late by almost ten minutes.",
]

ALL_LINES = QUESTION_LINES + STATEMENT_LINES


def build_corpus(num_blocks=40):
    """Each 'block' is all 20 lines in a freshly shuffled order, joined by
    newlines. Repeating with a different shuffle each time gives the model
    many (question -> {question, statement}) transitions to learn general
    STATISTICS from, rather than letting it memorize one fixed sequence --
    while still never containing a single (question -> direct answer) pair."""
    blocks = []
    for _ in range(num_blocks):
        lines = ALL_LINES.copy()
        random.shuffle(lines)
        blocks.append("\n".join(lines))
    return ("\n".join(blocks) + "\n").lower()


CORPUS = build_corpus()

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
# Hyperparameters -- same scale philosophy as Phase 02 Lesson 6.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 48
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
BATCH_SIZE = 32
NUM_ITERS = 1500
LEARNING_RATE = 3e-3


def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


# ---------------------------------------------------------------------------
# 1. The model -- identical architecture to Phase 02 Lesson 6's MiniGPT
# (decoder-only: token + positional embedding, N causal self-attention +
# FFN blocks, final LayerNorm, linear head over the vocabulary).
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids

    @torch.no_grad()
    def next_char_distribution(self, token_ids, top_k=6):
        context = token_ids[:, -self.block_size:]
        logits, _ = self(context)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        return [(itos[i.item()], p.item()) for p, i in zip(top_probs, top_ids)]


def main():
    print("=" * 70)
    print("0. THE TOY CORPUS -- raw, unstructured, no instruction formatting")
    print("=" * 70)
    print(f"{len(QUESTION_LINES)} question-shaped lines, {len(STATEMENT_LINES)} plain")
    print("statement lines, shuffled together into 40 blocks. Sample of the raw")
    print("corpus (first 5 lines of one block):")
    for line in CORPUS.split("\n")[:5]:
        print(f"  {line}")
    print(f"\nCorpus length: {len(CORPUS)} characters, vocabulary: {vocab_size} unique characters")
    print("Crucially: NOT ONE line in this corpus is a direct factual answer to a")
    print("question. There is no '...france? paris.' anywhere in the data.")

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameter count: {num_params:,} (same architecture as Phase 02 Lesson 6)")

    print("\n" + "=" * 70)
    print("TRAINING (plain next-token prediction -- exactly like pretraining)")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, NUM_ITERS + 1):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\n-> Loss falls the same way it did in Phase 02 Lesson 6: the model IS")
    print("   learning the statistics of its training data well. The question this")
    print("   demo asks is not 'can it learn?' but 'learn to do WHAT, exactly?'")

    print("\n" + "=" * 70)
    print("1. PROMPTING THE 'BASE MODEL' WITH A DIRECT QUESTION")
    print("=" * 70)
    prompt = "what is the capital of france?\n"
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long)

    print(f"Prompt fed to the model: {prompt!r}")
    print("\nA helpful assistant would continue this with something like 'paris.'")
    print("Let's look at what the model actually thinks should come next.\n")

    top_next = model.next_char_distribution(prompt_ids)
    print("Model's actual top-6 predicted next CHARACTERS (character, probability):")
    for ch, p in top_next:
        print(f"  {ch!r:>6}  {p:.3f}")
    print("\n-> Notice there is no unusual spike on 'p' (which would start 'paris').")
    print("   The top candidates are the same characters ('w', 'h', 't', 'i', ...)")
    print("   that start the OTHER lines in the corpus (other questions, or plain")
    print("   statements) -- because that is the only pattern this training data")
    print("   ever contained after a line ending in '?'.")

    generated = model.generate(prompt_ids, max_new_tokens=90)
    completion = decode(generated[0].tolist())[len(prompt):]
    print(f"\nFull generated continuation after the question:\n  {completion!r}")

    contains_paris = "paris" in completion.lower()
    print(f"\nDoes the continuation contain the word 'paris'? {contains_paris}")
    if not contains_paris:
        print("-> As expected: the word 'paris' never once appears anywhere in the")
        print("   training corpus, so the model has no statistical basis to produce")
        print("   it. Instead it continues with more question-shaped or statement-")
        print("   shaped text -- reproducing the ONLY behavior it was ever shown.")
    else:
        print("-> Even in the rare case some 'paris'-like character run appears by")
        print("   chance, it would not be a genuine 'answer' -- the model has no")
        print("   concept of answering; any resemblance is coincidental sampling.")

    print("\n" + "=" * 70)
    print("2. A SECOND EXAMPLE, WITH A QUESTION-WORD PROMPT")
    print("=" * 70)
    prompt2 = "why do cats sleep so much every day?\n"
    prompt2_ids = torch.tensor([encode(prompt2)], dtype=torch.long)
    generated2 = model.generate(prompt2_ids, max_new_tokens=90)
    completion2 = decode(generated2[0].tolist())[len(prompt2):]
    print(f"Prompt: {prompt2!r}")
    print(f"Continuation: {completion2!r}")
    starts_with_question_word = any(
        completion2.strip().startswith(w) for w in ("what", "how", "why", "the", "i ", "she", "he", "my")
    )
    print(f"\nDoes the continuation begin like one of the corpus's own lines "
          f"(a question or a statement)? {starts_with_question_word}")
    print("-> Either way, what it does NOT do is explain feline sleep biology --")
    print("   because nothing resembling that ever appeared in its training data.")

    print("\n" + "=" * 70)
    print("CONCLUSION: THE ALIGNMENT GAP, MADE MECHANICAL")
    print("=" * 70)
    print("This model trained successfully (loss dropped, as printed above) and it")
    print("faithfully reproduces the STATISTICS of its training corpus. But being a")
    print("good next-token predictor and being a helpful question-answerer are two")
    print("different behaviors, and only one of them was ever in the training data.")
    print("This is the entire motivation for the rest of this phase: reward")
    print("modeling, RLHF, DPO, and RLAIF/Constitutional AI all exist specifically")
    print("to teach a model the SECOND behavior, on top of the first.")


if __name__ == "__main__":
    main()
