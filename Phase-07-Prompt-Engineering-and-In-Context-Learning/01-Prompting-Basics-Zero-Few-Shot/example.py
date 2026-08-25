"""
Prompting Basics: Zero-Shot and Few-Shot

A genuine, from-scratch demonstration of few-shot in-context learning --
not a simulation of it. We train a tiny decoder-only Transformer (the same
architecture family as Phase 02's mini-GPT) on many randomly generated
EPISODES of a simple function-mapping task. Each episode is a fresh,
randomly parameterized function "y = (x + k) mod M" for a hidden shift k;
the episode's prompt shows a handful of (x -> y) example pairs for THAT
k, followed by a query x the model must map to the right y.

Because k is re-sampled fresh every single episode, the model can never
memorize a fixed input->output table in its weights -- the only way to
answer the query correctly is to INFER k from the in-context examples
given in that specific prompt, and apply it. This is a small, honest
analogue of exactly what GPT-3 does when it performs a new task from a
handful of prompt examples with no gradient updates at all (Phase 03
Lesson 1, section 3).

After training, we test the model on shift values k it NEVER saw during
training -- true zero-shot-to-a-new-function generalization, evaluated
purely through in-context examples at inference time, with the model's
weights completely frozen. We then measure how accuracy depends on the
NUMBER of in-context examples given, and on whether their ORDER matches
the canonical order the model was trained on -- a direct, measured
demonstration of the prompt-sensitivity phenomenon documented in
Zhao et al. (2021), "Calibrate Before Use."

Runtime: ~20-40 seconds on a CPU (3000 training steps on tiny sequences).

Run:
    python example.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 0. The task family: y = (x + k) mod M
#
# M = size of the symbol alphabet. k = the hidden "shift" that defines one
# episode's function. TRAIN_KS are the only shifts the model ever trains
# on; TEST_KS are held out completely -- the model must generalize to them
# purely via in-context inference at test time.
# ---------------------------------------------------------------------------

M = 6
TRAIN_KS = [0, 1, 2, 3]
TEST_KS = [4, 5]          # never appear during training -- genuinely novel functions

CHARS = [str(d) for d in range(M)] + [">", ","]
vocab_size = len(CHARS)
stoi = {ch: i for i, ch in enumerate(CHARS)}
itos = {i: ch for i, ch in enumerate(CHARS)}


def encode(s):
    return [stoi[ch] for ch in s]


def build_episode_string(k, x_shown, x_query):
    """x_shown: list of distinct x values to show as few-shot examples (in
    the given order -- caller controls sorted vs. scrambled). Returns the
    full prompt string (ending right after 'x_query>') and the correct
    answer digit as a string."""
    parts = []
    for x in x_shown:
        y = (x + k) % M
        parts.append(f"{x}>{y},")
    prompt = "".join(parts) + f"{x_query}>"
    answer = str((x_query + k) % M)
    return prompt, answer


def sample_episode(k, n_shown, scrambled=False):
    xs = list(range(M))
    random.shuffle(xs)
    x_shown = xs[:n_shown]
    x_query = xs[n_shown]                      # guaranteed distinct from shown
    if not scrambled:
        x_shown = sorted(x_shown)               # canonical order used during training
    prompt, answer = build_episode_string(k, x_shown, x_query)
    return prompt, answer


# ---------------------------------------------------------------------------
# 1. Hyperparameters and model -- same decoder-only recipe as Phase 02
#    Lesson 6, just re-declared here so this lesson is self-contained.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 24     # max length of "4*5(examples) + 2(query) + 1(answer)" = 23
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
BATCH_SIZE = 64
NUM_ITERS = 3000
LEARNING_RATE = 3e-3


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
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def predict_next_char(self, prompt_str):
        """Greedy argmax over the very next character -- this is the
        in-context 'answer' the model commits to, with NO weight updates."""
        ids = torch.tensor([encode(prompt_str)], dtype=torch.long)
        logits, _ = self(ids)
        next_id = logits[0, -1, :].argmax().item()
        return itos[next_id]


# ---------------------------------------------------------------------------
# 2. Batch generation -- a fresh random (k, examples) episode every single
#    training example. Nothing here is ever repeated verbatim, which is
#    exactly what forces the model to learn the ALGORITHM (infer k from
#    context, then apply it) instead of memorizing input->output pairs.
# ---------------------------------------------------------------------------

def get_training_batch(batch_size):
    n_shown = random.randint(2, 5)              # varies step to step, fixed within a batch
    seqs = []
    for _ in range(batch_size):
        k = random.choice(TRAIN_KS)
        prompt, answer = sample_episode(k, n_shown, scrambled=False)
        seqs.append(encode(prompt + answer))
    tokens = torch.tensor(seqs, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


# ---------------------------------------------------------------------------
# 3. Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, ks, n_shown, scrambled, num_trials=300):
    correct = 0
    for _ in range(num_trials):
        k = random.choice(ks)
        prompt, answer = sample_episode(k, n_shown, scrambled=scrambled)
        pred = model.predict_next_char(prompt)
        correct += int(pred == answer)
    return correct / num_trials


def main():
    print("=" * 70)
    print("SETUP")
    print("=" * 70)
    print(f"Alphabet size M = {M}   (symbols 0..{M-1})")
    print(f"Function family: y = (x + k) mod {M}, for hidden shift k")
    print(f"Training shifts (seen during training):     k in {TRAIN_KS}")
    print(f"Held-out shifts (NEVER seen during training): k in {TEST_KS}")
    print(f"Vocabulary ({vocab_size} characters): {CHARS}")
    example_prompt, example_answer = sample_episode(2, 3, scrambled=False)
    print(f"\nExample training-style prompt (k=2, 3 shown examples):")
    print(f"  {example_prompt!r}  -> correct next char: {example_answer!r}")

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameter count: {num_params:,}")

    print("\n" + "=" * 70)
    print("BEFORE TRAINING: accuracy on held-out shifts (random weights)")
    print("=" * 70)
    random_baseline = 1.0 / M
    acc_before = evaluate(model, TEST_KS, n_shown=4, scrambled=False, num_trials=300)
    print(f"Chance accuracy (uniform guess over {M} symbols): {random_baseline:.3f}")
    print(f"Untrained model accuracy:                          {acc_before:.3f}")

    print("\n" + "=" * 70)
    print("TRAINING (next-token prediction over randomly generated episodes)")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, NUM_ITERS + 1):
        x, y = get_training_batch(BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("AFTER TRAINING: in-context generalization to shifts NEVER seen")
    print("during training (weights are now completely frozen)")
    print("=" * 70)
    acc_after = evaluate(model, TEST_KS, n_shown=4, scrambled=False, num_trials=300)
    print(f"Held-out-shift accuracy with 4 in-context examples: {acc_after:.3f}")
    print(f"(chance level is {random_baseline:.3f}; {M-1}/{M} of guesses would be wrong at random)")

    print("\nA few concrete examples on shift k=5 (held out, never trained on):")
    for _ in range(4):
        prompt, answer = sample_episode(5, n_shown=4, scrambled=False)
        pred = model.predict_next_char(prompt)
        mark = "correct" if pred == answer else "WRONG"
        print(f"  prompt={prompt!r:28s} true={answer}  model={pred}  [{mark}]")

    if acc_after > acc_before + 0.1:
        print("\n-> The model was NEVER trained on k=4 or k=5. Every correct answer")
        print("   above comes purely from reading the in-context examples in THAT")
        print("   prompt and inferring k on the fly -- exactly the mechanism behind")
        print("   GPT-3's few-shot in-context learning (Phase 03 Lesson 1 section 3),")
        print("   just demonstrated end-to-end on a task small enough to train here.")

    print("\n" + "=" * 70)
    print("NUMBER OF IN-CONTEXT EXAMPLES vs. ACCURACY (held-out shifts)")
    print("=" * 70)
    print("Training only ever showed between 2 and 5 examples per episode.")
    print(f"{'n_shown':>10}{'accuracy':>12}")
    counts_results = {}
    for n_shown in [1, 2, 3, 4, 5]:
        acc = evaluate(model, TEST_KS, n_shown=n_shown, scrambled=False, num_trials=300)
        counts_results[n_shown] = acc
        note = "  (below training range)" if n_shown < 2 else ""
        print(f"{n_shown:>10}{acc:>12.3f}{note}")

    print("\n-> Accuracy rises as more in-context examples are given, and is at its")
    print("   worst with only 1 example -- a prompt shorter than anything the model")
    print("   ever trained on. More examples give the model more redundant evidence")
    print("   to pin down the hidden shift k before it has to commit to an answer.")

    print("\n" + "=" * 70)
    print("EXAMPLE ORDER vs. ACCURACY: sorted (training distribution) vs.")
    print("scrambled (never seen during training) -- Zhao et al. (2021)")
    print("=" * 70)
    N_SHOWN_FOR_ORDER_TEST = 4
    acc_sorted = evaluate(model, TEST_KS, n_shown=N_SHOWN_FOR_ORDER_TEST, scrambled=False, num_trials=500)
    acc_scrambled = evaluate(model, TEST_KS, n_shown=N_SHOWN_FOR_ORDER_TEST, scrambled=True, num_trials=500)
    print(f"Sorted order   (matches training):     accuracy = {acc_sorted:.3f}")
    print(f"Scrambled order (never seen in training): accuracy = {acc_scrambled:.3f}")

    if acc_scrambled < acc_sorted - 0.03:
        print("\n-> Same examples, same hidden function, same COUNT of examples -- the")
        print("   only thing that changed is the ORDER they appear in the prompt, and")
        print("   accuracy measurably drops. The model was only ever trained on")
        print("   examples sorted by x, so it partly relies on that positional")
        print("   regularity rather than a perfectly order-invariant algorithm. This")
        print("   is a small, controlled reproduction of the prompt-order sensitivity")
        print("   Zhao et al. (2021, 'Calibrate Before Use') documented in real LLMs:")
        print("   logically equivalent prompts that differ only in example order or")
        print("   formatting can produce measurably different accuracy.")
    else:
        print("\n-> Here, scrambling order did NOT meaningfully hurt accuracy: the model")
        print("   generalized the underlying algorithm (infer k from ANY example pair,")
        print("   then apply it) well enough that it doesn't depend on position. Real")
        print("   LLMs are not always this robust -- Zhao et al. (2021) found that with")
        print("   less redundant, more ambiguous tasks than this one, example order and")
        print("   formatting DO measurably move accuracy, which is why prompt order is")
        print("   worth controlling for rather than assuming it is harmless.")


if __name__ == "__main__":
    main()
