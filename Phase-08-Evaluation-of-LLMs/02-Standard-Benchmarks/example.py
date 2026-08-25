"""
Standard Benchmarks

Two independent, fully self-contained demos:

  1. Log-probability-based multiple-choice scoring, exactly as real MMLU/
     HellaSwag evaluation harnesses do it: train a tiny character-level
     mini-GPT (same recipe as Phase 02 Lesson 6) on a toy corpus of
     "question -> answer" facts, then, for each toy MMLU-style question,
     score EVERY candidate answer by the model's total log-probability of
     that candidate's characters (teacher-forced) and pick the argmax --
     never asking the model to literally type "A"/"B"/"C"/"D". Because the
     correct answers are deterministically baked into the training corpus,
     the resulting accuracy is verifiably meaningful (it tests whether the
     SCORING METHOD correctly recovers a known association, not whether the
     model is "smart").

  2. The pass@k unbiased estimator (Chen et al., 2021), implemented from
     scratch and sanity-checked two ways: against a brute-force Monte Carlo
     simulation, and against the textbook "at least one success in k i.i.d.
     Bernoulli trials" formula 1 - (1-p)^k in the large-n limit.

Runtime: ~30-60 seconds on a CPU (1500 training steps on a tiny model).

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
# PART 1: log-probability multiple-choice scoring with a real trained model
# ---------------------------------------------------------------------------

# Toy "knowledge base" -- each fact is a (question, correct_answer) pair.
# These are BAKED INTO the training corpus below, so a scoring method that
# works should recover every one of them with certainty.
FACTS = [
    ("what is the capital of france", "paris"),
    ("what is the capital of japan", "tokyo"),
    ("what is the largest planet", "jupiter"),
    ("what is the smallest planet", "mercury"),
    ("what is the chemical symbol for gold", "au"),
    ("who wrote hamlet", "shakespeare"),
]


def fact_sentence(question, answer):
    return f"q: {question}? a: {answer}.\n"


FACT_BLOCK = "".join(fact_sentence(q, a) for q, a in FACTS)
CORPUS = FACT_BLOCK * 40   # repeat so there's enough data, exactly like Phase 02 Lesson 6

chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


data = torch.tensor(encode(CORPUS), dtype=torch.long)

BLOCK_SIZE = 64
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_ITERS = 1500
LEARNING_RATE = 3e-3


def get_batch():
    max_start = len(data) - BLOCK_SIZE - 1
    starts = torch.randint(0, max_start, (BATCH_SIZE,))
    x = torch.stack([data[s:s + BLOCK_SIZE] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + BLOCK_SIZE] for s in starts])
    return x, y


# --- Mini-GPT: identical recipe to Phase 02 Lesson 6 (kept minimal here) ---

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
def score_continuation(model, prompt, continuation):
    """The core of real MMLU/HellaSwag-style harnesses: total log P(continuation | prompt),
    computed with ONE teacher-forced forward pass, never by generating text."""
    full_ids = encode(prompt + continuation)
    prompt_len = len(encode(prompt))
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
    target_ids = full_ids[1:]

    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # target_ids[t] is the token predicted by logits[:, t, :]. The continuation's
    # first character is predicted starting at t = prompt_len - 1.
    total_log_prob = 0.0
    for t in range(prompt_len - 1, len(target_ids)):
        total_log_prob += log_probs[0, t, target_ids[t]].item()
    return total_log_prob


def build_mc_questions():
    """Turn each fact into a toy MMLU-style multiple-choice question: the
    correct answer plus 3 distractors borrowed from the OTHER facts' answers
    (so every option is a real word the model has seen, just attached to a
    different question -- exactly what makes multiple choice non-trivial)."""
    all_answers = [a for _, a in FACTS]
    questions = []
    for question, correct in FACTS:
        distractors = [a for a in all_answers if a != correct]
        random.shuffle(distractors)
        options = [correct] + distractors[:3]
        random.shuffle(options)
        questions.append((question, correct, options))
    return questions


def multiple_choice_demo():
    print("=" * 78)
    print("1. LOG-PROBABILITY-BASED MULTIPLE-CHOICE SCORING (how MMLU is really graded)")
    print("=" * 78)
    print(f"Training corpus: {len(FACTS)} facts, repeated into {len(CORPUS)} characters.")
    print(f"Vocabulary: {vocab_size} unique characters.\n")

    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training the mini-GPT on next-token prediction over the fact corpus...")
    for step in range(1, NUM_ITERS + 1):
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:5d}  loss = {loss.item():.4f}")

    print("\nEvaluating: for each question, score EVERY candidate answer's total")
    print("log-probability given the shared question prefix, and pick the argmax.\n")

    questions = build_mc_questions()
    num_correct = 0
    for question, correct_answer, options in questions:
        prompt = f"q: {question}? a:"
        scores = {}
        for option in options:
            continuation = f" {option}."
            scores[option] = score_continuation(model, prompt, continuation)

        predicted = max(scores, key=scores.get)
        is_correct = predicted == correct_answer
        num_correct += int(is_correct)

        print(f"  Q: {question}?")
        for option in options:
            marker = " <-- picked" if option == predicted else ""
            gold = " (correct)" if option == correct_answer else ""
            print(f"      log P({option!r} | prompt) = {scores[option]:8.2f}{marker}{gold}")
        print(f"    {'CORRECT' if is_correct else 'WRONG'}\n")

    accuracy = num_correct / len(questions)
    print(f"Toy-MMLU accuracy: {num_correct}/{len(questions)} = {accuracy:.0%}")
    print("\n-> The model was NEVER asked to output a letter or generate free text --")
    print("   every candidate's full continuation was scored by the SAME model in")
    print("   teacher-forced mode, and the highest-log-probability option won. Since")
    print("   the correct associations were literally memorized during training, this")
    print("   accuracy number verifies the SCORING METHOD works correctly (it correctly")
    print("   recovers a known ground truth), which is exactly how lm-evaluation-harness")
    print("   scores real MMLU/HellaSwag questions against real trained LLMs.")


# ---------------------------------------------------------------------------
# PART 2: the pass@k unbiased estimator (Chen et al., 2021)
# ---------------------------------------------------------------------------

def pass_at_k(n, c, k):
    """1 - P(a random size-k subset of the n samples contains ZERO correct ones)."""
    if n - c < k:
        return 1.0   # fewer than k samples are wrong, so any subset must include a correct one
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def monte_carlo_pass_at_k(n, c, k, trials=200_000):
    """Brute-force sanity check: actually draw random size-k subsets (without
    replacement) from n samples (c of them marked correct) and measure the
    empirical fraction containing at least one correct sample."""
    items = [True] * c + [False] * (n - c)   # True = correct
    successes = 0
    for _ in range(trials):
        subset = random.sample(items, k)
        successes += any(subset)
    return successes / trials


def pass_at_k_demo():
    print("\n" + "=" * 78)
    print("2. THE pass@k UNBIASED ESTIMATOR: pass@k = 1 - C(n-c, k) / C(n, k)")
    print("=" * 78)

    print(f"{'n':>5}{'c':>5}{'k':>5}{'formula':>12}{'monte carlo':>14}")
    cases = [(10, 1, 1), (10, 5, 1), (10, 2, 5), (100, 10, 10), (20, 15, 3)]
    for n, c, k in cases:
        formula_val = pass_at_k(n, c, k)
        mc_val = monte_carlo_pass_at_k(n, c, k, trials=50_000)
        print(f"{n:>5}{c:>5}{k:>5}{formula_val:>12.4f}{mc_val:>14.4f}")

    print("\n-> The closed-form formula and the brute-force Monte Carlo estimate agree")
    print("   to within simulation noise on every case -- confirming the formula really")
    print("   does compute 'probability at least one of a random k-sample subset passed'.")

    print("\n" + "-" * 78)
    print("Sanity check: as n grows with p = c/n held fixed, pass@k should converge to")
    print("the textbook 'at least one success in k i.i.d. Bernoulli(p) trials' formula:")
    print("    1 - (1 - p)^k")
    print("(sampling k out of a large finite pool without replacement behaves more and")
    print("more like k independent draws as the pool size n grows).\n")

    p, k = 0.2, 3
    binomial_limit = 1 - (1 - p) ** k
    print(f"p = c/n = {p}, k = {k}  ->  binomial formula 1-(1-p)^k = {binomial_limit:.4f}\n")
    print(f"{'n':>10}{'c = p*n':>10}{'pass@k (formula)':>20}")
    for n in [10, 100, 1_000, 100_000]:
        c = int(round(p * n))
        print(f"{n:>10}{c:>10}{pass_at_k(n, c, k):>20.4f}")

    print(f"\n-> As n grows (with the same ratio c/n = {p}), the finite-pool pass@k formula")
    print(f"   converges toward the i.i.d.-Bernoulli value {binomial_limit:.4f} -- exactly the")
    print("   sanity check we'd want: pass@k IS 'probability of at least one success,' just")
    print("   computed exactly for a finite, already-drawn sample instead of assuming")
    print("   infinite independent draws.")


def main():
    multiple_choice_demo()
    pass_at_k_demo()


if __name__ == "__main__":
    main()
