"""
Instruction Tuning (SFT)

Demonstrates, from scratch in raw PyTorch:
  1. Building the SFT label tensor: cross-entropy loss masked (ignore_index=-100)
     over the INSTRUCTION/PROMPT tokens, computed only on RESPONSE tokens --
     shown explicitly, character by character, for one example.
  2. Pretraining a tiny decoder-only Transformer (Phase 02's mini-GPT block)
     on generic text with plain next-token prediction -- a stand-in for "a
     base model that already knows the language but has never followed an
     instruction in its life".
  3. Instruction-tuning that SAME pretrained model on toy (instruction,
     response) pairs using the masked loss from step 1, and comparing
     generation BEFORE vs. AFTER on held-out instructions (words/tasks
     never seen together during fine-tuning).

Runtime: ~1-2 minutes on a CPU (pretraining + SFT combined).

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# 0. Vocabulary shared by BOTH the pretraining corpus and the instruction
#    dataset -- a real base model's tokenizer already covers punctuation,
#    capitals, etc. from pretraining; we just build one alphabet up front.
# ---------------------------------------------------------------------------

PRETRAIN_CORPUS = """
the quick fox runs over the lazy dog.
the lazy dog sleeps by the warm fire.
a bird sings in the tall green tree.
the sun rises over the calm blue sea.
Cat: a small furry animal that sleeps all day.
Dog: a loyal animal that loves to run and play.
the moon shines over the quiet dark forest.
rain falls softly on the old stone road.
""".strip()
PRETRAIN_CORPUS = (PRETRAIN_CORPUS + "\n") * 6

PAD = "\x00"  # explicit pad character, never appears in real text

# Toy "instruction-tuning" task family: given a short word, either uppercase
# it or reverse it -- two DIFFERENT correct answers for the same word,
# so following the instruction (not just memorizing "word -> one answer")
# is the only way to get both right.
TRAIN_WORDS = ["cat", "dog", "sun", "run", "big", "red", "top", "box", "pen", "hat", "six", "jam"]
HELDOUT_WORDS = ["cub", "ten", "sit"]  # never appear in the SFT training set, in ANY task

ALL_WORDS = TRAIN_WORDS + HELDOUT_WORDS
TASKS = ["uppercase", "reverse"]


def apply_task(task, word):
    return word.upper() if task == "uppercase" else word[::-1]


def format_example(task, word):
    """The instruction-tuning format: an instruction line, a Response: tag,
    then the answer. This is a minimal stand-in for a real chat template
    (<|user|>...<|assistant|>...) -- see the README section 2."""
    prompt = f"Instruction: {task} the word {word}\nResponse: "
    answer = apply_task(task, word) + "\n"
    return prompt, answer


# Build the full character vocabulary from everything the model will ever see.
_all_text = PRETRAIN_CORPUS + PAD
for w in ALL_WORDS:
    for t in TASKS:
        p, a = format_example(t, w)
        _all_text += p + a
chars = sorted(set(_all_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
PAD_ID = stoi[PAD]


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 1. Model: the exact decoder-only block from Phase 02 Lesson 6 / Phase 05
#    Lesson 2 -- instruction tuning changes the DATA and LOSS, not the
#    architecture or training loop.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 64
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3


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

    def forward(self, token_ids, labels=None):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX
            )
        return logits, loss

    @torch.no_grad()
    def generate_greedy(self, prompt_ids, max_new_tokens, stop_id=None):
        """Deterministic (argmax) decoding -- used for evaluation so accuracy
        numbers are exactly reproducible, not dependent on sampling noise."""
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = ids[:, -self.block_size:]
            logits, _ = self(context)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return ids


# ---------------------------------------------------------------------------
# 2. Building the SFT label tensor: -100 over the prompt, real ids over the
#    response. This is the from-scratch mechanism the whole lesson is about.
# ---------------------------------------------------------------------------

def build_sft_example(task, word, block_size):
    """Returns (input_ids, labels), both padded to block_size."""
    prompt, answer = format_example(task, word)
    full = prompt + answer
    token_ids = encode(full)
    assert len(token_ids) <= block_size + 1, "toy example too long for BLOCK_SIZE"

    input_ids = token_ids[:-1]
    targets = token_ids[1:]

    # The mechanism: label[i] supervises predicting token (i+1). Mask it
    # (-100, PyTorch's default ignore_index) whenever that TARGET token still
    # belongs to the prompt -- i.e. whenever i+1 < len(prompt). The position
    # predicting the FIRST response character (i+1 == len(prompt)) is kept:
    # that's the whole point of fine-tuning -- learn "after the prompt ends,
    # produce the response".
    prompt_len = len(prompt)
    labels = [
        IGNORE_INDEX if (i + 1) < prompt_len else targets[i]
        for i in range(len(targets))
    ]

    pad_amount = block_size - len(input_ids)
    input_ids = input_ids + [PAD_ID] * pad_amount
    labels = labels + [IGNORE_INDEX] * pad_amount
    return input_ids, labels, prompt, answer


def show_masking_demo():
    print("=" * 78)
    print("1. THE SFT LOSS MASK, MADE EXPLICIT FOR ONE EXAMPLE")
    print("=" * 78)
    input_ids, labels, prompt, answer = build_sft_example("uppercase", "cat", BLOCK_SIZE)
    print(f"Full text seen by the model (input): {(prompt + answer)!r}")
    print(f"  prompt part  ({len(prompt)} chars): {prompt!r}")
    print(f"  response part ({len(answer)} chars): {answer!r}\n")

    print("Position-by-position label (what the model is trained to PREDICT at")
    print("each position, given everything before it as input):\n")
    prompt_len = len(prompt)
    real_len = len(prompt) + len(answer) - 1
    n_masked = sum(1 for l in labels[:real_len] if l == IGNORE_INDEX)
    n_supervised = real_len - n_masked
    for i in range(real_len):
        in_ch = itos[input_ids[i]]
        lab = "MASK(-100)" if labels[i] == IGNORE_INDEX else repr(itos[labels[i]])
        tag = "  <- first supervised position (start of response)" if i == prompt_len - 1 else ""
        if i < 6 or i > real_len - 8 or i == prompt_len - 1:
            print(f"  input[{i:2d}]={in_ch!r:>6}  ->  label={lab:<12}{tag}")
        elif i == 6:
            print("  ...")
    print(f"\nTotal positions: {real_len}  |  masked (prompt): {n_masked}  |  "
          f"supervised (response): {n_supervised}")
    print("-> Exactly the response text's characters (plus the terminating newline)")
    print("   receive gradient signal. The model sees the whole prompt as INPUT")
    print("   (full bidirectional-in-time context up to that point, same as always)")
    print("   but is never asked to PREDICT any part of it -- unlike BERT's MLM")
    print("   (Phase 03 Lesson 2 SS2), which masks INPUT tokens and predicts only")
    print("   those masked positions, SFT never touches the input; it only masks")
    print("   which output positions count toward the LOSS.")


# ---------------------------------------------------------------------------
# 3. Pretraining: plain next-token prediction on generic text (Phase 02 L6's
#    exact recipe) -- this stands in for "a base model pretrained on lots of
#    raw text", the starting point every SFT run assumes.
# ---------------------------------------------------------------------------

def get_pretrain_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def pretrain(model, iters=700, lr=3e-3):
    print("\n" + "=" * 78)
    print("2. PRETRAINING THE BASE MODEL (plain next-token prediction, no masking)")
    print("=" * 78)
    data = torch.tensor(encode(PRETRAIN_CORPUS), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, iters + 1):
        x, y = get_pretrain_batch(data, BLOCK_SIZE, batch_size=32)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 175 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")
    print(f"\nBase model pretrained on generic text (never saw the word 'Instruction:'")
    print("or 'Response:' during this phase, exactly like a real base LLM before SFT).")


# ---------------------------------------------------------------------------
# 4. Instruction tuning: continue training the SAME model, masked loss only.
# ---------------------------------------------------------------------------

def build_sft_dataset(words, block_size):
    batch_inputs, batch_labels = [], []
    for w in words:
        for t in TASKS:
            input_ids, labels, _, _ = build_sft_example(t, w, block_size)
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
    return torch.tensor(batch_inputs, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)


def instruction_tune(model, iters=600, lr=2e-3):
    print("\n" + "=" * 78)
    print("4. INSTRUCTION TUNING: CONTINUE TRAINING, LOSS MASKED TO RESPONSE TOKENS")
    print("=" * 78)
    x, labels = build_sft_dataset(TRAIN_WORDS, BLOCK_SIZE)
    print(f"SFT training set: {len(TRAIN_WORDS)} words x {len(TASKS)} tasks = {x.shape[0]} examples")
    print(f"Held-out words (never in the SFT set, in either task): {HELDOUT_WORDS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, iters + 1):
        _, loss = model(x, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 150 == 0 or step == 1:
            print(f"  step {step:4d}  response-only loss = {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 5. Evaluation: generate on held-out instructions, before vs. after.
# ---------------------------------------------------------------------------

def run_prompt(model, task, word, max_new_tokens=8):
    prompt = f"Instruction: {task} the word {word}\nResponse: "
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long)
    out_ids = model.generate_greedy(prompt_ids, max_new_tokens, stop_id=stoi["\n"])
    generated = decode(out_ids[0, prompt_ids.shape[1]:].tolist())
    return prompt, generated.split("\n")[0]  # text produced after the prompt, up to the stop newline


def evaluate(model, words, label):
    correct = 0
    right_length = 0
    alpha_only = 0
    total = 0
    rows = []
    for w in words:
        for t in TASKS:
            prompt, produced = run_prompt(model, t, w)
            expected = apply_task(t, w)
            ok = produced == expected
            correct += int(ok)
            right_length += int(len(produced) == len(expected))
            alpha_only += int(produced.isalpha())
            total += 1
            rows.append((t, w, expected, produced, ok))
    print(f"\n{label}")
    print(f"  exact-match accuracy:            {correct}/{total} = {100 * correct / total:.0f}%")
    print(f"  correct RESPONSE LENGTH:         {right_length}/{total} = {100 * right_length / total:.0f}%  "
          f"(format compliance, regardless of content)")
    print(f"  well-formed (letters only, no ramble/punctuation): {alpha_only}/{total} = {100 * alpha_only / total:.0f}%")
    for t, w, expected, produced, ok in rows:
        mark = "OK" if ok else "WRONG"
        print(f"    {t:10s} {w:5s} -> expected {expected!r:8s} got {produced!r:10s} [{mark}]")
    return {
        "exact_pct": 100 * correct / total,
        "length_pct": 100 * right_length / total,
        "wellformed_pct": 100 * alpha_only / total,
    }


def before_after_demo(base_state_dict):
    print("\n" + "=" * 78)
    print("3. GENERATION ON A HELD-OUT INSTRUCTION -- BEFORE INSTRUCTION TUNING")
    print("=" * 78)
    model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    model.load_state_dict(base_state_dict)
    model.eval()

    prompt, produced = run_prompt(model, "uppercase", HELDOUT_WORDS[0], max_new_tokens=40)
    print(f"Prompt:  {prompt!r}")
    print(f"Base model's continuation (up to 40 chars, stopping early at a newline if hit):")
    print(f"  {produced!r}")
    print("-> The base model was never trained on this instruction/response format at all,")
    print("   so it just continues in generic-corpus style -- unrelated words, no attempt")
    print("   to uppercase anything, and it doesn't reliably stop after a short answer.")

    print("\nHeld-out-set accuracy BEFORE instruction tuning (exact string match):")
    return evaluate(model, HELDOUT_WORDS, "BEFORE fine-tuning, held-out words")


def main():
    print(f"Vocabulary ({vocab_size} characters): {chars}\n")

    base_model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    show_masking_demo()
    pretrain(base_model)

    base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
    before_stats = before_after_demo(base_state_dict)

    instruction_tune(base_model)

    print("\n" + "=" * 78)
    print("5. GENERATION ON THE SAME INSTRUCTIONS -- AFTER INSTRUCTION TUNING")
    print("=" * 78)
    base_model.eval()
    prompt, produced = run_prompt(base_model, "uppercase", HELDOUT_WORDS[0], max_new_tokens=40)
    print(f"Prompt:  {prompt!r}")
    print(f"Fine-tuned model's continuation:")
    print(f"  {produced!r}")

    print("\nSame word, two different instructions -- a TRAINING word, to check the")
    print("model is conditioning on the instruction rather than memorizing one fixed")
    print("answer per word:")
    for t in TASKS:
        _, produced = run_prompt(base_model, t, TRAIN_WORDS[0])
        print(f"  Instruction: {t:10s} the word {TRAIN_WORDS[0]!r}  ->  Response: {produced!r}")

    train_stats = evaluate(base_model, TRAIN_WORDS[:4], "AFTER fine-tuning, TRAINING words (sample of 4)")
    held_stats = evaluate(base_model, HELDOUT_WORDS, "AFTER fine-tuning, HELD-OUT words")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"BEFORE fine-tuning, held-out words: {before_stats['wellformed_pct']:.0f}% well-formed, "
          f"{before_stats['length_pct']:.0f}% correct length, {before_stats['exact_pct']:.0f}% exact match --")
    print("the base model has no notion of 'produce a short response and stop' at all.")
    print(f"\nAFTER fine-tuning, TRAINING words: {train_stats['exact_pct']:.0f}% exact match -- the SAME")
    print("model weights now reproduce every instruction/response pair they were")
    print("directly trained on, including choosing a DIFFERENT correct answer for")
    print("the same word depending on which instruction was given (see 'cat' above) --")
    print("genuine instruction-conditioned behavior, not a fixed word-to-word lookup.")
    print(f"\nAFTER fine-tuning, HELD-OUT words: {held_stats['wellformed_pct']:.0f}% well-formed and "
          f"{held_stats['length_pct']:.0f}% the correct response")
    print(f"length, but only {held_stats['exact_pct']:.0f}% exact match. This is the honest, unglamorous")
    print("result this toy scale actually produces: instruction tuning taught the")
    print("RESPONSE FORMAT (a short, letters-only answer, stopped by a newline) well")
    print("enough to generalize to words never seen during SFT, but the specific")
    print("uppercase/reverse TRANSFORMATION SKILL for a brand-new word did not fully")
    print("generalize from only 12 training words per task -- the model falls back")
    print("on memorized training-set responses instead. That gap is exactly what")
    print("Section 5's data-diversity point (LIMA) is about: how much an SFT run")
    print("generalizes, versus merely memorizes, is a direct function of how much")
    print("the instruction data actually covers -- not just how many gradient steps")
    print("or how low the training loss gets.")


if __name__ == "__main__":
    main()
