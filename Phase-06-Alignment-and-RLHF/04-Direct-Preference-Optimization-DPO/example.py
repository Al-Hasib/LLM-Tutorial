"""
Direct Preference Optimization (DPO)

Implements the DPO loss (Rafailov et al., 2023) from scratch in PyTorch and
uses it to fine-tune a tiny decoder-only Transformer (the same mini-GPT
architecture from Phase 02 Lesson 6, reused here as both the policy and the
frozen reference model) directly on synthetic (prompt, chosen, rejected)
preference triples -- WITHOUT ever training a separate reward model
(Lesson 2) and WITHOUT any RL rollout / PPO update (Lesson 3). This is
exactly the simplification DPO is famous for: the same Bradley-Terry
preference objective, optimized with one supervised-looking loss and a
single backward pass.

Pipeline mirrored here, at toy scale:
  1. "Pretrain" a tiny LM on generic text that includes BOTH positive-
     sentiment and negative-sentiment continuations of some prompts (a
     stand-in for an SFT model that has *seen* both kinds of continuation
     but has no preference between them yet).
  2. Freeze a copy of it as pi_ref.
  3. Clone it again as pi_theta (the policy DPO will actually update).
  4. Fine-tune ONLY pi_theta with the DPO loss on preference triples
     (README section 2) -- no reward model, no sampling/rollouts, no PPO.
  5. Track the log-probability margin log pi(chosen) - log pi(rejected)
     growing over training, on both the training pairs and a small held-out
     set of prompts/completions never given a preference label during DPO
     fine-tuning (to honestly check whether the learned preference is a
     generalizable "prefer positive sentiment" rule or pure memorization).

Runtime: well under a minute on a CPU.

Run:
    python example.py
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 0. Toy preference data: (prompt, chosen, rejected) triples.
#
# The "preference" here is a simple, known ground-truth rule -- chosen
# completions are positive-sentiment, rejected completions are negative-
# sentiment continuations of the same prompt -- so we can later check
# whether DPO training actually learned that rule, not just memorized
# six sequences.
# ---------------------------------------------------------------------------

TRAIN_PAIRS = [
    ("the movie was", "absolutely wonderful and touching.", "absolutely dreadful and boring."),
    ("the food tasted", "incredibly delicious and fresh.", "incredibly bland and stale."),
    ("the service was", "truly excellent and attentive.", "truly rude and careless."),
    ("the weather today is", "bright sunny and pleasant.", "cold gloomy and miserable."),
    ("my whole day was", "wonderful and full of joy.", "terrible and full of stress."),
    ("the concert last night was", "amazing and unforgettable.", "awful and forgettable."),
]

# Held out of DPO fine-tuning entirely -- used only to check generalization.
# Their WORDS appear in the pretraining corpus below (so the base model has
# seen them), but DPO training never sees a preference label over them.
HELD_OUT_PAIRS = [
    ("the hotel room was", "clean bright and comfortable.", "dirty dark and uncomfortable."),
    ("the new phone is", "fast reliable and impressive.", "slow buggy and disappointing."),
]

ALL_PAIRS = TRAIN_PAIRS + HELD_OUT_PAIRS

# ---------------------------------------------------------------------------
# 1. A tiny word-level tokenizer (simpler to reason about token-by-token
# log-probabilities than characters, for a preference signal that lives at
# the word level -- "wonderful" vs "dreadful").
# ---------------------------------------------------------------------------


def tokenize(text):
    return re.findall(r"[a-z]+|[.]", text.lower())


vocab = sorted({tok for prompt, chosen, rejected in ALL_PAIRS
                for tok in tokenize(prompt) + tokenize(chosen) + tokenize(rejected)})
PAD_ID = 0
stoi = {tok: i + 1 for i, tok in enumerate(vocab)}   # ids 1..V, 0 reserved for PAD
itos = {i: tok for tok, i in stoi.items()}
VOCAB_SIZE = len(stoi) + 1


def encode(text):
    return [stoi[tok] for tok in tokenize(text)]


# ---------------------------------------------------------------------------
# 2. The model: identical block design to Phase 02 Lesson 6's MiniGPT
# (causal self-attention + feed-forward, Pre-LN residual blocks), just
# small enough to train in seconds on this toy vocabulary.
# ---------------------------------------------------------------------------

D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
BLOCK_SIZE = 20   # comfortably longer than any prompt+completion in this dataset


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
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, block_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)   # (batch, T, vocab_size)


# ---------------------------------------------------------------------------
# 3. Building padded (prompt, completion) tensors and a completion-only
# log-probability function -- the piece of machinery both pretraining and
# DPO need: "what log-probability does this model assign to exactly these
# completion tokens, given exactly this prompt?"
# ---------------------------------------------------------------------------


def build_example(prompt, completion):
    """Returns (full_token_ids, num_prompt_tokens)."""
    prompt_ids = encode(prompt)
    completion_ids = encode(completion)
    return prompt_ids + completion_ids, len(prompt_ids)


def make_batch(pairs, which):
    """which in {'chosen', 'rejected'}. Returns padded (input, target, mask)
    tensors of shape (batch, BLOCK_SIZE - 1), where mask[i, t] = 1 exactly at
    the target positions that correspond to a COMPLETION token (never a
    prompt token, never padding)."""
    inputs, targets, masks = [], [], []
    for prompt, chosen, rejected in pairs:
        completion = chosen if which == "chosen" else rejected
        full_ids, num_prompt = build_example(prompt, completion)
        n = len(full_ids)
        inp = full_ids[:-1] + [PAD_ID] * (BLOCK_SIZE - 1 - (n - 1))
        tgt = full_ids[1:] + [PAD_ID] * (BLOCK_SIZE - 1 - (n - 1))
        mask = [0] * (BLOCK_SIZE - 1)
        # target index i holds full_ids[i+1]; it's a completion token when
        # i+1 >= num_prompt, i.e. i >= num_prompt - 1.
        for i in range(num_prompt - 1, n - 1):
            mask[i] = 1
        inputs.append(inp)
        targets.append(tgt)
        masks.append(mask)
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(masks, dtype=torch.float))


def sequence_logprobs(model, input_ids, target_ids, mask):
    """Sum of log p(target_t | input_<=t) over exactly the masked (completion)
    positions, for each sequence in the batch -- this is log pi(completion | prompt)
    under the model's own causal factorization. Returns shape (batch,)."""
    logits = model(input_ids)                                  # (batch, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (batch, T)
    return (token_logprobs * mask).sum(dim=1)                  # (batch,)


# ---------------------------------------------------------------------------
# 4. The DPO loss itself (README section 2):
#
#   loss = -log( sigmoid( beta * [ (log pi(chosen|x) - log pi_ref(chosen|x))
#                                 - (log pi(rejected|x) - log pi_ref(rejected|x)) ] ) )
#
# No reward model anywhere in sight -- pi_ref stands in for both "the
# reward baseline" AND the KL anchor from Lesson 3, simultaneously.
# ---------------------------------------------------------------------------


def dpo_loss(policy, ref, chosen_batch, rejected_batch, beta):
    c_in, c_tgt, c_mask = chosen_batch
    r_in, r_tgt, r_mask = rejected_batch

    policy_chosen_logp = sequence_logprobs(policy, c_in, c_tgt, c_mask)
    policy_rejected_logp = sequence_logprobs(policy, r_in, r_tgt, r_mask)
    with torch.no_grad():
        ref_chosen_logp = sequence_logprobs(ref, c_in, c_tgt, c_mask)
        ref_rejected_logp = sequence_logprobs(ref, r_in, r_tgt, r_mask)

    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    implicit_reward_margin = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * implicit_reward_margin).mean()
    return loss, policy_chosen_logp.detach(), policy_rejected_logp.detach()


# ---------------------------------------------------------------------------
# 5. Pretraining: an ordinary next-token cross-entropy LM, trained on BOTH
# the chosen and the rejected continuation of every prompt (train + held-out
# alike), with no notion of preference at all -- exactly what an SFT model
# is: fluent, and capable of producing either continuation, before any
# preference-based stage touches it.
# ---------------------------------------------------------------------------


def pretrain(model, pairs, steps=300, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    chosen_batch = make_batch(pairs, "chosen")
    rejected_batch = make_batch(pairs, "rejected")
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        total_loss = 0.0
        for (inp, tgt, mask) in (chosen_batch, rejected_batch):
            logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=PAD_ID
            )
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  pretrain step {step:4d}   LM cross-entropy loss = {total_loss / 2:.4f}")


def average_margin(model, pairs):
    chosen_batch = make_batch(pairs, "chosen")
    rejected_batch = make_batch(pairs, "rejected")
    with torch.no_grad():
        c_logp = sequence_logprobs(model, *chosen_batch)
        r_logp = sequence_logprobs(model, *rejected_batch)
    margin = (c_logp - r_logp)
    accuracy = (margin > 0).float().mean().item()
    return margin.mean().item(), accuracy


def main():
    print("=" * 70)
    print("0. SETUP")
    print("=" * 70)
    print(f"Vocabulary size (word-level, toy corpus): {VOCAB_SIZE}")
    print(f"{len(TRAIN_PAIRS)} training preference triples, "
          f"{len(HELD_OUT_PAIRS)} held-out triples (no preference label used in DPO training)")
    print("Ground-truth rule the preference data encodes: chosen = positive-sentiment")
    print("continuation, rejected = negative-sentiment continuation of the same prompt.")

    print("\n" + "=" * 70)
    print("1. PRETRAINING A BASE (SFT-LIKE) MODEL ON BOTH CHOSEN AND REJECTED TEXT")
    print("=" * 70)
    print("This model sees BOTH sentiments as plain, equally-plausible text -- it has")
    print("no preference yet, exactly like an SFT model before any alignment stage.")
    base_model = MiniGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    pretrain(base_model, ALL_PAIRS, steps=300)

    ref_model = MiniGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    ref_model.load_state_dict(base_model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()   # pi_ref: frozen from here on, exactly as in Lesson 3's KL anchor

    policy_model = MiniGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    policy_model.load_state_dict(base_model.state_dict())   # pi_theta starts identical to pi_ref

    train_margin_before, train_acc_before = average_margin(policy_model, TRAIN_PAIRS)
    heldout_margin_before, heldout_acc_before = average_margin(policy_model, HELD_OUT_PAIRS)
    print(f"\nBEFORE any DPO training (policy == reference model):")
    print(f"  avg [log pi(chosen) - log pi(rejected)] on TRAIN pairs   = {train_margin_before:+.3f}"
          f"   (chosen preferred in {train_acc_before * 100:.0f}% of pairs)")
    print(f"  avg [log pi(chosen) - log pi(rejected)] on HELD-OUT pairs = {heldout_margin_before:+.3f}"
          f"   (chosen preferred in {heldout_acc_before * 100:.0f}% of pairs)")
    print("-> Close to a coin flip on both -- the pretrained model has no systematic")
    print("   preference for positive over negative sentiment; it just learned both are fluent.")

    print("\n" + "=" * 70)
    print("2. DPO FINE-TUNING -- NO REWARD MODEL, NO RL, JUST THIS LOSS:")
    print("=" * 70)
    print("loss = -log( sigmoid( beta * [ (log pi(c) - log pi_ref(c)) - (log pi(r) - log pi_ref(r)) ] ) )")
    print("Optimized directly on the 6 TRAINING preference pairs. pi_ref is frozen throughout.\n")

    BETA = 0.5
    LR = 5e-4
    NUM_STEPS = 400
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)

    chosen_batch = make_batch(TRAIN_PAIRS, "chosen")
    rejected_batch = make_batch(TRAIN_PAIRS, "rejected")

    history = []
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        loss, c_logp, r_logp = dpo_loss(policy_model, ref_model, chosen_batch, rejected_batch, BETA)
        loss.backward()
        optimizer.step()
        margin = (c_logp - r_logp).mean().item()
        history.append((step, loss.item(), margin))
        if step % 50 == 0 or step == 1:
            print(f"  DPO step {step:4d}   loss = {loss.item():.4f}   "
                  f"avg train margin log pi(chosen)-log pi(rejected) = {margin:+.3f}")

    print("\n" + "=" * 70)
    print("3. RESULTS: DID THE LOG-PROBABILITY MARGIN ACTUALLY GROW?")
    print("=" * 70)
    train_margin_after, train_acc_after = average_margin(policy_model, TRAIN_PAIRS)
    heldout_margin_after, heldout_acc_after = average_margin(policy_model, HELD_OUT_PAIRS)

    print(f"{'':>28}{'before DPO':>14}{'after DPO':>14}")
    print(f"{'TRAIN margin':>28}{train_margin_before:>14.3f}{train_margin_after:>14.3f}")
    print(f"{'TRAIN pairwise accuracy':>28}{train_acc_before:>14.2f}{train_acc_after:>14.2f}")
    print(f"{'HELD-OUT margin':>28}{heldout_margin_before:>14.3f}{heldout_margin_after:>14.3f}")
    print(f"{'HELD-OUT pairwise accuracy':>28}{heldout_acc_before:>14.2f}{heldout_acc_after:>14.2f}")

    print(f"\n-> On the {len(TRAIN_PAIRS)} TRAINING pairs, the margin log pi(chosen) - log pi(rejected)")
    print(f"   moved from {train_margin_before:+.3f} to {train_margin_after:+.3f} purely by minimizing the DPO loss --")
    print(f"   no reward model was ever instantiated, and no text was ever sampled/rolled")
    print(f"   out from the policy during this entire training loop.")

    if heldout_acc_after > heldout_acc_before:
        print(f"\n-> On the {len(HELD_OUT_PAIRS)} HELD-OUT pairs (no preference label used during DPO training,")
        print(f"   only seen as plain text during pretraining), pairwise accuracy also rose, from")
        print(f"   {heldout_acc_before:.2f} to {heldout_acc_after:.2f} -- evidence the model shifted probability mass")
        print(f"   toward positive-sentiment completions in general, not merely toward the six")
        print(f"   exact training sequences it was fine-tuned on.")
    else:
        print(f"\n-> On the {len(HELD_OUT_PAIRS)} HELD-OUT pairs, pairwise accuracy went from {heldout_acc_before:.2f} to")
        print(f"   {heldout_acc_after:.2f}. With only {len(TRAIN_PAIRS)} training pairs and a model this tiny, DPO here")
        print(f"   mainly memorizes the specific training sequences rather than learning a fully")
        print(f"   general 'prefer positive sentiment' direction -- a real system needs far more")
        print(f"   preference data and a far larger backbone for the implicit reward to generalize.")

    print(f"\nSample: implicit rewards beta*(log pi/pi_ref) for one TRAIN pair before vs after DPO")
    print(f"(this log-ratio is exactly the reward DPO optimizes WITHOUT ever training an explicit")
    print(f"reward model -- README section 3):")
    prompt, chosen, rejected = TRAIN_PAIRS[0]
    one_chosen = make_batch([TRAIN_PAIRS[0]], "chosen")
    one_rejected = make_batch([TRAIN_PAIRS[0]], "rejected")
    with torch.no_grad():
        ref_c = sequence_logprobs(ref_model, *one_chosen).item()
        ref_r = sequence_logprobs(ref_model, *one_rejected).item()
        pol_c = sequence_logprobs(policy_model, *one_chosen).item()
        pol_r = sequence_logprobs(policy_model, *one_rejected).item()
    print(f"  prompt = {prompt!r}")
    print(f"  chosen = {chosen!r}  |  rejected = {rejected!r}")
    print(f"  implicit reward of chosen   = beta*(log pi - log pi_ref) = "
          f"{BETA * (pol_c - ref_c):+.3f}")
    print(f"  implicit reward of rejected = beta*(log pi - log pi_ref) = "
          f"{BETA * (pol_r - ref_r):+.3f}")
    print("  -> DPO raised the implicit reward of the chosen response relative to the")
    print("     reference and lowered it for the rejected response, using only supervised-style")
    print("     gradient descent on labeled pairs -- exactly the RLHF objective from Lesson 2/3,")
    print("     reached without a reward model or an RL rollout loop.")


if __name__ == "__main__":
    main()
