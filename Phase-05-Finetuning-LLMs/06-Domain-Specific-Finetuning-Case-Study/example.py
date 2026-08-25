"""
Domain-Specific Fine-tuning Case Study

An end-to-end worked case study: adapt a pretrained mini-GPT to a narrow
domain (a distinctive "pirate speak" writing style) and quantify the
classic specialization-vs-forgetting trade-off from two angles:

  1. Parameter cost: LoRA (Lesson 2's LoRALinear, reused and applied to a
     REAL pretrained model's attention projections) vs. full fine-tuning --
     Lesson 1's memory-math argument, made concrete for this model size.
  2. Behavior: after fine-tuning on ONLY the narrow domain, measure BOTH
     (a) held-out loss on the narrow domain (did it specialize?) and
     (b) held-out loss on general, non-domain text (did it forget?) --
     for LoRA and for full fine-tuning, side by side, with real numbers.

Runtime: ~1-2 minutes on a CPU (one pretraining run + two short fine-tuning
runs on a tiny model).

Run:
    python example.py
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2024)

# ---------------------------------------------------------------------------
# 0. Two corpora: a general-English one (what the base model is pretrained
#    on) and a narrow "pirate speak" domain corpus (what we'll fine-tune
#    on). A few sentences from EACH are held out entirely for evaluation.
# ---------------------------------------------------------------------------

GENERAL_TRAIN = """
the sun rises over the quiet green hills.
a small bird sings in the old oak tree.
the children play in the park after school.
rain falls softly on the city streets tonight.
she reads a book by the warm window light.
the train arrives at the station every morning.
he cooks a simple meal in the small kitchen.
the river flows gently past the old stone bridge.
the teacher writes a lesson on the board.
a gentle wind blows across the open field.
the baker sells fresh bread every single day.
they walk the dog around the quiet neighborhood.
""".strip()

GENERAL_HELDOUT = [
    "the dog sleeps quietly by the fire.",
    "a cool breeze moves through the tall grass.",
    "the farmer waters the plants every evening.",
    "a young student studies at the small desk.",
]

DOMAIN_TRAIN = """
arr the ship sails over the rough dark sea.
the cap'n found a chest of golden treasure.
ye best be careful of the storm ahead matey.
the crew drank rum and sang a pirate song.
a parrot squawked upon the tall wooden mast.
we buried the treasure upon a lonely island.
the cap'n steers the ship past the rocky shore.
shiver me timbers there be a storm ahead.
""".strip()

DOMAIN_HELDOUT = [
    "the pirate crew searched the sea for treasure.",
    "arr matey the old ship sails into the storm.",
]

# Repeat the training text so there's enough signal to fit in a small
# number of steps -- exactly Lesson 4/Phase 02's approach for toy corpora.
GENERAL_TRAIN_TEXT = (GENERAL_TRAIN + "\n") * 10
DOMAIN_TRAIN_TEXT = (DOMAIN_TRAIN + "\n") * 10

_all_text = GENERAL_TRAIN_TEXT + DOMAIN_TRAIN_TEXT + "\n".join(GENERAL_HELDOUT + DOMAIN_HELDOUT)
chars = sorted(set(_all_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 1. Model: the same decoder-only block from Phase 02 / Lesson 2, plus
#    Lesson 2's LoRALinear reused verbatim (with one addition: wrapping an
#    ALREADY-PRETRAINED nn.Linear, which is what real LoRA fine-tuning does
#    -- Lesson 2's own demo applied it to a freshly-initialized layer).
# ---------------------------------------------------------------------------

BLOCK_SIZE = 64
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 3
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGETS = ("W_q", "W_v")  # the common real-world choice -- Lesson 5 section 2


class LoRALinear(nn.Module):
    """Identical mechanism to Lesson 2's LoRALinear: frozen base + trainable
    low-rank update W' = W + (alpha/r) * B @ A. Adds one classmethod so it
    can wrap an EXISTING pretrained nn.Linear instead of only a freshly
    initialized one -- this is what applying LoRA to a real model (Lesson 5's
    get_peft_model) actually does."""

    def __init__(self, in_features, out_features, r, alpha=None, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    @classmethod
    def from_pretrained_linear(cls, linear, r, alpha=None):
        """Freeze a COPY of an already-trained layer's weights as the base,
        instead of a random initialization -- this is the realistic case:
        the base model is pretrained first, LoRA is added afterward."""
        obj = cls(linear.in_features, linear.out_features, r=r, alpha=alpha,
                   bias=linear.bias is not None)
        with torch.no_grad():
            obj.base.weight.copy_(linear.weight)
            if linear.bias is not None:
                obj.base.bias.copy_(linear.bias)
        return obj

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A.T) @ self.B.T)


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
    def generate_greedy(self, prompt_ids, max_new_tokens):
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = ids[:, -self.block_size:]
            logits, _ = self(context)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def apply_lora_to_attention(model, r, alpha, targets):
    """The from-scratch equivalent of Lesson 5's peft.get_peft_model: freeze
    EVERY existing parameter, then replace the named attention projections
    with LoRA-wrapped versions (trainable A/B, frozen pretrained base)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for name in targets:
            linear = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear.from_pretrained_linear(linear, r=r, alpha=alpha))
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# 2. Training / evaluation utilities
# ---------------------------------------------------------------------------

def get_batch(data, block_size, batch_size):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y


def train_lm(model, text, iters, lr, params=None, log_every=None):
    data = torch.tensor(encode(text), dtype=torch.long)
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    log_every = log_every or max(iters // 4, 1)
    for step in range(1, iters + 1):
        x, y = get_batch(data, BLOCK_SIZE, batch_size=24)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            print(f"    step {step:4d}  loss = {loss.item():.4f}")
    return loss.item()


@torch.no_grad()
def held_out_loss(model, sentences):
    """Average next-token-prediction cross-entropy over held-out sentences
    NEVER seen in any training text -- the standard way to measure how well
    a model's learned distribution fits a piece of text it wasn't trained on."""
    model.eval()
    losses = []
    for s in sentences:
        ids = torch.tensor([encode(s)], dtype=torch.long)
        if ids.shape[1] < 2:
            continue
        x, y = ids[:, :-1], ids[:, 1:]
        x, y = x[:, -BLOCK_SIZE:], y[:, -BLOCK_SIZE:]
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def sample_continuation(model, prompt, max_new_tokens=40):
    model.eval()
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    out = model.generate_greedy(ids, max_new_tokens)
    model.train()
    return decode(out[0, ids.shape[1]:].tolist())


# ---------------------------------------------------------------------------
# 3. The case study
# ---------------------------------------------------------------------------

def main():
    print(f"Vocabulary ({vocab_size} characters)\n")

    # --- Step 1: pretrain a base model on GENERAL text only -----------------
    print("=" * 78)
    print("1. PRETRAINING THE BASE MODEL ON GENERAL TEXT")
    print("=" * 78)
    base_model = MiniGPT(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, BLOCK_SIZE)
    train_lm(base_model, GENERAL_TRAIN_TEXT, iters=900, lr=3e-3, log_every=225)
    total_params, _ = count_params(base_model)
    print(f"\nBase model total parameters: {total_params:,}")

    baseline_general = held_out_loss(base_model, GENERAL_HELDOUT)
    baseline_domain = held_out_loss(base_model, DOMAIN_HELDOUT)
    print(f"\nBEFORE any domain exposure, held-out loss (lower = fits better):")
    print(f"  general held-out sentences: {baseline_general:.3f}")
    print(f"  domain (pirate) held-out sentences: {baseline_domain:.3f}")
    print("-> The domain loss is higher: pirate-speak's vocabulary and phrasing")
    print("   ('arr', \"cap'n\", 'matey', 'ye') is out of distribution for a model")
    print("   that has only ever seen everyday general sentences.")

    # --- Step 2: parameter-cost comparison, Lesson 1/2 style -----------------
    print("\n" + "=" * 78)
    print("2. PARAMETER COST: FULL FINE-TUNING vs. LoRA, FOR THIS MODEL")
    print("=" * 78)
    lora_probe = copy.deepcopy(base_model)
    apply_lora_to_attention(lora_probe, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)
    _, lora_trainable = count_params(lora_probe)
    full_trainable = total_params
    print(f"Full fine-tuning trainable parameters: {full_trainable:,} (every parameter)")
    print(f"LoRA (r={LORA_R}, targeting {LORA_TARGETS} in all {NUM_LAYERS} layers) "
          f"trainable parameters: {lora_trainable:,}")
    print(f"LoRA trains {100 * lora_trainable / full_trainable:.1f}% as many parameters as full "
          f"fine-tuning -- by Lesson 1's 16-bytes-per-trainable-parameter\naccounting, that is "
          f"roughly the same ratio by which LoRA shrinks the optimizer-state memory needed to "
          f"fine-tune this model.")
    del lora_probe

    # --- Step 3: fine-tune two independent copies on the SAME domain data ---
    print("\n" + "=" * 78)
    print("3. FINE-TUNING ON THE NARROW DOMAIN: LoRA vs. FULL FINE-TUNING")
    print("=" * 78)

    print("\n[LoRA fine-tuning -- only A/B in W_q, W_v are trainable]")
    lora_model = copy.deepcopy(base_model)
    apply_lora_to_attention(lora_model, r=LORA_R, alpha=LORA_ALPHA, targets=LORA_TARGETS)
    lora_params = [p for p in lora_model.parameters() if p.requires_grad]
    train_lm(lora_model, DOMAIN_TRAIN_TEXT, iters=400, lr=5e-3, params=lora_params, log_every=100)

    print("\n[Full fine-tuning -- every parameter trainable]")
    full_model = copy.deepcopy(base_model)
    for p in full_model.parameters():
        p.requires_grad_(True)
    train_lm(full_model, DOMAIN_TRAIN_TEXT, iters=400, lr=3e-3, log_every=100)

    # --- Step 4: measure specialization AND forgetting, for both -----------
    print("\n" + "=" * 78)
    print("4. RESULT: DOMAIN IMPROVEMENT vs. GENERAL-CAPABILITY REGRESSION")
    print("=" * 78)

    lora_general = held_out_loss(lora_model, GENERAL_HELDOUT)
    lora_domain = held_out_loss(lora_model, DOMAIN_HELDOUT)
    full_general = held_out_loss(full_model, GENERAL_HELDOUT)
    full_domain = held_out_loss(full_model, DOMAIN_HELDOUT)

    def fmt_delta(after, before):
        sign = "+" if after >= before else ""
        return f"{after:.3f}  ({sign}{after - before:.3f} vs. baseline)"

    print(f"\n{'':22s}{'general held-out loss':>26s}{'domain held-out loss':>26s}")
    print(f"{'baseline (no domain FT)':22s}{baseline_general:>26.3f}{baseline_domain:>26.3f}")
    print(f"{'LoRA fine-tuned':22s}{fmt_delta(lora_general, baseline_general):>26s}"
          f"{fmt_delta(lora_domain, baseline_domain):>26s}")
    print(f"{'fully fine-tuned':22s}{fmt_delta(full_general, baseline_general):>26s}"
          f"{fmt_delta(full_domain, baseline_domain):>26s}")

    lora_general_drift = lora_general - baseline_general
    full_general_drift = full_general - baseline_general
    lora_domain_gain = baseline_domain - lora_domain
    full_domain_gain = baseline_domain - full_domain

    print(f"\nDomain improvement (loss reduction on pirate held-out text):")
    print(f"  LoRA:  {lora_domain_gain:.3f}   |   Full FT: {full_domain_gain:.3f}")
    print(f"General-capability drift (loss INCREASE on general held-out text, "
          f"higher = more forgetting):")
    print(f"  LoRA:  {lora_general_drift:+.3f}   |   Full FT: {full_general_drift:+.3f}")

    if full_general_drift > lora_general_drift:
        print("\n-> Full fine-tuning drifted further from the base model's general-text fit than")
        print("   LoRA did, while updating "
              f"{100 * full_trainable / lora_trainable:.0f}x more parameters to reach its domain")
        print("   improvement -- a direct, measured illustration of catastrophic forgetting: letting")
        print("   every weight move to fit a narrow domain risks quietly degrading everything else")
        print("   the model could previously do, exactly Lesson 1 section 3's concern. LoRA's frozen")
        print("   base constrains how much general behavior CAN change, which is showing up here as")
        print("   less general-loss drift for a comparable amount of domain specialization.")
    else:
        print("\n-> At this toy scale, LoRA's general-loss drift was not smaller than full")
        print("   fine-tuning's -- with a model and dataset this tiny, both quickly overfit the small")
        print("   repeated domain corpus. The qualitative generations below are the more reliable")
        print("   signal at this scale; a real-scale comparison (larger base model, more diverse")
        print("   domain data) is where this gap typically widens clearly in LoRA's favor.")

    # --- Step 5: qualitative check -- does a GENERIC prompt still generate
    #     sensibly, or has it been dragged into pirate-speak regardless of prompt?
    print("\n" + "=" * 78)
    print("5. QUALITATIVE CHECK: GENERIC PROMPT, BOTH FINE-TUNED MODELS")
    print("=" * 78)
    generic_prompt = "the "
    print(f"Prompt: {generic_prompt!r}\n")
    print(f"  base model (pre-domain-FT):  {sample_continuation(base_model, generic_prompt)!r}")
    print(f"  LoRA fine-tuned:             {sample_continuation(lora_model, generic_prompt)!r}")
    print(f"  fully fine-tuned:            {sample_continuation(full_model, generic_prompt)!r}")
    print("\n-> Compare how much each fine-tuned model's generic-prompt continuation still")
    print("   resembles ordinary general text vs. how much pirate vocabulary leaks in even")
    print("   though the prompt gave no domain cue -- a qualitative view of the same")
    print("   specialization-vs-forgetting trade-off measured quantitatively above.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("Both fine-tuning approaches specialize on the narrow pirate-speak domain")
    print("(domain held-out loss drops from the baseline in both cases). The question this")
    print("case study answers quantitatively is what that specialization COSTS on general,")
    print("non-domain capability -- and, per Lesson 1's memory-math argument, at what")
    print(f"trainable-parameter budget ({lora_trainable:,} for LoRA vs. {full_trainable:,} for full")
    print("fine-tuning) that specialization was bought. Whichever direction the numbers above")
    print("came out, they are this run's REAL measured numbers, not assumed ones -- that is the point")
    print("of evaluating on both the narrow domain AND general text, every time you fine-tune.")


if __name__ == "__main__":
    main()
