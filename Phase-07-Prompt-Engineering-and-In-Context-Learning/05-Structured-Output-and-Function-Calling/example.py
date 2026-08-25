"""
Structured Output and Function Calling

PART A -- CONSTRAINED DECODING, implemented for real, from scratch. We
define a small grammar for a restricted JSON object:

    {"action": "add" | "sub" | "mul", "value": <1-to-3-digit integer>}

and a character-level state machine, `next_valid_chars(prefix)`, that
inspects the string generated SO FAR and returns exactly the set of
characters that are legal to emit next (an empty set means the structure is
already complete and generation must stop). At every generation step of a
genuine (but deliberately UNTRAINED, randomly initialized) tiny decoder-only
Transformer, we take the model's raw logits over the character vocabulary
and set every logit for a character NOT in that valid set to -infinity
before sampling -- this is constrained decoding. We then show, over many
independent generations, that the output is 100% syntactically valid EVERY
SINGLE TIME despite the model's weights being pure random noise, and
contrast this against the same random model sampling with NO mask at all,
which almost never produces valid structured output. Validity here comes
entirely from the mask, not from the model having learned anything.

PART B -- FUNCTION CALLING, scripted end to end. A "model output" (a fixed
string, standing in for what a real LLM would emit) contains a structured
function-call request in the now-standard `{"name": ..., "arguments": {...}}`
format. External code parses that JSON, dispatches to a REAL Python function
(a calculator, and a small unit-conversion lookup) with the parsed
arguments, executes it, and splices the REAL return value back into a
final response -- demonstrating the full request -> parse -> execute ->
respond loop that underlies every production tool-calling LLM API.

Runtime: a few seconds on CPU (a tiny untrained model doing character-level
forward passes, no training loop at all).

Run:
    python example.py
"""

import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ===========================================================================
# PART A: CONSTRAINED DECODING FOR GUARANTEED-VALID STRUCTURED OUTPUT
# ===========================================================================

TEMPLATE_HEAD = '{"action": "'
TEMPLATE_MID = '", "value": '
ACTIONS = ["add", "sub", "mul"]

VALID_JSON_RE = None  # (kept as a plain check function below instead of regex, for clarity)


def is_fully_valid(s):
    """Independent ground-truth validity check, written completely
    separately from the grammar state machine below, so it can genuinely
    catch a bug in `next_valid_chars` rather than just agreeing with it by
    construction."""
    if not s.startswith(TEMPLATE_HEAD):
        return False
    rest = s[len(TEMPLATE_HEAD):]
    for action in ACTIONS:
        prefix = action + TEMPLATE_MID
        if rest.startswith(prefix):
            tail = rest[len(prefix):]
            if tail.endswith("}"):
                digits = tail[:-1]
                if digits.isdigit() and (digits == "0" or not digits.startswith("0")) and 1 <= len(digits) <= 3:
                    return True
    return False


def next_valid_chars(prefix):
    """The grammar, implemented as a character-level state machine driven
    purely by inspecting `prefix`. Returns a set of legal next characters,
    or an empty set to mean 'the structure is already complete -- stop.'"""
    if len(prefix) < len(TEMPLATE_HEAD):
        return {TEMPLATE_HEAD[len(prefix)]}
    rest = prefix[len(TEMPLATE_HEAD):]

    quote_idx = rest.find('"')
    if quote_idx == -1:
        partial = rest  # mid-way through the enum word, no closing quote yet
        valid = {w[len(partial)] for w in ACTIONS if w.startswith(partial) and len(w) > len(partial)}
        if partial in ACTIONS:
            valid.add('"')          # the word is complete -- allowed to close the string now
        return valid

    partial = rest[:quote_idx]      # guaranteed to be a full action word (see loop invariant below)
    after_quote = rest[quote_idx + 1:]
    if len(after_quote) < len(TEMPLATE_MID):
        return {TEMPLATE_MID[len(after_quote)]}

    digits_and_after = after_quote[len(TEMPLATE_MID):]
    brace_idx = digits_and_after.find("}")
    if brace_idx != -1:
        return set()               # closing brace already emitted -- structure complete, stop

    digits_so_far = digits_and_after
    if digits_so_far == "":
        return set("0123456789")                 # first digit: anything, including a lone '0'
    if digits_so_far == "0":
        return {"}"}                              # "0" cannot be followed by more digits (no leading zeros)
    if len(digits_so_far) < 3:
        return set("0123456789") | {"}"}          # 1-2 digits so far: may extend or close
    return {"}"}                                  # 3 digits: MUST close now


# --- A tiny, genuinely UNTRAINED decoder-only Transformer (random weights) ---

VOCAB_CHARS = sorted(set(TEMPLATE_HEAD + TEMPLATE_MID + "".join(ACTIONS) + "0123456789}"))
vocab_size = len(VOCAB_CHARS)
stoi = {ch: i for i, ch in enumerate(VOCAB_CHARS)}
itos = {i: ch for i, ch in enumerate(VOCAB_CHARS)}
BLOCK_SIZE = 40


class TinyCausalTransformer(nn.Module):
    """One tiny causal self-attention block. Weights are left at their
    random initialization ON PURPOSE -- Part A's entire point is that
    constrained decoding guarantees syntactic validity regardless of
    whether the model underneath has learned anything at all."""

    def __init__(self, vocab_size, d_model=32, num_heads=2, block_size=BLOCK_SIZE):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.num_heads = num_heads
        self.d_model = d_model
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, ids):
        batch, T = ids.shape
        positions = torch.arange(T)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        h = self.ln1(x)
        d_k = self.d_model // self.num_heads
        Q = self.W_q(h).view(batch, T, self.num_heads, d_k).transpose(1, 2)
        K = self.W_k(h).view(batch, T, self.num_heads, d_k).transpose(1, 2)
        V = self.W_v(h).view(batch, T, self.num_heads, d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).reshape(batch, T, self.d_model)
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return self.head(x)


@torch.no_grad()
def generate(model, constrained, max_len=BLOCK_SIZE, temperature=1.0, rng=None):
    """Generate one sequence, character by character. If `constrained` is
    True, logits for grammar-violating characters are set to -inf before
    every single sampling step. If False, the model samples freely over the
    whole vocabulary -- the baseline we compare against."""
    prefix = ""
    for _ in range(max_len):
        if constrained:
            valid = next_valid_chars(prefix)
            if not valid:
                break   # grammar says: structure is complete
        ids = torch.tensor([[stoi[c] for c in prefix]], dtype=torch.long) if prefix else torch.zeros((1, 1), dtype=torch.long)
        if prefix == "":
            logits = model(torch.zeros((1, 1), dtype=torch.long))[0, -1]
            # (an empty prefix has no tokens yet; we still need one forward pass
            # to get a first-step logit vector, so we seed with a dummy token 0
            # and immediately overwrite/ignore it via masking below)
        else:
            logits = model(ids)[0, -1]
        if constrained:
            mask = torch.full((vocab_size,), float("-inf"))
            for c in valid:
                mask[stoi[c]] = 0.0
            logits = logits + mask
        probs = F.softmax(logits / temperature, dim=-1)
        if torch.isnan(probs).any():
            break
        next_id = torch.multinomial(probs, num_samples=1, generator=rng).item()
        prefix += itos[next_id]
        if not constrained and len(prefix) >= max_len:
            break
    return prefix


def part_a_demo():
    print("=" * 78)
    print("PART A: CONSTRAINED DECODING GUARANTEES VALID STRUCTURED OUTPUT")
    print("=" * 78)
    print(f"Grammar: {TEMPLATE_HEAD}<action>{TEMPLATE_MID}<1-3 digit value>}}")
    print(f"  <action> in {ACTIONS}")
    print(f"Vocabulary ({vocab_size} characters): {VOCAB_CHARS}")

    model = TinyCausalTransformer(vocab_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: a {num_params:,}-parameter causal Transformer -- weights are")
    print("PURE RANDOM INITIALIZATION. No training happens anywhere in Part A.")

    print("\n" + "-" * 78)
    print("Self-check: verifying the grammar state machine against an INDEPENDENT")
    print("full-string validity checker, using pure random valid-choice walks")
    print("-" * 78)
    self_check_trials = 2000
    self_check_failures = 0
    for _ in range(self_check_trials):
        prefix = ""
        for _ in range(BLOCK_SIZE):
            valid = next_valid_chars(prefix)
            if not valid:
                break
            prefix += random.choice(sorted(valid))
        if not is_fully_valid(prefix):
            self_check_failures += 1
    print(f"{self_check_trials} random walks through the grammar's own valid-character sets;")
    print(f"failures against the INDEPENDENT validity checker: {self_check_failures}")
    print(f"-> The state machine and the independent checker agree {self_check_failures == 0}: "
          f"every path the grammar permits is genuinely valid.")

    print("\n" + "-" * 78)
    print("CONSTRAINED generations from the UNTRAINED model (mask applied every step)")
    print("-" * 78)
    num_generations = 8
    constrained_outputs = []
    for i in range(num_generations):
        rng = torch.Generator().manual_seed(100 + i)
        out = generate(model, constrained=True, rng=rng)
        constrained_outputs.append(out)
        print(f"  [{i}] {out!r}   valid={is_fully_valid(out)}")
    num_valid_constrained = sum(is_fully_valid(o) for o in constrained_outputs)
    print(f"\nValid outputs: {num_valid_constrained}/{num_generations}")

    print("\n" + "-" * 78)
    print("UNCONSTRAINED generations from the SAME untrained model (no mask at all)")
    print("-" * 78)
    unconstrained_outputs = []
    for i in range(num_generations):
        rng = torch.Generator().manual_seed(100 + i)
        out = generate(model, constrained=False, max_len=24, rng=rng)
        unconstrained_outputs.append(out)
        print(f"  [{i}] {out!r}   valid={is_fully_valid(out)}")
    num_valid_unconstrained = sum(is_fully_valid(o) for o in unconstrained_outputs)
    print(f"\nValid outputs: {num_valid_unconstrained}/{num_generations}")

    print(f"\n-> Same random, untrained model, same random seeds, same sampling procedure.")
    print(f"   With the grammar mask applied at every step: {num_valid_constrained}/{num_generations} valid.")
    print(f"   With no mask at all:                          {num_valid_unconstrained}/{num_generations} valid.")
    print("   Validity here comes ENTIRELY from masking illegal tokens to -inf before")
    print("   sampling, not from anything the model has learned -- constrained decoding")
    print("   makes a syntax guarantee that holds regardless of model quality, which is")
    print("   exactly why production structured-output APIs implement it at the")
    print("   decoding layer instead of just hoping a well-trained model complies.")


# ===========================================================================
# PART B: FUNCTION CALLING -- REQUEST, PARSE, EXECUTE, RESPOND
# ===========================================================================

def tool_calculator(expression):
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError(f"Unsupported characters in expression: {expression!r}")
    return eval(expression, {"__builtins__": {}}, {})


UNIT_CONVERSIONS = {
    ("miles", "km"): 1.60934,
    ("km", "miles"): 1 / 1.60934,
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 1 / 2.20462,
}


def tool_convert_units(value, from_unit, to_unit):
    factor = UNIT_CONVERSIONS.get((from_unit, to_unit))
    if factor is None:
        raise ValueError(f"No conversion known for {from_unit} -> {to_unit}")
    return value * factor


TOOLS = {
    "calculator": tool_calculator,
    "convert_units": tool_convert_units,
}


def run_function_call(model_output_text):
    """Parses a structured function-call request out of a (here: scripted)
    model output string, executes the REAL corresponding Python function
    with the REAL parsed arguments, and returns the tool's real result."""
    call = json.loads(model_output_text)
    name = call["name"]
    arguments = call["arguments"]
    if name not in TOOLS:
        raise ValueError(f"Model requested unknown tool: {name!r}")
    result = TOOLS[name](**arguments)
    return name, arguments, result


def part_b_demo():
    print("\n" + "=" * 78)
    print("PART B: FUNCTION CALLING -- STRUCTURED REQUEST -> PARSE -> EXECUTE -> RESPOND")
    print("=" * 78)
    print("Each 'model output' below is a SCRIPTED string (there is no LLM generating")
    print("it) standing in for what a real model would emit in the standard")
    print('{"name": ..., "arguments": {...}} function-calling format. Everything AFTER')
    print("that point -- JSON parsing, tool dispatch, and execution -- is real code")
    print("running on real inputs, with a real return value spliced back in.\n")

    scenarios = [
        {
            "user_query": "What is 128 times 37, plus 6?",
            "model_output": '{"name": "calculator", "arguments": {"expression": "128 * 37 + 6"}}',
            "response_template": "The result of 128 * 37 + 6 is {result}.",
        },
        {
            "user_query": "Convert 42 kilometers to miles.",
            "model_output": '{"name": "convert_units", "arguments": {"value": 42, "from_unit": "km", "to_unit": "miles"}}',
            "response_template": "42 km is approximately {result:.2f} miles.",
        },
    ]

    for i, scenario in enumerate(scenarios, start=1):
        print(f"--- Scenario {i} ---")
        print(f"User query:    {scenario['user_query']}")
        print(f"Model output:  {scenario['model_output']}")
        name, arguments, result = run_function_call(scenario["model_output"])
        print(f"Parsed call:   name={name!r}, arguments={arguments}")
        print(f"Tool executed. Real return value: {result!r}")
        final_response = scenario["response_template"].format(result=result)
        print(f"Final response (real tool result spliced in): {final_response!r}\n")

    # Independent verification that the spliced-in numbers are actually correct,
    # computed completely separately from the tool-calling machinery above.
    expected_1 = 128 * 37 + 6
    expected_2 = 42 * (1 / 1.60934)
    _, _, result_1 = run_function_call(scenarios[0]["model_output"])
    _, _, result_2 = run_function_call(scenarios[1]["model_output"])
    print(f"Independent check -- scenario 1: expected {expected_1}, tool returned {result_1}, "
          f"match={expected_1 == result_1}")
    print(f"Independent check -- scenario 2: expected {expected_2:.4f}, tool returned {result_2:.4f}, "
          f"match={abs(expected_2 - result_2) < 1e-9}")
    print("\n-> Both final responses embed a number the calling code could not have")
    print("   known in advance without actually running the requested tool -- this is")
    print("   the entire value of function calling: the model's job is reduced to")
    print("   emitting a syntactically valid REQUEST (which Part A showed can be")
    print("   guaranteed structurally), while a real system executes it and supplies")
    print("   the real answer back into the conversation.")


def main():
    part_a_demo()
    part_b_demo()


if __name__ == "__main__":
    main()
