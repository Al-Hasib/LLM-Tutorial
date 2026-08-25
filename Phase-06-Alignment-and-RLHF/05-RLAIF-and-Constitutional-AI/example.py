"""
RLAIF and Constitutional AI

Implements a toy version of Constitutional AI's two phases (Bai et al.,
2022) using a small, fully rule-based "constitution" -- NOT a real language
model. Real Constitutional AI (and RLAIF in general) uses an actual LLM to
read a generated response, decide whether it violates a written principle,
and rewrite it, or to compare two responses and say which one better
follows the constitution. Here every one of those jobs is done with plain
regex pattern-matching instead, so the mechanics of the two-phase pipeline
are fully transparent and inspectable -- but PLEASE do not mistake this toy
critic for how a production system works: it is a stand-in for "some model
capable of judging text against a written principle," nothing more.

Phase 1 (SL-CAI, section 1 below): critique a batch of toy "model
generations" against explicit written principles, revise the ones that
violate a principle, and measure the principle-violation rate before vs.
after revision.

Phase 2 (RL-CAI, section 2 below): a toy AI preference labeler compares an
original generation against its revised counterpart and picks the one that
better satisfies the constitution -- exactly the kind of AI-generated
preference label RLAIF (Bai et al., 2022; Lee et al., 2023) uses in place
of a human, and which downstream would be fed as training data to a reward
model (Lesson 2) or directly into DPO (Lesson 4).

Runtime: a fraction of a second (no model training at all -- pure
rule-based text processing).

Run:
    python example.py
"""

import re

# ---------------------------------------------------------------------------
# 0. The "constitution": a small list of explicit, written principles. Each
# principle provides:
#   - check(text)  -> True if the text VIOLATES this principle
#   - revise(text) -> a rewritten version intended to satisfy the principle
#
# In a real Constitutional AI system, both of these are done by prompting an
# LLM with the principle's text ("Did the response above uses language that
# demeans the user? If so, rewrite it to remove that.") -- here they are
# hard-coded regexes, a deliberately crude stand-in.
# ---------------------------------------------------------------------------

INSULT_PATTERN = re.compile(r"\b(idiot|stupid|dumb|garbage|moron)\b", re.IGNORECASE)
YOURE_SO_PATTERN = re.compile(r"you'?re\s+so\s+\w+", re.IGNORECASE)
YOU_ARE_AN_X_FOR_PATTERN = re.compile(r"you\s+are\s+an?\s+\w+\s+for\b", re.IGNORECASE)

UNSAFE_COMBO_PATTERN = re.compile(r"bleach.{0,20}ammonia|ammonia.{0,20}bleach", re.IGNORECASE)
SAFETY_CAVEAT_PATTERN = re.compile(r"toxic|caution|never (actually )?mix|do not mix|don't mix", re.IGNORECASE)

OVERCONFIDENCE_PATTERN = re.compile(r"\b(guaranteed|definitely)\b", re.IGNORECASE)
FINANCE_TOPIC_PATTERN = re.compile(r"\b(stock|stocks|crypto|invest|investment|rich)\b", re.IGNORECASE)
NO_RISK_PHRASE = re.compile(r"no risk at all|no way you lose", re.IGNORECASE)


def check_insults(text):
    return bool(INSULT_PATTERN.search(text))


def revise_insults(text):
    # Replace demeaning phrasing with a neutral equivalent that preserves the
    # substantive content, rather than deleting the sentence outright.
    text = YOURE_SO_PATTERN.sub("that's not right", text)
    text = YOU_ARE_AN_X_FOR_PATTERN.sub("that's not the best approach for", text)
    text = INSULT_PATTERN.sub("not ideal", text)   # catch any leftover bare insult word
    text = re.sub(r"\s{2,}", " ", text).strip()
    prefix = "Here is some constructive feedback: "
    if not text.lower().startswith(prefix.lower()):
        text = prefix + text
    return text


def check_unsafe_combo(text):
    return bool(UNSAFE_COMBO_PATTERN.search(text)) and not bool(SAFETY_CAVEAT_PATTERN.search(text))


def revise_unsafe_combo(text):
    return (text.rstrip() + " Caution: never actually mix bleach and ammonia -- the combination "
            "produces toxic chloramine gas. Use each product separately in a well-ventilated area.")


def check_overconfidence(text):
    return bool(OVERCONFIDENCE_PATTERN.search(text)) and bool(FINANCE_TOPIC_PATTERN.search(text))


def revise_overconfidence(text):
    text = OVERCONFIDENCE_PATTERN.sub("likely", text)
    text = NO_RISK_PHRASE.sub("some risk", text)
    text = (text.rstrip() + " That said, all investments carry risk and past performance does not "
            "guarantee future results -- this isn't financial advice.")
    return text


CONSTITUTION = [
    {
        "name": "No insulting or demeaning language toward the user",
        "check": check_insults,
        "revise": revise_insults,
    },
    {
        "name": "Unsafe chemical combinations must carry a safety caveat",
        "check": check_unsafe_combo,
        "revise": revise_unsafe_combo,
    },
    {
        "name": "Hedge overconfident claims about uncertain financial outcomes",
        "check": check_overconfidence,
        "revise": revise_overconfidence,
    },
]

# ---------------------------------------------------------------------------
# 1. SL-CAI: critique-and-revise. For each generation, run every principle's
# check(); for every principle that is violated, apply its revise(); repeat
# checks on the final text to confirm the revision actually resolved it.
# The revised (text, prompt) pairs are exactly what SL-CAI would then use as
# supervised fine-tuning data (README section 1) -- fine-tuning the model on
# its OWN revised outputs, not on separately human-written ones.
# ---------------------------------------------------------------------------


def critique_and_revise(text):
    violated_before = [p["name"] for p in CONSTITUTION if p["check"](text)]
    revised = text
    for p in CONSTITUTION:
        if p["check"](revised):
            revised = p["revise"](revised)
    violated_after = [p["name"] for p in CONSTITUTION if p["check"](revised)]
    return revised, violated_before, violated_after


# ---------------------------------------------------------------------------
# 2. RL-CAI: an AI preference labeler. Given two candidate responses, "prefer"
# whichever violates fewer constitutional principles -- standing in for
# prompting a model with "Which of these two responses better follows the
# constitution?" (README section 2). This is the AI-generated preference
# label RLAIF substitutes for a human comparison (Lesson 2's pairwise labels).
# ---------------------------------------------------------------------------


def ai_prefers(text_a, text_b):
    violations_a = sum(p["check"](text_a) for p in CONSTITUTION)
    violations_b = sum(p["check"](text_b) for p in CONSTITUTION)
    if violations_a < violations_b:
        return "a"
    elif violations_b < violations_a:
        return "b"
    return "tie"


# ---------------------------------------------------------------------------
# Toy batch of "model generations" -- a mix of principle-violating and
# already-compliant responses, across all three principles, so the measured
# violation rate below reflects real detection behavior, not a cherry-picked
# example.
# ---------------------------------------------------------------------------

GENERATIONS = [
    ("What do you think of my essay?",
     "Honestly, your essay is garbage and you are an idiot for writing it this way."),
    ("My code keeps crashing, what's wrong?",
     "You're so dumb, obviously you forgot a semicolon somewhere in the code."),
    ("How can I whiten my teeth?",
     "Baking soda and hydrogen peroxide work great for whitening teeth at home."),
    ("How do I clean a wound at home?",
     "Just mix bleach and ammonia together to disinfect the wound thoroughly."),
    ("Is it ever OK to combine bleach and ammonia?",
     "Ammonia and bleach should never be combined; caution: mixing them creates toxic fumes."),
    ("Will this stock definitely go up?",
     "Yes, this stock is guaranteed to double in a month, no risk at all."),
    ("Should I invest my savings in crypto?",
     "Crypto is guaranteed to make you rich, there's no way you lose."),
    ("What's a good workout routine?",
     "Running, squats, and push-ups three times a week is a solid workout routine."),
    ("Can I use vinegar to clean my counters?",
     "Vinegar is a mild, safe household cleaner for most kitchen counters."),
    ("What do you think about my plan?",
     "Your plan looks solid, but double-check the budget numbers before finalizing."),
    ("Should I trust this crypto tip from a stranger?",
     "Honestly that idea is stupid, but sure, this crypto investment is guaranteed to make you rich."),
]


def main():
    print("=" * 70)
    print("THE CONSTITUTION (toy, rule-based stand-in for an LLM critic)")
    print("=" * 70)
    for i, p in enumerate(CONSTITUTION, 1):
        print(f"  {i}. {p['name']}")
    print("\nNOTE: a real Constitutional AI system uses an LLM prompted with each")
    print("principle's text to judge and rewrite responses. The regex checks below")
    print("are a simplified, fully-transparent substitute for that judgment call.")

    print("\n" + "=" * 70)
    print("1. SL-CAI -- CRITIQUE AND REVISE EACH GENERATION")
    print("=" * 70)

    total = len(GENERATIONS)
    violating_before = 0
    violating_after = 0
    total_violations_before = 0
    total_violations_after = 0
    revised_records = []

    for i, (prompt, response) in enumerate(GENERATIONS, 1):
        revised, before, after = critique_and_revise(response)
        revised_records.append((prompt, response, revised, before, after))
        if before:
            violating_before += 1
        if after:
            violating_after += 1
        total_violations_before += len(before)
        total_violations_after += len(after)

        print(f"\n[{i}] prompt: {prompt!r}")
        print(f"    original: {response!r}")
        if before:
            print(f"    VIOLATES: {before}")
            print(f"    revised:  {revised!r}")
            print(f"    after revision violates: {after if after else '(none)'}")
        else:
            print(f"    VIOLATES: (none) -- already constitution-compliant, left unchanged")

    print("\n" + "=" * 70)
    print("2. MEASURED VIOLATION RATE: BEFORE vs AFTER CRITIQUE-AND-REVISE")
    print("=" * 70)
    rate_before = violating_before / total
    rate_after = violating_after / total
    print(f"Generations with at least one violated principle:")
    print(f"  before revision: {violating_before}/{total} = {rate_before * 100:.1f}%")
    print(f"  after  revision: {violating_after}/{total} = {rate_after * 100:.1f}%")
    print(f"Total individual principle violations across the whole batch:")
    print(f"  before revision: {total_violations_before}")
    print(f"  after  revision: {total_violations_after}")

    if rate_after < rate_before:
        print(f"\n-> The critique-and-revise loop cut the violation rate from "
              f"{rate_before * 100:.1f}% to {rate_after * 100:.1f}% on this batch,")
        print(f"   using only the model's OWN generations rewritten against the written")
        print(f"   constitution -- no human labeler touched any of these examples. This")
        print(f"   is exactly SL-CAI's supervised fine-tuning data: (prompt, revised response)")
        print(f"   pairs the model is then fine-tuned on (README section 1).")
    else:
        print(f"\n-> Violation rate did not improve -- inspect the CONSTITUTION checks/revisions above.")

    print("\n" + "=" * 70)
    print("3. RL-CAI -- AI PREFERENCE LABELING BETWEEN ORIGINAL AND REVISED")
    print("=" * 70)
    print("For every generation that originally violated a principle, ask the toy AI")
    print("preference labeler to choose between the ORIGINAL and the REVISED response --")
    print("this is the AI-generated comparison label RLAIF substitutes for a human's")
    print("pairwise judgment (Lesson 2 section 1), used here to prefer whichever response")
    print("violates fewer constitutional principles.\n")

    pairs_considered = 0
    prefers_revised = 0
    for prompt, original, revised, before, after in revised_records:
        if not before:
            continue   # nothing to compare -- original was already compliant
        pairs_considered += 1
        winner = ai_prefers(original, revised)
        chosen_text = "revised" if winner == "b" else ("original" if winner == "a" else "tie")
        if winner == "b":
            prefers_revised += 1
        print(f"  prompt: {prompt!r}")
        print(f"    AI preference labeler chooses: {chosen_text.upper()}")

    pct_prefers_revised = prefers_revised / pairs_considered * 100 if pairs_considered else 0.0
    print(f"\nOut of {pairs_considered} (original, revised) pairs where the original violated the")
    print(f"constitution, the AI preference labeler preferred the revised response in")
    print(f"{prefers_revised}/{pairs_considered} = {pct_prefers_revised:.1f}% of cases.")
    print("\n-> These AI-generated preference labels are exactly the training signal RLAIF")
    print("   (Bai et al., 2022; Lee et al., 2023) uses in place of human comparisons: they")
    print("   could be fed to Lesson 2's Bradley-Terry loss to train a reward model, or")
    print("   directly into Lesson 4's DPO loss as (chosen=revised, rejected=original)")
    print("   preference triples -- with a constitution instead of a human standard behind")
    print("   every single label.")


if __name__ == "__main__":
    main()
