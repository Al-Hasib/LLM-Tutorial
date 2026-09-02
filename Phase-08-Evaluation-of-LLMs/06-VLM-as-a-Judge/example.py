"""
VLM-as-a-Judge

This repo has no real vision model and no actual image pixels to work with,
so this script builds something smaller but genuinely real instead of a
hand-wavy description: a "scene" is an explicit, structured ground-truth
data structure (colors/shapes/counts + one spatial relation) standing in
for an image -- a tiny CLEVR-style scene graph, not a picture. From each
scene, a fixed caption template generates one fully correct caption and
four labeled error variants (object hallucination, count error, attribute
error, spatial error). Two judges then grade these captions:

  1. The GROUNDED judge is handed the scene graph and re-extracts every
     count/relation claim a caption makes with a small, honest regex-based
     parser, then checks each claim against the real ground truth.
  2. The UNGROUNDED judge sees only the caption text -- no scene -- and
     falls back to the one signal it has left: response length. This is
     Phase 08 Lesson 3's verbosity bias, not bolted on as an extra flaw,
     but as literally the only thing left for a judge that cannot check
     substance at all.

Both judges are run thousands of times on random scene/caption pairs, and
the REAL measured pairwise win rates (does the judge prefer the fully
correct caption over each error variant?) are reported for every error
type, plus one worked single-scene walkthrough.

Run:
    python example.py
"""

import random
import re

random.seed(0)

# ---------------------------------------------------------------------------
# 0. The scene graph -- a symbolic stand-in for an image
# ---------------------------------------------------------------------------

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle"]
RELATIONS = ["left of", "right of", "above", "below"]
OPPOSITE_RELATION = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}


def make_random_scene(rng):
    """A scene = two distinct (color, shape) objects, each with a count,
    plus one spatial relation between them. This is the ENTIRE ground
    truth -- everything a real image would need to encode for this task,
    made explicit and checkable instead of hidden in pixels."""
    all_combos = [(c, s) for c in COLORS for s in SHAPES]
    (color_a, shape_a), (color_b, shape_b) = rng.sample(all_combos, 2)
    objects = [
        {"color": color_a, "shape": shape_a, "count": rng.randint(1, 3)},
        {"color": color_b, "shape": shape_b, "count": rng.randint(1, 3)},
    ]
    relation = rng.choice(RELATIONS)
    return {
        "objects": objects,
        "relation": (color_a, shape_a, relation, color_b, shape_b),
    }


def true_caption(scene):
    """The one fully correct caption for a scene, from a fixed template --
    rigid on purpose, so the grounded judge's parser below can be a simple,
    honest regex instead of needing real NLP."""
    obj_sentences = [f"There are {o['count']} {o['color']} {o['shape']}(s)." for o in scene["objects"]]
    ca, sa, rel, cb, sb = scene["relation"]
    rel_sentence = f"The {ca} {sa} is {rel} the {cb} {sb}."
    return " ".join(obj_sentences + [rel_sentence])


# ---------------------------------------------------------------------------
# 1. Four labeled error injections
# ---------------------------------------------------------------------------

def inject_hallucination(scene, rng):
    """Append a sentence about an object combination that is NOT in the
    scene at all -- the caption claims to see something that isn't there."""
    present = {(o["color"], o["shape"]) for o in scene["objects"]}
    absent_combos = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
    color, shape = rng.choice(absent_combos)
    fake_count = rng.randint(1, 3)
    return true_caption(scene) + f" There are {fake_count} {color} {shape}(s)."


def inject_count_error(scene, rng):
    """Swap one object's stated count to a wrong number -- same length as
    the correct caption, so an ungrounded judge gets no length cue at all."""
    caption = true_caption(scene)
    obj = rng.choice(scene["objects"])
    wrong_count = obj["count"]
    while wrong_count == obj["count"]:
        wrong_count = rng.randint(1, 4)
    old = f"There are {obj['count']} {obj['color']} {obj['shape']}(s)."
    new = f"There are {wrong_count} {obj['color']} {obj['shape']}(s)."
    return caption.replace(old, new, 1)


def inject_attribute_error(scene, rng):
    """Swap one object's color to one that matches no object actually
    present -- again, no length change versus the correct caption."""
    caption = true_caption(scene)
    obj = rng.choice(scene["objects"])
    present_combos = {(o["color"], o["shape"]) for o in scene["objects"]}
    wrong_colors = [c for c in COLORS if (c, obj["shape"]) not in present_combos]
    wrong_color = rng.choice(wrong_colors)
    old = f"There are {obj['count']} {obj['color']} {obj['shape']}(s)."
    new = f"There are {obj['count']} {wrong_color} {obj['shape']}(s)."
    return caption.replace(old, new, 1)


def inject_spatial_error(scene, rng):
    """Flip the stated relation to its opposite -- again no length change."""
    caption = true_caption(scene)
    ca, sa, rel, cb, sb = scene["relation"]
    old = f"The {ca} {sa} is {rel} the {cb} {sb}."
    new = f"The {ca} {sa} is {OPPOSITE_RELATION[rel]} the {cb} {sb}."
    return caption.replace(old, new, 1)


ERROR_INJECTORS = {
    "hallucination": inject_hallucination,
    "count": inject_count_error,
    "attribute": inject_attribute_error,
    "spatial": inject_spatial_error,
}


# ---------------------------------------------------------------------------
# 2. The GROUNDED judge -- real claim extraction + real ground-truth check
# ---------------------------------------------------------------------------

COUNT_CLAIM_RE = re.compile(r"There are (\d+) (\w+) (\w+)\(s\)\.")
RELATION_CLAIM_RE = re.compile(r"The (\w+) (\w+) is (left of|right of|above|below) the (\w+) (\w+)\.")


def grounded_judge(scene, caption):
    """Parses every count/relation claim OUT of the caption text and checks
    each one against the scene's real ground truth. Returns (fraction of
    claims that are true, list of (claim_text, is_true, reason))."""
    true_counts = {(o["color"], o["shape"]): o["count"] for o in scene["objects"]}
    claims = []

    for match in COUNT_CLAIM_RE.finditer(caption):
        count, color, shape = int(match.group(1)), match.group(2), match.group(3)
        key = (color, shape)
        if key not in true_counts:
            claims.append((match.group(0), False, "no such object in the scene"))
        elif true_counts[key] != count:
            claims.append((match.group(0), False, f"actual count is {true_counts[key]}"))
        else:
            claims.append((match.group(0), True, "matches the scene"))

    for match in RELATION_CLAIM_RE.finditer(caption):
        claimed = match.groups()
        if claimed == scene["relation"]:
            claims.append((match.group(0), True, "matches the scene"))
        else:
            claims.append((match.group(0), False, "relation does not match the scene"))

    score = sum(is_true for _, is_true, _ in claims) / len(claims)
    return score, claims


# ---------------------------------------------------------------------------
# 3. The UNGROUNDED judge -- no scene access, only response length
# (exactly Lesson 3's verbosity bias, now the judge's ONLY available signal)
# ---------------------------------------------------------------------------

def ungrounded_judge(caption, rng, length_coef=0.05, noise_std=1.0):
    return length_coef * len(caption) + rng.gauss(0.0, noise_std)


# ---------------------------------------------------------------------------
# 4. A worked single-scene walkthrough
# ---------------------------------------------------------------------------

def worked_example():
    print("=" * 78)
    print("A WORKED EXAMPLE: ONE SCENE, ONE CAPTION PER ERROR TYPE")
    print("=" * 78)
    rng = random.Random(7)
    scene = make_random_scene(rng)

    print("Scene (ground truth):")
    for obj in scene["objects"]:
        print(f"  {obj['count']} x {obj['color']} {obj['shape']}")
    ca, sa, rel, cb, sb = scene["relation"]
    print(f"  relation: the {ca} {sa} is {rel} the {cb} {sb}\n")

    correct = true_caption(scene)
    print(f"Correct caption: {correct!r}")
    score, claims = grounded_judge(scene, correct)
    print(f"  grounded judge score: {score:.2f}  (all claims true: {score == 1.0})\n")

    for error_type, inject in ERROR_INJECTORS.items():
        flawed = inject(scene, rng)
        score, claims = grounded_judge(scene, flawed)
        print(f"[{error_type}] caption: {flawed!r}")
        for claim_text, is_true, reason in claims:
            flag = "OK   " if is_true else "FALSE"
            print(f"    {flag}  {claim_text!r}  ({reason})")
        print(f"    grounded judge score: {score:.2f}\n")


# ---------------------------------------------------------------------------
# 5. Aggregate pairwise win rates, over many random scenes
# ---------------------------------------------------------------------------

def aggregate_demo():
    print("=" * 78)
    print("AGGREGATE: DOES EACH JUDGE PREFER THE CORRECT CAPTION OVER THE FLAWED ONE?")
    print("=" * 78)
    print("For each error type, 3000 random scenes; a judge 'wins' the trial if it")
    print("scores the fully correct caption higher than the flawed one.\n")

    n_trials = 3000
    rng = random.Random(0)

    header = f"{'error type':16}{'grounded win rate':>20}{'ungrounded win rate':>22}{'caption length delta':>24}"
    print(header)
    print("-" * len(header))

    for error_type, inject in ERROR_INJECTORS.items():
        grounded_wins = 0
        ungrounded_wins = 0
        length_deltas = []

        for _ in range(n_trials):
            scene = make_random_scene(rng)
            correct = true_caption(scene)
            flawed = inject(scene, rng)
            length_deltas.append(len(flawed) - len(correct))

            grounded_correct_score, _ = grounded_judge(scene, correct)
            grounded_flawed_score, _ = grounded_judge(scene, flawed)
            if grounded_correct_score > grounded_flawed_score:
                grounded_wins += 1

            ungrounded_correct_score = ungrounded_judge(correct, rng)
            ungrounded_flawed_score = ungrounded_judge(flawed, rng)
            if ungrounded_correct_score > ungrounded_flawed_score:
                ungrounded_wins += 1

        grounded_rate = grounded_wins / n_trials
        ungrounded_rate = ungrounded_wins / n_trials
        avg_length_delta = sum(length_deltas) / len(length_deltas)
        print(f"{error_type:16}{grounded_rate:>19.1%} {ungrounded_rate:>21.1%} "
              f"{avg_length_delta:>+23.1f}")

    print("\n-> The grounded judge wins essentially every trial for every error type --")
    print("   it is CHECKING real structured facts, so there is nothing for a flawed")
    print("   caption to exploit. The ungrounded judge has no image access at all, so")
    print("   for count/attribute/spatial errors (~0 average length delta) it is at")
    print("   pure CHANCE -- its ~50% win rate reflects zero real signal, not partial")
    print("   competence. For hallucination errors (positive length delta: the flawed")
    print("   caption is LONGER), the ungrounded judge's verbosity bias actively")
    print("   prefers the WORSE, hallucinated caption -- its win rate for the correct")
    print("   one is pushed BELOW 50%, the wrong direction entirely. This is Lesson 3's")
    print("   verbosity bias, deployed here as literally the only signal an image-blind")
    print("   judge has left.")


def main():
    worked_example()
    print()
    aggregate_demo()


if __name__ == "__main__":
    main()
