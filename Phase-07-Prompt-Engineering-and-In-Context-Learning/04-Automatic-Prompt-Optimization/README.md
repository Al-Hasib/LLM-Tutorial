# Automatic Prompt Optimization

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `04-Automatic-Prompt-Optimization`

## Why this matters

Every lesson so far in this phase has asked "how does changing the prompt affect accuracy?" — [Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md) measured the effect of example count and order by hand; [Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) and [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md) each introduced a specific hand-designed prompting strategy. This lesson asks the natural follow-up question: instead of a human guessing which prompt variant is best and checking it manually, can the search itself be **automated**? Prompt design turns out to be a discrete, gradient-free optimization problem — no different in kind from hyperparameter search — and `example.py` builds and runs a real instance of that search end to end.

## What this lesson covers

- Framing prompt design as gradient-free, discrete optimization over a combinatorial space
- Automatic Prompt Engineer (APE): propose candidate prompts, score on a validation objective, keep the best (Zhou et al., 2022)
- Building a real, measurable prompt-component search space out of Lesson 1's task
- Why some prompt-component *combinations* score worse even when each component looks fine individually
- Random search vs. hill-climbing (greedy local search) as two concrete search strategies
- `example.py`: both methods find a reliably better prompt combination than picking one at random, verified against the true (brute-forced) optimum

## 1. Prompt design as discrete optimization

A prompt is built out of choices: which instruction wording, how many examples, what order, what separators. Each choice is a **discrete variable** with a small number of possible values, and the full prompt is one point in the **Cartesian product** of all those choices. Hand-tuning a prompt is a human doing local search by hand — try something, eyeball the output, tweak one thing, repeat. **Automatic prompt optimization** replaces the human's judgment with a real scoring function (accuracy on a held-out validation set) and a real search procedure, exactly the same shift that hyperparameter tuning made decades ago when grid/random search replaced hand-picked learning rates.

## 2. APE: Automatic Prompt Engineer (Zhou et al., 2022)

Zhou et al. (2022) formalized this loop for instruction prompts specifically:

1. **Propose**: generate a pool of candidate instructions (often by asking a model to infer plausible instructions from example input/output pairs, or by combining known-good instruction fragments).
2. **Score**: evaluate every candidate on a held-out validation set, using a real accuracy or log-likelihood metric — not a proxy, not a guess.
3. **Select / iterate**: keep the best-scoring candidates, optionally generate variations of them ("resample around the best," a form of hill-climbing), and repeat.

This is the exact template `example.py` follows, just with a search space small and cheap enough to score exhaustively for verification purposes, and using a genuinely trained small model as the scoring function instead of an external API.

## 3. Building a real, measurable search space

`example.py` retrains Lesson 1's `y = (x + k) mod M` in-context task, but this time the training distribution varies **three independent prompt components** every episode:

1. **`NUM_EXAMPLES`** in `{2, 3, 4, 5}` — how many in-context pairs are shown.
2. **`ORDER`** in `{sorted, scrambled}` — the order those pairs appear in.
3. **`PHRASING`** in `{'P', 'Q'}` — a single leading token standing in for two different instruction phrasings a prompt engineer might A/B test in a real natural-language system.

That's a `4 x 2 x 2 = 16`-combination discrete search space. Critically, the training data does **not** cover it evenly: marker `'Q'` only ever appears together with 2-3 shown examples, while marker `'P'` appears with the full range. This means `(marker='Q', n_shown in {4,5})` is a combination the trained model has genuinely never encountered — even though each individual component (`'Q'`, and `n_shown=5`) is completely familiar on its own. This mirrors a very real prompt-engineering trap: an instruction phrasing and a formatting choice can each look fine in isolation and still combine into something the model handles worse, purely because that specific *combination* was underrepresented wherever the model's behavior was shaped.

## 4. Two search strategies, checked against the true optimum

Because this toy space has only 16 combinations, `example.py` can afford to brute-force score every single one — a luxury real prompt-component spaces (with many more instruction variants, example pools, and formatting options) never have. That brute-force sweep is used purely as **ground truth to check the search methods against**, not as the proposed method itself. On top of it, two real search procedures are run:

- **Random search**: sample a handful of combinations uniformly at random, score each, keep the best.
- **Hill-climbing (greedy local search)**: start at one random combination; repeatedly look at every combination reachable by changing exactly *one* component, move to whichever neighbor scores highest, and stop once no neighboring change improves the score (a local optimum).

`example.py` reports, from its own live run: the mean accuracy across all 16 combinations (the "pick blindly" baseline), the best combination random search happened to sample, and where hill-climbing's local search converges — plus how close that converged result lands to the true brute-forced best. The concrete numbers are printed directly by the script rather than asserted here, since they come from one specific trained model's run.

## 5. Why this generalizes beyond this toy example

Nothing about random search or hill-climbing over a discrete space depends on the space being small or the scorer being a toy model. Swap in a real LLM API call as the scorer and a much larger space of instruction templates, example pools, and formatting choices, and the exact same two algorithms (or their more sophisticated descendants — Bayesian optimization, evolutionary search, or prompting a model to propose better prompts as APE does) are what production automatic-prompt-optimization tools actually run. The only thing that changes is the cost per evaluation and the size of the space — the search *logic* demonstrated here is unchanged.

## Video Script Outline

1. Motivation — stop hand-guessing prompts; treat prompt design as an optimization problem with a real scoring function
2. APE (Zhou et al. 2022): propose, score, select — the template every method here follows
3. Building the search space: 3 components, 16 combinations, deliberately uneven training coverage
4. Why "individually fine, jointly untested" combinations are a real prompt-engineering trap, not just a toy quirk
5. Walkthrough of `example.py`'s brute-force ground truth: which combinations score well, which don't, and why
6. Random search vs. hill-climbing, live results from the run
7. Comparing both methods against the true optimum: does automated search reliably beat picking blindly?
8. Recap + preview: once a prompt reliably contains structured content, how do you make the model's OUTPUT reliably structured too? (Lesson 5)

## Further Reading

- Zhou et al. (2022), *Large Language Models Are Human-Level Prompt Engineers* (APE)
- Shin et al. (2020), *AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts*
- Pryzant et al. (2023), *Automatic Prompt Optimization with "Gradient Descent" and Beam Search*
- Zhao et al. (2021), *Calibrate Before Use: Improving Few-Shot Performance of Language Models* (revisited from [Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md); the underlying prompt-sensitivity phenomenon this lesson's search is navigating)
