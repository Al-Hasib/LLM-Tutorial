# Tree-of-Thought and ReAct

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `03-Tree-of-Thought-and-ReAct`

## Why this matters

[Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) treated reasoning as **one linear chain** — a single sequence of steps from question to answer, possibly resampled independently several times and majority-voted at the end. This lesson removes that restriction in two different directions. Tree-of-Thought turns a single chain into an explicit **search tree**, exploring and comparing multiple partial reasoning paths *before* committing to one, which matters whenever an early wrong turn is hard to recover from. ReAct removes a different restriction: it lets the model's reasoning steps be interleaved with **actions that touch the outside world** (a tool call, a lookup, a calculator) rather than reasoning purely from what's already in the prompt. Both ideas are prerequisites for [Lesson 5](../05-Structured-Output-and-Function-Calling/README.md), which covers how a model's tool-call requests are formatted and parsed reliably in production systems.

## What this lesson covers

- Tree-of-Thought (ToT): reasoning as tree search over candidate "thoughts," with a heuristic evaluator judging and pruning branches
- Why naive greedy single-path reasoning can get provably stuck, and how keeping multiple candidate branches alive fixes it
- `example.py` Part A: BFS, ToT-style pruned best-first search, and naive greedy, compared head-to-head on an identical toy search puzzle
- ReAct: interleaving Reasoning, Actions, and Observations in a loop
- Why ReAct's tool calls matter: some information genuinely isn't in the prompt and reasoning alone cannot produce it
- `example.py` Part B: a scripted ReAct-style agent making real tool calls to answer a question it cannot answer from text alone

## 1. Tree-of-Thought (Yao et al., 2023): reasoning as search

Chain-of-Thought commits to one linear sequence of intermediate steps. **Tree-of-Thought (ToT)** instead treats problem-solving as **search over a tree of "thoughts"**, where each node is a partial solution state:

1. **Generate**: from the current state, propose several different next "thoughts" (candidate next steps) — this is the tree's branching factor.
2. **Evaluate**: score each candidate thought with a heuristic (a hand-written rule, or the model itself asked to self-assess promise).
3. **Prune / select**: keep only the most promising candidates and discard the rest, before recursing one level deeper.
4. **Search**: repeat generate-evaluate-prune level by level (breadth-first-style) or path by path (depth-first-style, with backtracking) until a solution is found or the search budget is exhausted.

This is a direct generalization of classical AI search (BFS, best-first search, beam search) applied to a reasoning trace instead of a game board — the "evaluator" plays the role a heuristic function (or, in games, a value network) plays in those classical algorithms.

## 2. Why naive greedy reasoning fails, and how ToT fixes it

A single Chain-of-Thought path is exactly a **greedy, single-path search**: at each step, the model commits to one continuation and never revisits that choice. This works fine when every reasonable next step keeps all options open — but on tasks where an early, locally-appealing choice can quietly close off the only route to a good final answer, greedy search has no way to recover: it has committed, and it does not backtrack.

`example.py` Part A makes this concrete and measurable with a genuine search puzzle: starting from an integer, reach a target integer in as few steps as possible using a fixed operation set (`+3`, `-1`, `*2`). Three real strategies are run on identical instances:

- **BFS oracle** — exhaustive, deduplicated breadth-first search; the ground truth for the true optimal (fewest-steps) solution.
- **ToT-style pruned best-first search** — at every depth, all children of the current frontier are generated (a raw branching factor of 3, so an *unpruned* tree would grow as `3^depth` — real branch explosion), each is scored by a heuristic (distance to target), and only the best `beam_width` survive to the next depth. This is exactly Yao et al.'s generate → evaluate → prune loop.
- **Naive greedy** — one path, no branching, no backtracking: always take whichever single next step looks locally best.

Across 200 random puzzle instances, the ToT-style pruned search matches the true optimal solution length the large majority of the time while examining only a small fraction of the nodes a full unpruned tree or exhaustive BFS would need — while naive greedy fails outright (gets stuck in a state it has already visited, with no way to try anything else) or lands on a strictly longer, worse solution in **over half** of instances. `example.py` prints the exact percentages from its own run, plus one concrete instance where greedy's single locally-best choice leads it to a solution that takes 50% more steps than the ToT-style search finds.

## 3. ReAct (Yao et al., 2022): Reason, Act, Observe, repeat

Every prompting technique so far reasons purely from what is already written in the context window. But plenty of real questions need information that **is not in the prompt and cannot be reasoned into existence** — today's date, a database lookup, a live API result, an arithmetic result too large to trust a language model's mental math on. ReAct addresses this with an explicit loop:

```
Thought: <what do I need to find out, and why>
Action: <call one specific tool with specific arguments>
Observation: <the tool's real return value, inserted into the context>
Thought: <given that observation, what's next>
...
Final Answer: <once nothing more is needed>
```

Each `Observation` is genuinely new information the model did not have access to before calling the tool — it comes from executing real code (a search API, a calculator, a database query) and feeding the result back into the context for the *next* Thought to condition on. This is the same "interleave reasoning with real actions" pattern that underlies every modern tool-using LLM agent.

## 4. `example.py` Part B: a scripted ReAct loop with real tool calls

There is no language model in this file's ReAct section — the "Thought" strings are scripted Python, written out explicitly so the *interaction pattern* is fully visible. What is genuinely real, and not scripted, is the tool execution: `lookup_capital`, `lookup_population`, and `calculator` are real Python functions that return real values the calling code did not know in advance. The demo task — "what is the population of a country's capital, divided by 1000, rounded?" — provably cannot be answered from the question text alone; it requires two chained lookups plus a real arithmetic computation, and the script prints every Thought/Action/Observation step plus an independent check that the agent's final answer matches directly computing the same quantity. A second run against a country with no lookup entry shows the loop terminating cleanly on a real "I can't proceed" observation, rather than hallucinating an answer.

## Video Script Outline

1. Motivation — one reasoning chain vs. a searchable tree of chains; reasoning-only vs. reasoning-that-can-touch-the-world
2. Tree-of-Thought (Yao et al. 2023): generate, evaluate, prune, repeat — the classical-search analogy
3. Why greedy single-path reasoning can get provably stuck: walk through one concrete puzzle instance
4. Walkthrough of `example.py` Part A: BFS oracle vs. ToT-style pruned search vs. naive greedy, live numbers from the 200-instance batch run
5. ReAct (Yao et al. 2022): the Thought / Action / Observation loop, and why some answers require real tool calls
6. Walkthrough of `example.py` Part B: the scripted agent's full trace, two chained tool calls, and the failure-path run
7. Recap: search breadth (ToT) and real-world grounding (ReAct) as two independent upgrades over plain Chain-of-Thought
8. Preview: how a model's tool-call *requests* get formatted, parsed, and validated reliably in production (Lesson 5)

## Further Reading

- Yao et al. (2023), *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (revisited from [Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md))
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (revisited from [Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md); another way of exploring multiple reasoning paths)
- Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools* (a model learning ReAct-style tool use directly, rather than being scripted or prompted into it)
