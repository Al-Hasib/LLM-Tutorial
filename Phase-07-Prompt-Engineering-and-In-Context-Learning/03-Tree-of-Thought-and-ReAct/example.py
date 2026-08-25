"""
Tree-of-Thought and ReAct

Two independent, fully self-contained demonstrations:

PART A -- Tree-of-Thought (Yao et al., 2023) as REAL search, not a metaphor.
Toy puzzle: starting from an integer `start`, reach an integer `target` in
as few steps as possible, where each step applies one operation from a
fixed set: +3, -1, *2. We implement three genuinely different search
strategies over the same problem and compare them on identical instances:

  1. Exhaustive BFS with state deduplication -- the ground-truth oracle for
     the true optimal (fewest-steps) solution length.
  2. Tree-of-Thought-style BEST-FIRST SEARCH WITH HEURISTIC PRUNING: at each
     depth, generate every child of the current frontier (branching factor
     3, so an UNPRUNED tree would grow as 3^depth -- real branch explosion),
     score each candidate with a heuristic (distance to target), and keep
     only the best `beam_width` candidates before continuing. This is
     exactly Yao et al.'s recipe: generate multiple "thoughts," have an
     evaluator judge them, keep the promising ones, discard the rest.
  3. NAIVE GREEDY single-path search: no branching at all, always take the
     single locally-best next step, never backtrack. We show empirically,
     across a batch of random puzzle instances, how often this gets stuck
     in a revisit loop or lands on a strictly worse (non-optimal) solution
     than both BFS and the pruned tree search find.

PART B -- ReAct (Yao et al., 2022): a small SCRIPTED agent loop (not a
trained model -- there is no LLM here) that interleaves Thought / Action /
Observation steps to solve a toy multi-step question that genuinely
requires calling external tools (a lookup table and a calculator) to
answer, printing the full trace at every step.

Runtime: well under 5 seconds on CPU (pure Python search + scripted logic,
no model training).

Run:
    python example.py
"""

import random
from collections import deque

random.seed(0)

# ===========================================================================
# PART A: TREE-OF-THOUGHT AS SEARCH
# ===========================================================================

OPS = [
    ("+3", lambda v: v + 3),
    ("-1", lambda v: v - 1),
    ("*2", lambda v: v * 2),
]
VALUE_BOUND = 500          # states outside [-VALUE_BOUND, VALUE_BOUND] are pruned as "too far afield"
MAX_DEPTH = 14


def in_bounds(v):
    return -VALUE_BOUND <= v <= VALUE_BOUND


def heuristic(value, target):
    """Simple, cheap evaluator: how far is this state from the target?
    (Not claimed to be admissible -- exactly the kind of rough, learned-or-
    hand-written evaluator Tree-of-Thought uses to JUDGE candidate thoughts,
    not a guarantee of optimality.)"""
    return abs(value - target)


def bfs_optimal(start, target, max_depth=MAX_DEPTH):
    """Ground-truth oracle: exhaustive breadth-first search over the
    DEDUPLICATED state graph. Because every edge has the same cost (one
    step), the first time BFS reaches a state is via a shortest path --
    this gives the true optimal step count to compare everything else against."""
    if start == target:
        return [], 0
    visited = {start}
    frontier = deque([(start, [])])
    nodes_visited = 0
    while frontier:
        value, path = frontier.popleft()
        for op_name, op_fn in OPS:
            new_value = op_fn(value)
            nodes_visited += 1
            if new_value == target:
                return path + [op_name], nodes_visited
            if in_bounds(new_value) and new_value not in visited and len(path) + 1 < max_depth:
                visited.add(new_value)
                frontier.append((new_value, path + [op_name]))
    return None, nodes_visited


def tot_beam_search(start, target, beam_width, max_depth=MAX_DEPTH):
    """Tree-of-Thought-style search: at every depth, expand ALL children of
    the current frontier (branching factor 3 -- this is the raw, unpruned
    tree), then keep only the `beam_width` children with the best (lowest)
    heuristic score before moving to the next depth. Returns the found path,
    the number of nodes GENERATED (i.e. how big the raw unpruned tree would
    have been at that point), and the number of nodes actually KEPT/explored
    after pruning."""
    if start == target:
        return [], 0, 0
    frontier = [(heuristic(start, target), start, [])]
    total_generated = 0
    total_kept = 1
    for depth in range(max_depth):
        candidates = []
        for _, value, path in frontier:
            for op_name, op_fn in OPS:
                new_value = op_fn(value)
                total_generated += 1
                if new_value == target:
                    return path + [op_name], total_generated, total_kept
                if in_bounds(new_value):
                    candidates.append((heuristic(new_value, target), new_value, path + [op_name]))
        if not candidates:
            return None, total_generated, total_kept
        candidates.sort(key=lambda c: c[0])
        frontier = candidates[:beam_width]
        total_kept += len(frontier)
    return None, total_generated, total_kept


def greedy_single_path(start, target, max_steps=MAX_DEPTH):
    """Naive greedy: ONE path, no branching, no backtracking. At each step,
    take whichever single operation most reduces the heuristic distance to
    the target (fixed tie-break order: +3, -1, *2). If the chosen next state
    was already visited earlier on this same path, greedy is stuck in a
    loop it cannot escape (it has no mechanism to try anything else) and we
    report failure rather than spin forever."""
    value = start
    path = []
    visited_on_path = {start}
    for _ in range(max_steps):
        if value == target:
            return path, "success"
        best = min(
            ((heuristic(op_fn(value), target), op_name, op_fn(value)) for op_name, op_fn in OPS),
            key=lambda c: c[0],
        )
        _, op_name, new_value = best
        if new_value == target:
            return path + [op_name], "success"
        if new_value in visited_on_path or not in_bounds(new_value):
            return path, "stuck (revisited a state / left bounds, no backtracking available)"
        visited_on_path.add(new_value)
        path.append(op_name)
        value = new_value
    return path, "gave up (exceeded max steps without reaching target)"


def part_a_demo():
    print("=" * 78)
    print("PART A: TREE-OF-THOUGHT AS REAL SEARCH")
    print("=" * 78)
    print(f"Puzzle: reach `target` from `start` using ops {[o[0] for o in OPS]}, fewest steps.")
    print(f"State bound: |value| <= {VALUE_BOUND}. Max depth: {MAX_DEPTH}.\n")

    print("-" * 78)
    print("One concrete instance, all three strategies side by side")
    print("-" * 78)
    start, target = 0, 35
    opt_path, bfs_nodes = bfs_optimal(start, target)
    beam_path, beam_generated, beam_kept = tot_beam_search(start, target, beam_width=3)
    greedy_path, greedy_status = greedy_single_path(start, target)
    print(f"start={start}, target={target}")
    print(f"  BFS oracle (exhaustive, deduplicated): {len(opt_path)} steps  {opt_path}")
    print(f"    nodes visited: {bfs_nodes}")
    print(f"  ToT beam search (beam_width=3):        {len(beam_path)} steps  {beam_path}")
    print(f"    raw tree nodes generated: {beam_generated}   nodes kept after pruning: {beam_kept}")
    unpruned_tree_size = sum(3 ** d for d in range(1, len(beam_path) + 1))
    print(f"    (an UNPRUNED tree exploring every branch to this depth would have")
    print(f"     generated {unpruned_tree_size} nodes -- pruning kept only {beam_kept}.)")
    print(f"  Naive greedy (single path, no backtracking): status = {greedy_status}")
    print(f"    path so far: {greedy_path}")

    print("\n" + "-" * 78)
    print("Batch comparison across 200 random puzzle instances")
    print("-" * 78)
    print("(start in [0,15], target in [-10,60], fixed random seed)\n")

    num_instances = 200
    greedy_fail_count = 0
    greedy_suboptimal_count = 0
    beam_matches_optimal_count = 0
    total_bfs_nodes = 0
    total_beam_generated = 0
    total_beam_kept = 0
    first_failure_example = None

    for _ in range(num_instances):
        s = random.randint(0, 15)
        t = random.randint(-10, 60)
        opt_path, bfs_nodes = bfs_optimal(s, t)
        if opt_path is None:
            continue  # skip the rare instance outside our search bound/depth entirely
        beam_path, beam_generated, beam_kept = tot_beam_search(s, t, beam_width=3)
        g_path, g_status = greedy_single_path(s, t)

        total_bfs_nodes += bfs_nodes
        total_beam_generated += beam_generated
        total_beam_kept += beam_kept

        if beam_path is not None and len(beam_path) == len(opt_path):
            beam_matches_optimal_count += 1

        if g_status != "success":
            greedy_fail_count += 1
            if first_failure_example is None:
                first_failure_example = (s, t, opt_path, g_path, g_status)
        elif len(g_path) > len(opt_path):
            greedy_suboptimal_count += 1
            if first_failure_example is None:
                first_failure_example = (s, t, opt_path, g_path, "succeeded but suboptimal")

    print(f"{'strategy':40s}{'result':>36}")
    print(f"{'BFS oracle: always optimal':40s}{'100% (by construction)':>36}")
    print(f"{'ToT beam search matches optimal length':40s}{beam_matches_optimal_count}/{num_instances:>4}"
          f" = {100*beam_matches_optimal_count/num_instances:.1f}%")
    fails_or_worse = greedy_fail_count + greedy_suboptimal_count
    print(f"{'Greedy fails OR is strictly suboptimal':40s}{fails_or_worse}/{num_instances:>4}"
          f" = {100*fails_or_worse/num_instances:.1f}%")
    print(f"  (of which outright stuck/gave up: {greedy_fail_count}, "
          f"succeeded but took more steps than optimal: {greedy_suboptimal_count})")

    print(f"\nAverage nodes examined per instance:")
    print(f"  BFS oracle (exhaustive):        {total_bfs_nodes/num_instances:.1f}")
    print(f"  ToT beam search -- generated:   {total_beam_generated/num_instances:.1f}"
          f"  (raw tree nodes proposed, pre-pruning)")
    print(f"  ToT beam search -- kept:        {total_beam_kept/num_instances:.1f}"
          f"  (nodes actually carried forward after pruning)")
    print(f"  Greedy (single path):           1 node examined per step, by definition")

    if first_failure_example:
        s, t, opt_path, g_path, g_status = first_failure_example
        print(f"\nConcrete failure case: start={s}, target={t}")
        print(f"  Optimal (BFS):  {len(opt_path)} steps -- {opt_path}")
        print(f"  Greedy result:  {g_status}, path so far {g_path} ({len(g_path)} steps)")

    print(f"\n-> Across {num_instances} random instances, ToT-style beam search matches the")
    print(f"   true optimal path length {100*beam_matches_optimal_count/num_instances:.1f}% of the time while examining only")
    print(f"   {total_beam_kept/num_instances:.1f} nodes on average per instance (vs {total_bfs_nodes/num_instances:.1f} for exhaustive BFS) --")
    print(f"   pruning the raw {total_beam_generated/num_instances:.1f}-node branching tree down to a fraction of its size.")
    print(f"   Naive greedy, with no ability to branch or backtrack, fails outright or")
    print(f"   ends up with a worse-than-optimal solution in {100*fails_or_worse/num_instances:.1f}% of instances -- exactly the")
    print("   failure mode Tree-of-Thought is designed to avoid: a single locally-best")
    print("   choice at one step can commit you to a state with no good continuation,")
    print("   whereas keeping several candidate branches alive (even a small beam)")
    print("   lets the search recover from a locally-tempting but globally bad move.")


# ===========================================================================
# PART B: REACT -- REASONING + ACTING, INTERLEAVED
# ===========================================================================

# A tiny "world" the toy agent can query -- stands in for real tools/APIs.
CAPITAL_LOOKUP = {
    "France": "Paris",
    "Japan": "Tokyo",
}
POPULATION_LOOKUP = {   # approximate, for demo purposes only
    "Paris": 2_148_000,
    "Tokyo": 13_960_000,
}


def tool_lookup_capital(country):
    return CAPITAL_LOOKUP.get(country, "UNKNOWN")


def tool_lookup_population(city):
    return POPULATION_LOOKUP.get(city, None)


def tool_calculator(expression):
    """A real calculator tool -- evaluates a restricted arithmetic
    expression string. This is a genuine function call with a genuine
    return value, not a fake API stub."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError(f"Unsupported characters in expression: {expression!r}")
    return eval(expression, {"__builtins__": {}}, {})


def react_agent(country):
    """A SCRIPTED ReAct-style loop (Yao et al., 2022): the agent's Thought
    steps are fixed here (there is no LLM generating them -- this file has
    no model in Part B at all), but the STRUCTURE -- alternating Thought,
    Action (a real tool call), and Observation (the tool's real return
    value), feeding each Observation into the next Thought -- is exactly
    the ReAct interaction pattern, applied to a task that genuinely
    requires two chained tool calls plus arithmetic to answer."""
    print(f"\nTask: \"What is the population of {country}'s capital city, divided by 1000,")
    print("       rounded to the nearest whole number?\"\n")

    print("Thought 1: I don't know the capital of this country. I need to look it up.")
    print(f"Action 1: lookup_capital(country={country!r})")
    capital = tool_lookup_capital(country)
    print(f"Observation 1: {capital!r}")
    if capital == "UNKNOWN":
        print("Thought: the lookup tool has no entry for this country. Stopping.")
        return None

    print(f"\nThought 2: Now I know the capital is {capital}. I need its population.")
    print(f"Action 2: lookup_population(city={capital!r})")
    population = tool_lookup_population(capital)
    print(f"Observation 2: {population!r}")
    if population is None:
        print("Thought: the lookup tool has no population entry for this city. Stopping.")
        return None

    print(f"\nThought 3: I have the population ({population}). The question asks for it")
    print("           divided by 1000 and rounded. I should use the calculator, not")
    print("           do the arithmetic myself, to guarantee the result is exact.")
    expr = f"{population} / 1000"
    print(f"Action 3: calculator(expression={expr!r})")
    raw_result = tool_calculator(expr)
    print(f"Observation 3: {raw_result!r}")

    final_answer = round(raw_result)
    print(f"\nThought 4: {raw_result} rounds to {final_answer}. I have everything needed to answer.")
    print(f"Final Answer: {final_answer}")
    return final_answer


def part_b_demo():
    print("\n" + "=" * 78)
    print("PART B: REACT -- A SCRIPTED REASON+ACT+OBSERVE LOOP")
    print("=" * 78)
    print("This agent's 'thoughts' are scripted Python strings (there is no language")
    print("model anywhere in this section) -- what's being demonstrated is the REACT")
    print("interaction PATTERN itself: Thought -> Action (real tool call) -> Observation")
    print("-> next Thought, chained until the task is answered, with each tool call")
    print("returning a REAL value that genuinely changes what happens next.\n")

    answer = react_agent("France")
    expected = round(POPULATION_LOOKUP["Paris"] / 1000)
    print(f"\n-> Verification: population of Paris ({POPULATION_LOOKUP['Paris']}) / 1000, rounded,")
    print(f"   is independently computed as {expected}. The agent's final answer ({answer}) matches:")
    print(f"   {answer == expected}. Two real, distinct tool calls (a lookup, then a")
    print("   calculator call) were required and executed to reach it -- neither tool's")
    print("   result was known to the agent in advance, exactly the point of Acting")
    print("   (gathering real information) interleaved with Reasoning (deciding what")
    print("   to do with it), rather than trying to answer from the question text alone.")

    print("\nRunning the same agent on a country with no lookup data on record, to show")
    print("the loop stopping cleanly on a real 'observation says I can't proceed' case:")
    react_agent("Germany")


def main():
    part_a_demo()
    part_b_demo()


if __name__ == "__main__":
    main()
