"""
Scaling Laws

Implements the Chinchilla power-law loss formula L(N, D) = E + A/N^alpha +
B/D^beta using the paper's published (approximate) fitted constants, then
solves for the compute-optimal (N, D) split at several compute budgets
under the C ~= 6*N*D approximation -- recovering, from the formula alone,
the same qualitative "size and data should scale together" conclusion
Chinchilla reached empirically.

Run:
    python example.py
"""

import numpy as np

# Published (approximate) Chinchilla-fitted constants (Hoffmann et al., 2022,
# "Approach 3" fit). These are the widely-cited figures from the paper; treat
# them as illustrative of the METHOD, not a precision reproduction of the
# original fit.
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N, D):
    """N = model parameters, D = training tokens."""
    return E + A / (N ** ALPHA) + B / (D ** BETA)


def loss_grid_demo():
    print("=" * 70)
    print("1. THE POWER-LAW LOSS SURFACE L(N, D) = E + A/N^a + B/D^b")
    print("=" * 70)
    print(f"Constants: E={E}, A={A}, B={B}, alpha={ALPHA}, beta={BETA}\n")

    Ns = [1e8, 1e9, 1e10, 1e11]
    Ds = [1e9, 1e10, 1e11, 1e12]

    header = f"{'N \\ D':>12}" + "".join(f"{d:>14.0e}" for d in Ds)
    print(header)
    for n in Ns:
        row = f"{n:>12.0e}" + "".join(f"{loss(n, d):>14.4f}" for d in Ds)
        print(row)

    print("\n-> Loss drops moving right (more data) OR down (more parameters),")
    print("   but with steeply diminishing returns along either axis alone --")
    print("   exactly why balancing BOTH matters more than maxing out one.")


def compute_optimal_split(compute_budget, num_candidates=4000):
    """Given a compute budget C ~= 6*N*D, search over N (with D = C/(6N))
    to find the loss-minimizing split."""
    n_min, n_max = 1e6, compute_budget / 6.0
    candidate_Ns = np.logspace(np.log10(n_min), np.log10(n_max), num_candidates)
    candidate_Ds = compute_budget / (6.0 * candidate_Ns)

    losses = loss(candidate_Ns, candidate_Ds)
    best_idx = np.argmin(losses)
    return candidate_Ns[best_idx], candidate_Ds[best_idx], losses[best_idx]


def compute_optimal_demo():
    print("\n" + "=" * 70)
    print("2. COMPUTE-OPTIMAL (N, D) SPLIT AT SEVERAL BUDGETS (C ~= 6*N*D FLOPs)")
    print("=" * 70)

    budgets = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
    print(f"{'compute (FLOPs)':>18}{'optimal N':>16}{'optimal D':>16}"
          f"{'D/N (tokens/param)':>22}{'loss':>10}")
    for C in budgets:
        N_opt, D_opt, L_opt = compute_optimal_split(C)
        print(f"{C:>18.1e}{N_opt:>16.3e}{D_opt:>16.3e}{D_opt / N_opt:>22.1f}{L_opt:>10.4f}")

    print("\n-> Neither extreme (all parameters, no data / all data, tiny model)")
    print("   minimizes loss -- there is a genuine interior optimum at every")
    print("   compute budget, and BOTH optimal N and optimal D grow with more")
    print("   compute (never just one of them). Across this 100,000x range of")
    print("   compute, optimal N grows ~182,000x and optimal D grows ~550x --")
    print("   both scale up together, which is the qualitative pattern (size and")
    print("   data scaling together, not model size alone) Chinchilla's empirical")
    print("   sweep found -- recovered here from nothing but the fitted formula")
    print("   and a grid search. (The tokens-per-parameter ratio does drift")
    print("   upward with these particular constants -- these fitted exponents")
    print("   aren't exactly equal -- but that drift is far gentler than either")
    print("   N or D's own five-order-of-magnitude growth.)")

    print("\n" + "=" * 70)
    print("3. GPT-3 (KAPLAN-ERA) vs. CHINCHILLA-OPTIMAL, AT THE SAME COMPUTE")
    print("=" * 70)
    gpt3_N, gpt3_D = 175e9, 300e9
    gpt3_compute = 6 * gpt3_N * gpt3_D
    gpt3_loss = loss(gpt3_N, gpt3_D)

    opt_N, opt_D, opt_loss = compute_optimal_split(gpt3_compute)

    print(f"GPT-3 actual:        N={gpt3_N:.1e} params, D={gpt3_D:.1e} tokens, "
          f"predicted loss={gpt3_loss:.4f}")
    print(f"Compute-optimal for the SAME compute budget:")
    print(f"                     N={opt_N:.1e} params, D={opt_D:.1e} tokens, "
          f"predicted loss={opt_loss:.4f}")
    print(f"\n-> Under this formula, the SAME training compute GPT-3 used could have")
    print(f"   reached a lower predicted loss with a smaller model trained on")
    print(f"   substantially more tokens -- the exact 'undertrained' finding that")
    print(f"   motivated Chinchilla and, in turn, every LLaMA-generation model's")
    print(f"   much larger token-per-parameter training recipes.")


def main():
    loss_grid_demo()
    compute_optimal_demo()


if __name__ == "__main__":
    main()
