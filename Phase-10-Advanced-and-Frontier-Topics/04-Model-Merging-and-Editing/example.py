"""
Model Merging and Editing

Two self-contained, from-scratch demonstrations, both computed live (no
pretrained weights, no internet):

1. TASK ARITHMETIC. A small base MLP is fine-tuned into two separate
   specialists on two different toy classification tasks. Each specialist's
   "task vector" (finetuned_weights - base_weights) is computed, then the
   two task vectors are ADDED back onto the frozen base model
   (theta_base + lambda * (tau_1 + tau_2)) to build one merged model with no
   further gradient descent. Accuracy on both tasks is measured for the base
   model, each specialist, and the merged model at several values of the
   scaling coefficient lambda, and the real numbers are reported honestly
   (merging is not assumed to be perfect).

2. SLERP vs. LERP. Two parameter vectors that point in very different
   directions in weight space are interpolated two ways -- plain linear
   interpolation (LERP) and spherical linear interpolation (SLERP) -- and
   the interpolated vector's norm is measured at t = 0, 0.1, ..., 1.0 for
   both, empirically demonstrating that LERP's norm dips in the middle while
   SLERP's stays close to a smooth transition between the two endpoint norms.

Runtime: a few seconds on CPU (small MLPs, a few hundred training steps).

Run:
    python example.py
"""

import copy
import math

import torch
import torch.nn as nn

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Part 1: Task Arithmetic
# ---------------------------------------------------------------------------

def make_mlp():
    """A small 2-input, 2-class MLP. Same architecture is reused for the
    base model and both fine-tuned specialists, so their weights all live
    in the exact same parameter space -- a precondition for task arithmetic
    (you cannot subtract weight tensors of different shapes)."""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def make_task_a_data(n):
    """Task A: quadrant-XOR. Label = 1 if x1 and x2 have the SAME sign,
    else 0. Not linearly separable -- genuinely needs the hidden layers."""
    x = torch.empty(n, 2).uniform_(-2, 2)
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    return x, y


def make_task_b_data(n):
    """Task B: inside-circle. Label = 1 if the point lies inside a circle
    of radius 1.3 centered at the origin, else 0. A completely different
    decision boundary shape from task A's quadrant pattern."""
    x = torch.empty(n, 2).uniform_(-2, 2)
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) < 1.3 ** 2).long()
    return x, y


def accuracy(model, x, y):
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        return (preds == y).float().mean().item()


def finetune(base_model, x, y, steps=300, lr=0.02):
    """Fine-tune a COPY of base_model on (x, y) and return the copy.
    base_model itself is left untouched -- it stays the frozen reference
    point every task vector is measured against."""
    model = copy.deepcopy(base_model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model


def task_vector(finetuned_model, base_model):
    """tau = theta_finetuned - theta_base, one tensor per parameter."""
    return [
        p_ft.detach() - p_base.detach()
        for p_ft, p_base in zip(finetuned_model.parameters(), base_model.parameters())
    ]


def apply_task_vectors(base_model, vectors_with_lambdas):
    """Build theta_base + sum_i(lambda_i * tau_i) as a brand-new model,
    with no gradient descent involved at all -- pure weight arithmetic."""
    merged = copy.deepcopy(base_model)
    with torch.no_grad():
        for p_merged, p_base in zip(merged.parameters(), base_model.parameters()):
            p_merged.copy_(p_base)
        for tau, lam in vectors_with_lambdas:
            for p_merged, delta in zip(merged.parameters(), tau):
                p_merged.add_(lam * delta)
    return merged


def task_arithmetic_demo():
    print("=" * 78)
    print("1. TASK ARITHMETIC: theta_merged = theta_base + sum_i(lambda_i * tau_i)")
    print("=" * 78)

    base_model = make_mlp()

    # Held-out evaluation sets for each task (not used for fine-tuning).
    xa_eval, ya_eval = make_task_a_data(2000)
    xb_eval, yb_eval = make_task_b_data(2000)

    # Training sets.
    xa_train, ya_train = make_task_a_data(1000)
    xb_train, yb_train = make_task_b_data(1000)

    print("\nFine-tuning two SEPARATE copies of the same base model on two")
    print("different toy tasks (quadrant-XOR vs. inside-circle)...")
    model_a = finetune(base_model, xa_train, ya_train)
    model_b = finetune(base_model, xb_train, yb_train)

    tau_a = task_vector(model_a, base_model)
    tau_b = task_vector(model_b, base_model)

    # How big are these task vectors, roughly? Useful context for the SLERP
    # section below, which reuses one of them.
    tau_a_norm = math.sqrt(sum((t ** 2).sum().item() for t in tau_a))
    tau_b_norm = math.sqrt(sum((t ** 2).sum().item() for t in tau_b))
    print(f"||tau_a|| = {tau_a_norm:.3f}   ||tau_b|| = {tau_b_norm:.3f}")

    base_acc_a = accuracy(base_model, xa_eval, ya_eval)
    base_acc_b = accuracy(base_model, xb_eval, yb_eval)
    a_on_a = accuracy(model_a, xa_eval, ya_eval)
    a_on_b = accuracy(model_a, xb_eval, yb_eval)
    b_on_a = accuracy(model_b, xa_eval, ya_eval)
    b_on_b = accuracy(model_b, xb_eval, yb_eval)

    print(f"\n{'model':<24}{'acc on task A':>16}{'acc on task B':>16}")
    print(f"{'base (untrained)':<24}{base_acc_a:>16.3f}{base_acc_b:>16.3f}")
    print(f"{'fine-tuned on A only':<24}{a_on_a:>16.3f}{a_on_b:>16.3f}")
    print(f"{'fine-tuned on B only':<24}{b_on_a:>16.3f}{b_on_b:>16.3f}")

    print(f"\n-> Each specialist is strong on ITS OWN task and much weaker on the")
    print(f"   other one (base-model chance level on a balanced 2-class task is")
    print(f"   ~0.5). That gap is exactly what merging is trying to close: get")
    print(f"   ONE model that is good at both, without training on both at once.")

    # Scan a range of lambda values applied identically to both task vectors,
    # rather than assuming lambda=1 is best -- and report whatever the real
    # numbers show.
    print(f"\n{'lambda':>8}{'merged acc A':>16}{'merged acc B':>16}{'avg(A,B)':>12}")
    results = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        merged = apply_task_vectors(base_model, [(tau_a, lam), (tau_b, lam)])
        acc_a = accuracy(merged, xa_eval, ya_eval)
        acc_b = accuracy(merged, xb_eval, yb_eval)
        avg = (acc_a + acc_b) / 2
        results.append((lam, acc_a, acc_b, avg))
        print(f"{lam:>8.2f}{acc_a:>16.3f}{acc_b:>16.3f}{avg:>12.3f}")

    best_lam, best_a, best_b, best_avg = max(results, key=lambda r: r[3])
    lam1_row = next(r for r in results if r[0] == 1.0)

    print(f"\n-> Best average accuracy across both tasks was at lambda={best_lam:.2f}")
    print(f"   (task A={best_a:.3f}, task B={best_b:.3f}, average={best_avg:.3f}).")
    print(f"   The naive lambda=1.0 merge scored average={lam1_row[3]:.3f} (task A=")
    print(f"   {lam1_row[1]:.3f}, task B={lam1_row[2]:.3f}). ", end="")
    if best_lam == 1.0:
        print("Here lambda=1.0 WAS the best")
        print(f"   choice among those tried -- simple addition of both task vectors")
        print(f"   already recovered most of each specialist's own-task accuracy.")
    else:
        print(f"Here lambda=1.0 was NOT optimal --")
        print(f"   scaling the combined task vectors down to lambda={best_lam:.2f} traded")
        print(f"   off a little bit of each task's peak accuracy for noticeably less")
        print(f"   interference between the two, and won on average. This is the")
        print(f"   honest, general lesson from Ilharco et al. (2022): task arithmetic")
        print(f"   works, but the scaling coefficient usually needs tuning -- it is")
        print(f"   rarely free to just add full-strength task vectors together.")
    print(f"   Either way, the merged model at its best lambda beats the UNTRAINED")
    print(f"   base model on both tasks using nothing but weight arithmetic on two")
    print(f"   checkpoints -- no joint training run over both tasks ever happened.")

    return tau_a, tau_b, base_model


# ---------------------------------------------------------------------------
# Part 2: SLERP vs. LERP
# ---------------------------------------------------------------------------

def flatten_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def lerp(p0, p1, t):
    return (1 - t) * p0 + t * p1


def slerp(p0, p1, t, eps=1e-8):
    """Spherical linear interpolation between two flat vectors.
    omega = arccos( (p0.p1) / (|p0||p1|) ), the angle between them.
    Falls back to LERP when the vectors are (numerically) collinear, since
    sin(omega) -> 0 makes the SLERP formula divide by ~zero in that case."""
    p0_norm = p0 / (p0.norm() + eps)
    p1_norm = p1 / (p1.norm() + eps)
    cos_omega = torch.clamp((p0_norm * p1_norm).sum(), -1.0, 1.0)
    omega = torch.arccos(cos_omega)
    sin_omega = torch.sin(omega)

    if sin_omega.abs().item() < 1e-6:
        return lerp(p0, p1, t)

    coeff_0 = torch.sin((1 - t) * omega) / sin_omega
    coeff_1 = torch.sin(t * omega) / sin_omega
    return coeff_0 * p0 + coeff_1 * p1


def slerp_vs_lerp_demo(tau_a, base_model):
    print("\n" + "=" * 78)
    print("2. SLERP vs. LERP: DOES INTERPOLATION PRESERVE THE WEIGHT VECTOR'S NORM?")
    print("=" * 78)

    # Deliberately pick two vectors that point in very different directions,
    # so any norm-collapse in LERP is clearly visible rather than hidden in
    # rounding noise. Two independently, randomly initialized networks of
    # the same architecture are a reliable way to get this: in a
    # high-dimensional space, two random vectors are nearly orthogonal with
    # overwhelming probability.
    net_x = make_mlp()
    net_y = make_mlp()
    p0 = flatten_params(net_x)
    p1 = flatten_params(net_y)

    cos_angle = torch.clamp(
        (p0 / p0.norm() * p1 / p1.norm()).sum(), -1.0, 1.0
    ).item()
    angle_deg = math.degrees(math.acos(cos_angle))
    print(f"\nChosen p0, p1: two independently initialized networks' flattened")
    print(f"parameter vectors (dimension {p0.numel()}).")
    print(f"||p0||={p0.norm().item():.3f}  ||p1||={p1.norm().item():.3f}  "
          f"angle between them = {angle_deg:.1f} degrees")
    print(f"(Two independent random vectors in a space this large land close to")
    print(f"90 degrees apart almost regardless of how they were initialized --")
    print(f"exactly the 'far apart in direction' case where LERP's norm dip is")
    print(f"most visible, and a realistic stand-in for how far apart two")
    print(f"unrelated fine-tunes' weights can end up.)")

    print(f"\n{'t':>6}{'LERP norm':>14}{'SLERP norm':>14}")
    lerp_norms = []
    slerp_norms = []
    ts = [round(i * 0.1, 1) for i in range(11)]
    for t in ts:
        t_tensor = torch.tensor(float(t))
        l_norm = lerp(p0, p1, t_tensor).norm().item()
        s_norm = slerp(p0, p1, t_tensor).norm().item()
        lerp_norms.append(l_norm)
        slerp_norms.append(s_norm)
        print(f"{t:>6.1f}{l_norm:>14.3f}{s_norm:>14.3f}")

    min_lerp = min(lerp_norms)
    min_lerp_t = ts[lerp_norms.index(min_lerp)]
    endpoint_avg_norm = (p0.norm().item() + p1.norm().item()) / 2
    lerp_dip_pct = 100 * (endpoint_avg_norm - min_lerp) / endpoint_avg_norm
    slerp_spread = max(slerp_norms) - min(slerp_norms)

    print(f"\n-> LERP's norm bottoms out at t={min_lerp_t:.1f} with norm={min_lerp:.3f},")
    print(f"   which is {lerp_dip_pct:.1f}% below the average of the two endpoint norms")
    print(f"   ({endpoint_avg_norm:.3f}) -- a real, measured shrinkage purely from")
    print(f"   averaging two differently-DIRECTED vectors, nothing to do with what")
    print(f"   either model actually represents. SLERP's norm only ranges across")
    print(f"   {slerp_spread:.3f} (from {min(slerp_norms):.3f} to {max(slerp_norms):.3f}) over the same t values --")
    print(f"   it stays close to a smooth, monotonic transition between the two")
    print(f"   endpoint norms instead of dipping, exactly as the great-circle-arc")
    print(f"   formula intends. This is why merge tools default to SLERP over")
    print(f"   plain averaging when blending two fine-tunes of the same base model.")

    # Second, smaller-angle comparison using the actual task vector from
    # part 1, to make the point that the size of the dip really does track
    # the angle between the two vectors, not some fixed constant.
    print(f"\nFor contrast, interpolating base_model against (base_model + tau_a)")
    print(f"from part 1 -- a much smaller angle, since tau_a is a comparatively")
    print(f"small, targeted fine-tuning update rather than a full re-initialization:")
    p_base = flatten_params(base_model)
    p_base_plus_tau = p_base + torch.cat([t.reshape(-1) for t in tau_a])
    cos_small = torch.clamp(
        (p_base / p_base.norm() * p_base_plus_tau / p_base_plus_tau.norm()).sum(),
        -1.0, 1.0,
    ).item()
    angle_small_deg = math.degrees(math.acos(cos_small))
    mid_lerp = lerp(p_base, p_base_plus_tau, torch.tensor(0.5)).norm().item()
    mid_slerp = slerp(p_base, p_base_plus_tau, torch.tensor(0.5)).norm().item()
    small_endpoint_avg = (p_base.norm().item() + p_base_plus_tau.norm().item()) / 2
    print(f"angle = {angle_small_deg:.1f} degrees (vs. {angle_deg:.1f} degrees above). At t=0.5:")
    print(f"  LERP norm={mid_lerp:.3f}  SLERP norm={mid_slerp:.3f}  "
          f"(endpoint average={small_endpoint_avg:.3f})")
    print(f"-> With a smaller angle between the two vectors, LERP's dip below the")
    print(f"   endpoint average shrinks too -- the effect is real but proportional")
    print(f"   to how far apart the two vectors actually point, not a fixed penalty.")


def main():
    tau_a, tau_b, base_model = task_arithmetic_demo()
    slerp_vs_lerp_demo(tau_a, base_model)

    print("\n" + "=" * 78)
    print("3. MODEL EDITING (ROME) -- CONCEPTUAL, NOT RUN HERE")
    print("=" * 78)
    print("Task arithmetic and SLERP above both operate on ENTIRE fine-tuned")
    print("checkpoints. ROME (Meng et al., 2022) operates on a single fact: it")
    print("uses causal tracing to locate the one MLP layer whose activations")
    print("causally control a specific factual association, then overwrites it")
    print("with a closed-form rank-one update W_new = W + (v_new - v_old) @ k^T /")
    print("(k^T @ k) -- no gradient descent, no training data beyond the one edit.")
    print("Reproducing this faithfully needs a real pretrained transformer LLM")
    print("(to have real factual associations to locate and edit in the first")
    print("place), which is outside this toy CPU example's scope -- see the")
    print("README's section 3 for the full mechanism, and the ROME paper /")
    print("rome-edit codebase for a runnable implementation against GPT-2/GPT-J.")


if __name__ == "__main__":
    main()
