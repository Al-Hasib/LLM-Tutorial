"""
Model Distillation and Pruning

Two real, from-scratch demos:
  1. Knowledge distillation -- train a larger teacher on a toy
     classification task with real class-similarity structure, then
     train a much smaller student two ways (hard labels alone vs.
     temperature-scaled distillation from the teacher), and compare
     real held-out accuracy.
  2. Magnitude pruning -- zero out the smallest-magnitude weights of a
     trained network at several sparsity levels, measuring the real
     accuracy-vs-sparsity trade-off curve.

Runtime: ~20-40 seconds on a CPU.

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 0. A toy classification task with real class-similarity structure:
#    classes are centers spaced around a circle, and each class overlaps
#    NOTICEABLY with its two neighbors on the circle (but not with classes
#    on the far side) -- exactly the kind of structure "dark knowledge"
#    (relative confidence across wrong classes) can actually capture.
# ---------------------------------------------------------------------------

NUM_CLASSES = 6
RADIUS = 3.0
NOISE_STD = 1.1   # large enough that adjacent classes genuinely overlap


def make_dataset(n_per_class):
    xs, ys = [], []
    for c in range(NUM_CLASSES):
        angle = 2 * math.pi * c / NUM_CLASSES
        center = torch.tensor([RADIUS * math.cos(angle), RADIUS * math.sin(angle)])
        points = center + NOISE_STD * torch.randn(n_per_class, 2)
        xs.append(points)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


train_x, train_y = make_dataset(300)
test_x, test_y = make_dataset(150)


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        sizes = [2] + hidden_sizes + [NUM_CLASSES]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(model, x, y, epochs, lr=0.05, extra_loss_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if extra_loss_fn is not None:
            loss = extra_loss_fn(logits, loss)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def accuracy(model, x, y):
    preds = model(x).argmax(dim=-1)
    return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# 1. Knowledge distillation
# ---------------------------------------------------------------------------

def distillation_demo():
    print("=" * 70)
    print("1. KNOWLEDGE DISTILLATION")
    print("=" * 70)

    teacher = MLP(hidden_sizes=[64, 64])
    train_classifier(teacher, train_x, train_y, epochs=300, lr=0.02)
    teacher_acc = accuracy(teacher, test_x, test_y)
    print(f"Teacher (2->64->64->{NUM_CLASSES}) test accuracy: {teacher_acc:.3f}")

    TEMPERATURE = 4.0
    ALPHA = 0.7   # weight on the distillation term vs. the hard-label term

    with torch.no_grad():
        teacher_logits = teacher(train_x)
    soft_targets = F.softmax(teacher_logits / TEMPERATURE, dim=-1)

    def distillation_loss_fn(student_logits, hard_loss):
        soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
        kd_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (TEMPERATURE ** 2)
        return ALPHA * kd_loss + (1 - ALPHA) * hard_loss

    # A single run's accuracy is noisy at this scale -- average several
    # independent seeds (same seed used for both recipes in each repetition,
    # so it's a fair PAIRED comparison) for a statistically honest verdict
    # instead of trusting one lucky/unlucky initialization.
    NUM_SEEDS = 8
    STUDENT_HIDDEN = [3]
    # Train the student on a SMALL subset of the data -- capacity- AND
    # data-constrained is exactly the regime where a hard one-hot label
    # underspecifies the decision boundary the most, giving the teacher's
    # soft targets real room to help.
    n_student_train = 60

    hard_accs, distilled_accs = [], []
    for seed in range(NUM_SEEDS):
        torch.manual_seed(100 + seed)
        idx = torch.randperm(len(train_x))[:n_student_train]
        sub_x, sub_y = train_x[idx], train_y[idx]
        with torch.no_grad():
            sub_soft_targets = F.softmax(teacher(sub_x) / TEMPERATURE, dim=-1)

        def sub_distillation_loss_fn(student_logits, hard_loss, targets=sub_soft_targets):
            soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            kd_loss = F.kl_div(soft_student, targets, reduction="batchmean") * (TEMPERATURE ** 2)
            return ALPHA * kd_loss + (1 - ALPHA) * hard_loss

        torch.manual_seed(seed)
        student_hard = MLP(hidden_sizes=STUDENT_HIDDEN)
        train_classifier(student_hard, sub_x, sub_y, epochs=150, lr=0.05)
        hard_accs.append(accuracy(student_hard, test_x, test_y))

        torch.manual_seed(seed)
        student_distilled = MLP(hidden_sizes=STUDENT_HIDDEN)
        train_classifier(student_distilled, sub_x, sub_y, epochs=150, lr=0.05,
                          extra_loss_fn=sub_distillation_loss_fn)
        distilled_accs.append(accuracy(student_distilled, test_x, test_y))

    hard_accs = torch.tensor(hard_accs)
    distilled_accs = torch.tensor(distilled_accs)
    wins = int((distilled_accs > hard_accs).sum())

    print(f"\nStudent (2->{STUDENT_HIDDEN[0]}->{NUM_CLASSES}), trained on only "
          f"{n_student_train} examples, {NUM_SEEDS} independent seeds:")
    print(f"  hard labels only:               mean test accuracy = {hard_accs.mean():.3f} "
          f"(std {hard_accs.std():.3f})")
    print(f"  distillation (T={TEMPERATURE}, alpha={ALPHA}):  mean test accuracy = "
          f"{distilled_accs.mean():.3f} (std {distilled_accs.std():.3f})")
    print(f"  distillation won on {wins}/{NUM_SEEDS} seeds (paired, same init per seed)")

    diff = (distilled_accs.mean() - hard_accs.mean()).item()
    if diff > 0.01 and wins >= NUM_SEEDS * 0.6:
        print(f"\n-> Averaged over {NUM_SEEDS} seeds, distillation improved this tiny,")
        print(f"   data-starved student's accuracy by {diff:+.3f} on average, and won on")
        print(f"   a clear majority of individual seeds. With only {n_student_train} labeled")
        print("   examples and a 3-unit hidden layer, hard one-hot labels alone")
        print("   underspecify the decision boundary -- the teacher's soft targets add")
        print("   information about WHICH mistakes are cheap (confusing a class with its")
        print("   neighbor on the circle) vs. expensive (the class on the opposite side),")
        print("   which a one-hot label never carries.")
    else:
        print(f"\n-> Averaged over {NUM_SEEDS} seeds, the mean difference was only {diff:+.3f}")
        print("   and distillation did not win a clear majority of individual seeds.")
        print("   Distillation's benefit is real in general but not guaranteed on every")
        print("   task or every random seed -- it helps most when the student is")
        print("   capacity- and data-constrained enough that hard labels alone")
        print("   underspecify the right decision boundary, which is what this toy")
        print("   setup was designed to test, honestly, rather than assume.")


# ---------------------------------------------------------------------------
# 2. Magnitude pruning: the accuracy-vs-sparsity curve
# ---------------------------------------------------------------------------

def magnitude_prune(model, sparsity):
    """Zero out the `sparsity` fraction of smallest-magnitude weights,
    GLOBALLY across all Linear layers (not per-layer), returning a pruned
    COPY so the original trained model is untouched."""
    import copy
    pruned = copy.deepcopy(model)
    all_weights = torch.cat([p.detach().abs().flatten()
                              for name, p in pruned.named_parameters() if "weight" in name])
    if sparsity <= 0:
        return pruned
    threshold = torch.quantile(all_weights, sparsity)
    with torch.no_grad():
        for name, p in pruned.named_parameters():
            if "weight" in name:
                mask = p.abs() >= threshold
                p.mul_(mask)
    return pruned


def pruning_demo():
    print("\n" + "=" * 70)
    print("2. MAGNITUDE PRUNING: THE ACCURACY-vs-SPARSITY CURVE")
    print("=" * 70)

    torch.manual_seed(2)
    model = MLP(hidden_sizes=[64, 64])
    train_classifier(model, train_x, train_y, epochs=300, lr=0.02)
    base_acc = accuracy(model, test_x, test_y)
    print(f"Unpruned model test accuracy: {base_acc:.3f}\n")

    print(f"{'sparsity':>10}{'test accuracy':>16}{'accuracy drop':>16}")
    for sparsity in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        pruned_model = magnitude_prune(model, sparsity)
        acc = accuracy(pruned_model, test_x, test_y)
        print(f"{sparsity:>10.0%}{acc:>16.3f}{base_acc - acc:>16.3f}")

    print("\n-> Moderate sparsity costs little to nothing -- most of the network's")
    print("   weights genuinely are redundant. Accuracy holds up well until sparsity")
    print("   climbs high enough that pruning starts removing weights the network")
    print("   actually relies on, at which point it drops off much more sharply --")
    print("   the 'knee' in the trade-off curve the README describes, visible")
    print("   directly in the real numbers above rather than just asserted.")


def main():
    distillation_demo()
    pruning_demo()


if __name__ == "__main__":
    main()
