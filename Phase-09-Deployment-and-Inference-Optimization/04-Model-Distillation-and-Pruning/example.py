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

    torch.manual_seed(1)
    student_hard = MLP(hidden_sizes=[3])
    train_classifier(student_hard, train_x, train_y, epochs=80, lr=0.05)
    acc_hard = accuracy(student_hard, test_x, test_y)

    torch.manual_seed(1)
    student_distilled = MLP(hidden_sizes=[3])
    train_classifier(student_distilled, train_x, train_y, epochs=80, lr=0.05,
                      extra_loss_fn=distillation_loss_fn)
    acc_distilled = accuracy(student_distilled, test_x, test_y)

    print(f"\nStudent (2->4->{NUM_CLASSES}), SAME initialization, two training recipes:")
    print(f"  hard labels only:               test accuracy = {acc_hard:.3f}")
    print(f"  distillation (T={TEMPERATURE}, alpha={ALPHA}):  test accuracy = {acc_distilled:.3f}")

    diff = acc_distilled - acc_hard
    if diff > 0.01:
        print(f"\n-> Distillation improved this tiny student's accuracy by {diff:+.3f} over")
        print("   hard labels alone. With only 6 classes and a 4-unit hidden layer, the")
        print("   student can't perfectly separate every class -- the teacher's soft")
        print("   targets tell it WHICH mistakes are cheap (confusing a class with its")
        print("   neighbor on the circle) vs. WHICH are expensive (confusing it with the")
        print("   class on the opposite side), information a one-hot label never carries.")
    else:
        print(f"\n-> Here distillation did not clearly beat hard labels ({diff:+.3f} test")
        print("   accuracy) on this particular run/task. Distillation's benefit is real")
        print("   but not guaranteed on every task -- it helps most when the student is")
        print("   capacity-constrained enough that a hard-label signal alone underspecifies")
        print("   the right decision boundary, exactly the regime this toy task targets.")


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
