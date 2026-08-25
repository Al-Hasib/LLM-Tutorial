"""
PyTorch Fundamentals

Tensors, autograd, nn.Module, loss/optimizer, and the standard training
loop -- rebuilding the exact same XOR network from
`02-Neural-Networks-Basics/example.py`, but letting PyTorch's autograd
compute the backward pass instead of doing it by hand.

Requires: torch (pip install torch)

Run:
    python example.py
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Tensor basics
# ---------------------------------------------------------------------------

def tensor_basics_demo():
    print("=" * 70)
    print("1. TENSOR BASICS")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"x = {x}, shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")

    W = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])   # shape (2, 3)
    y = W @ x                                                # matrix-vector product
    print(f"W shape={tuple(W.shape)} @ x shape={tuple(x.shape)} -> y = {y}")

    # Device-agnostic pattern used everywhere in this course.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")


# ---------------------------------------------------------------------------
# 2. Autograd: automatic differentiation vs. the manual math from Lesson 1
# ---------------------------------------------------------------------------

def autograd_demo():
    print("\n" + "=" * 70)
    print("2. AUTOGRAD")
    print("=" * 70)

    # f(x) = x^2 -> f'(x) = 2x, same function used in the math refresher.
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"x = {x.item()}, y = x^2 = {y.item()}")
    print(f"autograd x.grad = {x.grad.item()}  (analytic 2x = {2 * x.item()})")

    # Chain rule: y = sin(x^2), same composed function as in the math refresher.
    x2 = torch.tensor(1.5, requires_grad=True)
    y2 = torch.sin(x2 ** 2)
    y2.backward()
    analytic = torch.cos(x2.detach() ** 2) * (2 * x2.detach())
    print(f"\nd/dx sin(x^2) at x=1.5 -> autograd = {x2.grad.item():.6f}, "
          f"analytic = {analytic.item():.6f}")

    # no_grad / detach demo
    with torch.no_grad():
        y3 = x2 ** 2  # not tracked; no graph is built, saves memory during inference
    print(f"\nUnder torch.no_grad(): y3.requires_grad = {y3.requires_grad}")


# ---------------------------------------------------------------------------
# 3. nn.Module: the same XOR network as Lesson 2, framework-managed
# ---------------------------------------------------------------------------

class TwoLayerNet(nn.Module):
    """Same architecture as SimpleMLP in 02-Neural-Networks-Basics/example.py:
    2 inputs -> 4 hidden (sigmoid) -> 1 output (sigmoid)."""

    def __init__(self, n_in=2, n_hidden=4, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.sigmoid(self.fc1(x))
        out = self.sigmoid(self.fc2(h))
        return out


def training_loop_demo():
    print("\n" + "=" * 70)
    print("3. nn.Module + THE STANDARD TRAINING LOOP (XOR, PyTorch version)")
    print("=" * 70)

    torch.manual_seed(42)

    # Shape convention here is (batch, features) -- PyTorch's default --
    # which is the transpose of the (features, batch) convention used in
    # the from-scratch NumPy version in Lesson 2.
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    model = TwoLayerNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    epochs = 5000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()          # 1. clear old gradients
        output = model(X)              # 2. forward pass
        loss = criterion(output, Y)    # 3. compute the loss
        loss.backward()                # 4. backward pass (autograd computes every gradient)
        optimizer.step()               # 5. gradient descent update

        if epoch % 1000 == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  loss = {loss.item():.6f}")

    print(f"\nFinal loss: {loss.item():.6f}")
    with torch.no_grad():
        predictions = model(X)
    print("\nPredictions vs targets:")
    for i in range(X.shape[0]):
        pred = predictions[i, 0].item()
        target = Y[i, 0].item()
        print(f"  input={X[i].tolist()} -> predicted={pred:.4f}  "
              f"(rounded={round(pred)}, target={int(target)})")


def main():
    tensor_basics_demo()
    autograd_demo()
    training_loop_demo()


if __name__ == "__main__":
    main()
