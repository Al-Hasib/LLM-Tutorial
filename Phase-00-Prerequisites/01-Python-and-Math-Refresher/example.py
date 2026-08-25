"""
Python and Math Refresher

Hands-on demos of the linear algebra, calculus, and probability building
blocks used throughout this course: dot products, matrix multiplication,
norms, numerical vs. analytical derivatives, gradients, the chain rule,
and softmax / cross-entropy.

Run:
    python example.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Linear algebra
# ---------------------------------------------------------------------------

def linear_algebra_demo():
    print("=" * 70)
    print("1. LINEAR ALGEBRA")
    print("=" * 70)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    # Dot product: manual sum-of-products vs. NumPy
    manual_dot = sum(ai * bi for ai, bi in zip(a, b))
    numpy_dot = np.dot(a, b)
    print(f"a = {a}, b = {b}")
    print(f"dot product (manual)  = {manual_dot}")
    print(f"dot product (np.dot)  = {numpy_dot}")
    assert np.isclose(manual_dot, numpy_dot)

    # Matrix multiplication — shapes must line up: (d x n) @ (n x m) -> (d x m)
    W = np.array([[1.0, 0.0, -1.0],
                  [0.5, 2.0, 1.0]])          # shape (2, 3)
    x = np.array([1.0, 2.0, 3.0])            # shape (3,)
    y = W @ x                                # shape (2,)
    print(f"\nW shape = {W.shape}, x shape = {x.shape}")
    print(f"y = W @ x = {y}  (shape {y.shape})")

    # Transpose
    print(f"\nW^T =\n{W.T}")

    # Norms
    l2 = np.linalg.norm(a, ord=2)
    l1 = np.linalg.norm(a, ord=1)
    print(f"\n||a||_2 (Euclidean norm) = {l2:.4f}")
    print(f"||a||_1 (Manhattan norm) = {l1:.4f}")


# ---------------------------------------------------------------------------
# 2. Calculus: derivatives, gradients, chain rule
# ---------------------------------------------------------------------------

def numerical_derivative(f, x, h=1e-6):
    """Central-difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def calculus_demo():
    print("\n" + "=" * 70)
    print("2. CALCULUS")
    print("=" * 70)

    # f(x) = x^2  ->  f'(x) = 2x
    f = lambda x: x ** 2
    x0 = 3.0
    analytic = 2 * x0
    numeric = numerical_derivative(f, x0)
    print(f"f(x) = x^2 at x=3 -> f'(x) analytic = {analytic}, numeric = {numeric:.6f}")

    # Gradient of a multivariable function: g(x, y) = x^2 + 3xy
    def g(x, y):
        return x ** 2 + 3 * x * y

    def grad_g(x, y, h=1e-6):
        dgdx = (g(x + h, y) - g(x - h, y)) / (2 * h)
        dgdy = (g(x, y + h) - g(x, y - h)) / (2 * h)
        return np.array([dgdx, dgdy])

    x0, y0 = 2.0, 1.0
    numeric_grad = grad_g(x0, y0)
    analytic_grad = np.array([2 * x0 + 3 * y0, 3 * x0])  # [dg/dx, dg/dy]
    print(f"grad(g)(2, 1) analytic = {analytic_grad}, numeric = {numeric_grad}")

    # Chain rule: y = f(u), u = g(x)  ->  dy/dx = f'(u) * g'(x)
    # Let g(x) = x^2 (u = x^2), f(u) = sin(u)  ->  y = sin(x^2)
    def compose(x):
        return np.sin(x ** 2)

    x0 = 1.5
    chain_rule_analytic = np.cos(x0 ** 2) * (2 * x0)   # f'(u)*g'(x)
    chain_rule_numeric = numerical_derivative(compose, x0)
    print(f"d/dx sin(x^2) at x=1.5 -> chain rule = {chain_rule_analytic:.6f}, "
          f"numeric = {chain_rule_numeric:.6f}")
    print("(This exact mechanism, applied automatically layer by layer, is backpropagation.)")


# ---------------------------------------------------------------------------
# 3. Probability: softmax and cross-entropy
# ---------------------------------------------------------------------------

def softmax(logits):
    # Subtract max for numerical stability (does not change the result).
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def cross_entropy(probs, target_index):
    """Negative log-likelihood of the correct class under `probs`."""
    return -np.log(probs[target_index] + 1e-12)


def probability_demo():
    print("\n" + "=" * 70)
    print("3. PROBABILITY")
    print("=" * 70)

    # Pretend these are logits ("scores") the model produced for 4 candidate
    # next tokens, before the softmax turns them into a distribution.
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    probs = softmax(logits)
    print(f"logits = {logits}")
    print(f"softmax(logits) = {np.round(probs, 4)}")
    print(f"sum(probs) = {probs.sum():.6f}  (always ~1.0 for a valid distribution)")

    true_next_token = 0  # say the correct next token is index 0
    loss = cross_entropy(probs, true_next_token)
    print(f"\nTrue next token index = {true_next_token}")
    print(f"Cross-entropy loss = -log(p_correct) = {loss:.4f}")
    print("This exact loss (cross-entropy over the vocabulary) is what every "
          "LLM in this course minimizes during pretraining and fine-tuning.")


def main():
    linear_algebra_demo()
    calculus_demo()
    probability_demo()


if __name__ == "__main__":
    main()
