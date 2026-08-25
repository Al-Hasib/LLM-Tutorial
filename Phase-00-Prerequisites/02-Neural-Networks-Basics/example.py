"""
Neural Networks Basics

A minimal multi-layer perceptron (MLP) implemented from scratch with only
NumPy: forward pass, manual backpropagation, and gradient descent, trained
on the classic XOR problem (not linearly separable, so it *requires* a
hidden layer + non-linearity to solve — a perfect first demo of why depth
and non-linearity matter).

Run:
    python example.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# ---------------------------------------------------------------------------
# A tiny from-scratch MLP: input -> hidden (sigmoid) -> output (sigmoid)
# ---------------------------------------------------------------------------

class SimpleMLP:
    def __init__(self, n_input, n_hidden, n_output, seed=0):
        rng = np.random.default_rng(seed)
        # Small random init so neurons don't all start identical.
        self.W1 = rng.normal(0, 1, size=(n_hidden, n_input)) * 0.5
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.normal(0, 1, size=(n_output, n_hidden)) * 0.5
        self.b2 = np.zeros((n_output, 1))

    def forward(self, x):
        """x: shape (n_input, batch). Returns y_hat and cached activations."""
        z1 = self.W1 @ x + self.b1
        a1 = sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = sigmoid(z2)
        cache = (x, z1, a1, z2, a2)
        return a2, cache

    def backward(self, cache, y, learning_rate):
        """Manual backprop: chain rule, layer by layer, from loss to W1/b1."""
        x, z1, a1, z2, a2 = cache
        batch_size = x.shape[1]

        # Mean-squared-error loss: L = mean((a2 - y)^2)
        # dL/da2 = 2*(a2 - y) / batch_size
        dL_da2 = 2 * (a2 - y) / batch_size

        # Layer 2 gradients
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        dL_dW2 = dL_dz2 @ a1.T
        dL_db2 = dL_dz2.sum(axis=1, keepdims=True)

        # Propagate error back into layer 1
        dL_da1 = self.W2.T @ dL_dz2
        dL_dz1 = dL_da1 * sigmoid_derivative(z1)
        dL_dW1 = dL_dz1 @ x.T
        dL_db1 = dL_dz1.sum(axis=1, keepdims=True)

        # Gradient descent update: theta -= lr * dL/dtheta
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

    def train(self, X, Y, epochs=5000, learning_rate=0.5, log_every=1000):
        for epoch in range(1, epochs + 1):
            y_hat, cache = self.forward(X)
            loss = float(np.mean((y_hat - Y) ** 2))
            self.backward(cache, Y, learning_rate)
            if epoch % log_every == 0 or epoch == 1:
                print(f"epoch {epoch:5d}  loss = {loss:.6f}")
        return loss


def main():
    # XOR truth table. Shapes are (n_features, n_examples) to match the
    # W @ x convention used in the README's forward-pass equations.
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)          # (2 inputs, 4 examples)
    Y = np.array([[0, 1, 1, 0]], dtype=float)          # (1 output, 4 examples)

    print("Training a from-scratch NumPy MLP on XOR")
    print("(XOR is NOT linearly separable -> a single perceptron cannot solve it,")
    print(" which is exactly why we need a hidden layer + non-linearity.)\n")

    model = SimpleMLP(n_input=2, n_hidden=4, n_output=1, seed=42)
    final_loss = model.train(X, Y, epochs=5000, learning_rate=0.5, log_every=1000)

    print(f"\nFinal loss: {final_loss:.6f}")
    predictions, _ = model.forward(X)
    print("\nPredictions vs targets:")
    for i in range(X.shape[1]):
        x_pair = X[:, i]
        pred = predictions[0, i]
        target = Y[0, i]
        print(f"  input={x_pair.astype(int)} -> predicted={pred:.4f}  "
              f"(rounded={round(pred)}, target={int(target)})")


if __name__ == "__main__":
    main()
