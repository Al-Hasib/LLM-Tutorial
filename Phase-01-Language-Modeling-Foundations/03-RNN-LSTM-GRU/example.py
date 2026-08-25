"""
RNNs, LSTMs and GRUs

From-scratch NumPy forward passes for a vanilla RNN and an LSTM, plus a
direct numerical demonstration of the vanishing-gradient problem: we
measure how much the FINAL hidden state actually responds to a nudge in
the FIRST input, as sequence length grows, for both architectures.

Run:
    python example.py
"""

import numpy as np

rng = np.random.default_rng(0)

INPUT_DIM = 4
HIDDEN_DIM = 8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. Vanilla RNN forward pass
# ---------------------------------------------------------------------------

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim):
        scale = 1.0 / np.sqrt(hidden_dim)
        self.Wxh = rng.normal(0, scale, size=(hidden_dim, input_dim))
        self.Whh = rng.normal(0, scale, size=(hidden_dim, hidden_dim))
        self.bh = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        h = np.zeros(self.hidden_dim)
        for x in x_seq:
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        return h


# ---------------------------------------------------------------------------
# 2. LSTM forward pass
# ---------------------------------------------------------------------------

class LSTM:
    def __init__(self, input_dim, hidden_dim, forget_bias=2.0):
        scale = 1.0 / np.sqrt(hidden_dim)
        z_dim = input_dim + hidden_dim
        self.Wf = rng.normal(0, scale, size=(hidden_dim, z_dim))
        self.Wi = rng.normal(0, scale, size=(hidden_dim, z_dim))
        self.Wg = rng.normal(0, scale, size=(hidden_dim, z_dim))
        self.Wo = rng.normal(0, scale, size=(hidden_dim, z_dim))
        # A positive forget-gate bias is a well-known init trick: it starts
        # the cell "remembering by default" (sigmoid(2.0) ~ 0.88) instead of
        # forgetting by default (sigmoid(0.0) = 0.5).
        self.bf = np.full(hidden_dim, forget_bias)
        self.bi = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        for x in x_seq:
            z = np.concatenate([h, x])
            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            g = np.tanh(self.Wg @ z + self.bg)
            o = sigmoid(self.Wo @ z + self.bo)
            c = f * c + i * g          # additive update -- the key difference from vanilla RNN
            h = o * np.tanh(c)
        return h


# ---------------------------------------------------------------------------
# 3. Numerically measure how much the final hidden state depends on x[0]
# ---------------------------------------------------------------------------

def gradient_of_output_wrt_first_input(forward_fn, x_seq, eps=1e-4):
    """Central-difference gradient of sum(h_final) w.r.t. x_seq[0]."""
    x0 = x_seq[0]
    grad = np.zeros_like(x0)
    for k in range(len(x0)):
        seq_plus = [x.copy() for x in x_seq]
        seq_minus = [x.copy() for x in x_seq]
        seq_plus[0][k] += eps
        seq_minus[0][k] -= eps
        out_plus = forward_fn(seq_plus).sum()
        out_minus = forward_fn(seq_minus).sum()
        grad[k] = (out_plus - out_minus) / (2 * eps)
    return grad


def main():
    print("Measuring: 'how much does the FIRST input still affect the FINAL")
    print("hidden state?' as the sequence gets longer -- this is exactly what")
    print("a vanishing gradient means in practice: the training signal for an")
    print("early token shrinks toward zero the further away the loss is.\n")

    rnn = VanillaRNN(INPUT_DIM, HIDDEN_DIM)
    lstm = LSTM(INPUT_DIM, HIDDEN_DIM, forget_bias=2.0)

    max_len = 40
    full_sequence = [rng.normal(size=INPUT_DIM) for _ in range(max_len)]

    seq_lengths = [2, 5, 10, 20, 30, 40]
    print(f"{'seq_len':>8}  {'RNN grad norm':>15}  {'LSTM grad norm':>16}")
    for T in seq_lengths:
        x_seq = full_sequence[:T]
        rnn_grad = gradient_of_output_wrt_first_input(rnn.forward, x_seq)
        lstm_grad = gradient_of_output_wrt_first_input(lstm.forward, x_seq)
        print(f"{T:>8}  {np.linalg.norm(rnn_grad):>15.8f}  {np.linalg.norm(lstm_grad):>16.8f}")

    print("\n-> The vanilla RNN's gradient norm collapses toward zero within a")
    print("   handful of steps: information from the first token is effectively")
    print("   erased by the time the network reaches the end of the sequence.")
    print("-> The LSTM's gradient decays far more slowly, because its cell state")
    print("   update (c_t = f_t*c_{t-1} + i_t*g_t) is additive rather than a")
    print("   repeated matrix multiplication -- exactly the mechanism described")
    print("   in the README. This is why LSTMs could learn much longer-range")
    print("   dependencies than vanilla RNNs.")


if __name__ == "__main__":
    main()
