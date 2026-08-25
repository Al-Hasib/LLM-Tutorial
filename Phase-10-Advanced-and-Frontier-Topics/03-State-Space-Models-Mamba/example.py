"""
State Space Models (Mamba)

Two demos:
  1. The core S4 duality: a fixed-A/B/C linear state-space recurrence
     computed BOTH sequentially (one step at a time) and as a single
     global convolution against a fixed kernel -- verified to agree to
     floating-point precision.
  2. A selective (Mamba-style, input-dependent gating) SSM vs. a fixed
     (S4-style) SSM, compared on how much the FINAL state still depends
     on the FIRST input as sequence length grows -- the exact same
     methodology used to compare a vanilla RNN against an LSTM in
     Phase 01 Lesson 3.

Run:
    python example.py
"""

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. Sequential recurrence vs. global convolution -- the S4 duality
# ---------------------------------------------------------------------------

def sequential_ssm(x_seq, A, B, C):
    """h_t = A h_{t-1} + B x_t ; y_t = C h_t, run one step at a time."""
    N = A.shape[0]
    h = np.zeros(N)
    ys = []
    for x_t in x_seq:
        h = A @ h + B * x_t
        ys.append((C @ h))
    return np.array(ys)


def convolution_ssm(x_seq, A, B, C):
    """The exact same recurrence, computed as a single causal convolution
    against the kernel K_i = C @ A^i @ B (see README section 3)."""
    T = len(x_seq)
    N = A.shape[0]
    kernel = np.zeros(T)
    A_power = np.eye(N)
    for i in range(T):
        kernel[i] = C @ (A_power @ B)
        A_power = A @ A_power

    ys = np.zeros(T)
    for t in range(T):
        # y_t = sum_{k=0}^{t} kernel[t-k] * x_k   (0-indexed causal convolution)
        ys[t] = np.dot(kernel[: t + 1][::-1], x_seq[: t + 1])
    return ys


def duality_demo():
    print("=" * 70)
    print("1. RECURRENCE == CONVOLUTION: THE CORE S4 DUALITY")
    print("=" * 70)

    N = 4  # state dimension
    T = 20
    # A fixed, stable system: eigenvalues comfortably inside the unit circle
    # so the recurrence doesn't blow up over T=20 steps.
    raw = rng.normal(scale=0.5, size=(N, N))
    A = raw / (np.max(np.abs(np.linalg.eigvals(raw))) * 1.5)
    B = rng.normal(size=N)
    C = rng.normal(size=N)

    x_seq = rng.normal(size=T)

    y_sequential = sequential_ssm(x_seq, A, B, C)
    y_convolution = convolution_ssm(x_seq, A, B, C)

    max_diff = np.abs(y_sequential - y_convolution).max()
    print(f"State dimension N={N}, sequence length T={T}")
    print(f"Max abs difference between sequential and convolution outputs: {max_diff:.2e}")
    print("\nFirst 5 outputs, sequential vs. convolution:")
    for t in range(5):
        print(f"  t={t}  sequential={y_sequential[t]:.6f}   convolution={y_convolution[t]:.6f}")

    print("\n-> Identical (to floating-point precision) despite being computed two")
    print("   completely different ways: one step-by-step with a hard dependency")
    print("   chain (O(T), sequential-only), the other as one shot against a fixed")
    print("   kernel (parallelizable across the whole sequence at once). This")
    print("   agreement IS the duality that makes S4-style models trainable in")
    print("   parallel like attention, yet able to run as an O(1)-per-step")
    print("   recurrence like an RNN at inference time.")


# ---------------------------------------------------------------------------
# 2. Selective (Mamba-style) vs. fixed (S4-style) SSM: memory retention
# ---------------------------------------------------------------------------

def fixed_ssm_forward(x_seq, decay, hidden_dim):
    """A fixed, non-selective SSM: the SAME decay factor every step,
    regardless of input content -- a plain exponential-decay memory."""
    h = np.zeros(hidden_dim)
    for x_t in x_seq:
        h = decay * h + (1 - decay) * x_t
    return h


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def selective_ssm_forward(x_seq, W_delta, b_delta, hidden_dim):
    """A selective SSM: the retention factor a_t = sigmoid(W_delta . x_t + b_delta)
    is COMPUTED FROM THE CURRENT INPUT at every step, exactly like an LSTM's
    forget gate (Phase 01 Lesson 3). b_delta is initialized LARGE and positive
    -- the same forget-bias trick from Phase 01 Lesson 3 -- so the default
    behavior is "retain strongly," while individual inputs can still push a_t
    down (via W_delta) to selectively overwrite state when their content
    warrants it."""
    h = np.zeros(hidden_dim)
    for x_t in x_seq:
        a_t = sigmoid(W_delta * x_t + b_delta)   # elementwise, input-dependent retention
        h = a_t * h + (1 - a_t) * x_t
    return h


def gradient_of_output_wrt_first_input(forward_fn, x_seq, eps=1e-4):
    """Central-difference gradient of sum(h_final) w.r.t. x_seq[0] --
    the EXACT SAME methodology used to compare a vanilla RNN against an
    LSTM in Phase 01 Lesson 3's example.py."""
    x0 = x_seq[0]
    seq_plus = x_seq.copy()
    seq_minus = x_seq.copy()
    seq_plus[0] = x0 + eps
    seq_minus[0] = x0 - eps
    out_plus = forward_fn(seq_plus).sum()
    out_minus = forward_fn(seq_minus).sum()
    return (out_plus - out_minus) / (2 * eps)


def selectivity_demo():
    print("\n" + "=" * 70)
    print("2. SELECTIVE (MAMBA-STYLE) vs. FIXED (S4-STYLE) SSM: MEMORY RETENTION")
    print("=" * 70)
    print("Measuring: how much does the FINAL state still depend on the FIRST")
    print("input, as sequence length grows? (Same question, same method, as")
    print("Phase 01 Lesson 3's vanilla-RNN-vs-LSTM gradient comparison.)\n")

    hidden_dim = 8
    FIXED_DECAY = 0.85   # a moderate, content-independent decay every step

    W_delta = rng.normal(scale=0.3, size=hidden_dim)
    b_delta = np.full(hidden_dim, 3.0)   # LSTM-style forget-bias trick: retain by default

    max_len = 60
    full_sequence = rng.normal(size=max_len)

    print(f"{'seq_len':>8}  {'fixed SSM grad':>16}  {'selective SSM grad':>20}")
    for T in [2, 5, 10, 20, 40, 60]:
        x_seq = full_sequence[:T]

        fixed_grad = gradient_of_output_wrt_first_input(
            lambda seq: fixed_ssm_forward(seq, FIXED_DECAY, hidden_dim), x_seq
        )
        selective_grad = gradient_of_output_wrt_first_input(
            lambda seq: selective_ssm_forward(seq, W_delta, b_delta, hidden_dim), x_seq
        )
        print(f"{T:>8}  {abs(fixed_grad):>16.8f}  {abs(selective_grad):>20.8f}")

    print("\n-> The fixed SSM's dependence on the first input decays geometrically")
    print("   (multiplying by the same 0.85 factor every step, exactly like a")
    print("   vanilla RNN's repeated tanh-derivative multiplication) -- it has no")
    print("   way to decide, per token, to protect that early signal. The")
    print("   selective SSM's retention gate defaults toward 'keep the existing")
    print("   state' (via the large positive bias, the same trick used to")
    print("   initialize an LSTM's forget gate), so its dependence on the first")
    print("   input decays far more slowly across the same range of sequence")
    print("   lengths -- input-dependent gating, not a bigger state, is what")
    print("   buys the extra memory.")


def main():
    duality_demo()
    selectivity_demo()


if __name__ == "__main__":
    main()
