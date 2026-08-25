"""
Positional Encoding

Three demos:
  1. Proof that self-attention alone is permutation-invariant (a set
     operation) -- shuffling the input and un-shuffling the output
     gives back exactly the same per-token results.
  2. Sinusoidal positional encoding: nearby positions get similar
     "fingerprints" that smoothly decay with distance.
  3. The relative-position rotation property: PE(pos+k) is a fixed
     linear (rotation) function of PE(pos), independent of pos.

Run:
    python example.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# 1. Self-attention is permutation-invariant
# ---------------------------------------------------------------------------

def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def permutation_invariance_demo():
    print("=" * 70)
    print("1. SELF-ATTENTION IS PERMUTATION-INVARIANT (NO POSITIONAL INFO)")
    print("=" * 70)

    T, d_model, d_k = 5, 8, 8
    X = torch.randn(T, d_model)
    Wq = torch.randn(d_model, d_k) * 0.3
    Wk = torch.randn(d_model, d_k) * 0.3
    Wv = torch.randn(d_model, d_k) * 0.3

    output_original = self_attention(X, Wq, Wk, Wv)

    # Shuffle the tokens, run attention again, then undo the shuffle on the
    # output. If attention truly has no notion of position, this must give
    # back exactly the same result as the original, unshuffled run.
    perm = torch.randperm(T)
    inverse_perm = torch.argsort(perm)
    X_shuffled = X[perm]
    output_shuffled = self_attention(X_shuffled, Wq, Wk, Wv)
    output_unshuffled = output_shuffled[inverse_perm]

    max_diff = (output_original - output_unshuffled).abs().max().item()
    print(f"Random permutation applied: {perm.tolist()}")
    print(f"Max difference between original output and un-shuffled-shuffled "
          f"output: {max_diff:.2e}")
    print("-> Effectively 0: attention treats the input as a SET. The model")
    print("   genuinely cannot tell 'A B C' from 'C A B' unless something else")
    print("   (positional encoding) tells it which token came from where.")


# ---------------------------------------------------------------------------
# 2. Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(max_len, d_model):
    """Returns a (max_len, d_model) matrix, PE(pos, dim) as in the README."""
    positions = np.arange(max_len)[:, None]                      # (max_len, 1)
    dim_indices = np.arange(d_model)[None, :]                    # (1, d_model)
    angle_rates = 1.0 / (10000 ** ((2 * (dim_indices // 2)) / d_model))
    angles = positions * angle_rates                              # (max_len, d_model)

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def sinusoidal_demo():
    print("\n" + "=" * 70)
    print("2. SINUSOIDAL POSITIONAL ENCODING: SIMILARITY DECAYS SMOOTHLY")
    print("=" * 70)

    d_model, max_len = 32, 50
    pe = sinusoidal_positional_encoding(max_len, d_model)

    reference_pos = 10
    print(f"cosine similarity between PE({reference_pos}) and PE(p), for various p:")
    for p in [10, 11, 12, 15, 20, 30, 49]:
        sim = cosine_similarity(pe[reference_pos], pe[p])
        distance = abs(p - reference_pos)
        print(f"  p={p:3d}  (distance {distance:3d})  cosine similarity = {sim:.4f}")

    print("\n-> Similarity is highest at distance 0 and generally decays as the")
    print("   position gets farther away -- each position gets a unique, but")
    print("   smoothly-varying, 'fingerprint' vector.")


# ---------------------------------------------------------------------------
# 3. The relative-position rotation property
# ---------------------------------------------------------------------------

def rotation_matrix_for_offset(k, d_model):
    """Builds the block-diagonal rotation matrix M_k such that, in theory,
    PE(pos + k) = M_k @ PE(pos) for every pos (see README derivation)."""
    M = np.zeros((d_model, d_model))
    for i in range(0, d_model, 2):
        theta = k / (10000 ** (i / d_model))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Angle-addition identities give, for v = [sin(pos*w), cos(pos*w)]:
        #   sin((pos+k)*w) = sin(pos*w)cos(k*w) + cos(pos*w)sin(k*w) =  cos_t*v[i] + sin_t*v[i+1]
        #   cos((pos+k)*w) = cos(pos*w)cos(k*w) - sin(pos*w)sin(k*w) = -sin_t*v[i] + cos_t*v[i+1]
        M[i, i] = cos_t
        M[i, i + 1] = sin_t
        M[i + 1, i] = -sin_t
        M[i + 1, i + 1] = cos_t
    return M


def relative_position_rotation_demo():
    print("\n" + "=" * 70)
    print("3. PE(pos + k) IS A FIXED LINEAR (ROTATION) FUNCTION OF PE(pos)")
    print("=" * 70)

    d_model, max_len = 16, 60
    pe = sinusoidal_positional_encoding(max_len, d_model)

    k = 5
    M_k = rotation_matrix_for_offset(k, d_model)

    print(f"Fixed offset k={k}. Checking PE(pos+{k}) ~= M_{k} @ PE(pos) for several pos:")
    print(f"{'pos':>5}  {'max abs error':>14}")
    for pos in [0, 3, 10, 25, 40]:
        predicted = M_k @ pe[pos]
        actual = pe[pos + k]
        max_error = np.abs(predicted - actual).max()
        print(f"{pos:>5}  {max_error:>14.2e}")

    print("\n-> The same rotation matrix M_k (built only from the offset k, not")
    print("   from pos) correctly maps PE(pos) to PE(pos+k) at every pos tested.")
    print("   This is the mathematical basis for the claim that a linear layer")
    print("   could, in principle, learn to attend by RELATIVE position using")
    print("   sinusoidal encodings -- the seed of the idea RoPE later builds on")
    print("   directly (Phase 03: Long-Context Techniques).")


def main():
    permutation_invariance_demo()
    sinusoidal_demo()
    relative_position_rotation_demo()


if __name__ == "__main__":
    main()
