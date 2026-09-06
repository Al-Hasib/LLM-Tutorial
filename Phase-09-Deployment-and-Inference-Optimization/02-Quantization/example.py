"""
Quantization

Two demos, both implemented from scratch in plain PyTorch tensor ops
(no external quantization libraries):

  1. Symmetric INT8 and INT4 quantization/dequantization of a random
     weight matrix -- measuring reconstruction error (MSE, max absolute
     error) and the actual memory footprint reduction -- plus an FP8
     (E4M3-style) floating-point quantizer simulated the same way, so
     INT8's fixed step size can be compared directly against FP8's
     floating exponent at the same 8-bit budget.
  2. A simplified AWQ-style demo: naive uniform INT4 quantization vs.
     an activation-aware scheme that keeps the top-k% highest-magnitude
     -activation columns at full precision, compared at a matched
     effective average bit-width.

Run:
    python example.py
"""

import torch

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Symmetric quantization / dequantization from scratch
# ---------------------------------------------------------------------------

def quantize_symmetric(x, bits):
    """Uniform symmetric quantization (README section 2).

    scale = max(|x|) / (2^(bits-1) - 1)
    q     = round(x / scale), clamped to the representable integer range
    """
    qmax = 2 ** (bits - 1) - 1     # e.g. 127 for INT8, 7 for INT4
    scale = x.abs().max() / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale


def dequantize_symmetric(q, scale):
    return q * scale


def quantize_float_format(x, exp_bits, mantissa_bits):
    """Simulate a low-bit IEEE-754-style floating-point format (e.g. FP8
    E4M3 is exp_bits=4, mantissa_bits=3) -- README section 6. Unlike
    quantize_symmetric's fixed step size, each value keeps its own
    exponent (clamped to what exp_bits can represent) and only its
    mantissa is rounded to mantissa_bits of precision. This simulator
    skips subnormals/inf/NaN handling (immaterial to the reconstruction
    -error comparison here) but reproduces the core trade-off: a
    floating format spends its limited bits on RELATIVE precision
    (same number of significant bits at any magnitude), while INT8/INT4's
    fixed step size spends them on ABSOLUTE precision across one range.
    """
    sign = torch.sign(x)
    x_abs = torch.clamp(x.abs(), min=1e-12)   # avoid log2(0)
    bias = 2 ** (exp_bits - 1) - 1
    exponent = torch.clamp(torch.floor(torch.log2(x_abs)), -bias + 1, bias)
    mantissa_levels = 2 ** mantissa_bits
    frac = torch.clamp(x_abs / (2.0 ** exponent) - 1.0, 0.0, 1.0)   # in [0, 1)
    frac_q = torch.round(frac * mantissa_levels) / mantissa_levels
    x_hat = sign * (2.0 ** exponent) * (1.0 + frac_q)
    return torch.where(x.abs() == 0, torch.zeros_like(x_hat), x_hat)


def quantization_error_demo():
    print("=" * 70)
    print("1. SYMMETRIC INT8 / INT4 QUANTIZATION FROM SCRATCH")
    print("=" * 70)

    # A weight matrix at a realistic-ish shape for one Transformer FFN
    # projection (Phase 02 Lesson 5's FFN sublayer), filled with values
    # roughly matching a real trained layer's distribution (small, mostly
    # Gaussian, no need for anything fancier to make the point).
    rows, cols = 512, 512
    W = torch.randn(rows, cols) * 0.02
    num_elements = rows * cols
    fp32_bytes = num_elements * 4

    print(f"weight matrix shape: {tuple(W.shape)}  ({num_elements:,} elements)")
    print(f"original dtype: float32  ({fp32_bytes:,} bytes = {fp32_bytes / 1024:.1f} KB)\n")

    print(f"{'scheme':>10}{'bits':>7}{'MSE':>16}{'max abs err':>16}{'bytes':>12}{'reduction':>12}")
    results = {}
    for bits, packed_bytes_per_elem in [(8, 1.0), (4, 0.5)]:
        q, scale = quantize_symmetric(W, bits)
        W_hat = dequantize_symmetric(q, scale)

        mse = torch.mean((W - W_hat) ** 2).item()
        max_abs_err = (W - W_hat).abs().max().item()
        # INT4 values are packed two-per-byte in a real implementation;
        # the per-tensor scale (a single float32) is negligible overhead.
        quantized_bytes = num_elements * packed_bytes_per_elem
        reduction = fp32_bytes / quantized_bytes

        results[bits] = (mse, max_abs_err, quantized_bytes, reduction)
        print(f"{'INT' + str(bits):>10}{bits:>7}{mse:>16.3e}{max_abs_err:>16.5f}"
              f"{quantized_bytes:>12,.0f}{reduction:>11.1f}x")

    mse8, _, bytes8, red8 = results[8]
    mse4, _, bytes4, red4 = results[4]
    print(f"\n-> INT8 shrinks this layer's weights by {red8:.0f}x (from "
          f"{fp32_bytes/1024:.1f} KB to {bytes8/1024:.1f} KB) with a tiny MSE of "
          f"{mse8:.2e}.")
    print(f"-> INT4 shrinks it by {red4:.0f}x, but the MSE rises to {mse4:.2e} -- "
          f"{mse4/mse8:.1f}x worse than INT8 -- because only "
          f"{2**3 - 1} distinct positive integer levels are available to represent")
    print("   the entire range of weight values, versus 127 for INT8. This is exactly")
    print("   the accuracy cliff GPTQ and AWQ exist to soften (README sections 3-4).")

    # --- FP8 (E4M3-style) at the SAME 8-bit budget as INT8 (README section
    #     6), on TWO different distributions, to show FP8 isn't just "INT8
    #     with a different name" -- which format wins depends on the shape
    #     of the data, not just the bit count:
    #       (a) the same narrow, roughly-Gaussian weight matrix as above
    #       (b) a wide-dynamic-range tensor with a few large outliers,
    #           the shape real Transformer ACTIVATIONS tend to have
    #           (the same phenomenon the AWQ demo below exploits) ---
    def fp8_vs_int8(tensor, label):
        q8, scale8 = quantize_symmetric(tensor, bits=8)
        mse_int8 = torch.mean((tensor - dequantize_symmetric(q8, scale8)) ** 2).item()
        W_hat_fp8 = quantize_float_format(tensor, exp_bits=4, mantissa_bits=3)
        mse_fp8 = torch.mean((tensor - W_hat_fp8) ** 2).item()
        winner = "FP8" if mse_fp8 < mse_int8 else "INT8"
        print(f"{label:>34}{mse_int8:>16.3e}{mse_fp8:>16.3e}{winner:>16}")
        return mse_int8, mse_fp8

    outlier_tensor = torch.randn(rows, cols) * 0.02
    num_outliers = max(1, int(outlier_tensor.numel() * 0.001))
    flat = outlier_tensor.flatten()
    flat[torch.randperm(flat.numel())[:num_outliers]] *= 200.0   # a rare, large-magnitude tail

    print(f"\n{'distribution':>34}{'INT8 MSE':>16}{'FP8(E4M3) MSE':>16}{'lower error':>16}")
    fp8_vs_int8(W, "narrow, ~Gaussian (this layer's weights)")
    fp8_vs_int8(outlier_tensor, "wide-range, rare large outliers")

    print("\n-> Same 8-bit budget, same tensor SHAPE, opposite winner depending on the data's")
    print("   dynamic range. INT8's fixed step size is calibrated to the single largest")
    print("   value in the tensor (max(|x|)/127, README section 2) -- on a narrow, ")
    print("   well-behaved distribution that step is already fine-grained, so INT8 wins.")
    print("   Introduce a few rare, large-magnitude outliers and that SAME calibration")
    print("   rule blows the step size up for the whole tensor, crushing resolution")
    print("   everywhere else -- while FP8's floating exponent keeps giving every value")
    print("   the same RELATIVE precision regardless of magnitude, so it barely notices")
    print("   the outliers. This is exactly why FP8 is the more common choice for")
    print("   ACTIVATIONS (which have real outlier channels -- the AWQ demo below is")
    print("   built around this same phenomenon) while weights, often narrower-range,")
    print("   can do just as well or better in INT8. Either format needs the same 1")
    print("   byte/element to store, and the same Tensor Core support (Lesson 1 section 7)")
    print("   to realize a matching compute speedup, not just the memory saving.")


# ---------------------------------------------------------------------------
# 2. A simplified AWQ-style demo: protect activation-salient columns
# ---------------------------------------------------------------------------

def quantize_matrix_int4(W):
    """Quantize an entire matrix to INT4 uniformly (naive RTN, per-tensor scale)."""
    q, scale = quantize_symmetric(W, bits=4)
    return dequantize_symmetric(q, scale)


def awq_style_demo():
    print("\n" + "=" * 70)
    print("2. AWQ-STYLE DEMO: PROTECTING ACTIVATION-SALIENT COLUMNS")
    print("=" * 70)

    rows, cols = 256, 256
    W = torch.randn(rows, cols) * 0.02

    # Simulate calibration-data activation statistics: most input channels
    # (columns of W, since y = x @ W multiplies column j by input channel j)
    # have ordinary magnitude, but a small fraction are consistently large --
    # exactly the "outlier channel" phenomenon AWQ and LLM.int8() both exploit.
    activation_magnitude = torch.abs(torch.randn(cols))
    salient_fraction = 0.05
    num_salient = max(1, int(cols * salient_fraction))
    # Inject a few genuinely large-magnitude activation channels so the
    # salient-column effect is real and measurable, not just noise.
    outlier_cols = torch.randperm(cols)[:num_salient]
    activation_magnitude[outlier_cols] *= 15.0

    print(f"weight matrix shape: {tuple(W.shape)}")
    print(f"salient columns: top {salient_fraction*100:.0f}% by simulated activation "
          f"magnitude = {num_salient} of {cols} columns\n")

    # --- (a) naive uniform INT4 quantization of the whole matrix ---
    W_hat_naive = quantize_matrix_int4(W)

    # To make the output-error comparison meaningful (not just raw weight
    # error), weight each column's reconstruction error by how large the
    # activations multiplying it actually are -- this approximates the
    # effect of a quantization error on the LAYER'S OUTPUT (y = x @ W),
    # since a fixed weight error in a column with large activations
    # produces a proportionally larger error in y.
    def activation_weighted_error(W_orig, W_hat, act_mag):
        per_col_sq_err = ((W_orig - W_hat) ** 2).mean(dim=0)     # (cols,)
        weighted = (per_col_sq_err * act_mag ** 2).sum() / (act_mag ** 2).sum()
        return weighted.item()

    naive_output_err = activation_weighted_error(W, W_hat_naive, activation_magnitude)
    naive_raw_mse = torch.mean((W - W_hat_naive) ** 2).item()

    # --- (b) AWQ-style: keep the top-k% salient columns at fp32, quantize the rest ---
    salient_mask = torch.zeros(cols, dtype=torch.bool)
    salient_mask[outlier_cols] = True

    W_hat_awq = W.clone()
    non_salient = W[:, ~salient_mask]
    q, scale = quantize_symmetric(non_salient, bits=4)
    W_hat_awq[:, ~salient_mask] = dequantize_symmetric(q, scale)
    # salient columns (W_hat_awq[:, salient_mask]) are left untouched at fp32

    awq_output_err = activation_weighted_error(W, W_hat_awq, activation_magnitude)
    awq_raw_mse = torch.mean((W - W_hat_awq) ** 2).item()

    # Effective average bit-width: num_salient columns stay at 32 bits,
    # the remaining (cols - num_salient) columns drop to 4 bits.
    naive_effective_bits = 4.0
    awq_effective_bits = (num_salient * 32 + (cols - num_salient) * 4) / cols

    print(f"{'scheme':>28}{'effective bits/weight':>24}{'raw weight MSE':>18}"
          f"{'activation-weighted err':>26}")
    print(f"{'naive uniform INT4':>28}{naive_effective_bits:>24.2f}"
          f"{naive_raw_mse:>18.3e}{naive_output_err:>26.3e}")
    print(f"{'AWQ-style (protect top ' + f'{salient_fraction*100:.0f}%)':>28}"
          f"{awq_effective_bits:>24.2f}{awq_raw_mse:>18.3e}{awq_output_err:>26.3e}")

    print(f"\n-> Protecting just the top {salient_fraction*100:.0f}% of columns raises the "
          f"effective bit-width only from {naive_effective_bits:.2f} to "
          f"{awq_effective_bits:.2f} bits/weight (still overwhelmingly INT4), yet the "
          f"activation-weighted output error drops by "
          f"{naive_output_err / awq_output_err:.1f}x (from {naive_output_err:.2e} to "
          f"{awq_output_err:.2e}).")
    print("-> This is the AWQ insight made concrete: a small minority of weight columns,")
    print("   the ones multiplied by large-magnitude activations, dominate the layer's")
    print("   OUTPUT error under quantization -- so protecting just those few columns")
    print("   buys most of full-precision's accuracy at almost none of its memory cost.")
    print(f"   (For reference, raw unweighted weight MSE only improves by "
          f"{naive_raw_mse / awq_raw_mse:.2f}x -- the benefit is concentrated exactly where")
    print("   the activations are large, which is the whole point.)")


def main():
    quantization_error_demo()
    awq_style_demo()


if __name__ == "__main__":
    main()
