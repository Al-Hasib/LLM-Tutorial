"""
Decoder-Only Models: the GPT Family

Computes real parameter counts for the published GPT-1/GPT-2-family
configurations, using the exact same architecture formula as Phase 02's
mini-GPT (token embedding + positional embedding + N x (attention + FFN)
blocks + tied output head), and compares the exact count against the
"quick estimate" formula (~12 * n_layer * d_model^2) commonly used in
scaling-laws literature (Lesson 5 uses this same shorthand).

Run:
    python example.py
"""


def count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True):
    """Exact parameter count for a GPT-2-style decoder-only Transformer:
    - token embedding:      vocab_size * d_model
    - positional embedding: n_ctx * d_model            (learned, GPT-2 style)
    - per layer:
        attention (4 Linear(d_model, d_model), with bias): 4*(d_model^2 + d_model)
        2 LayerNorms (gamma + beta each):                   4*d_model
        FFN (d_model -> 4*d_model -> d_model, with bias):
            (d_model*d_ff + d_ff) + (d_ff*d_model + d_model), d_ff = 4*d_model
    - final LayerNorm: 2*d_model
    - output head: tied to token embedding in GPT-2 (0 extra params) or
      an untied vocab_size * d_model matrix otherwise.
    """
    d_ff = 4 * d_model

    token_embed = vocab_size * d_model
    pos_embed = n_ctx * d_model

    attn_params = 4 * (d_model * d_model + d_model)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    layernorm_params = 2 * (2 * d_model)   # 2 LayerNorms per layer, gamma+beta each
    per_layer = attn_params + ffn_params + layernorm_params

    final_layernorm = 2 * d_model
    output_head = 0 if weight_tying else vocab_size * d_model

    total = token_embed + pos_embed + n_layer * per_layer + final_layernorm + output_head
    return total


def quick_estimate(n_layer, d_model):
    """The ~12*n_layer*d_model^2 shorthand used throughout scaling-laws papers
    (see Lesson 5) -- it ignores embedding/vocab terms entirely, which is a
    fine approximation once d_model is large relative to n_ctx and vocab_size
    is a small multiple of d_model, but noticeably off for smaller configs."""
    return 12 * n_layer * d_model * d_model


GPT_CONFIGS = [
    # name,           n_layer, d_model, n_head, n_ctx, vocab_size, published_params
    ("GPT-1",         12,      768,     12,     512,   40000,      "~117M"),
    ("GPT-2 small",   12,      768,     12,     1024,  50257,      "~117M"),
    ("GPT-2 medium",  24,      1024,    16,     1024,  50257,      "~345M"),
    ("GPT-2 large",   36,      1280,    20,     1024,  50257,      "~774M"),
    ("GPT-2 XL",      48,      1600,    25,     1024,  50257,      "~1.5B"),
]


def main():
    print("=" * 90)
    print("PARAMETER COUNTS: PUBLISHED GPT CONFIGS vs. EXACT FORMULA vs. QUICK ESTIMATE")
    print("=" * 90)
    header = f"{'model':<14}{'layers':>8}{'d_model':>9}{'heads':>7}{'published':>12}" \
             f"{'exact count':>16}{'~12*L*d^2':>14}"
    print(header)
    for name, n_layer, d_model, n_head, n_ctx, vocab_size, published in GPT_CONFIGS:
        exact = count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True)
        estimate = quick_estimate(n_layer, d_model)
        print(f"{name:<14}{n_layer:>8}{d_model:>9}{n_head:>7}{published:>12}"
              f"{exact:>16,}{estimate:>14,}")

    print("\n-> The exact formula lands close to each model's published parameter")
    print("   count (small deviations come from GPT's real vocab size / embedding")
    print("   details, which vary slightly by exact release). The '~12*L*d^2'")
    print("   shorthand ignores the embedding table entirely -- notice it UNDERSHOOTS")
    print("   noticeably for GPT-2 small (where the ~38.6M-parameter embedding table")
    print("   is a large fraction of the model) but gets proportionally much closer")
    print("   for GPT-2 XL, where 48 huge transformer layers dwarf the embedding")
    print("   table. This is exactly why scaling-laws papers (Lesson 5) can get away")
    print("   with the simpler formula when studying large-scale trends.")

    print("\n" + "=" * 90)
    print("HOW MUCH OF EACH MODEL IS 'JUST' THE EMBEDDING TABLE?")
    print("=" * 90)
    for name, n_layer, d_model, n_head, n_ctx, vocab_size, published in GPT_CONFIGS:
        exact = count_decoder_only_params(vocab_size, n_ctx, d_model, n_layer, weight_tying=True)
        embed_params = vocab_size * d_model + n_ctx * d_model
        print(f"  {name:<14} embedding share = {embed_params / exact:.1%}")

    print("\n-> This share shrinks steadily as models get bigger -- exactly why the")
    print("   embedding-free shorthand estimate gets more accurate at larger scale.")


if __name__ == "__main__":
    main()
