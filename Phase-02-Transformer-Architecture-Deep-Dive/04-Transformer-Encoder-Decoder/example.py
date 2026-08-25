"""
Transformer Encoder-Decoder Architecture

A complete (small) encoder-decoder Transformer in PyTorch, assembled
from the pieces built in Lessons 1-3: token embeddings + sinusoidal
positional encoding, multi-head self-attention, masked self-attention,
cross-attention, and a position-wise feed-forward network. We run a
toy forward pass end to end and inspect shapes at every stage,
including the cross-attention weights that connect decoder to encoder.

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Building blocks from Lessons 1-3
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class MultiHeadAttention(nn.Module):
    """Same mechanism as Lesson 2, generalized so Q can come from a DIFFERENT
    sequence than K/V -- this is what makes it reusable for both self-attention
    (query_input is key_value_input) and cross-attention (they differ)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        batch, T, d_model = x.shape
        return x.view(batch, T, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        batch, num_heads, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch, T, num_heads * d_k)

    def forward(self, query_input, key_value_input, mask=None):
        Q = self._split_heads(self.W_q(query_input))
        K = self._split_heads(self.W_k(key_value_input))
        V = self._split_heads(self.W_v(key_value_input))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_output = self._combine_heads(weights @ V)
        return self.W_o(attn_output), weights


class PositionwiseFeedForward(nn.Module):
    """Same Linear -> activation -> Linear block covered in full in Lesson 5."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device)).bool()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x)             # bidirectional self-attention, no mask
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", sinusoidal_positional_encoding(max_len, d_model))
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, token_ids):
        T = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_encoding[:T]
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, causal_mask_tensor):
        self_attn_out, _ = self.self_attn(x, x, mask=causal_mask_tensor)
        x = self.norm1(x + self_attn_out)

        # Cross-attention: query from the DECODER, key/value from the ENCODER.
        cross_attn_out, cross_attn_weights = self.cross_attn(x, encoder_output)
        x = self.norm2(x + cross_attn_out)

        x = self.norm3(x + self.ffn(x))
        return x, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", sinusoidal_positional_encoding(max_len, d_model))
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, encoder_output):
        T = token_ids.shape[1]
        x = self.token_embed(token_ids) + self.pos_encoding[:T]
        mask = causal_mask(T, token_ids.device)

        last_cross_attn_weights = None
        for layer in self.layers:
            x, last_cross_attn_weights = layer(x, encoder_output, mask)

        logits = self.output_proj(x)   # (batch, T, vocab_size)
        return logits, last_cross_attn_weights


def main():
    print("=" * 70)
    print("FULL ENCODER-DECODER TRANSFORMER: TOY FORWARD PASS")
    print("=" * 70)

    src_vocab_size, tgt_vocab_size = 50, 60
    d_model, num_heads, d_ff, num_layers = 32, 4, 64, 2
    max_len = 20

    encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len)

    batch_size, src_len, tgt_len = 2, 7, 5
    src_tokens = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    print(f"source token ids shape: {tuple(src_tokens.shape)}  (batch, src_len)")
    print(f"target token ids shape: {tuple(tgt_tokens.shape)}  (batch, tgt_len)")

    encoder_output = encoder(src_tokens)
    print(f"\nencoder_output shape:   {tuple(encoder_output.shape)}  (batch, src_len, d_model)")
    print("-> Every source position now has a representation that has attended")
    print("   over the whole source sentence (bidirectional self-attention).")

    logits, cross_attn_weights = decoder(tgt_tokens, encoder_output)
    print(f"\ndecoder logits shape:   {tuple(logits.shape)}  (batch, tgt_len, tgt_vocab_size)")
    print(f"cross-attn weights shape: {tuple(cross_attn_weights.shape)}  "
          f"(batch, heads, tgt_len, src_len)")
    print("-> Note the LAST two dimensions: tgt_len x src_len, not tgt_len x tgt_len.")
    print("   Cross-attention queries come from the decoder (length tgt_len) but")
    print("   keys/values come from the encoder (length src_len) -- exactly the")
    print("   'decoder looks back at the whole source sentence' mechanism from")
    print("   Phase 01's Seq2Seq attention lesson, now fully general.")

    print("\nCross-attention weights for batch 0, head 0 (rows=target position,")
    print("columns=source position), each row still sums to 1:")
    print(cross_attn_weights[0, 0].detach().numpy().round(3))

    next_token_probs = F.softmax(logits[0, -1], dim=-1)
    print(f"\nSanity check: softmax over vocab at the last decoder position sums to "
          f"{next_token_probs.sum().item():.6f}")


if __name__ == "__main__":
    main()
