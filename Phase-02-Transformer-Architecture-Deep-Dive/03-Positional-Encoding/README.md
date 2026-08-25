# Positional Encoding

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `03-Positional-Encoding`

## Why this matters

Self-attention, as built in the previous lesson, computes a weighted average over a *set* of value vectors. Nothing about `softmax(QKᵀ/√d_k)V` refers to which position anything came from — if you shuffled the input tokens and shuffled the output back to match, you'd get identical results. That's a real problem: "the dog bit the man" and "the man bit the dog" would be indistinguishable to a Transformer with no notion of order. This lesson fixes that gap, which is entirely absent from an RNN (which processes tokens strictly in sequence and therefore gets order "for free") but has to be explicitly engineered into a Transformer.

## What this lesson covers

- Proving self-attention is permutation-invariant without extra help
- Sinusoidal absolute positional encoding: the formula and why it was chosen
- The relative-position property sinusoidal encodings have "for free"
- Learned positional embeddings as a simpler alternative
- A preview of relative position schemes (RoPE, ALiBi)

## 1. Self-attention has no idea what order tokens came in

Look again at scaled dot-product attention: every score `qᵢ · kⱼ` depends only on the *content* of tokens `i` and `j` — never on `i` or `j` themselves as numbers. If you permute every row of `Q`, `K`, and `V` the same way, the entire computation permutes right along with it and produces the exact same set of output vectors, just relabeled. Self-attention is a set operation wearing a sequence's clothes. `example.py` demonstrates this directly: permute a toy input, and the outputs (matched by token identity) come out identical.

## 2. The fix: add position information into the input

The Transformer paper's solution is almost aggressively simple: compute a vector that depends only on a token's position, and **add it directly to the token embedding** before the first attention layer:

```
input_to_layer_1 = token_embedding(token) + positional_encoding(position)
```

Since token embedding and positional encoding are added together, and attention has no way to tell them apart, position information now flows through the exact same `Q/K/V` projections and influences every attention score.

## 3. Sinusoidal positional encoding

The original paper's specific choice, for position `pos` and embedding dimension index `i` (out of `d_model` total dimensions):

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

Each pair of dimensions `(2i, 2i+1)` oscillates at its own frequency — low dimensions oscillate fast (change a lot between adjacent positions), high dimensions oscillate slowly (change gradually over long spans). This gives every position a unique "fingerprint" vector, and nearby positions get *similar* fingerprints (their cosine similarity is high), which decays smoothly as positions get farther apart — exactly the property `example.py` measures.

## 4. The relative-position trick

Here's the elegant part the paper points out: for **any fixed offset `k`**, `PE(pos + k)` can be written as a **linear function of `PE(pos)`** — specifically, a fixed rotation matrix `M_k` (built purely from `k`, not from `pos`) such that `PE(pos + k) = M_k · PE(pos)` for every `pos`, exploiting the angle-addition identities of sine and cosine. Because attention scores are computed via dot products (linear operations), this means the *relative* distance between two positions is, in principle, something a linear layer downstream could learn to extract — without needing to have seen every specific absolute position during training. `example.py` verifies this rotation relationship numerically.

## 5. Learned positional embeddings

A simpler, equally common alternative (used by GPT-2, BERT, and others): just make position embeddings a regular learned `nn.Embedding` table, indexed by position `0, 1, 2, ...`, trained by gradient descent exactly like a token embedding. Simpler to implement, works well in practice, but has one real limitation: it can never generalize to sequence positions longer than whatever the largest position seen during training was — there's simply no row in the table for position 5000 if the model only ever trained on sequences up to length 1024. Sinusoidal encodings, by contrast, can be computed for *any* position, even ones never seen during training, though in practice trained models still tend to perform worse far outside their training length regardless of which scheme is used.

## 6. Preview: relative position schemes

Both methods above encode **absolute** position. Later architectures found it more effective to encode **relative** position directly into the attention computation itself, rather than adding a separate vector at the input:

- **RoPE (Rotary Position Embedding)**: rotates the `Q` and `K` vectors by an angle proportional to position, so the dot product `Q · K` naturally depends on `(position_i - position_j)`.
- **ALiBi (Attention with Linear Biases)**: skips positional embeddings entirely and instead directly subtracts a distance-proportional penalty from the attention scores.

Both were designed specifically to generalize better to sequences longer than what the model was trained on — the full deep dive is in [Phase 03: Long-Context Techniques](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md).

## Video Script Outline

1. Motivation — "attention doesn't know what order anything came in; prove it live"
2. The fix: add a position-dependent vector to the embedding
3. Sinusoidal formula, visualize the wave patterns across dimensions and positions
4. The linear relative-position property, shown as a rotation
5. Learned embeddings as the simpler alternative, and their length-generalization limit
6. Walkthrough of `example.py`
7. Recap + preview RoPE/ALiBi for later

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.5
- Amirhossein Kazemnejad, *Transformer Architecture: The Positional Encoding* (blog, very thorough derivation of the rotation-matrix property)
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE, previewed here in full in Phase 03)
- Press, Smith, Lewis (2021), *Train Short, Test Long* (ALiBi)
