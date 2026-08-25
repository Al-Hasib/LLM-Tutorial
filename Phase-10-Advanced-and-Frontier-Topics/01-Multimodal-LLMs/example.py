"""
Multimodal LLMs -- a toy CLIP-style contrastive training loop from scratch.

This does NOT download any real images, text, or pretrained weights (no
internet access is assumed). Instead it builds a fully synthetic toy problem
that captures the one idea CLIP's loss is actually built to solve:

  - There are `NUM_CATEGORIES` hidden "concepts" (stand-ins for real-world
    categories like "cat", "beach", "mountain", ...). Each category has its
    own random latent vector.
  - For each paired training example, we sample a category, then generate
    an "image" feature vector and a "text" feature vector that BOTH derive
    from that same category latent, but through different, noisy,
    modality-specific transformations -- exactly like a real photo of a cat
    and the word "cat" share an underlying concept but look nothing alike
    at the raw-pixel / raw-token level.
  - Two small encoder MLPs (one per modality) are trained with the
    symmetric InfoNCE contrastive loss from the README, over batches of
    these synthetic pairs, with NO access to the category labels -- only
    the fact that image i and text i were paired together.
  - We then measure cross-modal retrieval accuracy (given an image, find
    its matching text among several candidates) BEFORE training (random
    encoder weights) and AFTER training, on a held-out set never seen
    during training, to show a real, measured improvement.

Runtime: a few seconds on CPU (small MLPs, 400 training steps).

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Synthetic paired (image, text) data generation
# ---------------------------------------------------------------------------

NUM_CATEGORIES = 10        # stand-ins for real concepts like "cat", "beach", ...
LATENT_DIM = 8             # dimensionality of the shared underlying concept
IMAGE_RAW_DIM = 24         # "raw" image-feature dimensionality before encoding
TEXT_RAW_DIM = 20          # "raw" text-feature dimensionality before encoding
EMBED_DIM = 16             # shared embedding dimension both encoders map into

# Fixed random "rendering" transforms: each modality turns a category's
# shared latent into its own raw feature space via a different random
# linear map, so an image-vector and a text-vector for the same category
# share signal but are not at all identical -- much like a photo of a dog
# and the string "dog" share a concept but not a representation.
image_render = torch.randn(LATENT_DIM, IMAGE_RAW_DIM)
text_render = torch.randn(LATENT_DIM, TEXT_RAW_DIM)

# One random latent vector per category, fixed for the whole script.
category_latents = torch.randn(NUM_CATEGORIES, LATENT_DIM)


def sample_pairs(n_pairs, noise_std=0.6):
    """Sample n_pairs of (image_raw, text_raw, category_id).

    Image and text raw vectors for the same pair are generated from the
    SAME category latent, then pushed through different modality-specific
    linear renderings and given independent noise -- so they share signal
    (the category) but are not trivially identical vectors.
    """
    category_ids = torch.randint(0, NUM_CATEGORIES, (n_pairs,))
    latents = category_latents[category_ids]                      # (n_pairs, LATENT_DIM)

    image_raw = latents @ image_render + noise_std * torch.randn(n_pairs, IMAGE_RAW_DIM)
    text_raw = latents @ text_render + noise_std * torch.randn(n_pairs, TEXT_RAW_DIM)
    return image_raw, text_raw, category_ids


# ---------------------------------------------------------------------------
# 2. The two encoders (one small MLP per modality)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Maps a modality's raw feature vector into the shared embedding space."""

    def __init__(self, in_dim, embed_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)   # unit-length embeddings, as in the README


# ---------------------------------------------------------------------------
# 3. The symmetric CLIP / InfoNCE contrastive loss
# ---------------------------------------------------------------------------

def clip_contrastive_loss(image_embeds, text_embeds, log_temperature):
    """Symmetric InfoNCE loss over a batch, exactly as described in the README.

    logits[i, j] = cosine_similarity(image_i, text_j) * exp(log_temperature)
    The correct match for row i (and column i) is always index i, since
    image_embeds and text_embeds are constructed as paired batches.
    """
    batch_size = image_embeds.shape[0]
    logits = (image_embeds @ text_embeds.T) * log_temperature.exp()
    labels = torch.arange(batch_size)

    loss_i2t = F.cross_entropy(logits, labels)          # each image picks its text
    loss_t2i = F.cross_entropy(logits.T, labels)         # each text picks its image
    return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------------
# 4. Retrieval evaluation: given an image, find its matching text
# ---------------------------------------------------------------------------

@torch.no_grad()
def retrieval_accuracy(image_encoder, text_encoder, n_queries=200, n_candidates=8):
    """For each query image, build a candidate pool of n_candidates texts
    (1 true match + (n_candidates - 1) distractors from OTHER pairs) and
    check whether nearest-neighbor cosine similarity picks the true match.

    This mirrors real cross-modal retrieval: "here's an image, which of
    these captions actually describes it?"
    """
    image_raw, text_raw, _ = sample_pairs(n_queries)
    image_embeds = image_encoder(image_raw)                  # (n_queries, EMBED_DIM)
    text_embeds = text_encoder(text_raw)                      # (n_queries, EMBED_DIM), text_embeds[i] matches image_embeds[i]

    correct = 0
    for i in range(n_queries):
        # Candidate pool: the true matching text, plus distractor texts
        # drawn from other queries' (unrelated) paired texts.
        distractor_pool = [j for j in range(n_queries) if j != i]
        distractor_idx = torch.tensor(
            distractor_pool[:n_candidates - 1]
            if len(distractor_pool) >= n_candidates - 1
            else distractor_pool
        )
        candidate_idx = torch.cat([torch.tensor([i]), distractor_idx])
        candidates = text_embeds[candidate_idx]               # (n_candidates, EMBED_DIM)

        sims = image_embeds[i] @ candidates.T                  # cosine sim (already unit-norm)
        predicted = sims.argmax().item()
        if predicted == 0:                                      # index 0 in candidate_idx is always the true match
            correct += 1

    return correct / n_queries


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------

def train_clip_style(image_encoder, text_encoder, steps=400, batch_size=32, lr=1e-2):
    log_temperature = torch.tensor(0.0, requires_grad=True)   # learned temperature, starts at exp(0)=1
    params = list(image_encoder.parameters()) + list(text_encoder.parameters()) + [log_temperature]
    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []
    for step in range(steps):
        image_raw, text_raw, _ = sample_pairs(batch_size)
        image_embeds = image_encoder(image_raw)
        text_embeds = text_encoder(text_raw)

        loss = clip_contrastive_loss(image_embeds, text_embeds, log_temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step == 0 or (step + 1) % 100 == 0:
            print(f"  step {step + 1:4d}/{steps}   contrastive loss = {loss.item():.4f}")

    return losses


# ---------------------------------------------------------------------------
# 6. Main script
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TOY CLIP: ALIGNING TWO MODALITIES WITH A CONTRASTIVE LOSS")
    print("=" * 70)
    print(f"{NUM_CATEGORIES} hidden categories, image_raw_dim={IMAGE_RAW_DIM}, "
          f"text_raw_dim={TEXT_RAW_DIM}, shared embed_dim={EMBED_DIM}")
    print("Image and text vectors for the same pair share a category latent")
    print("but are rendered through different random linear maps plus noise --")
    print("they are correlated, not identical, just like a photo and a caption.\n")

    image_encoder = Encoder(IMAGE_RAW_DIM, EMBED_DIM)
    text_encoder = Encoder(TEXT_RAW_DIM, EMBED_DIM)

    # A random-batch sanity check: with random raw features (no shared
    # category structure at all), a random encoder should retrieve at
    # roughly chance level. We instead evaluate on the REAL data-generating
    # process below, both before and after training, which is the fair,
    # honest comparison the README promises.
    n_candidates = 8
    chance_level = 1.0 / n_candidates

    print("-" * 70)
    print("RETRIEVAL ACCURACY BEFORE TRAINING (randomly initialized encoders)")
    print("-" * 70)
    acc_before = retrieval_accuracy(image_encoder, text_encoder, n_candidates=n_candidates)
    print(f"Task: given an image, pick its matching text among {n_candidates} candidates.")
    print(f"Chance level (uniform random guess): {chance_level:.3f}")
    print(f"Measured accuracy BEFORE training:    {acc_before:.3f}")
    print("-> With random encoder weights, accuracy is already somewhat above chance")
    print("   (not exactly at it): the raw image and text vectors are genuinely")
    print("   correlated by construction (same category latent), and random linear")
    print("   projections do not fully destroy that correlation. But the encoders")
    print("   have not been trained to EXPLOIT it yet -- watch this number rise")
    print("   substantially once the contrastive loss is applied below.\n")

    print("-" * 70)
    print("TRAINING WITH THE SYMMETRIC CONTRASTIVE (INFONCE) LOSS")
    print("-" * 70)
    losses = train_clip_style(image_encoder, text_encoder)
    print(f"\nLoss went from {losses[0]:.4f} (step 1) to {losses[-1]:.4f} (final step).")

    print("\n" + "-" * 70)
    print("RETRIEVAL ACCURACY AFTER TRAINING")
    print("-" * 70)
    acc_after = retrieval_accuracy(image_encoder, text_encoder, n_candidates=n_candidates)
    print(f"Measured accuracy AFTER training:     {acc_after:.3f}")
    print(f"(same task: 1 true match among {n_candidates} candidates, chance = {chance_level:.3f})")

    improvement = acc_after - acc_before
    print(f"\n-> Retrieval accuracy improved by {improvement:+.3f} after training")
    print("   (from {:.3f} to {:.3f}). The contrastive loss pulled each image's".format(acc_before, acc_after))
    print("   embedding toward its own paired text's embedding and away from")
    print("   unrelated texts in the batch -- with no labels beyond 'these two")
    print("   were paired' -- which is exactly the CLIP recipe from the README,")
    print("   just at toy scale with synthetic vectors instead of real images")
    print("   and captions.")

    # -----------------------------------------------------------------
    # Zero-shot-style demo: pick the right text for ONE new image out of
    # several candidates, printed in full for intuition.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ONE WORKED EXAMPLE: NEAREST-NEIGHBOR RETRIEVAL FOR A SINGLE IMAGE")
    print("=" * 70)
    with torch.no_grad():
        image_raw, text_raw, category_ids = sample_pairs(6)
        image_embeds = image_encoder(image_raw)
        text_embeds = text_encoder(text_raw)

        query_idx = 0
        sims = image_embeds[query_idx] @ text_embeds.T   # similarity to all 6 texts
        ranked = sims.argsort(descending=True).tolist()

        print(f"Query image belongs to hidden category {category_ids[query_idx].item()}")
        print(f"Candidate texts belong to categories:   {category_ids.tolist()}")
        print(f"Cosine similarities to each candidate:  {sims.numpy().round(3)}")
        print(f"Ranked candidate indices (best first):  {ranked}")
        top_pick_category = category_ids[ranked[0]].item()
        is_correct = ranked[0] == query_idx
        print(f"Top-ranked candidate is index {ranked[0]} (category {top_pick_category}), "
              f"true match is index {query_idx} (category {category_ids[query_idx].item()})")
        print(f"-> Top-1 retrieval on this single worked example is "
              f"{'CORRECT' if is_correct else 'INCORRECT'}.")


if __name__ == "__main__":
    main()
