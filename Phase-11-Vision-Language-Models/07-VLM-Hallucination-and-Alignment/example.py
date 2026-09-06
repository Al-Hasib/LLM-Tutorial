"""
VLM hallucination -- where it comes from, how POPE measures it, and what DPO
actually fixes. Everything trained and measured from scratch.

No downloads, no pretrained weights. CPU, a few minutes.

The world is built so that hallucination has a CAUSE we control:

  * 12 object types. Objects do not appear independently -- they come in
    correlated groups (a "table" scene tends to contain "chair" and "plate";
    a "street" scene tends to contain "car" and "sign"). This is the
    co-occurrence structure of real images, and it is the source of a VLM's
    language prior.
  * The model is trained on EXISTS questions ("is there a <object>?") whose
    answers are, like real VQA data, mostly YES.
  * The model's view of the scene is PARTIAL, which is the crucial ingredient.
    Each present object survives into the vision tokens only with probability
    VISIBILITY -- standing in for everything that loses evidence in a real
    VLM: a small object at 336px, a detail averaged away by token
    compression (Lesson 4), an occluded region. When the evidence for an
    object is missing, the model has nothing to fall back on but its prior,
    and that is precisely when hallucination happens.

Then we measure it the way the field does, with POPE's three negative-sampling
regimes, and we fix it with a real DPO update (the algorithm from Phase 06
Lesson 4, applied to preference pairs where the preferred answer is the
grounded one).

Run:
    python example.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

N_OBJECT_TYPES = 12
N_GROUPS = 3                      # correlated "scene types"
OBJECTS_PER_SCENE = 4
D_VISION = 24
D_MODEL = 64

YES, NO = 0, 1
OBJ0 = 2                          # object-name tokens
VOCAB = OBJ0 + N_OBJECT_TYPES

VISIBILITY = 0.6                  # chance a present object reaches the model

object_code = torch.randn(N_OBJECT_TYPES, D_VISION)

# Group g's objects are the contiguous block [4g, 4g+4). Objects from the same
# group co-occur constantly; objects from different groups rarely do.
GROUPS = [list(range(4 * g, 4 * g + 4)) for g in range(N_GROUPS)]

# Object popularity: object 0 of each group is very common, the last is rare.
POPULARITY = torch.tensor([4.0, 2.0, 1.0, 0.5] * N_GROUPS)


def sample_scenes(b):
    """Sample scenes with realistic co-occurrence: mostly one group, plus noise."""
    present = torch.zeros(b, N_OBJECT_TYPES, dtype=torch.bool)
    for i in range(b):
        g = torch.randint(0, N_GROUPS, (1,)).item()
        w = POPULARITY.clone()
        w[[o for o in range(N_OBJECT_TYPES) if o not in GROUPS[g]]] *= 0.15
        chosen = torch.multinomial(w, OBJECTS_PER_SCENE, replacement=False)
        present[i, chosen] = True
    return present


def scene_vision(present):
    """One vision token per present object -- except that each object only
    makes it into the tokens with probability VISIBILITY. Everything else is
    genuinely not in the model's input."""
    b = present.shape[0]
    vis = torch.zeros(b, OBJECTS_PER_SCENE, D_VISION)
    for i in range(b):
        objs = present[i].nonzero().squeeze(1)
        seen = objs[torch.rand(len(objs)) < VISIBILITY]
        vis[i, : len(seen)] = object_code[seen]
    return vis + 0.15 * torch.randn_like(vis)


def make_questions(present, negative_mode, yes_fraction=0.5):
    """Build EXISTS questions. `negative_mode` decides which ABSENT object we
    ask about -- this is exactly the axis POPE varies."""
    b = present.shape[0]
    q_obj, ans = [], []
    for i in range(b):
        objs = present[i].nonzero().squeeze(1).tolist()
        absent = [o for o in range(N_OBJECT_TYPES) if o not in objs]
        if torch.rand(1).item() < yes_fraction:
            q_obj.append(objs[torch.randint(0, len(objs), (1,)).item()])
            ans.append(YES)
        else:
            if negative_mode == "random":
                cand = absent
                weights = torch.ones(len(cand))
            elif negative_mode == "popular":
                cand = absent
                weights = POPULARITY[cand]
            elif negative_mode == "adversarial":
                # Absent objects that CO-OCCUR with what is present: the group
                # mates of the scene's objects. Maximum language-prior pressure.
                groupmates = set()
                for o in objs:
                    g = o // 4
                    groupmates.update(GROUPS[g])
                cand = [o for o in absent if o in groupmates] or absent
                weights = torch.ones(len(cand))
            else:
                raise ValueError(negative_mode)
            pick = cand[torch.multinomial(weights, 1).item()]
            q_obj.append(pick)
            ans.append(NO)
    return torch.tensor(q_obj), torch.tensor(ans)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class VLM(nn.Module):
    def __init__(self, n_layers=2, blind=False):
        super().__init__()
        self.blind = blind
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.projector = nn.Sequential(nn.Linear(D_VISION, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, D_MODEL))
        self.pos_embed = nn.Parameter(torch.randn(1, 16, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2)          # YES / NO

    def forward(self, vision, q_obj):
        q = self.tok_embed(q_obj + OBJ0).unsqueeze(1)
        x = q if self.blind else torch.cat([self.projector(vision), q], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, -1]))


def train_sft(steps=1200, bs=128, lr=2e-3, yes_fraction=0.8,
              negative_mode="random", blind=False):
    """Instruction tuning on EXISTS data. `yes_fraction` is the share of
    questions whose answer is YES -- real VQA data is heavily yes-skewed."""
    torch.manual_seed(1)
    model = VLM(blind=blind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        present = sample_scenes(bs)
        vision = scene_vision(present)
        q_obj, ans = make_questions(present, negative_mode, yes_fraction)
        loss = F.cross_entropy(model(vision, q_obj), ans)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def pope_eval(model, negative_mode, n=4000, bs=500):
    """POPE-style: balanced yes/no, report accuracy, yes-rate and the two error
    types separately. A model that always says yes scores 50% accuracy and a
    100% yes-rate -- the metric only means something with both numbers."""
    # Fixed evaluation data: save the training RNG stream, evaluate on the same
    # scenes every time, then restore it.
    rng_state = torch.get_rng_state()
    torch.manual_seed(1234)
    correct = tot = yes_pred = 0
    false_yes = false_no = n_neg = n_pos = 0
    for _ in range(n // bs):
        present = sample_scenes(bs)
        vision = scene_vision(present)
        q_obj, ans = make_questions(present, negative_mode, yes_fraction=0.5)
        pred = model(vision, q_obj).argmax(-1)
        correct += (pred == ans).sum().item()
        tot += bs
        yes_pred += (pred == YES).sum().item()
        false_yes += ((pred == YES) & (ans == NO)).sum().item()
        false_no += ((pred == NO) & (ans == YES)).sum().item()
        n_neg += (ans == NO).sum().item()
        n_pos += (ans == YES).sum().item()
    torch.set_rng_state(rng_state)
    return {
        "acc": correct / tot,
        "yes_rate": yes_pred / tot,
        "hallucination_rate": false_yes / max(n_neg, 1),   # said YES about an absent object
        "miss_rate": false_no / max(n_pos, 1),
    }


print("=" * 80)
print("1. WHERE HALLUCINATION COMES FROM")
print("=" * 80)
print("Three models, all trained on the same EXISTS task:")
print()

blind = train_sft(blind=True, yes_fraction=0.5)
skewed = train_sft(blind=False, yes_fraction=0.8)
balanced = train_sft(blind=False, yes_fraction=0.5)

models = [("BLIND (no vision at all)", blind),
          ("grounded, 80% YES training data", skewed),
          ("grounded, balanced training data", balanced)]

print(f"{'model':<36} {'accuracy':>9} {'yes-rate':>9} {'halluc.':>9} {'misses':>8}")
for name, m in models:
    r = pope_eval(m, "random")
    print(f"{name:<36} {r['acc']:>8.1%} {r['yes_rate']:>8.1%}"
          f" {r['hallucination_rate']:>8.1%} {r['miss_rate']:>7.1%}")
print()
print("The BLIND model cannot possibly know what is in the scene, and still")
print("beats 50%. That is the language prior doing the work: objects co-occur,")
print("so 'is there a chair?' is answerable at better than chance from the")
print("question alone. Every VLM sits somewhere on the spectrum between that")
print("model and a fully grounded one, and a benchmark that does not include")
print("this baseline cannot tell you where.")
print()
print("The 80%-YES model has learned the training distribution's prior on top of")
print("the image: it says yes too often, and its errors are almost all")
print("hallucinations rather than misses. Real VQA instruction data is skewed")
print("this way, which is a large part of why real VLMs over-affirm.")
print()


# ---------------------------------------------------------------------------
# 2. POPE's three negative-sampling regimes
# ---------------------------------------------------------------------------

print("=" * 80)
print("2. THE SAME MODEL, THREE WAYS OF CHOOSING THE 'NO' QUESTIONS (POPE)")
print("=" * 80)
print("Identical model, identical images, balanced yes/no. Only the choice of")
print("WHICH absent object to ask about changes.")
print()
print(f"{'negatives are':<14}{'chosen from':<40}{'acc':>7}{'halluc.':>10}")
for mode, desc in [("random", "any absent object"),
                   ("popular", "absent objects that are common overall"),
                   ("adversarial", "absent objects that co-occur with the scene")]:
    r = pope_eval(skewed, mode)
    print(f"{mode:<14}{desc:<40}{r['acc']:>7.1%}{r['hallucination_rate']:>10.1%}")
print()
print("One model, one set of images, three different scores -- and 'random',")
print("the setting a paper would quote if it quoted only one, is the easiest.")
print("Both harder settings work by aiming at the prior rather than at the")
print("model's eyesight: 'popular' asks about objects that are common in")
print("general, 'adversarial' about objects that usually accompany what IS in")
print("this scene. Which of the two bites hardest depends on which prior")
print("dominates in the data -- here overall frequency wins -- and that is")
print("exactly why POPE defines all three and why they should be quoted")
print("together. A hallucination number without its negative-sampling regime is")
print("not a number.")
print()


# ---------------------------------------------------------------------------
# 3. Fixing it with DPO on grounded preference pairs
# ---------------------------------------------------------------------------

def dpo_finetune(model, steps=400, bs=128, lr=3e-4, beta=0.2,
                 negative_mode="adversarial"):
    """Direct Preference Optimization (Phase 06 Lesson 4), applied to
    hallucination: the CHOSEN response is the grounded answer, the REJECTED
    response is the hallucinated one, for the same image and question.

    This is exactly the RLHF-V / POVID recipe in miniature: preference pairs
    that differ ONLY in whether the answer is faithful to the image.
    """
    policy = copy.deepcopy(model)
    ref = copy.deepcopy(model)                       # frozen reference
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for _ in range(steps):
        present = sample_scenes(bs)
        vision = scene_vision(present)
        q_obj, ans = make_questions(present, negative_mode, yes_fraction=0.5)
        chosen = ans                                  # the grounded answer
        rejected = 1 - ans                            # the hallucinated one

        logp = F.log_softmax(policy(vision, q_obj), dim=-1)
        with torch.no_grad():
            ref_logp = F.log_softmax(ref(vision, q_obj), dim=-1)
        idx = torch.arange(bs)
        pi_c, pi_r = logp[idx, chosen], logp[idx, rejected]
        rf_c, rf_r = ref_logp[idx, chosen], ref_logp[idx, rejected]
        loss = -F.logsigmoid(beta * ((pi_c - rf_c) - (pi_r - rf_r))).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy


print("=" * 80)
print("3. DPO ON GROUNDED PREFERENCE PAIRS")
print("=" * 80)
print("Preference pairs where the chosen answer is the grounded one and the")
print("rejected answer is the hallucinated one, for the SAME image and question.")
print()
aligned = dpo_finetune(skewed)
print(f"{'':<26} {'accuracy':>9} {'yes-rate':>9} {'hallucination rate':>20}")
for mode in ["random", "popular", "adversarial"]:
    before = pope_eval(skewed, mode)
    after = pope_eval(aligned, mode)
    print(f"{mode + ', before DPO':<26} {before['acc']:>8.1%} {before['yes_rate']:>8.1%}"
          f" {before['hallucination_rate']:>19.1%}")
    print(f"{mode + ', after  DPO':<26} {after['acc']:>8.1%} {after['yes_rate']:>8.1%}"
          f" {after['hallucination_rate']:>19.1%}")
print()
print("DPO moves the model toward the grounded answer without any reward model,")
print("exactly as in Phase 06 Lesson 4 -- the only multimodal change is where")
print("the preference pairs come from. Note the yes-rate moving toward 50%: what")
print("the update is really doing is removing the prior the SFT data installed.")
print()
print("The honest caveat this simulation makes visible: DPO is trained on")
print("adversarial pairs here, and it is on adversarial questions that it helps")
print("most. Preference data fixes the failure mode it was collected for. A")
print("preference set that only contains obvious hallucinations will not fix")
print("subtle ones, which is the multimodal restatement of Phase 06's point that")
print("an alignment method is only as good as the preferences behind it.")
