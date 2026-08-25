# Safety, Bias and Toxicity Mitigation

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `06-Safety-Bias-and-Toxicity-Mitigation`

## Why this matters

This phase has built an increasingly sophisticated toolkit for steering a model's *behavior* toward what humans (or an AI standing in for them, per [Lesson 5](../05-RLAIF-and-Constitutional-AI/README.md)) prefer: a reward model ([Lesson 2](../02-Reward-Modeling/README.md)), RL fine-tuning against it ([Lesson 3](../03-RLHF-with-PPO/README.md)), and a simplified direct optimizer for the same objective ([Lesson 4](../04-Direct-Preference-Optimization-DPO/README.md)). This closing lesson asks a different question than "how do we optimize toward a preference signal": **how do we know where the model is failing, and how do we actually fix it once we find it** — before, during, and after those training stages, and even after deployment. It brings together three distinct but complementary practices — red-teaming (finding failures), bias measurement (quantifying systematic unfairness), and toxicity classification (filtering harmful content) — and situates each of the alignment techniques from Lessons 2-5 as one possible *mitigation* applied at one specific stage of the pipeline, rather than the only lever available.

## What this lesson covers

- Red-teaming: deliberately probing a model with adversarial inputs to find failure modes before deployment
- Bias measurement: template-based probes for detecting systematic differences in model behavior across demographic groups
- Toxicity classifiers: dedicated models that score or filter generations, separate from the LLM itself
- Where mitigation can happen across the pipeline: pretraining data filtering, RLHF/DPO steering, and inference-time guardrails
- A hands-on demonstration: a controlled bias probe that recovers a known injected bias, and a real toxicity classifier trained from scratch and evaluated on held-out data

## 1. Red-teaming

**Red-teaming** is the practice of deliberately trying to make a model fail — probing it with adversarial, edge-case, or deceptively-phrased inputs specifically designed to surface harmful, biased, or otherwise unwanted outputs, before real users ever encounter them. Unlike the benchmark-style evaluation in [Phase 08](../../Phase-08-Evaluation-of-LLMs/README.md), which typically measures performance on a fixed, representative distribution of inputs, red-teaming is adversarial by design: testers (human or, increasingly, other language models, per Ganguli et al., 2022) actively search for the prompts that break the model, including jailbreak-style prompts that try to circumvent safety training, requests phrased to obscure harmful intent, and prompts that specifically target known model weaknesses. The output of a red-teaming exercise is a set of concrete failure cases, each of which then feeds back into one of the mitigation stages in section 4 — a red-teamed failure can become a new preference-pair example for [DPO](../04-Direct-Preference-Optimization-DPO/README.md), a new constitutional principle for [Constitutional AI](../05-RLAIF-and-Constitutional-AI/README.md), or a new training example for a toxicity classifier (section 3).

## 2. Bias measurement with template-based probes

**Bias**, in this context, means a *systematic* difference in a model's outputs or associations across otherwise-equivalent inputs that differ only in a demographic attribute — for instance, a sentiment or competence score that differs when a sentence mentions one gender versus another, holding everything else about the sentence fixed. The standard measurement methodology is a **template-based probe**: construct a sentence template with a slot for a demographic term (a pronoun, a name, a nationality, etc.), fill it with each candidate term while holding every other detail of the sentence fixed and identically distributed across groups, run the model or scorer being audited on each filled-in sentence, and compare the resulting score (or generated continuation) distributions across groups. Because everything *legitimate* about the input is held constant across groups by construction, any remaining systematic difference in the output is attributable to the demographic term alone — this is exactly the logic behind established fairness audits such as Bertrand and Mullainathan's (2004) resume-callback study and NLP-specific probes such as Rudinger et al.'s (2018) Winogender schemas and Caliskan et al.'s (2017) WEAT. `example.py` runs a controlled version of this: a toy scorer with a bias of a *known* size deliberately injected into it, so the probe's detected gap can be directly checked against ground truth rather than taken on faith.

## 3. Toxicity classifiers

A **toxicity classifier** is a separate, dedicated model — usually far smaller and cheaper than the LLM it works alongside — trained to take a piece of text and output a score (or a binary label) indicating whether it is toxic, harassing, or otherwise harmful. Unlike the reward model from [Lesson 2](../02-Reward-Modeling/README.md), which learns a broad notion of "which response a human would prefer," a toxicity classifier is narrowly scoped to one specific property, trained on labeled examples of toxic and non-toxic text with ordinary supervised classification (not pairwise preference) — architecturally this can be as simple as logistic regression over bag-of-words features, or as sophisticated as a fine-tuned Transformer (e.g. Google's Perspective API, or Meta's Llama Guard). It can be deployed at two different points: **at training-data-filtering time**, scoring and removing toxic documents from a pretraining or fine-tuning corpus before the LLM ever sees them, or **at inference time**, as a guardrail that screens the LLM's own generations right before they reach a user. `example.py` implements exactly this classifier from scratch — logistic regression on bag-of-words features, trained on a tiny labeled toy dataset — and evaluates it with precision and recall on phrases it never saw during training, then reuses the identical trained model as an inference-time filter on a fresh batch of candidate outputs.

A well-known and important caveat, documented by Dixon et al. (2018): toxicity classifiers trained on real-world data can learn a *spurious* correlation between certain identity terms and toxicity (scoring a neutral sentence that happens to mention a particular identity group as more toxic than an equivalent sentence that doesn't), which is itself exactly the kind of systematic bias section 2's probe methodology is designed to catch — bias measurement and toxicity classification are complementary practices, not independent ones, and a toxicity classifier is itself a model that should be bias-probed before being trusted as a safety mechanism.

## 4. Mitigation across the pipeline

Every technique in this phase — and in this lesson — is a mitigation applied at a *specific stage* of the overall LLM pipeline, and understanding which stage each one targets clarifies why real systems use several of them together rather than relying on just one:

```
1. PRETRAINING DATA FILTERING   -- remove toxic/low-quality documents from the corpus before
                                    pretraining even begins (Phase 04 Lesson 1's data pipeline);
                                    a toxicity classifier (section 3) is a standard tool here.
2. SFT                          -- teach the target behavior via curated demonstrations
                                    (Phase 05 Lesson 4); can include explicit refusal examples.
3. RLHF / DPO STEERING          -- use preference data (human, Lesson 2, or AI, Lesson 5) to
                                    push the POLICY itself toward safer, less biased behavior;
                                    this changes what the model tends to generate, not just what
                                    slips through afterward.
4. INFERENCE-TIME GUARDRAILS    -- a separate, fast classifier or rule system (section 3) screens
                                    every generation right before a user sees it, catching failures
                                    the earlier stages missed, without requiring retraining the LLM.
5. RED-TEAMING (section 1)      -- run continuously across every stage above, to find the failures
                                    that motivate the next round of data filtering, preference data,
                                    or guardrail rules.
```

No single stage is sufficient on its own: pretraining filters can't catch every harmful pattern, RLHF/DPO steering can be circumvented by an adversarial enough prompt (the exact failure red-teaming looks for), and an inference-time guardrail treats a symptom rather than the underlying tendency in the policy — which is precisely why production systems layer several of these together, and why an honest safety story reports what each individual layer does and does not catch.

## Video Script Outline

1. Motivation — the alignment toolkit built so far steers behavior; this lesson is about finding failures and fixing them at the right stage
2. Red-teaming: adversarial probing to surface failure modes before deployment (section 1)
3. Bias measurement: the template-based probe methodology, holding everything legitimate fixed except the demographic term (section 2)
4. Toxicity classifiers: a dedicated, narrowly-scoped model, used for data filtering or as an inference-time guardrail (section 3)
5. The Dixon et al. caveat: classifiers can themselves be biased, tying bias measurement and toxicity classification together
6. The five-stage mitigation pipeline and why no single stage is sufficient alone (section 4)
7. Walkthrough of `example.py` Part 1 -- injecting a known bias into a toy scorer and confirming the probe recovers it
8. Walkthrough of `example.py` Parts 2-3 -- training a real toxicity classifier from scratch, measuring precision/recall on held-out data, and reusing it as a deployment-time guardrail

## Further Reading

- Ganguli et al. (2022), *Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned*
- Dixon et al. (2018), *Measuring and Mitigating Unintended Bias in Text Classification*
- Bertrand, Mullainathan (2004), *Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination*
- Rudinger et al. (2018), *Gender Bias in Coreference Resolution* (the Winogender schemas)
- Caliskan, Bryson, Narayanan (2017), *Semantics Derived Automatically from Language Corpora Contain Human-like Biases* (the WEAT methodology)
- Welbl et al. (2021), *Challenges in Detoxifying Language Models*
