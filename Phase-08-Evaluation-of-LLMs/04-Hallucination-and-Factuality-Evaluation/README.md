# Hallucination and Factuality Evaluation

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `04-Hallucination-and-Factuality-Evaluation`

## Why this matters

Every metric in this phase so far assumes you already have a way to tell whether a generated sentence is *fluent* — [Lesson 1](../01-Evaluation-Metrics/README.md)'s BLEU/ROUGE, [Lesson 2](../02-Standard-Benchmarks/README.md)'s benchmark accuracy, and [Lesson 3](../03-LLM-as-a-Judge/README.md)'s judge scores all grade *how good the text is*, given that it's meant to be read as a claim. None of them ask the more basic question this lesson is about: **is the claim actually true?** A response can be perfectly fluent, well-organized, and even win a pairwise comparison against another response, while stating something false — this is **hallucination**, and it is one of the most consequential failure modes an LLM can have, because it is specifically the failure mode that looks the most trustworthy. This lesson is also the direct home of the "**honest**" third of the HHH alignment framing from [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing), which already flagged this exact lesson as where the honesty problem gets a measurement tool. The technique at the center of this lesson — checking whether a source *entails* a claim — also reappears directly as the rubric axis "accuracy" in [Lesson 5](../05-Human-Evaluation-Methodologies/README.md), where human annotators do the same entailment judgment manually.

## What this lesson covers

- Defining hallucination: fluent output that is factually incorrect or unsupported
- Intrinsic hallucination: the output contradicts a given source/context
- Extrinsic hallucination: the output is unverifiable or possibly fabricated, with no source to check against at all
- Retrieval-based verification: checking a claim against a trusted external source
- NLI/entailment-based checking: a 3-way classification of entailment, contradiction, or neutral
- A from-scratch toy factuality checker, trained on synthetic (source, claim, label) triples and applied to a toy generated summary

## 1. Defining hallucination

**Hallucination** is text that is fluent and confident but factually incorrect or unsupported by any available evidence. The word is deliberately borrowed from psychology/perception: like a visual hallucination, the output *looks* like a normal perception (here, a normal, well-formed sentence) but doesn't correspond to anything real. This is what makes hallucination dangerous in a way that a garbled or ungrammatical output isn't — a hallucinated sentence gives no surface signal that anything is wrong. It reads exactly like a correct one.

## 2. Intrinsic vs. extrinsic hallucination

Ji et al. (2023) draw a standard, useful distinction based on *what the hallucinated claim is being checked against*:

- **Intrinsic hallucination**: the output directly **contradicts** a source document or context the model was given. Example: given a source passage stating a company was founded in 1998, a summary that says it was founded in 2005 is intrinsically hallucinated — the contradiction is checkable against the exact text the model had in front of it. This is the easier of the two to catch automatically, because you have a concrete source to compare against.
- **Extrinsic hallucination**: the output makes a claim that the given source **neither confirms nor contradicts** — it's simply not addressed by the available context, so it can't be verified from that context at all. It might be an obscure, true fact the model recalled correctly from pretraining, or it might be fabricated outright; from the source's perspective these two cases look identical. This is the harder case, because catching it requires reaching outside the immediate context to some external, trusted source of truth (an encyclopedia, a database, a search engine) — or, absent that, cannot be automatically resolved at all.

## 3. Automated factuality checking: two complementary approaches

**Retrieval-based verification.** Break the generated output into individual factual claims, retrieve the most relevant passage(s) from a trusted knowledge source for each claim (a search engine, a fixed reference corpus, a knowledge base), and check the claim against what's retrieved. This is the natural fit for *extrinsic* hallucination — since there's no source in the prompt to check against, you have to go get one. It's also the approach behind fact-checking pipelines like FActScore (Min et al., 2023), which decompose a long generation into "atomic facts" and verify each one independently before aggregating into a single precision score.

**NLI/entailment-based checking.** When a source document already exists (the summarization/RAG case — *intrinsic* hallucination), you don't need to retrieve anything new: you can directly ask, for a given `(source, claim)` pair, does the source **entail** the claim? This reframes factuality checking as **Natural Language Inference (NLI)**, a classic NLP task with three labels:

```
label(source, claim) = ENTAILMENT     if the source logically supports the claim being true
                        CONTRADICTION  if the source implies the claim is false
                        NEUTRAL        if the source says nothing that confirms or denies the claim
```

This 3-way framing is exactly what makes NLI a sharper tool than simple word-overlap checks: a claim can reuse none of the source's words and still be strongly entailed (a valid paraphrase or a valid inference from stated facts), and a claim can reuse many of the source's words while still contradicting it (e.g., negating one key detail, swapping a number, reversing who-did-what-to-whom) — the same lexical-overlap blind spot as [Lesson 1's BLEU/ROUGE](../01-Evaluation-Metrics/README.md#5-the-shared-weakness-surface-overlap-is-not-meaning), just applied to a truth judgment instead of a quality judgment. A generated summary sentence is flagged as an (intrinsic) hallucination whenever the source's relationship to it is **CONTRADICTION**, and treated as unsupported/suspicious whenever it's **NEUTRAL** — only **ENTAILMENT** counts as a verified, faithful claim.

## 4. Why this doesn't fully solve the problem

An automated factuality checker is itself just another model, and inherits the same limits as everything else in this phase: it can be fooled, it needs to be evaluated against some ground truth (precision/recall against human-labeled hallucinations, exactly as `example.py` computes), and a NEUTRAL verdict from an NLI checker doesn't distinguish "true but not covered by this source" from "fabricated" — that ambiguity is the intrinsic/extrinsic line drawn above, and no single-source entailment check can resolve it. In practice, production factuality pipelines combine both approaches: NLI-style entailment checking against any provided context, backed by retrieval-based verification against an external source for claims the context doesn't cover.

## 5. What `example.py` demonstrates

A small MLP entailment classifier is trained from scratch on synthetic `(source, claim, label)` triples using bag-of-words features, then used to check every sentence of a toy generated "summary" against a toy source passage — with a few sentences deliberately written to contradict the source or introduce unsupported claims — and the checker's flags are scored against the known ground truth with precision and recall.

## Video Script Outline

1. Motivation — every metric so far grades quality or preference, none of them check truth; that's the gap this lesson fills
2. Defining hallucination: fluent output that's confidently wrong
3. Intrinsic vs. extrinsic hallucination, and why extrinsic is the harder case
4. Retrieval-based verification for claims with no given source
5. NLI-based checking: entailment / contradiction / neutral, and why 3-way beats simple overlap
6. Why NEUTRAL is genuinely ambiguous, and why real pipelines combine both approaches
7. Walkthrough of `example.py` — train the toy entailment classifier, apply it to a summary with injected hallucinations, read the precision/recall numbers
8. Recap + pointer to [Lesson 5](../05-Human-Evaluation-Methodologies/README.md), where the same entailment judgment becomes a human rubric axis ("accuracy")

## Further Reading

- Ji, Lee, Frieske et al. (2023), *Survey of Hallucination in Natural Language Generation* (the standard intrinsic/extrinsic taxonomy used in this lesson)
- Bowman, Angeli, Potts, Manning (2015), *A Large Annotated Corpus for Learning Natural Language Inference* (SNLI — the origin of the entailment/contradiction/neutral 3-way task framing)
- Min, Krishna, Lyu et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
- Maynez, Narayan, Bohnet, McDonald (2020), *On Faithfulness and Factuality in Abstractive Summarization* (the paper that established the intrinsic/extrinsic distinction for summarization specifically)
- Kryscinski, McCann, Xiong, Socher (2020), *Evaluating the Factual Consistency of Abstractive Text Summarization* (FactCC — an NLI-style factual consistency checker for summaries)
