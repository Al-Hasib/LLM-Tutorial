# Hallucination and Factuality Evaluation

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `04-Hallucination-and-Factuality-Evaluation`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর এ পর্যন্ত যত metric আছে, সবগুলোই ধরে নেয় যে আপনি ইতিমধ্যেই জানেন একটি generate করা বাক্য *সাবলীল* কি না — [Lesson 1](../01-Evaluation-Metrics/README.md)-এর BLEU/ROUGE, [Lesson 2](../02-Standard-Benchmarks/README.md)-এর benchmark accuracy এবং [Lesson 3](../03-LLM-as-a-Judge/README.md)-এর judge score — সবগুলোই গ্রেড করে *টেক্সটটি কতটা ভালো*, এই ভিত্তিতে যে এটি একটি দাবি (claim) হিসেবে পড়ার উদ্দেশ্যে। এদের কোনোটিই সেই মৌলিক প্রশ্নটি জিজ্ঞেস করে না যেটি এই lesson নিয়ে: **দাবিটি কি সত্যিই সত্য?** একটি response সম্পূর্ণ সাবলীল, সুসংগঠিত এবং এমনকি pairwise তুলনায় অন্য একটি response-এর বিরুদ্ধে জয়ী হতে পারে, অথচ মিথ্যা কিছু দাবি করতে পারে — এটিই **hallucination**, এবং এটি একটি LLM-এর হতে পারে এমন সবচেয়ে গুরুত্বপূর্ণ failure mode-গুলোর একটি, কারণ এটি বিশেষভাবে সেই failure mode যেটি সবচেয়ে বিশ্বস্ত দেখায়। এই lesson-টি Phase 06 Lesson 1-এর [HHH alignment framing](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing)-এর "**সৎ**" (honest) অংশের সরাসরি ঠিকানাও — সেখানে ইতিমধ্যেই চিহ্নিত করা হয়েছিল যে এই নির্দিষ্ট lesson-টিই হবে যেখানে honesty সমস্যাটি একটি পরিমাপ-টুল পায়। এই lesson-এর 중심ের কৌশলটি — উৎস কোনো claim-কে *entail* করে কি না তা পরীক্ষা করা — [Lesson 5](../05-Human-Evaluation-Methodologies/README.md)-এ rubric অক্ষ "accuracy" হিসেবে সরাসরি আবার দেখা দেয়, যেখানে মানব annotator-রা একই entailment রায় ম্যানুয়ালি করে।

## এই পাঠে যা যা আছে

- hallucination সংজ্ঞায়িত করা: সাবলীল আউটপুট যা বাস্তবে ভুল বা অসমর্থিত
- Intrinsic hallucination: আউটপুট দেওয়া source/context-এর সাথে সাংঘর্ষিক
- Extrinsic hallucination: আউটপুট যাচাই-অযোগ্য বা সম্ভবত বানানো, যাচাই করার মতো কোনো source-্ই নেই
- Retrieval-ভিত্তিক যাচাইকরণ: একটি বিশ্বস্ত বাহ্যিক source-এর বিরুদ্ধে claim পরীক্ষা করা
- NLI/entailment-ভিত্তিক পরীক্ষা: entailment, contradiction বা neutral — একটি 3-মুখী শ্রেণিবিভাগ
- একটি শূন্য থেকে তৈরি toy factuality checker, synthetic (source, claim, label) triples-এ প্রশিক্ষিত এবং একটি toy generated summary-তে প্রয়োগ করা

## 1. hallucination সংজ্ঞায়িত করা

**Hallucination** হলো এমন টেক্সট যা সাবলীল ও আত্মবিশ্বাসী কিন্তু বাস্তবে ভুল বা উপলব্ধ কোনো প্রমাণ দ্বারা অসমর্থিত। শব্দটি ইচ্ছাকৃতভাবে মনোবিজ্ঞান/উপলব্ধি থেকে ধার করা: ভিজ্যুয়াল hallucination-এর মতো, আউটপুটটি *দেখতে* একটি স্বাভাবিক উপলব্ধির মতো (এখানে, একটি স্বাভাবিক, সুগঠিত বাক্য) কিন্তু বাস্তব কোনো কিছুর সাথে মেলে না। এটিই hallucination-কে এমনভাবে বিপজ্জনক করে তোলে যেভাবে একটি এলোমেলো বা অ-ব্যাকরণগত আউটপুট নয় — একটি hallucinated বাক্যে কোনো কিছু ভুল হওয়ার surface সংকেত থাকে না। এটি হুবহু একটি সঠিক বাক্যের মতোই পড়া যায়।

## 2. Intrinsic বনাম extrinsic hallucination

Ji et al. (2023) *hallucinated claim-টিকে কীসের বিরুদ্ধে যাচাই করা হচ্ছে* তার ভিত্তিতে একটি আদর্শ, দরকারি পার্থক্য আঁকেন:

- **Intrinsic hallucination**: আউটপুট মডেলটিকে দেওয়া কোনো source document বা context-কে সরাসরি **contradict** করে। উদাহরণ: একটি source passage বলছে একটি কোম্পানি 1998 সালে প্রতিষ্ঠিত হয়েছিল, একটি summary যদি বলে এটি 2005 সালে প্রতিষ্ঠিত হয়েছিল, তাহলে সেটি intrinsically hallucinated — মডেলের সামনে থাকা exact টেক্সটটির বিরুদ্ধে এই দ্বন্দ্ব যাচাই করা যায়। দুটির মধ্যে এটিই স্বয়ংক্রিয়ভাবে ধরা পড়া সহজ, কারণ আপনার কাছে তুলনার জন্য একটি কংক্রিট source থাকে।
- **Extrinsic hallucination**: আউটপুট এমন একটি claim করে যা দেওয়া source **নিশ্চিতও করে না, contradict-ও করে না** — এটি উপলব্ধ context-এ মোটেই সম্বোধিত নয়, তাই সেই context থেকে এটি মোটেও যাচাই করা যায় না। এটি pretraining থেকে মডেল সঠিকভাবে মনে রেখেছে এমন একটি অস্পষ্ট, সত্য ঘটনা হতে পারে, অথবা সম্পূর্ণ বানানো হতে পারে; source-এর দৃষ্টিকোণ থেকে এই দুটি ক্ষেত্র অভিন্ন দেখায়। এটিই কঠিন ক্ষেত্র, কারণ এটি ধরতে হলে তাৎক্ষণিক context-এর বাইরে গিয়ে কিছু বাহ্যিক, বিশ্বস্ত সত্য-উৎসে (একটি encyclopedia, একটি database, একটি search engine) পৌঁছাতে হয় — অথবা সেটির অনুপস্থিতিতে, স্বয়ংক্রিয়ভাবে মোটেও সমাধান করা যায় না।

## 3. স্বয়ংক্রিয় factuality checking: দুটি পরিপূরক পদ্ধতি

**Retrieval-ভিত্তিক verification।** generate করা আউটপুটকে আলাদা আলাদা factual claim-এ ভেঙে ফেলুন, প্রতিটি claim-এর জন্য একটি বিশ্বস্ত knowledge source থেকে সবচেয়ে প্রাসঙ্গিক passage(s) retrieve করুন (একটি search engine, একটি নির্দিষ্ট reference corpus, একটি knowledge base), এবং claim-টিকে যা retrieve হয়েছে তার বিরুদ্ধে যাচাই করুন। এটি *extrinsic* hallucination-এর সাথে স্বাভাবিকভাবে মানানসই — যেহেতু prompt-এ যাচাই করার কোনো source নেই, আপনাকে নিজেই একটি সংগ্রহ করতে হয়। FActScore (Min et al., 2023)-এর মতো fact-checking pipeline-গুলোর পেছনেও এই পদ্ধতিই, যেখানে একটি দীর্ঘ generation-কে "atomic fact"-এ বিশ্লেষণ করা হয় এবং একটি একক precision score-এ একত্রিত করার আগে প্রতিটি স্বাধীনভাবে যাচাই করা হয়।

**NLI/entailment-ভিত্তিক checking।** যখন একটি source document ইতিমধ্যেই বিদ্যমান (summarization/RAG-এর ক্ষেত্রে — *intrinsic* hallucination), তখন নতুন কিছু retrieve করার দরকার নেই: আপনি সরাসরি জিজ্ঞেস করতে পারেন, একটি নির্দিষ্ট `(source, claim)` pair-এর জন্য, source কি claim-টিকে **entail** করে? এটি factuality checking-কে **Natural Language Inference (NLI)** হিসেবে পুনর্গঠন করে — একটি ক্লাসিক NLP কাজ, তিনটি label সহ:

```
label(source, claim) = ENTAILMENT     if the source logically supports the claim being true
                        CONTRADICTION  if the source implies the claim is false
                        NEUTRAL        if the source says nothing that confirms or denies the claim
```

এই 3-মুখী কাঠামোই NLI-কে সাধারণ word-overlap পরীক্ষার চেয়ে অনেক তীক্ষ্ণ টুল বানায়: একটি claim source-এর কোনো শব্দই পুনঃব্যবহার না করে জোরালোভাবে entailed হতে পারে (বলা বিবৃত ঘটনা থেকে একটি বৈধ paraphrase বা বৈধ inference), এবং একটি claim source-এর অনেক শব্দ পুনঃব্যবহার করেও তাকে contradict করতে পারে (যেমন একটি মূল বিবরণকে negation করা, একটি সংখ্যা বদলানো, কে-কাকে-কী-করল তা উল্টে দেওয়া) — [Lesson 1&#39;s BLEU/ROUGE](../01-Evaluation-Metrics/README.md#5-the-shared-weakness-surface-overlap-is-not-meaning)-এর মতো একই lexical-overlap অন্ধবিন্দু, শুধু গুণমানের রায়ের বদলে সত্যের রায়ে প্রয়োগ করা। source-এর সাথে একটি generated summary বাক্যের সম্পর্ক যখনই **CONTRADICTION** হয় তখনই সেটিকে (intrinsic) hallucination হিসেবে ফ্ল্যাগ করা হয়, এবং যখন **NEUTRAL** হয় তখন অসমর্থিত/সন্দেহজনক হিসেবে ধরা হয় — শুধুমাত্র **ENTAILMENT**-কেই একটি যাচাইকৃত, বিশ্বস্ত claim হিসেবে গণ্য করা হয়।

## 4. কেন এটি সমস্যাটি সম্পূর্ণ সমাধান করে না

একটি স্বয়ংক্রিয় factuality checker নিজেও কেবল আর একটি মডেল, এবং এই phase-এর অন্য সবকিছুর মতো একই সীমা উত্তরাধিকার সূত্রে পায়: এটি বোকা বানানো যায়, এটিকে কোনো ground truth-এর বিরুদ্ধে মূল্যায়ন করতে হয় (মানব-লেবেলকৃত hallucination-এর বিরুদ্ধে precision/recall, হুবহু যেমন `example.py` হিসাব করে), এবং একটি NLI checker-এর NEUTRAL রায় "সত্য কিন্তু এই source-এ আচ্ছাদিত নয়" আর "বানানো" এই দুটি আলাদা করে না — সেই অস্পষ্টতাই উপরে আঁকা intrinsic/extrinsic রেখা, এবং কোনো একক-source entailment check এটি মীমাংসা করতে পারে না। বাস্তবে, প্রোডাকশন factuality pipeline দুটি পদ্ধতিই একত্রিত করে: প্রদত্ত যেকোনো context-এর বিরুদ্ধে NLI-শৈলীর entailment checking, এবং context-এ আচ্ছাদিত নয় এমন claim-গুলোর জন্য একটি বাহ্যিক source-এর বিরুদ্ধে retrieval-ভিত্তিক verification-এর সমর্থন।

## 5. `example.py` কী প্রদর্শন করে

bag-of-words feature ব্যবহার করে synthetic `(source, claim, label)` triples-এ একটি ছোট MLP entailment classifier শূন্য থেকে প্রশিক্ষিত হয়, তারপর একটি toy generated "summary"-র প্রতিটি বাক্য একটি toy source passage-এর বিরুদ্ধে যাচাই করতে ব্যবহৃত হয় — কিছু বাক্য ইচ্ছাকৃতভাবে source-কে contradict করতে বা অসমর্থিত claim ঢোকানোর জন্য লেখা — এবং চেকারের ফ্ল্যাগগুলো precision ও recall দিয়ে পরিচিত ground truth-এর বিরুদ্ধে স্কোর করা হয়।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — এ পর্যন্ত যত metric আছে সবগুলো গুণমান বা পছন্দ গ্রেড করে, কেউই সত্য পরীক্ষা করে না; এই lesson সেই ফাঁকটিই পূরণ করে
2. hallucination সংজ্ঞায়িত: আত্মবিশ্বাসের সাথে ভুল সাবলীল আউটপুট
3. Intrinsic বনাম extrinsic hallucination, এবং কেন extrinsic কঠিন ক্ষেত্র
4. কোনো দেওয়া source নেই এমন claim-এর জন্য retrieval-ভিত্তিক verification
5. NLI-ভিত্তিক checking: entailment / contradiction / neutral — এবং কেন 3-মুখী পদ্ধতি সাধারণ overlap-কে হারায়
6. কেন NEUTRAL সত্যিই অস্পষ্ট, এবং কেন বাস্তব pipeline দুটি পদ্ধতিই একত্রিত করে
7. `example.py`-এর ভেতরে-বাইরে — toy entailment classifier প্রশিক্ষণ, injected hallucination-সহ একটি summary-তে প্রয়োগ, precision/recall সংখ্যা পড়া
8. Recap + [Lesson 5](../05-Human-Evaluation-Methodologies/README.md)-এর দিকে নির্দেশনা, যেখানে একই entailment রায় একটি মানব rubric অক্ষ ("accuracy") হয়ে ওঠে

## আরও পড়ুন

- Ji, Lee, Frieske et al. (2023), *Survey of Hallucination in Natural Language Generation* — এই lesson-এ ব্যবহৃত আদর্শ intrinsic/extrinsic ট্যাক্সোনমি
- Bowman, Angeli, Potts, Manning (2015), *A Large Annotated Corpus for Learning Natural Language Inference* (SNLI — entailment/contradiction/neutral 3-মুখী কাজের কাঠামোর উৎপত্তি)
- Min, Krishna, Lyu et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
- Maynez, Narayan, Bohnet, McDonald (2020), *On Faithfulness and Factuality in Abstractive Summarization* — নির্দিষ্টভাবে summarization-এর জন্য intrinsic/extrinsic পার্থক্য প্রতিষ্ঠা করা paper
- Kryscinski, McCann, Xiong, Socher (2020), *Evaluating the Factual Consistency of Abstractive Text Summarization* (FactCC — summary-এর জন্য একটি NLI-শৈলীর factual consistency checker)