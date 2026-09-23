# Standard Benchmarks

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `02-Standard-Benchmarks`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 1](../01-Evaluation-Metrics/README.md)-এ *সাধারণ-উদ্দেশ্যের* metric-গুলো নিয়ে আলোচনা হয়েছিল, যেগুলো আপনি যেকোনো (candidate, reference) জোড়ায় প্রয়োগ করতে পারেন। এই lesson-এ আলোচনা হয় *নির্দিষ্ট, মানসম্মত dataset*-গুলো নিয়ে — যেগুলোর উপর দিয়ে ফিল্ডটি আসলে সেই (ও অন্যান্য) metric-গুলো চালিয়ে মডেলগুলোর পারস্পরিক তুলনা করে। প্রতিটি model release-এর "results" টেবিলে যে সংখ্যাগুলো দেখেন (MMLU score, HellaSwag accuracy, GSM8K accuracy, HumanEval pass@1) সেগুলো এই benchmark-গুলো থেকেই আসে। এই benchmark-গুলো খুব কংক্রিট কারণে গুরুত্বপূর্ণ: "এই মডেলটি কি সত্যিই ভালো" প্রশ্নটির উত্তরের জন্য ফিল্ডে যা আছে, তার মধ্যে এগুলোই নিয়ন্ত্রিত পরীক্ষা-নিরীক্ষার (controlled experiment) সবচেয়ে কাছাকাছি — এবং এদের ভেতরে ঠিক *কীভাবে* স্কোরিং হয় তা বোঝাই একজন leaderboard সংখ্যাকে সঠিকভাবে পড়া আর ভুলভাবে পড়ার মধ্যে পার্থক্য তৈরি করে। এই lesson-টি সরাসরি [Lesson 3: LLM-as-a-Judge](../03-LLM-as-a-Judge/README.md)-এর ভিত্তিও তৈরি করে — এই ধরনের benchmark-গুলো কেবল কারণেই কাজ করে যে প্রতিটি প্রশ্নের একটি দ্ব্যর্থহীন সঠিক উত্তর থাকে যেটির বিরুদ্ধে যাচাই করা যায়; open-ended generation-এ (আমাকে একটি essay লিখো, একটি কথোপকথন চালাও) এমন কোনো gold answer থাকে না — এই ফাঁকটিই LLM-as-a-Judge তৈরি হওয়ার কারণ।

## এই পাঠে যা যা আছে

- MMLU: 57টি বিষয় জুড়ে বিস্তৃত multiple-choice knowledge
- HellaSwag: adversarially-filter করা commonsense sentence completion
- GSM8K: grade-school স্তরের math word problem, সাধারণত chain-of-thought prompting-এর সাথে ব্যবহৃত
- HumanEval: code generation, যাকে গ্রেড করা হয় আসলে unit test চালিয়ে, এবং pass@k metric
- multiple-choice benchmark-গুলো আসলে *কীভাবে* স্কোর হয় (এটা "মডেলকে A/B/C/D টাইপ করতে বলা" নয়)
- pass@k unbiased estimator সূত্র, derive এবং sanity-check সহ

## 1. MMLU: বিস্তৃত multiple-choice knowledge

MMLU (**Massive Multitask Language Understanding**, Hendrycks et al., 2021) একটি multiple-choice benchmark যা 57টি বিষয় জুড়ে বিস্তৃত — elementary math, US history, computer science, law, medicine, professional accounting এবং আরও অনেক কিছু — প্রতিটি প্রশ্নের সাথে 4টি উত্তর-বিকল্প (A/B/C/D)। এটির উদ্দেশ্য pretraining চলাকালীন অর্জিত বিস্তৃত factual ও reasoning knowledge পরীক্ষা করা, কোনো একক সংকীর্ণ দক্ষতা নয়। যেহেতু এটি এত বেশি অসম্পর্কিত বিষয় জুড়ে ছড়িয়ে, তাই একটি মাত্র MMLU accuracy সংখ্যাকে প্রায়ই একটি মডেলের সাধারণ knowledge-এর বিস্তৃতির মোটামুটি অনুমান (rough proxy) হিসেবে দেখা হয়।

## 2. HellaSwag: adversarial commonsense completion

HellaSwag (Zellers et al., 2019) একটি বাক্য বা ছোট পরিস্থিতি এবং 4টি সম্ভাব্য সমাপ্তি উপস্থাপন করে; মডেলকে বেছে নিতে হয় যে কোন সমাপ্তিটিকে একজন মানুষ প্রাকৃতিক continuation হিসেবে বিচার করতেন। এটিকে কঠিন করে তোলে *ভুল সমাপ্তিগুলো কীভাবে তৈরি করা হয়েছিল* — সেগুলো একটি language model দিয়ে generate করে তারপর **Adversarial Filtering**-এর মাধ্যমে ফিল্টার করা হয়, যাতে কেবল সেসব ভুল সমাপ্তি টিকে থাকে যেগুলোকে *আরেকটি* শক্তিশালী language model-ও বিশ্বাসযোগ্য (plausible) হিসেবে স্কোর করে (অর্থাৎ, ভুল বিকল্পগুলোকে বিশেষভাবে বেছে নেওয়া হয় মডেলকে বোকা বানানোর জন্য, অথচ মানুষের কাছে স্পষ্টতই ভুল রয়ে যায়)। এই adversarial নির্মাণই HellaSwag-কে এলোমেলোভাবে লেখা ভুল উত্তর-সম্বলিত benchmark-এর চেয়ে অর্থবহভাবে কঠিন করে তোলে — এটি সরাসরি language-model-ভিত্তিক distractor generation-এর failure mode-গুলোকে লক্ষ্য করে।

## 3. GSM8K: grade-school math word problem

GSM8K (Cobbe et al., 2021) হলো ~8,500টি grade-school স্তরের math word problem-এর একটি সেট, যেগুলোর জন্য multi-step arithmetic reasoning দরকার (যেমন "Natalia sold clips to 48 of her friends in April... how many clips did she sell altogether?")। MMLU/HellaSwag-এর মতো এটি multiple choice নয় — মডেলকে একটি মুক্ত-ফর্ম (free-form) সংখ্যাসূচক চূড়ান্ত উত্তর তৈরি করতে হয়, যা মডেলের আউটপুট টেক্সট থেকে বের করে নেওয়া হয় এবং ground-truth সংখ্যার সাথে exact match যাচাই করা হয়। যেহেতু সমস্যাগুলোর জন্য ধারাবাহিক কয়েকটি reasoning step দরকার, তাই GSM8K accuracy *মডেলটিকে কীভাবে prompt করা হয়* তার প্রতি অত্যন্ত সংবেদনশীল: সরাসরি "সংখ্যাটি আমাকে দাও" ধরনের prompt-এ মডেলগুলো ঐতিহাসিকভাবে খারাপ করত; **chain-of-thought prompting**-এ (চূড়ান্ত সংখ্যার আগে মাঝখানের reasoning step-গুলো দেখাতে বলা — দেখুন [Phase 07 Lesson 2](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md)) accuracy নাটকীয়ভাবে বেড়ে যায়। chain-of-thought prompting যে সাহায্য করে তা প্রদর্শনের জন্য GSM8K-ই আদর্শ benchmark।

## 4. HumanEval: code generation, execution দিয়ে গ্রেড করা হয়

HumanEval (Chen et al., 2021, Codex paper-টি) মডেলটিকে একটি function signature ও একটি docstring দেয় যেটি কাঙ্ক্ষিত আচরণ বর্ণনা করে, এবং function body-টি সম্পূর্ণ করতে বলে। গুরুত্বপূর্ণ বিষয় হলো, গ্রেডিং মোটেও text-overlap-ভিত্তিক **নয়** — একটি generate করা সমাধানকে আসলে *চালিয়ে* গ্রেড করা হয়, held-out unit test-এর একটি suite-এর বিরুদ্ধে। এটি Lesson 1-এর প্রতিটি দুর্বলতাকে সম্পূর্ণভাবে এড়িয়ে যায়: এখানে "surface overlap বনাম paraphrase" সমস্যাটিই নেই, কারণ দুটি সম্পূর্ণ আলাদা দেখতে implementation যেগুলো সব টেস্ট পাস করে, দুটোই সম্পূর্ণ সঠিক হিসেবে স্কোর হয়।

### pass@k

যেহেতু একটি language model থেকে sampling stochastic, তাই HumanEval শুধু "একটি greedy completion কি সঠিক ছিল" তা পরীক্ষা করে না — এটি প্রতি সমস্যায় `n`টি sample নেয় এবং জিজ্ঞেস করে: যদি আপনাকে সেই `n`টি sample-এর মধ্যে `k`টি submit করার অনুমতি দেওয়া হতো এবং *একটি* পাস করলেই হতো, তাহলে সাফল্যের probability কত? এটিকে আক্ষরিক অর্থেই `k` আকারের অনেকগুলো random subset টেনে naive ভাবে estimate করা অপচয়কারী এবং noisy। Chen et al. (2021)-এর বদলে একটি low-variance, unbiased **closed-form estimator** আছে। প্রতি সমস্যায় মোট `n`টি sample টানা হয়েছে এবং তার মধ্যে `c`টি সব unit test পাস করেছে, তখন:

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

যেখানে `C(a, b)` হলো "a choose b"। সূত্রটি সরাসরি পড়া যায়: `C(n-c, k) / C(n, k)` হলো সম্ভাবনা যে *এলোমেলোভাবে বেছে নেওয়া* `k` আকারের একটি subset-এ (টানা `n`টির মধ্যে থেকে) **শূন্য**টি সঠিক sample রয়েছে (আপনি সবগুলো `k` বেছে নিচ্ছেন কেবল `n-c`টি ভুল sample থেকে), তাই `1` বিয়োগ করলে পাওয়া যায় সম্ভাবনা যে subset-এ **কমপক্ষে একটি** সঠিক sample আছে — pass@k-এর অর্থ ঠিক এটিই। এটি প্রতি সমস্যায় হিসাব করে তারপর benchmark-এর সব সমস্যার উপর average করে চূড়ান্ত pass@k score পাওয়া যায়।

## 5. multiple-choice benchmark-গুলো আসলে কীভাবে স্কোর হয়

একটি প্রচলিত ভুল ধারণা: মডেলকে MMLU/HellaSwag-এ গ্রেড করা হয় আক্ষরিকভাবে টেক্সট generate করে এবং এটি "A", "B", "C", না "D" আউটপুট করছে কি না দেখে। আদর্শ standard evaluation harness-গুলো (যেমন EleutherAI-এর `lm-evaluation-harness`, যেটি দিয়ে বেশিরভাগ প্রকাশিত সংখ্যা তৈরি হয়) আসলে **এমন করে না**। বদলে:

1. প্রতিটি candidate উত্তর-এর জন্য প্রশ্নটিকে একটি একক ধারাবাহিক string আকারে ফর্ম্যাট করুন: `"{question}\n{answer_option_text}"` (কিছু harness বদলে `"{question}\nAnswer: {letter}"` ফর্ম্যাট করে — কিন্তু সেখানেও তারা *letter token-এর* log-probability স্কোর করে, মডেলের মুক্ত-টেক্সট আউটপুট নয়)।
2. প্রতিটি candidate-এর জন্য মডেলটিকে একবার, **teacher-forcing** মোডে চালান, এবং মডেলটি সেই নির্দিষ্ট উত্তর টেক্সটের token-গুলোতে যে log-probability বরাদ্দ করে, সেগুলো shared question prefix-এর শর্তে যোগ করুন।
3. যেকোনো candidate-এর **সবচেয়ে বেশি মোট (বা length-normalized) log-probability** আছে সেটিকেই বেছে নিন।

একটি খুব বাস্তব কারণে "মডেলকে একটি letter emit করতে বলা ও তার আউটপুট parse করা"-র চেয়ে এই log-likelihood-তুলনা পদ্ধতিটি পছন্দ করা হয়: এটি মডেলের ফরম্যাটিং-সংক্রান্ত quirks, refusal আচরণ এবং বাড়াবাড়ি (verbosity) সহ্য করতে পারে। একটি মডেল এমন একটি লম্বা-গড়ানো উত্তর তৈরি করতে পারে যা কখনোই পরিষ্কারভাবে একটি খালি letter আউটপুট করে না, অথবা নির্দিষ্ট letter-গুলোর প্রতি token-level bias থাকতে পারে যার সাথে উত্তর কী তা কোনো সম্পর্ক নেই — সম্পূর্ণ উত্তরের log-probability তুলনা করলে সরাসরি মাপা যায় প্রশ্নটি আসলে কী পরীক্ষা করার জন্য তৈরি হয়েছিল (মডেলের নিজস্ব distribution কোন সমাপ্তিটিকে পছন্দ করে), surface আউটপুট ফরম্যাটিং নির্বিশেষে। `example.py` এই স্কোরিং পদ্ধতিটিই একটি বাস্তব (যদিও ছোট) প্রশিক্ষিত মডেলের বিরুদ্ধে implement করে।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — benchmark হলো সেই মানসম্মত পরীক্ষা যেগুলোর মাধ্যমে ফিল্ডটি মডেলের তুলনা করে
2. MMLU: বিস্তৃত multiple-choice knowledge, 57টি বিষয়
3. HellaSwag: adversarial filtering, এবং কেন এটি যতটা মনে হয় তার চেয়ে কঠিন
4. GSM8K: মুক্ত-ফর্ম math উত্তর, এবং chain-of-thought-এর বিশাল প্রভাব
5. HumanEval: আসলে code চালিয়ে গ্রেড করা, এবং pass@k সূত্র derive করা
6. multiple-choice-এর আসল স্কোরিং প্রক্রিয়া: text generation নয়, log-probability তুলনা
7. `example.py`-এর ভেতরে-বাইরে — একটি ছোট প্রশিক্ষিত মডেলকে হুবহু এমনভাবে স্কোর করা যেমন একটি বাস্তব MMLU harness করত, সাথে pass@k সূত্র implement ও sanity-check
8. Recap: benchmark-এর দরকার একটি দ্ব্যর্থহীন সঠিক উত্তর — কেন open-ended generation-এর জন্য আলাদা টুল দরকার (LLM-as-a-Judge, পরবর্তী lesson) তা প্রিভিউ

## আরও পড়ুন

- Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding* (MMLU)
- Zellers et al. (2019), *HellaSwag: Can a Machine Really Finish Your Sentence?*
- Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K)
- Chen et al. (2021), *Evaluating Large Language Models Trained on Code* (HumanEval এবং pass@k estimator)
- Gao et al. (2021), *A Framework for Few-Shot Language Model Evaluation* (`lm-evaluation-harness` — log-likelihood-ভিত্তিক multiple-choice স্কোরিংয়ের de facto standard implementation)