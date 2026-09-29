# Prompting Basics: Zero-Shot এবং Few-Shot

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `01-Prompting-Basics-Zero-Few-Shot`

## কেন এটি গুরুত্বপূর্ণ

এখন পর্যন্ত প্রতিটি lesson ছিল *model-টিকে বদলানো* নিয়ে — architecture ([Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md)), pretraining objective ([Phase 04](../../Phase-04-Pretraining-LLMs/README.md)), weights ([Phase 05](../../Phase-05-Finetuning-LLMs/README.md)), অথবা এর preferences ([Phase 06](../../Phase-06-Alignment-and-RLHF/README.md))। এই phase ঠিক উল্টো বিষয় নিয়ে: weights এখন সম্পূর্ণ **frozen**, এবং হাতে থাকা একমাত্র lever হলো *prompt-এ আপনি কোন text রাখছেন*। [Phase 03 Lesson 1 §3](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#3-gpt-3-in-context-learning-emerges) ইতিমধ্যে মূল ফলাফলটি পরিচয় করিয়েছে — GPT-3 prompt-এ সরাসরি রাখা কয়েকটি example থেকেই একেবারে নতুন একটি task করতে পারে, কোনো gradient update ছাড়াই। এই lesson সেই mechanism-কে এমন একটি task-এ মূর্ত ও পরিমাপযোগ্য করে তোলে যা শুরু থেকে শেষ পর্যন্ত train ও probe করার মতো যথেষ্ট ছোট, এবং সেই শব্দভান্ডার (zero-shot, few-shot, in-context examples, prompt sensitivity) প্রতিষ্ঠা করে যার উপর এই phase-এর পরের প্রতিটি lesson — chain-of-thought ([Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md)), tree-of-thought ও ReAct ([Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md)), automatic prompt optimization ([Lesson 4](../04-Automatic-Prompt-Optimization/README.md)), এবং structured output ([Lesson 5](../05-Structured-Output-and-Function-Calling/README.md)) — সরাসরি দাঁড়িয়ে আছে।

## এই lesson যা কভার করে

- Zero-shot বনাম few-shot prompting: "shot" আসলে কী বোঝায়
- In-context learning (ICL): এমন এক ধরনের learning যা কোনো weight-ই update করে না
- কেন ICL কাজ করার আগে একটি model-কে *পরিবর্তনশীল* task/episode-এর উপর train করতে হয়
- `example.py`: hidden-parameter function-এর একটি family-র উপর একটি ক্ষুদ্র decoder-only Transformer train করা, তারপর অদেখা parameter-এ প্রকৃত in-context generalization পরীক্ষা করা
- In-context example-এর সংখ্যার সাথে accuracy কীভাবে বাড়ে
- Example-এর ক্রমের প্রতি prompt sensitivity (Zhao et al., 2021, "Calibrate Before Use")

## 1. Zero-shot বনাম few-shot: "shot" মানে কী

একটি **shot** হলো আসল query-র আগে prompt-এ রাখা একটি input/output example। **Zero-shot** prompting model-কে শুধু একটি instruction বা খালি query দেয় — কোনো worked example নেই। **Few-shot** prompting আসল প্রশ্নের আগে সরাসরি context window-এ `k` টি example pair জুড়ে দেয় (বাস্তব LLM-এর জন্য `k` সাধারণত 1-32), যেমন:

```
2>4,3>5,0>2,x_query>
```

এটি একটি 3-shot prompt: তিনটি example pair, তারপর একটি query যার উত্তর model-কে সেগুলোর সাথে pattern-matching করে দিতে হবে। গুরুত্বপূর্ণ বিষয় হলো, zero-shot ও few-shot-এর মধ্যে *model*-এর কিছুই বদলায় না — একমাত্র পার্থক্য context window-এর text-এ। একটি discipline হিসেবে prompt engineering-এর পুরো ভিত্তিই এটি: আচরণকে শুধু input দিয়েই নিয়ন্ত্রণ করা হচ্ছে।

## 2. In-context learning: weights frozen রেখে learning

GPT-2 zero-shot task transfer দেখিয়েছিল; GPT-3 দেখিয়েছিল যে prompt-এ কয়েকটি example সাজিয়ে দিলে এটি নাটকীয়ভাবে বেশি নির্ভরযোগ্য হয় — Brown et al. (2020) এই ঘটনার নাম দেন **in-context learning (ICL)**। ICL-কে যা অদ্ভুত করে, এবং যা নিয়ে ভাবার মতো, তা হলো এটি দেখতে learning-এর মতো (বেশি example দিলে performance বাড়ে) কিন্তু এতে **শূন্য gradient update** — কোনো backward pass নেই, কোনো optimizer step নেই, কোনো weight-এ কিছু লেখা হয় না। যে "learning"-ই ঘটুক না কেন, তা সম্পূর্ণভাবে forward pass-এ সম্পন্ন একটি গণনা, যেখানে in-context example-গুলো এমন data হিসেবে ব্যবহৃত হয় যা attention mechanism পড়ে এবং যার উপর condition করে।

এটি তৎক্ষণাৎ সেই প্রশ্নটি তোলে যার সরাসরি উত্তর দেওয়ার জন্যই `example.py` তৈরি: weights যদি কখনোই না বদলায়, তাহলে একটি *প্রকৃত নতুন* task সমাধানের ক্ষমতা আসে কোথা থেকে? উত্তর হলো, weights-গুলো সম্পর্কিত task-এর একটি বড় **distribution**-এর উপর pretrain করা হয়েছিল, এবং network-টি কোনো একক task-এর answer key মুখস্থ না করে একটি সাধারণ-উদ্দেশ্যের *algorithm* শিখেছে — "কিছু example পড়ো, তাদের input ও output-কে যুক্ত করা নিয়মটি অনুমান করো, সেই নিয়ম নতুন query-তে প্রয়োগ করো"। Inference-এর সময় ICL হলো একই family-র একটি নতুন instance-এর উপর সেই pretrained algorithm-এর চলা।

## 3. `example.py`: train ও যাচাই করার মতো যথেষ্ট ছোট একটি task family

বাস্তব GPT-3-এর in-context learning শুরু থেকে শেষ পর্যন্ত পরীক্ষা করা যায় না — mechanism যাচাই করতে কেউ GPT-3 আবার train করতে পারে না। তাই `example.py` এর বদলে একটি **ছোট, সৎ analogue** তৈরি করে, section 2-এর recipe হুবহু অনুসরণ করে:

- Task family হলো `y = (x + k) mod M`, যেখানে `M = 6` টি symbol এবং একটি hidden shift `k`।
- প্রতিটি training example একটি নতুন **episode**: একটি random `k` নেওয়া হয়, সেই `k`-এর অধীনে তৈরি কয়েকটি `(x, y)` pair in-context example হিসেবে দেখানো হয়, এবং model-কে আরও একটি query `x`-এর জন্য `y` predict করতে হয় — শুধু next-token prediction দিয়ে train করা, এই কোর্সের প্রতিটি GPT model-এর মতো একই objective।
- যেহেতু প্রতিটি episode-এ `k` নতুন করে sample করা হয়, model কখনোই একটি স্থির lookup table মুখস্থ করতে পারে না। সঠিক উত্তর দেওয়ার *একমাত্র* উপায় হলো সেই নির্দিষ্ট prompt-এ দেওয়া example থেকে `k` অনুমান করে প্রয়োগ করা — ঠিক section 2-এর "example পড়ো, নিয়ম অনুমান করো, প্রয়োগ করো" algorithm, যা training distribution নিজেই অস্তিত্বে আসতে বাধ্য করে।
- Architecture হলো [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)-এর হুবহু `MiniGPT` decoder-only stack (causal self-attention + feed-forward blocks), lesson-টিকে self-contained রাখতে এখানে স্থানীয়ভাবে আবার declare করা।
- শুধু `k in {0,1,2,3}` shift-এ train করার পর, model-কে `k in {4,5}` shift-এ evaluate করা হয় — এমন মান যা এটি **training-এর সময় কখনো দেখেনি**। এই held-out shift-গুলোতে `1/M` chance baseline-এর উপরে যেকোনো accuracy কেবল inference-এর সময় in-context example পড়া ও ব্যবহার করা থেকেই আসতে পারে, weights সম্পূর্ণ frozen রেখে। Section 2 যা বিমূর্তভাবে বর্ণনা করেছিল, এটি তার পরিমাপকৃত, পুনরুৎপাদনযোগ্য সংস্করণ।

## 4. In-context example-এর সংখ্যা বনাম accuracy

`example.py` দেখানো example-এর সংখ্যা (`n_shown = 1..5`) sweep করে এবং প্রতিটি বিন্দুতে held-out-shift accuracy মাপে। যেহেতু training-এ প্রতি episode-এ সবসময় 2 থেকে 5 টি example দেখানো হয়েছে, 1-shot prompt training distribution-এর বাইরে পড়ে এবং সেখানে accuracy পরিমাপযোগ্যভাবে সবচেয়ে খারাপ। বেশি example model-কে উত্তরে প্রতিশ্রুতিবদ্ধ হওয়ার আগে hidden `k` নির্ধারণ করার জন্য আরও redundant প্রমাণ দেয় — বাস্তব LLM-এর জন্য রিপোর্ট করা একই গুণগত curve, যেখানে একটি নতুন task-এ accuracy সাধারণত বেশি few-shot example যোগ করলে বাড়ে (ক্রমহ্রাসমান লাভসহ), যতক্ষণ না context window বা example-এর বৈচিত্র্য ফুরিয়ে যায়।

## 5. Prompt sensitivity: example-এর ক্রম গুরুত্বপূর্ণ (Zhao et al., 2021)

In-context example-এর *set* স্থির রেখে শুধু তাদের *ক্রম* বদলালে কোনো পার্থক্য হওয়া উচিত নয়, যদি model সত্যিই একটি order-invariant rule-extraction algorithm শিখে থাকে। `example.py` এটি সরাসরি পরীক্ষা করে: example-গুলো `x` অনুযায়ী sorted করে দেখালে (পুরো training জুড়ে ব্যবহৃত ক্রম) accuracy কত, আর model কখনো train করেনি এমন একটি এলোমেলো ক্রমে দেখালে কত — ঠিক একই example এবং ঠিক একই hidden `k` ব্যবহার করে তুলনা করে। Zhao et al. (2021), *"Calibrate Before Use: Improving Few-Shot Performance of Language Models,"* বাস্তব LLM-এ এই একই প্রভাব নথিভুক্ত করেছিলেন — যৌক্তিকভাবে সমতুল্য few-shot prompt যা শুধু example-এর ক্রমে (বা সামান্য formatting-এ) ভিন্ন, তা পরিমাপযোগ্যভাবে ভিন্ন accuracy দিতে পারে, কারণ model-এর training data কখনো তাকে এসব বাহ্যিক খুঁটিনাটির প্রতি নিখুঁত invariance শেখায়নি। Prompt লেখেন এমন যে কারো জন্য এটি একটি সরাসরি, ব্যবহারিক পরিণতি: example-এর ক্রম, formatting, এমনকি কোন example বেছে নেওয়া হলো — এগুলো নিরপেক্ষ পছন্দ নয়, এবং এগুলোকে নিরীহ ধরে না নিয়ে নিয়ন্ত্রণ করা উচিত (অথবা স্পষ্টভাবে randomize করে গড় নেওয়া উচিত) — একই প্রবৃত্তি যা [Lesson 4](../04-Automatic-Prompt-Optimization/README.md)-এর automatic prompt search-কে অনুপ্রাণিত করে।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. অনুপ্রেরণা — weights এখন frozen; হাতে থাকা একমাত্র lever হলো prompt
2. Zero-shot বনাম few-shot, একটি prompt example দিয়ে সুনির্দিষ্টভাবে সংজ্ঞায়িত
3. In-context learning হলো "forward pass-এ চলমান একটি algorithm," weight update নয় — GPT-3-এর মূল ফলাফলের recap
4. ধাঁধা: একটি frozen model কীভাবে একটি *নতুন* task শিখতে পারে? উত্তর: একটি স্থির task নয়, task-এর একটি distribution-এর উপর pretraining
5. `example.py`-এর episodic training setup-এর walkthrough — hidden shift `k`, held-out test shift, কেন মুখস্থ করা অসম্ভব
6. চালিয়ে দেখুন: কখনো-না-দেখা shift-এ training-এর আগে/পরে accuracy, শুধু in-context example-এর মাধ্যমে
7. দুটি পরিমাপকৃত curve: example-সংখ্যা বনাম accuracy, এবং sorted বনাম এলোমেলো ক্রম (Zhao et al. 2021)
8. Recap + preview: এরপর model-কে *তার কাজ দেখাতে* prompt করা (Chain-of-Thought, Lesson 2)

## আরও পড়ুন

- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3; যে paper in-context learning-এর নামকরণ ও জনপ্রিয় করেছে)
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2; zero-shot task transfer, পূর্বসূরি ফলাফল)
- Zhao et al. (2021), *Calibrate Before Use: Improving Few-Shot Performance of Language Models*
- Liu et al. (2021), *What Makes Good In-Context Examples for GPT-3?* (example নির্বাচন ও ক্রমের প্রভাব)
- Min et al. (2022), *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*
