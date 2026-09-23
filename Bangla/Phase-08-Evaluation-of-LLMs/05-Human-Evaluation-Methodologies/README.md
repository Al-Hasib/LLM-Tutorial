# Human Evaluation Methodologies

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `05-Human-Evaluation-Methodologies`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 3](../03-LLM-as-a-Judge/README.md)-এ একটি LLM-কে মানব evaluator-এর জায়গায় দাঁড় করানো হয়েছিল, অবিকল এই কারণে যে একটি বাস্তব মানব panel বড় মাপে চালানো ধীর ও ব্যয়বহুল — কিন্তু LLM judge-টির পুরো নকশা (pairwise comparison, rubric অক্ষ, সংশোধন করার bias-গুলো) সরাসরি তৈরি হয়েছিল বাস্তবে মানব মূল্যায়ন কীভাবে কাজ করে তার উপর ভিত্তি করে। এই lesson-এ সেই মূল, gold-standard সংস্করণটি নিয়ে আলোচনা: আপনি কীভাবে একটি rubric ডিজাইন করেন, কীভাবে এক গাদা pairwise মানব preference ভোটকে একটি একক ranking-এ রূপান্তর করেন, এবং — সবচেয়ে গুরুত্বপূর্ণ — আপনি কীভাবে যাচাই করেন যে আপনার মানব annotator-রা ভোটের উপর বিশ্বাস করার আগে আদৌ নিজেদের মধ্যে একমত কি না। এই lesson-এ ব্যবহৃত rubric অক্ষগুলো (helpfulness, accuracy, harmlessness) [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing)-এর **HHH** কাঠামোর সরাসরি অপারেশনালাইজেশন (operationalization), এবং "accuracy" অক্ষটি গ্রেড করা হয় [Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md#3-automated-factuality-checking-two-complementary-approaches)-এ তৈরি করা ঠিক সেই entailment রায় ব্যবহার করে। এখানে সংগ্রহ করা pairwise preference data-ও সেই কাঁচামাল যেটির উপর [Phase 06 Lesson 2 (Reward Modeling)](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md) একটি reward model প্রশিক্ষণ দেয় — মানব মূল্যায়ন শুধু পরবর্তীতে মডেল যাচাই করার পদ্ধতি নয়, এটি সেই data source যেটির উপর পুরো RLHF pipeline চলে।

## এই পাঠে যা যা আছে

- Rubric ডিজাইন: একটি একক সামগ্রিক "ভালো/খারাপ" score-এর বদলে একাধিক evaluation অক্ষ
- মানব-annotation protocol হিসেবে pairwise comparison এবং কেন
- Elo rating: দাবা রেটিং ব্যবস্থা, যা pairwise ভোট থেকে মডেল rank করার জন্য সম্পূর্ণ ধার করা
- Inter-annotator agreement: কেন কাঁচা percent-agreement বিভ্রান্তিকর
- Cohen's kappa: যে সূত্রটি percent-agreement-কে সুযোগের (chance) জন্য সংশোধন করে

## 1. Rubric ডিজাইন: একাধিক অক্ষ, একক score নয়

একজন মানব annotator-কে "এই response-টি কি ভালো?" জিজ্ঞেস করা বেশ কয়েকটি সত্যিই স্বাধীন প্রশ্নকে একটি সংখ্যায় চেপে ফেলে, যা ঠিক সেই তথ্যটিই লুকিয়ে রাখে যেটি আপনার একটি মডেল *কেন* খারাপ তা নির্ণয় করতে দরকার। আদর্শ অনুশীলন বদলে annotator-দের বেশ কয়েকটি অক্ষে আলাদাভাবে রেট করতে বলে, সবচেয়ে সাধারণভাবে [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing)-এর একই **HHH** triad-এর একটি সংস্করণ:

- **Helpfulness** — response-টি কি সত্যিই ব্যবহারকারীর অনুরোধ সম্বোধন করে?
- **Accuracy** — response-এর factual claim-গুলো কি সঠিক ও সমর্থিত ([Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md)-এর entailment check-এর মানব সংস্করণ)?
- **Harmlessness** — response-টি কি অনিরাপদ, আপত্তিকর বা অন্যথায় ক্ষতিকর বিষয়বস্তু এড়িয়ে চলে?

এগুলোর পৃথকীকরণ গুরুত্বপূর্ণ কারণ এগুলো স্বাধীনভাবে চলতে পারে — এবং প্রায়ই চলে: একটি response অত্যন্ত helpful এবং accurate হতে পারে অথচ সূক্ষ্মভাবে অনিরাপদ (কোনো বিপজ্জনক জিনিসের সঠিক, কার্যকর নির্দেশনা দেওয়া), অথবা সর্বাধিক harmless হতে পারে অথচ অকেজো (একটি নিরীহ অনুরোধ প্রত্যাখ্যান করা)। একটি একক সামগ্রিক score এই উত্তেজনাগুলোকে গড় করে মুছে দেয়; একটি multi-axis rubric সেগুলোকে দৃশ্যমান রাখে — [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing)-এ তিনটি H-এর মধ্যে ইতিমধ্যেই চিহ্নিত করা ঠিক একই উত্তেজনা।

## 2. pairwise comparison থেকে Elo rating

একটি একক response-কে absolute score দেওয়ার জন্য annotator-দের জিজ্ঞেস করার বদলে (noisy ও ক্যালিব্রেট করা কঠিন — দেখুন [Lesson 3 §1](../03-LLM-as-a-Judge/README.md#1-two-judge-protocols)), প্রভাবশালী মানব-evaluation protocol হলো **pairwise comparison**: একই prompt-এর প্রতি দুটি মডেলের response দেখাও, কোনটি ভালো (বা tie) জিজ্ঞেস করো, এবং অনেক prompt ও অনেক মডেল-জোড়া জুড়ে পুনরাবৃত্তি করো। তখন প্রশ্নটি হয়ে ওঠে: pairwise win/loss/tie ফলাফলের একটি বড় গাদা দেওয়া থাকলে, সেটিকে একটি একক ranked list-এ কীভাবে রূপান্তর করা যায়? এটিই ঠিক সেই সমস্যা যা দাবা আগেই সমাধান করেছিল।

**Elo rating ব্যবস্থা** (Arpad Elo, দাবার জন্য তৈরি, Chatbot Arena-র মতো পাবলিক leaderboard-গুলো সরাসরি model ranking-এর জন্য অভিযোজিত): প্রতিটি প্রতিযোগীর (এখানে প্রতিটি মডেল) একটি rating `R` আছে। rating `R_A` এবং `R_B` বিশিষ্ট দুটি প্রতিযোগী দেওয়া থাকলে, A-এর B-কে হারানোর *প্রত্যাশিত* probability হলো rating-ফাঁকের একটি logistic ফাংশন:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

লক্ষ্য করুন `E_A + E_B = 1` (এরা পরিপূরক), এবং যদি `R_A == R_B` হয় তাহলে `E_A = 0.5` — সমান rating ভবিষ্যদ্বাণী করে একটি মুদ্রা-নিক্ষেপের ফলাফল, ঠিক যেমনটি আশা করা যায়। পর্যবেক্ষিত ফলাফল `S_A` সহ একটি প্রকৃত ম্যাচের পরে (`S_A = 1` যদি A জেতে, `0` যদি A হারে, tie হলে `0.5`), দুটি rating-ই আপডেট হয় **সারপ্রাইজ** (surprise) দ্বারা — যা আসলে ঘটল এবং যা প্রত্যাশিত ছিল তার মধ্যে ব্যবধান:

```
R_A_new = R_A + K * (S_A - E_A)
R_B_new = R_B + K * (S_B - E_B)
```

`K` হলো একটি নির্দিষ্ট step size যা নিয়ন্ত্রণ করে rating প্রতি ম্যাচে কত দ্রুত সরে (`K` বড় হলে দ্রুত খাপ খায় কিন্তু বেশি noisy)। আপডেট নিয়মটি সরাসরি পড়া যায়: যদি A-কে `E_A = 0.9` দিয়ে জেতার আশা করা হয়েছিল এবং সে সত্যিই জেতে (`S_A = 1`), সারপ্রাইজ `S_A - E_A = 0.1` ছোট, তাই A-এর rating প্রায় সরে না — অনেক দুর্বল প্রতিপক্ষকে হারানো তথ্যপূর্ণ নয়। যদি A-কে *হারার* আশা করা ছিল (`E_A = 0.1`) কিন্তু সে তবু জেতে, সারপ্রাইজ `S_A - E_A = 0.9` বড়, এবং A-এর rating লাফিয়ে ওঠে — একটি উল্টো ফলাফল highly informative যে A আসলে তার বর্তমান rating-এর প্রতিফলনের চেয়ে শক্তিশালী। এই আপডেটটি হাজার হাজার pairwise ম্যাচ ফলাফলের উপর চালান (হুবহু যেভাবে Chatbot Arena লক্ষ লক্ষ মানব ভোটকে তার পাবলিক leaderboard-এ একত্রিত করে) — এবং, `example.py` পরিচিত সত্য গুণমানের মডেলগুলোর মধ্যে synthetic ম্যাচ দিয়ে প্রদর্শন করে, rating-গুলো একত্রিত হয়ে প্রতিযোগীদের সঠিকভাবে rank করবে, যদিও প্রতিটি পৃথক ম্যাচ ফলাফল noisy।

## 3. Inter-annotator agreement এবং কেন কাঁচা percent-agreement বিভ্রান্তিকর

কোনো মানব-annotated dataset (rubric score বা pairwise ভোট) বিশ্বাস করার আগে, আপনার জানা দরকার স্বাধীন annotator-রা আদৌ একে অপরের সাথে একমত কি না — যদি তারা না হয়, তাহলে "ground truth" label-গুলো সংকেতের (signal) চেয়ে শব্দের (noise) কাছাকাছি। naive পদ্ধতিটি হলো **percent agreement**: সেই ভগ্নাংশ যেমনটা দুই annotator একই label দিয়েছেন। এটি একটি নির্দিষ্ট, সুপরিচিত কারণে বিভ্রান্তিকর: **এটি কখনোই সেই চুক্তির জন্য সংশোধন করে না যা আপনি খাঁটি সুযোগ থেকে আশা করবেন**, এবং label যখনই অসামঞ্জস্যপূর্ণ (imbalanced) হবে তখনই chance-agreement খুব বেশি হতে পারে। যদি একটি safety-labeling কাজে 95% response "safe" এবং মাত্র 5% "unsafe" হয়, তাহলে দুইজন annotator যারা প্রত্যেকে প্রতিবারই কেবল "safe" অনুমান করে, তারা 95%+ আইটেমে একমত হবে — একটি percent-agreement score যা চমৎকার দেখায় অথচ *শূন্য* প্রকৃত বিচার প্রতিফলিত করে।

## 4. Cohen's kappa

**Cohen's kappa** (Cohen, 1960) স্পষ্টভাবে chance agreement বিয়োগ করে এটির সমাধান করে:

```
kappa = (p_o - p_e) / (1 - p_e)
```

- `p_o` — **পর্যবেক্ষিত চুক্তি (observed agreement)**: দুই annotator একই label দেওয়া আইটেমের কাঁচা ভগ্নাংশ (এটি §3-এর হুবহু naive percent-agreement)।
- `p_e` — **সুযোগ অনুযায়ী প্রত্যাশিত চুক্তি (expected agreement by chance)**: দুই annotator প্রত্যেকে নিজের পর্যবেক্ষিত label distribution অনুযায়ী স্বাধীনভাবে অনুমান করলে যে agreement rate তৈরি হতো। হিসাব: `p_e = sum_k( P1(k) * P2(k) )`, যেখানে `P1(k)` এবং `P2(k)` হলো প্রতিটি annotator class `k` হিসেবে label দেওয়া আইটেমের ভগ্নাংশ — অর্থাৎ, দুজনই সুযোগক্রমে স্বাধীনভাবে class `k` বেছে নেওয়ার সম্ভাবনা, সব class-জুড়ে যোগ করা।

সূত্রটি সরাসরি পড়া যায়: `p_o - p_e` হলো সুযোগের *উপরে এবং বাইরে* অর্জিত চুক্তির পরিমাণ, এবং `1 - p_e` (সুযোগের বাইরে সর্বোচ্চ সম্ভাব্য চুক্তি) দিয়ে ভাগ করলে তা 0-থেকে-1-ধরনের স্কেলে normalise হয়। `kappa = 1` মানে নিখুঁত চুক্তি; `kappa = 0` মানে annotator-রা ঠিক ততবার একমত হয় যতটা সুযোগ একাই ভবিষ্যদ্বাণী করত (তাদের আপাত চুক্তিতে কোনো প্রকৃত সংকেত নেই); `kappa < 0` মানে তারা সুযোগের ভবিষ্যদ্বাণীর চেয়ে *কম* একমত। এটিই কারণ যে কোনো গুরুতর মানব-evaluation গবেষণায় inter-annotator নির্ভরযোগ্যতার মানক রিপোর্ট করা পরিসংখ্যানটি kappa, percent agreement নয় — উচ্চ `p_e` সহ একটি উচ্চ `p_o` (উপরের imbalanced-label উদাহরণের মতো) নিম্ন বা প্রায়-শূন্য kappa-তে সংকুচিত হয়, সঠিকভাবে প্রকাশ করে যে কাঁচা চুক্তি সংখ্যাটি label imbalance-এর একটি মরীচিকা ছিল, বাস্তব annotator ঐকমত্য নয়।

## 5. `example.py` কী প্রদর্শন করে

Elo rating শূন্য থেকে implement করা হয় এবং বিভিন্ন গোপন প্রকৃত গুণমান স্তরের কয়েকটি toy মডেলের মধ্যে দীর্ঘ synthetic pairwise "মানব পছন্দ" ম্যাচ ফলাফলের উপর চালানো হয় — দেখানো হয় যে প্রতিটি পৃথক ম্যাচ একটি noisy মুদ্রা-নিক্ষেপ হওয়া সত্ত্বেও ফলে আসা rating-গুলো সঠিক গুণমান ranking-এ একত্রিত হয়। এরপর Cohen's kappa শূন্য থেকে implement করা হয় এবং দুটি বিপরীত পরিস্থিতির দুটি synthetic annotator label সেটে হিসাব করা হয়: একটি প্রকৃত উচ্চ-চুক্তি ক্ষেত্র, এবং একটি imbalanced-label ক্ষেত্র যেখানে কাঁচা percent agreement উচ্চ দেখায় কিন্তু kappa সঠিকভাবে প্রকাশ করে যে এটি বেশিরভাগই একটি chance/imbalance আর্টিফ্যাক্ট।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — মানব মূল্যায়ন হলো সেই gold standard যেটির উপর [Lesson 3](../03-LLM-as-a-Judge/README.md)-এর LLM judge তৈরি হয়েছিল, এবং [Phase 06&#39;s reward models](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md)-এর সরাসরি data source
2. Rubric ডিজাইন: helpfulness / accuracy / harmlessness স্বাধীন অক্ষ হিসেবে, HHH-এর প্রতিধ্বনিসহ
3. পছন্দের annotation protocol হিসেবে pairwise comparison, এবং কেন এটি absolute scoring-কে হারায়
4. Elo rating: দাবার সূত্র, expected-score ফাংশন, এবং "surprise"-চালিত আপডেট নিয়ম
5. `example.py` অংশ 1-এর ভেতরে-বাইরে — পরিচিত গুণমানের মডেলগুলোর মধ্যে synthetic ম্যাচ, Elo rating-কে সত্য ranking-এ একত্রিত হতে দেখুন
6. কেন percent agreement বিভ্রান্তিকর: imbalanced-label মরীচিকা
7. Cohen's kappa: সূত্রটি, এবং p_o, p_e ও normalization আসলে কী বোঝায়
8. `example.py` অংশ 2-এর ভেতরে-বাইরে — kappa সঠিকভাবে প্রকৃত চুক্তিকে chance-চালিত চুক্তি থেকে আলাদা করে, বাস্তব সংখ্যা সহ

## আরও পড়ুন

- Elo, A. (1978), *The Rating of Chessplayers, Past and Present* (মূল Elo rating ব্যবস্থা)
- Cohen, J. (1960), *A Coefficient of Agreement for Nominal Scales* (মূল Cohen's kappa paper)
- Chiang, Zheng, Sheng et al. (2024), *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* (Elo-শৈলীর ranking লক্ষ লক্ষ pairwise মানব ভোট থেকে বড় মাপে LLM-এ প্রয়োগ)
- Bai, Jones, Ndousse et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (reward modeling-এর অন্তর্নিহিত মানব-preference data সংগ্রহ pipeline, দেখুন [Phase 06 Lesson 2](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md))
- Artstein, Poesio (2008), *Inter-Coder Agreement for Computational Linguistics* (NLP annotation-এর জন্য kappa ও সম্পর্কিত agreement পরিসংখ্যানের একটি গভীর সমীক্ষা)