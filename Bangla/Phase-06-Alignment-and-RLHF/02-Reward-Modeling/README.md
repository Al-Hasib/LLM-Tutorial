# Reward Modeling

**ফেজ:** [Alignment and RLHF](../README.md) · **টপিক ফোল্ডার:** `02-Reward-Modeling`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 1](../01-The-Alignment-Problem/README.md) প্রতিষ্ঠা করেছে যে একটি মডেলের "এখানে একটি ভালো response-এর একটি উদাহরণ" (যা [SFT](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md) যতটুকু দেয়) — এইটুকুর চেয়ে বেশি feedback দরকার; মডেলকে জানতে হবে যে সে নিজে যেসব অনেকগুলো সম্ভাব্য response তৈরি করতে পারে, তার মধ্যে *একটি response আরেকটির তুলনায় কতটা ভালো*। একটি **reward model (RM)** হলো সেই উপাদান যা মানুষের বিচারকে একটি differentiable scalar signal-এ রূপান্তর করে, যেটিকে একটি training algorithm সত্যিই optimize করতে পারে। এই ফেজের প্রতিটি পরের লেসন তার উপর বা তার বিকল্পের উপর নির্ভর করে: [Lesson 3 (RLHF with PPO)](../03-RLHF-with-PPO/README.md) RM-এর স্কোরকে সরাসরি reinforcement learning-এর reward signal হিসেবে ব্যবহার করে; [Lesson 4 (DPO)](../04-Direct-Preference-Optimization-DPO/README.md) দেখায় যে এখানে উদ্ভূত একই underlying objective-কে optimize করার সময় কীভাবে একটি স্পষ্ট RM প্রশিক্ষণই এড়িয়ে যাওয়া যায়; [Lesson 5 (RLAIF)](../05-RLAIF-and-Constitutional-AI/README.md) এই লেসনের পাইপলাইনের মানুষের labeler-কে একটি AI labeler দিয়ে প্রতিস্থাপন করে কিন্তু reward modeling-এর বাকি সবকিছু অপরিবর্তিত রাখে।

## এই লেসন যা যা কভার করে

- কেন preference data absolute স্কোরের বদলে pairwise comparison হিসেবে সংগ্রহ করা হয়
- Preference-এর Bradley-Terry মডেল, এবং এর loss function কোথা থেকে আসে
- Reward model architecture: একটি language model-কে scalar scorer-এ রূপান্তর
- Reward-model training loop
- পরিচিত সীমাবদ্ধতা: reward hacking এবং reward model-এর নিজস্ব capacity ceiling

## 1. কেন pairwise comparison, absolute স্কোর নয়

মানুষের feedback সংগ্রহ করার নিভি পদ্ধতি হবে একটি একক response-কে "helpfulness"-এর জন্য absolute স্কেলে (ধরা যাক, 1 থেকে 10) রেট করতে বলা। অনুশীলনে এটি preference data সংগ্রহ করার সবচেয়ে কম নির্ভরযোগ্য উপায়গুলোর একটি: ভিন্ন labeler-রা স্কেলটি ভিন্নভাবে ব্যবহার করে, একই labeler বিভিন্ন session-এ নিজের সাথেই অসঙ্গত, এবং prompt-এর সামান্য শব্দগত পার্থক্যে "7" মানে কী — সেটি বদলে যেতে পারে। **Pairwise comparison** — একই *prompt*-এর দুইটি (বা ততোধিক) প্রার্থী response দেখিয়ে labeler-কে কেবল জিজ্ঞাসা করা "কোনটি ভালো?" — সংগ্রহ করা অনেক বেশি সুসংগত এবং নির্ভরযোগ্য, কারণ এটি একটি অনেক সহজ এবং ভালোভাবে ক্রমাঙ্কিত জ্ঞানীয় কাজ: মানুষ আপেক্ষিক বিচারে ("A, B-এর চেয়ে ভালো") absolute বিচারের ("A হলো 7.3") চেয়ে অনেক ভালো। এটি সেই data-collection সিদ্ধান্ত যা InstructGPT (Ouyang et al., 2022) এবং কার্যত প্রতিটি পরবর্তী RLHF পাইপলাইন নিয়েছিল, এবং নিচের পুরো গাণিতিক কাঠামোটি স্কোরের বদলে comparison-এর চারপাশে গড়ে ওঠার এটাই কারণ।

## 2. Preference-এর Bradley-Terry মডেল

একটি prompt `x` এবং দুইটি প্রার্থী response `y_w` ("বিজয়ী," অর্থাৎ যেটি মানুষের পছন্দ) ও `y_l` ("পরাজিত") দেওয়া থাকলে, আমরা এমন একটি scalar reward function `r(x, y)` fit করতে চাই যাতে মানুষ যে response-গুলো পছন্দ করে সেগুলি বেশি স্কোর পায়। **Bradley-Terry মডেল** (pair-ভিত্তিক comparison-এর একটি ধ্রুপদী পরিসংখ্যানগত মডেল, মূলত দাবা/ক্রীড়া প্রতিযোগীদের র্যাংকিংয়ের জন্য বিকশিত) ধরে নেয় যে `y_w`-কে `y_l`-এর উপর পছন্দ করার সম্ভাবনা তাদের underlying স্কোরের *পার্থক্যের* একটি logistic function:

```
P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )
                 = exp(r(x, y_w)) / ( exp(r(x, y_w)) + exp(r(x, y_l)) )
```

এর একটি সহজাত আকৃতি আছে: যদি `r(x, y_w)` `r(x, y_l)`-এর চেয়ে অনেক বড় হয়, মডেল প্রায় নিশ্চিতভাবে ভবিষ্যদ্বাণী করে যে একজন মানুষ `y_w`-ই বেছে নেবে; যদি দুইটি স্কোর কাছাকাছি হয়, তা মুদ্রা ছোড়ার মত কিছু ভবিষ্যদ্বাণী করে। মূল বিষয় হলো, **স্কোরের মধ্যে শুধুমাত্র পার্থক্যটুকুই গুরুত্বপূর্ণ** — Bradley-Terry মডেল `r(x, y_w)` ও `r(x, y_l)` দুইটিতে যেকোনো ধ্রুবক যোগ করলে অপরিবর্তিত থাকে, তাই প্রশিক্ষিত reward model-এর কাঁচা স্কোর কেবল পরস্পরের তুলনাতেই অর্থপূর্ণ, কখনো absolute, ক্রমাঙ্কিত রাশি হিসেবে নয়।

## 3. Reward model loss

আমরা একটি neural network `r_theta(x, y)` (weight `theta` দ্বারা প্যারামিটারাইজড) প্রশিক্ষণ দিই যাতে প্রতিটি সংগ্রহ করা জোড়া `(x, y_w, y_l)`-তে মানুষের প্রকৃত পর্যবেক্ষিত পছন্দের প্রতি Bradley-Terry মডেল যে likelihood নির্ধারণ করে তা সর্বাধিক হয়। `N` সংখ্যক comparison-এর একটি dataset-এর উপর negative log-likelihood নিলে reward-model loss পাওয়া যায়:

```
loss(theta) = - E_{(x, y_w, y_l)} [ log( sigmoid( r_theta(x, y_w) - r_theta(x, y_l) ) ) ]
```

এটি হুবহু binary cross-entropy, যেখানে "label" সবসময় 1 (বিজয়ীকে সবসময় বিজয়ী হিসেবেই উপস্থাপন করা হয়) এবং "logit" হলো স্কোরের পার্থক্য `r_theta(x, y_w) - r_theta(x, y_l)`। এটি minimize করলে, যখনই মডেল একটি জোড়া ভুল করে বা under-confident হয়, `r_theta(x, y_w)` উপরে এবং `r_theta(x, y_l)` নিচে ঠেলে দেয় — এবং তা করে *আপেক্ষিকভাবে*; RM-কে কোনো একক response-এর absolute "সঠিক" স্কোর প্রয়োজনও হয় না এবং দেওয়াও হয় না, কেবল বলা হয় দুইটির মধ্যে কোনটি ভালো। `example.py` এই loss-টি scratch থেকে implement করে এবং যাচাই করে যে এটি সত্যিই একটি sensible ranking পুনরুদ্ধার করে।

## 4. Reward model architecture

অনুশীলনে, একটি reward model তৈরি করা হয় একটি pretrained (এবং সাধারণত SFT-করা) language model নিয়ে এবং **এর language-modeling head-কে একটি একক scalar output head দিয়ে প্রতিস্থাপন করার মাধ্যমে** — final hidden state-কে vocabulary-র উপর distribution-এ project করার বদলে (যেমন এই কোর্সের এ পর্যন্ত প্রতিটি মডেলে ছিল), এটি project করে একটি সংখ্যায়:

```
hidden_states = Transformer(x, y)          # policy model-এর মতো একই backbone
pooled = hidden_states[:, -1, :]           # যেমন, শেষ token-এর hidden state
reward = Linear(d_model -> 1)(pooled)      # এই (x, y) জোড়ার জন্য একটি একক scalar স্কোর
```

এলোমেলোভাবে initialize করা network-এর বদলে pretrained/SFT backbone থেকে শুরু করা গুরুত্বপূর্ণ: response ভালোভাবে বিচার করতে RM-কে সত্যিই ভাষা, তথ্যনির্ভরতা এবং কাজের গুণমান বোঝার দরকার, এবং ইতিমধ্যে প্রশিক্ষিত মডেলের representation পুনরায় ব্যবহার করা তুলনামূলকভাবে ছোট preference dataset-এ scratch থেকে ভাষা বোঝা শেখার চেয়ে অনেক বেশি sample-efficient। `example.py` একটি ছোট MLP ব্যবহার করে খেলনা fixed-length feature vector-এর উপর একটি "scorer head"-এর পরিবর্ত হিসেবে, যাতে ডেমোটি দ্রুত থাকে এবং সম্পূর্ণ Transformer backbone প্রশিক্ষণের বদলে শুধুমাত্র Bradley-Terry loss-এর প্রক্রিয়ার উপর দৃষ্টি নিবদ্ধ থাকে।

## 5. পরিচিত সীমাবদ্ধতা

- **Reward hacking**: যেহেতু RM সত্যিকারের মানব পছন্দের শুধুমাত্র একটি approximation, তাই একটি downstream optimizer (যেমন [Lesson 3](../03-RLHF-with-PPO/README.md)-এর PPO) আসলে ভালো না হয়েও RM-তে উচ্চ স্কোর পাওয়া response খুঁজে পেতে পারে — মানুষ যে intent-কে approximate করার কথা ছিল তা সত্যিই পূরণ না করে, RM যা শিখেছে তার quirks বা অন্ধদাগ (blind spots) শোষণ করা।
- **Distribution shift**: RM-কে *কোনো* policy-র (প্রায়শই SFT মডেল) response-এর comparison-এর উপর প্রশিক্ষণ দেওয়া হয়। RL fine-tuning যখন policy-র আউটপুটকে response-স্পেসের নতুন অঞ্চলে ঠেলে দেয়, তখন সেখানে RM-এর বিচার ক্রমশ কম নির্ভরযোগ্য হয়ে পড়ে, কারণ সে তার নিজের প্রশিক্ষণের সময় সেরকম আউটপুটের comparison কখনো দেখেনি।
- **Label noise এবং মতানৈক্য**: প্রকৃত মানুষ labeler-রা সত্যিই অস্পষ্ট comparison-গুলোর একটি উল্লেখযোগ্য অংশে পরস্পরের সাথে মতভেদ করে, যা যেকোনো reward model-কে — যত বড়ই হোক — কতটা আত্মবিশ্বাসের সাথে প্রশিক্ষণ দেওয়া যায় তার একটি সীমা (ceiling) তৈরি করে।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "মানুষের preference আছে" থেকে "একটি সংখ্যা যা একটি training loop optimize করতে পারে" তৈরিতে রূপান্তর
2. কেন pairwise comparison, data collection-এর জন্য absolute rating-কে হারায়
3. Bradley-Terry মডেল: স্কোর পার্থক্যের sigmoid হিসেবে preference probability
4. Bradley-Terry-র অধীনে negative log-likelihood হিসেবে reward-model loss উদ্ভাবন
5. Architecture: language model backbone + scalar head, LM head-কে প্রতিস্থাপন
6. `example.py`-এর walkthrough — পরিচিত ground truth থেকে synthetic preference pair, RM-কে প্রশিক্ষণ, এর শেখা ranking ground truth-এর সাথে সম্পর্কযুক্ত কিনা যাচাই
7. সীমাবদ্ধতা: reward hacking এবং distribution shift — Lesson 3-এর KL penalty-র ভিত্তি স্থাপন

## আরও পড়ুন

- Bradley, Terry (1952), *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons* (মূল পরিসংখ্যানগত মডেল)
- Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences* (modern deep-RL reward-modeling-from-preferences পেপার)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — সম্পূর্ণ RLHF পাইপলাইনের stage 2 হিসেবে reward model প্রশিক্ষণ)
- Stiennon et al. (2020), *Learning to Summarize from Human Feedback*