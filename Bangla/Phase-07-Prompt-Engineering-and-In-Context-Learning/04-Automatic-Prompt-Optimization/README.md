# Automatic Prompt Optimization

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `04-Automatic-Prompt-Optimization`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর এখন পর্যন্ত প্রতিটি lesson জিজ্ঞেস করেছে "prompt বদলালে accuracy-তে কী প্রভাব পড়ে?" — [Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md) example-এর সংখ্যা ও ক্রমের প্রভাব হাতে মেপেছে; [Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) এবং [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md) প্রত্যেকে একটি নির্দিষ্ট হাতে-ডিজাইন করা prompting strategy পরিচয় করিয়েছে। এই lesson স্বাভাবিক পরবর্তী প্রশ্নটি করে: কোন prompt variant সেরা তা একজন মানুষ অনুমান করে হাতে যাচাই করার বদলে, search-টিকেই কি **automate** করা যায়? দেখা যায় prompt design একটি discrete, gradient-free optimization সমস্যা — প্রকৃতিগতভাবে hyperparameter search থেকে ভিন্ন কিছু নয় — এবং `example.py` সেই search-এর একটি প্রকৃত instance শুরু থেকে শেষ পর্যন্ত তৈরি করে চালায়।

## এই lesson যা কভার করে

- Prompt design-কে একটি combinatorial space-এর উপর gradient-free, discrete optimization হিসেবে দেখা
- Automatic Prompt Engineer (APE): প্রার্থী prompt প্রস্তাব করো, একটি validation objective-এ score করো, সেরাটি রাখো (Zhou et al., 2022)
- Lesson 1-এর task থেকে একটি প্রকৃত, পরিমাপযোগ্য prompt-component search space তৈরি করা
- কেন কিছু prompt-component *combination* খারাপ score করে, যদিও প্রতিটি component আলাদাভাবে ঠিক দেখায়
- দুটি মূর্ত search strategy হিসেবে random search বনাম hill-climbing (greedy local search)
- `example.py`: দুটি পদ্ধতিই random-ভাবে একটি বেছে নেওয়ার চেয়ে নির্ভরযোগ্যভাবে ভালো prompt combination খুঁজে পায়, প্রকৃত (brute-forced) optimum-এর বিরুদ্ধে যাচাই করা

## 1. Discrete optimization হিসেবে prompt design

একটি prompt কিছু পছন্দ দিয়ে তৈরি: কোন instruction-এর শব্দচয়ন, কতগুলো example, কোন ক্রম, কোন separator। প্রতিটি পছন্দ একটি **discrete variable** যার অল্প কয়েকটি সম্ভাব্য মান আছে, এবং পূর্ণ prompt হলো ওই সব পছন্দের **Cartesian product**-এর একটি বিন্দু। হাতে prompt tune করা মানে একজন মানুষের হাতে local search করা — কিছু একটা চেষ্টা করো, output-এ চোখ বুলাও, একটা জিনিস বদলাও, পুনরাবৃত্তি করো। **Automatic prompt optimization** মানুষের বিচারকে একটি প্রকৃত scoring function (একটি held-out validation set-এ accuracy) এবং একটি প্রকৃত search procedure দিয়ে প্রতিস্থাপন করে — ঠিক সেই একই পরিবর্তন যা কয়েক দশক আগে hyperparameter tuning-এ ঘটেছিল, যখন grid/random search হাতে-বাছা learning rate-কে প্রতিস্থাপন করেছিল।

## 2. APE: Automatic Prompt Engineer (Zhou et al., 2022)

Zhou et al. (2022) বিশেষভাবে instruction prompt-এর জন্য এই loop-টিকে formalize করেছিলেন:

1. **Propose**: প্রার্থী instruction-এর একটি pool তৈরি করো (প্রায়ই একটি model-কে example input/output pair থেকে সম্ভাব্য instruction অনুমান করতে বলে, অথবা পরিচিত-ভালো instruction fragment একত্র করে)।
2. **Score**: প্রতিটি প্রার্থীকে একটি held-out validation set-এ evaluate করো, একটি প্রকৃত accuracy বা log-likelihood metric ব্যবহার করে — কোনো proxy নয়, কোনো অনুমান নয়।
3. **Select / iterate**: সর্বোচ্চ-score প্রার্থীদের রাখো, ঐচ্ছিকভাবে তাদের variation তৈরি করো ("সেরাটির চারপাশে resample করো," hill-climbing-এর একটি রূপ), এবং পুনরাবৃত্তি করো।

`example.py` হুবহু এই template অনুসরণ করে, শুধু একটি search space নিয়ে যা যাচাইয়ের উদ্দেশ্যে exhaustively score করার মতো যথেষ্ট ছোট ও সস্তা, এবং একটি বাইরের API-র বদলে scoring function হিসেবে একটি প্রকৃতপক্ষে trained ছোট model ব্যবহার করে।

## 3. একটি প্রকৃত, পরিমাপযোগ্য search space তৈরি করা

`example.py` Lesson 1-এর `y = (x + k) mod M` in-context task আবার train করে, কিন্তু এবার training distribution প্রতিটি episode-এ **তিনটি স্বাধীন prompt component** বদলায়:

1. **`NUM_EXAMPLES`** in `{2, 3, 4, 5}` — কতগুলো in-context pair দেখানো হয়।
2. **`ORDER`** in `{sorted, scrambled}` — pair-গুলো কোন ক্রমে আসে।
3. **`PHRASING`** in `{'P', 'Q'}` — একটি একক leading token, যা দুটি ভিন্ন instruction phrasing-এর প্রতিনিধিত্ব করে, যেগুলো একজন prompt engineer একটি বাস্তব natural-language system-এ A/B test করতে পারেন।

এটি একটি `4 x 2 x 2 = 16`-combination discrete search space। গুরুত্বপূর্ণভাবে, training data এটিকে সমানভাবে **cover করে না**: marker `'Q'` সবসময় শুধু 2-3 টি দেখানো example-এর সাথে আসে, যেখানে marker `'P'` পুরো পরিসরের সাথে আসে। এর অর্থ `(marker='Q', n_shown in {4,5})` এমন একটি combination যা trained model প্রকৃতপক্ষে কখনো দেখেনি — যদিও প্রতিটি আলাদা component (`'Q'`, এবং `n_shown=5`) নিজে নিজে সম্পূর্ণ পরিচিত। এটি একটি অত্যন্ত বাস্তব prompt-engineering ফাঁদের প্রতিফলন: একটি instruction phrasing এবং একটি formatting পছন্দ আলাদাভাবে প্রতিটি ঠিক দেখাতে পারে, অথচ একসাথে মিলে এমন কিছু হতে পারে যা model খারাপভাবে সামলায় — শুধু এই কারণে যে model-এর আচরণ যেখানে গড়ে উঠেছে সেখানে ওই নির্দিষ্ট *combination*-টি কম প্রতিনিধিত্ব পেয়েছিল।

## 4. দুটি search strategy, প্রকৃত optimum-এর বিরুদ্ধে যাচাই করা

যেহেতু এই toy space-এ মাত্র 16 টি combination, `example.py` প্রতিটিকে brute-force score করতে পারে — এমন একটি বিলাসিতা যা বাস্তব prompt-component space-এর (অনেক বেশি instruction variant, example pool, এবং formatting option-সহ) কখনো থাকে না। সেই brute-force sweep শুধু **search পদ্ধতিগুলো যাচাইয়ের ground truth** হিসেবে ব্যবহৃত হয়, প্রস্তাবিত পদ্ধতি হিসেবে নয়। এর উপরে দুটি প্রকৃত search procedure চালানো হয়:

- **Random search**: uniformly random-ভাবে কয়েকটি combination sample করো, প্রতিটি score করো, সেরাটি রাখো।
- **Hill-climbing (greedy local search)**: একটি random combination থেকে শুরু করো; বারবার ঠিক *একটি* component বদলে পৌঁছানো যায় এমন প্রতিটি combination দেখো, যে neighbor সর্বোচ্চ score করে সেখানে যাও, এবং কোনো প্রতিবেশী পরিবর্তন score না বাড়ালে থামো (একটি local optimum)।

`example.py` নিজের live run থেকে রিপোর্ট করে: সব 16 টি combination জুড়ে গড় accuracy ("অন্ধভাবে বেছে নাও" baseline), random search যে সেরা combination-টি sample করতে পেরেছে, এবং hill-climbing-এর local search কোথায় converge করে — সাথে সেই converged ফলাফল প্রকৃত brute-forced সেরার কতটা কাছে পৌঁছায়। মূর্ত সংখ্যাগুলো এখানে দাবি করার বদলে সরাসরি script দিয়ে print করা হয়, যেহেতু সেগুলো একটি নির্দিষ্ট trained model-এর run থেকে আসে।

## 5. কেন এটি এই toy example-এর বাইরেও সাধারণীকরণযোগ্য

একটি discrete space-এর উপর random search বা hill-climbing-এর কিছুই space ছোট হওয়া বা scorer একটি toy model হওয়ার উপর নির্ভর করে না। Scorer হিসেবে একটি প্রকৃত LLM API call এবং instruction template, example pool ও formatting পছন্দের অনেক বড় একটি space বসিয়ে দিন, তাহলে হুবহু এই একই দুটি algorithm (অথবা তাদের আরও পরিশীলিত উত্তরসূরি — Bayesian optimization, evolutionary search, অথবা APE-এর মতো একটি model-কে আরও ভালো prompt প্রস্তাব করতে prompt করা) হলো সেগুলোই যা production automatic-prompt-optimization tool আসলে চালায়। একমাত্র যা বদলায় তা হলো প্রতি evaluation-এর খরচ এবং space-এর আকার — এখানে প্রদর্শিত search *logic* অপরিবর্তিত।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. অনুপ্রেরণা — হাতে prompt অনুমান করা বন্ধ করুন; prompt design-কে একটি প্রকৃত scoring function-সহ optimization সমস্যা হিসেবে দেখুন
2. APE (Zhou et al. 2022): propose, score, select — এখানকার প্রতিটি পদ্ধতি যে template অনুসরণ করে
3. Search space তৈরি: 3 টি component, 16 টি combination, ইচ্ছাকৃতভাবে অসম training coverage
4. কেন "আলাদাভাবে ঠিক, একসাথে অপরীক্ষিত" combination একটি বাস্তব prompt-engineering ফাঁদ, শুধু একটি toy অদ্ভুততা নয়
5. `example.py`-এর brute-force ground truth-এর walkthrough: কোন combination ভালো score করে, কোনটি করে না, এবং কেন
6. Random search বনাম hill-climbing, run থেকে live ফলাফল
7. দুটি পদ্ধতিকে প্রকৃত optimum-এর সাথে তুলনা: automated search কি নির্ভরযোগ্যভাবে অন্ধভাবে বেছে নেওয়াকে হারায়?
8. Recap + preview: একবার prompt নির্ভরযোগ্যভাবে structured বিষয়বস্তু ধারণ করলে, model-এর OUTPUT-কেও কীভাবে নির্ভরযোগ্যভাবে structured করবেন? (Lesson 5)

## আরও পড়ুন

- Zhou et al. (2022), *Large Language Models Are Human-Level Prompt Engineers* (APE)
- Shin et al. (2020), *AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts*
- Pryzant et al. (2023), *Automatic Prompt Optimization with "Gradient Descent" and Beam Search*
- Zhao et al. (2021), *Calibrate Before Use: Improving Few-Shot Performance of Language Models* ([Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md) থেকে পুনরায় দেখা; এই lesson-এর search যে মৌলিক prompt-sensitivity ঘটনার মধ্য দিয়ে পথ খোঁজে)
