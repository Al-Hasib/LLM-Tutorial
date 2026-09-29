# Chain-of-Thought এবং Reasoning Prompts

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `02-Chain-of-Thought-and-Reasoning-Prompts`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md) প্রতিষ্ঠা করেছে যে একটি frozen model-এর একমাত্র lever হলো prompt, এবং মেপে দেখিয়েছে in-context example-এর *সংখ্যা* ও *ক্রম* কীভাবে accuracy বদলায়। এই lesson একটি ভিন্ন প্রশ্ন করে: যেসব সমস্যায় এক ধাপের বেশি reasoning দরকার (arithmetic word problem, multi-hop logic), সেখানে prompt-এর *বিষয়বস্তু* — বিশেষত model-কে মধ্যবর্তী কাজ দেখাতে বলা — কি একটিও নতুন example যোগ না করে নিজে থেকেই accuracy বদলাতে পারে? এবং একবার model একটি reasoning trace তৈরি করতে পারলে, শুধু কয়েকবার sample করেই কি বিনামূল্যে এর থেকে আরও নির্ভরযোগ্যতা বের করা যায়? দ্বিতীয় প্রশ্নটির একটি পরিষ্কার, প্রমাণযোগ্য উত্তর আছে, এবং এটিই এই lesson-এর `example.py`-এর কেন্দ্রবিন্দু। দুটি ধারণাই [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md)-এর পূর্বশর্ত, যা "একটি reasoning chain"-কে "reasoning branch-এর একটি searchable tree"-তে সাধারণীকরণ করে।

## এই lesson যা কভার করে

- Chain-of-Thought (CoT) prompting: few-shot worked example দিয়ে মধ্যবর্তী reasoning step বের করে আনা
- Zero-shot CoT: "Let's think step by step" কৌশল, কোনো worked example লাগে না
- Self-consistency: একাধিক স্বাধীন reasoning path sample করে চূড়ান্ত উত্তরে majority-vote করা
- Self-consistency কেন কাজ করে তার গাণিতিক মেরুদণ্ড হিসেবে Condorcet Jury Theorem
- `example.py`: self-consistency-র সুবিধার একটি exact-formula-plus-simulation প্রমাণ — এবং সেই সৎ ক্ষেত্রটিরও, যেখানে এটি উল্টো ফল দেয়
- কেন ভুল উত্তরের বিভাজন (vote-splitting) বাস্তব, open-ended task-এ self-consistency-কে আরও কার্যকর করে

## 1. Chain-of-Thought prompting (Wei et al., 2022)

সাধারণ few-shot prompting (Lesson 1) model-কে `(input, final_answer)` pair দেখায়। **Chain-of-Thought (CoT)** prompting এর বদলে `(input, reasoning_steps, final_answer)` triple দেখায় — few-shot example-গুলো নিজেরাই সমস্যাটি *ধাপে ধাপে সমাধান করা* প্রদর্শন করে, শুধু উত্তর বলে না:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls is 6 balls. 5 + 6 = 11. The answer is 11.

Q: <new question>
A:
```

Wei et al. (2022) দেখিয়েছিলেন যে multi-step arithmetic, commonsense ও symbolic reasoning benchmark-এ এই একটিমাত্র পরিবর্তন — উত্তর *কী* তা নয়, *কীভাবে* reasoning করতে হয় তা দেখানো — যথেষ্ট বড় model-এ বিশাল accuracy উল্লম্ফন ঘটায়, অথচ ছোট model-এ প্রভাব নগণ্য। এটি একটি **emergent capability**, [Phase 03 Lesson 1-এর Further Reading](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#আরও-পড়ুন)-এ ব্যবহৃত অর্থে: একটি নির্দিষ্ট scale-এর নিচে CoT prompting-এর সুবিধা প্রায় থাকেই না এবং তার উপরে বড় হয়ে ওঠে, prompting কৌশলে কোনো পরিবর্তন ছাড়াই।

## 2. Zero-shot Chain-of-Thought (Kojima et al., 2022)

প্রতিটি task-এর জন্য হাতে পূর্ণ worked-example reasoning trace লেখা ব্যয়বহুল। Kojima et al. (2022) একটি আশ্চর্যরকম সহজ বিকল্প দেখিয়েছিলেন: প্রশ্নের পরে আক্ষরিক বাক্যাংশ **"Let's think step by step"** জুড়ে দিন, *কোনো* worked example ছাড়াই, এবং উত্তর দেওয়ার আগে model-কে শূন্য থেকে নিজের reasoning trace তৈরি করতে দিন। এটি একই benchmark-এ few-shot CoT-এর সুবিধার একটি উল্লেখযোগ্য অংশ পুনরুদ্ধার করে, শুধু একটি স্থির instruction string থেকে। এটি একটি চমকপ্রদ প্রদর্শন যে scale কতটা সুপ্ত reasoning ক্ষমতা উন্মুক্ত করে, এমনকি কোনো demonstration ছাড়াই — model ইতিমধ্যে সমস্যাটি ভাঙতে "জানে"; তাকে মূলত বলে দিতে হয় যে চূড়ান্ত উত্তরে প্রতিশ্রুতিবদ্ধ হওয়ার আগে আসলে সেটা করুক, তাৎক্ষণিক উত্তর না দিয়ে।

## 3. Self-consistency (Wang et al., 2022): অনেকবার sample করো, vote দাও

CoT এবং zero-shot CoT দুটিই একটি মাত্র reasoning trace তৈরি করে, সাধারণত greedy বা low-temperature decoding দিয়ে। Wang et al. (2022) লক্ষ করেছিলেন যে ভিন্ন ভিন্ন reasoning path একই সমস্যায় ভিন্ন দিক থেকে পৌঁছাতে পারে, এবং সবগুলো একইভাবে ব্যর্থ হয় না — তাই একবার decode করার বদলে, **self-consistency**:

1. *একই* প্রশ্নের জন্য `k` টি স্বাধীন reasoning trace sample করে (temperature/top-p sampling দিয়ে, যাতে প্রতিটি trace প্রকৃতপক্ষে ভিন্ন হতে পারে);
2. প্রতিটি trace-এর চূড়ান্ত উত্তর বের করে;
3. সব `k` টি trace জুড়ে **majority-vote** উত্তরটি ফেরত দেয়, আলাদা আলাদা reasoning trace পুরোপুরি বাদ দিয়ে।

এর খরচ একটি একক sample-এর `k` গুণ inference compute, বিনিময়ে বেশি accuracy — একটি compute/accuracy trade-off, বিনামূল্যের কিছু নয়। সেই compute ঠিক কতটা accuracy কেনে, এবং কোন অবস্থায় এটি উল্টো ফল দিতে পারে, সেটিই `example.py` সরাসরি প্রমাণ করে।

## 4. Condorcet Jury Theorem: voting কেন কাজ করে — এবং কখন করে না

একটি reasoning sample-কে বিমূর্তভাবে model করুন: এটি `p` সম্ভাব্যতায় সঠিক চূড়ান্ত উত্তরে পৌঁছায়, অন্য প্রতিটি sample থেকে স্বাধীনভাবে। এটি হুবহু **Condorcet Jury Theorem** (1785)-এর setup: `k` জন স্বাধীন "voter" দেওয়া থাকলে, যাদের প্রত্যেকে `p` সম্ভাব্যতায় সঠিক, এবং majority rule থাকলে,

- যদি `p > 0.5` হয়, **majority** সঠিক হওয়ার সম্ভাব্যতা `k -> infinity` হওয়ার সাথে সাথে একঘেয়েভাবে **1**-এর দিকে বাড়ে;
- যদি `p < 0.5` হয়, majority সঠিক হওয়ার সম্ভাব্যতা `k -> infinity` হওয়ার সাথে সাথে একঘেয়েভাবে **0**-এর দিকে কমে;
- যদি ঠিক `p == 0.5` হয়, প্রতিটি `k`-এর জন্য majority accuracy ঠিক **0.5**-এই থাকে — বাড়িয়ে তোলার মতো কোনো signal নেই।

`example.py` "`k` টি স্বাধীন Bernoulli(p) trial-এর মধ্যে `k/2`-এর বেশি সঠিক হওয়ার সম্ভাব্যতা"-র exact binomial formula গণনা করে, একটি স্বাধীন Monte Carlo simulation দিয়ে তা যাচাই করে, এবং তারপর `k` ও `p` দুটোই sweep করে তিনটি regime সংখ্যায় দেখায়। `p < 0.5` regime এড়িয়ে যাওয়ার মতো কোনো কাল্পনিক edge case নয় — এটি theorem-এর সৎ, প্রয়োজনীয় অপর পিঠ, এবং এর অর্থ self-consistency তখনই ভালো ধারণা যখন বিশ্বাস করার স্বাধীন কারণ আছে যে মূল reasoning পদ্ধতিটি হাতের task-এ ইতিমধ্যে chance-কে হারায় (`p > 0.5`); chance-এর চেয়ে *খারাপ* কোনো পদ্ধতিতে এটি অন্ধভাবে প্রয়োগ করলে চূড়ান্ত উত্তর নির্ভরযোগ্যভাবে আরও খারাপ হয়, ভালো নয়।

## 5. Binary-র বাইরে: vote-splitting বাস্তব self-consistency-কে আরও শক্তিশালী করে

উপরের Condorcet model ধরে নেয় যে ঠিক দুটি সম্ভাব্য উত্তর আছে, "সঠিক" এবং "ভুল উত্তরটি" — একটি একক সংঘবদ্ধ বিরোধী পক্ষ। বাস্তব CoT task-এর (arithmetic, multi-hop QA) answer space open-ended: একটি সংখ্যাসূচক উত্তর, একটি free-text span। দুটি ভিন্ন ত্রুটিপূর্ণ reasoning path কদাচিৎ *হুবহু একই* ভুল সংখ্যায় পৌঁছায়। `example.py`-এর শেষ experiment `(1 - p)` probability mass-কে একটি নয়, কয়েকটি আলাদা ভুল label-এ ছড়িয়ে দেয়, এবং দেখায় যে এই বিভক্ত answer space-এর উপর **plurality voting** এমন কিছু regime-এও সঠিক-উত্তর accuracy পুনরুদ্ধার করে যেখানে শুধু binary-case formula হতাশাজনক দেখাত, কেবল এই কারণে যে ভুল vote-গুলো একটি একক প্রতিদ্বন্দ্বী জোটে একত্রিত না হয়ে পরস্পরের বিরুদ্ধে বিভক্ত হয়ে যায় (classic vote-splitting)। এটি একটি সৎ, mechanistic কারণ যে কেন self-consistency-র রিপোর্ট করা বাস্তব-জগতের লাভ শুধু সাধারণ binary Condorcet model-এর পূর্বাভাসকে ছাড়িয়ে যেতে পারে — অথচ একই মৌলিক সীমা মেনে চলে: ভুল উত্তর যথেষ্ট অসংখ্য এবং `p` যথেষ্ট কম হলে, সেগুলোকে আরও পাতলা করে ভাগ করাও vote-কে বাঁচাতে যথেষ্ট নয়।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. অনুপ্রেরণা — CoT prompt-এর reasoning-এ *কী আছে* তা বদলায়, শুধু কতগুলো example তা নয়; self-consistency তারপর compute খরচের বিনিময়ে সেই reasoning থেকে আরও নির্ভরযোগ্যতা নিংড়ে বের করে
2. Chain-of-Thought (Wei et al. 2022): worked-example prompt format দেখানো, এবং scale-এর সাথে emergent ফলাফল
3. Zero-shot CoT (Kojima et al. 2022): এক লাইনের কৌশল হিসেবে "Let's think step by step"
4. Self-consistency (Wang et al. 2022): k টি reasoning trace sample করা, চূড়ান্ত উত্তরে majority-vote
5. Condorcet Jury Theorem: p > 0.5 হলে voting কেন সাহায্য করে তা derive করা
6. `example.py`-এর exact-formula + Monte Carlo প্রমাণের walkthrough — সহায়ক ক্ষেত্র, p=0.5 dead zone, এবং সৎ p < 0.5 ব্যর্থতার ক্ষেত্র
7. Vote-splitting extension: কেন বাস্তব, open-ended-answer self-consistency binary গণিতের ইঙ্গিতের চেয়েও ভালো করতে পারে
8. Recap + preview: একটি reasoning chain-কে অনেকগুলোর একটি searchable tree-তে পরিণত করা (Tree-of-Thought, Lesson 3)

## আরও পড়ুন

- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners*
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- de Condorcet (1785), *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix* (মূল Jury Theorem)
- Wei et al. (2022), *Emergent Abilities of Large Language Models* ([Phase 03 Lesson 1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#আরও-পড়ুন) থেকে পুনরায় দেখা; ব্যাখ্যা করে কেন CoT-এর সুবিধা scale-নির্ভর)
