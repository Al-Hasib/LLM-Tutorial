# Reasoning Models এবং GRPO

**ফেজ:** [Alignment and RLHF](../README.md) · **টপিক ফোল্ডার:** `07-Reasoning-Models-and-GRPO`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 3 (PPO)](../03-RLHF-with-PPO/README.md) এবং [Lesson 4 (DPO)](../04-Direct-Preference-Optimization-DPO/README.md) — দুটোই *মানুষের পছন্দ* থেকে উদ্ভূত একটি signal-এর বিরুদ্ধে policy optimize করে — একটি শেখা reward model, অথবা preference pair থেকে DPO-র implicit reward। কোনোটিই কখনো ground truth-এর বিরুদ্ধে কিছু যাচাই করে না; দুটোই কেবল জিজ্ঞাসা করে "কোন response-টি একজন মানুষ (বা একজন মানুষকে অনুকরণ করতে প্রশিক্ষিত একটি মডেল) পছন্দ করেছিল?" আধুনিক reasoning model (OpenAI-এর o1, DeepSeek-R1) fundamentally ভিন্ন ধরনের reward দিয়ে RL-এ প্রশিক্ষণ নেয়: একটি **verifiable** — চূড়ান্ত সংখ্যাগত উত্তরটি কি সঠিক, উৎপন্ন কোডটি কি তার unit test পাস করেছে — যেটি একটি শেখা মডেল নয়, একটি deterministic checker গণনা করে, এবং reward-hacking ব্যর্থতা-মোডের প্রতিরোধী যেটির বিরুদ্ধে [Lesson 3 section 3](../03-RLHF-with-PPO/README.md#3-why-plain-policy-gradient-breaks-reward-hacking)-কে KL penalty দিয়ে রক্ষা করতে হয়েছিল। এই লেসন সেই পরিবর্তন, সেই RL অ্যালগরিদম (**GRPO**) — যা PPO-র critic network-টি সম্পূর্ণ সরিয়ে দিয়ে এটিতে প্রশিক্ষণকে স্কেলে চালানোর পক্ষে যথেষ্ট সস্তা করেছে — এবং এর মাধ্যমে উন্মুক্ত দ্বিতীয়, পরিপূরক লিভারটি কভার করে: *inference সময়ে* বেশি compute ব্যয় করা — [Phase 07 Lesson 2](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md)-এর self-consistency ধারণাকে "একটি frozen model-এর জন্য একটি prompting কৌশল" থেকে "এই মডেলগুলোকে কীভাবে ব্যবহার করার উদ্দেশ্য — তার সচেতন নকশা-বিন্দু"-তে সরাসরি সম্প্রসারণ।

## এই লেসন যা যা কভার করে

- Verifiable reward বনাম শেখা/preference-ভিত্তিক reward (Lesson 2-4-এর জগৎ বনাম এই লেসনের)
- GRPO (Shao et al., 2024 — DeepSeekMath): critic-বিহীন PPO — group-relative advantage estimation
- সেই group-relative advantage-এর দুইটি বৈশিষ্ট্য, সরাসরি মাপা: একটি সত্যিকারের "ব্যর্থতা থেকে দূরে ঠেলা" signal, এবং সহজ/মাঝারি/কঠিন prompt জুড়ে একটি ধারাবাহিক signal scale
- একটি খেলনা verifiable-reward কাজে একটি বাস্তব, চালানোযোগ্য GRPO training loop — এবং এই নির্দিষ্ট খেলনা কাজটি কী প্রমাণ করে ও কী প্রমাণ করে না তার একটি সৎ দৃষ্টিপাত
- Test-time scaling: [Phase 03 Lesson 5](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md)-এর train-time scaling law থেকে স্বতন্ত্র একটি দ্বিতীয় অক্ষ হিসেবে inference compute (আরও sample, majority vote) ব্যয়
- একটি বাস্তব পরীক্ষা: ইচ্ছাকৃতভাবে এখনও অপূর্ণ একটি policy-তে majority-vote নির্ভুলতা বনাম sample-সংখ্যা

## 1. দুই ধরনের reward

[Lesson 2](../02-Reward-Modeling/README.md)-এর reward model `r_phi(x, y)` মনে করুন: একটি neural network যা মানুষের pairwise preference অনুকরণ করতে প্রশিক্ষিত, তারপর "এই response কতটা ভালো" এর proxy হিসেবে ব্যবহৃত। একটি **verifiable reward** সেই proxy-কে একটি সস্তা, deterministic function দিয়ে প্রতিস্থাপন করে — completion-এর এবং একটি পরিচিত সঠিক উত্তরের — যেমন একটি গণিত সমস্যার জন্য `reward = 1 if extracted_final_answer == ground_truth else 0`, অথবা উৎপন্ন কোডের জন্য `reward = fraction of unit tests passed`। RL সময়ে reward *নিজের* জন্য কোনো human label-এর প্রয়োজন নেই (লেবেলগুলো কেবল আগে দরকার ছিল, সঠিক উত্তর জানতে বা পরীক্ষা লিখতে) — এবং গুরুত্বপূর্ণভাবে, এই reward-কে একটি শেখা reward model-কে যেভাবে ঠকানো যায় ([Lesson 3 section 3](../03-RLHF-with-PPO/README.md#3-why-plain-policy-gradient-breaks-reward-hacking)-এর reward hacking) সেভাবে ঠকানো যায় না, কারণ এটি একটি প্রকৃত যাচাইযোগ্য সত্য পরীক্ষা করছে, কোনো proxy মডেলের মতামত নয়।

## 2. GRPO: critic-বিহীন PPO

PPO-র উপাদানগুলো স্মরণ করুন ([Lesson 3 section 5](../03-RLHF-with-PPO/README.md#5-ppos-clipped-surrogate-objective)): একটি policy, advantage-এর জন্য per-state baseline অনুমানকারী একটি value/critic network, এবং clipped surrogate objective। সেই critic-কে প্রশিক্ষণ ও চালানো প্রায় policy-টির মতোই ব্যয়বহুল। GRPO-র সমাধান (Shao et al., 2024, *DeepSeekMath*): প্রতিটি prompt-এর জন্য বর্তমান policy থেকে `G` সংখ্যক completion-এর একটি **group** sample করুন, প্রতিটিকে verifiable reward দিয়ে স্কোর করুন, এবং প্রতিটি completion-এর reward-কে সরাসরি advantage-এ পরিণত করতে **group-এর নিজস্ব গড় ও মানক বিচ্যুতি** ব্যবহার করুন — কোথাও কোনো critic network নেই:

```
advantage_i = ( reward_i - mean(rewards in group) ) / ( std(rewards in group) + eps )
```

বাকিটা Lesson 3-এর সাথে অভিন্ন: একই clipped-surrogate-per-token objective (সেই লেসনের `L_CLIP` কাঠামো পুনরায় ব্যবহার করে), এবং একই frozen reference policy-র বিরুদ্ধে KL penalty — শুধু শেখা baseline-এর জায়গায় এই group-relative advantage।

## 3. কেন group-relative baseline কাজ করে — দুইটি বৈশিষ্ট্য, সরাসরি মাপা

একটি ভালো RL baseline-এর জন্য কেবল bias প্রবর্তন না করেই variance কমানো প্রয়োজন — sample নেওয়া action-এর উপর নির্ভর করে না এমন *যেকোনো* রাশি বিয়োগ করলেই সেটি পূরণ হয়, এবং group গড়, যা সেই সুনির্দিষ্ট prompt-এর `G` সংখ্যক সত্যিকারের rollout থেকে প্রতি-prompt তাজাভাবে গণনা করা হয়, হুবহু এমন একটি রাশি। `example.py` এর দুইটি concrete পরিণতি মাপে, যেকোনো নির্দিষ্ট প্রশিক্ষণ চলমান থেকে স্বাধীন:

- **একটি সত্যিকারের negative-advantage signal।** কোনোরকম baseline ছাড়া, একটি ব্যর্থ completion (`reward = 0`) হুবহু `0` advantage পায় — এটি কখনো সক্রিয়ভাবে নিচে ঠেলা হয় না, শুধু সফলতাগুলো উপরে ঠেলা হয়। GRPO-র group-relative advantage group-গড়ের নিচের যেকোনো completion-এর জন্য *ঋণাত্মক*, যার মধ্যে প্রতিটি ব্যর্থতা অন্তর্ভুক্ত যখনই group-এর *যেকোনো* সদস্য সফল হয়েছে — একটি প্রকৃত "এখান থেকে সরে যাও" signal যা শুধু-reward REINFORCE কাঠামোগতভাবে দিতে পারে না।
- **কাজের কঠিনতা নির্বিশেষে একটি ধারাবাহিক signal scale।** একটি কাঁচা, অ-স্বাভাবিককৃত reward-এর variance হলো `p(1-p)` যেখানে `p` হলো per-sample সফলতার সম্ভাবনা — এটি `p` 0 বা 1-এর কাছে গেলে *মিলিয়ে যায়*, অর্থাৎ একটি খুব সহজ বা খুব কঠিন prompt প্রায় কোনো gradient signal-ই দেয় না, যদিও শেখার মতো প্রচুর বাকি থাকতে পারে (যেমন 95% থেকে 99.9% সফলতায় যাওয়া)। GRPO-র z-scoring advantage-এর স্কেলকে পুরো কঠিনতা পরিসর জুড়ে ধ্রুবক কাছাকাছি রাখে, কারণ এটি সর্বদা সেই নির্দিষ্ট prompt-এর নিজস্ব পর্যবেক্ষিত বিস্তারের বিরুদ্ধে পুনরায় স্বাভাবিক করে।

## 4. `example.py` Part A — প্রক্রিয়াটি, তারপর একটি সৎ end-to-end চলমান

খেলনা কাজটি: `NUM_PROMPTS` সংখ্যক স্বতন্ত্র prompt, প্রতিটি `SEQ_LEN` সংখ্যক (0-5) অঙ্ক চায় যেগুলোর যোগফল একটি নির্দিষ্ট টার্গেট — একটি ছোট কিন্তু সত্যিই verifiable সঠিকতা-পরীক্ষা, [Lesson 3](../03-RLHF-with-PPO/README.md)-এর ইচ্ছাকৃতভাবে-ঠকানো যায় "hack token" এর মতো কোনো বানোয়াট শোষণযোগ্য reward নয়। Part A তিনটি জিনিস চালায়:

1. একটি প্রায়-এলোমেলো policy-র একটি rollout-এ উপরের negative-advantage-fraction পরিমাপ — সত্যিকারের মুদ্রিত সংখ্যা সহ।
2. 3% থেকে 97% সিমুলেটেড সফলতার সম্ভাবনা জুড়ে সোয়েপ করা উপরের signal-scale-consistency পরিমাপ — সত্যিকারের মুদ্রিত সংখ্যা সহ।
3. একই কাজে no-baseline ablation-এর পাশাপাশি সম্পূর্ণ GRPO update-এর (group-relative advantage + clipped surrogate + KL penalty) একটি প্রকৃত end-to-end প্রশিক্ষণ চলমান।

Section 3-এর সাথে একটি সৎ caveat আসে যা স্ক্রিপ্ট নিজেই মুদ্রণ করে: এই নির্দিষ্ট ছোট, নিম্ন-মাত্রিক খেলনা অনুসন্ধান স্থানে (প্রতি prompt-এ 216টি সম্ভাব্য অঙ্ক-ক্রম), *দুইটি* পদ্ধতিই শেষ পর্যন্ত কাজটি সমাধান করে, কারণ Adam-এর নিজস্ব adaptive step sizing আংশিকভাবে কাঁচা reward-এর অসংগত স্কেলের জন্য এখানে ক্ষতিপূরণ করে। এটি এই নির্দিষ্ট খেলনা উদাহরণের একটি বৈশিষ্ট্য, এটির প্রমাণ নয় যে baseline- গুরুত্বপূর্ণ নয় — section 1-2 সরাসরি GRPO যে প্রকৃত প্রক্রিয়া প্রদান করে তা মাপে, এবং অনুসন্ধান স্থান যখন 216টি তিন-অঙ্কের সংমিশ্রণ নয়, বরং বহু-পদক্ষেপ token sequence-এর একটি বাস্তব স্থান, তখন সেই প্রক্রিয়াটিই "sparse, verifiable reward থেকে আদৌ শেখা"-কে "না শেখা" থেকে পৃথক করে।

## 5. Test-time scaling: train-time scaling থেকে লম্ব (orthogonal) একটি দ্বিতীয় লিভার

[Phase 03 Lesson 5](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) *train-time* compute-র স্কেলিং কভার করে — আরও প্যারামিটার, আরও ডেটা, আরও প্রশিক্ষণ FLOP — loss কমানোর জন্য। Reasoning model একটি দ্বিতীয়, স্বাধীন লিভার যোগ করে: একটি নির্দিষ্ট, ইতিমধ্যে প্রশিক্ষিত মডেলে *inference সময়ে* বেশি compute ব্যয় করুন, একটি প্রশ্নের জন্য আরও ভালো উত্তর পেতে — যেমন `N` সংখ্যক completion sample করে majority vote নেওয়া। এটি হুবহু [Phase 07 Lesson 2 section 3](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#3-self-consistency-wang-et-al-2022-sample-many-vote)-এর self-consistency প্রক্রিয়া, এখানে একটি সাধারণ মডেলের উপর লাগানো prompting কৌশলের বদলে একটি reasoning model-এর সচেতন নকশা-বিন্দু হিসেবে কাঠামোবদ্ধ।

## 6. `example.py` Part B — majority-vote নির্ভুলতা বনাম N

ইতিমধ্যে ~100% one-shot নির্ভুলতায় থাকা একটি policy-র জন্য majority voting উন্নত করার মতো কিছুই অবশিষ্ট থাকে না, তাই Part B GRPO দিয়ে একটি **নতুন** policy প্রশিক্ষণ দেয় কিন্তু ইচ্ছাকৃতভাবে আগেই থামিয়ে দেয় — Part A-র যত iteration-এর একটি ছোট ভগ্নাংশের পরে — এটিকে প্রকৃতপক্ষে অপূর্ণ রেখে। সেই আংশিক-প্রশিক্ষিত policy-টি, অপরিবর্তিত, ব্যবহার করে Part B প্রতি prompt-এ `N` সংখ্যক completion sample করে, প্রতিটির অঙ্ক-যোগফলকে তার "উত্তর" হিসেবে বের করে, এবং সমস্ত `N`-এর মধ্যে majority-vote উত্তর নেয় — `N` ∈ `{1, 3, 5, 9, 15, 25}`-এর জন্য, বহু স্বাধীন trial-এর উপর। প্রকৃত, মুদ্রিত নির্ভুলতা `N=1` থেকে `N=25`-তে যথেষ্ট উপরে উঠে, `N` বাড়ার সাথে সাথে per-sample উন্নতির হার ক্রমে ক্ষীণ হয় — [Phase 07 Lesson 2 section 4](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#4-the-condorcet-jury-theorem-why-voting-works--and-when-it-doesnt)-এর Condorcet Jury Theorem প্রতি কোনো sample-নির্ভুলতার জন্য সুযোগের উপরে যে diminishing-returns আকৃতি ভবিষ্যদ্বাণী করে, তা-ই। কারণ একটি ভুল অঙ্ক-ক্রম একটি একক প্রতিযোগী উত্তরের বদলে কয়েকটি ভিন্ন ভুল যোগফলের উপর পড়তে পারে, এটি সত্যিই [Lesson 2 section 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#5-beyond-binary-vote-splitting-makes-real-self-consistency-even-stronger)-এর multi-candidate plurality-voting পরিসর, সাধারণ binary ক্ষেত্রটি নয় — ভুল ভোট কয়েকটি ভুল যোগফলের মধ্যে পরস্পরের বিরুদ্ধে বিভক্ত হয়, যেটি কাঁচা per-sample নির্ভুলতা একা যা ইঙ্গিত করে তার চেয়ে সঠিক উত্তর জয়ী হওয়া সহজ হওয়ার একটি কারণ।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — Lesson 3-4 মানব পছন্দের বিরুদ্ধে optimize করে; reasoning model তার বদলে একটি verifiable, পরীক্ষাযোগ্য সত্যের বিরুদ্ধে
2. Verifiable reward বনাম শেখা reward model, এবং কেন প্রাক্তনটিকে পরবর্তীটির মতো reward-hack করা যায় না
3. GRPO: PPO-র critic সরিয়ে দিন, তার জায়গায় সত্যিকারের rollout থেকে গণনা করা একটি group-relative, per-prompt baseline বসান
4. `example.py` Part A demo 1-এর walkthrough — negative-advantage-fraction পরিমাপ, প্রকৃত সংখ্যা
5. `example.py` Part A demo 2-এর walkthrough — সহজ/মাঝারি/কঠিন prompt জুড়ে signal-scale ধারাবাহিকতা, প্রকৃত সংখ্যা
6. `example.py` Part A demo 3-এর walkthrough — সম্পূর্ণ GRPO update end to end প্রশিক্ষণ, এবং এই খেলনা কাজের স্কেল সম্পর্কে সৎ caveat
7. Test-time scaling একটি দ্বিতীয়, inference-time লিভার হিসেবে — Phase 03 Lesson 5-এর train-time scaling law থেকে স্বতন্ত্র
8. `example.py` Part B-এর walkthrough — ইচ্ছাকৃতভাবে অপূর্ণ একটি policy-তে majority-vote নির্ভুলতা বনাম N, Phase 07 Lesson 2-এর self-consistency ও vote-splitting-এ ফিরে বাঁধা

## আরও পড়ুন

- Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO প্রবর্তন করে)
- DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- OpenAI (2024), *Learning to Reason with LLMs* (o1 ঘোষণা, verifiable ফলাফলের বিরুদ্ধে RL প্রশিক্ষণ বর্ণনা করে)
- Snell, Lee, Xu, Kumar (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — ইতিমধ্যে [Phase 07 Lesson 2](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md)-তে উদ্ধৃত, এখানে test-time-scaling প্রক্রিয়া হিসেবে পুনর্বিবেচিত