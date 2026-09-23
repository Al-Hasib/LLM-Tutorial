# RLAIF and Constitutional AI

**ফেজ:** [Alignment and RLHF](../README.md) · **টপিক ফোল্ডার:** `05-RLAIF-and-Constitutional-AI`

## কেন এটি গুরুত্বপূর্ণ

এই ফেজে এ পর্যন্ত নির্মিত প্রতিটি পাইপলাইন একটি ব্যয়বহুল উপাদান ভাগ করে: [Lesson 2-এর reward model](../02-Reward-Modeling/README.md) এবং তার উপর নির্মিত দুইটি optimizer-ই ([Lesson 3-এর PPO](../03-RLHF-with-PPO/README.md) ও [Lesson 4-এর DPO](../04-Direct-Preference-Optimization-DPO/README.md)) **মানুষের** pairwise preference label-এর একটি dataset অনুমান করে। আধুনিক LLM-গুলোর প্রয়োজনের স্কেলে সেই label সংগ্রহ করা ধীর এবং ব্যয়বহুল; তাছাড়া মানুষ labeler-রা ক্লান্ত হয়ে পড়ে, মতভেদ করে, এবং হাজার হাজার comparison জুড়ে কতগুলো নীতি তারা ধারাবাহিকভাবে প্রয়োগ করতে পারে তার সীমাও আছে। **RLAIF** — Reinforcement Learning from AI Feedback (Bai et al., 2022; Lee et al., 2023) — প্রশ্ন করে যে একজন সক্ষম language model কি সেটি তৈরি করতে পারে, খরচের সামান্য ভগ্নাংশে। এই লেসন Lesson 2-4-এর প্রতিটি যন্ত্রপাতি হুবহু আগের মতো রাখে; শুধুমাত্র পরিবর্তন হয় *কে* "কোন response ভালো" এই label-টি তৈরি করে। **Constitutional AI** (Anthropic, Bai et al., 2022) এই ধারণার সবচেয়ে পরিপূর্ণ প্রকাশ: এটি শুধু preference labeler-ই নয়, মানুষের লেখা SFT demonstration-গুলোকেও প্রতিস্থাপন করে — একটি লিখিত **constitution** (নীতির একটি সুস্পষ্ট তালিকা) ব্যবহার করে সেই মানদণ্ড হিসেবে যার বিপরীতে একটি AI মডেল নিজের আউটপুট সমালোচনা ও উন্নত করে। এটি সরাসরি [Lesson 6](../06-Safety-Bias-and-Toxicity-Mitigation/README.md)-কে প্রস্তুত করে, যেখানে একটি লিখিত safety মানদণ্ড একটি একক training stage-এর বদলে সম্পূর্ণ deployment পাইপলাইন জুড়ে পদ্ধতিগতভাবে প্রয়োগ করা হয়।

## এই লেসন যা যা কভার করে

- RLAIF: Lesson 2-এর পাইপলাইনের মানব preference labeler-কে একটি AI বিচারক দিয়ে প্রতিস্থাপন
- Constitutional AI-এর দুইটি পর্যায়: SL-CAI (self-critique and revision) এবং RL-CAI (constitution-এর বিরুদ্ধে AI preference labeling)
- কেন একটি *লিখিত* constitution, বিশেষভাবে, Constitutional AI-এর AI feedback-কে একটি সীমাবদ্ধ "একটি মডেলকে জিজ্ঞাসা করুন এটি ভালো কিনা"-র চেয়ে বেশি scalable এবং auditable করে তোলে
- কীভাবে SL-CAI-এর সংশোধিত আউটপুট supervised fine-tuning data হয়ে যায়, এবং কীভাবে RL-CAI-এর AI preference label Lesson 2 ও 4-এর হুবহু একই Bradley-Terry / DPO যন্ত্রপাতিকে খাওয়ায়
- একটি hands-on প্রদর্শন: একটি rule-based critic (LLM বিচারকের পরিবর্ত) যা constitution লঙ্ঘন শনাক্ত করে, সেগুলো সংশোধন করে, এবং লঙ্ঘনের হার কমে যাওয়া মাপে, প্লাস একটি খেলনা AI preference labeler যা মূল ও সংশোধিত আউটপুটের মধ্যে বেছে নেয়

## 1. RLAIF: মানব labeler-এর বদলে একটি AI বিচারক

RLAIF পাইপলাইন কাঠামোগতভাবে RLHF-এর হুবহু অনুরূপ (Lesson 1 section 4-এর ডায়াগ্রাম): একটি prompt সংগ্রহ করুন, বর্তমান policy থেকে অল্পসংখ্যক প্রার্থী response তৈরি করুন, এবং তাদের উপর একটি pairwise preference তৈরি করুন। RLHF একজন মানুষকে সেই pairwise বিচার করতে বলে; RLAIF-এর বদলে একটি পৃথক (প্রায়শই বড় বা বিশেষভাবে নির্দেশিত) language model-কে দুইটি প্রার্থী দেখিয়ে পরে জিজ্ঞাসা করে কোনটি ভালো বেছে নিতে — ঐচ্ছিকভাবে একটি rubric সহ যা বর্ণনা করে এই কাজের জন্য "ভালো" মানে কী। সেই AI-উৎপন্ন preference label তখন হুবহু মানুষের label-এর মতো ব্যবহার হয়: [Lesson 2-এর Bradley-Terry loss](../02-Reward-Modeling/README.md#3-the-reward-model-loss) দিয়ে একটি reward model প্রশিক্ষণ দিতে, অথবা সরাসরি `(chosen, rejected)` triple হিসেবে [Lesson 4-এর DPO loss](../04-Direct-Preference-Optimization-DPO/README.md#3-the-dpo-loss)-এ দেওয়া যায়। Lee et al. (2023) দেখেছেন যে AI-লেবেলকৃত preference কয়েকটি কাজে নাটকীয়ভাবে কম labeling খরচে মানব-লেবেলকৃত RLHF-এর সমান বা তার চেয়ে ভালো হতে পারে — তবে AI বিচারক নিজের যে bias ও অন্ধদাগ আছে সেগুলোই উত্তরাধিকারসূত্রে পায়, যেটি একটি পাদটীকা নয়, একটি বাস্তব সীমাবদ্ধতা (Lesson 6 সরাসরি bias পরিমাপ নিয়ে আলোচনা করবে)।

## 2. Constitutional AI, পর্যায় 1: SL-CAI (self-critique and revision)

Constitutional AI-এর "মানুষের বদলে AI ব্যবহার করুন" ধারণাটি কেবল preference-labeling পদক্ষেপে নয়, একটি সম্পূর্ণ alignment পাইপলাইনে প্রয়োগ করে — supervised পর্যায় দিয়ে শুরু করে। একটি **constitution** দেওয়া থাকলে (সুস্পষ্ট natural-language নীতির একটি ছোট তালিকা, যেমন "সেই response-টি বেছে নিন যেটি অবৈধ বা বিপজ্জনক কার্যকলাপে উৎসাহিত করার সম্ভাবনা সবচেয়ে কম," বা "সেই response-টি বেছে নিন যেটি বেশি সম্মানজনক এবং user-কে অপমান করে না") — মডেলটিকেই একটি লুপে ব্যবহার করা হয়:

```
1. একটি (সম্ভবত adversarial) prompt-এর প্রাথমিক response তৈরি করুন
2. CRITIQUE:  মডেলকে বলুন response-টি কোন কোন উপায়ে একটি constitutional নীতি লঙ্ঘন করে তা শনাক্ত করতে
3. REVISE:    মডেলকে বলুন response-টি আবার লিখতে যাতে এটি আর সেই নীতি লঙ্ঘন না করে
4. প্রয়োজনে একাধিক নীতি / একাধিক critique-revise রাউন্ডে পুনরাবৃত্তি করুন
```

মডেলটিকে তারপর **নিজের সংশোধিত আউটপুটে fine-tune** করা হয় — [Phase 05 Lesson 4](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md)-এর হুবহু supervised fine-tuning, তবে `(prompt, good response)` জোড়াগুলি মানুষ scratch থেকে একটি demonstration লেখার বদলে একটি লিখিত মানদণ্ডের বিরুদ্ধে নিজেকে সমালোচনা ও সংশোধন করে মডেলটি তৈরি করেছে। এটিই SL-CAI-এর "SL": Constitution-AI-উৎপন্ন ডেটায় **s**upervised **l**earning।

## 3. Constitutional AI, পর্যায় 2: RL-CAI (constitution-এর বিরুদ্ধে AI preference labeling)

দ্বিতীয় পর্যায় হলো RLAIF — বিশেষভাবে constitution-কে বিচারের মানদণ্ড হিসেবে প্রয়োগ: মানুষের বদলে দুইটি response তুলনা করে, মডেলটিকেই (অথবা একটি পৃথক AI evaluator-কে) constitution-এর নীতি এবং একই prompt-এর দুইটি প্রার্থী response দেখানো হয়, এবং জিজ্ঞাসা করা হয় কোনটি সেগুলো ভালোভাবে পূরণ করে। এটি হুবহু সেই pairwise `(chosen, rejected)` preference data তৈরি করে যার চারপাশে [Lesson 2](../02-Reward-Modeling/README.md) গড়ে ওঠা — সাধারণ RLAIF (section 1)-এর থেকে একমাত্র পার্থক্য হলো বিচারের মানদণ্ডটি "গুণমান"-এর একটি অনির্দিষ্ট ধারণা নয়, বরং একটি স্পষ্ট, লিখিত, auditable নীতিসমূহের সেট। সেই AI-উৎপন্ন preference তখন একটি reward model প্রশিক্ষণ দেয় (বা সরাসরি DPO-কে খাওয়ায়), এবং RL fine-tuning [Lesson 3](../03-RLHF-with-PPO/README.md)-এর হুবহু মতোই এগোয় — তাই নাম "RL-CAI"। সম্পূর্ণ দুই-পর্যায়ের পাইপলাইনটি (SL-CAI, তারপর RL-CAI) Anthropic Claude-কে আরও harmless হতে প্রশিক্ষণের জন্য ব্যবহার করেছিল, মানুষের labeler-দের সরাসরি ক্ষতিকর বিষয়বস্তু পড়তে ও বিচার করতে না লাগিয়ে — যা একটি স্পষ্ট লক্ষ্যও ছিল: প্রশিক্ষণের সময় মানুষের রেটারদের কতটা বিরক্তিকর বিষয়বস্তুর সংস্পর্শে আসে তা কমানো।

## 4. কেন বিশেষভাবে একটি লিখিত constitution

Constitution শুধু একটি prompt-engineering কৌশল নয় — এটিই AI feedback-কে **scalable ও auditable** করে তোলে এমন একভাবে যেভাবে একটি সীমাবদ্ধ "এই response কি ভালো?" প্রশ্ন কখনো করে না: প্রতিটি নীতি স্পষ্টভাবে বর্ণিত, পৃথকভাবে পরীক্ষা, সংশোধন, যোগ বা অপসারণ করা যায় সিস্টেম নির্মাতাদের দ্বারা, এবং প্রতিটি critique বা preference বিচার নীতিগতভাবে খুঁজে বের করা যায় কোন নির্দিষ্ট নীতি এটিকে প্রেরণা দিয়েছে। এটি সাধারণ RLAIF (section 1)-এর থেকে একটি অর্থপূর্ণভাবে ভিন্ন নকশা-বিন্দু, যেখানে বিচারক মডেলের নিজস্ব প্রশিক্ষণে সেঁকা "ভালো"-এর একটি নির্বিচার, সম্ভবত implicit মানদণ্ড ব্যবহার করা যায়। `example.py` এটিকে একটি ছোট, স্পষ্ট, পরীক্ষাযোগ্য নীতিসমূহের সেট দিয়ে concrete করে — LLM বিচারকের বদলে ইচ্ছাকৃতভাবে rule-based pattern চেক-এ সরলীকৃত, যাতে "একটি লিখিত মানদণ্ডের বিরুদ্ধে সমালোচনা, তারপর সংশোধন, তারপর প্রভাব পরিমাপ" এই প্রক্রিয়াটি সম্পূর্ণ স্বচ্ছ হয়।

## 5. এই লেসনের খেলনা বাস্তবায়নের সৎ সীমাবদ্ধতা

`example.py`-এর critic ও AI preference labeler **rule-based pattern matcher**, language model নয় — তারা কেবল সেই নির্দিষ্ট টেক্সটুয়াল প্যাটার্নগুলো শনাক্ত করতে পারে যেগুলো ধরার জন্য লেখা হয়েছিল (একটি নির্দিষ্ট অপমান শব্দ, একটি নির্দিষ্ট অনিরাপদ রাসায়নিক সংমিশ্রণ, একটি নির্দিষ্ট overconfidence বাক্যাংশ)। একটি বাস্তব Constitutional AI সিস্টেম একটি প্রকৃত LLM ব্যবহার করে natural language-এ বর্ণিত একটি নীতির বিরুদ্ধে নির্বিচার টেক্সট পড়তে ও বিচার করতে, যা যেকোনো নির্দিষ্ট regex সেটের চেয়ে অনেক বেশি generalization করে। এখানে খেলনা সংস্করণটিকে সেই বিচার-সিদ্ধান্তের একটি সম্পূর্ণ স্বচ্ছ, পরীক্ষাযোগ্য *পরিবর্ত* হিসেবে পড়া উচিত — দুই-পর্যায়ের প্রক্রিয়াটি শুরু থেকে শেষ দেখাার জন্য দরকারি, বাস্তবসম্মত safety classifier হিসেবে নয় (toxicity classification-এর আরও যত্নশীল বিবেচনার জন্য [Lesson 6](../06-Safety-Bias-and-Toxicity-Mitigation/README.md) দেখুন)।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — মানব preference labeling একটি বাধা; মডেল কি এর বদলে label সরবরাহ করতে পারে?
2. RLAIF: RLHF-এর মতো একই pairwise-preference পাইপলাইন, মানব labeler-এর জায়গায় AI বিচারক প্রতিস্থাপিত (section 1)
3. Constitutional AI পর্যায় 1 (SL-CAI): একটি লিখিত constitution-এর বিরুদ্ধে self-critique and revision, তারপর সংশোধিত আউটপুটে fine-tuning (section 2)
4. Constitutional AI পর্যায় 2 (RL-CAI): একই constitution-এর বিরুদ্ধে AI preference labeling, সরাসরি Lesson 2/4-এর যন্ত্রপাতিকে খাওয়ানো (section 3)
5. কেন একটি স্পষ্ট, লিখিত constitution একটি সীমাবদ্ধ AI বিচারের চেয়ে বেশি scalable ও auditable (section 4)
6. `example.py`-এর rule-based constitution-এর walkthrough: তিনটি খেলনা নীতি এবং তাদের check/revise function
7. Live ফলাফল: critique-and-revise-এর আগে বনাম পরে লঙ্ঘনের হার, এবং মূল বনাম সংশোধিত আউটপুটের মধ্যে AI preference labeler-এর পছন্দ
8. সৎ caveat: এটি LLM critic-এর একটি rule-based পরিবর্ত, এবং Lesson 6-এর bias/toxicity পরিমাপের একটি প্রিভিউ

## আরও পড়ুন

- Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*
- Bai et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (RLHF baseline যার সাথে Constitutional AI তুলনা করে এবং যার উপর নির্মাণ করে)
- Lee et al. (2023), *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*
- Ganguli et al. (2022), *Red Teaming Language Models to Reduce Harms* (Anthropic-এর red-teaming পদ্ধতি, Lesson 6-এ আরও উল্লেখিত)