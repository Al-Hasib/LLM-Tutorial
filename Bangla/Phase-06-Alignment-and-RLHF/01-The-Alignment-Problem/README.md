# Alignment সমস্যা (The Alignment Problem)

**ফেজ:** [Alignment and RLHF](../README.md) · **টপিক ফোল্ডার:** `01-The-Alignment-Problem`

## কেন এটি গুরুত্বপূর্ণ

এ পর্যন্ত প্রতিটি লেসনের মূল বিষয় ছিল এমন একটি মডেল তৈরি করা যা ঠিক একটি কাজে ভালো: টেক্সটের পরের token (next token) পূর্বাভাস করা — যেমন [Phase 04: Pretraining LLMs](../../Phase-04-Pretraining-LLMs/README.md) লেসনে প্রশিক্ষণ দেওয়া হয় যে objective ব্যবহার করে, যেটি প্রথমবার [Phase 01 Lesson 1](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md#1-what-a-language-model-actually-is)-এ প্রবর্তিত হয়েছিল এবং [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)-তে একটি বাস্তব architecture-এর উপর end-to-end প্রশিক্ষিত হয়েছিল। সেই objective — "এখন পর্যন্ত যতটুকু টেক্সট দেখা হয়েছে, তার পরের token কী হওয়ার সম্ভাবনা *এই ধরনের টেক্সটে* সবচেয়ে বেশি?" — কাঁচা টেক্সট থেকে grammar, তথ্য, reasoning pattern এবং স্টাইল শেখার জন্য অসাধারণভাবে শক্তিশালী। কিন্তু এটি নীরবে **একই objective নয়** যেটি হলো "একজন সহায়ক, সৎ assistant হয়ে user-এর প্রশ্নের উত্তর দেওয়া।" শুধুমাত্র সেই objective দিয়ে প্রশিক্ষিত একটি মডেলের "helpful" হওয়ার কোনো ধারণাই থাকে না; সে কেবল "বিশ্বাসযোগ্য ধারাবাহিকতা (plausible continuation)" জানে। এই লেসন সেই ব্যবধানটিকে concrete করে তোলে এবং ব্যাখ্যা করে কেন এই ফেজের বাকি পুরো অংশ ([reward modeling](../02-Reward-Modeling/README.md), RLHF, DPO, RLAIF/Constitutional AI এবং safety mitigation) pretraining ও instruction tuning-এর পরে যুক্ত একটি **আলাদা training stage** হিসেবে বিদ্যমান ([Phase 05 Lesson 4: Instruction Tuning (SFT)](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md))।

## এই লেসন যা যা কভার করে

- কেন শুধুমাত্র next-token prediction একটি assistant তৈরি করে না
- অনুশীলনে "aligned" বলতে কী বোঝায় তার helpful, honest, harmless (HHH) কাঠামো
- কাঁচা pretrained ("base") মডেলকে chatbot হিসেবে ব্যবহার করলে যে তিনটি স্বতন্ত্র failure mode দেখা যায়
- কেন SFT ব্যবধান কমায় কিন্তু পুরোপুরি বন্ধ করে না, এবং কেন আরও একটি preference-ভিত্তিক stage প্রয়োজন
- একটি hands-on প্রদর্শন: unlabeled, unstructured টেক্সটে একটি ছোট base model প্রশিক্ষণ দেওয়া এবং দেখা যে এটি একটি প্রশ্নের "উত্তর" দিতে ব্যর্থ হয় — এবং তার সৎ কারণ হলো, উত্তর কেমন দেখতে লাগে তা তাকে কখনো দেখানোই হয়নি

## 1. Pretraining objective, নির্ভুলভাবে পুনর্ব্যক্ত

একটি pretrained language model নিম্নোক্ত বিষয়টি সর্বাধিক করতে প্রশিক্ষিত হয়:

```
P(token_t | token_1, ..., token_{t-1})
```

বই, ওয়েবসাইট, ফোরাম, কোড ইত্যাদি থেকে সংগ্রহ করা বিশাল কাঁচা টেক্সট কর্পাসের উপর ([Phase 04 Lesson 1: Pretraining Data Pipeline](../../Phase-04-Pretraining-LLMs/01-Pretraining-Data-Pipeline/README.md))। সেই objective-এর ভেতরে এমন কিছুই নেই যা "প্রশ্নের সহায়ক উত্তর দেয় এমন টেক্সট"-কে "আরও প্রশ্ন দিয়ে প্রশ্ন চালিয়ে যাওয়া টেক্সট," "কথা শেষ না করে ঝিমিয়ে পড়া টেক্সট" বা "ক্ষতিকর অনুরোধকে ক্ষতিকর বিষয়বস্তু দিয়ে চালিয়ে যাওয়া টেক্সট" থেকে আলাদা করে — মডেল শুধুমাত্র তার *training distribution-এর পরিসংখ্যানের* সাথে মিল রাখার জন্যই পুরস্কৃত হয়। যদি training distribution-এ এমন FAQ তালিকা থাকে যেখানে প্রশ্নের পর আরও প্রশ্ন আসে, এমন ফোরাম থ্রেড থাকে যেখানে অনুরোধের পর ঘুরিয়ে দেওয়া হয়, বা ক্ষতিকর টেক্সটের পর আরও ক্ষতিকর টেক্সট থাকে, তাহলে মডেল ঠিক একই আনন্দে সেই প্যাটার্নগুলো শিখবে যত আনন্দে সে একটি সহায়ক উত্তর শিখত — কারণ **loss function-এর দৃষ্টিকোণ থেকে এগুলো ঠিক একই ধরনের prediction কাজ।** একটি base model হলো একটি অত্যন্ত সক্ষম *next-token statistics engine*, কোনো "এই ব্যক্তিকে সাহায্য করো"-ধরনের উদ্দেশ্যসহ agent নয়।

## 2. কাঁচা base model-এর তিনটি concrete failure mode

কাঁচা pretrained মডেলকে (কোনো instruction tuning নেই, কোনো alignment নেই) যেন chatbot, সেইভাবে prompt করলে সাধারণত কয়েকটি বৈশিষ্ট্যপূর্ণ ব্যর্থতার একটি দেখা যায়:

1. **উত্তরের বদলে প্রশ্ন চালিয়ে যাওয়া।** যদি training data-তে প্রশ্ন-সদৃশ টেক্সটের পরে সাধারণত আরও প্রশ্ন-সদৃশ টেক্সট আসে (একটি FAQ পেজ, একটি জরিপ, আলোচনার prompt-এর তালিকা), তাহলে একটি প্রশ্নের পরিসংখ্যানগতভাবে সবচেয়ে সম্ভাব্য ধারাবাহিকতা হলো *আরেকটি প্রশ্ন*, উত্তর নয় — কারণ মডেল সেটাই আসলে দেখেছে।
2. **থেমে যাওয়া বা পুনরাবৃত্তি করা।** যদি training data-তে "প্রশ্নের ঠিক পরেই সরাসরি, সংক্ষিপ্ত উত্তর" এই প্যাটার্নটি খুব কমই থাকে, তাহলে মডেলের অনুকরণ করার মতো কিছুই থাকে না এবং সে কেবল সাধারণ, প্রসঙ্গ-বহির্ভূত টেক্সট দিয়ে চালিয়ে যেতে পারে।
3. **ক্ষতিকর বিষয়বস্তুকে ক্ষতিকরভাবে চালিয়ে যাওয়া।** যদি একটি ক্ষতিকর prompt-এর পরিসংখ্যানগতভাবে সম্ভাব্য ধারাবাহিকতা (কারণ training data-তে এমন prompt-এর পর ক্ষতিকর টেক্সট থাকে — ফোরাম, কল্পকাহিনি, বিতর্ক) আরও ক্ষতিকর টেক্সট হয়, তাহলে কাঁচা base model-এর প্রত্যাখ্যান করার কোনো বিল্ট-ইন প্রক্রিয়া নেই — প্রত্যাখ্যান হলো একটি *আচরণ* যা স্পষ্টভাবে শেখাতে হয়, এটি ডিফল্ট নয়।

এই তিনটি ব্যর্থতারই একটি মূল কারণ আছে: **মডেল ঠিক সেটাই করছে যেটা করতে তাকে প্রশিক্ষণ দেওয়া হয়েছিল (বিশ্বাসযোগ্য ধারাবাহিকতা পূর্বাভাস করা); "helpful," "honest" এবং "harmless" — এই তিনটির কোনোটি কখনো সেই training signal-এর অংশ ছিল না।**

## 3. HHH কাঠামো

Anthropic-এর alignment কাঠামো (Askell et al., 2021, *A General Language Assistant as a Laboratory for Alignment*) লক্ষ্য আচরণকে তিনটি বৈশিষ্ট্যে সাজায়, যেগুলো প্রায়শই **HHH** নামে সংক্ষিপ্ত করা হয়:

- **Helpful (সহায়ক)** — সত্যিই user-এর অনুরোধ পূরণের চেষ্টা করা, প্রয়োজন হলে স্পষ্টীকরণমূলক প্রশ্ন করা, সরাসরি এবং কার্যকর উত্তর দেওয়া।
- **Honest (সৎ)** — মডেলের প্রকৃত সেরা অনুমান জানানো, ক্রমাঙ্কিত (calibrated) অনিশ্চয়তা প্রকাশ করা এবং তথ্য উদ্ভাবন এড়ানো ([hallucination সমস্যা](../../Phase-08-Evaluation-of-LLMs/04-Hallucination-and-Factuality-Evaluation/README.md)-এর একটি রূপ, যা আংশিকভাবে pretraining-বনাম-alignment সমস্যাও বটে)।
- **Harmless (নিরাপদ)** — বাস্তব ক্ষতি করবে এমন অনুরোধ প্রত্যাখ্যান বা নিরাপদে ঘুরিয়ে দেওয়া, তবে এতটা সতর্ক হয়ে যাওয়াও নয় যে helpful থাকা বন্ধ হয়ে যায়।

HHH-র কোনো অংশই pretraining loss থেকে অনুমিত হয় না। প্রতিটিই একটি *আচরণগত নির্দিষ্টকরণ (behavioral specification)* যা আলাদাভাবে শেখাতে হয় — এবং, সবচেয়ে গুরুত্বপূর্ণ, তিনটি বৈশিষ্ট্য প্রায়শই **পরস্পরের সাথে সংঘাতে** থাকে (একটি অতিসতর্ক মডেল যে সব কিছু প্রত্যাখ্যান করে সে harmless কিন্তু helpful নয়; যে মডেল কখনো সন্দেহ প্রকাশ করে না সে অসহায়ভাবে overconfident এবং, সূক্ষ্ম একটি অর্থে, কম honest)। এই সংঘাতের ভারসাম্য বজায় রাখাই এই ফেজের বাকি অংশের মূল বিষয়।

## 4. কেন SFT ব্যবধান কমায় কিন্তু পুরোপুরি বন্ধ করে না

Instruction tuning / SFT ([Phase 05 Lesson 4](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md)) base model-কে curated `(instruction, good response)` জোড়ার উপর fine-tune করে, যা সরাসরি সহায়ক-উত্তর আচরণের *ফরম্যাট ও অস্তিত্ব* শেখায় — একা এটি কাঁচা base model-এর তুলনায় বিশাল উন্নতি। কিন্তু SFT মডেলটিকে কেবল *ভালো আচরণের উদাহরণ* দেখায় যা তাকে অনুকরণ করতে হবে; এটি কখনো বলে না *একটি response আরেকটির তুলনায় কতটা ভালো*, কখনো training set-এ না থাকা সূক্ষ্ম-খারাপ-কিন্তু-বিশ্বাসযোগ্য response-কে শাস্তি দেয় না, এবং প্রশিক্ষণ শেষে আবিষ্কৃত মডেলের নিজস্ব idiosyncratic failure mode সংশোধনের কোনো প্রক্রিয়া দেয় না। তার জন্য mডেলের *নিজের* আউটপুটের উপর তুলনামূলক feedback (comparative feedback) প্রয়োজন — আর ঠিক সেটাই [Lesson 2: Reward Modeling](../02-Reward-Modeling/README.md) এবং [Lesson 3: RLHF with PPO](../03-RLHF-with-PPO/README.md) উপর থেকে যোগ করে।

## 5. `example.py` যা প্রদর্শন করে

এই ব্যবধানকে শুধু দাবি না করে অনস্বীকার্য করতে, `example.py` একটি সত্যিকারের ছোট GPT-স্টাইল মডেলকে ([Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)-এর ঠিক সেই decoder-only architecture) একটি ছোট, ইচ্ছাকৃতভাবে **unstructured, কাঁচা, ইন্টারনেট-সদৃশ** খেলনা কর্পাসে প্রশিক্ষণ দেয়: সাধারণ বিবৃতি ও প্রশ্নের মিশ্রণ, যেখানে **কোনো instruction formatting নেই এবং data-র কোথাও একটি প্রশ্নের সরাসরি উত্তর দেওয়ার কোনো উদাহরণ নেই**। প্রশিক্ষণের পর মডেলটিকে একটি সরাসরি তথ্যগত প্রশ্ন দিয়ে prompt করা হয়। যেহেতু training data-তে "প্রশ্ন -> সরাসরি উত্তর" প্যাটার্ন কখনো ছিল না, তাই মডেল সেটি তৈরি করতে পারে না — বদলে সে উপরের failure mode-গুলোর ঠিক যেমনটি ভবিষ্যদ্বাণী করে তেমনটিই করে: সে আরেকটি প্রশ্নের আকারের লাইন বা সম্পর্কহীন বিবৃতি দিয়ে চালিয়ে যায়, কখনো তথ্যগত উত্তর দেয় না। এটি এমন কোনো "মূর্খ মডেল" সমস্যা নয় যা শুধু স্কেল বাড়ালেই ঠিক হয়ে যাবে; এটি training objective ও training data-তে কী ছিল এবং কী ছিল না — তার একটি সরাসরি, যান্ত্রিক পরিণতি।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "pretraining শেখায় next-token prediction; এটি কখনো 'সহায়ক হও' শেখায় না"
2. Pretraining objective পুনর্ব্যক্ত, এবং কেন এটি helpful টেক্সটকে অন্য যেকোনো পরিসংখ্যানগতভাবে সম্ভাব্য টেক্সট থেকে আলাদা করতে পারে না
3. তিনটি failure mode: আরও প্রশ্ন দিয়ে উত্তর দেওয়া, থেমে যাওয়া, ক্ষতিকর টেক্সট চালিয়ে যাওয়া
4. HHH কাঠামো (helpful, honest, harmless) এবং তিনটির মধ্যকার সংঘাত
5. কেন SFT সাহায্য করে কিন্তু একা যথেষ্ট নয়
6. `example.py`-এর live walkthrough — ছোট base model-টি প্রশিক্ষণ দিন, এটিকে একটি সত্যিকারের প্রশ্ন দিয়ে prompt করুন, এবং একটি যান্ত্রিকভাবে স্পষ্ট কারণে এটিকে উত্তর দিতে ব্যর্থ হতে দেখুন
7. প্রিভিউ: Lesson 2-6 যন্ত্রপাতি তৈরি করে (reward model, RLHF, DPO, RLAIF, safety) যা আসলে এই ব্যবধান বন্ধ করে

## আরও পড়ুন

- Askell et al. (2021), *A General Language Assistant as a Laboratory for Alignment* (HHH কাঠামো)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — যে পেপারটি সম্পূর্ণ pretraining -> SFT -> RLHF পাইপলাইন জনপ্রিয় করেছে)
- Bai et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*
- Bender, Gebru et al. (2021), *On the Dangers of Stochastic Parrots* (pretrained LM-গুলো আসলে কী "বোঝে" এবং কী বোঝে না — তার একটি সমালোচনামূলক দৃষ্টিভঙ্গি)