# Scaling Laws

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `05-Scaling-Laws`

## কেন এটি গুরুত্বপূর্ণ

এ পর্যন্ত প্রতিটি পাঠই ছিল *কী* তৈরি করতে হবে তা নিয়ে। এই পাঠটি এমন একটি প্রশ্ন নিয়ে, যা ঠিক সমান গুরুত্বপূর্ণ বলে প্রমাণিত হয়েছে: একটি নির্দিষ্ট compute budget-এ, মডেলটি কত বড় হওয়া উচিত, এবং আপনি কত data-তে training করা উচিত? অভিজ্ঞতামূলক উত্তরটি — শত শত training run-এ power-law curve ফিট করে আবিষ্কৃত — সরাসরি ব্যাখ্যা করে কেন LLM-গুলো এত বড় হয়ে গেল, কেন কিছু বিখ্যাত বড় মডেল (মূল 175B GPT-3-এর মতো) যুক্তিযুক্তভাবেই *undertrained* ছিল, এবং কেন আধুনিক LLM-গুলো প্রথম যুগের মডেলগুলোর তুলনায় প্রতি parameter-এ অনেক বেশি data-তে training করে।

## এই পাঠে যা শেখা হবে

- loss, model size, dataset size এবং compute-এর মধ্যে অভিজ্ঞতামূলক scaling-law সম্পর্ক
- Kaplan et al. (2020): মূল scaling laws, এবং তাদের "বড়ই ভালো" সুপারিশ
- Chinchilla (Hoffmann et al., 2022): compute-optimal সংশোধন
- Power-law loss সূত্র, এবং মডেলের performance ceiling আসলে কোথা থেকে আসে
- আজকের বাস্তব LLM-গুলো কীভাবে training হয় তার জন্য ব্যবহারিক প্রভাব

## 1. অভিজ্ঞতামূলক সম্পর্ক

বিভিন্ন সাইজের মডেল (`N` parameter) বিভিন্ন পরিমাণ data-তে (`D` token) training করুন, log-log স্কেলে চূড়ান্ত training loss-কে `N` এবং `D`-এর বিপরীতে প্লট করুন, আর একটি আকর্ষণীয়ভাবে পরিষ্কার প্যাটার্ন উঠে আসবে: loss **power law** হিসেবে কমে — log-log প্লটে একটি সরল রেখা, বহু order of magnitude জুড়ে মসৃণভাবে বিস্তৃত। Kaplan et al. (2020)-এর মূল অভিজ্ঞতামূলক আবিষ্কার ছিল এটি: scaling আচরণ লক্ষণীয়ভাবে *পূর্বাভাসযোগ্য*, অর্থাৎ আপনি extrapolate করতে পারেন: কয়েকটি ছোট, সস্তা মডেল training করে curve-টি ফিট করুন, এবং অনেক বড়, ব্যয়বহুল মডেল training করার আগেই এর পারফরম্যান্স পূর্বাভাস করুন।

## 2. Power-law loss সূত্র

একটি ব্যাপক ব্যবহৃত functional form (Chinchilla-এর সংস্করণ) প্রাপ্তযোগ্য loss-কে তিনটি additive অংশে বিশ্লেষণ করে:

```
L(N, D) = E + A / N^α + B / D^β
```

- `E` — **irreducible loss**: natural language-এর নিজস্ব entropy; কোনো পরিমাণ স্কেল loss-কে এই floor-এর নিচে নামাতে পারে না।
- `A / N^α` — **সসীম parameter সংখ্যা** থাকার loss penalty; `N` বাড়ার সাথে সাথে কমে।
- `B / D^β` — **সসীম training data** থাকার loss penalty; `D` বাড়ার সাথে সাথে কমে।

`example.py` এই সঠিক সূত্রটি (প্রকাশিত Chinchilla-fitted constants ব্যবহার করে) ফিট ও ভিজ্যুয়ালাইজ করে trade-off-টিকে কংক্রিট করতে।

## 3. Kaplan et al. (2020): মূল সুপারিশ

প্রথম বড় scaling-laws paper খুঁজে পেয়েছিল যে, একটি নির্দিষ্ট compute budget-এ, dataset বাড়ানোর চেয়ে মডেল বড় করলে loss দ্রুত উন্নত হয় — যা ব্যবহারিক সুপারিশে পৌঁছায়: **খুব বড় মডেল training করুন, এবং তুলনামূলকভাবে পরিমিত data-তে এটিকে পূর্ণ convergence পর্যন্ত training না করার চিন্তা করবেন না।** GPT-3 (175B parameter, ~300B token-এ training) মূলত এই নির্দেশনাই অনুসরণ করে তৈরি হয়েছিল।

## 4. Chinchilla (2022): মডেলগুলো undertrained ছিল

Hoffmann et al. আরও বিস্তৃত, আরও সতর্কভাবে নিয়ন্ত্রিত training-run জরিপ দিয়ে প্রশ্নটি পুনর্বিবেচনা করে ভিন্ন সিদ্ধান্তে পৌঁছান: একটি নির্দিষ্ট compute budget-এ, **model size এবং dataset size মোটামুটি একসাথে স্কেল হওয়া উচিত** — মোটামুটিভাবে, প্রতি parameter-এ প্রায় 20টি training token — model size-কে এত বেশি প্রাধান্য না দিয়ে। তাদের শিরোনামের ফলাফল: **1.4 ট্রিলিয়ন token-এ training করা 70B-parameter মডেল** ("Chinchilla") অনেক বড় **280B-parameter Gopher** মডেল-কে ছাড়িয়ে গিয়েছিল, যেটি শুধু ~300B token-এ training হয়েছিল, উভয়ের জন্য **একই মোট training compute** ব্যবহার করে। এই বিশ্লেষণ অনুযায়ী, GPT-3-স্কেলের মডেলগুলো তাদের parameter সংখ্যার তুলনায় উল্লেখযোগ্যভাবে *undertrained* ছিল — একই compute budget ব্যবহারের জন্য ছোট মডেলটিকে বেশি দিন training করা আরও ভালো ব্যবহার হতো।

## 5. কেন আসল মুদ্রা শুধু parameter নয়, compute

দুটি paper-ই প্রশ্নটিকে একইভাবে ফ্রেম করে: compute budget `C` (মোটামুটি `C ≈ 6ND` FLOPs, যেহেতু প্রতিটি token-এর forward pass-এ মোটামুটি `2N` FLOPs এবং backward pass-এ `4N` FLOPs লাগে) দিলে, আপনি কীভাবে তা `N` এবং `D`-এর মধ্যে ভাগ করবেন? `example.py` সরাসরি এই optimization-টি সম্পাদন করে: বিভিন্ন compute budget-এর জন্য, এটি `C = 6ND` সন্তুষ্টকারী সম্ভাব্য `(N, D)` ভাগগুলো খোঁজে এবং power-law সূত্রের অধীনে `L(N, D)` কে সর্বনিম্ন করে এমনটি বের করে — শুধু সূত্র থেকেই Chinchilla অভিজ্ঞতামূলকভাবে যে গুণগত সিদ্ধান্তে পৌঁছেছিল সেটিই ফিরে পায়।

## 6. ব্যবহারিক প্রভাব

এজন্যই Chinchilla-র পরে প্রকাশিত প্রায় প্রতিটি LLM (LLaMA, Mistral এবং বেশিরভাগ অন্যগুলো) "মাত্র" 7B-70B parameter-এর মাত্রাতেও *প্রতি মডেলে* শত শত বিলিয়ন থেকে ট্রিলিয়ন token-এ training করে — GPT-3-যুগের training রেসিপি থেকে মারাত্মক প্রস্থান। এটিও কেন ছোট, "Chinchilla-optimal-এর বাইরে over-trained" মডেলগুলো ব্যবহারিক ডিপ্লয়মেন্টের জন্য জনপ্রিয় হয়ে উঠল: Chinchilla-optimal একটি নির্দিষ্ট loss-এর জন্য *training* compute-কে সর্বনিম্ন করে, কিন্তু *inference* সময়ে চালানো সস্তা এমন একটি ছোট মডেল প্রায়ই সূত্রের training-compute-optimal বিন্দুর পরামর্শের চেয়েও বেশি data-তে training করা মূল্যবান, কারণ inference খরচ (প্রতিটি user request-এ একবার, চিরকালের জন্য) ডিপ্লয়মেন্ট স্কেলে training খরচকে (একবার) ছাপিয়ে যেতে পারে।

## Video Script Outline

1. Motivation — "একটি নির্দিষ্ট budget-এ, ব্যয় করার সবচেয়ে চতুর উপায় কী?"
2. Power-law আবিষ্কার: loss বনাম N ও D, log-log সরল রেখা
3. Additive loss সূত্র: irreducible + finite-N + finite-D টার্ম
4. Kaplan-এর "আরও বড় করো" যুগ বনাম Chinchilla-এর সংশোধন, একটি before/after গল্প হিসেবে
5. `C ≈ 6ND` compute-budget ফ্রেমিং
6. `example.py`-এর ওয়াকথ্রু — সূত্রটি ফিট করা, তারপর বেশ কয়েকটি budget-এ compute-optimal `(N, D)` সমাধান করা
7. Recap: কেন post-Chinchilla LLM-গুলো GPT-3-এর চেয়ে অনেক বেশি data-তে training করে, এবং training বনাম inference-cost মোচড়

## Further Reading

- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla)
- Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models* (সস্তা inference-এর জন্য Chinchilla-optimal-এর বাইরে training-এর একটি স্পষ্ট বাস্তব-বিশ্ব প্রয়োগ)