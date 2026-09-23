# Safety, Bias এবং Toxicity প্রশমন (Mitigation)

**ফেজ:** [Alignment and RLHF](../README.md) · **টপিক ফোল্ডার:** `06-Safety-Bias-and-Toxicity-Mitigation`

## কেন এটি গুরুত্বপূর্ণ

এই ফেজ একটি মডেলের *আচরণকে* মানুষ যেটা পছন্দ করে (বা, [Lesson 5](../05-RLAIF-and-Constitutional-AI/README.md) অনুযায়ী, একজন AI যা তাদের প্রতিনিধিত্ব করে) সেদিকে চালনার জন্য ক্রমশ পরিশীলিত একটি যন্ত্রপাতি সেট তৈরি করেছে: একটি reward model ([Lesson 2](../02-Reward-Modeling/README.md)), তার বিরুদ্ধে RL fine-tuning ([Lesson 3](../03-RLHF-with-PPO/README.md)), এবং একই objective-এর জন্য একটি সরলীকৃত সরাসরি optimizer ([Lesson 4](../04-Direct-Preference-Optimization-DPO/README.md))। এই সমাপনী লেসন "কীভাবে একটি preference signal-এর দিকে optimize করব?" তার চেয়ে ভিন্ন প্রশ্ন করে: **কীভাবে জানবো মডেলটি কোথায় ব্যর্থ হচ্ছে, এবং একবার খুঁজে পেলে কীভাবে সত্যিই ঠিক করব** — সেই training stage-গুলোর আগে, সময়ে ও পরে, এমনকি deployment-এর পরেও। এটি তিনটি স্বতন্ত্র কিন্তু পরিপূরক অনুশীলনকে একত্র করে — red-teaming (ব্যর্থতা খোঁজা), bias measurement (পদ্ধতিগত অবিচার পরিমাপ) এবং toxicity classification (ক্ষতিকর বিষয়বস্তু ছাঁকা) — এবং Lesson 2-5-এর প্রতিটি alignment কৌশলকে পাইপলাইনের এক নির্দিষ্ট stage-এ প্রয়োগকৃত একটি সম্ভাব্য *প্রশমন* হিসেবে স্থাপন করে, একমাত্র উপলব্ধ লিভার হিসেবে নয়।

## এই লেসন যা যা কভার করে

- Red-teaming: deployment-এর আগে failure mode খুঁজতে ইচ্ছাকৃতভাবে adversarial input দিয়ে একটি মডেল পরীক্ষা
- Bias measurement: বিভিন্ন জনতাতাত্ত্বিক গোষ্ঠীতে মডেল আচরণের পদ্ধতিগত পার্থক্য শনাক্ত করার template-ভিত্তিক probe
- Toxicity classifier: LLM-থেকে আলাদা, generation-কে স্কোর বা ছাঁকা নিবেদিত মডেল
- পাইপলাইন জুড়ে যেখানে প্রশমন ঘটতে পারে: pretraining data filtering, RLHF/DPO steering, এবং inference-time guardrail
- একটি hands-on প্রদর্শন: একটি নিয়ন্ত্রিত bias probe যা একটি পরিচিত injected bias পুনরুদ্ধার করে, এবং একটি বাস্তব toxicity classifier যা scratch থেকে প্রশিক্ষিত এবং held-out data-তে মূল্যায়িত

## 1. Red-teaming

**Red-teaming** হলো ইচ্ছাকৃতভাবে একটি মডেলকে ব্যর্থ করানোর চেষ্টা করার অনুশীলন — প্রকৃত user-রা কখনো সেগুলো দেখার আগেই ক্ষতিকর, biased বা অবাঞ্ছিত আউটপুট বের করে আনার জন্য বিশেষভাবে নকশা করা adversarial, edge-case বা প্রতারণামূলকভাবে বাক্যগঠিত input দিয়ে পরীক্ষা। [Phase 08](../../Phase-08-Evaluation-of-LLMs/README.md)-এর benchmark-স্টাইল মূল্যায়নের মতো নয় — যেখানে সাধারণত input-এর একটি নির্দিষ্ট, প্রতিনিধিত্বমূলক distribution-এ কর্মক্ষমতা মাপা হয় — red-teaming নকশা অনুযায়ীই adversarial: পরীক্ষকরা (মানুষ অথবা, ক্রমবর্ধমানভাবে, অন্য language model, Ganguli et al., 2022 অনুযায়ী) সক্রিয়ভাবে সেই prompt-গুলো খোঁজে যা মডেল ভেঙে দেয়, যার মধ্যে safety training ঘুরিয়ে দেওয়ার চেষ্টা করা jailbreak-স্টাইল prompt, ক্ষতিকর intent গোপন করার জন্য বাক্যগঠিত অনুরোধ, এবং বিশেষভাবে পরিচিত মডেল দুর্বলতাগুলোকে লক্ষ্য করা prompt অন্তর্ভুক্ত। একটি red-teaming অনুশীলনের আউটপুট হলো concrete failure case-গুলির একটি সেট; প্রতিটি তারপর section 4-এর প্রশমন stage-গুলোর একটিতে আবার ফিরে যায় — একটি red-teamed ব্যর্থতা [DPO](../04-Direct-Preference-Optimization-DPO/README.md)-র জন্য একটি নতুন preference-pair উদাহরণ, [Constitutional AI](../05-RLAIF-and-Constitutional-AI/README.md)-র জন্য একটি নতুন constitutional principle, অথবা একটি toxicity classifier-এর (section 3) জন্য একটি নতুন প্রশিক্ষণ উদাহরণ হতে পারে।

## 2. Template-ভিত্তিক probe দিয়ে bias পরিমাপ

এই প্রসঙ্গে **bias** বলতে বোঝায় একটি মডেলের আউটপুট বা সম্পর্কের *পদ্ধতিগত* পার্থক্য, এমন অন্যথায় সমতুল্য input-গুলির মধ্যে যেগুলি শুধুমাত্র একটি জনতাত্ত্বিক বৈশিষ্ট্যে পৃথক — যেমন, একটি বাক্যে এক লিঙ্গের উল্লেখ করলে যে sentiment বা competence স্কোর আসে তা অন্য লিঙ্গের উল্লেখে ভিন্ন হয়, বাক্যের বাকি সবকিছু স্থির রেখে। মানক পরিমাপ পদ্ধতি হলো একটি **template-ভিত্তিক probe**: একটি জনতাত্ত্বিক পদের জন্য একটি স্লটসহ (সর্বনাম, নাম, জাতীয়তা ইত্যাদি) একটি বাক্য template তৈরি করুন, বাক্যের প্রতিটি অন্যান্য বিবরণ সব গোষ্ঠীতে স্থির ও identically distributed রেখে প্রতিটি প্রার্থী পদ দিয়ে পূরণ করুন, প্রতিটি পূরণকৃত বাক্যে মনিটর করা মডেল বা scorer চালান, এবং গোষ্ঠীর মধ্যে ফলে আসা স্কোর (বা উৎপন্ন ধারাবাহিকতা) distribution তুলনা করুন। যেহেতু input-এর প্রতিটি *বৈধ* বিষয় গঠনকাঠামো অনুযায়ী গোষ্ঠীর মধ্যে ধ্রুব রাখা হয়, তাই আউটপুটে অবশিষ্ট যেকোনো পদ্ধতিগত পার্থক্য শুধুমাত্র জনতাত্ত্বিক পদের জন্য দায়ী — এটি অবিকল সেই যুক্তি যা Bertrand ও Mullainathan-এর (2004) resume-callback গবেষণা এবং NLP-নির্দিষ্ট probe যেমন Rudinger et al.-এর (2018) Winogender schema এবং Caliskan et al.-এর (2017) WEAT-এর মতো প্রতিষ্ঠিত fairness অডিটগুলোর পেছনে। `example.py` এর একটি নিয়ন্ত্রিত সংস্করণ চালায়: একটি খেলনা scorer যাতে *পরিচিত* আকারের একটি bias ইচ্ছাকৃতভাবে injected, যাতে probe-এর শনাক্তকৃত ব্যবধান শুধু বিশ্বাস করার বদলে সরাসরি ground truth-এর বিরুদ্ধে যাচাই করা যায়।

## 3. Toxicity classifier

একটি **toxicity classifier** হলো একটি পৃথক, নিবেদিত মডেল — সাধারণত তার পাশে কাজ করা LLM-এর চেয়ে অনেক ছোট ও সস্তা — যা টেক্সটের একটি অংশ নিয়ে এটি toxic, harassing বা অন্যথায় ক্ষতিকর কিনা নির্দেশক একটি স্কোর (বা binary label) আউটপুট করতে প্রশিক্ষিত। [Lesson 2](../02-Reward-Modeling/README.md)-এর reward model-এর মতো নয় — যেটি "কোন response একজন মানুষ পছন্দ করবে"-এর একটি বিস্তৃত ধারণা শেখে — একটি toxicity classifier একটি নির্দিষ্ট বৈশিষ্ট্যে সংকীর্ণভাবে সীমাবদ্ধ, toxic ও non-toxic টেক্সটের labeled উদাহরণে সাধারণ supervised classification দিয়ে প্রশিক্ষিত (pairwise preference নয়) — স্থাপত্যিকভাবে এটি bag-of-words feature-এর উপর logistic regression-এর মতো সহজ বা fine-tuned Transformer-এর মতো পরিশীলিত হতে পারে (যেমন Google-এর Perspective API, বা Meta-এর Llama Guard)। এটি দুইটি ভিন্ন বিন্দুতে মোতায়েন করা যায়: **training-data-filtering সময়**, LLM সেগুলো দেখার আগেই একটি pretraining বা fine-tuning কর্পাস থেকে toxic document স্কোর করে সরিয়ে ফেলা, অথবা **inference সময়**, একটি guardrail হিসেবে যা LLM-এর নিজস্ব generation একটি user-এর কাছে পৌঁছানোর ঠিক আগে ছাঁকে। `example.py` হুবহু এই classifier-টি scratch থেকে implement করে — bag-of-words feature-এর উপর logistic regression, একটি ছোট labeled খেলনা dataset-এ প্রশিক্ষিত — এবং প্রশিক্ষণের সময় কখনো না দেখা বাক্যাংশে precision ও recall দিয়ে মূল্যায়ন করে, তারপর সেই একই প্রশিক্ষিত মডেলটিকে প্রার্থী আউটপুটের একটি নতুন batch-এ inference-time ফিল্টার হিসেবে পুনরায় ব্যবহার করে।

একটি সুপরিচিত ও গুরুত্বপূর্ণ caveat, Dixon et al. (2018) দ্বারা নথিভুক্ত: বাস্তব-বিশ্বের ডেটায় প্রশিক্ষিত toxicity classifier একটি নির্দিষ্ট পরিচয়-শব্দ ও toxicity-র মধ্যে একটি *ভুয়া* (spurious) সম্পর্ক শিখতে পারে (একটি নির্দিষ্ট পরিচয় গোষ্ঠীর উল্লেখ করে এমন একটি নিরপেক্ষ বাক্যকে, সমতুল্য নিরপেক্ষ বাক্যের চেয়ে বেশি toxic স্কোর করা), যা নিজেই section 2-এর probe পদ্ধতি যে ধরনের পদ্ধতিগত bias ধরার জন্য নকশা করা হয়েছে — bias measurement ও toxicity classification পরিপূরক অনুশীলন, স্বাধীন নয়, এবং একটি toxicity classifier নিজেও একটি মডেল যাকে safety প্রক্রিয়া হিসেবে বিশ্বাস করার আগে bias-probe করা উচিত।

## 4. পাইপলাইন জুড়ে প্রশমন

এই ফেজের প্রতিটি কৌশল — এবং এই লেসনের প্রতিটি কৌশল — সামগ্রিক LLM পাইপলাইনের একটি *নির্দিষ্ট stage*-এ প্রয়োগকৃত একটি প্রশমন, এবং প্রতিটি কোন stage-কে লক্ষ্য করে তা বোঝা পরিষ্কার করে কেন বাস্তব সিস্টেমগুলো শুধু একটির উপর নির্ভর না করে বেশ কয়েকটি একসাথে ব্যবহার করে:

```
1. PRETRAINING DATA FILTERING   -- pretraining শুরু হওয়ার আগেই কর্পাস থেকে toxic/নিম্নমানের document
                                    সরিয়ে ফেলা (Phase 04 Lesson 1-এর data pipeline);
                                    এখানে একটি toxicity classifier (section 3) একটি মানক হাতিয়ার।
2. SFT                          -- curated demonstration দিয়ে লক্ষ্য আচরণ শেখানো
                                    (Phase 05 Lesson 4); স্পষ্ট refusal উদাহরণ অন্তর্ভুক্ত হতে পারে।
3. RLHF / DPO STEERING          -- preference data (মানুষের, Lesson 2, অথবা AI-র, Lesson 5) ব্যবহার করে
                                    POLICY-টিকেই নিরাপদ, কম biased আচরণের দিকে ঠেলা;
                                    এটি কেবল পরে যা পিছলে যায় তা নয়, মডেল যা তৈরি করতে প্রবণ তা পরিবর্তন করে।
4. INFERENCE-TIME GUARDRAILS    -- একটি পৃথক, দ্রুত classifier বা rule সিস্টেম (section 3) প্রতিটি
                                    generation-কে user দেখার ঠিক আগে ছাঁকে, আগের stage-গুলো যে ব্যর্থতা
                                    মিস করেছে তা ধরে, LLM-কে retrain করার প্রয়োজন ছাড়াই।
5. RED-TEAMING (section 1)      -- উপরের প্রতিটি stage জুড়ে অবিচ্ছিন্নভাবে চালানো, সেই ব্যর্থতাগুলো খোঁজার
                                    জন্য যেগুলো data filtering, preference data বা guardrail rule-এর
                                    পরবর্তী রাউন্ডকে প্রেরণা দেয়।
```

কোনো একক stage একা যথেষ্ট নয়: pretraining filter প্রতিটি ক্ষতিকর প্যাটার্ন ধরতে পারে না, RLHF/DPO steering-কে যথেষ্ট adversarial একটি prompt ঘুরিয়ে দিতে পারে (ঠিক সেই ব্যর্থতা যা red-teaming খোঁজে), এবং একটি inference-time guardrail উপসর্গের চিকিৎসা করে, policy-র অন্তর্নিহিত প্রবণতার নয় — অবিকল এই কারণেই production সিস্টেমগুলো এগুলোর কয়েকটিকে স্তরে স্তরে রাখে, এবং কেনও একটি সৎ safety গল্প রিপোর্ট করে প্রতিটি পৃথক স্তর কী ধরে এবং কী ধরে না।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — এ পর্যন্ত নির্মিত alignment যন্ত্রপাতি আচরণ চালায়; এই লেসন ব্যর্থতা খোঁজার এবং সঠিক stage-এ সেগুলো ঠিক করার
2. Red-teaming: deployment-এর আগে failure mode বের করে আনার জন্য adversarial probing (section 1)
3. Bias measurement: template-ভিত্তিক probe পদ্ধতি, জনতাত্ত্বিক পদ ছাড়া সব বৈধ বিষয় স্থির রেখে (section 2)
4. Toxicity classifier: একটি নিবেদিত, সংকীর্ণ-সীমাবদ্ধ মডেল, data filtering-এর জন্য বা inference-time guardrail হিসেবে ব্যবহৃত (section 3)
5. Dixon et al. caveat: classifier-গুলো নিজেদেরই biased হতে পারে, bias measurement ও toxicity classification-কে একসাথে বাঁধে
6. পাঁচ-পর্যায়ের প্রশমন পাইপলাইন এবং কেন কোনো একক stage একা যথেষ্ট নয় (section 4)
7. `example.py` Part 1-এর walkthrough -- একটি খেলনা scorer-এ একটি পরিচিত bias inject করা এবং probe সেটি পুনরুদ্ধার করে তা নিশ্চিত করা
8. `example.py` Part 2-3-এর walkthrough -- scratch থেকে একটি বাস্তব toxicity classifier প্রশিক্ষণ, held-out data-তে precision/recall মাপা, এবং এটিকে deployment-time guardrail হিসেবে পুনরায় ব্যবহার

## আরও পড়ুন

- Ganguli et al. (2022), *Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned*
- Dixon et al. (2018), *Measuring and Mitigating Unintended Bias in Text Classification*
- Bertrand, Mullainathan (2004), *Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination*
- Rudinger et al. (2018), *Gender Bias in Coreference Resolution* (Winogender schema)
- Caliskan, Bryson, Narayanan (2017), *Semantics Derived Automatically from Language Corpora Contain Human-like Biases* (WEAT পদ্ধতি)
- Welbl et al. (2021), *Challenges in Detoxifying Language Models*