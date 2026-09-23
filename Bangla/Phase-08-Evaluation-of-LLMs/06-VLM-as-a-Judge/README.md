# VLM-as-a-Judge

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `06-VLM-as-a-Judge`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 3](../03-LLM-as-a-Judge/README.md)-এ খাঁটি টেক্সটের জন্য LLM-as-a-Judge প্রতিষ্ঠিত হয়েছিল: একটি judge একটি prompt এবং একটি candidate response পড়ে গ্রেড করে, এবং পুরো lesson-টি সেই গ্রেডিং প্রক্রিয়ার bias-গুলো নিয়ে — position, verbosity, self-preference — যেগুলো judge-এর রায়কে বিকৃত করে *এমনকি যখন তার কাছে সঠিকভাবে গ্রেড করার জন্য যা যা দরকার সবই থাকে*। যে মুহূর্তে গ্রেড করা জিনিসটির সাথে একটি ছবি জড়িত — একটি caption, একটি visual question উত্তর, একটি বর্ণিত চার্ট বা screenshot — তখন একটি টেক্সট-কেবল judge সেই "সম্পূর্ণ অ্যাক্সেস" ধারণাটি সম্পূর্ণ হারিয়ে ফেলে: এটি আপনাকে বলতে পারে caption-টি সাবলীলভাবে পড়া যায় কি না, কিন্তু বলতে পারে না caption-টি আসলে ছবির ভেতরে যা আছে তা বর্ণনা করছে কি না, কারণ এটি ছবিই কখনো দেখেনি। একটি **VLM-as-a-judge** — একটি vision-language model (একই architecture পরিবার যা [Phase 10 Lesson 1](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#3-from-an-aligned-space-to-a-multimodal-llm-llava)-এ তৈরি হয়েছিল) যা টেক্সট ও rubric-এর পাশাপাশি ছবিটিও গ্রহণ করে — সেই ফাঁকটি বন্ধ করে, কিন্তু শুধুমাত্র যদি এটি claim যাচাই করার জন্য ছবিটি সত্যিই ব্যবহার করে, কেবল আরেকটি সাবলীল-শোনানো অনুমান তৈরি করার বদলে। এই lesson-টি সেই নতুন failure mode নিয়ে — একটি judge-এর *grounding* — যা Lesson 3-এ ইতিমধ্যে আচ্ছাদিত সবকিছুর উপরে বসে।

## এই পাঠে যা যা আছে

- কেন একটি টেক্সট-কেবল judge কাঠামোগতভাবে একটি ছবি সম্পর্কে claim যাচাই করতে পারে না
- নতুন, vision-নির্দিষ্ট judge failure mode: object hallucination, counting error, spatial/relational error
- এগুলো Lesson 3-এর bias-গুলো থেকে কীভাবে আলাদা: সেগুলো judge-এর *আচরণ* বিকৃত করে, এগুলো বোঝায় এটি claim-টি আদৌ *যাচাই* করতে পারে কি না
- vision কাজের জন্য rubric ডিজাইন: কেবল groundable, যাচাইযোগ্য claim-গুলো গ্রেড করা
- একটি বাস্তব, শূন্য থেকে তৈরি simulation: একটি judge যে পরিচিত ground-truth scene-এর বিরুদ্ধে caption যাচাই করে, বনাম একটি যে পারে না
- কেন একটি ungrounded "judge" অবনমিত হয়ে হুবহু Lesson 3-এর length/fluency-চালিত bias-এ ফিরে যায়

## 1. কেন একটি টেক্সট-কেবল judge multimodal আউটপুট গ্রেড করতে পারে না

Lesson 3-এর দুটি protocol — pairwise comparison এবং absolute rubric scoring — দুটোই নীরবে ধরে নেয় যে judge স্বাধীনভাবে মূল্যায়ন করতে পারে একটি response-টি বাস্তবতার সাথে কতটা ভালোভাবে মেলে যেটিকে গ্রেড করার জন্য যথেষ্ট। কোনো reference উত্তর ছাড়া open-ended টেক্সটের জন্য, সেটি ইতিমধ্যেই কঠিন, এবং Lesson 3-এর পুরো বিষয়বস্তুই হলো এমনকি তখনও judge-এর রায় কীভাবে বিকৃত হয়। captioning বা visual-QA কাজের জন্য, ধারণাটি কেবল অসম্পূর্ণভাবে নয় বরং সম্পূর্ণভাবে ভেঙে যায়: একটি judge যার ছবিতে কোনো অ্যাক্সেস নেই, সে আক্ষরিক অর্থেই "আকাশ নীল এবং তিনটি পাখি আছে" থেকে "আকাশ সবুজ এবং পাঁচটি পাখি আছে" আলাদা করতে পারে না। এটি কেবল fluency, plausibility এবং surface শৈলীতে ফিরে যেতে পারে — এগুলোর কোনোটিই claim-টি সত্যিই সত্য কি না তার সাথে সম্পর্কিত নয়।

## 2. vision grounding-এর জন্য নির্দিষ্ট নতুন failure mode

- **Object hallucination** — caption এমন একটি object বা attribute-এর কথা বলে যা ছবিতে আসলে নেই; [Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md#1-defining-hallucination)-এ আচ্ছাদিত টেক্সট hallucination-এর vision-language অ্যানালগ; বাস্তব VLM captioning গবেষণায় এটি এত ভালোভাবে নথিভুক্ত যে এর জন্য নিবেদিত নিজস্ব benchmark-ও আছে (CHAIR, POPE — দেখুন Further Reading)।
- **Counting errors** — সঠিকভাবে চিহ্নিত করা যে একটি object বিদ্যমান, কিন্তু কতগুলো আছে তা ভুল পাওয়া।
- **Spatial/relational errors** — বাম/ডান, উপরে/নিচে, সামনে/পেছনে ভুলভাবে বলা।

Lesson 3-এর তিনটি bias *নির্বিশেষে* টিকে থাকে যে judge নীতিগতভাবে response-গুলো সঠিকভাবে আলাদা করতে পারত কি না — সেগুলো বোঝায় judge একটি তুলনাকে কীভাবে বিকৃত করে যেটি অন্যথায় করতে সক্ষম। এই failure mode-গুলো আরও এক স্তর মৌলিক: সেগুলো বোঝায় judge একটি বৈধ তুলনা **আদৌ** করতে পারে কি না।

## 3. Protocol এবং rubric ডিজাইন

[Lesson 3 §1](../03-LLM-as-a-Judge/README.md#1-two-judge-protocols)-এর একই pairwise/absolute পার্থক্য এখনও প্রযোজ্য — একটি VLM-judge-এর context-এ টেক্সটের পাশাপাশি কেবল ছবিটিও থাকে এবং (absolute scoring-এর জন্য) একটি rubric। একমাত্র যোগ যা সত্যিই গুরুত্বপূর্ণ: একটি ব্যবহারযোগ্য vision rubric-কে পৃথকভাবে **groundable** claim-এ বিভক্ত হতে হয় — caption কি আসলে বিদ্যমান প্রতিটি প্রধান salient object-এর নাম বলে? count-টি কি সঠিক পায়? spatial relation-টি কি সঠিক পায়? — একটি অস্পষ্ট সামগ্রিক মানদণ্ড যেমন "এটি কি একটি ভালো caption" বলা ছাড়া। ছবির বিরুদ্ধে কেউ যাচাই করতে পারে না এমন একটি মানদণ্ড এখানে ঠিক ততটাই অকেজো যতটা Lesson 3-এ একটি অনিয়ন্ত্রিত confound ছিল — এটিকে একটি বাস্তব, পরিমাপযোগ্য সংকেতে রূপান্তর করা যায় না।

## 4. `example.py` — একটি বাস্তব, grounded VLM-judge, একটি symbolic scene থেকে তৈরি

এই repository-তে কাজ করার মতো কোনো বাস্তব vision model বা প্রকৃত pixel নেই, তাই `example.py` হাত-নেড়া বর্ণনার বদলে ছোট কিন্তু সত্যিই বাস্তব ও যাচাইযোগ্য কিছু তৈরি করে: একটি **scene** হলো একটি স্পষ্ট, কাঠামোবদ্ধ ground-truth data structure — (color, shape, count) object-এর একটি ছোট list যার সাথে দুটির মধ্যে একটি spatial relation, একটি ছোট CLEVR-শৈলীর scene graph যা একটি ছবির প্রতিনিধিত্ব করে। প্রতিটি scene থেকে, একটি নির্দিষ্ট caption template একটি সম্পূর্ণ সঠিক caption এবং চারটি লেবেলযুক্ত error variant তৈরি করে: একটি object-hallucination বাক্য যোগ করা হয় যার color/shape সংমিশ্রণটি scene-এ কখনোই দেখা যায় না, একটি count ভুল সংখ্যায় বদলানো, একটি attribute (color) এমন একটিতে বদলানো যা আসলে বিদ্যমান কোনো object-এর সাথে মেলে না, এবং একটি spatial relation তার বিপরীতে উল্টে দেওয়া।

দুটি judge তখন এই caption-গুলো গ্রেড করে:

- **The grounded judge** প্রতিটি caption-এর count ও relation claim-কে একটি ছোট, সৎ regex-ভিত্তিক extractor দিয়ে আবার বের করে আনে (এটির আসল NLP-এর দরকার নেই — caption শব্দভাণ্ডার নির্মাণের মাধ্যমেই নির্দিষ্ট, ঠিক এই repository-র অন্যান্য সীমাবদ্ধ-ভাষা কাজের জন্য ব্যবহৃত একই "ছোট কিন্তু বাস্তব প্রক্রিয়া") এবং scene-এর প্রকৃত ground truth-এর বিরুদ্ধে প্রতিটি claim যাচাই করে।
- **The ungrounded judge** শুধু caption টেক্সটটিই দেখে — কোনো scene নেই — এবং তার কাছে আসলে থাকা একমাত্র সংকেতে ফিরে যায়: response length, অর্থাৎ হুবহু [Lesson 3 §3](../03-LLM-as-a-Judge/README.md#3-verbosity-bias)-এর verbosity bias, এখানে যোগ করা ত্রুটি হিসেবে নয় বরং claim-এর সারবস্তু যাচাই করার কোনো উপায় নেই এমন একটি judge-এর জন্য যা *সম্পূর্ণ* রয়ে যায় সেটি হিসেবে।

`example.py` উভয় judge-কে হাজার হাজার বার, random scene/caption pair-এ চালায় এবং প্রতিটি error type-এর জন্য বাস্তব পরিমাপকৃত pairwise win rate রিপোর্ট করে (judge কি প্রতিটি error variant-এর তুলনায় সম্পূর্ণ সঠিক caption-টিকে সঠিকভাবে পছন্দ করে?), সাথে একটি কার্যকরী single-scene walkthrough যা দেখায় grounded judge কোন claim-গুলোকে সত্য বা মিথ্যা চিহ্নিত করে এবং কেন।

## 5. Recap: Lesson 3-এর পরিপূরক, প্রতিস্থাপন নয়

এমনকি একটি সঠিকভাবে grounded VLM-judge — যেটি ছবির বিরুদ্ধে প্রতিটি claim সঠিকভাবে যাচাই করে — এখনও position bias, verbosity bias এবং self-preference bias প্রদর্শন করতে পারে দুটি ইতিমধ্যে-যাচাইকৃত caption-কে সে *ওজন* করার পদ্ধতিতে। Grounding (এই lesson) এবং bias mitigations ([Lesson 3 §5](../03-LLM-as-a-Judge/README.md#5-mitigations): swap-and-average ordering, length control, বিভিন্ন judge panel) ভিন্ন ভিন্ন সমস্যা সমাধান করে, এবং একটি প্রোডাকশন multimodal eval pipeline-এ দুটোই দরকার — একটি ছাড়া অন্যটি এখনও একটি বাস্তব, পরিমাপযোগ্য failure mode খোলা রেখে দেয়।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — Lesson 3-এর judge গ্রেড করার জন্য যা যা দরকার সব দেখতে পেত; ছবিতে কোনো অ্যাক্সেস ছাড়া একটি multimodal judge পারে না
2. নতুন failure mode: object hallucination, counting error, spatial/relational error — এবং কীভাবে এগুলো Lesson 3-এর bias-গুলো থেকে ধরণে আলাদা
3. Rubric ডিজাইন: কেবল groundable, যাচাইযোগ্য claim-গুলোই আদৌ গ্রেডযোগ্য
4. Scene graph: একটি বাস্তব ছবির প্রতিনিধিত্বকারী একটি ছোট, কাঠামোবদ্ধ, CLEVR-শৈলীর ground truth
5. চারটি লেবেলযুক্ত error injection, এবং grounded judge-এর scene-এর বিরুদ্ধে বাস্তব claim-by-claim যাচাই
6. Ungrounded judge: ছবিতে কোনো অ্যাক্সেস নেই, তার একমাত্র অবশিষ্ট সংকেত হিসেবে Lesson 3-এর verbosity bias-এ ফিরে যায়
7. `example.py`-এর aggregate win-rate টেবিলের ভেতরে-বাইরে — grounded প্রায়-নিখুঁত, ungrounded chance-এ বা সক্রিয়ভাবে উল্টো
8. Recap — grounding এবং Lesson 3-এর bias mitigations ভিন্ন সমস্যা সমাধান করে; একটি বাস্তব pipeline-এ দুটোই দরকার

## আরও পড়ুন

- Rohrbach, Hendricks, Burns, Darrell, Saenko (2018), *Object Hallucination in Image Captioning* (CHAIR metric-টির পরিচয় — image captioning-এর জন্য মূল object-hallucination পরিমাপ)
- Li et al. (2023), *Evaluating Object Hallucination in Large Vision-Language Models* (POPE benchmark-টির পরিচয়)
- Liu, Li, Wu, Lee (2023), *Visual Instruction Tuning* — LLaVA paper, [Phase 10 Lesson 1](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#3-from-an-aligned-space-to-a-multimodal-llm-llava) থেকে পুনরালোচনা; একটি VLM-judge সাধারণত যে architecture হয় সেটি
- [Phase 11 Lesson 7 — VLM Hallucination and Alignment](../../Phase-11-Vision-Language-Models/07-VLM-Hallucination-and-Alignment/README.md) — এই lesson-এ গ্রেড করা grounding ব্যর্থতাগুলো আসলে কোথা থেকে আসে; এবং [Phase 11 Lesson 9 — Evaluating VLMs](../../Phase-11-Vision-Language-Models/09-Evaluating-VLMs/README.md) — একই সমস্যার benchmark-স্তরের সংস্করণের জন্য
- Zheng, Chiang, Sheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — [Lesson 3](../03-LLM-as-a-Judge/README.md) থেকে পুনরালোচনা; এই lesson-টি যে টেক্সট-কেবল judge bias-গুলোর উপর নির্মিত