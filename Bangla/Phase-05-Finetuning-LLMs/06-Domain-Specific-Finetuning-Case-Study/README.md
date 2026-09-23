# Domain-Specific Fine-tuning কেস স্টাডি

**Phase:** [LLM ফাইন-টিউনিং](../README.md) · **টপিক ফোল্ডার:** `06-Domain-Specific-Finetuning-Case-Study`

## কেন এটি গুরুত্বপূর্ণ

এই lesson-টি এই phase-এর সমাপনী (capstone): একটি একক, কাজ করা উদাহরণ, যা [Lesson 1-এর full-fine-tuning-vs-PEFT সিদ্ধান্ত](../01-Full-Finetuning-vs-PEFT/README.md), [Lesson 2-এর LoRA mechanism](../02-LoRA-and-QLoRA/README.md) আর [Lesson 5-এর বাস্তব-জগতের tooling](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md)-কে একসাথে টেনে নিয়ে আসে সেই প্রশ্নে, যা দিয়ে একটি প্রকৃত fine-tuning প্রজেক্ট সত্যিই শুরু হয়: *"আমি চাই এই model-টি আমার সংকীর্ণ domain-এ ভালো হোক — আমি এটা কীভাবে করব, আর কী ঝুঁকি নিচ্ছি?"* এই phase-এর বাকি প্রতিটি lesson একটি করে mechanism আলাদাভাবে পরীক্ষা করেছে; এই lesson-টি একটি প্রকৃত before/after তুলনা চালায় এবং কী ঘটল তা রিপোর্ট করে — সেই অংশগুলো-সহ, যা কোনো পদ্ধতিকেই তেমন শোভন দেখায় না।

## এই lesson-এ যা যা শেখানো হবে

- একটি সংকীর্ণ domain-এর জন্য data curation-বিষয়ক বিবেচনাগুলো
- একটি domain-adaptation প্রজেক্টের জন্য full fine-tuning বনাম PEFT বেছে নেওয়া — কংক্রিটভাবে পুনরালোচনা
- কেন domain adaptation-কে *দুটি* অক্ষে মূল্যায়ন করতে হয়, একটিতে নয়: in-domain gain ও general-capability regression
- একটি কাজ করা কেস স্টাডি: একটি ছোট pretrained model-কে LoRA দিয়ে এবং full fine-tuning দিয়ে একটি বিশেষ "pirate speak" স্টাইলে অভিযোজিত করা — পাশাপাশি মেপে

## ১. একটি সংকীর্ণ domain-এর জন্য data curation

একটি সংকীর্ণ domain-এর জন্য fine-tuning (একটি কোম্পানির support-ticket স্টাইল, একটি legal-document register, একটি নির্দিষ্ট লেখার ভয়েস) শুরু হয় সেই একই data-প্রশ্ন দিয়ে, যা [Lesson 4 §5](../04-Instruction-Tuning-SFT/README.md#5-data-quality-and-diversity-beat-raw-quantity) সাধারণভাবে instruction data-র জন্য তুলেছিল — তবে এবার domain-এর ক্ষেত্রে আরও ধারালো করে: "সংকীর্ণ"-কে কতটা সংকীর্ণভাবে curated করতে হবে? domain-এর ভেতরে খুব কম বৈচিত্র্য (প্রায় অভিন্ন কয়েকটি উদাহরণ) ঝুঁকি তৈরি করে যে model-টি অন্তর্নিহিত স্টাইল বা vocabulary-এর বদলে উপরিতলের প্যাটার্ন মুখস্থ করবে — [Lesson 4-এর নিজস্ব toy run](../04-Instruction-Tuning-SFT/README.md#6-what-examplepy-does) training-set ও held-out accuracy-এর মধ্যে ঠিক এই ফারাকটিই সরাসরি মেপেছিল। Training উদাহরণে টার্গেট domain থেকে খুব বেশি সরে গেলে (মিশ্র-মানের scrape-করা text, প্রসঙ্গ-বহির্ভূত ভরাট-সামগ্রী) সেই সংকেত পাতলা হয়ে যায়, যাকে ঘন করতে fine-tuning run-টি চায়। অনুশীলনে এর অর্থ: এমন উদাহরণ সংগ্রহ করো, যেগুলো স্পষ্টতই টার্গেট domain-এর vocabulary ও phrase-গঠনের *প্রতিনিধিত্বমূলক*; মূল্যায়নের জন্য সত্যিই আলাদা একটি অংশ হাতে রেখে দাও (কখনো সেই data নয়, যাতে model train হয়েছে — ঠিক যেমন `example.py` নিচে করে); আর একদম শুরুতেই একটি আলাদা general-purpose মূল্যায়ন-সেট হাতে রাখো — আগে থেকে "before" কেমন দেখাচ্ছে তা ঠিক না করলে, ঘটনার পরে forgetting মাপা সম্ভব নয়।

## ২. Domain adaptation-এর জন্য full fine-tuning বনাম PEFT

[Lesson 1](../01-Full-Finetuning-vs-PEFT/README.md) বিমূর্ত trade-off-টি সাজিয়ে দিয়েছে: full fine-tuning প্রতিটি parameter update করে (সবচেয়ে উঁচু সিলিং, সবচেয়ে বেশি memory খরচ, সবচেয়ে বেশি forgetting ঝুঁকি), অন্যদিকে [LoRA](../02-LoRA-and-QLoRA/README.md)-র মতো PEFT পদ্ধতিগুলো base-টি freeze করে ছোট একটি add-on train করে। Domain adaptation এমনই একটি প্রেক্ষাপট, যেখানে এই trade-off-টি সবচেয়ে কংক্রিট, কারণ — প্রায় সংজ্ঞা অনুযায়ীই — fine-tuning data pretraining corpus-এর চেয়ে সংকীর্ণ ও কম বৈচিত্র্যপূর্ণ; এই অবস্থাতেই full fine-tuning-এর প্রতিটি weight সরানোর সীমাহীন স্বাধীনতা base model-এর আগে-থেকেই থাকা সাধারণ দক্ষতাকে ওভাররাইট করার সম্ভাবনা সবচেয়ে বেশি — শুধুই কারণ training data বা loss-এর কোনো কিছুই এটিকে নিরুৎসাহিত করে না। LoRA-র frozen base এটির বিরুদ্ধে একটি কাঠামোগত বাধা: rank-`r`-এর একটি update দিয়ে কয়েকটি projection matrix-এর মাধ্যমে যেই আচরণে পৌঁছানো যায় না, তা কেবল বদলায়ই না — ভালো দিক থেকে (সাধারণ দক্ষতা সুরক্ষিত) আর খারাপ দিক থেকেও (অর্জনযোগ্য domain-নির্দিষ্ট উন্নতির একটি সিলিং আছে, যা rank নির্ধারণ করে)। `example.py` একই *pretrained model*-টিকে দুইভাবে — একই *domain data*-তে — fine-tune করে, আর মেপে দেখে এই run-এর প্রকৃত সংখ্যা অনুযায়ী trade-off-টি আসলে কোথায় গিয়ে দাঁড়াল।

## ৩. দুটি অক্ষে মূল্যায়ন: specialization এবং regression

Domain fine-tuning-এর লেখা-লেখিতে সবচেয়ে সাধারণ ভুলটি হলো শুধু সেই মেট্রিকটি রিপোর্ট করা, যা উন্নত হয়েছে। একটি domain fine-tune, যা টার্গেট domain-এ ব্যাপক সাহায্য করলেও নীরবে model-টিকে সাধারণ কাজে খারাপ করে দেয়, স্পষ্টতই ভালো trade নয় — এটি মূল্যবান কি না পুরোপুরি নির্ভর করে model-টি আসলে কীভাবে ব্যবহৃত হবে, আর এই সিদ্ধান্ত আপনি নিতে পারবেনই কেবল, যদি regression-টি একেবারেই মাপা হয়ে থাকে। এই lesson-এ ব্যবহৃত পদ্ধতি এতটাই সরল ও সাধারণ যে এটি যেকোনো প্রকৃত domain-adaptation প্রজেক্টে প্রয়োগযোগ্য:

1. Model-এ হাত দেওয়ার আগে, দুটি held-out মূল্যায়ন-সেট স্থির করো: একটি **in-domain** (যে নিখুঁত স্টাইলটিকে আপনি টার্গেট করছেন তার নমুনা, fine-tuning-এর সময় কখনো দেখা হয়নি) এবং একটি **general** (model-টি আগে কী করতে পারত তার সাধারণ, বিস্তৃত-কভারেজ উদাহরণ)।
2. দুটোই fine-tuning-এর **আগে** মাপো — এটি আপনার catastrophic forgetting-এর জন্য baseline, শুধু উন্নতির baseline নয়।
3. Fine-tune করো।
4. দুটো আবার মাপো। In-domain **improvement** আর general-capability **drift** দুটো একসাথে রিপোর্ট করো, আলাদাভাবে নয় — বড় regression-এর বিনিময়ে কেনা বড় improvement, সেই একই improvement-এর চেয়ে গুণগতভাবে ভিন্ন ফলাফল, যা কোনো খরচই নেয়নি।

`example.py` দুই পাশেই মেট্রিক হিসেবে held-out **next-token-prediction loss** ব্যবহার করে (কম = model-এর শেখা distribution সেই text-এর সাথে ভালো মানানসই), কারণ এটিই একমাত্র মেট্রিক, যা task-নির্দিষ্ট scorer-এর প্রয়োজন ছাড়াই যেকোনো text-এ — in-domain বা general — অভিন্নভাবে প্রযোজ্য।

## ৪. `example.py` কী করে

1. একটি ছোট `MiniGPT` ([Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) ও [Lesson 2](../02-LoRA-and-QLoRA/README.md)-এর সেই একই block) শুধু সাধারণ, দৈনন্দিন-ইংরেজি বাক্যে pretrain করে।
2. সাধারণ text-এ এবং একটি স্বতন্ত্রভাবে ভিন্ন "pirate speak" domain-এ (`"arr the ship sails..."`, `"the cap'n found a chest of treasure"`) তার held-out loss মাপে — যা সে কখনো দেখেনি — ফলে fine-tuning-আগের baseline দুটি অক্ষেই প্রতিষ্ঠিত হয়।
3. এই নির্দিষ্ট model-টির জন্য full fine-tuning বনাম LoRA-র trainable-parameter খরচ প্রিন্ট করে, [Lesson 2-এর `LoRALinear`](../02-LoRA-and-QLoRA/README.md#5-what-this-lessons-code-does-and-what-a-real-workflow-uses-instead) ব্যবহার করে — এখানে একটি `from_pretrained_linear` constructor যোগ করে, যেন এটি *ইতিমধ্যে-pretrained* একটি layer-এর weights মোড়ে (বাস্তবসম্মত ক্ষেত্র; Lesson 2-এর নিজস্ব demo একটি সদ্য-initialize করা layer মোড়ত)।
4. Pretrained model-টির দুটি স্বাধীন কপি একই সংকীর্ণ pirate-speak training data-তে fine-tune করে: একটি যেখানে শুধু attention-এর `W_q`/`W_v` projection-গুলোতে LoRA প্রয়োগ (বাকি সবকিছু frozen), আরেকটি যেখানে প্রতিটি parameter-ই trainable।
5. দুটো fine-tuned model-এর জন্য domain ও general — দুটি মূল্যায়ন-সেটেই — held-out loss আবার মাপে, আর domain improvement ও general-capability drift পাশাপাশি রিপোর্ট করে — সাথে একটি domain-নিরপেক্ষ prompt থেকে qualitative generation, যাতে দেখা যায় pirate vocabulary এমন text-এ ফুটে ওঠে কি না, যেখানে কোনো domain-ইঙ্গিতই ছিল না।

শেষে প্রিন্ট হওয়া সিদ্ধান্তটি সরাসরি সেই run-এর প্রকৃত সংখ্যা থেকে নেওয়া — যেদিকে-ই যাক না কেন — আগে-থেকে-স্থির কোনো গল্প থেকে নয়, কারণ এই lesson-এর পুরো উদ্দেশ্যই হলো, এই trade-off-টি আগে থেকে জানা যায় না; মেপেই দেখতে হয়।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "Lessons 1, 2 আর 5-কে একসাথে একটি বাস্তব প্রশ্নে: MY domain-এর জন্য fine-tune, কত খরচে?"
2. একটি সংকীর্ণ domain-এর জন্য data curation: প্রতিনিধিত্বমূলক, যথেষ্ট-বৈচিত্র্যপূর্ণ, সত্যিই-held-out
3. Full fine-tuning বনাম LoRA — বিশেষভাবে domain-adaptation প্রেক্ষাপটের জন্য পুনরালোচনা
4. কেন আপনাকে দুটি অক্ষে মূল্যায়ন করতেই হবে — in-domain gain ও general-capability drift — একটি নয়
5. `example.py`-এর সেটআপের ওয়াকথ্রু: pretraining, দুটি held-out সেট, `LoRALinear.from_pretrained_linear`
6. LoRA বনাম full-fine-tuning ফলাফলের ওয়াকথ্রু: parameter খরচ, domain improvement, general-loss drift, আর qualitative generic-prompt পরীক্ষা
7. পুরো phase-এর recap: PEFT/LoRA mechanism -> instruction tuning -> বাস্তব tooling -> এই case study, যা সবকিছুকে একসাথে বাঁধে
8. Preview: [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md) একটি fine-tuned, instruction-following model থেকে শুরু করে প্রশ্ন করে — কীভাবে এর আচরণকে মানুষের পছন্দের সাথে align করা যায়

## আরও পড়ার জন্য

- Kirkpatrick et al. (2017), *Overcoming Catastrophic Forgetting in Neural Networks* (এখানে সরাসরি মাপা সাধারণ ঘটনাটি — প্রথমে উত্থাপিত [Lesson 1 §3](../01-Full-Finetuning-vs-PEFT/README.md#3-catastrophic-forgetting)-এ)
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (এই case study-তে full fine-tuning-এর সাথে তুলনা করা পদ্ধতিটি — পূর্ণ ব্যুৎপত্তি [Lesson 2](../02-LoRA-and-QLoRA/README.md)-তে)
- Gururangan et al. (2020), *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (একই অন্তর্নিহিত সমস্যার জন্য একটি সম্পর্কিত, বৃহত্তর-স্কেল কৌশল হিসেবে domain-adaptive pretraining)
- Zhou et al. (2023), *LIMA: Less Is More for Alignment* (§1-এ domain data-তে প্রয়োগ করা data-curation কোণ — প্রথমে আলোচিত [Lesson 4](../04-Instruction-Tuning-SFT/README.md#5-data-quality-and-diversity-beat-raw-quantity)-এ)