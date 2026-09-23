# Full Fine-tuning বনাম Parameter-Efficient Fine-tuning

**Phase:** [LLM ফাইন-টিউনিং](../README.md) · **টপিক ফোল্ডার:** `01-Full-Finetuning-vs-PEFT`

## কেন এটি গুরুত্বপূর্ণ

[Phase 04](../../Phase-04-Pretraining-LLMs/README.md)-তে দেখেছি কীভাবে একটি base model কাঁচা text থেকে সাধারণ ভাষাগত দক্ষতা শেখে, আর [Phase 03 Lesson 1 §1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#1-gpt-1-pretrain-then-fine-tune)-এ GPT-1-এর মূল paper থেকে পরের ধাপটির ইঙ্গিত ইতিমধ্যেই পেয়েছি: একবার pretrain করো, তারপর নির্দিষ্ট একটি আচরণের জন্য fine-tune করো। এই lesson-টি সেই ব্যবহারিক প্রশ্নটি নিয়ে, যা বাস্তব জগতের fine-tuning সিদ্ধান্তে সবচেয়ে বেশি প্রাধান্য পায়: fine-tune করব *কীভাবে*? প্রতিটি parameter, নাকি ছোট একটি সস্তা add-on? এই উত্তরই ঠিক করে দেয় — বড় একটি model-কে fine-tune করতে GPU-তে ভরা একটি ঘর দরকার, নাকি একটি সাধারণ consumer card-ই যথেষ্ট। আর এই lesson-টি এই phase-এর বাকি সব lesson-এরও ভিত্তি তৈরি করে, কারণ [LoRA](../02-LoRA-and-QLoRA/README.md), [prompt/prefix tuning এবং adapters](../03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md), আর [Hugging Face PEFT/TRL workflow](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) — সবই হলো "আমরা কীভাবে full fine-tuning-এর খরচ এড়াবো?" এই প্রশ্নের নির্দিষ্ট উত্তর।

## এই lesson-এ যা যা শেখানো হবে

- Full fine-tuning আসলে কী update করে, এবং কেন এটি train করা এত ব্যয়বহুল (শুধু সংরক্ষণ করা নয়)
- AdamW optimizer-state-এর memory-র ব্যয়, এবং "প্রতি parameter 16 bytes" এই থাম্ব-নিয়মটি কোথা থেকে আসে
- Catastrophic forgetting: প্রতিটি weight update করলে কেন সাধারণ (general) দক্ষতা মুছে যাওয়ার ঝুঁকি থাকে
- সাধারণ Parameter-Efficient Fine-Tuning (PEFT) ধারণা: base-টি freeze করো, ছোট একটি add-on train করো
- Multi-task deployment-এর খরচ: প্রতিটি task-এর জন্য একটি করে সম্পূর্ণ model copy বনাম একটি shared base + অসংখ্য ছোট adapter

## ১. Full fine-tuning: সবকিছু update করা

Full fine-tuning-এ একটি pretrained model-এর weight-গুলোকে *initialization* হিসেবে নিয়ে সাধারণ gradient descent চালিয়ে যাওয়া হয় — [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction)-এর ঠিক সেই একই forward pass, cross-entropy loss আর backward pass — শুধু পার্থক্য এই যে, এবার model-টির প্রতিটি বিদ্যমান parameter-ই একটি trainable leaf, যা নতুন, task- বা domain-নির্দিষ্ট data-র উপর gradient update পায়। ধারণাগতভাবে এটিই সবচেয়ে সহজ বিকল্প, এবং এর quality-র সম্ভাব্য সিলিংও সবচেয়ে উঁচু: model-টির কোনো কিছুই সীমাবদ্ধ থাকে না, তাই নীতিগতভাবে এটি weight-space-এর সেই সব বিন্দুতে পৌঁছাতে পারে, যেখানে সাধারণ pretraining-ও পৌঁছাতে পারত। কিন্তু এই খরচই একটি নির্দিষ্ট আকারের পরে এটিকে অবাস্তব (impractical) করে তোলে — এবং এই খরচ আসলে weight সংরক্ষণের নয়, বরং সেগুলোকে *train* করার।

## ২. আসল memory-র খরচ: optimizer, weight নয়

শুধু model-এর weight সংরক্ষণ করলে খরচ train করার তুলনায় কম। AdamW (Transformers-এর জন্য de facto মানক optimizer) দিয়ে train করতে হলে **প্রতি trainable parameter-এর জন্য** প্রয়োজন হয়:

```
weights (fp16)                     2 bytes
gradients (fp16)                   2 bytes
fp32 master copy of the weights    4 bytes   (kept for numerically stable updates)
Adam first moment  (m, fp32)       4 bytes
Adam second moment (v, fp32)       4 bytes
---------------------------------------------
total                             16 bytes / trainable parameter
```

এটি মানক mixed-precision-training memory breakdown (একই হিসাবের জন্য দেখো Rajbhandari et al., 2020, *ZeRO*)। সেখান থেকে মূল সিদ্ধান্ত: **একটি 7-বিলিয়ন-parameter model-কে full fine-tune করতে হলে শুধু weights + gradients + optimizer state-এর জন্যই আনুমানিক 16 x 7B ≈ 112 GB** দরকার — backward pass-এর জন্য একটি activation-ও সংরক্ষণ করার আগেই, এবং training batch-এর নিজস্ব memory-র খরচ তো ধরাই হয়নি। তুলনায়, একই model-কে inference-এর জন্য শুধু *লোড* করতে হলে লাগে তার 2-byte-র weight-গুলো: প্রায় 14 GB। Full fine-tuning-এর memory-চাহিদার একেবারে বড় অংশজুড়ে থাকে optimizer-এর bookkeeping — যে parameter-গুলো update হয় তাদের পেছনে — আর এটি ঠিক সেই পরিমাণ, যেটিকে PEFT পদ্ধতিগুলো সরাসরি আক্রমণ করে।

## ৩. Catastrophic forgetting

খরচ ছাড়াও, full fine-tuning-এ একটি বাস্তব quality-ঝুঁকিও আছে: যেহেতু প্রতিটি weight-ই যেকোনো দিকে যেতে স্বাধীন, তাই নতুন, সংকীর্ণ dataset-এ চলা gradient descent base model-এর সেই সাধারণ জ্ঞান ও দক্ষতাগুলোকে ওভাররাইট করতে পারে, যা অর্জনে model-টি বিপুল pretraining compute খরচ করেছিল। এটিই **catastrophic forgetting** — model-টি fine-tuning task-টিতে আগের চেয়ে ভালো হয়ে ওঠে, কিন্তু নীরবে খারাপ হয়ে যায় সেই সব কাজে, যা আগে সে করতে পারত। [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md) এই trade-off-টি সরাসরি এবং পরিমাণগতভাবে মাপে। এটি model-এর *কতটুকু* অংশ বদলাতে দেওয়া হবে তা সীমাবদ্ধ রাখাকে পছন্দ করার দ্বিতীয়, স্বাধীন কারণ — শুধু কাঁচা memory-খরচের বাইরে।

## ৪. PEFT ধারণা: প্রায় সবকিছু freeze করা

Parameter-Efficient Fine-Tuning (PEFT) হলো সেই সব কৌশলের পরিবার, যাদের ভিত্তি একটি অভিন্ন কৌশল: **পুরো pretrained base model-টি freeze করো** (প্রতিটি মূল parameter-এ `requires_grad=False` — এটি forward pass-এ অংশ নেয়, কিন্তু কখনো gradient update পায় না) **এবং অল্প কিছু নতুন, trainable parameter যোগ করো**, যাদের কাজ হলো frozen model-টির আচরণকে পথ দেখানো বা সামঞ্জস্য করা। যেহেতু শুধু এই ছোট নতুন অংশটুকুরই gradient, fp32 master copy আর Adam moment দরকার, তাই 16-bytes-per-parameter ব্যয়টি প্রযোজ্য হয় model-টির সামান্য একটি অংশে মাত্র:

```
Full fine-tuning memory  =  N_total x 16 bytes
PEFT memory              =  N_total x 2 bytes (frozen, inference-only)  +  N_trainable x 16 bytes
```

যেহেতু `N_trainable` সাধারণত `N_total`-এর 0.01%-1% হয় ([Lesson 2](../02-LoRA-and-QLoRA/README.md#3-counting-parameters)-এ বিশেষভাবে LoRA-র জন্য কংক্রিট সংখ্যা হিসাব করা আছে), তাই দ্বিতীয় পদটি প্রায় শূন্যের কাছাকাছি চলে যায়, আর PEFT-এর মোট memory-র বড় অংশজুড়ে থাকে base model-এর সস্তা frozen-inference খরচ — base model যত বড়ই হোক না কেন। `example.py` বাস্তবসম্মত বিভিন্ন model size-এর ওপর এই তুলনাটি সরাসরি হিসাব করে।

## ৫. Multi-task deployment-এর কোণ

Memory-র যুক্তিটি *training*-এর সময় সবচেয়ে গুরুত্বপূর্ণ, কিন্তু PEFT-এর *deployment*-এর সময় একটি দ্বিতীয়, স্বাধীন লাভও আছে। পাঁচটি ভিন্ন task-এর জন্য fine-tuned model দরকার হলে, full fine-tuning তৈরি করবে **model-এর weight-এর পাঁচটি সম্পূর্ণ কপি** (প্রতিটিই একটি সম্পূর্ণ checkpoint, কারণ প্রতিটি weight-ই সম্ভবত বদলেছে)। PEFT তৈরি করে **একটি shared frozen base আর পাঁচটি ছোট adapter ফাইল** — প্রতি adapter প্রায়ই এক ডিজিটের কয়েক MB, যেখানে প্রতিটি সম্পূর্ণ checkpoint-এর জন্য লাগে gigabytes — কারণ task-গুলোর মধ্যে পার্থক্য কেবল সেই ছোট যোগ করা parameter-গুলোতে। এই কারণেই একই base model-এর অনেক fine-tuned ভ্যারিয়েন্ট সার্ভ করে এমন production system (যেমন প্রতিটি customer বা task-এর জন্য একটি করে LoRA adapter) প্রায় সবসময়ই PEFT ব্যবহার করে: memory-তে adapter বদলানো সম্পূর্ণ model বদলানোর চেয়ে অনেক সস্তা।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "একটি 7B model fine-tune করতে 100+ GB লাগে কেন, যখন model-টি নিজেই মাত্র 14 GB?"
2. Full fine-tuning-এর সংক্ষিপ্ত পুনরালোচনা: pretraining-এর মতোই training loop, শুধু একটি checkpoint থেকে আরও চালিয়ে যাওয়া
3. Memory আসলে কোথায় খরচ হয়: weights, gradients, আর AdamW optimizer-state-এর ব্যয় — বিস্তারিত
4. সবকিছু update করার দ্বিতীয়, স্বাধীন খরচ হিসেবে catastrophic forgetting
5. PEFT ধারণা: base-টি freeze করো, ছোট্ট একটি add-on train করো, প্রায় কিছুইতে 16-byte-র ব্যয় দাও
6. `example.py`-এর ওয়াকথ্রু — বিভিন্ন model size-জুড়ে memory-footprint ক্যালকুলেটর, আর multi-task storage তুলনা
7. Recap + preview: Lesson 2 "ছোট add-on"-টিকে LoRA-র মাধ্যমে কংক্রিট করে তুলবে

## আরও পড়ার জন্য

- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP* (যে paper-টি adapter-এর মাধ্যমে PEFT ধারণাটির নামকরণ ও জনপ্রিয়করণ করেছে; [Lesson 3](../03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md)-এ সম্পূর্ণভাবে আলোচিত)
- Rajbhandari, Rasley, Ruwase, He (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (উপরে ব্যবহৃত mixed-precision Adam memory হিসাব)
- Kirkpatrick et al. (2017), *Overcoming Catastrophic Forgetting in Neural Networks*