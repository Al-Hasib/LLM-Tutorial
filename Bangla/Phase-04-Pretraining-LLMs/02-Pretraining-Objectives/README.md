# Pretraining Objectives

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `02-Pretraining-Objectives`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 1](../01-Pretraining-Data-Pipeline/README.md) আপনাকে প্রচুর পরিমাণ পরিষ্কার text দিয়েছে। এই পাঠ পরের প্রশ্নটির উত্তর দেয়: *এ থেকে আপনি আসলে কী training signal বের করবেন?* এই কোর্সে ইতিমধ্যেই Phase 02-03 জুড়ে তিনটি ভিন্ন উত্তর তৈরি করা হয়েছে — causal language modeling ([Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction)), masked language modeling ([Phase 03 Lesson 2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#2-masked-language-modeling-mlm)) এবং span corruption ([Phase 03 Lesson 3](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#2-t5s-pretraining-objective-span-corruption)) — তবে সেগুলো আলাদা আলাদা architecture পাঠে ছড়ানো ছিল। এই পাঠ তিনটিকেই (এবং চতুর্থ একটি, prefix LM) **একই বাক্যের উপর পাশাপাশি** বসিয়ে দেয়, যাতে আসল অন্তর্দৃষ্টিটি অমিসযোগ্য হয়ে ওঠে: এগুলো ভিন্ন architecture নয়, বরং মূলত একই Transformer stack-এর উপর প্রয়োগ করা ভিন্ন ভিন্ন **loss-masking recipe**।

## এই পাঠে কী শেখানো হয়

- Recap: causal LM, masked LM ও span corruption, এখন সরাসরি তুলনা করা হয়েছে
- Prefix LM: চতুর্থ একটি হাইব্রিড বৈকল্পিক (bidirectional prefix, causal continuation)
- প্রতিটি objective-এর অধীনে কোন কোন token position loss-এ অবদান রাখে, এবং প্রতিটি position কী target predict করে
- কোন objective কোন architecture family-র জন্য উপযুক্ত, এবং কেন
- "এক architecture, অনেক recipe" — মূল কথা

## 1. চারটি objective, এক নজরে

| Objective | Attention pattern | কী predict করা হয় | Loss কোথায় (কোন position-এ) গণনা করা হয় |
|---|---|---|---|
| **Causal LM** ([Phase 02 L6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)) | Causal (প্রতিটি position শুধু অতীত দেখে) | প্রতিটি position *পরবর্তী* token predict করে | Sequence-এর প্রতিটি position |
| **Masked LM** ([Phase 03 L2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md)) | সম্পূর্ণ bidirectional (কোনো mask নেই) | প্রায় 15% position, context থেকে, তাদের *নিজের মূল* token predict করে | শুধু masked position-গুলো |
| **Span corruption** ([Phase 03 L3](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md)) | Bidirectional encoder + causal decoder | Decoder *অনুপস্থিত spans*-কে predict করে, সেগুলো sentinel-এর পর যুক্ত করা হয় | (ছোট) target sequence-এর প্রতিটি position |
| **Prefix LM** (এই পাঠ) | Prefix-এর উপর bidirectional, বাকি অংশে causal | শুধু non-prefix position-গুলো *পরবর্তী* token predict করে | শুধু non-prefix position-গুলো |

## 2. Recap: causal, masked ও span corruption

**Causal LM** token `0..T-2` খাওয়ায় এবং প্রতিটি position-কে পরের token predict করতে তত্ত্বাবধান (supervise) করে, কঠোর causal mask ব্যবহার করে যাতে কোনো position সামনে তাকিয়ে "প্রতারণা" করতে না পারে — [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction)-এ প্রশিক্ষিত recipe-টি অবিকল এটিই। এটি **decoder-only** মডেলের জন্য স্বাভাবিক পছন্দ, কারণ generation ও training-এ *অভিন্ন* forward-pass আকৃতি ব্যবহৃত হয়: inference-এও আপনার কাছে শুধু "এখন পর্যন্ত সবকিছু" থাকে এবং পরবর্তীটা predict করতে হয়।

**Masked LM** causal mask একেবারেই ব্যবহার করতে পারে না — **encoder-only** মডেলের পুরো উদ্দেশ্যই হলো প্রতিটি position এক পাসে পুরো sequence-কে দুদিক থেকে দেখা ([Phase 03 Lesson 2 §1](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#1-just-the-encoder-stack--bidirectional-by-construction))। যেহেতু bidirectional position অন্যথায় নিজের input কপি করে তুচ্ছভাবে নিজেকে "predict" করতে পারত, MLM বদলে input-এর প্রায় 15% position-কে corrupt করে (বেশিরভাগকে `[MASK]`-এ), এবং অন্য সব জায়গায় loss `-100`/ignore-index ব্যবহার করে শুধু সেই corrupted position-গুলোর *মূল* পরিচয় ফিরিয়ে আনার চেষ্টা করায়।

**Span corruption** হলো encoder-decoder মডেল যা করবে, MLM-এর একই অন্তর্নিহিত ধারণা (input-এর কিছু অংশ corrupt, যা সরানো হয়েছে তা predict) দিয়ে — তবে হাতে একটি সম্পূর্ণ autoregressive decoder থাকায় পুনর্গঠিত: বিক্ষিপ্ত token mask করার বদলে এটি ধারাবাহিক *spans*-কে সরিয়ে দেয়, আর decoder-কে শুধু অনুপস্থিত বিষয়বস্তু generate করতে হয় (sentinel দিয়ে ট্যাগ করা), পুরো মূল sequence নয় — অনেক ছোট, সস্তা target ([Phase 03 Lesson 3 §2](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#2-t5s-pretraining-objective-span-corruption))।

## 3. Prefix LM: চতুর্থ, হাইব্রিড বৈকল্পিক

Prefix LM (UniLM-এ ব্যবহৃত, এবং PaLM-এর training-এর একটি বৈকল্পিকে) জিজ্ঞেস করে: *যদি আপনি একটাই decoder-only stack রাখেন, কিন্তু sequence-এর শুধু প্রথম অংশে causal mask শিথিল করেন?* কংক্রিটভাবে, প্রতিটি training sequence-কে একটি **prefix** (ধরা যাক প্রথম `k`টি token) এবং একটি **continuation**-এ ভাগ করুন:

```
positions 0..k-1  (prefix):        every position can attend to every OTHER prefix position, bidirectionally
positions k..T-1  (continuation):  each position attends causally -- to the whole prefix, plus everything
                                    in the continuation up to and including itself
```

Loss **শুধু continuation position-গুলোর** উপর গণনা করা হয়, প্রতিটি তার পরের token predict করে — হুবহু causal LM-এর মতো; prefix position-গুলোতে কোনো loss-ই নেই, সেগুলো শুধু আরও সমৃদ্ধ, bidirectional-প্রক্রিয়াজাত context সরবরাহের জন্য থাকে। এর ফলে দুটি আলাদা stack ছাড়াই একটি সত্যিই দরকারি ধর্ম পাওয়া যায়: input-এর "প্রশ্ন" বা "instruction" অংশের bidirectional বোঝাপড়া, সাথে "উত্তর" অংশের জন্য সাধারণ autoregressive generation — সব এক শেয়ার্ড weight-set এবং একটি attention implementation-এর ভেতরে, শুধু শুদ্ধ causal LM বা শুদ্ধ MLM-এর চেয়ে ভিন্ন mask আকৃতিসহ। যান্ত্রিকভাবে এটা Lesson 1-এর causal mask-ই, যার উপরের-বাঁয়ের `k x k` block-টি un-mask করা।

## 4. কোন objective কোন architecture-এর জন্য, এবং কেন

প্যাটার্নটি কাকতালীয় নয় — প্রতিটি objective আকৃতিগ্রহণ করেছে সংশ্লিষ্ট architecture-কে যেভাবে ভালো হতে হবে তার দ্বারা:

- **Decoder-only + causal LM**: training ও inference-এর *একই* গণনাগত আকৃতি (সবসময় এখন-পর্যন্ত-সবকিছু থেকে পরের token predict করা), যার জন্যই decoder-only মডেল খোলামেলা (open-ended) generation-এর স্বাভাবিক পছন্দ — দেখুন [Phase 03 Lesson 1 §5](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#5-why-decoder-only-won)।
- **Encoder-only + MLM**: architecture-তে "পরের token generate" করার কোনো ধারণাই নেই (প্রতিটি position ইতিমধ্যেই পুরো input দেখে), তাই self-supervised signal পাওয়ার একমাত্র উপায় কিছু *তথ্য* লুকিয়ে সেটা ফিরে চাওয়া — bidirectional বোঝার কাজ (classification, embedding)-ই ফল।
- **Encoder-decoder + span corruption / denoising**: encoder MLM-এর দর্শনের মতোই corrupted input-কে bidirectional ভাবে প্রক্রিয়া করে, কিন্তু decoder একটি সম্পূর্ণ generative stack, তাই target masked slot-এর উপর fixed-vocabulary classification-এর বদলে মুক্ত-রূপে generate-করা text হতে পারে — এটিই T5/BART-কে summarization, translation-এর মতো input→output generation কাজে ভালো করে তোলে।
- **Prefix LM**: ইচ্ছা করেই মাঝখানে অবস্থান করে — একটি decoder-only-আকৃতির stack, যা একটি "conditioning" অংশের জন্য bidirectional context ধার নেয়; instruction-ধরনের input-এর জন্য দরকারি, আলাদা দ্বিতীয় decoder stack-এর engineering খরচ ছাড়াই ([Phase 03 Lesson 3 §5](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#5-why-decoder-only-won-anyway-for-general-purpose-llms))।

## 5. মূল কথা: একই Transformer, ভিন্ন recipe

`example.py` একটি toy tokenized বাক্য নেয় এবং চারটি masking recipe-ই প্রয়োগ করে, ছাপে প্রতিটি objective মডেলকে ঠিক কোন input খাওয়ায়, প্রতিটি position-এ কোন target predict করতে বলে, আর কোন position-গুলো আসলে loss-এ অবদান রাখে। অন্তর্নিহিত self-attention ও feed-forward math ([Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) সম্পূর্ণ) চারটির মধ্যে কখনোই বদলায় না — শুধু attention mask-এর আকৃতি এবং loss mask বদলায়। এটা আত্মস্থ করাই পাঠের আসল উদ্দেশ্য: "decoder-only বনাম encoder-only বনাম encoder-decoder" আসলে *masking recipe*-র পার্থক্য, একটি শেয়ার্ড architectural toolkit-এর উপর স্তরিত।

## Video Script Outline

1. Motivation — "দুটি phase জুড়ে তিনটি objective ইতিমধ্যেই মিলেছে, এবার প্রথমবারের মতো পাশাপাশি"
2. দ্রুত recap টেবিল: causal LM, MLM, span corruption
3. চতুর্থ বৈকল্পিক হিসেবে prefix LM-এর পরিচয় — "bidirectional prefix, causal continuation" mask
4. কেন প্রতিটি objective তার architecture family-র প্রকৃত use case-এর সাথে মানানসই
5. `example.py`-এর walkthrough — একটি বাক্য, চারটি masking recipe, একটি মুদ্রিত তুলনা
6. বড় takeaway: এক শেয়ার্ড Transformer toolkit, চারটি ভিন্ন training recipe
7. Recap + Lesson 3-এর প্রিভিউ: objective স্থির হয়ে গেলে, বহু মেশিনে স্কেলে আসলে কীভাবে প্রশিক্ষণ দেবেন?

## Further Reading

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1, causal LM)
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (MLM)
- Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, span corruption)
- Dong et al. (2019), *Unified Language Model Pre-training for Natural Language Understanding and Generation* (UniLM — prefix LM masking ধারণা)
- Chowdhery et al. (2022), *PaLM: Scaling Language Modeling with Pathways* (prefix LM objective-র একটি বৈকল্পিক স্কেলে প্রশিক্ষণ দেয়)