# Serving Frameworks

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `04-Serving-Frameworks`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 2](../02-Quantization/README.md) *model*-টিকে ছোট করেছে, আর [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) প্রতিটি generation ধাপের *কাজ*-কে ছোট করেছে (KV cache, speculative decoding)। এই lesson দুটোরই উপরের স্তর নিয়ে: প্রকৃত serving software — যেটি একটি প্রশিক্ষিত (সম্ভবত quantized) model নেয়, অনেক সমকালীন ব্যবহারকারীর KV caches-কে memory-তে পরিচালনা করে, এবং সিদ্ধান্ত নেয় কীভাবে requests-গুলোকে বাস্তব hardware-এ batch করা যায়। এমনকি একটি পুরোপুরি অপ্টিমাইজ করা model-কেও অপচয়কারীভাবে serve করা যায় — এই lesson সেই ধারণাগুলো কভার করে যা "একটি model চালানো"-কে রূপান্তরিত করেছে "একটি model *দক্ষভাবে, একসাথে অনেক ব্যবহারকারীর জন্য* চালানো"-তে: vLLM-এর memory management, Hugging Face TGI-এর request scheduling, SGLang-এর আরও সাধারণ prefix sharing, NVIDIA TensorRT-LLM-এর compiled kernels, আর llama.cpp-এর GPU-বিহীন deployment path। [Lesson 6](../06-Cost-and-Latency-Optimization/README.md) সরাসরি এখানকার batching ধারণার উপর ভর করে fleet স্তরে cost ও latency trade-offs যুক্তি করার জন্য।

## এই lesson যা কভার করে

- কেন naive KV-cache memory allocation অগাধ পরিমাণ accelerator memory নষ্ট করে
- vLLM-এর PagedAttention: OS-style virtual memory paging, KV cache-এ প্রয়োগ করা
- Variable-length generations-এ static batching-এর throughput সমস্যা
- Hugging Face TGI-এর continuous (in-flight) batching
- Chunked prefill: continuous batching-এর ভেতরে একটি দীর্ঘ prompt-এর head-of-line blocking ঠিক করা
- SGLang-এর RadixAttention: automatic, general-purpose prefix sharing
- TensorRT-LLM: portability-এর বদলে একটি compiled, kernel-fused execution model
- llama.cpp এবং GGUF: একেবারেই GPU ছাড়া serving
- `example.py`: discrete-event simulations যা real numbers-সহ প্রমাণ করে continuous batching-এর throughput সুবিধা, এবং chunked prefill-এর latency-spike ফিক্স

## 1. সমস্যা: naive KV-cache allocation memory নষ্ট করে

[Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) থেকে মনে করুন, autoregressive generation প্রতিটি token-এর K ও V vectors cache করে যাতে প্রতি ধাপে পুনর্গণনার দরকার না হয়। একটি naive serving implementation প্রতিটি sequence-এর জন্য একটি **বড়, contiguous** buffer বরাদ্দ করে, server-এর সমর্থিত *সর্বোচ্চ সম্ভাব্য* sequence দৈর্ঘ্যের জন্য সাইজ করা (ধরা যাক 4096 token) — যদিও একটি নির্দিষ্ট request হয়তো কখনো end-of-sequence token-এ আঘাত করার আগে মাত্র 30 টি token-ই তৈরি করে।

এটি দুভাবে memory নষ্ট করে:

- **Internal fragmentation**: প্রতিটি sequence-এর reserved buffer-এর বেশিরভাগই তার পুরো lifetime-এ খালি পড়ে থাকে।
- **নিরাপদে বাড়তে না পারা**: দুটি sequence যদি memory-তে পাশাপাশি বসানো হয় এবং একটির প্রত্যাশার চেয়ে বেশি জায়গা দরকার হয়, তাহলে copying ছাড়া তার বাড়ার জায়গা নেই।

যেহেতু GPU memory-ই সেই শক্ত সীমা যা নির্ধারণ করে কতগুলো sequence *সমকালীনভাবে* (batch size) serve করা যায়, নষ্ট হওয়া KV-cache memory সরাসরি throughput-কে সীমিত করে — এটিই naive serving code-এর সবচেয়ে বড় একক inefficiency।

## 2. vLLM-এর PagedAttention

Kwon et al. (2023) operating-systems virtual memory থেকে সরাসরি একটি ধারণা ধার করে: একটি sequence-এর জন্য একটি contiguous buffer-এর বদলে, KV cache-কে ছোট, নির্দিষ্ট-আকারের **blocks** ("pages")-এ ভাগ করা হয়, সাধারণত প্রতিটিতে 16 token-এর জন্য যথেষ্ট। একটি per-sequence **block table** কিছু logical token positions-কে physical blocks-এ ম্যাপ করে, যেগুলো একটি shared physical memory pool-এর *যেকোনো* জায়গায় থাকতে পারে — অগত্যা contiguous নয়, অগত্যা আগে থেকে reserved নয়:

```
Naive:     [ sequence A: reserved for 4096 tokens, uses 30 ]  <- 4066 tokens wasted
           [ sequence B: reserved for 4096 tokens, uses 800 ] <- 3296 tokens wasted

PagedAttention:  physical block pool (shared, fixed-size blocks)
                 sequence A block table -> [ block 7 ]                (30 tokens -> 1 block)
                 sequence B block table -> [ block 2, block 9, ... ]  (800 tokens -> 50 blocks)
```

Blocks বরাদ্দ হয় **on demand**, একবারে একটি, যখনই একটি sequence প্রকৃতপক্ষে বাড়ে — ঠিক যেমন একটি paging OS একটি process-এর virtual address space-এ lazily physical pages বরাদ্দ করে। এটি internal fragmentation প্রায় সম্পূর্ণভাবে দূর করে (একমাত্র waste প্রতিটি sequence-এর *শেষ, আংশিকভাবে-ভরা* block-এর ভেতরে) এবং মানে memory-তে যে number of sequences ফিট হয় তা *প্রকৃত* মোট generated tokens-এর দ্বারা সীমিত, কোনো worst-case reservation-এর দ্বারা নয়। বাস্তবে এটি vLLM-কে একই GPU memory-তে নাটকীয়ভাবে বেশি concurrent sequences প্যাক করতে দেয়, যা — নিচের continuous batching-এর সাথে মিলে — naive Hugging Face `transformers` serving-এর উপর vLLM-এর ব্যাপক প্রতিবেদিত throughput gains-এর উৎস। PagedAttention সস্তা **memory sharing**-ও সক্ষম করে: যদি অনেক requests একটি অভিন্ন prompt prefix ভাগ করে (যেমন একটি system prompt), তাদের block tables একই *physical* blocks-এ (copy-on-write) নির্দেশ করতে পারে, সম্পূর্ণরূপে অপ্রয়োজনীয় KV-cache storage এড়িয়ে — [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)-এর prefix-caching ধারণার একটি প্রিভিউ।

## 3. Static batching-এর throughput ceiling

দক্ষ memory থাকলেও, *কীভাবে* requests-গুলো একটি batch-এ grouped হয় তা গুরুত্বপূর্ণ। সরলতম স্কিম, **static batching**, `N` টি requests-এর একটি নির্দিষ্ট-আকারের batch গঠন করে এবং সেগুলো ধাপে ধাপে একসাথে চালায় যতক্ষণ না প্রত্যেকটি শেষ হয়, শুধুমাত্র তখনই পরের `N` টি অপেক্ষমাণ requests-কে ভর্তি করে:

```
batch = [req_1 (needs 10 tokens), req_2 (needs 200 tokens), req_3 (needs 15 tokens), req_4 (needs 180 tokens)]
```

যেহেতু বাস্তব requests-এর wildly ভিন্ন সংখ্যক output tokens দরকার হয় (এক লাইনের উত্তর বনাম একটি দীর্ঘ ব্যাখ্যা), পুরো batch তার **সবচেয়ে ধীর সদস্য**-এর মতোই দ্রুত। `req_1` ও `req_3` 10-15 ধাপে শেষ হয় কিন্তু তাদের batch slots **idle** পড়ে থাকে — কোনো দরকারী কাজ করছে না — যতক্ষণ না `req_2` ও `req_4` শেষে ধাপ 200-এর কাছাকাছি শেষ হয় — তখনই পরের 4 টি অপেক্ষমাণ requests-এর batch শুরু হতে পারে, যদিও 4 টি slot-এর 2 টি 185 ধাপ ধরে ফ্রি ছিল।

## 4. Hugging Face TGI: continuous (in-flight) batching

TGI (এবং অন্তর্নিহিত ধারণা Orca থেকে, Yu et al. 2022) ঠিক এটিই ঠিক করে: batch-টি একটি নির্দিষ্ট, static group হওয়ার বদলে, server একটি নির্দিষ্ট সংখ্যক **concurrent slots** বজায় রাখে, এবং যে মুহূর্তে কোনো slot-এর sequence শেষ হয়, সেই slot **অবিলম্বে** queue-তে অপেক্ষমাণ পরবর্তী request দিয়ে backfill হয় — বাকি batch-এর জন্য কোনো অপেক্ষা নেই। (জগতে **dynamic batching** শব্দটিও দেখতে পাবেন, অসঙ্গতভাবে ব্যবহৃত: কখনো ঠিক এই per-step rescheduling-এর সমার্থক হিসেবে, কখনো একটি দুর্বল মধ্যম পথ হিসেবে — একটি batch-এর *আকার* admission time-এ কে অপেক্ষা করছে তার ভিত্তিতে dynamically ঠিক হয়, কিন্তু continuous batching যেভাবে করে সেভাবে একটি request mid-flight-এ reshuffle বা backfill হয় না। নিচের scheduling আচরণই গুরুত্বপূর্ণ, লেবেল নয়।)

```
Static:      [====req_1====][xxxxxxxxxxxxxxxxxxxx idle xxxxxxxxxxxxxxxxxxxx]
             [====req_2==========================================================]
             (next batch cannot start until req_2's slot is free too)

Continuous:  [====req_1====][====req_5====][==req_8==][...]     <- slot immediately reused
             [====req_2==========================================================]
```

এটি প্রতিটি slot-কে generation lengths-এর বিতরণ যতই skewed হোক না কেন প্রায় সবসময় দরকারী কাজে ব্যস্ত রাখে — যা ঠিক সেই workload যা বাস্তব chat/completion traffic তৈরি করে (কিছু উত্তর এক বাক্য, কিছু পৃষ্ঠা দীর্ঘ)। `example.py` একটি অভিন্ন workload-এ উভয় স্কিমের প্রকৃত discrete-event simulation গড়ে তোলে এবং ফলে হওয়া মোট সময় ও slot utilization পার্থক্য সরাসরি মাপে।

## 5. Chunked prefill: continuous batching-এর head-of-line blocking ঠিক করা

Continuous batching (§4) *decode*-side idling সমাধান করে, কিন্তু একটি দীর্ঘ prompt আসামাত্রই একটি নতুন সমস্যা প্রবর্তন করে: একটি request-এর **prefill** প্রসেস করা — তার পুরো prompt-এর উপর একক forward pass, সম্ভবত হাজারো token — সাধারণত একটি atomic step হিসেবে dispatch হয়। সcheduler যদি সেই atomic prefill-কে সবার চলমান **decode** ধাপের (প্রতিটি একটি করে নতুন token) সাথে একই slot-tick-এ চালায়, তাহলে সেই অন্য প্রতিটি, ইতিমধ্যে-in-flight request সেই পুরো prefill যতক্ষণ লাগে ততক্ষণ থমকে যায় — মাঝ-কথোপকথনে থাকা interactive users-এর জন্য একটি প্রকৃত latency spike, মাত্র কয়েক মুহূর্ত পরে continuous batching-এর তাদের প্রতি-টিক slot-এর প্রতিশ্রুতির। এটি **head-of-line blocking**: queue-এর সামনে একটি দীর্ঘ কাজের একক পরিমাণ তার পেছনের সবকিছুকে বিলম্বিত করে, এমনকি কাজ যা অন্যথায় তাৎক্ষণিক হতো।

**Chunked prefill** (Sarathi কর্মসূত্র, Agrawal et al.) এটি সরাসরি ঠিক করে: একটি দীর্ঘ prefill-কে ছোট chunks-এ ভাগ করা (প্রতিটি কয়েকশো token), এবং প্রতিটি রাউন্ডে অন্য প্রতিটি in-flight request-এর একটি সাধারণ decode step-এর সাথে prefill-এর একটি chunk interleave করা — হয় একটি দীর্ঘ অবিরাম prefill, অথবা বিশুদ্ধ decode-only rounds-এর বদলে:

```
Atomic prefill (blocks everyone else for the FULL prefill length):
  round 1:  [=========== prefill: all 2000 tokens in one shot ===========]   <- other slots FROZEN this whole time
  round 2:  [decode][decode][decode]...                                      <- other slots resume

Chunked prefill (interleaved, chunk size 256):
  round 1:  [prefill chunk 1/8][decode][decode][decode]...   <- other slots only wait ~256 tokens' worth
  round 2:  [prefill chunk 2/8][decode][decode][decode]...
  ...
  round 8:  [prefill chunk 8/8][decode][decode][decode]...   <- long request's prefill now fully done too
```

প্রতিটি রাউন্ড এখন দুটো workload-এর একটু করে মেশায়: prefill-এর একটি chunk (compute-bound, অনেক token একসাথে random বড় matmuls) এবং অন্য কয়েকটি requests-এর decode ধাপের handful (memory-bandwidth-bound, প্রতিটি একটি token) — batch-এর প্রতিটি request-এর দেখা worst-case per-step latency-কে মসৃণ করে, chunked হওয়া request-টির জন্য একটি ছোট, বাস্তব overhead-এর খরচে (নিজের prefill ভাগ করা ও আবার শুরু করা পুরোপুরি বিনামূল্যে নয়)। `example.py` §3 ঠিক এই trade-off-কে প্রকৃত মাপা সংখ্যায় সিমুলেট করে: chunking অন্য in-flight requests-কে যে worst-case delay-এর হাত থেকে রক্ষা করে, এবং chunked request-টি নিজে তার জন্য যে সামান্য অতিরিক্ত মোট সময় দেয়।

## 6. SGLang: automatic, general-purpose prefix sharing

SGLang-এর (Zheng et al., 2024) প্রধান serving ধারণা **RadixAttention**: PagedAttention §2-এ সক্ষম হওয়া prefix-sharing-টির একটি কঠোরভাবে আরও সাধারণ সংস্করণ। vLLM-এর copy-on-write prefix sharing তখন কাজ করে যখন requests একটি *পূর্ব-নির্ধারিত* prefix ভাগ করে, যেমন একটি নির্দিষ্ট system prompt। RadixAttention বদলে প্রতিটি sequence-এর cached KV blocks-কে একটি shared **radix tree**-তে সংগঠিত করে, এখন পর্যন্ত দেখা প্রকৃত token sequences দ্বারা keyed — তাই *যে কোনো* দুটি requests ঘটনাক্রমে *যে কোনো* সাধারণ prefix ভাগ করলে, আগে থেকে কনফিগার করা নয় বরং স্বয়ংক্রিয়ভাবে আবিষ্কৃত, সেই সংশ্লিষ্ট cached blocks ভাগ করে, memory চাপে tree-এর least-recently-used শাখাগুলো স্বয়ংক্রিয়ভাবে evict হয়, একইভাবে একটি LRU cache যেমন করবে। এটি prefix caching-কে "একটি prefix-এর জন্য কাজ করে যার কথা আপনি server-কে বলেছিলেন" থেকে "যে prefixes বাস্তব traffic-এ আসে সেগুলোর জন্য কাজ করে" পর্যন্ত সাধারণীকরণ করে। SGLang একটি structured-generation-মুখী programming model-ও প্রবর্তন করে — explicit constraints ও control flow-এর সাথে generation interleave করা — serving কর্মক্ষমতার বাইরে একটি দ্বিতীয়, বৈশিষ্ট্যসূচক বৈশিষ্ট্য হিসেবে; এটি [Phase 07 Lesson 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/05-Structured-Output-and-Function-Calling/README.md)-এ গভীরভাবে আচ্ছাদিত একই constrained/structured-generation অঞ্চল, এখানে পুনরুৎপাদিত নয়।

## 7. TensorRT-LLM: compiled, kernel-fused inference

NVIDIA-এর TensorRT-LLM vLLM, TGI ও SGLang — যারা সবাই একটি সাধারণ Python serving loop চালায় যা সাধারণ framework (যেমন PyTorch) অপারেশনগুলো dispatch করে — থেকে একটি মৌলিকভাবে ভিন্ন execution model নেয়। TensorRT-LLM বদলে একটি নির্দিষ্ট model-কে, আগে থেকে, **compile** করে — একটি অত্যন্ত অপ্টিমাইজড hand-fused CUDA kernels-এর graph-এ, *সুনির্দিষ্ট* GPU-কে লক্ষ্য করে যেখানে এটি চলবে — portability বেছে নেওয়ার বদলে (একটি compiled artifact একটি model, একটি precision এবং একটি target GPU generation-এর সাথে বাঁধা, এবং অন্যটিতে যেতে পুনরায় compile করতে হয়) একটি সাধারণ Python loop যা মেলাতে পারে না এমন kernel fusion ও compile-time specialization থেকে আসা সর্বোচ্চ single-GPU, single-vendor throughput-এর জন্য। এটি অন্যদের মতো একই উচ্চ-স্তরের ধারণাগুলোতে মিলিত হয়েছে — in-flight (continuous) batching এবং paged KV-cache management দুটোই সমর্থিত — প্রকৃত পার্থক্যসূচক পছন্দ হলো compiled kernel graph বনাম সাধারণ serving loop, প্রতিটি framework কোন scheduling ধারণাগুলো ব্যবহার করে তা নয়; এই frameworks ধারণাগুলো নতুন আবিষ্কারে প্রতিযোগিতার চেয়ে অনেক বেশি ভাগ করে।

## 8. llama.cpp: GPU ছাড়াই serving

llama.cpp (Gerganov et al.) সম্পূর্ণ ভিন্ন একটি অক্ষ নেয়: GPU throughput সর্বাধিক করার বদলে, এটি **CPUs এবং consumer hardware**-এ দক্ষতার সাথে LLMs চালানোকে লক্ষ্য করে, একেবারেই কোনো GPU প্রয়োজন ছাড়াই। এটি **GGUF**-এ models লোড করে — quantized weights (সরাসরি [Lesson 2](../02-Quantization/README.md)-এর INT8/INT4 quantization-এর সাথে যুক্ত) যোগ tokenizer ও metadata সংরক্ষণকারী একটি একক-ফাইল ফরম্যাট — এবং CPU SIMD instruction sets-এর জন্য hand-optimized quantized matrix-multiply kernels বাস্তবায়ন করে। এটি সেই পথ যা laptop বা phone-এ একটি 7-13B parameter model চালানো সম্ভব করে — vLLM/TGI-এর multi-GPU, many-concurrent-user data-center serving-এর চেয়ে একটি মৌলিকভাবে ভিন্ন deployment target, কিন্তু Lesson 2-এর হুবহু একই quantization ধারণাগুলোর উপর নির্মিত।

## 9. এদের মধ্যে নির্বাচন

| Framework        | Optimizes for                                    | KV-cache strategy                                                  | Typical deployment target       |
| ---------------- | ------------------------------------------------- | -------------------------------------------------------------------| -------------------------------- |
| vLLM              | Max throughput, many concurrent users             | PagedAttention (paged, on-demand)                                  | Multi-GPU data center serving    |
| Hugging Face TGI  | Utilization under variable-length traffic         | Continuous batching (+ can combine with paged memory)              | GPU data center serving          |
| SGLang            | Automatic prefix reuse across arbitrary traffic   | RadixAttention (radix-tree paged cache, LRU eviction)               | GPU data center serving, structured-generation workloads |
| TensorRT-LLM      | Maximum single-GPU throughput on NVIDIA hardware  | Paged cache + in-flight batching, inside a compiled kernel graph    | Latency/throughput-critical NVIDIA deployments |
| llama.cpp         | Running at all without a GPU                      | Simple contiguous cache, but a tiny quantized model to begin with  | Laptop / CPU / edge / phone      |

এগুলোর কোনোটি পারস্পরিক-বহির্ভূত ধারণা নয় — আধুনিক serving stacks ক্রমবর্ধমানভাবে paged memory management *এবং* continuous (বা chunked-prefill-aware) batching *এবং* quantized weights একসাথে একই সঙ্গে মেশায়; উপরের frameworks কেবল যেখানে প্রতি ধারণাটি প্রথমে বড় পরিসরে জনপ্রিয় হয়েছিল, অথবা, TensorRT-LLM-এর ক্ষেত্রে, একমাত্র framework যা অন্যদের portability-কে compiled, hardware-specific গতির বিনিময়ে ছাড়ে।

## Video Script Outline

1. Motivation — একটি পুরোপুরি অপ্টিমাইজ করা model-কেও অপচয়কারীভাবে serve করা যায়; এই lesson নিজেই serving software স্তর নিয়ে
2. Naive contiguous KV-cache buffer সমস্যা এবং কেন এটি concurrent batch size-কে সীমিত করে
3. vLLM-এর PagedAttention — KV cache-এ প্রয়োগ করা OS-style paging, block tables, on-demand allocation, prefix sharing
4. Static batching-এর throughput ceiling — পুরো batch তার সবচেয়ে ধীর সদস্যের জন্য অপেক্ষা করে
5. Hugging Face TGI-এর continuous/in-flight batching — তাৎক্ষণিক slot backfill, Orca-র সাথে সংযোগ
6. Chunked prefill — head-of-line blocking concretely, আর interleaved-chunk ফিক্স
7. SGLang-এর RadixAttention — radix tree-র মাধ্যমে automatic prefix sharing, PagedAttention-এর prefix caching-কে সাধারণীকরণ
8. TensorRT-LLM — compiled, kernel-fused execution বনাম একটি সাধারণ serving loop, portability/gতি trade-off
9. llama.cpp এবং GGUF — GPU-মুক্ত deployment path, Lesson 2-এর quantization-এর সাথে সংযোগ
10. `example.py`-এর walkthrough — static বনাম continuous batching-এর, এবং chunked বনাম atomic prefill-এর head-of-line-blocking ফিক্সের discrete-event simulations, সব measured numbers-সহ
11. Recap + pointer Lesson 6-এর cost/latency trade-offs-এর দিকে, যা আজকের batching ধারণাগুলোর উপর নির্মিত

## Further Reading

- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper)
- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models* (origin of continuous/iteration-level batching, which Hugging Face TGI implements)
- Agrawal et al., the Sarathi / Sarathi-Serve line of work on piggybacking decode steps with chunked prefill to fix head-of-line blocking in continuous batching
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention and structured generation)
- NVIDIA, *TensorRT-LLM* documentation and GitHub repository
- Gerganov et al., the `llama.cpp` project and the GGUF file format specification
- Hugging Face, *Text Generation Inference* (TGI) documentation