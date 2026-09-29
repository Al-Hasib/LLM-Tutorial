# Frontier Inference Systems

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `11-Frontier-Inference-Systems`

## কেন এটি গুরুত্বপূর্ণ

এই lesson পুরো phase-এর capstone — কোনো নতুন স্বাধীন কৌশল নয়, বরং সেই জায়গা যেখানে আগের প্রতিটি lesson-এর ধারণাকে এমন একটি বিন্দুর বাইরে ঠেলে দেওয়া হয় যেখানে একটি একক server বা একটি একক, একরূপ batching policy আর যথেষ্ট নয়। [Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) প্রতিষ্ঠা করেছিল যে prefill compute-bound এবং decode memory-bound; [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) দিয়েছিল KV cache এবং এটিকে সংকুচিত করার GQA/MQA ফর্মুলা; [Lesson 4](../04-Serving-Frameworks/README.md) দিয়েছিল একটি server-এর ভেতরে PagedAttention এবং স্বয়ংক্রিয় prefix sharing; [Lesson 6](../06-Cost-and-Latency-Optimization/README.md) দিয়েছিল batching ও prefix caching-এর cost/latency trade-off; এবং [Lesson 10](../10-Production-Serving-and-Benchmarking/README.md) দিয়েছিল অনেক replica জুড়ে fleet-scale routing ও observability। বাস্তব জগতের সবচেয়ে বড় serving deployment-গুলো — যেগুলো আসলে বিশাল সমান্তরাল scale-এ frontier model চালাচ্ছে — ওই প্রতিটি ধারণাকে নিয়ে আরও প্রসারিত করে: একটি GPU-তে দুটি বিপরীত-আকৃতির workload interleave করার বদলে, সেগুলোকে hardware-এর আলাদা, বিশেষায়িত pool-এ ভাগ করে দাও; একটি server-এর local prefix cache-এর বদলে, cache-topology সচেতনতা নিয়ে পুরো একটি fleet জুড়ে route করো; অভিন্ন accelerator-এর একটি একরূপ fleet-এর বদলে, প্রতিটি workload-এর resource profile-এর সাথে মেলাতে ইচ্ছাকৃতভাবে বিভিন্ন hardware generation মিশিয়ে দাও; এবং শুধু heads জুড়ে K/V ভাগ করে (GQA/MQA) KV cache সংকুচিত করার বদলে, per-head K/V আদৌ cache না করে এটিকে আরও অনেক বেশি সংকুচিত করো। এর কোনোটিই কোনো স্থির, সমাপ্ত recipe নয় — এটিই সেই জায়গা যেখানে এই মুহূর্তে এই ক্ষেত্রের engineering প্রচেষ্টা কেন্দ্রীভূত।

## এই lesson যা কভার করে

- কেন continuous batching-এর একক-GPU-তে prefill ও decode-কে interleave করা মৌলিকভাবে একটি আপস, সমাধান নয়
- Prefill/decode disaggregation: প্রতিটি phase-এর জন্য আলাদা GPU pool, এবং তাদের মধ্যে KV cache স্থানান্তরের প্রকৃত খরচ
- Fleet scale-এ KV-cache-aware routing: RadixAttention-style prefix sharing-কে একটি server থেকে একটি সম্পূর্ণ load-balanced fleet-এ প্রসারিত করা
- Heterogeneous serving: একটি একরূপ fleet ধরে নেওয়ার বদলে GPU generation-কে workload-এর আকৃতির (এবং খরচের) সাথে মেলানো
- Multi-head Latent Attention (MLA): per-head K/V-এর বদলে প্রতি token-এ একটি একক low-dimensional latent-এ KV cache সংকুচিত করা
- কীভাবে একটি বাস্তব frontier serving stack quantization, MLA/GQA, PagedAttention, disaggregation, heterogeneous hardware এবং fleet routing সবকিছু একসাথে compose করে

## 1. Recap: continuous batching prefill ও decode-কে একটি GPU ভাগ করে নিতে বাধ্য করে

[Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) সেই মূল টানাপোড়েনটি প্রতিষ্ঠা করেছিল যা এই পুরো lesson সমাধান করে: একটি সম্পূর্ণ prompt-এর উপর prefill-এর একটি বড় matmul compute-bound, অথচ decode-এর এক-সময়ে-এক-token matmul-গুলো memory-bound — বিপরীত resource profile-সহ দুটি workload, একই weight matrices-এর উপর, একই GPU-তে চলছে। [Lesson 4](../04-Serving-Frameworks/README.md)-এর continuous batching (সেখানে §4) decode slot-গুলো পূর্ণ রাখে, এবং chunked prefill (সেখানে §5) একটি দীর্ঘ prompt-এর prefill-কে অন্য সবার decode ধাপ আটকে দেওয়া থেকে বিরত রাখে — দুটোই একটি *একক* GPU-তে দুটি workload-কে interleave করার চতুর উপায়, যাতে একটি অন্যটিকে বঞ্চিত না করে। কিন্তু interleaving তবুও একটি আপস: compute-bound কাজে ভালো হওয়ার জন্য provision করা একটি GPU (বেশি raw FLOPs) কখনোই একই সাথে memory-bandwidth-bound কাজের জন্য আদর্শ পছন্দ নয়, এবং উল্টোটাও সত্য। Decode ধাপের একটি round-এর মধ্যে গুঁজে দেওয়া prefill-এর প্রতিটি chunk, যত ছোটই হোক, তবুও এমন সময় যখন সেই GPU তার সবচেয়ে ভালো কাজটি করছিল না। Interleaving কৌশল টানাপোড়েনটিকে লুকিয়ে রাখে; এটিকে দূর করে না।

## 2. Prefill/decode disaggregation: আলাদা pool, একটি প্রকৃত network খরচ

আরও মৌলিক সমাধান হলো দুটি phase-এর মধ্যে একটি GPU ভাগ করা পুরোপুরি বন্ধ করা। **Prefill/decode disaggregation** prefill ও decode-কে GPU-এর শারীরিকভাবে *আলাদা* pool-এ ভাগ করে, যার প্রতিটি নিজের workload-এর জন্য provision ও tune করা — উদাহরণস্বরূপ, prefill pool-এর জন্য কম কিন্তু বেশি compute-ভারী chip (যেখানে raw FLOPs-ই bottleneck), এবং decode pool-এর জন্য বেশি সংখ্যক, সস্তা, memory-bandwidth-অনুকূল chip (যেখানে FLOPs বেশিরভাগই এমনিতেও নষ্ট হয়, §1 অনুযায়ী)। একটি request-এর prefill পুরোপুরি prefill pool-এ চলে, প্রথম output token এবং prompt-এর জন্য একটি সম্পূর্ণ KV cache তৈরি করে; সেই KV cache-কে তারপর decode pool-এর একটি machine-এ **network-এর মাধ্যমে স্থানান্তর** করতে হয়, যা সেখান থেকে generation চালিয়ে যায়, token-এর পর token, prefill pool অন্য request-গুলোর জন্য যা-ই করুক না কেন তা থেকে সম্পূর্ণ স্বাধীনভাবে।

```
Interleaved (single pool):     [prefill chunk][decode][decode][prefill chunk][decode]...
                                one GPU pool, one scheduling policy, workloads compete for the same hardware

Disaggregated (two pools):     prefill pool:  [=== prefill ===] --KV cache transfer--> decode pool
                                decode pool:                                            [decode][decode][decode]...
                                each pool scheduled and scaled independently, on hardware suited to its own workload
```

এটি §1-এর interleaving আপসকে পুরোপুরি দূর করে: prefill pool-কে পুরোপুরি compute-bound throughput-কে কেন্দ্র করে scale ও schedule করা যায়, decode pool-কে পুরোপুরি memory-bandwidth এবং অনেক সমান্তরাল decode stream batching-কে কেন্দ্র করে, এবং কোনোটির scheduling policy-কেই অন্যটির জন্য আপস করতে হয় না। তবে প্রকৃত খরচ সম্পর্কে সৎ থাকুন — এটি একটি সত্যিকারের trade-off, বিনামূল্যের জয় নয়। স্থানান্তরিত KV cache বড় হতে পারে ([Lesson 3 §3-4](../03-KV-Cache-and-Speculative-Decoding/README.md#3-the-kv-cache-pay-for-each-tokens-kv-exactly-once)-এর memory ফর্মুলা মনে করুন: এটি `batch * seq_len * num_kv_heads * d_k * num_layers`-এর সাথে scale করে), তাই একটি দীর্ঘ prompt মানে decode শুরু হওয়ার আগেই দুটি শারীরিক machine-এর মধ্যে network পার হয়ে প্রকৃত, উল্লেখযোগ্য পরিমাণ data যেতে হবে — একটি অতিরিক্ত latency খরচ যা single-pool design-এ কখনো ছিল না। প্রকৃত অতিরিক্ত system জটিলতাও আছে: schedule ও ভারসাম্যপূর্ণ রাখার জন্য দুটি pool (একটির বদলে), একটি network transfer path যাকে দ্রুত ও নির্ভরযোগ্য হতে হবে, এবং একটি কঠিনতর capacity-planning সমস্যা (সময়ের সাথে বদলাতে পারে এমন একটি traffic mix-এর জন্য কতগুলো prefill machine বনাম কতগুলো decode machine) যা একটি একক homogeneous pool কখনো তৈরি করেনি।

## 3. Fleet scale-এ KV-cache-aware routing

[Lesson 4 §6](../04-Serving-Frameworks/README.md#6-sglang-automatic-general-purpose-prefix-sharing)-এর RadixAttention একটি radix tree-এর মাধ্যমে স্বয়ংক্রিয়ভাবে একটি server-এর memory-র *ভেতরে* cached prefix ভাগ করে; [Lesson 6 §2](../06-Cost-and-Latency-Optimization/README.md#2-prefix-caching-paying-for-a-shared-prompt-once) একটি shared prompt-এর prefill-এর জন্য দুবার pay না করার অর্থনীতি কভার করে। এই দুটি ধারণাই ধরে নেয় যে প্রশ্নের request-টি ইতিমধ্যে সেই একটি server-এ (অথবা, §2-এর পরে, সেই একটি decode-pool machine-এ) পৌঁছে গেছে যেখানে ঘটনাক্রমে প্রাসঙ্গিক prefix-টি cached আছে। Fleet scale-এ, অনেক স্বাধীন replica-সহ — অথবা, prefill ও decode disaggregated হয়ে গেলে, প্রতিটিতে অনেক machine-সহ আলাদা pool — সেই অনুমান আর বিনামূল্যে থাকে না: **load balancer নিজেকেই** সিদ্ধান্ত নিতে হয় একটি নতুন request কোন replica-তে যাবে, এবং একটি naive policy-র জানার কোনো উপায় নেই কোন replica ইতিমধ্যে তার cache-এ একটি নির্দিষ্ট prefix ধরে রেখেছে।

[Lesson 10 §3](../10-Production-Serving-and-Benchmarking/README.md#3-load-balancing-across-replicas)-এর round-robin এবং least-outstanding-requests policy দুটোই গঠনগতভাবে cache state-এর প্রতি অন্ধ — তারা load-এর ভারসাম্য রাখে, cache locality-র নয়। এমন একটি replica-তে একটি request route করুন যেটি আগে কখনো তার prefix দেখেনি, এবং সেই replica পুরো, uncached prefill খরচ আবার নতুন করে pay করে, ঠিক সেই redundant খরচ যা এড়ানোর জন্য §2 (Lesson 6-এর মাধ্যমে) আছে — শুধু এখন redundancy-টি একটি routing দুর্ঘটনা, caching-layer সীমাবদ্ধতা নয়। একটি **cache-aware router** এটি ঠিক করে fleet স্তরে track করে কোন replica প্রতিটি পরিচিত prefix-কে সবচেয়ে সম্প্রতি serve করেছে (এবং তাই সম্ভবত এখনো cached রেখেছে), এবং একই prefix ভাগ করা যেকোনো নতুন request-এর জন্য সেই replica-কে অগ্রাধিকার দেয় — কেবল তখনই একটি সাধারণ load-balancing policy-তে ফিরে যায় যখন কোনো replica-তে প্রাসঙ্গিক prefix cached নেই, অথবা যখন যে replica-তে আছে সেটি ইতিমধ্যে overloaded। এটি Lesson 4-এর §4/§6-এর একই prefix-sharing ধারণা, শুধু এক স্তর উপরে তোলা: একটি server-এর local radix tree-এর বদলে, router পুরো fleet-এর cache *topology*-র (একটি আনুমানিক রূপ) বজায় রাখে, এবং এটিকে routing সিদ্ধান্তের নিজেরই একটি first-class input হিসেবে ব্যবহার করে, পরে ভাবার বিষয় হিসেবে নয়।

## 4. Heterogeneous serving: hardware-কে workload-এর আকৃতির সাথে মেলানো

একটি serving fleet-এর কোনো কিছুই এর প্রতিটি GPU-কে একই generation-এর, এমনকি একই ভূমিকার হতে বাধ্য করে না। §1-2 ইতিমধ্যে এই ধারণা পরিচয় করিয়েছে যে prefill ও decode ভিন্ন resource profile চায়; **heterogeneous serving** সেই ধারণাটি নিয়ে অভিন্ন accelerator-এর একটি একরূপ fleet ধরে নেওয়ার বদলে সরাসরি বাস্তব, মিশ্র hardware-এ প্রয়োগ করে। একটি সুনির্দিষ্ট প্যাটার্ন: নতুন, দামি, উচ্চতর-peak-FLOPs chip-গুলো prefill pool-এ রাখুন, যেখানে §1-এর compute-bound workload নতুন silicon-এর প্রতি ডলারে বেশি raw FLOPs থেকে সরাসরি উপকৃত হয় — এবং পুরনো, সস্তা chip-গুলো (অবসরে পাঠানোর বদলে রেখে দেওয়া, অথবা প্রতি GPU-তে সস্তা বলে কেনা) decode pool-এ রাখুন, যেখানে workload memory-bandwidth-bound এবং একটি chip-এর কত peak FLOPs আছে তার প্রতি মূলত সংবেদনশীল নয়, যতক্ষণ এর memory bandwidth ও capacity যথেষ্ট।

এটি সরাসরি [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)-এর cost-engineering কাঠামোর সাথে যুক্ত: hardware-কে workload-এর আকৃতির সাথে মেলানো একটি **cost lever**, শুধু performance lever নয়। Fleet-এর প্রতিটি ভূমিকার জন্য সবচেয়ে নতুন, সবচেয়ে দামি accelerator কেনা এমন decode machine-এ টাকা নষ্ট করে যেগুলো কখনোই সেই chip-এর অতিরিক্ত FLOPs ব্যবহার করার মতো যথেষ্ট compute-bound হবে না, ঠিক যেমন শুধু সস্তা, bandwidth-সীমাবদ্ধ chip কেনা একটি compute-bound prefill pool-কে বঞ্চিত করবে। একটি heterogeneous fleet হলো §1-2-এর disaggregation ধারণার যৌক্তিক পরিণতি: শুধু *আলাদা* pool নয়, বরং *ভিন্ন* hardware থেকে তৈরি pool, যার প্রতিটি সেই নির্দিষ্ট bottleneck-এর (compute বনাম bandwidth) জন্য বেছে নেওয়া যা তার workload আসলে আঘাত করে।

## 5. MLA: Multi-head Latent Attention cache-কে GQA/MQA-এর চেয়েও বেশি সংকুচিত করে

[Lesson 3 §4](../03-KV-Cache-and-Speculative-Decoding/README.md#4-recap-grouped-query-attention-shrinks-the-cache) KV-cache-size ফর্মুলা দিয়েছিল:

```
KV cache bytes = 2 * batch * seq_len * num_kv_heads * d_k * num_layers * bytes_per_value
```

GQA ও MQA এই ফর্মুলাকে `num_kv_heads` কমিয়ে সংকুচিত করে — কম *স্বতন্ত্র* K/V projections, query heads-এর গ্রুপ জুড়ে ভাগ করা (অথবা, MQA-এর ক্ষেত্রে, সবগুলো জুড়ে)। এটি সাহায্য করে, কিন্তু এটি একটি ভোঁতা হাতিয়ার: একটি shared গ্রুপের প্রতিটি head-কে আক্ষরিক অর্থেই অভিন্ন K/V vectors ব্যবহার করে attend করতে বাধ্য করা হয়, একটি ছোট cache-এর বিনিময়ে কিছু representational বৈচিত্র্য ছেড়ে দিয়ে।

**Multi-head Latent Attention** (MLA, DeepSeek-V2-এর সাথে পরিচিত, DeepSeek-AI 2024) একটি ভিন্ন, আরও আক্রমণাত্মক পদ্ধতি নেয়। প্রতি head-এ আলাদা — এমনকি shared হলেও — K/V vectors আদৌ cache করার বদলে, MLA প্রতি layer-এ প্রতি token-এর জন্য একটি একক, অনেক নিম্নতর-dimensional **latent vector** cache করে, এবং attention-এর সময় শেখা per-head up-projection matrices-এর মাধ্যমে তাৎক্ষণিকভাবে সম্পূর্ণ, per-head K/V পুনর্গঠন করে:

```
GQA/MQA cache:  store num_kv_heads separate (d_k-dimensional) K/V vectors per token per layer
                -> attend directly against the cached K/V, no reconstruction needed

MLA cache:      store ONE d_latent-dimensional latent vector per token per layer  (d_latent << num_kv_heads * d_k)
                -> at attention time: up-project the latent, per head, into full-sized K/V on the fly
                   K_head_i = W_up_K_i @ latent   (and similarly for V), THEN attend as usual
```

ফলস্বরূপ cache-size ফর্মুলা `num_kv_heads * d_k` term-টিকে একটি একক ছোট `d_latent` দিয়ে প্রতিস্থাপন করে:

```
MLA cache bytes = 2 * batch * seq_len * d_latent * num_layers * bytes_per_value
```

যেহেতু `d_latent` সাধারণত এমনকি *একটি* GQA গ্রুপের K/V-এর (`d_k`) চেয়েও অনেক ছোট, `num_kv_heads * d_k`-এর কথা তো বাদই দিন, MLA-এর cache এমনকি আক্রমণাত্মক GQA বা MQA setting যা অর্জন করে তার চেয়েও অর্থপূর্ণভাবে ছোট হতে পারে — যখন per-head up-projection (আনুমানিকভাবে) পূর্ণ multi-head expressiveness সংরক্ষণ করে, GQA যেভাবে heads-এর একটি পুরো গ্রুপকে অভিন্ন, অ-পৃথকীকৃত K/V ভাগ করতে বাধ্য করে সেভাবে নয়। তবে trade-off সম্পর্কে সৎ থাকুন: *প্রতিটি* attention গণনায় latent থেকে per-head K/V পুনর্গঠন করা সামান্য অতিরিক্ত compute যোগ করে — up-projection matmul — যা GQA/MQA-এর সরাসরি-cached-ও-পুনর্ব্যবহৃত K/V-কে pay করতেই হয় না। MLA একটি উল্লেখযোগ্যভাবে ছোট memory footprint-এর বিনিময়ে সামান্য অতিরিক্ত decode-time compute দেয়; সেই বিনিময় সার্থক কিনা তা নির্ভর করে একটি deployment memory-capacity-সীমাবদ্ধ (খুব দীর্ঘ context, খুব বড় batch) নাকি ইতিমধ্যে আরামদায়কভাবে তার memory budget-এর মধ্যে আছে তার উপর। `example.py` §1 একটি সুনির্দিষ্ট দৃষ্টান্তমূলক model configuration-এ, context length-এর একটি sweep জুড়ে, MHA, GQA, MQA ও MLA-এর জন্য পাশাপাশি প্রকৃত cache-memory সংখ্যা গণনা করে।

## 6. Recap: কীভাবে একটি frontier serving stack এই সবকিছু compose করে

একটি বাস্তব frontier serving stack খুব কমই এই phase-এর মাত্র একটি ধারণা ব্যবহার করে — এটি প্রায় সবগুলোকে একসাথে compose করে। Quantized weights ([Lesson 2](../02-Quantization/README.md)) এবং একটি MLA বা GQA architecture ([Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) / উপরের §5) একই সাথে দুটি স্বাধীন দিক থেকে KV cache সংকুচিত করে। PagedAttention এবং continuous batching ([Lesson 4](../04-Serving-Frameworks/README.md)) সেই (এখন ছোট) cache দক্ষতার সাথে পরিচালনা করে এবং প্রতিটি pool-এর *ভেতরে* GPU slot-গুলো পূর্ণ রাখে। Prefill ও decode disaggregated pool-এ চলে (§2), সম্ভবত প্রতিটি pool-এর নির্দিষ্ট bottleneck-এর জন্য বেছে নেওয়া heterogeneous hardware থেকে তৈরি (§4)। একটি cache-aware fleet router (§3) pool-গুলোকে একসাথে বাঁধে, প্রতিটি request-কে সেখানে পাঠায় যেখানে তার prefix ইতিমধ্যে cached এবং তার লক্ষ্য pool-এর capacity আছে, সবকিছুই সেই observability ও autoscaling layer-এর অধীনে যা [Lesson 10](../10-Production-Serving-and-Benchmarking/README.md) পুরো fleet জুড়ে প্রদান করে। এখানে কোনো একক ধারণা সব কাজ করে না — জয় আসে সস্তা weights, একটি ছোট cache, স্মার্টতর memory management, বিশেষায়িত hardware এবং cache-aware routing একটির উপর আরেকটি স্তূপ করা থেকে। এবং এর কোনোটিই সমাপ্ত নয়: disaggregation অনুপাত, cache-aware routing algorithm, এবং MLA-এর মতো attention variant — সবই এখনো সক্রিয়, দ্রুত-চলমান research ও engineering ক্ষেত্র, কোনো স্থির, চূড়ান্ত architecture নয় — frontier এগিয়ে চলতে থাকে কারণ এই প্রতিটি lever-এর এখনো ঠেলার মতো প্রকৃত জায়গা বাকি আছে।

## Video Script Outline

1. Motivation — এটি capstone lesson: phase-এর আগের প্রতিটি ধারণা, একটি server বা একটি একরূপ policy যা করতে পারে তার বাইরে ঠেলে দেওয়া
2. Recap: continuous batching একটি GPU-তে prefill ও decode-কে interleave করে, কিন্তু তাদের বিপরীত resource profile-এর জন্য interleaving একটি আপস, সমাধান নয়
3. Prefill/decode disaggregation: আলাদা pool, স্বাধীন scaling, এবং একটি প্রকৃত KV-cache network transfer-এর সৎ খরচ
4. KV-cache-aware routing: RadixAttention-style prefix sharing-কে একটি server-এর memory থেকে একটি পুরো fleet-এর load balancer-এ তুলে আনা
5. Heterogeneous serving: prefill-এর জন্য নতুন compute-ভারী chip, decode-এর জন্য সস্তা bandwidth-অনুকূল chip, একটি cost lever হিসেবে উপস্থাপিত
6. MLA: GQA/MQA-এর shared-K/V-per-group ধারণা থেকে প্রতি token-এ একটি একক latent vector-এ, attention-এর সময় per-head up-projection-এর মাধ্যমে পুনর্গঠিত
7. `example.py` §1-এর walkthrough — একটি context-length sweep জুড়ে MHA বনাম GQA বনাম MQA বনাম MLA-এর প্রকৃত গণনাকৃত KV-cache-memory সংখ্যা
8. `example.py` §2-এর walkthrough — cache-aware বনাম round-robin fleet routing-এর একটি Monte Carlo simulation, এবং তাদের মধ্যে প্রকৃত hit-rate ব্যবধান
9. Recap: কীভাবে quantization, MLA/GQA, PagedAttention, disaggregation, heterogeneous hardware এবং fleet routing একটি বাস্তব frontier stack-এ একসাথে compose হয়
10. সমাপনী নোট: এগুলো সক্রিয়ভাবে বিকশিত হওয়া system ও research ক্ষেত্র, কোনো সমাপ্ত recipe নয়

## Further Reading

- DeepSeek-AI (2024), *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (Multi-head Latent Attention পরিচয় করিয়ে দেয়)
- Zhong et al. (2024), *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving* (একটি serving architecture হিসেবে prefill/decode disaggregation)
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — ইতিমধ্যে [Lesson 4 §6](../04-Serving-Frameworks/README.md#6-sglang-automatic-general-purpose-prefix-sharing)-এ উদ্ধৃত; RadixAttention হলো সেই single-server prefix-sharing ধারণা যা উপরের §3 fleet scale-এ প্রসারিত করে
- Pope et al. (2022), *Efficiently Scaling Transformer Inference* — ইতিমধ্যে [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) ও [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)-এ উদ্ধৃত; disaggregated ও heterogeneous serving-এর নিচে থাকা at-scale throughput/latency বিশ্লেষণ
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — ইতিমধ্যে [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md)-এ উদ্ধৃত; উপরের §5-এ MLA-এর সাথে যে cache-size baseline-এর তুলনা করা হয়েছে
