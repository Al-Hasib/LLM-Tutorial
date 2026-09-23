# Quantization

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `02-Quantization`

## কেন এটি গুরুত্বপূর্ণ

এখন পর্যন্ত এই কোর্সের প্রতিটি model 32-bit (বা 16-bit) floating point-এ প্রশিক্ষিত ও সংরক্ষিত হয়েছে। প্রশিক্ষণের জন্য এটি সঠিক পছন্দ — gradients-কে সঠিকভাবে জমা হতে নির্ভুলতা প্রয়োজন — কিন্তু *serving*-এর জন্য এটি ব্যয়বহুল: একটি 7-billion-parameter model float32-এ মাত্র তার weights ধরে রাখতেই 28 GB দরকার, কোনো token-এর context প্রসেস করার আগেই। Quantization সেই ব্যবধান বন্ধ করার প্রথম এবং সবচেয়ে প্রভাবশালী হাতিয়ার: এটি weights (এবং কখনো কখনো activations)-কে 8-bit বা 4-bit integers-এ সংকুচিত করে, memory footprint এবং প্রায়ই latency 2-8x কমায়, বিনিময়ে অল্প, যত্নসহকারে নিয়ন্ত্রিত পরিমাণের সংখ্যাগত error নিয়ে।

এই lesson হল [Lesson 1](../01-GPU-and-Hardware-Fundamentals/README.md)-এর পরে স্বাভাবিক পরবর্তী ধাপ, কারণ এটি সেই lesson-এ প্রতিষ্ঠিত memory-bandwidth-bound বাস্তবতাকে সরাসরি কাজে লাগায়: memory-bandwidth-bound hardware-এ, ছোট weight মানে প্রতি ধাপে সরাতে কম, শুধু সংরক্ষণ করতে কম নয়। এর পরে সবকিছু এর সাথে যুক্ত হয়: [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md)-এ আচ্ছাদিত KV cache প্রায়শই হ্রাস-নির্ভুলতায় সংরক্ষিত হয়; [Phase 03 Lesson 7](../../Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md#4-grouped-query-attention-gqa-a-new-practically-important-variant)-এর GQA/MQA cache-size আলোচনা এবং quantization একই memory bottleneck-এর উপর পরিপূরক, স্বাধীন হাতিয়ার; আর [Lesson 4](../04-Serving-Frameworks/README.md)-এর serving frameworks (বিশেষ করে llama.cpp) বিশেষভাবে quantized model formats দক্ষতার সাথে চালানোর চারপাশে নির্মিত। এটি [Lesson 5](../05-Model-Distillation-and-Pruning/README.md)-এর distillation এবং pruning-এর সাথেও স্বাভাবিকভাবে জোড়া যায় — তিনটিই "স্ক্র্যাচ থেকে retraining ছাড়াই model ছোট/সস্তা করা" কৌশল, ভিন্ন কোণ থেকে — এবং সরাসরি [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)-এর cost/latency trade-off-কে খাওয়ায়।

## এই lesson যা কভার করে

- মূল trade-off: model size এবং গতি বনাম সংখ্যাগত নির্ভুলতা
- INT8/INT4 quantization mechanics: scale, zero-point, quantize/dequantize
- Symmetric বনাম asymmetric quantization
- GPTQ: layer-by-layer, Hessian-aware post-training quantization
- AWQ: second-order math ব্যবহার না করে activation-salient weights রক্ষা করা
- কেন একটি quantized layer-এর output-এ প্রতিটি weight সমানভাবে গুরুত্বপূর্ণ নয়
- FP8 এবং FP4: নতুনতর floating-point formats যাদের পরিশোধ করতে Tensor Core support দরকার, কেবল কম bits নয়
- KV-cache quantization: শুধু weights নয়, [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) যা cache করে তাও quantize করা

## 1. মূল trade-off

একটি model-এর weights কেবল সংখ্যার tensors। প্রতিটি সংখ্যা সংরক্ষণে কম bits ব্যবহার সরাসরি সংকুচিত করে:

- **Memory footprint** — model ধরে রাখতে কতটা RAM/VRAM দরকার (এবং তাই কতগুলো সমকালীন request মানায়, অথবা model কোনো নির্দিষ্ট device-এ আদৌ মানায় কিনা)
- **Memory bandwidth খরচ** — আধুনিক hardware-এ, memory থেকে compute units-এ weights সরানো প্রায়শই প্রকৃত bottleneck (arithmetic নিজে নয়, [Lesson 1](../01-GPU-and-Hardware-Fundamentals/README.md)-এর memory-bound regime), তাই ছোট weights মানে প্রায়ই *দ্রুততর* inference-ও, শুধু কম storage নয়
- **সংখ্যাগত নির্ভুলতা** — weight-এর মূল সঠিক মান হারিয়ে যায়; শুধু একটি approximation টিকে থাকে

Quantization গবেষণা যে engineering সমস্যা সমাধান করে তা হলো: **কীভাবে সবচেয়ে বেশি bits ফেলে দিয়ে সবচেয়ে কম নির্ভুলতা হারানো যায়?** প্রতিটি weight-কে নিষ্পাপভাবে নিকটতম উপস্থাপনযোগ্য low-bit মানে বৃত্তাকার করা ("round-to-nearest," RTN) হল সরলতম পদ্ধতি এবং INT8 পর্যন্ত আশ্চর্যজনকভাবে ভালো কাজ করে, কিন্তু INT4 এবং নিচে মারাত্মকভাবে অবনত হয় — যার জন্যই GPTQ এবং AWQ (নিচে) বিদ্যমান।

## 2. INT8/INT4 quantization mechanics

মানক স্কিম হলো **uniform affine quantization**। float32 weights-এর একটি tensor `x`-এর জন্য, symmetric quantization (zero-point নেই, ধরে নেয় মানগুলো মোটামুটিভাবে শূন্যের চারপাশে কেন্দ্রীভূত — neural network weights-এর জন্য একটি ভালো মিল) এভাবে কাজ করে:

```
scale = max(|x|) / (2^(bits-1) - 1)

quantize:    q = round(x / scale)              # q is an integer in [-(2^(bits-1)-1), 2^(bits-1)-1]
dequantize:  x_hat = q * scale                  # x_hat approximates x, in float
```

`bits=8`-এর জন্য `q` মোটামুটি [-127, 127] জুড়ে; `bits=4`-এর জন্য মাত্র [-7, 7] — একটি সম্পূর্ণ weight matrix-এর worth-এর সংখ্যা উপস্থাপন করতে মাত্র 15টি ভিন্ন integer মান, যার কারণে INT4 INT8-এর চেয়ে অনেক বেশি error-প্রবণ।

**Asymmetric quantization** একটি **zero-point** `z` যোগ করে এবং দরকারী যখন মানগুলো শূন্যের চারপাশে কেন্দ্রীভূত নয় (post-activation মানের জন্য সাধারণ, যা একটি ReLU-এর পরে সব non-negative হয়):

```
scale = (max(x) - min(x)) / (2^bits - 1)
z     = round(-min(x) / scale)

quantize:    q = round(x / scale) + z
dequantize:  x_hat = (q - z) * scale
```

`example.py` র্যান্ডম weight matrices-এর উপর 8-bit এবং 4-bit দুইটাতেই স্ক্র্যাচ থেকে symmetric স্কিম বাস্তবায়ন করে, এবং মূল weights ও dequantized approximation-এর মধ্যে reconstruction error (মূল weights এবং dequantized approximation-এর mean squared error এবং max absolute error) প্রকৃত memory সাশ্রয়ের পাশাপাশি সরাসরি মাপে।

## 3. GPTQ: Hessian-aware post-training quantization

Round-to-nearest প্রতিটি weight আলাদাভাবে quantize করে, weights কীভাবে মিথস্ক্রিয়া করে তা উপেক্ষা করে। **GPTQ** (Frantar et al., 2022) এর উন্নতি করে: এটি একটি layer-এর weight matrix **column by column** quantize করে, এবং প্রতিটি column quantize করার পরে, এটি *অবশিষ্ট, এখনো-quantize-হয়নি* columns-কে সামান্য সামঞ্জস্য করে এইমাত্র সৃষ্ট error-এর প্রতিকারের জন্য — layer-এর **Hessian**-এর একটি approximation (একটি ছোট calibration dataset-এর activations থেকে প্রাপ্ত second-order curvature তথ্য) ব্যবহার করে বের করে কোন compensating adjustment-এর ফলে **layer-এর output**-এ ফলস্বরূপ error ন্যূনতম হয়, কেবল কাঁচা weight মানে নয়।

মূল ধারণাগত পরিবর্তন: GPTQ জিজ্ঞাসা করে না "এই quantized weight কি মূল weight-এর কাছাকাছি?" — এটি জিজ্ঞাসা করে "এই quantized *layer*, বাস্তব calibration data-তে চালালে, আসল layer-এর মতো কাছাকাছি *output* দেয় কি?" এগুলো ভিন্ন উদ্দেশ্য, এবং দ্বিতীয়টি সরাসরি অপ্টিমাইজ করাই GPTQ-কে একই bit-width-এ naive RTN quantization-এর চেয়ে অনেক ছোট quality loss-সহ 4-bit (অথবা এমনকি 3-bit) পর্যন্ত ঠেলে দিতে দেয়।

## 4. AWQ: activation-aware weight quantization

**AWQ** (Lin et al., 2023) একটি সস্তা, পরিপূরক পর্যবেক্ষণ নেয়: এটির কোনো second-order Hessian গণিতের প্রয়োজনই নেই। পরিবর্তে, এটি পর্যবেক্ষণ করে যে যখন আপনি বাস্তব calibration data একটি layer দিয়ে চালান, তখন **weight columns-এর একটি ছোট শতাংশ** — বিশেষ করে, যেগুলো ধারাবাহিকভাবে **বড়-প্রস্থের activations** দিয়ে গুণিত হয় — layer-এর output-এর কাছে অসমভাবে গুরুত্বপূর্ণ। সেই নির্দিষ্ট columns-কে মোটা করে (low bits-এ) quantize করলে অসমানুপাতিক পরিমাণ output error হয়, যদিও *weight values নিজেরাই* অগত্যা অস্বাভাবিক নয় — বড় activations-এর সাথে তাদের মিথস্ক্রিয়াই সেখানে rounding errors-কে ডাউনস্ট্রিমে বর্ধিত করে।

AWQ-এর সমাধান: activation statistics থেকে columns-এর সেই ছোট salient ভগ্নাংশ শনাক্ত করুন (backward pass বা Hessian প্রয়োজন নেই — calibration set-এর উপর শুধু একটি forward pass), এবং হয় সেই columns-কে উচ্চ নির্ভুলতায় রাখুন (একটি mixed-precision layer), অথবা মূল পত্রে, quantization-এর আগে per-channel সেগুলোকে স্কেল-আপ করে পরে নিচে ফিরিয়ে আনুন (গাণিতিকভাবে সীমিত quantization "resolution"-এর বেশি অংশ সেই columns-এর দিকে ঠেলে দেয় যেগুলোর প্রয়োজন) যাতে একটি সাধারণ uniform quantizer তাদের নিহিতভাবে রক্ষা করে। উভয় ব্যবস্থা একই লক্ষ্য অর্জন করে: অসম গুরুত্বের জন্য অসম সুরক্ষা।

```
naive RTN INT4:      quantize every weight column uniformly to INT4
AWQ-style INT4:       identify top-s% columns by |activation| magnitude
                      keep those columns at full precision (or scaled favorably)
                      quantize the remaining (1-s)% columns aggressively to INT4
```

`example.py` এই দ্বিতীয় mixed-precision কৌশলের একটি সরলীকৃত সংস্করণ সরাসরি বাস্তবায়ন করে: একটি কৃত্রিম per-column "activation magnitude" vector বাস্তব calibration statistics-এর stand-in হিসেবে দাঁড়ায়, এবং সেই magnitude অনুযায়ী top-k% columns fp32-এ রাখা হয় বাকিগুলো INT4-এ quantized হয় — তারপর মেলানো *কার্যকর* গড় bit-width-এ naive uniform INT4-এর বিরুদ্ধে reconstruction error তুলনা করা হয়।

## 5. Quantization-এর খরচ আপনার কী, আর কী নয়

- **যা আপনি সাশ্রয় করেন**: memory (প্রতি bits-per-weight-এ রৈখিক), memory bandwidth, এবং প্রায়ই wall-clock latency, বিশেষত memory-bandwidth-bound hardware-এ
- **যা আপনি ঝুঁকি নেন**: নির্ভুলতার অবনতি, যা GPTQ বা AWQ-এর মতো বুদ্ধিমান স্কিম ছাড়া INT8-এর নিচে তীক্ষ্ণভাবে বাড়ে; নির্দিষ্ট layer (যেমন শেষ output projection, অথবা LLM.int8() পত্রে নথিভুক্ত নির্দিষ্ট "outlier" activation channels) অন্যদের চেয়ে বেশি সংবেদনশীল এবং অন্যথায় 4-bit model-েও কখনো কখনো উচ্চ নির্ভুলতায় রাখা হয়
- **যা পরিবর্তন হয় না**: model-এর architecture, parameter count, বা প্রশিক্ষণ — quantization *প্রশিক্ষণের পরে* প্রয়োগ করা হয় (post-training quantization, PTQ) এবং মূল model-এ কোনো gradient update প্রয়োজন হয় না (যদিও এটি একটি ছোট calibration dataset ব্যবহার করতে পারে), যার কারণেই স্ক্র্যাচ থেকে একটি ছোট model পুনরায় প্রশিক্ষণের ([Lesson 5](../05-Model-Distillation-and-Pruning/README.md)) তুলনায় এটি এত সস্তা, জনপ্রিয় প্রথম হাতিয়ার

## 6. FP8 এবং FP4: নতুনতর, Tensor-Core-native formats

§1-5-এর সবকিছু **ইন্টিজার** formats (INT8/INT4)-এ quantize করে: সম্পূর্ণ representable range জুড়ে একটি স্থির step size, যেখানে বিতরণের সমস্ত "আকৃতি" বাইরে থেকে scale এবং zero-point দ্বারা সামলানো হয়। **FP8** এবং **FP4** পরিবর্তে একটি *floating-point* বিন্যাস রাখে — sign, exponent, mantissa bits, একই ধারণা যা [Phase 04 Lesson 4 §1](../../Phase-04-Pretraining-LLMs/04-Mixed-Precision-and-Optimization/README.md#1-fp32-vs-fp16-vs-bf16) fp32/fp16/bf16 তুলনা করতে ব্যবহার করেছিল — শুধু মোট bits অনেক কম সহ:

```
FP8 E4M3:  1 sign + 4 exponent bits + 3 mantissa bits   (more precision, less range -- good for weights/activations)
FP8 E5M2:  1 sign + 5 exponent bits + 2 mantissa bits   (more range, less precision -- good for gradients)
FP4 E2M1:  1 sign + 2 exponent bits + 1 mantissa bit    (only 16 representable magnitudes total)
```

একটি floating-point exponent (INT8/INT4-এর স্থির step size-এর বদলে) রাখার অর্থ হলো FP8/FP4 খুব ছোট এবং খুব বড় উভয় মানকে উপস্থাপন করে, সমস্ত কাজ একটি পৃথক per-tensor scale ছাড়াই — LLM activations-এর যে বিস্তৃত dynamic range থাকে তার জন্য দরকারী। ধরা, [Lesson 1 §7](../01-GPU-and-Hardware-Fundamentals/README.md#7-compute-units-in-practice-cuda-cores-vs-tensor-cores)-এ hardware পাশ থেকে আচ্ছাদিত: FP8/FP4 কেবল সেই GPU generations-এ *গতি* জয় (শুধু memory নয়) সরবরাহ করে যাদের Tensor Cores nativeভাবে সেই format-এ matmuls কার্যকর করে (যেমন FP8-এর জন্য NVIDIA Hopper, FP4-এর জন্য Blackwell) — পুরোনো hardware-এ একই FP8 tensor চালালে এটি compute-এর আগে upcast করতে হয়, memory সাশ্রয় রেখে কিন্তু throughput লাভ হারিয়ে। `example.py` এর স্ক্র্যাচ-থেকে-লেখা quantizer-কে একটি FP8 (E4M3-style) সিমুলেটর দিয়ে বাড়িয়ে দেয় এবং একই *8-bit* বাজেটে INT8-এর সাথে দুটি ভিন্ন tensor-এ তুলনা করে — একটি সংকীর্ণ, Gaussian-সদৃশ (উপরের weight matrix-এর মতো) এবং অন্যটি কয়েকটি বিরল, বড়-প্রস্থের outliers-সহ (বাস্তব activations-এর মতো) — এবং দেখায় যে বিজয়ী আসলে উল্টে যায়: INT8-এর স্থির step size সংকীর্ণ tensor-এ জেতে, কিন্তু outliers range প্রশস্ত করলে FP8-এর floating exponent জেতে, কারণ INT8-এর একক per-tensor scale একক বৃহত্তম মান থেকে ক্রমাঙ্কিত হয় এবং সেই মান বাড়লে অন্য সব জায়গায় অবনত হয়।

## 7. KV-cache quantization: শুধু weights নয়, Lesson 3 যা cache করে তা quantize করা

এখন পর্যন্ত এই lesson-এর প্রতিটি কৌশল **weights** quantize করে — প্রশিক্ষণের পরে স্থির থাকা parameters। কিন্তু [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) একটি দ্বিতীয়, *ক্রমবর্ধমান* memory ভোক্তা প্রতিষ্ঠা করেছিল: KV cache, যা model size-এর বদলে batch size এবং context length-এর সাথে বাড়ে, এবং দীর্ঘ context-এ weights-এর নিজস্ব memory footprint-ও অতিক্রম করতে পারে। সেই cache-ও quantize করা যায়, স্বাধীনভাবে যে যেকোনো নির্ভুলতায় weights সংরক্ষিত আছে — সাধারণত INT8 বা FP8-এ, কখনো কখনো asymmetrically per-channel (যেহেতু বিভিন্ন attention heads-এর K/V বিতরণ scale-এ ভিন্ন হতে পারে, প্রতিধ্বনি করে কেন AWQ (§4) columns-কে uniform নয় বরং অসমভাবে ব্যবহার করে)।

এখানে trade-off weights-এর চেয়ে তীক্ষ্ণ: weights একবার, offline, আপনার পছন্দমতো যত calibration data-সহ quantize করা হয়; KV cache **অনলাইনে** quantize হয়, generation চলাকালীন প্রতিবার একটি নতুন token-এর K/V vector-এর সাথে — scale/zero-point ক্রমাঙ্কনের জন্য ভবিষ্যতের data-এর কোনো ব্যাচ নেই, শুধু একটি চলমান অনুমান। এই কারণেই KV-cache quantization স্কিমগুলো (যেমন KIVI, Liu et al. 2024) কৌশলের উপর ঝুঁকে থাকে যেমন একটি সংক্ষিপ্ত সাম্প্রতিক window পূর্ণ নির্ভুলতায় রাখা এবং শুধুমাত্র পুরোনো, ইতিমধ্যে-"স্থির" এন্ট্রিগুলো quantize করা, অথবা একটি একক দুর্ভাগ্যজনক scale অনুমানের সৃষ্ট error নিয়ন্ত্রণ করতে per-tensor-এর বদলে per-channel quantize করা। ফল, সরাসরি [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md#3-the-kv-cache-pay-for-each-tokens-kv-exactly-once)-এর cache-size ফর্মুলায়, একই `bytes_per_value` পদ যা GQA/MQA `num_kv_heads` হ্রাস করে সংকুচিত করে — quantization এবং GQA/MQA সেই একই ফর্মুলার উপর দুটি স্বাধীন, স্তুপযোগ্য হাতিয়ার, একটি cached vectors-এর *সংখ্যা* সংকুচিত করে, অন্যটি প্রতিটির *আকার*।

## Video Script Outline

1. Motivation — একটি 7B float32 model-কে শুধু memory-তে থাকতেই 28 GB দরকার; quantization তা সংকুচিত করার সবচেয়ে সস্তা উপায়
2. মূল mechanism: scale, quantize, dequantize — symmetric বনাম asymmetric, হাতে ধরে কাজ করা
3. কেন round-to-nearest INT4-এ ভেঙে পড়ে: কম representable মান, বড় rounding error
4. GPTQ: column-by-column quantize, অবশিষ্ট columns-এর প্রতিকারে Hessian তথ্য ব্যবহার, LAYER OUTPUT error ন্যূনতম
5. AWQ: Hessian গণিত এড়িয়ে যান, শুধু activation-salient weight columns-এর ছোট শতাংশ রক্ষা করুন
6. FP8/FP4: INT8/INT4-এর একটি floating-point বিকল্প, এবং কেন speedup ঠিক সেই format-এর জন্য Tensor Core support-এর উপর নির্ভর করে
7. KV-cache quantization: একটি স্থির, offline weight tensor-এর বদলে একটি ক্রমবর্ধমান, online cache quantize করা — এবং কেন এটি একটি কঠিনতর calibration সমস্যা
8. `example.py`-এর walkthrough — স্ক্র্যাচ থেকে INT8/INT4/FP8 quantization, পরিমাপিত error এবং memory সাশ্রয়, তারপর AWQ-style mixed-precision demo
9. Recap: quantization এই phase-এর তিনটি "training-এর পরে সংকুচিত করুন" কৌশলের প্রথমটি, KV-cache কৌশল এবং distillation/pruning-এর পাশাপাশি

## Further Reading

- Frantar, Ashkboos, Hoefler, Alistarh (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Lin, Tang, Tang, Yang, Dang, Han (2023), *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- Dettmers, Lewis, Belkada, Zettlemoyer (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*
- Micikevicius et al. (2022), *FP8 Formats for Deep Learning*
- Liu, Wang, Dao, et al. (2024), *KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache*