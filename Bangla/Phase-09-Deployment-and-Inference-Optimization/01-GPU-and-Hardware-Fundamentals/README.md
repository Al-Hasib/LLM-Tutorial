# GPU এবং Hardware Fundamentals

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `01-GPU-and-Hardware-Fundamentals`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর পরের প্রতিটি lesson নীরবে hardware-এর একটি ছোট গুচ্ছ সত্যের উপর ভর করে, কখনো সেগুলো স্পষ্টভাবে না বলেই: [Lesson 2](../02-Quantization/README.md) বলে যে ছোট weight প্রায়ই *দ্রুততর* weight; [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) বলে যে একটি decode ধাপ "memory-bandwidth-bound, compute-bound নয়"; [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md) বলে যে FlashAttention-এর পুরো উদ্দেশ্যই হলো "ধীরগতির HBM"-এ যাওয়ার round trip এড়ানো। এই দাবিগুলোর কোনোটিই কোথাও থেকে উদ্ভূত নয় — এগুলো কেবল ঘোষিত, কারণ এদের নিচের hardware বাস্তবতাকে assumed জ্ঞান হিসেবে ধরা হয়। এই lesson সেই হারিয়ে যাওয়া ভিত্তি: একটি GPU কীভাবে নির্মিত এবং তাতে data কীভাবে চলাচল করে সেই সম্পর্কে অল্প কয়েকটি সত্য, একবারের জন্য আনুষ্ঠানিকভাবে স্থির করে দেওয়া — যাতে এই phase-এর পরের প্রতিটি "এটা memory-bound" বা "এতে bandwidth সাশ্রয় হয়" দাবির জন্য বিশ্বাসের উপর নির্ভর না করে বাস্তব ভিত্তি থাকে।

## এই lesson যা কভার করে

- LLM inference-এর জন্য যতটুকু জরুরি সেই স্তরে GPU architecture: cores, Streaming Multiprocessors, warps — CUDA programming কোর্স নয়
- Memory hierarchy: HBM বনাম SRAM, এবং কেন তাদের মধ্যে size/speed ব্যবধানই GPU performance-এর কেন্দ্রীয় সত্য
- Memory bandwidth একটি কঠিন, সসীম সম্পদ হিসেবে
- FLOPs: এগুলো কী, এবং Transformer-কে প্রভাবিত করা অপারেশন (matrix multiplication)-এর জন্য কীভাবে গণনা করতে হয়
- Roofline model: একটি মাত্র ratio — arithmetic intensity — যা নির্ধারণ করে একটি অপারেশন compute-bound নাকি memory-bound
- ফলাফল: কেন LLM **prefill** compute-bound এবং **decode** memory-bound, এবং কীভাবে এই একক সত্যই এই phase-এর পরবর্তী অনেক optimization কৌশলের অর্ধেক ব্যাখ্যা করে
- CUDA Cores বনাম Tensor Cores: কেন নিম্ন-নির্ভুলতার matmul (Lesson 2-এর quantized formats) কেবল ছোট নয়, বরং সেগুলো নির্দিষ্ট বিশেষায়িত hardware-এ dispatch হয় যা per instruction-এ সত্যিকার অর্থেই দ্রুততর

## 1. GPU architecture, এখানে যতটুকু গুরুত্বপূর্ণ সেই স্তরে

একটি GPU অনেকগুলো **Streaming Multiprocessor (SM)** দিয়ে গঠিত — একটি আধুনিক data-center chip-এ কয়েক ডজন থেকে শতাধিক — যার প্রত্যেকটি বৃহৎ thread-গোষ্ঠীকে lockstep-এ চালায়, যাদের **warp** বলে (32টি thread একই সময়ে ভিন্ন data-তে একই instruction কার্যকর করে)। এটি CPU-এর design point-এর সম্পূর্ণ বিপরীত: একটি CPU-তে হাতে গোনা কয়েকটি জটিল, উচ্চ-clock-speed cores থাকে, প্রতিটি পৃথকভাবে অপ্টিমাইজ করা হয় দীর্ঘ, শাখাপ্রশাখাপূর্ণ, নির্ভরশীল instruction-এর শৃঙ্খল যত দ্রুত সম্ভব শেষ করার জন্য; একটি GPU-তে থাকে অনেক বেশি সংখ্যক, অনেক সরল compute unit, যারা পৃথকভাবে ধীর ও কম বুদ্ধিমান, কিন্তু *ঠিক একই instruction* বিপুল সংখ্যক স্বাধীন data পয়েন্টে *একই সঙ্গে* প্রয়োগ করার জন্য অপ্টিমাইজ করা। Matrix multiplication — Transformer-এর প্রতিটি layer-কে প্রভাবিত করা অপারেশন (`Q`/`K`/`V` projections, FFN, output head; মনে করুন [Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) এবং [Phase 02 Lesson 5](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#5-the-position-wise-feed-forward-network-ffn)) — ঠিক এই ধরনের workload: একই multiply-accumulate অপারেশন, বিপুল সংখ্যক স্বাধীন output element-এর উপর বারবার। এই মিলটিই একমাত্র কারণ যে LLM চালায় GPU-ই (CPU নয়)।

## 2. Memory hierarchy: HBM বনাম SRAM

একটি GPU-তে দুই ধরনের খুব ভিন্ন memory থাকে। **HBM** (High Bandwidth Memory) হলো বড় পুল — একটি আধুনিক data-center card-এ কয়েক tens of gigabytes — যা model-এর weights, activations এবং KV cache ধারণ করে; এটি পরম মানে দ্রুত, কিন্তু *chip-এর পাশেই বসে থাকা compute units-এর সাপেক্ষে* ধীর। **SRAM** — প্রতিটি SM-এর কাছে অবস্থিত ছোট, দ্রুত on-chip memory (কখনো কখনো shared memory বলা হয়), প্রায়ই পুরো chip জুড়ে মাত্র কয়েক megabytes — সেখানে access করা নাটকীয়ভাবে দ্রুততর, কিন্তু একটি model-এর weights ধারণ করার জন্য অনেক ছোট। প্রতিটি অপারেশনের inputs-কে HBM থেকে SRAM-এ (এবং সেখান থেকে registers-এ) যাত্রা করতে হয়, তার আগে SM সেগুলোর সাথে কিছু গণনা করতে পারে, এবং ফলাফলগুলোকে আবার ফিরে যেতে হয়। এই ব্যবধান — ছোট-ও-দ্রুত বনাম বড়-ও-অপেক্ষাকৃত-ধীর — GPU performance সম্পর্কে সবচেয়ে গুরুত্বপূর্ণ একক সত্যই, এবং [Phase 02 Lesson 7-এর FlashAttention](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md#2-flashattention-dao-et-al-2022-exact-but-io-aware) বিশেষভাবে attention-এর জন্য ঠিক এটিই কাজে লাগায়: `Q`/`K`/`V`-কে এমন block-এ tiling করা যেগুলো SRAM-এ মানায়, ফলে পূর্ণ attention score matrix কখনোই HBM-এ লিখতে হয় না। এই lesson সেই একই HBM/SRAM ব্যবধানের সাধারণ রূপ; সেই lesson তার সবচেয়ে বিখ্যাত worked example।

## 3. Memory bandwidth একটি কঠিন সম্পদ হিসেবে

HBM এবং compute units-এর মধ্যে bytes চলাচল করে একটি স্থির, সসীম হারে — যাকে **memory bandwidth** বলা হয়, GB/s বা TB/s-এ পরিমাপ করা হয়। একটি আধুনিক data-center GPU দৈনন্দিন মানদণ্ডে অত্যন্ত উচ্চ bandwidth দেয় (একটি উচ্চ-শেষের accelerator-এর ক্ষেত্রে প্রতি সেকেন্ডে প্রায় কয়েক terabytes), কিন্তু তবুও এটি একটি কঠিন সীমা: প্রতিটি অপারেশনকে এই খরচ — সময়ের মধ্যে — দিতেই হয়, HBM থেকে তার inputs পড়তে এবং outputs ফিরিয়ে লিখতে, যত দ্রুতই compute units নীতিগতভাবে কাজ করতে পারুক না কেন। Bandwidth এবং compute speed দুটি *পৃথক* hardware স্পেসিফিকেশন, এবং — পরের দুই অংশে যেভাবে নির্ভুল করা হবে — কোনো নির্দিষ্ট অপারেশন কত দ্রুত চলে তা সম্পূর্ণ নির্ভর করে data হাতে পেয়ে সেই অপারেশন কী করছে তার উপর।

## 4. FLOPs: প্রকৃত arithmetic গণনা

একটি **FLOP** হলো একটি floating-point অপারেশন (একটি যোগ, অথবা একটি গুণ)। একটি `(m, k)` matrix-কে একটি `(k, n)` matrix দিয়ে গুণ করে `(m, n)` output উৎপাদনের জন্য, `m*n` প্রতিটি output element হলো `k` পদের একটি dot product — `k` গুণ এবং `k-1` (~`k`) যোগ — সুতরাং মোট খরচ:

```
FLOPs(matmul) = 2 * m * k * n
```

**Peak FLOPs** একটি স্থির hardware স্পেসিফিকেশন: compute units প্রতি সেকেন্ডে সর্বোচ্চ কতগুলো অপারেশন শারীরিকভাবে করতে পারত, যদি তাদের কখনো data-এর জন্য অপেক্ষা না করতে হতো। এটি memory bandwidth (§3) থেকে সম্পূর্ণ পৃথক একটি সংখ্যা — একটি chip-এর peak FLOPs এবং peak bandwidth দুটি স্বাধীন সীমা, এবং একটি অপারেশন কেবলমাত্র যত দ্রুত চলতে পারে, যেই সীমাটি আসলে সে স্পর্শ করে।

## 5. Roofline model: arithmetic intensity নির্ধারণ করে compute-bound বনাম memory-bound

একটি অপারেশনের **arithmetic intensity** সংজ্ঞায়িত করি: এটি প্রতি byte data-তে কতটা arithmetic করে যা তাকে সরাতে হয়:

```
arithmetic intensity = FLOPs performed / bytes moved from memory
```

প্রত্যেক accelerator-এর একটি **ridge point** থাকে — `peak FLOPs / peak memory bandwidth` — সেই arithmetic intensity, যেখানে compute সীমা এবং memory-bandwidth সীমা ঠিক ভারসাম্যপূর্ণ। যে অপারেশনের arithmetic intensity ridge point-এর **উপরে**, তা **compute-bound**: প্রতি byte সরানোর জন্য এত বেশি arithmetic হয় যে compute units-ই bottleneck থাকে, memory bandwidth তাদের সম্পূর্ণরূপে পরিতৃপ্ত করছে না, এবং একটি দ্রুততর chip (আরও peak FLOPs) সত্যিই এটিকে দ্রুত করবে। যে অপারেশন ridge point-এর **নিচে**, তা **memory-bound**: compute units নিষ্ক্রিয় বসে থাকে, bytes আসার অপেক্ষায়, এবং অতিরিক্ত peak FLOPs কোনোভাবেই সাহায্য করবে না — শুধু আরও bandwidth (অথবা কম bytes সরানো) সাহায্য করবে। এই একক ratio-ই "roofline model" (Williams, Waterman & Patterson, 2009)-এর সম্পূর্ণ বিষয়বস্তু — achievable performance-কে arithmetic intensity-র বিপরীতে প্লট করলে, প্রতিটি বাস্তব অপeração ঢালু ছাদ থেকে সমতল ছাদে ওঠা-শীর্ষ একটি graph-এর কোথাও না কোথাও পড়ে, ridge point ঠিক কোণায়।

## 6. ফলাফল: কেন prefill compute-bound এবং decode memory-bound

এটি উপরের সবকিছুর LLM inference-এ সবচেয়ে গুরুত্বপূর্ণ প্রয়োগ, এবং [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) সরাসরি এই ধারণার উপর ভর করে। LLM inference-এর দুটি নামযুক্ত phase আছে: **prefill** — সেই একক সমান্তরাল forward pass যা পুরো input prompt একবারে প্রসেস করে এবং প্রথম output token উৎপাদন করে — এবং **decode** — প্রতিটি পরবর্তী forward pass যা ঠিক আরও একটি token উৎপাদন করে, এইমাত্র তৈরি হওয়া token দ্বারা চালিত। দেখুন প্রতিটি phase একই weight matrix-কে কী করতে বলে:

- **Prefill** পুরো prompt-এর worth-এর activations — `seq_len` token, সম্ভবত হাজার হাজার — একটি matmul-এ প্রতিটি weight matrix-এর বিরুদ্ধে গুণ করে। Weight matrix HBM থেকে একবার পড়া হয়, এবং তার bytes প্রতিটি `seq_len` token-এর worth-এর arithmetic-এ পুনর্ব্যবহৃত হয়: উচ্চ arithmetic intensity, ridge point-এর অনেক উপরে, **compute-bound**। ঠিক এই কারণেই prefill throughput একটি chip-এর কাঁচা FLOPs-এর সাথে বাড়ে, এবং অনেকগুলো prompt-এর prefill একসাথে batching করলে আরও সাহায্য হয়।
- **Decode** ঠিক *এক* token-এর worth-এর activations-কে একই weight matrix-এর বিরুদ্ধে গুণ করে। *সম্পূর্ণ* weight matrix (এবং KV cache থাকলে ক্রমবর্ধমান cache — [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md)) তবুও HBM থেকে পড়তে হয়, কিন্তু এখন কেবল একটি token-এর worth-এর arithmetic-এর জন্য: arithmetic intensity প্রায় তাত্ত্বিক তলায় ভেঙে পড়ে, ridge point-এর অনেক নিচে, **memory-bound**। `example.py` নিচে বাস্তব, কংক্রিট Transformer-আকৃতির matrices-এর জন্য এই পার্থক্য গণনা করে — decode-এর arithmetic intensity প্রায় ঠিক সেই "প্রতি byte-এ এক FLOP-জোড়া" তলায় পড়ে যা একটি single-token matmul হারাতে পারে না, অথচ prefill-এরটি কয়েক order of magnitude উচ্চতর।

এটিই *সেই* কারণ যে decode latency নির্ধারণ করে HBM থেকে weights কত দ্রুত স্রোতের মতো বের হতে পারে, chip-এর peak FLOPs নয় — এবং এটি অনেকগুলো *decode* request একত্রে batching-এর সম্পূর্ণ যুক্তি (অনেকগুলো token-এর worth-এর decode কাজ একটি weight read ভাগ করে নেয়), যা [Lesson 4](../04-Serving-Frameworks/README.md)-এর continuous batching-এর অস্তিত্বের কারণ। `example.py` বিভিন্ন batch size জুড়ে decode-এর arithmetic intensity-ও সরাসরি ঘুরিয়ে দেখায়, কংক্রিটভাবে দেখায় যে decode-কে ridge point-এর উপরে ফিরিয়ে আনতে কতগুলো সমকালীন decode request প্রয়োজন।

## 7. Compute units বাস্তবে: CUDA Cores বনাম Tensor Cores

§1 SM এবং warp-কে সাধারণ শব্দে বর্ণনা করেছিল — "অনেক data পয়েন্টে একই instruction চালানো সরল compute units"। বাস্তবে, একটি আধুনিক data-center GPU-র প্রতিটি SM-এর ভেতরে **দুই ধরনের** arithmetic unit বসে, এবং একটি matmul আসলে কোনটিতে পড়ে তা রoofline গণিতের (§5-6) মতোই গুরুত্বপূর্ণ।

**CUDA Cores** হলো সাধারণ-উদ্দেশ্যের unit: একটি core প্রতি clock cycle-এ একটি scalar fused-multiply-add (`a*b+c`) করে, সাধারণ FP32 (বা FP64/INT32) operands-এ। এগুলোই non-matmul কাজ চালায় (elementwise ops, normalization, activation functions) এবং, ঐতিহাসিকভাবে, matmul-ও চালাত — প্রতি বারে একটি dot-product term করে, অনেক cores-এ সমান্তরালে।

**Tensor Cores** (NVIDIA-র Volta architecture, 2017-এর সাথে প্রবর্তিত) উদ্দেশ্য-নির্মিত matrix-multiply-accumulate unit: একটি একক Tensor Core instruction একই সঙ্গে দুটি ছোট *matrix tile* (যেমন 4x4 বা তার চেয়ে বড় fragment) গুণ ও সঞ্চয় করে, একটি scalar জোড়ার বদলে — CUDA cores দিয়ে একই matmul চালানোর চেয়ে per cycle-এ order-of-magnitude বেশি FLOPs দেয় — কিন্তু **শুধুমাত্র** সেই নির্দিষ্ট হ্রাস-নির্ভুলতা input formats-এর জন্য যেগুলোর জন্য সেগুলো নির্মিত (FP16, BF16, INT8, এবং নতুন chip-এ FP8/FP4; দেখুন [Lesson 2](../02-Quantization/README.md))। FP32-এর একটি matmul সাধারণত Tensor Core-এর দ্রুততম পথগুলো মোটেও ব্যবহার করতে পারে না, অথবা একটি ধীরতর fallback mode ব্যবহার করে; FP16/BF16-এ (বা INT8/FP8-এ quantized) একই matmul ব্যবহার করতে পারে।

এটিই "quantization memory সংকুচিত করে" (§2-3) এবং "quantization inference-কে *দ্রুততর* করে, কেবল ছোট নয়"-এর মধ্যে হারিয়ে যাওয়া সংযোগ: একটি quantized weight শুধু HBM থেকে সরাতে সস্তা নয় (§6-এর memory-bound গল্প) — Tensor Cores-সহ hardware-এ, এটি সেই নির্ভুলতায় strictভাবে উচ্চতর peak FLOPs-সহ একটি hardware unit দিয়েও গণনা করা হয়, যা একটি CUDA core যে FP32 পথ নিত তার চেয়ে বেশি। দুটি প্রভাব যৌগিক: সরাতে কম bytes *এবং* পৌঁছালে দ্রুততর unit দিয়ে গণনা। এটিই কারণ [Lesson 2](../02-Quantization/README.md#6-fp8-and-fp4-newer-tensor-core-native-formats)-এর FP8/FP4 formats শুধু memory আরও চেপে ধরার বাইরেও গুরুত্বপূর্ণ — সেগুলো কেবলমাত্র সেই GPU generation-গুলিতে বাস্তবে পরিশোধ করে যাদের Tensor Cores nativeভাবে সমর্থন করে।

## Video Script Outline

1. Motivation — এই phase-এর প্রতিটি পরের lesson "memory-bound" বা "bandwidth সাশ্রয়" দাবি করে কখনো প্রমাণ না করেই; এই lesson সেটি প্রমাণ করে
2. GPU architecture: SMs এবং warps, কেন matmul-আকৃতির কাজের জন্য বিশাল সরল parallelism কয়েকটি জটিল core-কে হারায়
3. HBM বনাম SRAM: size/speed ব্যবধান, FlashAttention (Phase 02 Lesson 7) কংক্রিট worked example হিসেবে
4. Memory bandwidth একটি কঠিন ছাদ হিসেবে, compute গতির থেকে স্বাধীন
5. FLOPs: matmul-এর arithmetic গণনা, এবং peak FLOPs একটি পৃথক hardware স্পেসিফিকেশন হিসেবে
6. Roofline model: arithmetic intensity, ridge point, compute-bound বনাম memory-bound এক চিত্রে
7. `example.py` Part A-এর walkthrough — prefill-আকৃতির বনাম decode-আকৃতির matmuls-এর জন্য বাস্তব arithmetic-intensity সংখ্যা, বাস্তব GPU-র ridge point-এর বিরুদ্ধে শ্রেণীবদ্ধ, এবং batch-size sweep দেখায় কোথায় decode আবার অতিক্রম করে
8. `example.py` Part B-এর walkthrough — CPU-পরিমাপনযোগ্য analogy, এবং Lesson 2-4-এর optimizations-এ recap, সবগুলো এখন বাস্তব ভিত্তিতে দাঁড়ানো
9. CUDA Cores বনাম Tensor Cores: কেন হ্রাস-নির্ভুলতা formats একটি *দ্রুততর* compute unit পায়, কেবল ছোট memory footprint নয়

## Further Reading

- Williams, Waterman, Patterson (2009), *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (মূল roofline model)
- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md) থেকে পুনর্বিবেচিত, HBM/SRAM ব্যবধানের কংক্রিট worked example যা এই lesson সাধারণীকরণ করে
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* — memory hierarchies এবং throughput-oriented processor design-এর সাধারণ রেফারেন্স
- নির্দিষ্ট chip-গুলির বাস্তব, সাম্প্রতিক peak-FLOPs, Tensor Core throughput এবং HBM-bandwidth স্পেসিফিকেশনের জন্য NVIDIA-র পাবলিক GPU architecture ডকুমেন্টেশন (যেমন Volta, Ampere/Hopper architecture whitepapers)