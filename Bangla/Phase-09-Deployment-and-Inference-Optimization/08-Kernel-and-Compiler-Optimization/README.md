# Kernel এবং Compiler Optimization

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `08-Kernel-and-Compiler-Optimization`

## কেন এটি গুরুত্বপূর্ণ

এই phase এখন পর্যন্ত যত optimization কভার করেছে, প্রতিটি একটি request-এর জন্য *কতটা* কাজ বা memory traffic লাগে তা কমায়: [Lesson 2](../02-Quantization/README.md) প্রতিটি weight সরাতে যত byte খরচ হয় তা সংকুচিত করে, [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) সেই redundant গণনা কমায় যা autoregressive generation অন্যথায় পুনরাবৃত্তি করত, এবং [Lesson 4](../04-Serving-Frameworks/README.md)-এর PagedAttention ও continuous batching অনেকগুলো একযোগে চলা request জুড়ে অপচয়িত memory ও অলস সময় কমায়। এর কোনোটিই বাকি থাকা প্রতিটি কাজের টুকরোকে প্রথমত GPU-তে *dispatch* করার স্থির, per-operation খরচ সম্পর্কে কিছু বলে না — এবং [Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) ইতিমধ্যে দেখিয়েছে যে decode-এর প্রতিটি ধাপ ক্ষুদ্র ও memory-bound, যা ঠিক সেই regime যেখানে একটি স্থির per-launch overhead নগণ্য থাকা বন্ধ করে এবং প্রকৃত গণনার সাথেই পাল্লা দিতে শুরু করে। এই lesson সেই বাকি ফাঁকটি বন্ধ করে: ছোট operation-এর chain-কে কম সংখ্যক, বড় kernel-এ fuse করা যাতে launch করার মতো জিনিসই কম থাকে, এবং যে kernel-গুলো বাকি থাকে তাদের launch overhead দূর করা — optimization-এর সেই layer যা quantization, KV cache, এবং serving-level batching-এর কোনোটির সাথে প্রতিযোগিতা না করে বরং তাদের নিচে বসে থাকে।

## এই lesson যা কভার করে

- Kernel launch: প্রতিটি GPU operation dispatch করার প্রকৃত, স্থির CPU-side খরচ, এবং কেন তা decode-এর ছোট, memory-bound ধাপে সবচেয়ে বেশি গুরুত্বপূর্ণ
- Kernel fusion: elementwise/normalization op-এর chain-কে একটি kernel-এ একত্র করে launch overhead ও HBM round-trip দুটোই কমানো
- Triton: custom fused GPU kernel লেখার জন্য একটি Python-embedded language, এবং FlashAttention-এর প্রকৃত reference implementation হিসেবে এর ভূমিকা
- CUDA Graphs: kernel launch-এর একটি স্থির sequence একবার capture করে একটিমাত্র dispatch দিয়ে replay করা
- torch.compile / TorchInductor: PyTorch-এর built-in JIT compiler যা fusion, Triton kernel generation, এবং CUDA Graph wrapping automate করে
- Quantization, KV cache, এবং serving-level batching-এর পাশাপাশি একটি স্বাধীন, stack করার যোগ্য অক্ষ হিসেবে kernel/compiler optimization কোথায় বসে

## 1. Kernel launch-এর একটি প্রকৃত, স্থির খরচ আছে

GPU যে প্রতিটি operation চালায় — এমনকি দুটি ছোট tensor-এর একটি ক্ষুদ্র elementwise add-ও — CPU থেকে একটি আলাদা **kernel launch** হিসেবে dispatch হয়: driver-কে কাজটি queue করতে হয়, এর argument সেট আপ করতে হয়, এবং GPU-র scheduler-এর হাতে তুলে দিতে হয়, সবই প্রকৃত গণনার একটি একক ইউনিট শুরু হওয়ার *আগে*। এই overhead microsecond-এ মাপা হয়, এবং kernel শুরু হওয়ার পর আসলে কত বেশি বা কত কম কাজ করে তা নির্বিশেষে প্রতি launch-এ একবার দিতে হয়। [Lesson 1](../01-GPU-and-Hardware-Fundamentals/README.md)-এর prefill-এর জন্য — পুরো একটি prompt-এর সমপরিমাণ token-এর উপর একটি বিশাল matmul — এই overhead গণনার পাশে একেবারেই নগণ্য; kernel millisecond ধরে চলে, launch খরচ microsecond, এবং কেউ লক্ষই করে না। কিন্তু [Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) ইতিমধ্যে প্রতিষ্ঠা করেছে যে decode ভিন্ন: পূর্ণ weight matrix-এর বিরুদ্ধে একটি token-এর সমপরিমাণ activation একটি ছোট, memory-bound operation, এবং একটি একক decode ধাপ এরকম কয়েক ডজন ছোট operation একসাথে chain করে (attention, projection, normalization, activation, প্রতি layer-এ, প্রতি layer-এ)। সেই scale-এ, স্থির per-launch overhead যে kernel-কে dispatch করছে তার প্রকৃত গণনার সময়ের সমান বা এমনকি তার চেয়ে বেশি হতে পারে — GPU পরবর্তী কী করতে হবে তা বলে দেওয়ার অপেক্ষায় ততটাই সময় কাটায় যতটা সময় কাজটি করতে কাটায়।

## 2. Kernel fusion: কম সংখ্যক, বড় kernel

**Kernel fusion** হলো সরাসরি সমাধান: একটি chain-এর প্রতিটি op-এর জন্য আলাদা kernel dispatch করার বদলে — ধরুন, একটি residual add, তারপর একটি LayerNorm, তারপর একটি activation function — সবগুলোকে একটি একক kernel-এ একত্র করুন যা একটি dispatch-এ তিনটিই করে। এটি একসাথে দুটি সত্যিকারের আলাদা উপায়ে সাহায্য করে। প্রথমত, এটি সহজভাবেই কম launch, তাই §1-এর স্থির per-launch overhead তিনবারের বদলে একবার দিতে হয়। দ্বিতীয়ত — এবং এটি সেই একই HBM/SRAM ব্যবধান যা [Lesson 1 §2](../01-GPU-and-Hardware-Fundamentals/README.md#2-the-memory-hierarchy-hbm-vs-sram) ইতিমধ্যে বর্ণনা করেছে — তিনটি kernel-এর একটি *unfused* chain প্রতিটি মধ্যবর্তী ফলাফল ধীর HBM-এ লিখে ফেরত পাঠায় এবং তারপর ঠিক পরের kernel-এর জন্য আবার পড়ে আনে, এমন data-র জন্য তিনটি আলাদা round trip যার chip ছেড়ে যাওয়ারই দরকার ছিল না; একটি *fused* kernel পুরো chain জুড়ে প্রতিটি মধ্যবর্তী মান on-chip register বা SRAM-এ রাখে এবং শুধু চূড়ান্ত ফলাফল একবার HBM-এ লেখে। তাই fusion শুধু "তিনটির বদলে একটি dispatch" নয় — এটি তিনটির বদলে একটি HBM write-ও, ঠিক যেভাবে [Lesson 1 §7](../01-GPU-and-Hardware-Fundamentals/README.md#7-compute-units-in-practice-cuda-cores-vs-tensor-cores)-এ quantization-এর ছোট footprint ও দ্রুততর Tensor Core path একে অপরের সাথে যুক্ত হয়ে লাভ বাড়ায়।

## 3. Triton: হাতে লেখা CUDA ছাড়াই fused kernel লেখা

ঐতিহাসিকভাবে, একটি custom fused kernel লেখা মানে ছিল হাতে CUDA C++ লেখা: thread index, shared-memory tile, এবং low-level scheduling হাতে পরিচালনা করা — একটি প্রকৃত, বিশেষায়িত দক্ষতা যা খুব কম ML engineer-এর আছে। **Triton** (Tillet, Kung, Cox, 2019) একটি Python-embedded language যা সেই ফাঁকের বেশিরভাগটাই পূরণ করে: আপনি data-র *block*-এর স্তরে একটি kernel লেখেন (যেমন "input-এর এই tile-টি load করো, এর উপর এই elementwise গণিত করো, এই tile-টি লিখে ফেরত পাঠাও"), এবং Triton-এর নিজস্ব compiler low-level খুঁটিনাটি সামলায় — একটি block-এর ভেতরে thread scheduling, shared-memory allocation, memory-access coalescing — যার জন্য আগে বিশেষজ্ঞ হাতে-tuning লাগত। ফলাফল হলো এমন একটি language যা শুধু একজন CUDA বিশেষজ্ঞ নয়, একজন ML engineer-কেও সাধারণ Python-এর কাছাকাছি কিছুতে একটি সত্যিকারের fused, সত্যিকারের দ্রুত custom kernel লিখতে দেয়। এটি কোনো কাল্পনিক সুবিধা নয়: [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md)-এর FlashAttention হলো এর মূর্ত ফল। সেই lesson-এর নিজস্ব `example.py` স্পষ্ট ছিল যে PyTorch op-এর উপর এর সাধারণ-Python tiling loop কখনোই FlashAttention-এর প্রকৃত wall-clock সুবিধা দেখাতে পারত না, কারণ সেই সুবিধা সম্পূর্ণভাবে আসে **একটি fused kernel থেকে যা প্রতিটি tile-কে on-chip SRAM-এ রাখে, কখনো GPU ছেড়ে না গিয়ে বা Python-level dispatch overhead না দিয়ে** — এবং FlashAttention-এর বহুল-ব্যবহৃত বাস্তব-জগতের reference implementation নিজেই Triton-এ লেখা, হাতে লেখা CUDA-তে নয়। Triton হলো সেই tool যা [Phase 02 Lesson 7 §3](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md#3-the-online-softmax-trick)-এর online-softmax algorithm-কে সেই একক fused kernel-এ পরিণত করে যা [Lesson 1 §2](../01-GPU-and-Hardware-Fundamentals/README.md#2-the-memory-hierarchy-hbm-vs-sram)-এ সাধারণভাবে বর্ণিত HBM round-trip এড়িয়ে যায়।

## 4. CUDA Graphs: একবার capture, বহুবার replay

Fusion (§2-3) একটি chain-এ kernel-এর *সংখ্যা* কমায়। **CUDA Graphs** যে launch-গুলো এখনও বাকি থাকে তাদের জন্য §1-এর overhead-কে ভিন্ন উপায়ে আক্রমণ করে: প্রতিটি decode ধাপে একটি sequence-এর প্রতিটি kernel একে একে dispatch করার বদলে, kernel launch-এর পুরো স্থির sequence-টি একটি graph হিসেবে **একবার capture** করা হয় — ঠিক কোন kernel-গুলো, কোন ক্রমে, কোন argument দিয়ে চলে তার একটি static বর্ণনা — এবং পরবর্তী প্রতিটি call CPU থেকে একটি একক dispatch দিয়ে সেই পুরো graph-টি সহজভাবে **replay** করে। যে workload হুবহু একই operation sequence বারবার পুনরাবৃত্তি করে তার জন্য §1-এর per-launch overhead-এর প্রায় সবটাই অদৃশ্য হয়ে যায়, এবং একটি decode ধাপ ঠিক সেটাই: একই model, একই layer, প্রতিবার একই op sequence, শুধু প্রকৃত tensor data ধাপ থেকে ধাপে বদলায়।

এটি [Lesson 3 §5-6](../03-KV-Cache-and-Speculative-Decoding/README.md#5-speculative-decoding-verify-several-tokens-for-the-price-of-one-pass)-এর speculative decoding-এর একটি ঘনিষ্ঠ কাঠামোগত সদৃশ: দুটি কৌশলই বাইরে থেকে প্রতি "call"-এ বেশি কাজ করে একটি স্থির per-step খরচকে অনেক ধাপ জুড়ে amortize করে। Speculative decoding একটি স্থির memory-bandwidth খরচ (পুরো model-এর weights পড়া) প্রতি target-model pass-এ কয়েকটি যাচাইকৃত token জুড়ে amortize করে; CUDA Graphs একটি স্থির CPU-dispatch খরচ (chain-এর প্রতিটি kernel queue করা) অনেকগুলো replay করা decode ধাপ জুড়ে amortize করে। ভিন্ন bottleneck, সমাধানের একই আকার: স্থির খরচ একবার দাও, বহুবার পুনরায় ব্যবহার করো।

## 5. torch.compile / TorchInductor: উপরের সবকিছু automate করা

হাতে Triton kernel লেখা (§3) বা হাতে CUDA Graph capture করা (§4) দুটোই কাজ করে, কিন্তু দুটোতেই সচেতন, per-model engineering প্রচেষ্টা লাগে। **torch.compile**, PyTorch-এর **TorchInductor** compiler backend-এর সমর্থনে, পুরো pipeline automate করে: একটি সাধারণ eager-mode PyTorch model দেওয়া হলে, এটি forward pass trace করে, fuse করা যায় এমন operation-এর chain চিহ্নিত করে (§2), হাতে লেখা CUDA-র বদলে প্রকৃত Triton code হিসেবে fused kernel তৈরি করে (§3), এবং — যেখানে op sequence call-এর পর call হুবহু একইভাবে পুনরাবৃত্তি হওয়ার মতো যথেষ্ট static — ফলাফলটিকে স্বয়ংক্রিয়ভাবে একটি CUDA Graph-এও (§4) wrap করতে পারে। এটি যে trade-off তৈরি করে তা একটি প্রকৃত মধ্যপন্থা, শুধু §3-4-এর "সহজ সংস্করণ" নয়: নিজে হাতে Triton kernel লেখা বেশি নিয়ন্ত্রণ দেয় (ঠিক কোন op কীভাবে fuse হবে তা tune করতে পারেন) বেশি engineering প্রচেষ্টার বিনিময়ে, অন্যদিকে torch.compile একটি অপরিবর্তিত `nn.Module` থেকে স্বয়ংক্রিয়ভাবে সেই speedup-এর বেশিরভাগটাই দেয়, compilation আবার বন্ধ করলে সাধারণ eager PyTorch হিসেবে এখনও debug ও edit করা যায়। এর সাথে [Lesson 4 §7](../04-Serving-Frameworks/README.md#7-tensorrt-llm-compiled-kernel-fused-inference)-এর TensorRT-LLM-এর তুলনা করুন, যা একটি model-কে **ahead of time** একটি স্থির, হাতে-optimize করা kernel graph-এ compile করে যা একটি নির্দিষ্ট model, precision, এবং target GPU generation-এর সাথে বাঁধা — সর্বোচ্চ single-GPU throughput, কিন্তু একটি সত্যিকারের আলাদা compilation artifact যা ওই তিনটির যেকোনোটি বদলাতে আবার build করতে হয়। torch.compile হাতে লেখা Triton এবং TensorRT-LLM-এর পূর্ণ ahead-of-time compilation-এর মাঝখানে বসে: এখনও একটি live, edit করার যোগ্য PyTorch model, কিন্তু এর নিচে just-in-time compiled, fused, graph-replayed execution।

## 6. পূর্ণ inference stack-এ এটি কোথায় বসে

এই phase-এর এখন পর্যন্ত প্রতিটি lesson, এবং এটিও, inference খরচের একটি *ভিন্ন*, স্বাধীন অক্ষ optimize করে — এগুলোর কোনোটিই একে অপরের সাথে প্রতিযোগিতা করে না, এবং একটি প্রকৃত production stack চারটিই একসাথে stack করে:

| অক্ষ | Lesson | যে প্রশ্নের উত্তর দেয় |
| --- | --- | --- |
| Precision | [Lesson 2: Quantization](../02-Quantization/README.md) | Weights ও activation কোন precision-এ store ও compute করা হয়? |
| Recomputation | [Lesson 3: KV Cache and Speculative Decoding](../03-KV-Cache-and-Speculative-Decoding/README.md) | প্রতি ধাপে পুনরায় গণনা করার বদলে কোন কাজ cache বা skip করা হয়? |
| Scheduling ও memory | [Lesson 4: Serving Frameworks](../04-Serving-Frameworks/README.md) | অনেকগুলো একযোগে চলা request কীভাবে memory ও GPU time ভাগ করে নেয়? |
| Dispatch ও fusion | Lesson 8 (এই lesson) | বাকি থাকা প্রতিটি ধাপ GPU-তে আসলে কতটা দক্ষতার সাথে launch ও execute হয়? |

vLLM-এর মাধ্যমে serve করা একটি quantized, KV-cached, continuously-batched model-ও প্রতিটি decode ধাপে kernel-এর একটি প্রকৃত sequence dispatch করে — quantization, KV cache, এবং PagedAttention সবাই ঠিক করে *কোন* কাজ হবে এবং *কীভাবে তা ভাগ হবে*, কিন্তু একবার ঠিক হয়ে যাওয়ার পর সেই কাজ GPU-র হাতে *কতটা সস্তায়* তুলে দেওয়া হয় সে সম্পর্কে কিছু বলে না। এই lesson ঠিক সেই ফাঁকটি বন্ধ করে, এবং এজন্যই TensorRT-LLM-এর মতো framework ([Lesson 4 §7](../04-Serving-Frameworks/README.md#7-tensorrt-llm-compiled-kernel-fused-inference)) এবং torch.compile-accelerated serving stack Lessons 2-4-এর সবকিছুর বদলে নয়, বরং তার উপরে kernel/compiler optimization layer করে।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. অনুপ্রেরণা — এই phase-এর আগের প্রতিটি lesson কতটা কাজ লাগে তা কমায়; কোনোটিই বাকি থাকা কাজ dispatch করার স্থির খরচ স্পর্শ করে না
2. Kernel launch: প্রতিটি GPU operation-এর প্রকৃত microsecond-scale CPU-side overhead, এবং কেন তা decode-এর ছোট, memory-bound ধাপে প্রাধান্য পায় কিন্তু prefill-এর বড় ধাপে নয়
3. Kernel fusion: op-এর একটি chain-কে একটি kernel-এ একত্র করা, launch সংখ্যা ও HBM round-trip দুটোই কমানো (Lesson 1 §2-এর HBM/SRAM ব্যবধানের সাথে যোগসূত্র)
4. Triton: fused kernel লেখার জন্য একটি Python-embedded language, এবং মূর্ত ফল হিসেবে FlashAttention-এর প্রকৃত reference implementation (Phase 02 Lesson 7-এর সাথে যোগসূত্র)
5. CUDA Graphs: একটি স্থির launch sequence একবার capture করে একটি dispatch দিয়ে replay করা — এবং speculative decoding-এর amortization-এর সাথে কাঠামোগত সাদৃশ্য (Lesson 3 §5-6)
6. torch.compile / TorchInductor: স্বয়ংক্রিয় fusion, স্বয়ংক্রিয় Triton kernel generation, স্বয়ংক্রিয় CUDA Graph wrapping
7. torch.compile বনাম হাতে লেখা Triton বনাম TensorRT-LLM-এর ahead-of-time compilation — একই control/effort/portability trade-off-এর তিনটি বিন্দু
8. `example.py`-এর walkthrough — kernel fusion-এর জন্য একটি প্রকৃত পরিমাপকৃত CPU dispatch-overhead analogue, এবং বেশি decode ধাপের সাথে CUDA Graph replay-এর সুবিধা বৃদ্ধির একটি illustrative সংখ্যাগত model
9. Recap table — quantization, KV cache, serving/batching, এবং kernel/compiler optimization চারটি স্বাধীন, stack করার যোগ্য অক্ষ হিসেবে

## আরও পড়ুন

- Tillet, Kung, Cox (2019), *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*
- NVIDIA, *CUDA Graphs* programming guide documentation
- PyTorch team, *torch.compile* / TorchInductor documentation, এবং "Accelerating Generative AI with PyTorch" blog post series
- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md#আরও-পড়ুন)-এ ইতিমধ্যে পূর্ণভাবে উদ্ধৃত; এখানে অনুপ্রেরণাদায়ক বাস্তব-জগতের Triton kernel হিসেবে cross-reference করা
- NVIDIA, *TensorRT-LLM* documentation এবং GitHub repository — torch.compile-এর ahead-of-time compiled বিপরীত দৃষ্টান্ত হিসেবে [Lesson 4 §7](../04-Serving-Frameworks/README.md#7-tensorrt-llm-compiled-kernel-fused-inference) থেকে পুনরায় দেখা
