# Efficient Attention: FlashAttention, Sparse and Linear Attention

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `07-Efficient-Attention-FlashAttention-and-Approximations`

## কেন এই বিষয়টি গুরুত্বপূর্ণ

[Lesson 2](../02-Self-Attention-and-Multi-Head-Attention/README.md) self-attention সঠিকভাবে তৈরি করেছিল কিন্তু এর খরচ সমাধান করেনি: `Q @ K.T` গণনা করা একটি সম্পূর্ণ `(T, T)` matrix materialize করে, যা sequence দৈর্ঘ্য `T`-তে compute ও memory দুই ক্ষেত্রেই দ্বিঘাত — এমন একটি খরচ যার কথা [Phase 01 Lesson 5 §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity)-এও বলা হয়েছিল এবং তখন থেকেই স্থগিত ছিল। এই কোর্সের আরও দুটি লেসন প্রতিটি একটি *পার্শ্ববর্তী* খরচের মোকাবিলা করে — [Phase 03 Lesson 6](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md) model-এর *position*-ধারণা কতদূর কার্যকরভাবে generalize করতে পারে তা ঠিক করে, আর [Phase 09 Lesson 3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md) autoregressive generation-এর সময় অপ্রয়োজনীয় *পুনঃগণনা* ঠিক করে — কিন্তু কোনোটিই একটি একক সম্পূর্ণ attention pass-এর raw `O(T²)` খরচ স্পর্শ করে না। এই লেসন অবশেষে তা করে — তিনটি সত্যিকারের ভিন্ন সমাধানে: **FlashAttention**, যা প্রকৃত হার্ডওয়্যারে একই ফলাফল অনেক বেশি কার্যকরভাবে গণনা করে, আর **sparse** ও **linear attention**, যা প্রকৃত sub-quadratic scaling-এর বিনিময়ে কী কী গণনা করা হয়ই তা বদলে দেয়।

## এই লেসনে যা যা শেখানো হবে

- কেন self-attention সময় ও memory দুই-ই `O(T²)` — বাস্তব সংখ্যা দিয়ে concrete করা
- FlashAttention: একটি exact, IO-aware পুনর্গঠন — গুণমানের কোনো আপস নেই, কেবল প্রকৃত হার্ডওয়্যারে দ্রুত/স্লিম
- Online-softmax কৌশল, যা কখনোই সম্পূর্ণ `(T, T)` matrix materialize না করেই tiled, exact attention সম্ভব করে
- Sparse attention: কোন অবস্থান কোনটি দেখতে পারে তা সীমিত করা, কিছু context হারানোর খরচে compute কমানো
- Linear attention: softmax-এর বদলে kernel feature map ব্যবহার করে প্রকৃত `O(T)` compute, আর এর RNN-সমতুল্য recurrent রূপ
- কী exact বনাম approximate — সেই সৎ তুলনা, এবং production system-গুলো আসলে কী ব্যবহার করে

## ১. খরচ, concrete করা

Self-attention `scores = Q @ K.T` (shape `(T, T)`) গণনা করে, তারপর softmax, তারপর `@ V`। Compute (`~T² · d`) এবং — ঠিক ততটাই গুরুত্বপূর্ণ — সেই score matrix ধরে রাখার প্রয়োজনীয় memory (`T²`টি float) — দুটোই `T`-তে দ্বিঘাতে বাড়ে। `example.py` প্রকৃত context দৈর্ঘ্যে এটি ছাপে: শুধুমাত্র ৩২টি head-এ float32-এ একটি মাত্র layer-এর score matrix হিসাব করলেই, `T=4,096`-তে প্রায় 2 GB প্রয়োজন, কিন্তু `T=32,768` — ৮ গুণ লম্বা context, যা আধুনিক long-context model-এর জন্য সম্পূর্ণ বাস্তবসম্মত — প্রয়োজন 137 GB-রও বেশি, আর `T=131,072`-এ প্রয়োজন **2 terabyte-এর বেশি**। Context দৈর্ঘ্য ৪ গুণ করলে খরচ প্রায় ১৬-৩২ গুণ বাড়ে, ৪ গুণ নয়। নিচের প্রতিটি কৌশল যে সমস্যার মোকাবিলা করে, এটি তারই concrete আকৃতি।

## ২. FlashAttention (Dao et al., 2022): exact, কিন্তু IO-aware

মূল অন্তর্দৃষ্টিটি **কোনো ভিন্ন গাণিতিক ফলাফল নয়** — FlashAttention naive সূত্রের মতোই attention output গণনা করে, bit-for-bit (ভাসমান-বিন্দু rounding পর্যন্ত)। অন্তর্দৃষ্টিটি হলো, একটি প্রকৃত GPU-তে attention-এর bottleneck সাধারণত raw FLOP নয়, বরং **memory bandwidth**: ধীরগতির HBM (high-bandwidth memory — যা GPU-র on-chip SRAM-এর চেয়ে অনেক ধীর) থেকে সম্পূর্ণ `(T, T)` score matrix বারবার লেখা ও পড়াই wall-clock সময়ের আধিপত্য করে। FlashAttention `Q`, `K`, `V`-কে দ্রুত on-chip SRAM-এ ফিট করার মতো ছোট block-এ tile করে এবং block ধরে block attention প্রক্রিয়া করে, ফলে সম্পূর্ণ `(T, T)` matrix **কখনোই HBM-এ লেখা হয় না**।

## ৩. Online-softmax কৌশল

Softmax-এর একটি চলমান max (সংখ্যাগত স্থায়িত্বের জন্য) ও *সব* `K`/`V`-র ওপর একটি যোগফল প্রয়োজন — কিন্তু tiling-এর অর্থ হলো এক সময়ে কেবল একটি block-এর `K`/`V` দৃশ্যমান। সমাধান: block-গুলোর ওপর হাঁটার সময় একটি চলমান max `m`, একটি চলমান (rescaled) যোগফল `l`, এবং একটি চলমান (rescaled) output accumulator `O` বজায় রাখা। যখনই কোনো নতুন block-এর local max চলমান max-কে ছাড়িয়ে যায়, নতুন block-এর অবদান যোগ করার আগে এ পর্যন্ত যা জমা হয়েছে তা `exp(old_max - new_max)` দিয়ে **rescale** করুন:

```
for each K/V block j:
    scores_j = Q @ K_j.T / sqrt(d_k)                 # (T, block_size) -- never (T, T)
    m_new    = max(m, rowmax(scores_j))
    alpha    = exp(m - m_new)                         # rescale factor for OLD accumulators
    p_j      = exp(scores_j - m_new)
    l        = alpha * l + rowsum(p_j)
    O        = alpha * O + p_j @ V_j
    m        = m_new
output = O / l
```

এটি *exact* softmax-কে ক্রমবর্ধমানভাবে, একবারে একটি block করে গণনা করে, এক সময়ে কখনোই একাধিক block-এর score matrix memory-তে না রেখে। `example.py` এই loop-টি বাস্তবায়ন করে এবং naive attention-এর সাথে যাচাই করে — এমন sequence দৈর্ঘ্যও অন্তর্ভুক্ত যা block size দিয়ে সমানভাবে ভাগ যায় না, এবং causal ও non-causal masking দুই-ই — আর প্রতিটি রান ভাসমান-বিন্দু নির্ভুলতায় পৌঁছায় (পার্থক্য `1e-7` ক্রমে, কেবল "কাছে" নয়)। এটি প্রতিটি পদ্ধতি এক সময়ে ধারণ করা পিক intermediate matrix-ও ছাপে: naive-রটি `T × T`-রূপে বাড়ে; tiled সংস্করণটিরটি `T × block_size`-এ থাকে, নির্দিষ্ট block size-এর জন্য `T`-তে linear — FlashAttention-এর tiling-এর ভিত্তি সেই প্রকৃত memory-shape সাশ্রয়।

**একটি সৎ সতর্কবাণী, সরাসরি script-এ বলা**: `example.py`-এর tiled বাস্তবায়ন সাধারণ PyTorch tensor op-এর ওপর একটি সাধারণ Python `for` loop, কোনো fused CUDA kernel নয় — তাই এটির wall-clock সময়কে একটি বড় `naive_attention` matmul call-এর সাথে মাপলে দেখা যায় tiled সংস্করণ এখানে *ধীর* চলে, সম্পূর্ণরূপে Python-স্তরের loop ও dispatch overhead-এর কারণে। এটি প্রত্যাশিত ও সৎ, কোনো বাগ নয়: প্রকৃত FlashAttention-এর wall-clock সুবিধা সম্পূর্ণরূপে একটি fused kernel থেকে আসে, যা প্রতিটি block-কে on-chip SRAM-এ resident রাখে, কখনো GPU ছেড়ে যায় না বা Python overhead দেয় না — এমন জিনিস যা PyTorch op-এর ওপর কোনো pure-Python loop পুনরুৎপাদন করতে পারে না। এই লেসনের কোড যা *প্রকৃতপক্ষে* প্রমাণ করে: অ্যালগরিদম exact, আর এর পিক materialized-matrix আকার সত্যিই `T²` থেকে `T · block_size`-এ সঙ্কুচিত হয়।

## ৪. Sparse attention: যা গণনা করা হয় তা বদলে ফেলা

FlashAttention-এর বিপরীতে, এটি *result*-কেই বদলায়: প্রতিটি query অবস্থানকে সব `T`-এর বদলে নির্দিষ্ট key অবস্থানের একটি উপসেটে সীমাবদ্ধ করে। `example.py` দুটি প্যাটার্ন একত্রিত করে:

- **Local window** — কেবল একটি নির্দিষ্ট আকারের neighborhood-এর মধ্যে মনোযোগ। এটি ঠিক [Phase 03 Lesson 6 §3-এর sliding-window attention](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md#3-sliding-window-local-attention-fixing-the-compute-not-just-position), যা সেখানে ইতিমধ্যে বাস্তবায়িত — নতুন করে derive করার কিছু নেই, কেবল ধারণাটি পুনরায় ব্যবহার করুন।
- **Global token** (Longformer/BigBird-শৈলী) — কয়েকটি নির্ধারিত অবস্থান, যারা *প্রতিটি* অবস্থানের প্রতি মনোযোগ দেয় এবং *প্রতিটি* অবস্থানও তাদের দিকে মনোযোগ দেয়; এরা hub বিন্দুর মতো কাজ করে, যাতে স্থানীয় সীমাবদ্ধতা সত্ত্বেও তথ্য পুরো sequence জুড়ে রুট হতে পারে।

`example.py` এই সম্মিলিত mask তৈরি করে এবং ঠিক হিসাব করে, সম্পূর্ণ `T²` জোড়ার কত ভগ্নাংশ সত্যিই গণনা করা হয়: window/global-token সংখ্যা স্থির রাখলে, `T` বাড়ার সাথে সাথে সেই ভগ্নাংশ ক্রমাগত সঙ্কুচিত থাকে (কয়েক হাজারের নিচে `T`-তে এক-অঙ্কের শতাংশ) — `O(T · window)` খরচের concrete স্বাক্ষর, `T`-তে linear। গুরুত্বপূর্ণভাবে, ও Parts 2-3-এর FlashAttention-এর বিপরীতে, এই masked attention-কে সম্পূর্ণ attention-এর মতো একই random `Q`/`K`/`V`-তে চালালে **সত্যিই ভিন্ন** output উৎপন্ন হয় (প্রকৃত, অশূন্য mean পার্থক্য, যুক্তিসঙ্গত যেকোনো tolerance-এ সংখ্যাগতভাবে সমান নয়) — sparse attention একটি প্রকৃত approximation; masked-out অবস্থান যা অবদান রাখতে পারত, তা তার হ্রাসকৃত খরচের বিনিময়ে বিলিয়ে দেয়।

## ৫. Linear attention: kernel feature map-এর মাধ্যমে প্রকৃত `O(T)` compute

`softmax(Q K^T) V`-কে `phi(Q) @ (phi(K)^T @ V)` দিয়ে প্রতিস্থাপন করুন, যেখানে `phi` একটি elementwise ধনাত্মক feature map (এখানে `phi(x) = elu(x) + 1`, Katharopoulos et al. 2020 থেকে)। যেহেতু matrix multiplication associative, তাই `phi(K)^T @ V`-কে **প্রথমে** গণনা করলে একটি `(d_k, d_v)` matrix পাওয়া যায় — যা `T`-এর থেকে স্বাধীন — ফলে সম্পূর্ণ গণনার খরচ `O(T² · d)`-এর বদলে `O(T · d²)`: sequence দৈর্ঘ্যে linear, আর (FlashAttention-এর tiling-এর বিপরীতে) `(T, T)` matrix কখনোই গণনা করা হয় না, এমনকি একবারে একটি block করেও নয়।

একটি causal, **recurrent** সমতুল্য রূপও আছে: একটি চলমান `(d_k, d_v)` state একবারে একটি token আপডেট হয়, হুবহু একটি RNN-এর hidden state-এর মতো — `state_t = state_{t-1} + phi(k_t) ⊗ v_t` — সাধারণ softmax attention-এর cache-র ([Phase 09 Lesson 3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md)) বিপরীতে, কোনো বাড়তে থাকা per-token KV cache-রই দরকার নেই। `example.py` ধাপে-ধাপে recurrent রূপ এবং একটি সমান্তরাল cumulative-sum রূপ — দুই-ই বাস্তবায়ন করে এবং যাচাই করে যে তারা সংখ্যাগতভাবে অভিন্ন — causal linear attention গাণিতিকভাবে **একটি RNN**, নিছক RNN-*সদৃশ* নয়।

যেহেতু এটি — Parts 2-3-এর tiling loop-এর বিপরীতে — ভাসমান-বিন্দু অপারেশন সংখ্যায় একটি প্রকৃত অ্যালগরিদমিক হ্রাস (fair, apples-to-apples matmul-বনাম-matmul তুলনা, কোনো Python-loop-বনাম-একক-call বিভ্রান্তি নেই), তাই `example.py`-এর non-causal linear attention-কে naive attention-এর সাথে দেওয়া সময়-তুলনায় `T` বাড়ার সাথে সাথে একটি **প্রকৃত, পরিমাপযোগ্য** wall-clock সুবিধা দেখা যায়: একটি লাইভ রানে, ছোট `T`-তে linear attention naive attention-এর সাথে প্রায় সমতুল্য শুরু হয় এবং `T=2,048`-এ ১০x-এর বেশি wall-clock সুবিধায় পৌঁছে যায়; আর `T` বাড়ার সাথে সাথে naive-র প্রতি-দ্বিগুণকরণ বৃদ্ধি-গুণক ধারাবাহিকভাবে linear attention-এর চেয়ে অনেক বেশি থাকে — `O(T²)` বনাম `O(T)`-এর concrete, পরিমাপিত আকৃতি। (সঠিক গুণকগুলো timing noise-র কারণে রান-ভেদে পরিবর্তিত হয়; script এখানে নির্দিষ্ট মান ধরে না নিয়ে নিজের লাইভ সংখ্যা নিজে ছাপে।)

## ৬. তুলনা — এবং production system-গুলো আসলে কী ব্যবহার করে

| | FlashAttention | Sparse attention | Linear attention |
|---|---|---|---|
| Output বদলায়? | না — exact | হ্যাঁ — প্রকৃত approximation | হ্যাঁ — সম্পূর্ণ ভিন্ন ফাংশন |
| Asymptotic খরচ | এখনও `O(T²)` compute, কিন্তু IO-optimal | `O(T · window)` | `O(T · d²)`, `T`-তে linear |
| আজকের সাধারণ ব্যবহার | কার্যত প্রতিটি প্রকৃত training/inference stack-এ ডিফল্ট | নিবেদিত long-context / local-attention স্থাপত্য (ক্রস-লিংক [Phase 03 Lesson 6](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md)) | প্রভাবশালী, কিন্তু ভাষার কাজে softmax attention-এর তুলনায় প্রকৃত ঐতিহাসিক গুণমানের ব্যবধান এটিকে বেশিরভাগ শীর্ষ LLM-এর বাইরে রেখেছে |

Linear attention-এর recurrent-state দৃষ্টিভঙ্গি — একবারে একটি token আপডেট হয় এমন নির্দিষ্ট-আকারের state — ধারণাগতভাবে [Phase 10 Lesson 3](../../Phase-10-Advanced-and-Frontier-Topics/03-State-Space-Models-Mamba/README.md)-এ আলোচিত state-space-model ধারণার কাছাকাছি: দুটোই attention-এর বাড়তে থাকা, সব-জোড়া গণনাকে একটি ধ্রুব-আকারের চলমান state দিয়ে প্রতিস্থাপন করে, এবং ঠিক একই জিনিস (content-ভিত্তিক, সব-জোড়া selectivity) বিলিয়ে দিয়ে সেখানে পৌঁছায়। অন্যদিকে FlashAttention ইচ্ছাকৃতভাবেই ব্যতিক্রম: এটি একেবারেই কোনো tradeoff নয়, আর তাই — sparse বা linear attention নয় — এটিই প্রায়-সর্বজনীন default হয়ে উঠেছে।

## Video Script Outline

1. Motivation — Lesson 2 attention সঠিকভাবে তৈরি করেছিল কিন্তু তার `O(T²)` খরচ কখনো সমাধান করেনি; তিনটি ভিন্ন সমাধান, তিনটি ভিন্ন tradeoff
2. খরচ, আসল gigabyte-এ, বাস্তব context দৈর্ঘ্যে
3. FlashAttention: exact output, IO-aware — memory-bandwidth bottleneck-ই প্রকৃত লক্ষ্য, FLOP নয়
4. Online-softmax recurrence, ধাপে ধাপে কাজ করা
5. `example.py` Part A-এর ওয়াকথ্রু — naive attention-এর সাথে সংখ্যাগতভাবে exact যাচাই, সঙ্কুচিত পিক-matrix-size টেবিল, এবং সৎ Python-loop-বনাম-fused-kernel wall-clock সতর্কবাণী
6. Sparse attention: local window (Phase 03 Lesson 6-এর রিক্যাপ) + global token — এবার একটি প্রকৃত approximation
7. Linear attention: kernel-feature-map কৌশল, আর recurrent রূপের RNN সমতুল্যতা, সংখ্যাগতভাবে প্রমাণিত
8. `example.py` Part C-এর প্রকৃত wall-clock scaling তুলনার ওয়াকথ্রু, ও চূড়ান্ত তিন-পথ তুলনা টেবিল

## Further Reading

- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao (2023), *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer*
- Zaheer, Guruganesh, Dubey, et al. (2020), *Big Bird: Transformers for Longer Sequences*
- Katharopoulos, Vyas, Pappas, Fleuret (2020), *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*
- Choromanski, Likhosherstov, Dohan, et al. (2021), *Rethinking Attention with Performers*