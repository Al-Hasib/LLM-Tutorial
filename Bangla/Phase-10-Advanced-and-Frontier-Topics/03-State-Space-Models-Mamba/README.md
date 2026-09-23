# State Space Models (Mamba)

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `03-State-Space-Models-Mamba`

## কেন এটি গুরুত্বপূর্ণ

কোর্সের এই পর্যায়ে এসে তুমি sequence-modeling trade-off-এর দুটি প্রান্তই কাছ থেকে দেখে ফেলেছ। [Phase 01 Lesson 3: RNNs, LSTMs and GRUs](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md)-এ তুমি এমন একটি model পেয়েছ যার প্রতি sequence-এ খরচ `O(T)` কিন্তু এর বিনিময়ে দিতে হয় কঠোরভাবে ক্রমিক (strictly sequential) recurrence (`h_t`-কে `h_{t-1}`-এর আগে হিসাব করা যায় না) এবং — LSTM/GRU-র gating সত্ত্বেও — দীর্ঘ দূরত্বে gradient এবং তথ্য ক্ষয় হতে থাকে। [Phase 01 Lesson 5: Introduction to Transformers](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md)-এ তুমি বিপরীত trade-off পেয়েছ: self-attention-এর যেকোনো দুটি token-এর মধ্যে path length হুবহু 1 (দূরত্বে signal-ম্লান হওয়ার কোনো সমস্যাই নেই) এবং প্রশিক্ষণের সময় sequence জুড়ে সম্পূর্ণ parallel, কিন্তু compute ও memory দুই-ই `O(T^2)` খরচ হয় — আর এ কারণেই long-context inference-এর খরচ বেশি।

State Space Model — এবং বিশেষ করে Mamba — একটি গুরুতর প্রশ্ন তোলার প্রচেষ্টা: **একই model-এ কি আমরা RNN-এর linear-time inference এবং Transformer-এর parallel trainability — দুটোই পেতে পারি?** এই প্রশ্নের চমকপ্রদ গাণিতিক উত্তর, যা S4 ধারার কাজে আবিষ্কৃত এবং Mamba-তে আরও তীক্ষ্ণ হয়েছে, তা হলো "বেশিরভাগ ক্ষেত্রেই হ্যাঁ।" এই lesson-টি সেই উত্তরটিকে প্রথম নীতি থেকে (first principles) গড়ে তোলে: একটি ধ্রুপদী linear state-space model, convolution কৌশল যা একে parallelizable করে, এবং "selectivity" ধারণা যা একে একটি নির্দিষ্ট linear filter থেকে এমন কিছুতে রূপান্তরিত করে যা আসলেই বিষয়বস্তু নিয়ে যুক্তি করতে পারে — যেমনটি LSTM-এ তুমি ইতিমধ্যেই gate-দের সাথে দেখা করেছ। এখান থেকে কোর্সটি এগোয় [Model Merging and Editing](../04-Model-Merging-and-Editing/README.md)-এর দিকে; পূর্ববর্তী lesson ছিল [Mixture of Experts, Advanced](../02-Mixture-of-Experts-Advanced/README.md), যেটি এখানে আলোচিত *sequence-length* trade-off-এর বদলে *parameter-count-vs-compute* trade-off-কে আক্রমণ করেছিল — গবেষণার এই দুটি ধারা পরস্পরের পরিপূরক, আর production মডেলগুলো ক্রমবর্ধমানভাবে দুটোকেই একত্রিত করছে।

## এই lesson-এ কী কী আচ্ছাদিত হবে

- trade-off-টি পুনর্বিবৃত্তি: attention `O(T^2)` কিন্তু parallel, RNN `O(T)` কিন্তু sequential এবং gradient-সীমাবদ্ধ — "উভয় জগতের সেরা" বলতে আসলে কী বোঝায়
- ধ্রুপদী linear state-space model: একটি Kalman filter-এর মতো একই গাণিতিক বস্তু, নির্দিষ্ট `A`, `B`, `C` matrix-সহ
- মূল duality যা S4-স্টাইল model-কে কাজ করায়: একটি linear recurrence-কে একটি নির্দিষ্ট kernel-এর বিপরীতে একটি বৈশ্বিক (global) convolution হিসেবে গণনা করা যায় — প্রশিক্ষণে parallel, চালাতে sequential
- কেন শুধু এই duality যথেষ্ট নয়: fixed matrix-গুলো content-based reasoning করতে পারে না
- Mamba-র selective state space: `A`, `B`, `C`-কে বর্তমান input-এর function বানানো
- কেন selectivity সাধারণ convolution কৌশলটি ভেঙে দেয়, আর hardware-aware parallel scan কীভাবে parallel-ism ফিরিয়ে আনে
- এতে sequence-mixing primitive-গুলোর পরিসর (landscape) কোথায় দাঁড়ায়

## 1. "উভয় জগতের সেরা" দেখতে আসলে কেমন হতো?

দুটি চরম প্রান্তকে পাশাপাশি রাখো:

```
Self-attention:  O(T^2) compute/memory,  fully parallel across positions,  path length 1 (no decay with distance)
Vanilla RNN:     O(T)   compute/memory,  strictly sequential,               path length T (vanishing gradients)
```

যে model attention-এর মতো প্রশিক্ষণ নেবে (এক ধাক্কায়, পুরো sequence জুড়ে parallel, GPU-বান্ধব) কিন্তু RNN-এর মতো চলবে (constant-size state, প্রতি নতুন token-এ `O(1)` কাজ, মোট `O(T)`) — তা দীর্ঘ sequence-এর জন্য অত্যন্ত আকর্ষণীয় হতো: attention-এর memory ও compute-ই তখন বাধা (bottleneck) হয়ে দাঁড়ায় যখন `T` বড় হয় (দীর্ঘ দলিল, audio, DNA, agent trajectory), আর সেখানেই তুমি সবচেয়ে বেশি RNN-শৈলীর খরচ-বক্ররেখা চাইবে। State Space Model-গুলো একটি বিশেষ গাণিতিক সত্যের চারপাশে গড়ে ওঠে যা *linear* recurrence-এর জন্য এই সংমিশ্রণটিকে সম্ভব করে — এমন কিছু যা কোনো nonlinear RNN (vanilla RNN, LSTM, GRU) কাজে লাগাতে পারে না, কারণ তাদের recurrence-তে প্রতিটি ধাপে একটি nonlinearity (`tanh`, gate) প্রয়োগ করা হয়, যার কোনো সমতুল্য closed-form convolution নেই।

## 2. ধ্রুপদী linear state space model

একটি recurrent network-কে তার সবচেয়ে মৌলিক linear রূপে ছোট করলে পাওয়া যায় একটি **state space model (SSM)** — ঠিক সেই একই গাণিতিক বস্তু, যা control তাত্ত্বিক ও পরিসংখ্যানবিদরা কয়েক দশক ধরে চলমান বস্তু ট্র্যাক করা একটি Kalman filter-এর মতো সিস্টেম বর্ণনা করতে ব্যবহার করে আসছেন:

```
h_t = A h_{t-1} + B x_t      # state update
y_t = C h_t                  # output/readout
```

- `h_t` হলো নির্দিষ্ট মাপ `N`-এর একটি hidden state vector (যাকে "state dimension" বলা হয়) — SSM-এর স্মৃতি (memory)।
- `x_t` হলো সময় `t`-এ input, `y_t` হলো সময় `t`-এ output।
- `A` (`N x N`), `B` (`N x input_dim`), `C` (`output_dim x N`) হলো **fixed** matrix — প্রতি timestep-এ একই তিনটি matrix, এবং গুরুত্বপূর্ণভাবে, এই ধ্রুপদী রূপে এরা input-এর উপর **মোটেও** নির্ভর করে না। এগুলো একবার শেখা হয় (অথবা ধ্রুপদী signal processing-এর মতো হাতে-নকশা করা হয়) এবং তারপর model-এর দেখা প্রতিটি input sequence-এর জন্য frozen থাকে।

এটি [Phase 01 Lesson 3](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#1-the-vanilla-rnn)-এর vanilla RNN-এর `h_t = tanh(W_xh x_t + W_hh h_{t-1})`-এর সাথে প্রায় হুবহু একই দেখায় — কিন্তু বিপুলভাবে গুরুত্বপূর্ণ পার্থক্য হলো **নিখোঁজ nonlinearity**। recurrence-টি খাঁটি linear হওয়ায় আমরা এটিকে শুধু সংখ্যাগতভাবে (numerically) নয়, বরং প্রতীকীভাবে (symbolically) unroll করতে পারি।

*(বাস্তব S4 model-গুলো আসলে অবিচ্ছিন্ন-সময়ের (continuous-time) differential equation হিসেবে সংজ্ঞায়িত — `h'(t) = A h(t) + B x(t)` — এবং তারপর একটি "discretization" ধাপের মাধ্যমে উপরের discrete recurrence-এ রূপান্তরিত হয়; যেমন zero-order hold, যা একটি continuous `A`-কে কোনো step size `Δ`-এর জন্য discrete `Ā = exp(ΔA)`-এ পরিণত করে। সেই বিবরণ S4 কীভাবে `A`-কে ভালোভাবে initialize করে (HiPPO matrix) তা বোঝার জন্য গুরুত্বপূর্ণ, কিন্তু পরের অংশের কৌশলটি দেখার জন্য দরকার নেই, তাই আমরা সরাসরি discrete `A`, `B`, `C` রূপ নিয়েই কাজ করব।)*

## 3. মূল কৌশল: একটি linear recurrence-ই একটি global convolution

recurrence-টিকে হাতে unroll করো, `h_0 = 0` থেকে শুরু করে:

```
h_1 = B x_1
h_2 = A B x_1 + B x_2
h_3 = A^2 B x_1 + A B x_2 + B x_3
...
h_t = sum_{k=1}^{t} A^{t-k} B x_k
```

readout `y_t = C h_t` প্রয়োগ করো:

```
y_t = sum_{k=1}^{t} (C A^{t-k} B) x_k
```

সেই সহগ `C A^{t-k} B`-টিকে ভালোভাবে লক্ষ করো: এটি **শুধুমাত্র lag `t - k`-এর উপর** নির্ভর করে, কখনোই পৃথকভাবে `t` এবং `k`-এর উপর নয়। ঠিক সেই lag-সূচিবদ্ধ সহগগুলো দিয়ে একটি kernel সংজ্ঞায়িত করো:

```
K = ( C B,  C A B,  C A^2 B,  ...,  C A^{T-1} B )        # one scalar/vector per lag i = 0 .. T-1
```

তাহলে সম্পূর্ণ output sequence-টি হলো এই নির্দিষ্ট kernel-এর বিপরীতে input-এর একটি একক **causal convolution**:

```
y_t = sum_{k=1}^{t} K_{t-k} x_k             i.e.        y = x * K
```

এটিই S4 (Gu, Goel, Re 2021)-এর পেছনের সম্পূর্ণ কৌশল। এটি বলে, *হুবহু একই সংখ্যাগুলো* (`y_1 ... y_T`) দুটি সম্পূর্ণ ভিন্ন উপায়ে তৈরি করা যায়:

1. **Sequentially**, একবারে একটি ধাপ: `h_t = A h_{t-1} + B x_t`, মোট কাজ `O(T)`, কিন্তু প্রতিটি ধাপ আগের ধাপের জন্য অপেক্ষা করে।
2. **In parallel**, একটি convolution হিসেবে: kernel `K`-টিকে একবার তৈরি করো (`A`-এর `T` সংখ্যক matrix power), তারপর পুরো input-টিকে এক ধাক্কায় এর বিপরীতে convolve করো — কোনো ধাপকে অন্য কোনো ধাপের জন্য অপেক্ষা করতে হয় না, তাই এটিকে attention-এর মতোই প্রশিক্ষণ দেওয়া যায়, পুরো sequence জুড়ে parallel, এমনকি FFT-ভিত্তিক convolution দিয়ে ত্বরান্বিত (accelerate) করা যায়।

`example.py` একটি ছোট fixed-`A/B/C` সিস্টেম তৈরি করে এবং এর output *দুই* উপায়েই হিসাব করে, তারপর যাচাই করে দুটি উত্তর floating-point precision পর্যন্ত মিলে কিনা — এই মিলটিই হলো সেই duality-টি বাস্তবে রূপ নেওয়া।

এটি সত্যিই "উভয় জগতের সেরা", এমন অর্থে যা attention বা vanilla RNN কোনোটিই দাবি করতে পারে না: convolution রূপে প্রশিক্ষণ দাও (attention-এর মতোই parallel, যদিও FFT-সহ `O(T log T)` বা সরলভাবে `O(T^2)` — যেখানে attention-এর `O(T^2)`-এর প্রতি-জোড়া খরচ অনেক বেশি), তারপর recurrence রূপে deploy করো (প্রতি নতুন token-এ `O(1)` memory ও compute, হুবহু RNN-এর মতো, সাথে বহন করার মতো কোনো `O(T^2)` attention cache নেই)। Attention parallel কিন্তু quadratic; vanilla RNN linear-time কিন্তু শুধু sequential এবং gradient-সীমাবদ্ধ; এই রূপের একটি linear SSM linear-time **ও** parallelizable, কারণ linearity-ই recurrence-কে closed form-এ convolution হিসেবে নতুন করে প্রকাশ করতে দেয়।

## 4. ধরা: fixed matrix-গুলো selective হতে পারে না

একটি fixed convolution kernel, কাঠামোগতভাবে, একটি **linear time-invariant (LTI) filter** — input-এর ভেতরে আসলে কী আছে তা নির্বিশেষে একই filter-আকৃতি প্রয়োগ করা হয়। এটি একটি বাস্তব সীমাবদ্ধতা: এর অর্থ model-টি *এইমাত্র-দেখা token-টি কী* তার ভিত্তিতে সিদ্ধান্ত নিতে পারে না যে হঠাৎ কোনো গুরুত্বপূর্ণ জিনিস মনে রাখবে বা অপ্রাসঙ্গিক কিছু বর্জন করবে। একে [Phase 01 Lesson 3 §4](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#4-lstms-gates-and-a-protected-cell-state)-এর LSTM-এর forget gate `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`-এর সাথে তুলনা করো: এটি বর্তমান input থেকে স্পষ্টভাবে হিসাব করে অতীতের কতটুকু রাখতে হবে। একটি সাধারণ S4-স্টাইল SSM-এর তেমন কোনো সমতুল্য নেই — `A`, `B`, `C` বিষয়বস্তু নির্বিশেষে একই থাকে, তাই এটি gated RNN-এর মতো "এই token-টি হুবহু কপি করো," "ফিলার token-গুলো উপেক্ষা করো," বা "ডিলিমিটারে memory রিসেট করো" বাস্তবায়ন করতে পারে না। বাস্তবে এর ফলে vanilla S4 কাঁচা দীর্ঘ-পাল্লার signal বিস্তারে অসাধারণ ছিল (এর খ্যাতি ছিল Long Range Arena benchmark) কিন্তু language modeling-এর মতো content-based reasoning দরকার হয় এমন কাজে attention-এর চেয়ে স্পষ্টভাবে দুর্বল ছিল।

## 5. Mamba-র মূল ধারণা: SSM-কে selective করা

Mamba (Gu & Dao, 2023) একই linear-recurrence কঙ্কাল রাখে কিন্তু `A`, `B`, `C`-কে — বা Mamba-র নির্দিষ্ট parameterization-এ, discretization step size `Δ`-কে সাথে `B` এবং `C`-কে — **বর্তমান input token-এর function** বানিয়ে দেয়, যা ছোট learned linear projection দিয়ে হিসাব করা হয়:

```
B_t = Linear_B(x_t)          # input-dependent, recomputed at every timestep
C_t = Linear_C(x_t)          # input-dependent, recomputed at every timestep
Δ_t = softplus(Linear_Δ(x_t))  # input-dependent step size -> controls the effective A_t = exp(Δ_t A)

h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

এটি ঠিক সেই একই পদক্ষেপ যা LSTM-কে কাজ করিয়েছিল: প্রতি ধাপে প্রয়োগ করা একটি fixed multiplicative factor (vanilla RNN-এর `W_hh`, বা সাধারণ SSM-এর fixed `A`) না রেখে, "কতটুকু রাখব বনাম কতটুকু মুছে দেব" সিদ্ধান্তটি এখন **বর্তমান input থেকে হিসাব করা হয়**, ফলে model-টি প্রতি token-এ বাছাই করতে পারে কী মনে রাখবে বা ভুলবে — একটি marker token "এটিকে দীর্ঘ সময় state-এ ধরে রাখো" ট্রিগার করতে পারে, আবার একটি filler token "এটিকে দ্রুত ম্লান হতে দাও" ট্রিগার করে। একটি বড় `Δ_t` আচরণ করে LSTM-এর forget gate 0-এর কাছাকাছি থাকার মতো (নতুন বিষয়বস্তু দিয়ে state-কে জোরালোভাবে আবার লিখে ফেলা); একটি ছোট `Δ_t` আচরণ করে forget gate 1-এর কাছাকাছি থাকার মতো (state প্রায় আপডেট না করা, কার্যকরভাবে token-টি এড়িয়ে যাওয়া)। `example.py` ঠিক এই "input থেকে হিসাব করা gate, ডিফল্টভাবে ধরে রাখার দিকে টিউন করা" ধারণাটি দিয়ে একটি ছোট selective SSM তৈরি করে — Phase 01-এর LSTM-এর forget-bias initialization-এর মতো একই কৌশল — এবং দেখায় যে sequence length বাড়ার সাথে সাথে এটি একটি প্রাথমিক "marker" input-এর প্রভাবকে চূড়ান্ত state-এর উপর একটি fixed-decay SSM-এর চেয়ে অনেক ভালোভাবে সংরক্ষণ করে।

## 6. selectivity-এর খরচ, এবং hardware-aware parallel scan

selectivity বিনামূল্যে নয়। যে মুহূর্তে `B_t`, `C_t` (এবং কার্যকর `A_t`) `t`-এর উপর নির্ভর করে, output sum-এ `x_k`-কে গুণ করা সহগটি হয় `C_t A_{k+1} A_{k+2} ... A_t B_k` — এটি শুধু lag `t - k` নয়, পরম অবস্থান `t` এবং `k`-এর উপরও নির্ভর করে। এটি সেই shift-invariance ভেঙে দেয় যার উপর Section 3-এর kernel নির্ভর করছিল: আর এমন একটি একক fixed kernel `K` থাকে না যার বিপরীতে পুরো sequence-কে convolve করা যাবে, কারণ "filter"-টি এখন প্রতিটি অবস্থানে ভিন্ন। Mamba content-based selection-এর বিনিময়ে সরল global-convolution প্রশিক্ষণ কৌশলটি হারায়।

Mamba যা ফিরিয়ে আনে তা হলো একটি **hardware-aware parallel scan**। recurrence `h_t = A_t h_{t-1} + B_t x_t`-টি এখনও তথাকথিত *associative* অপারেশন — দুটি পরপর linear update-এর সংমিশ্রণ নিজেই একই রূপের একটি linear update, এবং এটিই সেই বীজগাণিতিক গুণ যা `O(T)` sequential ধাপের বদলে `O(log T)` parallel গভীরতায় একটি চলমান যোগফল বা গুণফল হিসাব করতে দেয় (একটি "parallel scan," parallel prefix-sum-এর পেছনের একই আদিম অপারেশন)। Mamba এই scan-টিকে একটি কাস্টম GPU kernel হিসেবে বাস্তবায়ন করে যা (ছোট) প্রতি-ধাপ state-গুলোকে ধীর GPU memory (HBM)-তে বারবার লিখে ফেলার বদলে দ্রুত on-chip SRAM-এ রাখে — এটি একটি systems-level অপ্টিমাইজেশন, গণিতের কোনো পরিবর্তন নয়। ফলাফল: Mamba logarithmic-depth parallelism-সহ প্রশিক্ষণ নেয় (LTI convolution বা attention-এর single-matmul parallelism নয়, তবে একটি সরল `O(T)` sequential loop-এর চেয়ে অনেক ভালো) আর একইসাথে inference-এর সময় একটি সত্যিকারের `O(1)`-প্রতি-ধাপ, constant-memory recurrence হিসেবে চলতে সক্ষম থাকে — attention-এর মতো sequence length-এর সাথে বাড়তে থাকা কোনো KV cache নেই।

## 7. এটি পরিসরটিকে কোথায় রাখে

```
                      train-time parallelism        inference cost         content-based selection
Self-attention        full (one matmul)              O(T) per new token,    yes (softmax over all past
                                                       growing KV cache        keys/values, every step)
Vanilla RNN/LSTM/GRU   none (strictly sequential)     O(1) per new token,    yes (LSTM/GRU gates)
                                                       constant memory
Linear SSM (S4)        full (global convolution)      O(1) per new token,    no (fixed A, B, C)
                                                       constant memory
Selective SSM (Mamba)  parallel scan, O(log T) depth  O(1) per new token,    yes (input-dependent A, B, C)
                                                       constant memory
```

কোনো বিকল্পই প্রতিটি অক্ষে সবার উপরে নয় — এটি একটি প্রকৃত নকশা-পরিসর (design space), সমাধান হয়ে যাওয়া সমস্যা নয়। Mamba এবং এর উত্তরসূরিরা (Mamba-2, Jamba-এর মতো hybrid Mamba/attention architecture) একটি সক্রিয় গবেষণা ক্ষেত্র, সুনির্দিষ্টভাবে কারণ selectivity, hardware দক্ষতা এবং গুণগত মান — তিনটিই একসাথে পাওয়া কঠিন, আর ভিন্ন downstream কাজ এই trade-off-গুলোর উপর ভিন্নভাবে চাপ দেয়।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. trade-off-এর পুনরালোচনা: attention `O(T^2)` কিন্তু parallel, RNN `O(T)` কিন্তু sequential ও gradient-সীমাবদ্ধ
2. ধ্রুপদী linear SSM-এর পরিচয় (`h_t = Ah_{t-1} + Bx_t`, `y_t = Ch_t`) এবং Kalman-filter পরিবারের সাদৃশ্যটি উল্লেখ করা
3. হাতে-হাতে recurrence-is-equal-convolution duality-টি বের করা, `h_t` unroll করে এবং শুধু-lag নির্ভরতাটি চিহ্নিত করা
4. `example.py` Part 1-এর walkthrough: sequential বনাম convolution output floating-point precision পর্যন্ত মিলছে
5. ধরাটি ব্যাখ্যা: fixed `A/B/C` content-based selection করতে পারে না, LSTM-এর gate-দের থেকে ভিন্ন
6. Mamba-র selectivity-র পরিচয়: `B`, `C`, `Δ` input-এর function হিসেবে; LSTM-এর forget gate-এর সাথে স্পষ্ট সংযোগ
7. কেন selectivity convolution কৌশলটি ভেঙে দেয় তা ব্যাখ্যা, এবং hardware-aware parallel scan কীভাবে দক্ষ প্রশিক্ষণ ফিরিয়ে আনে
8. `example.py` Part 2-এর walkthrough: sequence length বাড়ার সাথে selective বনাম fixed SSM-এর gradient ধারণ; পুনরালোচনা এবং Model Merging and Editing-এর পূর্বাভাস

## আরও পড়ার জন্য

- Gu, Goel, Re (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4)
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2)