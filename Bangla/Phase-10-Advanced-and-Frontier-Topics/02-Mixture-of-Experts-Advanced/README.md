# Mixture of Experts, Advanced

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `02-Mixture-of-Experts-Advanced`

## কেন এটি গুরুত্বপূর্ণ

[Phase 03 Lesson 4](../../Phase-03-LLM-Architectures-and-Types/04-Mixture-of-Experts/README.md)-এ মূল MoE ধারণাটি পরিচয় করানো হয়েছিল: একটি dense FFN-এর বদলে `E` সংখ্যক expert বসাও, প্রতিটি token-কে একটি ছোট router দিয়ে তার top-`k` expert-এর কাছে route করো, আর প্রতি token-এ আনুপাতিক কম্পিউট (compute) ব্যয় না বাড়িয়েই অনেক বেশি মোট parameter পেয়ে যাও। সেখানে মূল ব্যর্থতার ধরন (failure mode)-টিও দেখানো হয়েছিল — router collapse, যেখানে কোনো একটি expert-এর সামান্য প্রাথমিক সুবিধা ক্রমাগত বেড়ে সেই expert-ই বেশিরভাগ token শুষে নেয় — আর এর মানক সমাধানও, একটি auxiliary load-balancing loss, যা routing-কে uniform-এর দিকে ঠেলে দেয়।

এই প্রাথমিক চিত্রটুকু *কেন* MoE কাজ করে তা বোঝার জন্য যথেষ্ট, কিন্তু কঠোর latency এবং memory budget-সহ লক্ষ লক্ষ request পরিবেশনকারী production সিস্টেমে routing আসলে কীভাবে করা হয়, তা এতে বাদ থেকে যায়। [Multimodal LLM](../01-Multimodal-LLMs/README.md) নিয়ে পূর্ববর্তী lesson-কে অনুসরণ করে, এই lesson-টি routing-এর কার্যপ্রণালী নিয়ে আরও এক ধাপ গভীরে যায়: একটি সম্পূর্ণ ভিন্ন routing paradigm যেখানে load balancing নির্মাণগতভাবে (by construction) এড়িয়ে যাওয়া হয়, top-k routing-এর বাস্তব বাস্তবায়নে থাকা token-dropping আচরণ (যা Phase 03-এর toy example-কে মডেল করতে হয়নি), এবং fine-grained "অনেক ছোট expert + shared expert" ডিজাইন, যেখানে আধুনিক open MoE মডেলগুলো মিলিত হয়েছে। পরের lesson, [State Space Models (Mamba)](../03-State-Space-Models-Mamba/README.md), MoE-কে সম্পূর্ণ বাদ দিয়ে attention-এর একটি non-Transformer বিকল্প দেখে — এটি একটি স্মরণ করিয়ে দেওয়া যে sparsity-via-routing এবং sequence-mixing-without-attention হলো দুটি স্বাধীন অক্ষ (axis), যেগুলোর সাথে সাথে মানক Transformer পদ্ধতিকে ঠেলে এগিয়ে নেওয়া হচ্ছে।

## এই lesson-এ কী কী আচ্ছাদিত হবে

- top-k token-choice routing-এর একটি দ্রুত পুনরালোচনা এবং কেন এটির একটি auxiliary loss দরকার (Phase 03-এর দিকে নির্দেশ, নতুন করে বের করা নয়)
- Expert-Choice routing (Zhou et al., 2022): কে কাকে বাছাই করে তা উল্টে দেওয়া — token-রা expert বাছাই করার বদলে expert-রা token বাছাই করে
- কেন Expert-Choice *by construction* নিখুঁত load balance নিশ্চিত করে, আর সেটা পেতে এটি কী বিসর্জন দেয় (token dropping, multiple selection)
- সাধারণ top-k routing-এ capacity factor এবং token dropping: বাস্তব বাস্তবায়নগুলোর যে বাস্তবে-ঘটা অদক্ষতা সামলাতে হয়, যা মৌলিক lesson-এর toy example-এ কখনও ট্রিগার হয় না
- Fine-grained expert segmentation এবং shared expert (DeepSeekMoE-স্টাইল): কিছু বড় expert-এর বদলে অনেকগুলো ছোট expert, সাথে সবসময় চালু থাকা shared expert
- Expert-Choice routing-এর একটি scratch থেকে তৈরি PyTorch বাস্তবায়ন, Phase 03-এর হুবহু একই router-bias পরিস্থিতিতে token-choice-এর সাথে মুখোমুখি তুলনা, এবং একটি capacity-factor/token-dropping সিমুলেশন

## 1. পুনরালোচনা: top-k token-choice routing এবং এর collapse সমস্যা

Phase 03-এর routing কাঠামোতে প্রতিটি **token** হচ্ছে সক্রিয় পক্ষ: এটি প্রতিটি expert-এর বিপরীতে একটি score হিসাব করে, নিজের top-`k` বেছে নেয়, আর সেই expert-দের FFN-এ প্রক্রিয়াজাত হয়।

```
router_logits = x @ W_router                            # (E,) one score per expert, per token
router_probs  = softmax(router_logits)
top_k_experts, top_k_weights = top_k(router_probs, k)
output = Σ_{i in top_k_experts} top_k_weights[i] * Expert_i(x)
```

এই গঠনে এমন কিছুই নেই যা ঠিক করে দেয় একটি নির্দিষ্ট expert-এর কাছে কতগুলো token পৌঁছাবে — এটি একটি প্রশিক্ষণ প্রক্রিয়ার উদীয়মান (emergent) বৈশিষ্ট্য, এবং সেই প্রক্রিয়ার প্রতিটি কারণ এটি হওয়ার পক্ষে *অসম* (ভালোভাবে প্রশিক্ষিত expert-রা router-এর কাছে বেশি আকর্ষণীয় দেখায়, তাই তাদের বেশি বাছাই করা হয়, তাই তারা বেশি gradient update পায়, তাই তারা আরও ভালো হয়ে ওঠে)। [Phase 03 Lesson 4](../../Phase-03-LLM-Architectures-and-Types/04-Mixture-of-Experts/README.md#3-the-load-balancing-problem)-এ এই collapse গতিশীলতা এবং এর auxiliary-loss সমাধান বিস্তারিতভাবে আচ্ছাদিত — সেই উপাদান এখানে আবার বলা হচ্ছে না। এই lesson-এর জন্য গুরুত্বপূর্ণ হলো সমাধানের *আকৃতি*: এটি loss function-এ যোগ করা একটি নরম, পরিসংখ্যানগত ঠেলা (nudge), যা একটি hyperparameter (aux-loss weight) দিয়ে টিউন করা হয়, এবং যা শুধু batch-এর গড়ে balance উৎসাহিত করে। এটি কোনো পৃথক batch-এর জন্য balance নিশ্চিত করে না, আর aux-loss weight ভুল করা নিজেই একটি বাস্তব টিউনিং দুঃস্বপ্ন — খুব দুর্বল হলে collapse ফিরে আসে, খুব শক্তিশালী হলে router এমন routing সিদ্ধান্ত থেকে সরে যায় যেগুলো অন্যথায় task loss-কে সাহায্য করত।

Expert-Choice routing একটি ভিন্ন প্রশ্ন করে: যদি load balance কাঠামোগতভাবে (structurally) প্রয়োগ করা হতো, তাহলে কি আদৌ কোনো auxiliary loss-এর দরকার হতো না?

## 2. Expert-Choice routing: expert-রা token বাছাই করে

Zhou et al. (2022) routing সিদ্ধান্তটিকে উল্টে দেয়। প্রতিটি token তার top-`k` expert বেছে নেওয়ার বদলে **প্রতিটি expert batch থেকে তার top-`C` token বেছে নেয়**, যেখানে `C` (expert-টির *capacity*) হলো একটি নির্দিষ্ট সংখ্যা যা batch size থেকে আগেই হিসাব করা হয়।

আনুষ্ঠানিকভাবে, `T` সংখ্যক token এবং `E` সংখ্যক expert-এর একটি batch-এর জন্য, আগের মতোই একটি affinity matrix হিসাব করো — প্রতি (token, expert) জোড়ার জন্য একটি score — কিন্তু এটিকে *অন্য* অক্ষ বরাবর পড়ো:

```
A = X @ W_router                     # (T, E) affinity matrix, one row per token, one column per expert
G = softmax(A, dim=-1)               # per-token normalization, used only as the gate value later

for expert e in 1..E:
    scores_e = A[:, e]                       # column e: every token's affinity to expert e
    top_C_tokens = top_k(scores_e, C)         # THIS expert's top-C tokens, by score
    output[top_C_tokens] += G[top_C_tokens, e] * Expert_e(x[top_C_tokens])
```

capacity `C`-টিকে লক্ষ্য গড় লোড হিসেবে আগে থেকেই নির্ধারণ করা হয়, সাধারণত

```
C = (T * k) / E
```

যেখানে `k` হলো প্রতিটি token শেষ পর্যন্ত যতগুলো expert-এর সাথে মিলে যাওয়ার কথা তার গড় সংখ্যা (token-choice-এর top-`k`-এর অনুরূপ; `k=1` বা `k=2` সাধারণ পছন্দ)। তারপর প্রতিটি expert `A`-এর নিজের কলাম থেকে *affinity score অনুযায়ী* ঠিক তার top-`C` token পড়ে নেয় — এটি একটি per-column top-`C`, per-row top-`k` নয়।

**কেন এটি by construction নিখুঁত load balance নিশ্চিত করে:** প্রতিটি expert সবসময় ঠিক `C` সংখ্যক token বাছাই করে, এখানেই শেষ। প্রশিক্ষণ-গতিশীলতার উপর কোনো নির্ভরতা নেই, সংশোধন করার মতো কোনো rich-get-richer ফিডব্যাক লুপ নেই, কারণ একজন expert-এর কাছে অন্য expert-এর চেয়ে বেশি token পৌঁছানোর কোনো উপায়ই নেই — top-`C` অপারেশনটি সংজ্ঞা অনুযায়ী এটিকে ঠিক `C`-তেই সীমাবদ্ধ রাখে। এমনকি একটি gradient step-এর আগে, random initialization-এও এটি সত্য: এখানে balance কোনো প্রশিক্ষণের *ফলাফল* নয়, এটি routing অপারেশনটির নিজেই একটি কাঠামোগত *অপরিবর্তনীয়তা* (invariant)। এই কারণেই Expert-Choice-এর কোনো auxiliary load-balancing loss-ই লাগে না।

**এই নিশ্চয়তা পেতে যেটা দিতে হয় (খরচ):**

- **Token dropping।** Expert-রা একে অপরের থেকে স্বাধীনভাবে token বাছাই করায়, একটি token কোনো expert-এর top-`C` তালিকাতেই না-ও থাকতে পারে — প্রতিটি expert তার বদলে `C` সংখ্যক অন্য token-কে প্রাধান্য দিয়েছে। সেই token এই layer-এ কোনো expert computation-ই পায় না (যদি পারিপার্শ্বিক architecture একটি residual connection সরবরাহ করে, তবে এটি শুধু সেই residual connection দিয়েই চলে যায়)।
- **Multiple selection।** Symmetric ভাবে, একটি token একইসাথে *বেশ কয়েকটি* expert-এর top-`C` তালিকায় থাকতে পারে (expert 3 এবং expert 7 দুজনেই একই জনপ্রিয় token-টি চাইতে পারে, তা আটকায় এমন কিছুই নেই), তাই এটি একাধিক expert থেকে অবদান পায় — প্রতি-token "আমি কতজন expert পেলাম" গণনাটি আর একটি নির্দিষ্ট `k` থাকে না, বরং batch-জুড়ে শুধু `k`-এর একটি *গড়* থাকে।
- **Batch dependence।** top-`C` নির্বাচনটি *batch-এ তখনকার উপস্থিত অন্য token-গুলোর* সাপেক্ষে হিসাব করা হয় বলে, একই token-টি তার batch-এ আর কী আছে তার উপর নির্ভর করে ভিন্নভাবে routed হতে পারে — এটি এমন একটি বৈশিষ্ট্য যা token-choice routing-এর নেই (সেখানে একটি token-এর routing শুধু নিজের উপর নির্ভর করে)। এটি autoregressive generation-কে জটিল করে তোলে, যেখানে তুমি চাইবে একটি token-এর routing-টি তার পাশাপাশি কী decode হচ্ছে তার উপর নির্ভর না করে স্থিতিশীল থাকুক; আর এটি এমন একটি কারণ যে Expert-Choice-কে causal decoding-এর চেয়ে encoder/training-time পরিস্থিতিতে বেশি আলোচনা করা হয়।

Expert-Choice load balancing-এর জন্য *নরম, টিউনযোগ্য, অসম্পূর্ণ* সমাধান (aux loss)-টির বদলে একটি *কঠিন, সুনির্দিষ্ট, কাঠামোগত* সমাধান নেয় — এর বিনিময়ে একটি নতুন failure mode (dropped এবং multiply-served token) যুক্ত হয়, যা top-k routing-এর aux loss তৈরি করে না।

## 3. top-k routing-এ capacity factor এবং token dropping

Token-choice routing-এরও capacity সীমার নিজস্ব সংস্করণ আছে, আর শুধু একটি toy implementation দেখলে (যেমন Phase 03-এর `MoELayer`, যেখানে প্রতিটি নির্বাচিত expert-কে তার কাছে routed হওয়া প্রতিটি token প্রক্রিয়া করতে দেওয়া হয়, কোনো সীমা ছাড়াই) সেটি সহজেই চোখ এড়িয়ে যায়। বাস্তব distributed MoE বাস্তবায়ন এটি করতে পারে না: প্রতিটি expert সাধারণত একটি নির্দিষ্ট accelerator-এ থাকে, যেখানে নির্দিষ্ট মাপের একটি buffer বরাদ্দ করা থাকে যে একটিমাত্র forward pass-এ এটি কতগুলো token প্রক্রিয়া করতে পারবে, কারণ সেই buffer-এর মাপ *batch দেখার আগেই* ঠিক করতে হয় (training step-এর মাঝখানে matmul-এর আকৃতি dynamic ভাবে বদলানো যায় না, কারণ তাতে পুরো pipeline থেমে যায়)।

তাই প্রতিটি expert-কে একটি কঠিন **capacity** দেওয়া হয় — একটি নির্দিষ্ট batch-এ এটি সর্বাধিক কতগুলো token গ্রহণ করবে — যা এভাবে হিসাব করা হয়:

```
capacity = capacity_factor * (num_tokens / num_experts)          # top-1 routing
capacity = capacity_factor * (num_tokens / num_experts) * top_k  # general top-k routing
```

`num_tokens / num_experts` হলো *সম্পূর্ণ সমান* routing-এর অধীনে প্রতিটি expert যে লোড পেত; **capacity factor** (যেমন `1.0`, `1.25`, `2.0`) হলো সেই আদর্শ লোডের উপরে যোগ করা অতিরিক্ত জায়গা (slack), যাতে routing imbalance শোষণ করা যায় — এমন imbalance যা aux loss uniform-এর দিকে টানলেও অনিবার্যভাবে থেকেই যায়।

যখন একটি token এমন কোনো expert-এর কাছে routed হয় যেটি এই batch-এর জন্য *ইতিমধ্যেই নিজের capacity পূর্ণ করে ফেলেছে* (token-গুলোকে ক্রমে প্রক্রিয়া করা হয়, যেটি আগে পৌঁছায়), মানক বাস্তবায়নগুলো (তাদের মধ্যে Switch Transformer) সেই token-টিকে সেই expert-এর জন্য **drop** করে দেয়: এটি সেই expert থেকে কোনো computation-ই পায় না এবং residual stream দিয়েই layer-টি পেরিয়ে যায়, ঠিক যেমনটি ঘটত যদি এটিকে কোথাও routed-ই না করা হতো। এটি একটি বাস্তব, নীরব দক্ষতা-ক্ষতি — dropped token-গুলোর উপর capacity-র ভেতরে ঢুকে পড়া token-গুলোর চেয়ে কম model capacity প্রয়োগ হয়, যা সম্পূর্ণভাবে তাদের batch-এ তখন আর কী ছিল তার একটি উপজাত (artifact), token-টি নিজেই এর জন্য দায়ী নয়।

- **`capacity_factor = 1.0`** কোনো slack-ই দেয় না: সামান্য পরিমাণ routing imbalance-ও (যা বাস্তবে সবসময় থাকে, aux loss থাকুক বা না থাকুক) drop-কে বাধ্য করে।
- **capacity factor বাড়ানো** (`1.25`, `2.0`, ...) expert-দের imbalance শোষণের জায়গা দেয় token drop না করেই, সরাসরি বিনিময়ে নষ্ট compute এবং memory — প্রতিটি expert-এর buffer তার *সবচেয়ে খারাপ-পরিস্থিতির* (worst-case) লোডের জন্য মাপা হয়, তাই বেশিরভাগ expert বেশিরভাগ সময় আংশিক খালি থাকে, padding-এর উপর computation চালায়।

এটি একটি সত্যিকারের তিন-মুখী trade-off: capacity factor বড় মানে কম dropped token কিন্তু বেশি নষ্ট compute; capacity factor ছোট মানে কম অপচয় কিন্তু বেশি dropped token; আর aux-loss weight দুটোর সাথেই মিথস্ক্রিয়া করে, কারণ ভালোভাবে balanced routing-এর drop এড়াতে কম capacity slack দরকার। `example.py` §3 এটি সরাসরি সিমুলেট করে এবং কয়েকটি capacity factor-এ প্রকৃত dropped-token ভগ্নাংশ রিপোর্ট করে।

## 4. Fine-grained expert segmentation এবং shared expert

DeepSeekMoE (Dai et al., 2024) একটি ভিন্ন অক্ষে এগোয়: সামান্য কয়েকটি *বড়* expert-এর (যেমন 8 জন expert, প্রত্যেকে মানক FFN-এর মাপের) বদলে এটি **অনেক বেশি, ছোট** expert ব্যবহার করে — প্রতিটি সম্ভাব্য expert-এর parameter-গুলোকে কয়েকটি সূক্ষ্ম (finer-grained) টুকরোয় ভাগ করা হয় (যেমন 8টি বড় expert-কে 64টি ছোটে উপবিভক্ত করা) এবং সেই সূক্ষ্ম সেটের উপর আনুপাতিকভাবে বড় `k` দিয়ে top-`k` routing করা হয়। অন্তর্দৃষ্টিটি হলো: একটি একক বড় expert, যাকে একটি একীভূত (monolithic) ইউনিট হিসেবে route করা হয়, তার কাছে পৌঁছানো প্রতিটি token-এর জন্য একটি "সবেতেই ওস্তাদ" (jack of all trades) হতে বাধ্য, অন্যদিকে অনেকগুলো ছোট expert router-কে প্রতিটি token-এর জন্য আরও সুনির্দিষ্ট, আরও বিশেষায়িত সমন্বয় রচনা করতে দেয় — নিকটতম-মানানসই generalist বেছে নেওয়ার বদলে সংকীর্ণ specialist-দের ঠিক-সঠিক মিশ্রণটি বাছাই করার কাছাকাছি।

fine-grained routed expert-গুলোর উপরে DeepSeekMoE অল্প কয়েকটি **shared expert** যোগ করে: এমন expert যাদের *আদৌ* route করা হয় না — প্রতিটি token-ই যে কোনো routed expert-এর পাশাপাশি শর্তহীনভাবে এদের সবকটির মধ্য দিয়ে যায়। ধারণাটি হলো, shared expert-রা সেই সাধারণ, জেনেরিক computation-গুলো শুষে নেবে যা মূলত প্রতিটি token-এরই দরকার (general-purpose রূপান্তর, যা অন্যথায় অনেকগুলো ভিন্ন routed expert-কে অপ্রয়োজনীয়ভাবে আবার-আবার শিখতে হতো), ফলে routed expert-রা আসলে token-নির্দিষ্ট বিষয়ে বিশেষজ্ঞ হওয়ার অবকাশ পায় — প্রতিবেশীদের মতো একই জেনেরিক রূপান্তর নতুন করে বের করার পেছনে capacity খরচ না করে। এটিই বড় একটি কারণ যে DeepSeek-V2/V3 তাদের বিশাল *মোট* parameter সংখ্যার সাপেক্ষে প্রতি token-এ *সক্রিয়* (active) parameter-এর কম ভগ্নাংশ দিয়েও শক্তিশালী গুণগত মান অর্জন করে।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. এক নিশ্বাসে পুনরালোচনা: token-choice top-k routing এবং যে collapse সমস্যাটিকে লড়তে এটির একটি aux loss দরকার (Phase 03-এর দিকে নির্দেশ)
2. উল্টোটা — "যদি expert-রাই token বাছাই করত?" — Expert-Choice routing-এর পরিচয়
3. affinity-matrix গঠনের মধ্য দিয়ে হাঁটা: আগের মতোই matrix, অন্য অক্ষ বরাবর পড়া
4. কেন নিখুঁত load balance বিনামূল্যে এসে যায়, এমনকি initialization-এও, কোথাও কোনো aux loss ছাড়াই
5. সেই নিশ্চয়তার দাম: dropped token এবং multiply-served token, আর কেন Expert-Choice causal decoding-এর চেয়ে training/encoder-এর জন্য বেশি উপযুক্ত
6. সাধারণ top-k routing-এ capacity factor: batch দেখার আগেই ঠিক করা একটি কঠিন buffer size, আর তার ফলে সৃষ্ট token dropping
7. Fine-grained segmentation এবং shared expert, DeepSeekMoE-শৈলীতে — অনেকগুলো ছোট specialist আর সবসময়-চালু generalist
8. `example.py`-এর walkthrough — scratch থেকে তৈরি একটি Expert-Choice layer, Phase 03-এর একই router-bias পরিস্থিতিতে token-choice-এর বিরুদ্ধে মুখোমুখি, আর capacity factor জুড়ে মাপা প্রকৃত token-dropping হার

## আরও পড়ার জন্য

- Zhou et al. (2022), *Mixture-of-Experts with Expert Choice Routing*
- Fedus, Zoph, Shazeer (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (একটি বাস্তব, ব্যাপকভাবে স্থাপিত বাস্তবায়নে capacity factor এবং token dropping)
- Dai et al. (2024), *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (fine-grained expert segmentation এবং shared expert)