# Generation এবং Decoding কৌশল

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `07-Generation-and-Decoding-Strategies`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর এখন পর্যন্ত প্রতিটি lesson নিঃশব্দে একটি ধাপ এড়িয়ে গেছে। [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) পুরো একটি lesson খরচ করে Key/Value tensors cache করার পেছনে যাতে decode সস্তা হয়, এবং এর speculative-decoding rejection-sampling নিয়ম (["কেন এটি নির্ভুল, এবং কেন এটি দ্রুততর"](../03-KV-Cache-and-Speculative-Decoding/README.md#6-why-this-is-exact-and-why-its-faster)) এই অনুমানের উপর দাঁড়িয়ে যে target model প্রতিটি ধাপে একটি প্রকৃত *sampling distribution* তৈরি করে, একটি একক deterministic token নয় — কিন্তু lesson-টি কখনো বলে না সেই distribution কী, এর আকৃতি কীভাবে নির্ধারিত হয়, বা এর মধ্য থেকে একটি token আসলে কীভাবে বের করে আনা হয়। [Lesson 4](../04-Serving-Frameworks/README.md) হাজার হাজার সমকালীন generation request schedule ও batch করে, কিন্তু কখনো জিজ্ঞেস করে না কোন সিদ্ধান্ত প্রতিটি request শেষ করে। [Phase 02 Lesson 6-এর mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#4-autoregressive-generation) — যে model-এর উপর এই প্রতিটি lesson নির্মিত — ইতিমধ্যে `softmax(logits / temperature)` গণনা করে এবং একবার `torch.multinomial` call করে, `0.8`-এর একটি স্থির temperature-এ, তারপর এগিয়ে যায়, কখনো জিজ্ঞেস না করে *কেন* ঠিক ওই মান, বিকল্পগুলো কী, বা সীমান্ত ক্ষেত্রগুলোতে (`temperature -> 0`, অথবা কোনো temperature scaling-ই না থাকলে) কী ঘটে। "logits থেকে একটি token sample করো" — এটিকে যেখানেই দেখা গেছে সেখানেই একটি সমাধান-হয়ে-যাওয়া, এক-লাইনের implementation detail হিসেবে ধরা হয়েছে। এটি এক লাইন নয় — এটি একটি পুরো design space, যার দুই প্রান্তেই বাস্তব, পরিমাপযোগ্য ব্যর্থতার ধরন আছে (এক প্রান্তে deterministic degenerate loops, অন্য প্রান্তে অসংলগ্ন noise), এবং এই lesson হলো সেই হারানো টুকরো: প্রকৃত সিদ্ধান্ত-নিয়ম যা logits-এর একটি vector-কে পরবর্তী token-এ পরিণত করে, এই কোর্সে এখন পর্যন্ত ব্যবহৃত প্রতিটি generation loop-এ।

## এই lesson যা কভার করে

- Logits থেকে একটি probability distribution: softmax recap, এবং "sampling" আসলে কী বোঝায়
- Greedy decoding: সবসময় argmax নেওয়া, এবং এর প্রকৃত ব্যর্থতার ধরন (পুনরাবৃত্তিমূলক loops)
- Temperature: softmax-এর আগে logits rescale করা, এবং কীভাবে `T -> 0` আনুষ্ঠানিকভাবে greedy decoding ফিরিয়ে আনে
- Top-k sampling: sample করার আগে সবচেয়ে সম্ভাব্য k টি token-এ ছেঁটে ফেলা
- Top-p (nucleus) sampling: একটি স্থির সংখ্যার বদলে একটি cumulative-probability mass পর্যন্ত ছেঁটে ফেলা
- বাস্তব pipelines কীভাবে temperature, top-k, top-p ও sampling একত্রিত করে, এবং কেন ক্রমটি গুরুত্বপূর্ণ
- Repetition penalty এবং frequency/presence penalties: greedy-এর degenerate-loop ব্যর্থতাকে সরাসরি patch করা
- Beam search: cumulative log-probability-এর উপর একটি deterministic *search*, আদৌ sampling নয়, এবং কেন এটি open-ended chat-এর জন্য খারাপ মানানসই
- Stopping criteria: EOS, max-new-tokens, এবং stop strings, এবং কীভাবে একটি শেষ হওয়া সিকোয়েন্সের KV-cache slot serving-এ ফিরে যায়

## 1. Recap: logits থেকে একটি probability distribution

এই কোর্সের প্রতিটি model একটি forward pass একইভাবে শেষ করে: একটি শেষ linear layer শেষ hidden state-কে raw scores-এর একটি vector-এ project করে, প্রতি vocabulary entry-তে একটি, যাদের বলা হয় **logits** ([Phase 02 Lesson 6 §2](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#2-the-full-model))। Logits normalize করা নয় — এগুলো যেকোনো বাস্তব সংখ্যা হতে পারে, ধনাত্মক বা ঋণাত্মক, এবং নিজেরা মিলে অর্থপূর্ণ কিছুতে যোগ হয় না। **Softmax** এগুলোকে vocabulary-এর উপর একটি প্রকৃত probability distribution-এ রূপান্তর করে:

```
P(token_i) = exp(logit_i) / sum_j( exp(logit_j) )
```

`P`-এর প্রতিটি মান ধনাত্মক এবং পুরো vector-এর যোগফল `1` — এটি vocabulary-এর উপর একটি খাঁটি probability distribution, এবং `example.py` পুরোটা জুড়ে এর উপরেই entropy ও top-token-probability সংখ্যা print করে। এই lesson-এর সবকিছু — greedy, temperature, top-k, top-p, beam search — এই একটি distribution-কে একটি প্রকৃত পরবর্তী token-এ রূপান্তর করার ভিন্ন ভিন্ন নিয়ম। অন্য কোনো input নেই: model প্রতিটি ধাপে generation pipeline-এর বাকি অংশকে কেবল সংখ্যার এই একটি vector-ই দেয়।

## 2. Greedy decoding: সবসময় argmax নাও

সবচেয়ে সরল সম্ভাব্য নিয়ম: প্রতিটি ধাপে, সর্বোচ্চ-probability-এর একক token-টি নাও এবং এগিয়ে যাও।

```
next_token = argmax(P)     # equivalently, argmax(logits) -- softmax is monotonic, doesn't change WHICH is largest
```

Greedy decoding সম্পূর্ণ **deterministic** — একই model ও একই prompt সবসময় হুবহু একই output তৈরি করে, loop-এর কোথাও কোনো randomness নেই। এই determinism-ই এটিকে একটি সুবিধাজনক baseline বানায় (এটিই [Lesson 3-এর `example.py`](../03-KV-Cache-and-Speculative-Decoding/example.py) ব্যবহার করে যাচাই করতে যে একটি naive ও একটি KV-cached generation loop byte-for-byte অভিন্ন token তৈরি করে — একটি random sampling নিয়ম হলে সেই তুলনা প্রতি run-এ অর্থহীন হয়ে যেত)। কিন্তু open-ended generation-এর জন্য, greedy decoding-এর একটি সুপরিচিত, বাস্তব ব্যর্থতার ধরন আছে: এটি **পুনরাবৃত্তিমূলক loops**-এ আটকে যায়। যদি model কখনো এমন একটি token-কে সর্বোচ্চ probability দেয় যা তাকে এমন একটি অবস্থায় ফিরিয়ে নিয়ে যায় যেখানে সে আগেই ছিল — একটি phrase একবার দেখা দেওয়ার পরে এবং model-এর নিজের attention এখন সেই phrase-কে অত্যন্ত predictable মনে করলে এটি একটি সাধারণ ফলাফল — তাহলে greedy decoding-এর পালানোর কোনো ব্যবস্থা নেই: একই input সবসময় একই "সবচেয়ে সম্ভাব্য" পরবর্তী token তৈরি করে, তাই loop চিরকাল পুনরাবৃত্ত হয়, অথবা যতক্ষণ না একটি max-length cutoff (§9) এটিকে থামতে বাধ্য করে। `example.py`-এর তৃতীয় demo একটি প্রকৃত প্রশিক্ষিত model-এ হুবহু এই loop তৈরি ও যাচাই করে, তারপর দেখায় একটি repetition penalty (§7) এটিকে ভেঙে দিচ্ছে।

## 3. Temperature: softmax-এর আগে distribution rescale করা

**Temperature** softmax প্রয়োগের আগে logits-কে `1/T` দিয়ে rescale করে:

```
P_T(token_i) = exp(logit_i / T) / sum_j( exp(logit_j / T) )
```

যেহেতু softmax exponential, `T < 1` temperature দিয়ে ভাগ করলে exponentiate করার আগে logits-এর মধ্যকার ব্যবধানগুলো *বেড়ে যায়*, distribution-এর mass-কে আরও বেশি করে ইতিমধ্যে-সর্বোচ্চ-probability token-গুলোর দিকে ঠেলে দেয় — distribution হয়ে ওঠে **আরও তীক্ষ্ণ**, আরও আত্মবিশ্বাসী, একটি একক spike-এর কাছাকাছি। `T > 1` temperature ঠিক উল্টোটা করে: এটি logits-এর মধ্যকার ব্যবধান *সংকুচিত* করে, probability mass-কে শীর্ষ প্রার্থীদের থেকে সরিয়ে বাকি সবকিছুর দিকে টেনে নেয় — distribution হয়ে ওঠে **আরও সমতল**, vocabulary-এর উপর uniform-এর কাছাকাছি। এটি ঠিক সেই knob যা [Phase 02 Lesson 6-এর `generate` method](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/example.py) প্রয়োগ করে (একটি স্থির `T=0.8`-এ), কখনো ব্যাখ্যা না করেই।

দুটি সীমা নির্ভুলভাবে বলার যোগ্য, কারণ এদের একটি সরাসরি §2-এর সাথে যুক্ত:

```
T -> 0+ :  P_T concentrates ALL probability mass on argmax(logits)  -->  sampling from P_T becomes
           IDENTICAL to greedy decoding (§2) -- greedy is temperature-sampling's zero-temperature limit.
T -> infinity :  P_T -> uniform distribution over the entire vocabulary  -->  every token equally likely,
           regardless of what the model actually predicted.
```

`example.py`-এর প্রথম demo প্রকৃত প্রশিক্ষিত-model logits-এর উপর এটি সরাসরি গণনা করে: একই logit vector, একটি নিম্ন temperature, `1` temperature, এবং একটি উচ্চ temperature-এ softmax করা, প্রতিটিতে entropy ও top-token probability print করা — উপরের গদ্য নয়, বরং sharpening/flattening প্রভাব দেখানো একটি ছোট, পরিমাপিত সংখ্যা।

## 4. Top-k sampling: একটি স্থির সংখ্যায় ছেঁটে ফেলা

Temperature (§3) *পুরো* distribution-এর আকার বদলায় কিন্তু কখনো কোনো token-কে বিবেচনা থেকে সরিয়ে দেয় না — খুব নিম্ন temperature-এও, খুব ছোট probability-র একটি token-এর sample হওয়ার কিছু অশূন্য সম্ভাবনা থাকে, এবং যথেষ্ট দীর্ঘ generation-এ বিরল দুর্ঘটনাগুলো জমতে থাকে। **Top-k sampling** (Fan, Lewis & Dauphin, 2018) দীর্ঘ লেজটি সরাসরি বাদ দিয়ে এটি ঠিক করে:

```
1. Sort tokens by probability, descending.
2. Keep only the top k tokens; set every other token's probability to 0.
3. Renormalize the kept probabilities so they sum to 1 again.
4. Sample from this truncated, renormalized distribution.
```

একটি ছোট `k` (ধরা যাক, `k=1`) top-k sampling-কে greedy decoding (§2)-এর সাথে অভিন্ন করে — sample করার জন্য কেবল একটি token বাকি থাকে। একটি বড় `k` (vocabulary size-এর সমান বা বেশি) এটিকে কোনো truncation ছাড়া সাধারণ temperature sampling (§3)-এর সাথে অভিন্ন করে — কিছুই বাদ পড়ে না। এই দুই প্রান্তের মাঝে, top-k নিশ্চিত করে যে model কখনো তার একেবারে-সর্বনিম্ন-probability token-গুলোর একটি sample করতে পারবে না, কোনো নির্দিষ্ট ধাপে distribution ক্ষণিকের জন্য যত সমতলই হোক না কেন, অথচ বাকি থাকা token-গুলোর মধ্যে প্রকৃত বৈচিত্র্যের সুযোগ রেখে দেয়। `example.py`-এর দ্বিতীয় demo এই truncate-renormalize-sample নিয়মটি সাধারণ tensor operations হিসেবে implement করে (mechanism লুকিয়ে ফেলে এমন কোনো library call ছাড়া) এবং ফলাফলস্বরূপ output diversity সরাসরি পরিমাপ করে।

## 5. Top-p (nucleus) sampling: একটি probability mass পর্যন্ত ছেঁটে ফেলা

Top-k-এর স্থির cutoff-এর একটি বাস্তব দুর্বলতা আছে: রাখার জন্য *সঠিক* token সংখ্যা স্থির নয় — এটি নির্ভর করে ওই নির্দিষ্ট ধাপে model-এর distribution কতটা তীক্ষ্ণ বা সমতল তার উপর। যখন model খুব আত্মবিশ্বাসী (একটি token আধিপত্য করে), তখন একটি ছোট স্থির `k`-ও অপ্রয়োজনে বেশ কয়েকটি প্রায়-শূন্য-probability token অন্তর্ভুক্ত করতে পারে; যখন model সত্যিই অনেক সম্ভাব্য ধারাবাহিকতার মধ্যে অনিশ্চিত, তখন একই স্থির `k` প্রকৃত, যুক্তিসঙ্গত প্রার্থীদের কেটে ফেলতে পারে যারা cutoff-এর ঠিক নিচে র‍্যাঙ্ক করেছে। **Top-p (nucleus) sampling** (Holtzman, Buys, Du, Forbes & Choi, 2020) স্থির সংখ্যাটিকে একটি স্থির cumulative-probability threshold দিয়ে প্রতিস্থাপন করে:

```
1. Sort tokens by probability, descending.
2. Walk down the sorted list, accumulating probability mass, until the running
   total first reaches or exceeds p (e.g. p = 0.9).
3. Keep exactly that prefix of tokens (the "nucleus"); set every other token's
   probability to 0; renormalize the kept probabilities so they sum to 1.
4. Sample from this truncated, renormalized distribution.
```

Top-k (§4)-এর সাথে মূল পার্থক্য: রাখা set-এর *আকার* আর স্থির নয় — এটি প্রতিটি আলাদা ধাপে distribution-এর আকৃতির সাথে স্বয়ংক্রিয়ভাবে খাপ খাইয়ে নেয়। যখন model তীক্ষ্ণভাবে আত্মবিশ্বাসী, nucleus-এ হয়তো কেবল একটি বা দুটি token থাকে (ছোট কার্যকর `k`, দ্রুত পৌঁছানো); যখন model সত্যিই অনেক সম্ভাব্য পরবর্তী token-এর মধ্যে ছড়িয়ে আছে, nucleus `p` threshold পার হওয়ার আগে কয়েক ডজন token পর্যন্ত প্রসারিত হতে পারে। Top-k এটি করতে পারে না: একটি স্থির `k` হয় আত্মবিশ্বাসী ধাপগুলোর জন্য খুব বড়, নয়তো অনিশ্চিত ধাপগুলোর জন্য খুব ছোট, কারণ ওই ধাপে probability mass আসলে কীভাবে বণ্টিত তা দেখার কোনো উপায় এর নেই। Holtzman et al.-এর paper-টিই সেই মূল empirical পর্যবেক্ষণের উৎস যার উপর এই পুরো lesson নিচের §8-এর জন্য নির্ভর করে: likelihood-এর বিশুদ্ধ greedy/beam-ধাঁচের maximization এমন text তৈরি করে যা মানুষের text-এর চেয়ে পরিমাপযোগ্যভাবে বেশি নিরস ও পুনরাবৃত্তিমূলক, আর ঠিক এই কারণেই open-ended generation-এর জন্য একটি maximization নিয়ম নয়, বরং top-p-এর মতো একটি *sampling*-ভিত্তিক নিয়মই standard। `example.py`-এর দ্বিতীয় demo top-p-কে হুবহু উপরের তিনটি ধাপ হিসেবে, সাধারণ tensor operations দিয়ে, top-k-এর পাশাপাশি, একই logits-এর উপর implement করে।

## 6. কৌশলগুলো একত্রিত করা: temperature, তারপর top-k, তারপর top-p, তারপর sample

বাস্তব generation pipelines (Hugging Face `transformers`, vLLM, এবং বেশিরভাগ বাণিজ্যিক chat API-এর `temperature`/`top_p`/`top_k` parameters) কদাচিৎ §3-5-এর কেবল একটিকে আলাদাভাবে প্রয়োগ করে — এগুলো তিনটিকেই একত্রিত করে, একটি নির্দিষ্ট ক্রমে প্রয়োগ করে:

```
1. logits <- logits / T                          # temperature: reshape the whole distribution first
2. logits <- top_k_filter(logits, k)              # truncate to the k highest logits, mask the rest to -inf
3. probs  <- top_p_filter(softmax(logits), p)     # softmax, then truncate to the smallest p-mass prefix
4. next_token <- sample(probs)                    # finally, sample one token from what's left
```

ক্রমটি গুরুত্বপূর্ণ কারণ প্রতিটি ধাপ আগেরটির *output*-এর উপর কাজ করে। Temperature-এর আগে top-k প্রয়োগ করলে truncation হতো *আকার-না-বদলানো* distribution-এর ভিত্তিতে — একটি বড় `k` এমন token রেখে দিতে পারে যেগুলোকে একটি নিম্ন temperature এমনিতেই নগণ্যভাবে অসম্ভাব্য করে তুলত, অথবা একটি ছোট `k` এমন token বাদ দিতে পারে যেগুলোকে একটি উচ্চ temperature বিশেষভাবে খেলায় রাখতে চাইছিল। Top-k-এর পরে top-p চালানো (এর বদলে নয়) ইচ্ছাকৃতও: top-k প্রথমে সবচেয়ে খারাপ ক্ষেত্রকে সীমাবদ্ধ করে (কখনো `k`-এর বেশি প্রার্থী টিকে থাকে না, top-p-এর cumulative sum-কে কতটা কাজ করতে হবে তা সীমিত করে এবং tail risk-এর একটি ঊর্ধ্বসীমা নিশ্চিত করে), আর top-p তারপর যা বাকি আছে তার প্রকৃত আকৃতির ভিত্তিতে সেই সীমাবদ্ধ set-কে adaptively আরও সংকুচিত করে। `k`-কে পূর্ণ vocabulary size-এ (অথবা `0`-তে, বেশিরভাগ library-র প্রচলন অনুযায়ী, অর্থাৎ "disabled") সেট করে কেবল `p`-এর উপর নির্ভর করা, বা উল্টোটা — দুটোই একই pipeline-এর সাধারণ বিশেষ ক্ষেত্র, ভিন্ন কোনো mechanism নয়।

## 7. Repetition penalty: greedy-এর degenerate-loop ব্যর্থতাকে সরাসরি patch করা

§2 greedy decoding-এর মূর্ত ব্যর্থতার ধরন দেখিয়েছে: একবার একটি phrase পুনরাবৃত্ত হলে, model-এর নিজের attention প্রায়ই সেটিকে *আবার* পুনরাবৃত্ত করাকেই সর্বোচ্চ-probability পদক্ষেপ বানিয়ে ফেলে, এবং একটি deterministic নিয়মের বেরোনোর কোনো পথ নেই। Sampling (§3-6) মাঝে মাঝে একটি non-top token বেছে নিয়ে সাহায্য করে, কিন্তু loop-এর কারণ হওয়া mechanism-কে বিশেষভাবে লক্ষ্য করে না — এটি একটি নির্দিষ্ট সমস্যার একটি সাধারণ সমাধান। একটি **repetition penalty** (Keskar et al., 2019, `CTRL`) loop-কে সরাসরি আক্রমণ করে, model-কে ইতিমধ্যে তৈরি করা token-গুলো পুনরায় বেছে নিতে নিরুৎসাহিত করে:

```
for token_id in set(already_generated_tokens):
    if logits[token_id] > 0:
        logits[token_id] /= penalty      # penalty > 1: shrink a positive logit toward 0
    else:
        logits[token_id] *= penalty      # penalty > 1: push a negative logit further negative
```

অসম multiply/divide (একটি সমতল বিয়োগের বদলে) penalty-র প্রভাবকে একটি logit-এর চিহ্ন বা মাত্রা নির্বিশেষে মোটামুটি আনুপাতিক রাখে। দুটি সম্পর্কিত, সরলতর রূপ একটি গুণগত threshold-এর বদলে একটি স্থির threshold সেট করে: একটি **frequency penalty** একটি token ইতিমধ্যে *কতবার* এসেছে তার সমানুপাতিক একটি পরিমাণ বিয়োগ করে (বারবার পুনরাবৃত্তি প্রতিবার আরও কঠোরভাবে penalize হয়), এবং একটি **presence penalty** যেকোনো token যা *আদৌ* এসেছে তার জন্য একটি একক সমতল পরিমাণ বিয়োগ করে, সংখ্যা নির্বিশেষে (কোনো কিছু অনেকবার পুনরায় ব্যবহার করার জন্য বাড়তি শাস্তি ছাড়াই পুনঃব্যবহার নিরুৎসাহিত করে)। তিনটিই temperature/top-k/top-p (§6)-এর *আগে* logits-এ প্রয়োগ করা হয়, যাতে pipeline-এর বাকি অংশ একটি ইতিমধ্যে-penalize-করা distribution-এর আকার বদলায় ও ছাঁটে। `example.py`-এর তৃতীয় demo হুবহু এই গুণগত penalty চালায় সেই একই toy model ও prompt-এ যা §2-এ একটি greedy loop-এ আটকে যায়, একই deterministic argmax নিয়মের অধীনে, এবং code-এ যাচাই করে যে loop ভেঙে যায়।

## 8. Beam search: একটি search, sample নয়

এখন পর্যন্ত প্রতিটি কৌশল — greedy, temperature, top-k, top-p — প্রতি ধাপে একটি সিদ্ধান্ত নেয় এবং কখনো পেছনে তাকায় না। **Beam search** এর বদলে প্রতিটি ধাপে একই সাথে `B` টি সর্বোচ্চ cumulative-log-probability সিকোয়েন্স ("beams") track করে:

```
1. Start with B copies of the prompt (or, at step 1, the single prompt expanded to
   its top-B next-token candidates).
2. At each step, for EACH of the B current beams, compute the model's next-token
   distribution and consider extending that beam by every candidate token.
3. Out of all B * vocab_size candidate extensions, keep only the B with the
   highest cumulative log-probability (sum of log P(token) over the whole
   sequence so far) -- discard the rest.
4. Repeat until every beam has produced an end-of-sequence token or hit the
   length budget; return the surviving beam with the highest cumulative
   log-probability.
```

এটি নির্ভুলভাবে বলা গুরুত্বপূর্ণ: beam search আদৌ **একটি sampling method নয়** — একটি স্থির model, prompt ও beam width `B` দেওয়া থাকলে, এটি সবসময় হুবহু একই output ফেরত দেয়, deterministically, ঠিক greedy decoding (§2)-এর মতো। একে বরং সেই একই objective-এর উপর একটি *search* হিসেবে বোঝা ভালো যা greedy decoding (স্থানীয়ভাবে) optimize করছে — model-এর অধীনে সর্বোচ্চ-probability সিকোয়েন্স — কেবল একটি প্রশস্ততর, কম অদূরদর্শী search: greedy হলো `B=1`-সহ beam search, এবং একটি প্রশস্ততর beam সিদ্ধান্ত নেওয়ার আগে উচ্চ-probability সিকোয়েন্সগুলোর space-এর আরও বেশি অংশ অন্বেষণ করে, যে কারণে beam search একই prompt ও model-এ greedy-এর *অন্তত সমান উচ্চ* cumulative log-probability-র একটি সিকোয়েন্স খুঁজে পেতে পারে (এবং `example.py`-এর চতুর্থ demo প্রকৃত সংখ্যা দিয়ে তা যাচাই করে)।

এই নিশ্চয়তাই ঠিক সেই কারণ যে জন্য beam search open-ended chat ও instruction-following-এর জন্য খারাপ মানানসই, যদিও এর নিজস্ব objective অনুযায়ী এটি "ভালো" (উচ্চতর-likelihood) সিকোয়েন্স খুঁজে পায়: Holtzman et al.-এর পর্যবেক্ষণ, যা ইতিমধ্যে §5-এ উদ্ধৃত, হলো sequence likelihood maximize করা — beam search ঠিক যা করে, greedy-এর চেয়ে আরও পুঙ্খানুপুঙ্খভাবে — open-ended text-এর জন্য *ভুল* objective। মানুষের লেখা text প্রতিটি ধাপে সর্বোচ্চ-likelihood ধারাবাহিকতা নয়; এটি বৈচিত্র্যময়, কখনো কখনো স্থানীয়ভাবে অপ্রত্যাশিত, এবং একক সবচেয়ে সম্ভাব্য সিকোয়েন্সের পেছনে beam search-এর নিরলস ছোটাছুটি লক্ষণীয়ভাবে নিরস, পুনরাবৃত্তিমূলক, গতানুগতিক text তৈরি করে — greedy-এর loops (§2)-এর মতোই একই গুণগত ব্যর্থতার ধরন, কেবল শনাক্ত করা কঠিন কারণ এটি একটি স্পষ্ট পুনরাবৃত্ত phrase-এর বদলে পুরো একটি সিকোয়েন্স জুড়ে ছড়িয়ে থাকে। Chat ও instruction-following model-গুলোর জন্য standard পছন্দ হিসেবে top-p sampling (§5)-এর empirical প্রেরণা ঠিক এটিই। Beam search এখনো এমন কাজের জন্য ভালো মানানসই, এবং এখনো ব্যাপকভাবে ব্যবহৃত, যাদের output space অনেক সংকীর্ণ এবং একটি বা কয়েকটি প্রকৃত "সঠিক" উত্তর থাকার কাছাকাছি — machine translation ও summarization হলো ক্লাসিক উদাহরণ — যেখানে একক সর্বোচ্চ-likelihood output-এর জন্য একটি পদ্ধতিগত search কাজটি আসলে যা চায় তার বেশি কাছাকাছি, শৈলীগত বৈচিত্র্য অন্বেষণের চেয়ে।

## 9. Stopping criteria: EOS, length budget, এবং stop strings

প্রতিটি generation loop-এর *কখন থামতে হবে* তার একটি নিয়ম দরকার, কোন per-step decoding কৌশল (§2-8) ব্যবহৃত হচ্ছে তা নির্বিশেষে:

- **EOS (end-of-sequence) token**: model নিজেই প্রশিক্ষিত হয়েছে একটি বিশেষ end-of-sequence token predict করতে যখন সে response-কে সম্পূর্ণ মনে করে; যে মুহূর্তে সেই token sample হয় বা argmax হিসেবে নির্বাচিত হয়, অন্য যেকোনো token-এর মতোই, generation থেমে যায়।
- **Max-new-tokens budget**: একটি একক request কতগুলো token তৈরি করতে পারবে তার একটি কঠোর ঊর্ধ্বসীমা, EOS কখনো আসুক বা না আসুক — সেই backstop যা নিশ্চিত করে যে একটি request (এবং, §2 অনুযায়ী, একটি greedy loop যা নিজে থেকে কখনো বেরোয় না) শেষ পর্যন্ত সমাপ্ত হয়।
- **Stop sequences / stop strings**: caller-প্রদত্ত literal strings-এর একটি তালিকা (যেমন একটি chat template-এ `"\n\nUser:"`), যেগুলো generated text-এ দেখা দেওয়ামাত্র response আগেভাগে শেষ করে দেয় — উপযোগী যখন একটি model-এর output format-এর একটি application-level terminator থাকে যা model-এর নিজস্ব EOS token নয়।

এটি serving-এর জন্য গুরুত্বপূর্ণ, কেবল বিচ্ছিন্নভাবে একটি একক request-এর জন্য নয়: এই তিনটি শর্তের যেকোনো একটি যে মুহূর্তে ঘটে, সেই সিকোয়েন্সের batch-এর slot — এবং এর জন্য সংরক্ষিত KV cache memory ([Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md#3-the-kv-cache-pay-for-each-tokens-kv-exactly-once)-এর per-sequence cache) — আর কোনো উপযোগী কাজ করছে না এবং অবশ্যই মুক্ত করতে হবে। [Lesson 4-এর continuous batching](../04-Serving-Frameworks/README.md#4-hugging-face-tgi-continuous-in-flight-batching) হলো ঠিক সেই serving-side mechanism যা একটি মুক্ত slot-কে অলস ফেলে না রেখে তাৎক্ষণিকভাবে পরবর্তী অপেক্ষমাণ request দিয়ে পূরণ করে — এখানে প্রতি সিকোয়েন্সে নেওয়া থামার সিদ্ধান্তটিই সেই ঘটনা যা ওই পুরো lesson-এর বিষয়বস্তু slot-management আচরণকে ট্রিগার করে।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. প্রেরণা — এই phase-এর এখন পর্যন্ত প্রতিটি lesson (এবং Phase 02-এর mini-GPT) নিঃশব্দে ধরে নিয়েছে যে "logits থেকে একটি token বেছে নাও" একটি সমাধান-হয়ে-যাওয়া ধাপ; আজ আর তা ধরে নেওয়া হচ্ছে না
2. Softmax recap: logits থেকে একটি প্রকৃত probability distribution, এবং মূর্তভাবে "sampling" মানে কী
3. Greedy decoding: সবসময় argmax, সম্পূর্ণ deterministic, এবং এর প্রকৃত repetitive-loop ব্যর্থতার ধরন
4. Temperature: logits-কে `1/T` দিয়ে rescale করা, sharpening বনাম flattening, এবং greedy decoding-এর সীমা হিসেবে `T -> 0`
5. Top-k sampling: একটি স্থির সংখ্যায় ছেঁটে ফেলা, renormalize, sample
6. Top-p (nucleus) sampling: এর বদলে একটি cumulative-probability mass-এ ছেঁটে ফেলা, এবং কেন এটি প্রতি ধাপে খাপ খায় যেখানে top-k পারে না
7. বাস্তব pipelines কীভাবে temperature -> top-k -> top-p -> sample একত্রিত করে, এবং কেন ক্রমটি যথেচ্ছ নয়
8. Repetition penalty এবং frequency/presence penalties: greedy-এর degenerate loops-এর একটি সরাসরি patch
9. Beam search: B টি beam জুড়ে cumulative-log-probability search, sampling নয়, এবং কেন এটি open-ended chat-এর জন্য ভুল হাতিয়ার কিন্তু translation/summarization-এর জন্য সঠিক
10. Stopping criteria: EOS, max-new-tokens, stop strings, এবং কীভাবে একটি মুক্ত slot continuous batching-কে খাওয়ায়
11. `example.py`-এর walkthrough — পরিমাপিত entropy/temperature সংখ্যা, top-k/top-p diversity, repetition penalty দিয়ে ভাঙা একটি প্রকৃত greedy loop, এবং greedy-এর উপর beam search-এর cumulative log-probability সুবিধা, সবই code-এ যাচাইকৃত
12. Recap + [Lesson 4: Serving Frameworks](../04-Serving-Frameworks/README.md)-এর দিকে ইঙ্গিত, যেখানে একটি থেমে যাওয়া সিকোয়েন্সের মুক্ত slot-ই ঠিক সেটি যা continuous batching পূরণ করে

## আরও পড়ুন

- Holtzman, Buys, Du, Forbes & Choi (2020), *The Curious Case of Neural Text Degeneration* (top-p/nucleus sampling উপস্থাপন করে এবং open-ended text-এর জন্য likelihood-maximizing decoding-এর বিরুদ্ধে empirical যুক্তি দেয়)
- Fan, Lewis & Dauphin (2018), *Hierarchical Neural Story Generation* (top-k sampling উপস্থাপন করে)
- Keskar, McCann, Varshney, Xiong & Socher (2019), *CTRL: A Conditional Transformer Language Model for Controllable Generation* (repetition penalty)
- Graves (2012), *Sequence Transduction with Recurrent Neural Networks*, এবং পরবর্তী অনেক encoder-decoder MT paper যেগুলো translation-এর standard decoding method হিসেবে beam search-কে জনপ্রিয় করেছে
