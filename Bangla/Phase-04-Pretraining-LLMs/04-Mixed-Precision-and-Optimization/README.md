# Mixed Precision and Optimization

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `04-Mixed-Precision-and-Optimization`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 3](../03-Distributed-Training-Basics/README.md) দেখিয়েছে একটি training step-এর byte ও FLOP *কোথায়* device-গুলোর মধ্যে ছড়ায়। এই পাঠ সেসব byte-এর আকার এবং প্রতিটি optimizer step-এর গুণমানকে সরাসরি আক্রমণ করে — আপনার কাছে কয়টি device আছে তার থেকে স্বাধীনভাবে। প্রতিটি বাস্তব pretraining run **mixed precision** ব্যবহার করে (সাধারণ fp32 নয় — এই কোর্সে এ পর্যন্ত লেখা প্রতিটি PyTorch example-এর ডিফল্ট) এবং **AdamW with a warmup+decay schedule** (সাধারণ SGD নয়)। দুটি পছন্দই implementation detail-এর মতো শোনায়, কিন্তু প্রতিটি নির্দিষ্ট, সুপরিচিত একটি failure mode-এর সরাসরি সমাধান — আর [Lesson 3-এর memory calculator-এ](../03-Distributed-Training-Basics/README.md#5-zero--fsdp-shard-dont-replicate) আপনি যে দুটি সংখ্যা হিসাব করেছেন (সেই `16` bytes/parameter চিত্র), তার সরাসরি উৎস এখানে ব্যাখ্যা করা precision-পছন্দগুলোই।

## এই পাঠে কী শেখানো হয়

- fp32 বনাম fp16 বনাম bf16: range/precision-এর ট্রেড-অফ, এবং কেন LLM training-এ bf16 জিতেছে
- Loss scaling: কেন fp16 gradient underflow হয়, এবং কীভাবে scaling তা ঠিক করে
- AdamW: momentum + adaptive learning rate, এবং সাধারণ Adam-এর উপরে weight-decay সংশোধন
- Learning-rate schedule: কেন warmup গুরুত্বপূর্ণ, এবং কেন তার পরে cosine/linear decay আসে

## 1. fp32 বনাম fp16 বনাম bf16

তিনটিই 4-byte/2-byte বাইনারি floating-point ফরম্যাট, শুধু পার্থক্য এই যে তাদের bits **exponent** (নিয়ন্ত্রণ করে *range* — কোনো সংখ্যা কত বড় বা ছোট হতে পারে) এবং **mantissa** (নিয়ন্ত্রণ করে *precision* — কতটি উল্লেখযোগ্য অঙ্ক উপস্থাপন করতে পারে) — এর মধ্যে কীভাবে ভাগ হয়:

```
fp32 (standard, "full precision"): 1 sign + 8 exponent bits + 23 mantissa bits
fp16 ("half precision"):           1 sign + 5 exponent bits + 10 mantissa bits
bf16 ("bfloat16"):                 1 sign + 8 exponent bits +  7 mantissa bits
```

- **fp32** হলো নিরাপদ ডিফল্ট, যা এই কোর্সের প্রতিটি আগের পাঠ ব্যবহার করেছে — সম্পূর্ণ range, সম্পূর্ণ precision, প্রতি মান 4 bytes।
- **fp16** memory অর্ধেক করে এবং fp16 tensor core-যুক্ত হার্ডওয়্যারে throughput মোটামুটি দ্বিগুণ করে, কিন্তু তার 5টি exponent bit তাকে fp32-এর চেয়ে *অনেক* সংকীর্ণ representable range দেয় (normal range-এ মোটামুটি `6e-5` থেকে `65504`) — এই range-এর বাইরের মান **infinity-তে overflow বা zero-তে underflow হয়**। Deep network-এর gradient নিয়মিতভাবেই খুব ছোট magnitude নেয়, যা প্রশিক্ষণের সময় fp16 underflow-কে একটি বাস্তব, ঘন ঘন সমস্যা করে তোলে (§2)।
- **bf16** fp32-এর *একই 8টি exponent bit* রাখে (fp32-এর একই range, তাই কোনো overflow/underflow বিস্ময় নেই) তবে তার বদলে precision বিসর্জন দেয়, মাত্র 7টি mantissa bit-এ নামিয়ে আনে। এই ট্রেড — fp32-এর একই range, fp16-এর চেয়ে কম precision — নিউরাল নেট train করার জন্য ঠিক সঠিকটি প্রমাণিত হয়েছে: overflow/underflow প্রশিক্ষণ run-কে একেবারে ভেঙে দেয়, অন্যদিকে হ্রাসপ্রাপ্ত precision বেশিরভাগই সামান্য numerical noise যোগ করে, যা gradient descent ইতিমধ্যেই ভালোভাবে সহ্য করে। এই কারণেই আধুনিক LLM train করার জন্য (যে হার্ডওয়্যার এটি সমর্থন করে — TPU শুরু থেকেই; NVIDIA GPU Ampere প্রজন্ম থেকে) bf16 (fp16 নয়) প্রমিত পছন্দ হয়ে উঠেছে।

## 2. Loss scaling: fp16-এর underflow সমস্যা, এবং সমাধান

একটি বাস্তব gradient মান ধরুন: `0.00003` (একটি বৃহৎ network-এর গভীরে একেবারে সাধারণ magnitude)। fp16-এর ক্ষুদ্রতম representable *normal* মান প্রায় `6e-5` — ফলে `0.00003` fp16-তে ঢালাই (cast) হওয়ার মুহূর্তেই সরাসরি **zero**-তে গোল হয়, আর gradient signal-এর সেই পুরো অংশটি নিঃশব্দে বিলীন হয়ে যায়। যথেষ্ট parameter-এর ক্ষেত্রে এটি ঘটলে প্রশিক্ষণ থেমে যায় বা আপাত কোনো কারণ ছাড়াই diverge করে।

**Loss scaling** গণিত মোটেও বদল না করেই এটি ঠিক করে: `.backward()` ডাকার *আগে* loss-টিকে একটি বড় ধ্রুবক `S` (যেমন `2^16 = 65536`) দিয়ে গুণ করুন। gradient যেহেতু নিজের উৎস loss-এর সাথে রৈখিকভাবে স্কেল হয়, backward pass-এর প্রতিটি gradient একই গুণক `S`-তে স্কেল পায়, পূর্বে-বিলুপ্ত হওয়া ছোট মানগুলোকে fp16-এর representable range-এর ভেতরে ফিরিয়ে আনে। Backprop-এর পরে, optimizer step-এর আগে gradient **unscale** করুন (`S` দিয়ে ভাগ), গাণিতিকভাবে সঠিক gradient magnitude ফিরে পান:

```
scaled_loss = loss * S
scaled_loss.backward()                  # gradients come out ~S times larger, now representable in fp16
unscaled_grad = fp16_grad.float() / S   # recover the true gradient before the optimizer step
```

`example.py` এটি সংখ্যাগতভাবে দেখায়: একটি ছোট gradient মান সরাসরি fp16-তে ঢালাই করলে তা ঠিক `0.0`-তে underflow হয়, অথচ আগে scale তারপর পরে unscale করলে মূল মান প্রায় হুবহু ফিরে আসে। (bf16-এর অনেক বড় range মানে loss scaling খুব কমই দরকার হয় — এর পক্ষে আরেকটি যুক্তি, এবং mixed-precision training recipe-কে একবার হার্ডওয়্যার সমর্থন করা শুরু করলে সহজ করার একটি কারণ।)

## 3. AdamW: decoupled weight decay

পূর্বের optimizer আলোচনা থেকে Adam-কে মনে করুন: এটি প্রতিটি parameter-এর gradient-এর চলমান exponential moving average (`m`, momentum/first-moment estimate) এবং প্রতিটি gradient-এর *বর্গ* magnitude-র average (`v`, second-moment estimate) রাখে, তারপর প্রতিটি parameter-কে তার momentum দিয়ে আপডেট করে, এর variance estimate-এর বর্গমূলের *বিপরীত* অনুপাতে স্কেল করে — ফলে প্রতিটি parameter নিজস্ব adaptive effective learning rate পায়:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)              # bias correction (প্রারম্ভিক ধাপগুলো 0-এর দিকে skewed থাকে)
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

সাধারণ "Adam + L2 regularization" weight decay বাস্তবায়ন করে gradient-এর সাথে `lambda * theta` সরাসরি যোগ করে, যা উপরের momentum/variance estimates-এ পৌঁছানোর আগেই — যার মানে decay পদটিও নিজে, gradient-এর মতোই, `v_hat` দিয়ে adaptively পুনরায়-স্কেল হয়। **AdamW** (Loshchilov & Hutter, 2019) এটিকে decouple করে: weight decay-কে adaptive-learning-rate মেশিনারি থেকে সম্পূর্ণ বাইরে, parameter-এর একটি পৃথক, সরাসরি সংকোচন হিসেবে প্রয়োগ করে:

```
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * theta_{t-1}
```

পার্থক্যটি সূক্ষ্ম কিন্তু পরিমাপযোগ্য: সাধারণ Adam+L2-এ, বড় historical gradient variance-যুক্ত (বড় `v_hat`) parameter-গুলো gradient-এর মতো একই `sqrt(v_hat)` দিয়ে decay পদ বিভক্ত হওয়ায় তাদের "উচিত"-এর তুলনায় *কম* কার্যকর weight decay পায়। AdamW-এর decoupled decay প্রতিটি parameter-এ gradient history থেকে স্বাধীনভাবে *একই* আনুপাতিক সংকোচন প্রয়োগ করে — যা অভিজ্ঞতাগতভাবে ভালো generalize করে, আর এজন্যই মূলত প্রতিটি আধুনিক LLM pretraining run সাধারণ Adam-এর বদলে AdamW ব্যবহার করে। `example.py` এই exact update rule স্ক্র্যাচ থেকে বাস্তবায়ন করে এবং `torch.optim.AdamW`-এর বিপরীতে যাচাই করে।

## 4. Learning-rate schedules: warmup তারপর decay

বাস্তব pretraining run কখনোই ধ্রুব learning rate ব্যবহার করে না। প্রমিত recipe-র দুটি ধাপ:

```
lr(step) = lr_max * (step / warmup_steps)                                              if step < warmup_steps
lr(step) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))                  otherwise
    where progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

- **Warmup** (প্রথম কয়েকশ/কয়েক হাজার ধাপে learning rate-কে ~0 থেকে `lr_max`-এ রৈখিকভাবে চড়ানো) গুরুত্বপূর্ণ, কারণ training-এর শুরুতে weights এলোমেলোভাবে initialized থাকে আর gradients বড় ও noisy হয় — সঙ্গে সঙ্গেই পূর্ণ-আকারের optimizer step নিলে মডেল এমন একটি খারাপ অঞ্চলে ঠেলে দিতে পারে, যেখান থেকে সে আর ফিরে আসে না। ছোট, ক্রমবর্ধমান learning rate মডেলের প্রারম্ভিক পরিসংখ্যানগুলোকে (যেমন Adam-এর নিজস্ব `m`/`v` চলমান estimate, যেগুলো zero থেকে শুরু হয় এবং নিজেরাও তখনো "warm up" হতে থাকে) পূর্ণ step size-কে বিশ্বাস করার আগে স্থির হতে দেয়।
- **Cosine decay** (বা, আরও সহজভাবে, linear decay) warmup-এর পরে বাকি প্রশিক্ষণ জুড়ে learning rate-কে মসৃণভাবে হ্রাস করে, একটি ছোট (প্রায়ই শূন্যের কাছাকাছি) চূড়ান্ত মানের দিকে অভিমুখী হয়ে — বড় step শুরুতে দ্রুত অগ্রগতি দেয়; প্রশিক্ষণের শেষের দিকে ছোট step মডেলটিকে একটি চওড়া minimum-এর চারপাশে ঘোরাঘুরির বদলে একটি তীক্ষ্ণতর, ভালো-generalizing minimum-এ বসতে দেয়।

`example.py` এই warmup+cosine schedule-টি হুবহু বাস্তবায়ন ও টেবুলেট করে।

## Video Script Outline

1. Motivation — "দুটি implementation detail, যা নিঃশব্দে প্রতিটি বাস্তব training run-কে বানায় বা ভাঙে"
2. fp32 বনাম fp16 বনাম bf16: range বনাম precision, এবং কেন bf16 জিতেছে
3. Underflow সমস্যা, একটি কংক্রিট ছোট gradient মান দিয়ে দেখানো
4. Loss scaling: backward-এর আগে scale, optimizer step-এর আগে unscale
5. Adam recap, তারপর AdamW-এর decoupled weight decay, পাশাপাশি
6. Warmup + cosine decay, এবং প্রতিটি ধাপের অন্তর্দৃষ্টি
7. `example.py`-এর walkthrough — স্ক্র্যাচ থেকে AdamW বনাম `torch.optim.AdamW`, LR schedule টেবিল, এবং fp16 underflow/loss-scaling demo
8. Recap + Lesson 5-এর প্রিভিউ: Lessons 1-4-এর সবকিছু একটি বাস্তব (যদিও ক্ষুদ্র) training run-এ একত্রিত

## Further Reading

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW)
- Micikevicius et al. (2018), *Mixed Precision Training* (মূল fp16 + loss-scaling recipe)
- Kalamkar et al. (2019), *A Study of BFLOAT16 for Deep Learning Training*
- Loshchilov & Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine annealing schedules)