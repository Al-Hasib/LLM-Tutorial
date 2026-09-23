# Prompt Tuning, Prefix Tuning এবং Adapters

**Phase:** [LLM ফাইন-টিউনিং](../README.md) · **টপিক ফোল্ডার:** `03-Prompt-Tuning-Prefix-Tuning-Adapters`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 2](../02-LoRA-and-QLoRA/README.md)-এ আমরা সেই PEFT পদ্ধতিটি দেখেছি, যা আপনি বাস্তবে সবচেয়ে বেশি ব্যবহার করবেন (LoRA), কিন্তু এটি প্রথম PEFT পদ্ধতি ছিল না, আর "frozen model-এর সাথে ছোট একটি trainable add-on" কেমন হতে পারে তার একমাত্র রূপও নয়। এই lesson-টি [Lesson 1](../01-Full-Finetuning-vs-PEFT/README.md)-এর সেই একই প্রশ্নের — base freeze করো, ছোট কিছু train করো — তিনটি পূর্ববর্তী, কাঠামোগতভাবে ভিন্ন উত্তর নিয়ে আলোচনা করে, যেখানে প্রতিটিই তার trainable অংশটি model-এর ভিন্ন একটি জায়গায় বসায়। তিনটিকেই পাশাপাশি দেখলে, LoRA-সহ, "PEFT" একটি নির্দিষ্ট কৌশল থেকে পরিণত হয় প্রকৃত design space-এ — বাস্তব trade-off সহ — এবং [Hugging Face `peft` library](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) ঠিক এভাবেই এটিকে উপস্থাপন করে: LoRA, prompt tuning, prefix tuning আর adapters — সবই সেই একই frozen-base ধারণার ওপর ভিন্ন ভিন্ন `PeftConfig` subclass মাত্র।

## এই lesson-এ যা যা শেখানো হবে

- Prompt Tuning: input-এর আগে জুড়ে দেওয়া trainable "soft prompt" embedding
- Prefix Tuning: শুধু input-এ নয়, প্রতিটি layer-এ trainable key/value vector জুড়ে দেওয়া
- Adapters: প্রতিটি Transformer layer-এর ভেতরে বসানো ছোট trainable bottleneck MLP
- প্রতিটি পদ্ধতির যোগ করা parameter শারীরিকভাবে কোথায় থাকে, আর এই অবস্থান কেন গুরুত্বপূর্ণ
- এক পাশাপাশি trainable-parameter তুলনা — পদ্ধতিগুলো একে অপরের এবং LoRA-র বিপরীতে

## ১. Prompt Tuning: trainable tokens, বাকি সবকিছু frozen

Lester, Al-Rfou, Constant (2021) এই ধারণাটির সবচেয়ে সরল সংস্করণটি প্রস্তাব করেন: `k` সংখ্যক নতুন, random-ভাবে initialize করা embedding vector নাও (সাধারণত `k = 10`-`100`), সেগুলোকে input sequence-এর token embedding-গুলোর **আগে জুড়ে দাও**, আর model-এর বাকি সবকিছু — সেই সাধারণ token embedding table-সহ — পুরোপুরি freeze করো। শুধু এই `k` সংখ্যক নতুন "soft prompt" vector-ই কখনো gradient update পায়:

```
input to layer 1 = [ soft_prompt_1, ..., soft_prompt_k, embed(token_1), ..., embed(token_T) ]
```

এই vector-গুলো কোনো প্রকৃত token-এর word embedding নয় — এরা মুক্ত-ভাসমান parameter, যাদেরকে gradient descent এমন একটি representation-এ রূপ দেয়, যা frozen model-টিকে কাঙ্ক্ষিত task-এর দিকে সবচেয়ে ভালোভাবে চালিত করে। এরা কাজ করে একটি স্থায়ীভাবে-অপটিমাইজ করা, continuous (বিচ্ছিন্ন শব্দের বদলে) prompt-এর মতো, যা প্রকৃত input-এর আগে বসানো থাকে। যেহেতু এরা সরাসরি শুধু প্রথম layer-এর input-কেই প্রভাবিত করে, তাই গভীর layer-গুলোতে এদের প্রভাব পরোক্ষ — পরের প্রতিটি layer-ই এদের "অনুভব" করে সীমিত পরিমাণে, যতটুকু প্রথম layer-এর self-attention এদের প্রভাব সামনের দিকে ছড়িয়ে দিয়েছে।

## ২. Prefix Tuning: প্রতিটি layer-এ প্রভাব

Li and Liang (2021) একই সময়ে, স্বাধীনভাবে, একটি ঘনিষ্ঠভাবে সম্পর্কিত কিন্তু আরও শক্তিশালী ভ্যারিয়েন্ট প্রস্তাব করেন: শুধু *input* embedding-এর আগে trainable vector জুড়ে দেওয়ার বদলে, সরাসরি **প্রতিটি Transformer layer-এর self-attention-এর keys ও values-এর** আগে trainable vector জুড়ে দাও:

```
layer i's attention keys   = [ prefix_K_i,  K_1, ..., K_T ]
layer i's attention values = [ prefix_V_i,  V_1, ..., V_T ]
```

প্রতিটি layer-ই পায় তার *নিজস্ব* trainable prefix vector (প্রতি layer-এ আলাদা একটি `prefix_K_i`, `prefix_V_i` জোড়া), তাই যোগ করা parameter-গুলো network-এর গভীরে attention-এর আচরণকেও সরাসরি রূপ দিতে পারে, শুধু input-এ নয়। ফলে প্রশিক্ষণের জন্য parameter-এর একই বাজেটে, prompt tuning-এর তুলনায় prefix tuning model-এর আচরণের ওপর লক্ষণীয়ভাবে বেশি প্রভাব ফেলে। এর খরচ: বেশি trainable parameter (যা layer-সংখ্যার সাথে বাড়ে, শুধু input-এ একবার নয়) এবং আরও বেশি হস্তক্ষেপকারী বাস্তবায়ন (কেবল embedding layer-ই নয়, প্রতিটি layer-এর attention computation-এ প্রবেশের দরকার হয়)।

## ৩. Adapters: প্রতিটি layer-এর ভেতরে ছোট bottleneck MLP

Houlsby et al. (2019) — কালানুক্রমিকভাবে তিনটির মধ্যে সবচেয়ে আগেরটি, আর সেই paper, যেটি প্রথম "base freeze করো, ছোট add-on train করো" এই সাধারণ PEFT কৌশলটিকে জনপ্রিয় করে — আবার ভিন্ন পথ বেছে নেয়: প্রতিটি Transformer block-এর ভেতরে প্রতিটি sublayer-এর (attention ও feed-forward) পরে একটি ছোট **bottleneck MLP module** বসাও:

```
Adapter(x) = x + W_up @ activation(W_down @ x)

W_down : bottleneck x d_model     (projects DOWN to a small bottleneck dimension)
W_up   : d_model x bottleneck     (projects back UP to the original width)
bottleneck << d_model                (e.g. 8-64, versus d_model in the hundreds/thousands)
```

adapter-এর চারপাশের residual connection (`x + ...`) মানে হলো adapter-কে এমনভাবে initialize করা হয় যেন এর output শূন্য হয় (`W_up`-কে প্রায় শূন্য দিয়ে initialize করে), তাই — LoRA-র শূন্য-init `B` matrix-এর মতোই — প্রশিক্ষণ শুরু হয় frozen model-এর মূল আচরণ থেকে, আর ধীরে ধীরে সরে আসে। prompt/prefix tuning-এর মতো নয়, adapters প্রতিটি forward pass-এ অল্প পরিমাণ **অতিরিক্ত compute** যোগ করে (প্রতি adapter-এ, প্রতি token-এ দুটি অতিরিক্ত ছোট matrix multiplication), কারণ এরা computation graph-এর ভেতরে inline থাকে — শুধু input sequence বা attention keys/values-কে দীর্ঘায়িত করার বদলে।

## ৪. Parameter-গুলো কোথায় থাকে: একটি পাশাপাশি তুলনা

| পদ্ধতি                                            | Trainable parameter-গুলো কোথায় থাকে                            | প্রতি forward pass-এ compute যোগ করে?                       | merge করে zero overhead-এ ফেরা যায়?                        |
| ------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Prompt Tuning                                     | নতুন embedding vector, একবার input-এ মাত্র, আগে জুড়ে দেওয়া    | Sequence সামান্য দীর্ঘ হয়                                    | না                                                           |
| Prefix Tuning                                     | নতুন K/V vector, প্রতিটি attention layer-এ জুড়ে দেওয়া        | প্রতি layer-এ কার্যকর sequence সামান্য দীর্ঘ                  | না                                                           |
| Adapters                                          | নতুন bottleneck MLP, প্রতিটি layer-এর ভেতরে বসানো               | হ্যাঁ — প্রতি layer-এ দুটি অতিরিক্ত ছোট matmul                | না                                                           |
| LoRA ([Lesson 2](../02-LoRA-and-QLoRA/README.md)) | বিদ্যমান weight matrix-গুলোর পাশাপাশি low-rank update           | Train-এর সময় কোনো অতিরিক্ত compute নেই; merge-এর পরে **zero** | **হ্যাঁ** — সরাসরি বিদ্যমান weights-এ merge হয়               |

LoRA-র মূল weight matrix-গুলোতে আক্ষরিক অর্থেই zero inference-time overhead-এ merge হয়ে যাওয়ার ক্ষমতাটিই মূলত এর অনুশীলনে ডিফল্ট পছন্দ হয়ে ওঠার বড় কারণ — অন্য তিনটি পদ্ধতিই প্রতিটি inference forward pass-এ একটি ছোট স্থায়ী ব্যয় রেখে যায় (দীর্ঘ কার্যকর sequence, বা অতিরিক্ত matmul), যাকে merge করে দূর করা যায় না, কারণ তাদের trainable parameter-গুলো কোনো বিদ্যমান weight matrix-এর আকৃতির সাথে মেলে না।

## ৫. Trainable-parameter count, সরাসরি তুলনা করা

একই base model-এর জন্য, পদ্ধতি ও তার hyperparameter-এর ওপর নির্ভর করে (`k` সংখ্যক soft-prompt token, adapter bottleneck size, বা LoRA rank `r`) trainable parameter count-গুলো কয়েক অর্ডার অফ ম্যাগনিচিউড পর্যন্ত আলাদা হয়। `example.py` একটি বাস্তবসম্মত model configuration-এর জন্য এই তুলনাটি সরাসরি হিসাব ও প্রিন্ট করে, সাথে দুটি হাতে-কলমে gradient-flow প্রদর্শনীও: একটি কার্যকর soft-prompt-tuning সেটআপ এবং একটি কার্যকর adapter module — প্রতিটিই যাচাই করা হয় যে frozen base model-এর gradient-গুলো নিখুঁত শূন্যই থেকে যায়।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "LoRA-ই এখানে প্রথম ধারণা ছিল না, আর বিকল্পগুলো তাদের trainable parameter-গুলো সত্যিই ভিন্ন ভিন্ন জায়গায় রাখে"
2. Prompt Tuning: input-এর আগে জুড়ে দেওয়া soft embeddings, বাকি সবকিছু frozen
3. Prefix Tuning: একই ধারণা প্রতিটি layer-এর K/V-তে ঠেলে দেওয়া, বেশি প্রভাবের জন্য
4. Adapters: inline-এ বসানো bottleneck MLP, সাথে zero-init residual
5. পাশাপাশি টেবিল: parameter কোথায় থাকে, আর শুধু LoRA-র থাকা merge-to-zero-overhead বৈশিষ্ট্য
6. `example.py`-এর ওয়াকথ্রু — একটি soft prompt ও একটি adapter train করো, frozen-gradient বিচ্ছিন্নতা যাচাই করো, চারটি পদ্ধতির parameter count তুলনা করো
7. Recap + preview: Lesson 4 একটি ছোট model-এর full fine-tuning ব্যবহার করে শুধু instruction-tuning objective-এর ওপরই মনোযোগ দেয়, আর Lesson 5 এসব পদ্ধতির জন্য প্রকৃত Hugging Face `peft` API দেখায়

## আরও পড়ার জন্য

- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP* (Adapters)
- Li and Liang (2021), *Prefix-Tuning: Optimizing Continuous Prompts for Generation*
- Lester, Al-Rfou, Constant (2021), *The Power of Scale for Parameter-Efficient Prompt Tuning*
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (সরাসরি তুলনার জন্য [Lesson 2](../02-LoRA-and-QLoRA/README.md) থেকে পুনরালোচিত)