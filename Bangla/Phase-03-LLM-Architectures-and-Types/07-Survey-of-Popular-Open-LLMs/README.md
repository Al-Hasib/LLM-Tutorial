# জনপ্রিয় Open LLM-গুলোর জরিপ

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `07-Survey-of-Popular-Open-LLMs`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর প্রতিটি ধারণা — decoder-only architecture, Mixture of Experts, RoPE, sliding-window attention — একাডেমিক ট্রিভিয়া নয়; এটি সেই open LLM-গুলোর আক্ষরিক parts list, যেগুলো আপনি আজ ডাউনলোড করে চালাতে পারেন। এই পাঠটি একটি ধাপে ধাপে ভ্রমণ, যা প্রতিটি বাস্তব, ব্যাপক ব্যবহৃত open মডেলকে এই কোর্সের সেই নির্দিষ্ট পাঠগুলোর সাথে ম্যাপ করে যা ব্যাখ্যা করে এটি কীভাবে কাজ করে, সাথে একটি সত্যিই নতুন, বাস্তবে গুরুত্বপূর্ণ কৌশল যা এই জরিপই প্রেরণা দেয়: **Grouped-Query Attention (GQA)** — multi-head attention-এর একটি ভ্যারিয়েন্ট যা বিশেষভাবে inference সস্তা করার জন্য ডিজাইন করা।

## এই পাঠে যা শেখা হবে

- LLaMA (1/2/3): আধুনিক open-model architecture baseline
- Mistral এবং Mixtral: প্রোডাকশনে sliding-window attention এবং Mixture of Experts
- অন্যান্য উল্লেখযোগ্য ফ্যামিলি: Falcon, Qwen, Gemma, Phi, DeepSeek
- Grouped-Query Attention (GQA): multi-head/multi-query-এর মধ্যভাগ
- নতুন চোখে একটি মডেলের config file পড়া

## 1. LLaMA: আধুনিক open baseline

Meta-র LLaMA ফ্যামিলি (Touvron et al., 2023) সেই architectural রেসিপি প্রতিষ্ঠা করেছে যা পরবর্তী বেশিরভাগ open decoder-only মডেল একত্রিত হয়:

- Positional encoding-এর জন্য **RoPE** ([Lesson 6 §1](../06-Long-Context-Techniques/README.md#1-rope-rotating-q-and-k-instead-of-adding-to-the-input))
- LayerNorm-এর বদলে **RMSNorm** ([Phase 02 Lesson 5 §2](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#2-layer-normalization)) — একটি সরলীকৃত normalization যা re-centering বাদ দেয় (কোনো mean subtraction নেই, শুধু root-mean-square দিয়ে rescaling), গণনা করতে সস্তা এবং গুণমানে সামান্য ক্ষতি
- FFN-এ GELU-এর বদলে **SwiGLU** activation ([Phase 02 Lesson 5 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#5-the-position-wise-feed-forward-network-ffn)) — একটি gated ভ্যারিয়েন্ট যা FFN-এর hidden layer-এ একটি learned multiplicative gate যোগ করে
- **Pre-LN** residual কাঠামো ([Phase 02 Lesson 5 §4](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#4-pre-ln-vs-post-ln))
- সস্তা inference-এর জন্য Chinchilla-optimal token সংখ্যার অনেক পরে training করা ([Lesson 5 §6](../05-Scaling-Laws/README.md#6-practical-implications))

Architecturally, LLaMA প্রায় হুবহু [Phase 02 Lesson 6-এর mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) — নির্দিষ্ট ছোট, সুপরীক্ষিত substitution-গুলোর এই সেট, অনেক বড় স্কেলে।

## 2. Mistral এবং Mixtral

Mistral 7B (Jiang et al., 2023) LLaMA-স্টাইল রেসিপির উপরে **sliding-window attention** ([Lesson 6 §3](../06-Long-Context-Techniques/README.md#3-sliding-window-local-attention-fixing-the-compute-not-just-position)) যোগ করেছে, সাথে **Grouped-Query Attention** (নিচে সম্পূর্ণ ভাবে পরিচয় করানো হয়েছে) inference-এর memory খরচ কমাতে। Mixtral (Jiang et al., 2024) একই base architecture নেয় এবং dense FFN-টিকে একটি **Mixture of Experts** layer ([Lesson 4](../04-Mixture-of-Experts/README.md)) দিয়ে প্রতিস্থাপন করে — 8 টি expert, top-2 routing — যার ফলে একই per-token compute খরচের একটি dense মডেলের তুলনায় অনেক বড় মোট parameter সংখ্যা পাওয়া যায়।

## 3. অন্যান্য উল্লেখযোগ্য ফ্যামিলি

| Model family | উল্লেখযোগ্য architectural পছন্দ |
|---|---|
| **Falcon** | LLaMA-স্টাইল backbone; সাবধানে ফিল্টার করা web-scale dataset-সহ একটি প্রাথমিক বড়-scale open রিলিজ |
| **Qwen** | শক্তিশালী multilingual tokenizer কভারেজ-সহ LLaMA-স্টাইল backbone; বড় রিলিজগুলো MoE ব্যবহার করে |
| **Gemma** | কিছু normalization placement tweak-সহ LLaMA-স্টাইল backbone; বড় অভ্যন্তরীণ মডেল থেকে distilled |
| **Phi** | তুলনামূলকভাবে ছোট মডেল, বিশাল স্কেলের বদলে অত্যন্ত উচ্চ-গুণমানের, কিউরেটেড training data-তে জোর দেয় |
| **DeepSeek-V2/V3** | আক্রমণাত্মক Mixture of Experts (অনেক ছোট expert, fine-grained routing) long-context কৌশলের সাথে মিলিত |

প্রায় সবগুলোর প্যাটার্ন: **এই কোর্সের একই ছোট architectural building block-গুলোর সেট, পুনরায় মিলিয়ে ও পুনরায় টিউন করা**, প্রতিবার মৌলিকভাবে নতুন architecture নয়।

## 4. Grouped-Query Attention (GQA): একটি নতুন, বাস্তবে গুরুত্বপূর্ণ ভ্যারিয়েন্ট

Multi-head attention ([Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md)) প্রতিটি head-কে তার *নিজস্ব* `K` এবং `V` projection দেয়। Inference-এ, autoregressive generation প্রতিটি আগের token-এর `K` ও `V` vector cache করে (এটি হলো "KV cache," সম্পূর্ণ আলোচনা [Phase 09: KV Cache and Speculative Decoding](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md)-এ), যাতে প্রতিটি নতুন generation step-এ সেগুলো পুনরায় compute করতে না হয় — এবং সেই cache-এর সাইজ সরাসরি মডেলের কতগুলো পৃথক K/V projection আছে তার সাথে স্কেল করে।

দুটি ভ্যারিয়েন্ট cache সাইজের বিনিময়ে গুণমানের ব্যবসা করে:

- **Multi-Query Attention (MQA)**: প্রতিটি head **একটি মাত্র** `K`/`V` projection ভাগ করে (শুধু `Q` থাকে per-head) — নাটকীয়ভাবে ছোট KV cache, কিন্তু এতটা কমিয়ে দেওয়া representational diversity থেকে লক্ষণীয় গুণমান খরচ।
- **Grouped-Query Attention (GQA)**: মধ্যভাগ। Heads-কে `g` টি group-এ ভাগ করো; group-এর ভিতরের heads একটি `K`/`V` projection **ভাগ** করে, কিন্তু ভিন্ন group-গুলো তাদের নিজস্ব পায়। `g=1` হলে হুবহু MQA; `g=num_heads` হলে হুবহু সাধারণ multi-head attention।

```
MHA:  num_heads separate K/V projections   (best quality, largest KV cache)
GQA:  g groups, g separate K/V projections  (tunable middle ground)
MQA:  1 shared K/V projection for all heads (smallest KV cache, most quality loss)
```

LLaMA 2 70B, LLaMA 3 এবং Mistral — সবাই বিশেষভাবে GQA ব্যবহার করে দীর্ঘ-conversation serving-কে সস্তা করতে, MQA-র পুরো গুণমান খরচ না দিয়েই।

## Video Script Outline

1. Motivation — "এই phase-এর প্রতিটি ধারণা, আপনি সত্যিই ডাউনলোড করতে পারেন এমন মডেলের উপর ম্যাপ করা"
2. LLaMA: RoPE + RMSNorm + SwiGLU + Pre-LN + trained-past-Chinchilla, প্রতিটিকে এর পাঠের সাথে যুক্ত করা
3. Mistral (sliding-window) এবং Mixtral (MoE), Lessons 4 ও 6-এর সাথে যুক্ত করা
4. দ্রুত ভ্রমণ: Falcon, Qwen, Gemma, Phi, DeepSeek — সত্যিই কী ভিন্ন, কী নয়
5. Grouped-Query Attention: KV-cache-size প্রেরণা, MHA/GQA/MQA একটি spectrum হিসেবে
6. `example.py`-এর ওয়াকথ্রু — GQA বাস্তবায়ন, MHA/GQA/MQA-তে সঠিক KV-cache সাইজ পরিমাপ, এবং জরিপ করা মডেলগুলোর একটি ছোট architecture-তুলনা টেবিল তৈরি
7. Recap: তিনটি architecture ফ্যামিলি, MoE, scaling laws, long-context, এবং এখন GQA → এই phase-এর সম্পূর্ণ টুলকিট, Phase 04-এর pretraining-এর জন্য প্রস্তুত

## Further Reading

- Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models* এবং *Llama 2: Open Foundation and Fine-Tuned Chat Models*
- Jiang et al. (2023), *Mistral 7B*; Jiang et al. (2024), *Mixtral of Experts*
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)