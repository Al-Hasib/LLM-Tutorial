# Long-Context Techniques

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `06-Long-Context-Techniques`

## কেন এটি গুরুত্বপূর্ণ

এই কোর্সের আগের অংশ থেকেই দুটি অমীমাংসিত সমস্যা চিহ্নিত ও স্থগিত করা হয়েছে: self-attention-এর `O(T²)` খরচ ([Phase 01 §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity)), এবং absolute positional encoding-এর training-এ দেখা দৈর্ঘ্যের বাইরে generalize করতে না পারা ([Phase 02 Lesson 3 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md#5-learned-positional-embeddings))। এই পাঠটি দুটোই সমাধান করে, সেই দুটি কৌশল দিয়ে যা কার্যত প্রতিটি আধুনিক long-context LLM সত্যিই ব্যবহার করে: RoPE এবং sliding-window/local attention, সাথে একটি শিক্ষণীয় বিকল্প হিসেবে ALiBi।

## এই পাঠে যা শেখা হবে

- RoPE (Rotary Position Embedding): attention-এ সরাসরি relative position বেক করা
- ALiBi: একটি parameter-free বিকল্প যা দূরত্ব দিয়ে attention score-তে bias যোগ করে
- Sliding-window (local) attention: linear-time compute-এর জন্য full context-এর বিনিময়
- বাস্তব ডিপ্লয়ড মডেলগুলোতে এগুলো কীভাবে মিলিত হয় ([Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md)-তে আরও প্রিভিউ)

## 1. RoPE: input-এ যোগ করার বদলে Q এবং K ঘোরানো

মনে পড়ুন [Phase 02 Lesson 3 §4](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md#4-the-relative-position-trick): sinusoidal encoding-এর এমন একটি বৈশিষ্ট্য আছে যে `PE(pos+k)` হলো `PE(pos)`-এর একটি নির্দিষ্ট rotation, যা একটি linear layer *নীতিগতভাবে* relative-position reasoning-এর জন্য কাজে লাগাতে শিখতে পারত — কিন্তু কেউই জোর করে না। **RoPE** (Su et al., 2021) এটিকে *একমাত্র* বিকল্প বানায়, input-এ একটি পৃথক vector যোগ করার বদলে rotation-টিকে সরাসরি attention mechanism-এর ভিতরে বেক করে:

```
q_rotated = Rotate(q, position_i)
k_rotated = Rotate(k, position_j)
q_rotated · k_rotated  depends ONLY on (position_i - position_j), never on the absolute positions
```

এটি [Phase 02 Lesson 3-এর `example.py`](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/example.py)-এর হুবহু একই per-dimension-pair rotation matrix ব্যবহার করে — একমাত্র পার্থক্য *কোথায়* rotation প্রয়োগ করা হয়: input embedding-এ একবার, additively নয়, বরং `Q` এবং `K` নিজেদের উপর (প্রতিটি layer-এর প্রতিটি attention গণনায়)। যেহেতু relative position এখন dot product-এর নিজেই একটি গাণিতিক নিশ্চয়তা, তাই RoPE-ভিত্তিক মডেলগুলো সাধারণত absolute-position প্রকল্পের তুলনায় training-এ দেখা দৈর্ঘ্যের বাইরের sequence length-এ ভালো generalize করে। RoPE LLaMA, Mistral এবং বেশিরভাগ আধুনিক open LLM-তে ব্যবহৃত হয়।

## 2. ALiBi: কোনো position embedding-ই নেই

Press, Smith, Lewis (2021) একটি ভিন্ন, আশ্চর্যজনকভাবে সহজ পদ্ধতি নিয়েছিল: **input-এ বা Q/K-তে কোথাও position encode করবে না।** বদলে, softmax-এর আগে, raw attention score থেকে সরাসরি একটি দূরত্ব-সমানুপাতিক penalty বিয়োগ করুন:

```
scores[i, j] = (q_i · k_j) - m · |i - j|      (only for j <= i, under a causal mask)
```

`m` একটি নির্দিষ্ট, head-নির্দিষ্ট slope (ভিন্ন head ভিন্ন, geometrically-spaced slope পায়, তাই কিছু head আরও স্থানীয়ভাবে এবং কিছু আরও ব্যাপকভাবে ফোকাস করে)। position-এর জন্য **শূন্য learned parameter** আছে — penalty দূরত্বের একটি নির্দিষ্ট ফাংশন, একবার গণনা করা হয় এবং যেকোনো sequence length-এর জন্য পুনর্ব্যবহৃত হয়, training-এর বাইরের দৈর্ঘ্যসহ। এই "train short, test long" দৃঢ়তাই ছিল ALiBi-এর বিশেষ বিক্রয়-বিন্দু।

## 3. Sliding-window (local) attention: শুধু position নয়, compute ঠিক করা

RoPE এবং ALiBi দুটোই সম্বোধন করে *positional generalization* — কোনোটিই অন্তর্নিহিত `O(T²)` compute খরচ স্পর্শ করে না। **Sliding-window attention** সরাসরি compute সমস্যাটিকে আক্রমণ করে: প্রতিটি token-কে পুরো sequence-এর বদলে শুধুমাত্র `w` টি কাছাকাছি token-এর একটি নির্দিষ্ট-সাইজ window-এ attend করার মধ্যে সীমাবদ্ধ রাখো (অনেক বাস্তবায়নে, সাথে কয়েকটি মনোনীত "global" token, যেগুলো সবাই দেখতে পারে):

```
Full attention:            each token attends to all T tokens      -> O(T^2)
Sliding-window attention:  each token attends to only w tokens      -> O(T * w), linear in T
```

Mistral এটিকে প্রোডাকশন স্কেলে জনপ্রিয় করেছে, RoPE-এর সাথে মিলিয়ে, দেখিয়েছে যে দীর্ঘ sequence-এর বেশিরভাগ দরকারি context যেকোনো নির্দিষ্ট token-এর জন্য বেশিরভাগ সময় সত্যিই স্থানীয় — এবং বেশ কয়েকটি sliding-window layer স্ট্যাক করলেও তথ্য *শেষ পর্যন্ত* পুরো sequence জুড়ে ছড়িয়ে পড়ে (layer `l`-এর সাইজ-`w` window, layer `l-1`-এর সাথে মিলে দুই layer-এর পরে `~2w`-এর একটি কার্যকর receptive field দেয়, গভীরতার সাথে বাড়তে থাকে) — CNN-এ ছোট convolutional kernel স্ট্যাক করে বড় কার্যকর receptive field গড়ার spirit-এর সাথে সাদৃশ্যপূর্ণ।

## 4. বাস্তবে এগুলো কীভাবে মিলিত হয়

বাস্তব long-context মডেলগুলো মিশিয়ে নেয়: relative-position-aware, length-generalizing attention-এর জন্য RoPE, মাঝে মাঝে কিছু (বা সব) layer-এ sliding-window attention-এর সাথে মিলিয়ে সরাসরি compute খরচ নিয়ন্ত্রণ করতে, এবং মাঝে মাঝে কয়েকটি full-attention layer-এর পাশাপাশি, কিছু সত্যিই global, সীমাহীন context ধরে রাখতে। [Lesson 7-এর জরিপ](../07-Survey-of-Popular-Open-LLMs/README.md) দেখাবে কোন বাস্তব, open মডেল কোন সংমিশ্রণ ব্যবহার করে।

## Video Script Outline

1. Motivation — "Phase 01/02 থেকে চিহ্নিত দুটি সমস্যা, অবশেষে সমাধান"
2. RoPE: input-এ যোগ নয়, Q/K ঘোরাও — relative position গাণিতিকভাবে নিশ্চিত হয়
3. ALiBi: position embedding সম্পূর্ণ এড়িয়ে যাও, দূরত্ব দিয়ে সরাসরি scores-এ bias করো
4. Sliding-window attention: O(T²) → O(T*w), এবং কীভাবে স্ট্যাক করা layer-গুলো তারপরও দূর দেখে
5. `example.py`-এর ওয়াকথ্রু — RoPE বাস্তবায়ন করে বাস্তব Q/K vector-এ সরাসরি relative-position বৈশিষ্ট্য যাচাই; ALiBi-এর bias matrix বাস্তবায়ন; sliding-window attention-এর linear বনাম full attention-এর quadratic বৃদ্ধি পরিমাপ
6. Recap + কোন বাস্তব মডেল কোন সংমিশ্রণ ব্যবহার করে তার জন্য [Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md)-এ pointer

## Further Reading

- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE)
- Press, Smith, Lewis (2021), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi)
- Jiang et al. (2023), *Mistral 7B* (প্রোডাকশন স্কেলে sliding-window attention)
- Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer* (একটি আগের, প্রভাবশালী local + global attention প্যাটার্ন)