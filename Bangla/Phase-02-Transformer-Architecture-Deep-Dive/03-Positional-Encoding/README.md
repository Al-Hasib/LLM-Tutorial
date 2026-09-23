# Positional Encoding

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `03-Positional-Encoding`

## কেন এই বিষয়টি গুরুত্বপূর্ণ

আগের লেসনে তৈরি self-attention value vector-গুলোর একটি *set*-এর উপর weighted average গণনা করে। `softmax(QKᵀ/√d_k)V`-এর কোনো অংশই বোঝায় না যে কিছু কোন অবস্থান থেকে এসেছে — যদি input token-গুলো shuffle করে output-কে আবার মিলিয়ে ফেরত সাজান, অভিন্ন ফল পাবেন। এটি একটি প্রকৃত সমস্যা: "the dog bit the man" এবং "the man bit the dog" — ক্রমবোধহীন একটি Transformer-এর কাছে দুটি অবিচ্ছেদ্য। এই লেসন সেই ফাঁকটি পূরণ করে — একটি RNN-তে এই ফাঁকটি একেবারেই নেই (কারণ এটি token-গুলোকে কঠোরভাবে ক্রমান্বয়ে প্রক্রিয়া করে, তাই ক্রমটি "ফ্রি" পায়), কিন্তু Transformer-এ এটি স্পষ্টভাবে নির্মাণ করা আবশ্যক।

## এই লেসনে যা যা শেখানো হবে

- অতিরিক্ত সাহায্য ছাড়াই self-attention যে permutation-invariant তা প্রমাণ
- Sinusoidal absolute positional encoding: সূত্র ও কেন এটি বেছে নেওয়া হয়েছিল
- Sinusoidal encoding-এর "ফ্রি" relative-position বৈশিষ্ট্য
- সরল বিকল্প হিসেবে learned positional embedding
- Relative position স্কিমগুলোর পূর্বাভাস (RoPE, ALiBi)

## ১. Self-attention-এর কোনো ধারণাই নেই token-গুলো কী ক্রমে এসেছে

Scaled dot-product attention আরেকবার দেখুন: প্রতিটি স্কোর `qᵢ · kⱼ` কেবল token `i` ও `j`-এর *content*-এর উপর নির্ভর করে — `i` বা `j`-এর সংখ্যা হিসেবে নিজেদের উপর কখনো নয়। `Q`, `K`, `V`-র প্রতিটি row-কে একইভাবে permute করলে, পুরো গণনাটিও সাথে সাথে permute হয় এবং ঠিক একই set output vector দেয়, শুধু label বদলায়। Self-attention হলো একটি set operation, যা sequence-এর পোশাক পরে আছে। `example.py` এটি সরাসরি দেখায়: একটি toy input permute করলে, output-গুলো (token পরিচয় অনুযায়ী মিলিয়ে) অভিন্ন বেরিয়ে আসে।

## ২. সমাধান: input-এ অবস্থান-তথ্য যোগ করা

Transformer কাগজের (paper) সমাধানটি প্রায় কৌতুককরভাবে সরল: এমন একটি vector গণনা করুন যা কেবল token-এর অবস্থানের উপর নির্ভর করে, এবং প্রথম attention layer-এর আগে **সেটিকে সরাসরি token embedding-এর সাথে যোগ করুন**:

```
input_to_layer_1 = token_embedding(token) + positional_encoding(position)
```

যেহেতু token embedding ও positional encoding একসাথে যোগ হয়, আর attention-এর তাদের আলাদা করার কোনো উপায় নেই, তাই অবস্থান-তথ্য এখন ঠিক একই `Q/K/V` projection-গুলোর মধ্য দিয়ে প্রবাহিত হয় এবং প্রতিটি attention score-কে প্রভাবিত করে।

## ৩. Sinusoidal positional encoding

মূল কাগজের নির্দিষ্ট পছন্দ, অবস্থান `pos` ও embedding dimension সূচক `i`-র জন্য (`d_model` মোট মাত্রার মধ্যে):

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

মাত্রার প্রতিটি জোড়া `(2i, 2i+1)` নিজস্ব ফ্রিকোয়েন্সিতে দোদুল্যমান হয় — নিম্ন মাত্রাগুলো দ্রুত দোদুল্যমান হয় (পার্শ্ববর্তী অবস্থানের মধ্যে বেশি পরিবর্তন), উচ্চ মাত্রাগুলো ধীরে (দীর্ঘ পরিসরের উপরে ধীরে ধীরে পরিবর্তন)। এর ফলে প্রতিটি অবস্থান একটি অনন্য "fingerprint" vector পায়, আর নিকটবর্তী অবস্থানগুলো *অনুরূপ* fingerprint পায় (তাদের cosine similarity বেশি), যা অবস্থান দূরত্ব বাড়ার সাথে সাথে মসৃণভাবে ক্ষয় পায় — `example.py` ঠিক এই বৈশিষ্ট্যটিই পরিমাপ করে।

## ৪. Relative-position কৌশলটি

এখানে কাগজটি যে মার্জিত অংশটি নির্দেশ করে: **যেকোনো নির্দিষ্ট offset `k`-এর জন্য**, `PE(pos + k)`-কে `PE(pos)`-এর একটি **রৈখিক ফাংশন** হিসেবে লেখা যায় — বিশেষভাবে, একটি নির্দিষ্ট rotation matrix `M_k` (সম্পূর্ণভাবে `k` থেকে নির্মিত, `pos` থেকে নয়) যেন প্রতিটি `pos`-এর জন্য `PE(pos + k) = M_k · PE(pos)` হয়; এখানে sine ও cosine-এর angle-addition identity ব্যবহার করা হয়। যেহেতু attention scores dot product (রৈখিক অপারেশন) দিয়ে গণনা করা হয়, তাই এর অর্থ হলো, দুই অবস্থানের *আপেক্ষিক* দূরত্ব — নীতিগতভাবে — নিচের কোনো linear layer শিখে বের করতে পারবে, এমনকি training-এ প্রতিটি নির্দিষ্ট absolute position না দেখলেও। `example.py` এই rotation সম্পর্ককে সংখ্যাগতভাবে যাচাই করে।

## ৫. Learned positional embedding

একটি সরল, সমানভাবে সাধারণ বিকল্প (GPT-2, BERT ও অন্যান্যদের মধ্যে ব্যবহৃত): position embedding-গুলোকে একটি নিয়মিত learned `nn.Embedding` টেবিল বানিয়ে ফেলুন, যেটি অবস্থান `0, 1, 2, ...` দিয়ে সূচিত হয় এবং token embedding-এর মতোই gradient descent দিয়ে প্রশিক্ষিত হয়। বাস্তবায়ন করা সহজ, অনুশীলনে ভালো কাজ করে, তবে এর একটি প্রকৃত সীমাবদ্ধতা আছে: training-এ দেখা সবচেয়ে বড় অবস্থানের চেয়ে লম্বা sequence অবস্থানের জন্য এটি কখনোই generalize করতে পারে না — model যদি কেবল দৈর্ঘ্য 1024 পর্যন্ত sequence-এ training পেয়ে থাকে, তবে অবস্থান 5000-এর জন্য টেবিলে কেবল row-ই নেই। অন্যদিকে sinusoidal encoding *যেকোনো* অবস্থানের জন্য গণনা করা যায়, এমনকি training-এ কখনো দেখা যায়নি এমন অবস্থানের জন্যও; তবে অনুশীলনে, যেকোনো স্কিম ব্যবহার করা হোক, trained model-গুলো এখনও তাদের training দৈর্ঘ্যের অনেক বাইরে খারাপ পারফর্ম করার প্রবণতা রাখে।

## ৬. পূর্বাভাস: relative position স্কিম

উপরের দুই পদ্ধতিই **absolute** অবস্থান এনকোড করে। পরবর্তী স্থাপত্যগুলো (architectures) দেখল যে input-এ পৃথক vector যোগ করার বদলে **relative** অবস্থানকে সরাসরি attention গণনার ভেতরেই এনকোড করা বেশি কার্যকর:

- **RoPE (Rotary Position Embedding)**: `Q` ও `K` vector-গুলোকে position-এর সমানুপাতিক একটি কোণে ঘোরায়, ফলে dot product `Q · K` স্বাভাবিকভাবেই `(position_i - position_j)`-এর উপর নির্ভর করে।
- **ALiBi (Attention with Linear Biases)**: positional embedding সম্পূর্ণভাবে বাদ দিয়ে, বদলে attention score-গুলো থেকে একটি দূরত্ব-সমানুপাতিক penalty সরাসরি বিয়োগ করে।

দুটি-ই এমনভাবে ডিজাইন করা হয়েছে যাতে model যত দৈর্ঘ্যে training পেয়েছে, তার চেয়ে লম্বা sequence-এ ভালো generalize করে — সম্পূর্ণ গভীর ডাইভ আছে [Phase 03: Long-Context Techniques](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md)-এ।

## Video Script Outline

1. Motivation — "attention জানে না কিছু কী ক্রমে এসেছে; লাইভ প্রমাণ করুন"
2. সমাধান: embedding-এর সাথে অবস্থান-নির্ভর vector যোগ করা
3. Sinusoidal সূত্র, মাত্রা ও অবস্থান জুড়ে তরঙ্গ-প্যাটার্ন visualize করা
4. Rotation হিসেবে দেখানো linear relative-position বৈশিষ্ট্য
5. সরল বিকল্প হিসেবে learned embedding, ও তাদের দৈর্ঘ্য-generalization সীমা
6. `example.py`-এর ওয়াকথ্রু
7. রিক্যাপ + পরে ব্যবহারের জন্য RoPE/ALiBi-র পূর্বাভাস

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.5
- Amirhossein Kazemnejad, *Transformer Architecture: The Positional Encoding* (blog, rotation-matrix বৈশিষ্ট্যের অত্যন্ত বিস্তারিত derivation)
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE, এখানে প্রিভিউ, সম্পূর্ণভাবে Phase 03-তে)
- Press, Smith, Lewis (2021), *Train Short, Test Long* (ALiBi)