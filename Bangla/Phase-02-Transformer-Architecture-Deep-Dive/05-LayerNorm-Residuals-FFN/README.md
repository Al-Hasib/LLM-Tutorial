# Layer Norm, Residuals and Feed-Forward Sublayers

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `05-LayerNorm-Residuals-FFN`

## কেন এই বিষয়টি গুরুত্বপূর্ণ

আগের লেসনে residual connection (`x = x + sublayer(x)`), `LayerNorm` ও feed-forward block ব্যবহার করা হয়েছিল, কিন্তু সেগুলোর কোনো ন্যায্যতা দেওয়া হয়নি — এগুলোকে "স্থাপত্যকে কাজ করানো boring plumbing" হিসেবে দেখা হয়েছিল। এই লেসনে থেমে প্রতিটির প্রকৃত ব্যাখ্যা দেওয়া হয়েছে। এগুলোর কোনোটিই ঐচ্ছিক বিবরণ নয়: residual connection ছাড়া প্রকৃত Transformer (১২, ৪৮, এমনকি ১০০+ layer গভীর) কেবল training-ই হয় না; LayerNorm ছাড়া training অস্থিতিশীল; আর feed-forward sublayer, self-attention-এর চেয়ে অনেক কম মনোযোগ পেলেও, একটি সাধারণ Transformer layer-এর প্রায় দুই-তৃতীয়াংশ parameter ধারণ করে।

## এই লেসনে যা যা শেখানো হবে

- Residual (skip) connection ও কেন তারা অতি-গভীর network সক্ষম করে
- Layer Normalization: সূত্র ও কেন এটি প্রয়োজন
- LayerNorm বনাম BatchNorm — Transformer-গুলো বিশেষভাবে কেন LayerNorm ব্যবহার করে
- Pre-LN বনাম Post-LN স্থাপত্য
- Position-wise feed-forward network, ও কেন এতে এত বেশি parameter থাকে

## ১. Residual connection: কেন গভীরতা gradient-কে মেরে ফেলে না

[Phase 01: RNNs, LSTMs and GRUs](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#3-vanishing-and-exploding-gradients)-এর vanishing-gradient গল্পটি মনে করুন: রূপান্তরগুলো বারবার composition করলে gradient-গুলো অনেক layer-এর মধ্য দিয়ে backpropagate হওয়ার সময় সঙ্কুচিত হতে থাকে। Transformer sublayer-এর একটি সাধারণ গভীর stack, `x = sublayer(x)`,-এরও একই সমস্যা হবে — ২৪ বা ৯৬ layer গভীরে গিয়ে প্রথমদিকের layer-এ পৌঁছানো gradient সম্পূর্ণ নিশ্চিহ্ন হয়ে যেতে পারে।

সমাধান (He et al., 2015; মূলত CNN-এর জন্য, যা Transformer সরাসরি গ্রহণ করেছে): `x`-কে `sublayer(x)` দিয়ে *প্রতিস্থাপন* না করে, sublayer-এর output মূল input-এর সাথে **যোগ** করুন:

```
x = x + Sublayer(x)
```

Backpropagation-এর সময়, একটি যোগফলের gradient হলো gradient-গুলোর যোগফল — তাই এই সংযোগ দিয়ে পিছনে প্রবাহিত gradient-এর একটি সরাসরি, অবাধ পথ থাকে (`+ x`-এর gradient কেবল `1`), যা `Sublayer`-এর মধ্য দিয়ে যাওয়া পথ *উপরন্তু* প্রথমদিকের layer-গুলোতে সরাসরি পৌঁছে যায়। এমনকি `Sublayer`-এর নিজের gradient অবদান ছোট হলেও, identity পথটি নিশ্চিত করে যে সংকেত তবুও প্রথমদিকের layer-গুলোতে পৌঁছায়। এই একটিমাত্র সংযোজনই একটি বড় কারণ, যার ফলে Transformer-গুলো pre-residual স্থাপত্যের চেয়ে অনেক বেশি গভীরে স্তূপীকৃত (stacked) হতে পারে। `example.py` এই প্রভাবটি সরাসরি পরিমাপ করে — [Phase 01 Lesson 3](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/example.py)-এর একই gradient-flow পরিমাপ কৌশলটিকে প্রসারিত করে।

## ২. Layer Normalization

গভীর network-গুলো তাদের মধ্য দিয়ে প্রবাহিত activation-এর *স্কেল* (scale)-এর প্রতি কুখ্যাতভাবে সংবেদনশীল — layer-এর পর layer মান বাড়লে বা কমলে training অস্থিতিশীল হয়। **Layer Normalization** (Ba, Kiros, Hinton, 2016) প্রতিটি token-এর activation vector-কে পুনঃকেন্দ্রিক (re-center) ও পুনঃস্কেল (re-scale) করে mean 0 ও variance 1-এ আনে (তারপর একটি learned scale `γ` ও shift `β` প্রয়োগ করে, যাতে network কোনো নির্দিষ্ট layer-এর জন্য normalization পূর্বাবস্থায় ফেরানো ভালো মনে করলে সেটা করতে পারে):

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

যেখানে `μ` ও `σ²` হলো mean ও variance, যেগুলো গণনা করা হয় **একক token-এর vector-এর features জুড়ে** (batch জুড়ে নয়, আর অন্য token-গুলোর জুড়েও নয়)।

## ৩. কেন LayerNorm, BatchNorm নয়?

BatchNorm — CNN-তে সর্বব্যাপী — **batch dimension** জুড়ে normalize করে: প্রতিটি feature-এর জন্য, বর্তমান batch-এর প্রতিটি উদাহরণ নিয়ে mean/variance গণনা করে। দুটি সমস্যার কারণে এটি sequence model-এর জন্য অনুপযুক্ত:

- **পরিবর্তনশীল sequence দৈর্ঘ্য।** খুব ভিন্ন দৈর্ঘ্যের বাক্যের একটি batch (padding সহ) batch পরিসংখ্যানকে noisy করে তোলে এবং padding-কে যেভাবে সামলানো হয় তার উপর নির্ভরশীল করে।
- **Sequential/autoregressive generation।** Inference-এর সময় আপনি একবারে একটি token (বা অতি ছোট একটি batch) প্রক্রিয়া করতে পারেন — batch পরিসংখ্যান তখন অর্থহীন বা পুরোপুরি অনুপলব্ধ হয়ে যায়।

LayerNorm দুইটিকেই এড়িয়ে চলে: এর পরিসংখ্যান গণনা করা হয় **প্রতি token**-এ, batch size বা অন্য token-এর উপস্থিতি নির্বিশেষে; ফলে training-এর সময় ৫১২টি sequence-এর batch প্রক্রিয়া করা হোক বা inference-এর সময় একবারে একটি token তৈরি করা হোক, এটি সমানভাবে সুসংজ্ঞায়িত।

## ৪. Pre-LN বনাম Post-LN

মূল Transformer কাগজ (ও [Lesson 4](../04-Transformer-Encoder-Decoder/README.md)-এর উপস্থাপনা) `LayerNorm` residual যোগ করার **পরে** প্রয়োগ করে — "Post-LN": `x = LayerNorm(x + Sublayer(x))`। অনুশীলনে, model-গুলো গভীর হতে হতে দেখা গেল, সতর্কভাবে টিউন করা learning-rate warmup ছাড়া এতে training অস্থিতিশীল হয়। অধিকাংশ আধুনিক LLM (GPT-2 থেকে শুরু করে) বদলে **Pre-LN** ব্যবহার করে: sublayer-এর *আগে* normalize করে, আর সম্পূর্ণ জিনিসের চারপাশে residual যোগ করে:

```
Pre-LN:  x = x + Sublayer(LayerNorm(x))
```

Pre-LN residual পথটিকে input থেকে output পর্যন্ত একটি সত্যিকারের পরিষ্কার, অপরিবর্তিত identity connection-এ রাখে (LayerNorm কেবল sublayer-এর *ভেতরে* ঢোকা শাখাটিকে স্পর্শ করে), যা empirically খুব গভীর stack-এ অনেক বেশি স্থিতিশীল gradient দেয় — খরচ হলো ভালোভাবে টিউন করা Post-LN model-এর তুলনায় চূড়ান্ত model গুণমানে সামান্য ক্ষতি। এটি দেখতে ছোট একটি স্থাপত্য-পছন্দ, কিন্তু বড় পরিসরে trainability-তে এর প্রভাব অপরিসীম।

## ৫. Position-wise feed-forward network (FFN)

Attention-ই একমাত্র sublayer যেখানে token-গুলো একে অপরের সাথে তথ্য বিনিময় করে। FFN sublayer বিপরীত কাজ করে: এটি **প্রতিটি token-এর vector-কে স্বাধীনভাবে ও সদৃশভাবে প্রক্রিয়া করে** — প্রতিটি অবস্থানে একই weight সহ প্রয়োগ করা একটি ছোট 2-স্তর MLP:

```
FFN(x) = W₂ · activation(W₁ x + b₁) + b₂
```

মূল কাগজে ReLU ব্যবহৃত হয়েছে; GPT-2 থেকে সাধারণত **GELU** ব্যবহৃত হয় ([Phase 00: Neural Networks Basics §2](../../Phase-00-Prerequisites/02-Neural-Networks-Basics/README.md#2-activation-functions) মনে করুন)। Hidden dimension `d_ff`-কে প্রচলিতভাবে `d_model`-এর **৪ গুণ** রাখা হয় (যেমন GPT-2-small-এ `d_model=768` → `d_ff=3072`) — এই expand-then-contract আকৃতি sublayer-কে প্রতিটি token-এর representation রূপান্তরের যথেষ্ট ক্ষমতা দেয়। যেহেতু `W₁` ও `W₂` সম্পূর্ণ `(d_model, d_ff)` matrix, তাই FFN sublayer সাধারণত **একটি Transformer layer-এর মোট parameter-এর প্রায় দুই-তৃতীয়াংশ** ধারণ করে — attention-ই সবচেয়ে বেশি ধারণাগত মনোযোগ পায় (শব্দের খেলা সদর্থক), কিন্তু raw ধরে parameter গোনা হলে বেশি "কাজ" করছে FFN-ই।

## Video Script Outline

1. Motivation — "৯৬-স্তর-বিশিষ্ট model-কে সত্যিই trainable করে তোলে এমন boring plumbing"
2. Residual connection: gradient-highway intuition, Phase 01-এর vanishing gradient-এর সাথে যুক্ত করা
3. LayerNorm সূত্র, ও কেন per-token (per-batch নয়) normalization sequence-এর সাথে খাপ খায়
4. Pre-LN বনাম Post-LN, এবং কেন আধুনিক model-গুলো Pre-LN-এ সরে গেছে
5. FFN sublayer: expand-then-contract, ও parameter-সংখ্যার আশ্চর্য
6. `example.py`-এর ওয়াকথ্রু — residual-connection gradient-flow প্রভাব সরাসরি পরিমাপ, hand-rolled LayerNorm-কে PyTorch-এর সাথে যাচাই, এবং FFN বনাম attention parameter গণনা
7. রিক্যাপ: প্রতিটি অংশ এখন তৈরি → পরবর্তী লেসনে একটি সম্পূর্ণ trainable mini-GPT জোড়া লাগানো হবে

## Further Reading

- He, Zhang, Ren, Sun (2015), *Deep Residual Learning for Image Recognition* (ResNets — residual-connection ধারণার উৎস)
- Ba, Kiros, Hinton (2016), *Layer Normalization*
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* (Pre-LN বনাম Post-LN training-স্থিতিশীলতা বিশ্লেষণ)
- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.3 (FFN sublayer)