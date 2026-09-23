# Sequence-to-Sequence ও Attention

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `04-Seq2Seq-and-Attention`

## কেন এটি গুরুত্বপূর্ণ

এটি এই phase-এর সবচেয়ে গুরুত্বপূর্ণ পাঠ: **attention** — যে mechanism দিয়ে এই পুরো কোর্সের model-গুলোর নামকরণ ও নির্মাণ — ঠিক এখানেই আবিষ্কৃত হয়েছিল, RNN-ভিত্তিক translation model-এর একটি নির্দিষ্ট, সীমিত ব্যর্থতার প্যাচ হিসেবে। এটি কোন সমস্যা সমাধানের জন্য ডিজাইন করা হয়েছিল তা একবার দেখলে, [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md)-এর self-attention আর একটি রহস্যময় নতুন ধারণা বলে মনে হয় না — বরং আপনি এখানে হাতে যা তৈরি করতে যাচ্ছেন তার সুস্পষ্ট generalisation বলে মনে হতে শুরু করে।

## এই পাঠে যা যা শেখা হবে

- Sequence-to-Sequence (Seq2Seq) architecture: encoder + decoder
- Fixed-context bottleneck সমস্যা
- Attention: decoder-কে প্রতিটি encoder state-এ ফিরে তাকাতে দেওয়া
- Attention সূত্র: scores → softmax → weighted context
- Additive (Bahdanau) বনাম multiplicative (Luong / dot-product) attention
- কেন এটি ইতিমধ্যে গোপনে "query, key, value"

## 1. Sequence-to-Sequence (Seq2Seq)

Machine translation-এর জন্য প্রবর্তিত (Sutskever et al., 2014), একটি Seq2Seq model হলো শৃঙ্খলিত দুটি RNN:

- **Encoder**: পুরো উৎস sentence-কে একবারে একটি শব্দ করে পড়ে এবং এটিকে একটি একক নির্দিষ্ট-আকারের vector-এ সংকুচিত করে — তার final hidden state। একে প্রায়ই "context vector" বা "thought vector" বলা হয়।
- **Decoder**: আরেকটি RNN, সেই context vector দিয়ে আরম্ভ করা, যা লক্ষ্য sentence-কে একবারে একটি শব্দ তৈরি করে, প্রতিটি generated শব্দকে নিজের পরবর্তী input হিসেবে ফিরিয়ে খাওয়ায়।

```
source sentence -> [Encoder RNN] -> context vector -> [Decoder RNN] -> target sentence
```

## 2. Fixed-context bottleneck

এখানে সমস্যা: **একটি নির্বিচারে দীর্ঘ উৎস sentence-এর পুরো অর্থ একটি নির্দিষ্ট আকারের vector-এর মধ্য দিয়ে চাপিয়ে দিতে হয়।** 5-শব্দের বাক্যের জন্য এটি প্রায় অলক্ষিত; 40-শব্দের বাক্যের জন্য, encoder যখন শেষে পৌঁছায় ততক্ষণে বাক্যের শুরুর তথ্য সাধারণত ইতিমধ্যেই মিশ্রিত বা মুছে ফেলা হয়ে থাকে — [RNNs, LSTMs and GRUs](../03-RNN-LSTM-GRU/README.md) থেকে সেই vanishing-signal সমস্যা, এখন সবকিছুকে একটি একক vector-এর মধ্য দিয়ে জোর করার কারণে আরও জটিল। উৎস sentence দীর্ঘ হলে translation মান পরিমাপযোগ্যভাবে অবনত হয়, আর এই bottleneck-ই ঠিক তার কারণ।

## 3. সমাধান: attention

Bahdanau et al. (2014) একটি সরাসরি সমাধান প্রস্তাব করেছিলেন: **পুরো sentence-কে একটি vector-এ সংকুচিত করা বন্ধ করো।** বদলে, *প্রতিটি* encoder hidden state ধরে রাখো, এবং প্রতিটি generation ধাপে decoder-কে সেগুলোর সবগুলোর দিকে ফিরে তাকাতে ও ঠিক করতে দাও কোনগুলো এই মুহূর্তে প্রাসঙ্গিক।

কংক্রিটভাবে, প্রতিটি decoder ধাপ `t`-এ, decoder state `h_t^{dec}` এবং encoder hidden states `h_1^{enc}, ..., h_n^{enc}` সহ:

```
score(h_t^{dec}, h_i^{enc}) = encoder position i এই মুহূর্তে কতটা প্রাসঙ্গিক?
α_{t,i} = softmax_i( score(h_t^{dec}, h_i^{enc}) )        # attention weights, যোগফল 1
context_t = Σ_i α_{t,i} · h_i^{enc}                        # weighted sum = attention output
```

সেই `context_t` — পুরো উৎস sentence-এর একটি উপযোগী মিশ্রণ, প্রতিটি decoding ধাপে নতুন করে পুনর্গণনা করা — তারপর decoder state-এর সাথে একত্রিত হয়ে পরবর্তী output শব্দ উৎপন্ন করে। **এটি হুবহু একটি differentiable, weighted lookup**: softmax কাঁচা score-কে probability-সদৃশ বণ্টনে রূপান্তর করে ([Phase 00 §4](../../Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md#4-probability)), আর "উত্তর" হলো একটি weighted average — একক নির্দিষ্ট position-এর কঠিন পছন্দ নয়।

## 4. Additive বনাম multiplicative attention

Score গণনার দুটি জনপ্রিয় উপায়:

- **Additive / Bahdanau attention**: `score = vᵀ tanh(W₁ h^{dec} + W₂ h^{enc})` — একটি ছোট feedforward network scoring function শেখে।
- **Multiplicative / Luong (dot-product) attention**: `score = h^{dec} · h^{enc}` (বা `h^{dec}ᵀ W h^{enc}`) — শুধু একটি dot product, ঐচ্ছিকভাবে মাঝখানে একটি শেখা matrix সহ।

Dot-product attention সস্তা (প্রতি জোড়ায় ছোট neural net-এর বদলে একটি matrix multiply) এবং, সঠিকভাবে scaled হলে, *ঠিক* সেই mechanism যা [Transformers](../05-Intro-to-Transformers/README.md) model-এর প্রতিটি attention গণনায় প্রমিত করে।

## 5. এটিই ইতিমধ্যে Query/Key/Value

অংশগুলো পুনরায় লেবেল করলে Transformer পরিভাষার সাথে সংযোগ স্পষ্ট হয়ে যায়:

| Seq2Seq attention                       | Transformer terminology                                               |
| --------------------------------------- | --------------------------------------------------------------------- |
| decoder state`h_t^{dec}`              | **query** — "আমি এই মুহূর্তে কী খুঁজছি?"                       |
| প্রতিটি encoder state`h_i^{enc}`       | **key** — "এখানে আমি যা ধারণ করি, আমার সাথে তুলনা করো"             |
| প্রতিটি encoder state`h_i^{enc}` (আবার) | **value** — "আমি প্রাসঙ্গিক হলে আসলে যা retrieve করা হবে"         |
| `softmax(score) · encoder states`      | attention output                                                        |

Transformer-এর তৈরি একমাত্র প্রকৃত generalisation (পরবর্তী পাঠে, এবং পূর্ণরূপে Phase 02-তে) হলো: **attention-কে "decoder encoder-এর দিকে তাকাচ্ছে"-তে সীমাবদ্ধ করা বন্ধ করো।** Sequence-এর *প্রতিটি* position-কে *প্রতিটি* অন্য position-এ মনোযোগ দিতে দাও, একই sequence-এর ভেতরেও — সেটিই **self**-attention, এবং এটিই অবশেষে field-কে recurrence সম্পূর্ণ বাদ দিতে দিল।

## Video Script Outline

1. Motivation — "translation model-এর একটি নির্দিষ্ট bug ছিল; সমাধানটি আধুনিক AI-এর সবচেয়ে গুরুত্বপূর্ণ ধারণা হয়ে গেল"
2. Seq2Seq architecture diagram: encoder squeeze, decoder generate
3. Bottleneck, একটি দীর্ঘ sentence উদাহরণ দিয়ে কংক্রিট করা
4. Attention: scores -> softmax -> weighted context, ধাপে ধাপে
5. Additive বনাম multiplicative scoring
6. `example.py`-র walkthrough — একটি টয় copy task-এ attention একটি "soft lookup" হিসেবে, এবং bottleneck-vs-attention তুলনা
7. Recap: Q/K/V হিসেবে পুনরায় লেবেল -> self-attention এবং recurrence-এর অবসানের পূর্বাভাস

## Further Reading

- Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning with Neural Networks*
- Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong, Pham, Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation*
- Jay Alammar, *Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)* (blog)