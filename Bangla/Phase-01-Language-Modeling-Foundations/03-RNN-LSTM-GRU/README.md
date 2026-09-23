# RNN, LSTM ও GRU

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `03-RNN-LSTM-GRU`

## কেন এটি গুরুত্বপূর্ণ

N-gram (পাঠ 1) ব্যর্থ হয় কারণ তাদের context window নির্দিষ্ট ও ক্ষুদ্র। Word embedding (পাঠ 2) *কোনো শব্দের অর্থ কী* তা ঠিক করে কিন্তু *ক্রম বা sequence* সম্পর্কে কিছু বলে না। Recurrent Neural Network ছিল field-এর প্রথম গুরুতর প্রচেষ্টা এমন একটি network-এর, যা নির্বিচারে দীর্ঘ sequence ধরে একটি memory বহন করে — এবং প্রায় এক দশক ধরে (2014-2017-এর কাছাকাছি) তারা প্রতিটি গুরুতর NLP সিস্টেমের মেরুদণ্ড ছিল, Google Translate-সহ। কেন তারা শেষ পর্যন্ত Transformer-এর কাছে হেরে গেল (তাদের sequential, ধাপে-ধাপে প্রকৃতি) তা হুবহু বোঝাই [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md)-তে self-attention-এর লাভকে বোধগম্য করে তোলে।

## এই পাঠে যা যা শেখা হবে

- Vanilla RNN: recurrence, সময়ের মধ্যে unrolling
- Backpropagation Through Time (BPTT) এবং কেন এটি ব্যয়বহুল/sequential
- Vanishing/exploding gradient সমস্যা
- LSTM: cell state এবং এর তিনটি gate
- GRU: LSTM-এর একটি সরু বিকল্প
- কেন recurrence নিজেই — শুধু gradient সমস্যা নয় — যা Transformer শেষ পর্যন্ত সরিয়ে দেয়

## 1. Vanilla RNN

একটি RNN একটি sequence-কে একবারে একটি উপাদান প্রক্রিয়া করে, একটি **hidden state** `h_t` বজায় রেখে যা এখন পর্যন্ত দেখা সবকিছুর সারাংশ ধারণ করে:

```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
y_t = W_hy h_t + b_y
```

*একই* weight matrix (`W_xh`, `W_hh`, `W_hy`) প্রতিটি timestep-এ পুনর্ব্যবহৃত হয় — এই weight sharing-ই RNN-কে নির্দিষ্ট সংখ্যক parameter দিয়ে যেকোনো দৈর্ঘ্যের sequence পরিচালনা করতে দেয়। `T` timestep ধরে recurrence-কে "unroll" করলে তা একটি অতি-গভীর feedforward network-এ (depth = sequence length) পরিণত হয়, যার প্রতিটি layer-এ tied weight থাকে।

## 2. Backpropagation Through Time (BPTT)

RNN train করা মানে প্রতিটি unrolled timestep-এর মধ্য দিয়ে loss-কে backward করা, recurrence-এর মাধ্যমে বারবার chain rule প্রয়োগ করা ([Phase 00 §3](../../Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md#3-calculus))। এর থেকে সরাসরি দুটি পরিণতি ঝরে পড়ে:

1. **এটি সহজাতভাবে sequential।** আগে `h_{t-1}` না পেলে `h_t` গণনা করা যায় না — training বা inference-এর সময় timestep জুড়ে parallelize করার কোনো উপায় নেই। এই একক সত্যই Transformer-এর শেষ পর্যন্ত RNN-কে প্রতিস্থাপনের প্রধান কারণ: আধুনিক GPU parallel কাজের জন্য তৈরি, আর একটি sequential dependency chain সেই ক্ষমতার প্রায় পুরোটাই নষ্ট করে।
2. **Gradient বারবার একই Jacobian দিয়ে গুণিত হয়।** প্রাথমিক timestep-এ ফিরে যাওয়া gradient প্রতিটি মধ্যবর্তী timestep-এ একটি `tanh'(·) · W_hh` গুণকের মধ্য দিয়ে গেছে।

## 3. Vanishing ও exploding gradient

যেহেতু প্রতিটি timestep-এ একই গুণক গুণিত হয়, gradient একটি geometric series-এর মতো আচরণ করে:

- যদি সেই গুণকের মান ধারাবাহিকভাবে `< 1` হয় (খুব সাধারণ — `tanh'` সর্বোচ্চ `1` এবং প্রায়ই এর চেয়ে অনেক ছোট), gradient sequence দৈর্ঘ্যের সাথে **exponentially** সঙ্কুচিত হয়: **vanishing gradient problem**। বাস্তবে, এর অর্থ প্রায় 10-20 ধাপের বেশি দূরত্বের dependency একটি vanilla RNN মূলত শিখতে পারে না — একটি অনুচ্ছেদের শুরুতে থাকা শব্দের শেষে পৌঁছানো training signal-এ কোনো প্রকৃত প্রভাব থাকে না।
- যদি সেই গুণকের মান ধারাবাহিকভাবে `> 1` হয়, gradient **exponentially** বেড়ে যায়: **exploding gradient problem**, যা বাস্তবে সাধারণত gradient clipping দিয়ে ঠিক করা হয়।

`example.py` এটি সরাসরি মাপে: ক্রমবর্ধমান sequence দৈর্ঘ্যের জন্য, প্রথম input-এ একটি ক্ষুদ্র নাড়ার (nudge) প্রতি final hidden state আসলে কতটা পরিবর্তিত হয় তা এটি গণনা করে।

## 4. LSTM: gate এবং একটি সুরক্ষিত cell state

The Long Short-Term Memory cell (Hochreiter & Schmidhuber, 1997) একটি দ্বিতীয় recurrent পরিমাণ যোগ করে vanishing gradient ঠিক করে — **cell state** `c_t` — যার মধ্য দিয়ে তথ্য অনেক কম multiplicative ক্ষয় নিয়ে প্রবাহিত হতে পারে, তিনটি শেখা **gate** দ্বারা নিয়ন্ত্রিত (প্রতিটি একটি sigmoid, তাই এটি `[0, 1]`-এ মান বের করে, একটি নরম on/off সুইচ হিসেবে কাজ করে):

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # forget gate: পুরনো cell state-এর কতটুকু রাখতে হবে
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # input gate: কত নতুন তথ্য লিখতে হবে
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # candidate নতুন বিষয়বস্তু
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         # cell state আপডেট — বেশিরভাগই additive, multiplicative নয়!
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # output gate: cell state-এর কতটুকু প্রকাশ করতে হবে
h_t = o_t ⊙ tanh(c_t)
```

মূল অন্তর্দৃষ্টি: `c_t`-এর আপডেট **additive** (`f_t ⊙ c_{t-1} + ...`), vanilla RNN-এর hidden state-এর মতো পুনরাবৃত্ত matrix multiplication নয়। যখন forget gate `f_t` `1`-এর কাছাকাছি থাকে, gradient অনেক timestep পিছিয়ে স্বল্প-পরিবর্তিতভাবে প্রবাহিত হতে পারে — `example.py` ঠিক এটিই সংখ্যাগতভাবে প্রদর্শন করে।

## 5. GRU: একটি সরল বিকল্প

The Gated Recurrent Unit (Cho et al., 2014) forget ও input gate-কে একটি একক **update gate**-এ একীভূত করে, আলাদা cell state সম্পূর্ণ বাদ দেয়, সবকিছু hidden state-এ ভাঁজ করে:

```
z_t = σ(W_z · [h_{t-1}, x_t])                          # update gate
r_t = σ(W_r · [h_{t-1}, x_t])                          # reset gate
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])                 # candidate hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t                   # পুরনো ও নতুনের মিশ্রণ
```

LSTM-এর চেয়ে কম parameter, প্রায়ই তুলনামূলক কর্মক্ষমতা, এবং এটি 2010-এর দশকের মাঝামাঝি একটি জনপ্রিয় ডিফল্ট ছিল। মূল কৌশলটি — খাঁটি matrix-multiply recurrence-এর বদলে একটি gated, প্রধানত-additive আপডেট — আত্মায় LSTM-এর মতোই।

## 6. Transformer আসলে কী সরিয়ে দেয়

ভাবতে লোভনীয় যে "LSTM RNN সমস্যা সমাধান করেছে," কিন্তু তারা কেবল *gradient* সমস্যাই সমাধান করেছে — **sequential bottleneck** (token `t-1`-কে token `t`-এর আগে প্রক্রিয়া করতে হবে) LSTM ও GRU-তেও থেকে যায়। সেই bottleneck-ই internet-scale ডেটায় প্রশিক্ষণকে recurrent model-এর জন্য নিষিদ্ধমূলক ধীর করে তুলেছিল, আর এটিই সেই নির্দিষ্ট জিনিস যা self-attention সরিয়ে দেয়: প্রতিটি position **সমান্তরালে** প্রক্রিয়া করা যায়, সরাসরি অন্য প্রতিটি position-কে দেখে, কোনো recurrence ছাড়াই। সেটিই পরবর্তী: [Sequence-to-Sequence and Attention](../04-Seq2Seq-and-Attention/README.md)।

## Video Script Outline

1. Motivation — "embedding আমাদের অর্থ দিল, এখন আমাদের দরকার sequence জুড়ে memory"
2. Vanilla RNN recurrence + unrolling diagram
3. BPTT এবং sequential-dependency সমস্যা
4. Vanishing gradient — geometric-series অন্তর্দৃষ্টি তৈরি করুন
5. LSTM gate, cell state, gradient প্রবাহের জন্য "কেন additive multiplicative-কে হারায়"
6. GRU সরু চাচাতো ভাই হিসেবে
7. `example.py`-র walkthrough — RNN বনাম LSTM-এ gradient ক্ষয় সরাসরি মাপুন
8. Recap: gradient ঠিক হয়েছে, কিন্তু recurrence নিজেই রয়ে গেছে -> attention-এর পূর্বাভাস

## Further Reading

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (GRU)
- Christopher Olah, *Understanding LSTM Networks* (colah.github.io) — আদর্শ চাক্ষুষ ব্যাখ্যাকারী
- Pascanu, Mikolov, Bengio (2013), *On the difficulty of training Recurrent Neural Networks* (vanishing/exploding gradients, gradient clipping)