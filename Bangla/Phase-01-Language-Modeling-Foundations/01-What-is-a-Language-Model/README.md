# Language Model কী

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `01-What-is-a-Language-Model`

## কেন এটি গুরুত্বপূর্ণ

এই পুরো কোর্সের প্রতিটি জিনিস — একটি 7B-parameter chatbot-ও এর মধ্যে — একটি language model, আর প্রতিটি language model ঠিক একটি মাত্র কাজ করে: শব্দের (বা token-এর) একটি sequence-কে একটি probability দেওয়া, এবং সেই probability দিয়ে অনুমান করা পরবর্তীতে কী আসবে। Neural network-এ হাত দেওয়ার আগে, এই ধারণাটিকে তার সবচেয়ে সহজ সম্ভব রূপে দেখা দরকার — একটি corpus-এ শব্দ গুনে ফেলা — কারণ সেই সহজ রূপের শক্তিগুলো এবং, আরও গুরুত্বপূর্ণভাবে, *ব্যর্থতাগুলোই* হল সেই মূল কারণ যা এই কোর্সের বাকি অংশের প্রতিটি neural কৌশলকে অনুপ্রাণিত করে।

## এই পাঠে যা যা শেখা হবে

- Language model-এর আনুষ্ঠানিক সংজ্ঞা
- N-gram model এবং Markov assumption
- Corpus থেকে count করে probability অনুমান করা
- Zero-probability সমস্যা এবং smoothing
- Perplexity: একটি language model ভালো কিনা তা আমরা কীভাবে মাপি
- An n-gram model থেকে sampling করে text তৈরি করা
- N-gram কেন ভেঙে পড়ে, এবং পরবর্তী সবকিছুর জন্য তার অর্থ কী

## 1. Language model আসলে কী

একটি language model যেকোনো শব্দ sequence-কে `w₁, w₂, ..., w_T` একটি probability দেয়। Probability-র chain rule ব্যবহার করলে, সেই যৌথ probability ভেঙে যায় পরবর্তী-শব্দ পূর্বাভাসের গুণফল:

```
P(w₁, w₂, ..., w_T) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(w_T|w₁,...,w_{T-1})
```

ব্যস, এটাই — পুরো ধারণাটা এটুকুই। `P(next word | everything so far)` ভালোভাবে অনুমান করার জন্য একটি model-কে train করুন, আর আপনি পারবেন: যেকোনো বাক্য কতটা সম্ভবপর তা score করা, এবং সেই next-word distribution থেকে বারবার sampling করে নতুন text তৈরি করা। এই কোর্সের প্রতিটি architecture (n-grams, RNNs, Transformers) হলো `P(w_t | w₁, ..., w_{t-1})` আনুমানিক করার একটি ভিন্ন উপায় মাত্র।

## 2. N-gram model এবং Markov assumption

*সম্পূর্ণ* পূর্ববর্তী ইতিহাসের উপর শর্ত বসানো ডেটা থেকে অনুমান করা কার্যত অসম্ভব — সম্ভবপর ইতিহাস প্রায় অসীম। **Markov assumption** এটি সরল করে: ধরে নাও পরবর্তী শব্দটি শুধু পূর্ববর্তী `n-1`টি শব্দের উপর নির্ভর করে।

- **Bigram model** (n=2): `P(w_t | w₁,...,w_{t-1}) ≈ P(w_t | w_{t-1})`
- **Trigram model** (n=3): `P(w_t | w₁,...,w_{t-1}) ≈ P(w_t | w_{t-2}, w_{t-1})`

## 3. Count করে probability অনুমান করা (MLE)

একটি training corpus দেওয়া থাকলে, একটি bigram probability-র maximum-likelihood estimate হলো শুধুই আপেক্ষিক কম্পাঙ্ক:

```
P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
```

জোড়াটি কতবার ঘটেছে তা count করুন, প্রথম শব্দটি মোট কতবার ঘটেছে তা দিয়ে ভাগ করুন।

## 4. Zero-probability সমস্যা এবং smoothing

যদি কোনো শব্দ জোড়া training-এ কখনোই দেখা না যায়, তার count হয় `0`, তাই model তাকে probability `0` দেয় — আর যেকোনোকিছুকে `0` দিয়ে গুণ করলে *পুরো বাক্যটিই* অসম্ভব হয়ে পড়ে, এমনকি যদি শুধু একটি bigram-ও অদেখা থাকে। যেকোনো বাস্তব text-এর জন্য এটি অত্যন্ত ভঙ্গুর। **Laplace (add-one) smoothing** এটি ঠিক করে এই ভান করে যে প্রতিটি সম্ভাব্য জোড়া অন্তত একবার ঘটেছে:

```
P(w_t | w_{t-1}) = (count(w_{t-1}, w_t) + 1) / (count(w_{t-1}) + V)
```

যেখানে `V` হলো vocabulary size। আধুনিক smoothing পদ্ধতির (Kneser-Ney, ইত্যাদি) তুলনায় এটি মোটামুটি, তবে মূল ধারণাটি একই — model-কে কখনোই এমন কোনো কিছুতে ঠিক শূন্য probability দিতে দেওয়া যাবে না যা সে কেবল দেখেনি — এই ধারণাটি বারবার ফিরে আসে, যেমন LLM training-এর সময় label smoothing-এ।

## 5. Perplexity

Perplexity হলো language model score করার প্রমিত উপায়, যা প্রতি token-এর exponentiated average negative log-likelihood হিসেবে সংজ্ঞায়িত:

```
PPL = exp( -(1/T) Σ log P(w_t | w_{t-1}) )
```

স্বাভাবিকভাবে: perplexity হল "প্রতিটি ধাপে model মনে করে সে কতগুলো সম-সম্ভাব্য পছন্দের মধ্যে থেকে বেছে নিচ্ছে তার গড় সংখ্যা।" একটি নিখুঁত model, যা সঠিক পরবর্তী শব্দকে সর্বদা probability `1` দেয়, তার `PPL = 1`। একটি সম্পূর্ণ বিপথগামী model, যা `V`টি vocabulary শব্দের মধ্যে সমানভাবে অনুমান করে, তার `PPL = V`। **কম হলে ভালো।** এই হুবহু metric-টি (শুধু n-gram count-এর পরিবর্তে neural model-এর probability দিয়ে গণনা করা) এখনও প্রতিটি LLM-এর জন্য report করা হয়, যা দেখতে পাবেন [Phase 08: Evaluation Metrics](../../Phase-08-Evaluation-of-LLMs/01-Evaluation-Metrics/README.md)-এ।

## 6. An n-gram model থেকে text তৈরি করা

একবার প্রতিটি শব্দের জন্য `P(w_t | w_{t-1})` পেলে, আপনি text তৈরি করতে পারেন: একটি seed word দিয়ে শুরু করুন, তার conditional distribution থেকে পরবর্তী শব্দ sample করুন, তারপর পুনরাবৃত্তি করুন — প্রতিটি sampled শব্দকে আবার নতুন context হিসেবে feed করুন। এই sample-then-feed-back লুপটি *ঠিক* সেই উপায়, যেভাবে এই কোর্সের প্রতিটি LLM inference-এর সময় text তৈরি করে; শুধু যে probability distribution থেকে sampling করা হয় তা আরও পরিশীলিত হয়ে ওঠে।

## 7. N-gram কেন ভেঙে পড়ে

- **Sparsity**: বেশিরভাগ যুক্তিসঙ্গত trigram একটি বিশাল corpus-েও কখনো দেখা যায় না, তাই count-গুলো অবিশ্বস্ত — আরও context ধরার জন্য 4-gram বা 5-gram-এ গেলে sparsity বিপর্যয়করভাবে আরও খারাপ হয়।
- **No generalization**: model-টির কোনো ধারণা নেই যে "cat" এবং "dog" একই ধরনের প্রাণী — `P(w | "the cat")` এবং `P(w | "the dog")` সম্পূর্ণ স্বাধীনভাবে estimate করা হয়, cat ও dog শব্দার্থগতভাবে যতই একই হোক না কেন। এটি [Phase 00: Introduction to NLP](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#5-one-hot-encoding-and-its-limits)-এর সেই হুবহু one-hot-encoding অন্ধবিন্দু।
- **নির্দিষ্ট, ক্ষুদ্র context window**: প্রকৃত ভাষা 2-3 শব্দের window-এর অনেক বাইরের শব্দের উপর নির্ভর করে, যা n-gram গঠনগতভাবেই দেখতে পারে না।

এই তিনটি ব্যর্থতাই এই phase-এর বাকি অংশকে অনুপ্রাণিত করে: [Word Embeddings](../02-Word-Embeddings/README.md) "no generalization" সমস্যাটি ঠিক করে, আর [RNNs](../03-RNN-LSTM-GRU/README.md) এবং শেষ পর্যন্ত [Transformers](../05-Intro-to-Transformers/README.md) "নির্দিষ্ট, ক্ষুদ্র context" সমস্যাটি ঠিক করে।

## Video Script Outline

1. Motivation — "একটি language model হলো শুধুই next-word prediction; এখানে সবচেয়ে সহজ সংস্করণ যা কাজ করতে পারে"
2. Probability-র chain rule -> Markov assumption -> bigram/trigram model
3. একটি বাস্তব উদাহরণে counting-ভিত্তিক estimation, তারপর zero-probability প্রাচীরে ধাক্কা
4. Smoothing এটি ঠিক করে, perplexity এটি মাপে
5. `example.py`-র walkthrough — একটি bigram model train, score এবং generate করুন
6. Recap: তিনটি নির্দিষ্ট ব্যর্থতার ধরন -> ঠিক কোন ভবিষ্যৎ পাঠ কোনটি ঠিক করবে তার পূর্বাভাস

## Further Reading

- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 (N-gram Language Models)
- Chen & Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*
- Shannon (1948), *A Mathematical Theory of Communication* — ভাষার জন্য entropy/perplexity-জাতীয় পরিমাণ ব্যবহারের উৎস