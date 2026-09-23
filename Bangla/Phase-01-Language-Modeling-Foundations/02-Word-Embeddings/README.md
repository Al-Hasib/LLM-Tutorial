# Word Embedding কী

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `02-Word-Embeddings`

## কেন এটি গুরুত্বপূর্ণ

পূর্ববর্তী পাঠ শেষ হয়েছিল n-gram-এর মারাত্মক ত্রুটি দিয়ে: তারা "cat" ও "dog"-কে দুটি সম্পূর্ণ সম্পর্কহীন প্রতীক হিসেবে দেখে, তাই একটির সম্পর্কে শেখা কিছুই অপরটিতে স্থানান্তরিত হয় না। Word embedding হলো সেই সমাধান — raw text থেকে শেখা dense vector, এমনভাবে অবস্থিত যে একই ধরনের context-এ ব্যবহৃত শব্দ শেষ পর্যন্ত জ্যামিতিকভাবে কাছাকাছি হয়ে পড়ে। এই একক ধারণাটি (একটি discrete token-কে অর্থ ধারণকারী continuous space-এর একটি বিন্দুতে রূপান্তর করা) পরবর্তী সবকিছুর ভিত্তি: প্রতিটি Transformer-এর প্রথম layer এখনও, মূলত, একটি embedding lookup।

## এই পাঠে যা যা শেখা হবে

- কেন dense embedding one-hot vector-কে হারায় (recap + [Phase 00 §5-6](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#5-one-hot-encoding-and-its-limits)-এর সম্প্রসারণ)
- Word2Vec: CBOW ও Skip-gram
- Negative sampling (কেন পুরো vocabulary-র উপর নির্বোধভাবে softmax করা যায় না)
- GloVe: global co-occurrence statistics
- FastText: subword embedding এবং out-of-vocabulary সমস্যা
- Embedding arithmetic এবং এর সীমাবদ্ধতা
- কেন *static* embedding এখনও যথেষ্ট নয় (attention ও Transformer-এর সেতু)

## 1. One-hot থেকে dense vector-এ

50,000-শব্দের vocabulary-র জন্য একটি one-hot vector হলো 50,000 মাত্রার, প্রায় সবই শূন্য, এবং যেকোনো দুটি ভিন্ন শব্দের cosine similarity হুবহু `0` (দেখুন [Phase 00-এর demo](../../Phase-00-Prerequisites/03-Intro-to-NLP/example.py))। একটি **embedding** তার পরিবর্তে একটি ছোট dense vector (যেমন 100-300 মাত্রার) দেয়, যেখানে distance ও direction অর্থপূর্ণ। [Distributional hypothesis](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#6-the-distributional-hypothesis) আমাদের বলে কীভাবে কোনো manual labeling ছাড়াই এমন কিছু শেখা যায়: **এমন একটি model train করুন যা শব্দ থেকে context (বা context থেকে শব্দ) পূর্বাভাস করে, আর এর ভেতরে যা কিছু internal representation তৈরি করতে বাধ্য হয় তা উপ-পণ্য হিসেবে অর্থ ধারণ করতে দেখা যায়।**

## 2. Word2Vec: CBOW ও Skip-gram

দুটি আয়নার-প্রতিবিম্ব প্রশিক্ষণ উদ্দেশ্য, দুটোই Mikolov et al. (2013) থেকে:

- **CBOW (Continuous Bag-of-Words)**: পারিপার্শ্বিক context শব্দগুলো থেকে কেন্দ্রীয় শব্দ পূর্বাভাস করা।
- **Skip-gram**: কেন্দ্রীয় শব্দ থেকে প্রতিটি পারিপার্শ্বিক context শব্দ পূর্বাভাস করা (অভিজ্ঞতাগতভাবে ছোট dataset / বিরল শব্দে ভালো কাজ করে, এবং নিচে আমরা এটিই implement করি)।

কংক্রিটভাবে, একটি বাক্য এবং window size `w`-র জন্য, skip-gram `(center, context)` জোড়া তৈরি করে — যেমন `"the cat sat on the mat"` এবং window 2-এর জন্য, কেন্দ্রীয় শব্দ `"sat"` প্রশিক্ষণ জোড়া `(sat, cat)`, `(sat, on)`, `(sat, the)`, `(sat, mat)` তৈরি করে। Model দুটি embedding matrix শেখে — একটি শব্দ "center" হিসেবে, একটি শব্দ "context" হিসেবে — যেন সহ-ঘটমান (co-occurring) জোড়াগুলোর জন্য `center · context` বড় হয় এবং বাকিদের জন্য ছোট।

## 3. Negative sampling

`center · context` সাদৃশ্যকে probability-তে রূপান্তর করা স্বাভাবিকভাবে প্রতিটি প্রশিক্ষণ জোড়ার জন্য *পুরো vocabulary*-র উপর softmax প্রয়োজন করে — বাস্তব vocabulary size-এ গণনাগতভাবে দুর্বিষহ। **Negative sampling** প্রশিক্ষণকে পরিবর্তে সরল binary classification-এ পুনর্গঠিত করে: "(center, context) কি ডেটা থেকে আসা প্রকৃত জোড়া, না এলোমেলোভাবে নেওয়া জাল?" প্রতিটি প্রকৃত জোড়ার জন্য কয়েকটি এলোমেলো "negative" context শব্দ sample করুন এবং তাদের dot product নিচের দিকে ঠেলে দিন, সাথে প্রকৃত জোড়ার dot product উপরের দিকে। এটি প্রতি ধাপে O(vocab size) softmax-কে O(k) binary সিদ্ধান্তে রূপান্তর করে, যেখানে `k` ছোট (5-20)।

## 4. GloVe: global co-occurrence statistics

Word2Vec সর্বদা শুধু স্থানীয় window-গুলো দেখে, একবারে একটি জোড়া। **GloVe** (Pennington et al., 2014) তার পরিবর্তে প্রথমে পুরো corpus-এর উপর একটি পূর্ণ word-word co-occurrence matrix তৈরি করে, তারপর এটিকে factorize করে যেন দুটি word vector-এর dot product তাদের co-occurrence count-এর log আনুমানিক করে। মোটামুটিভাবে: Word2Vec অনেক ছোট স্থানীয় দৃষ্টিভঙ্গি থেকে শেখে; GloVe একটি বৈশ্বিক পরিসংখ্যান থেকে শেখে। বাস্তবে, উভয়েই তুলনামূলক মানের embedding উৎপাদন করে।

## 5. FastText: subword দিয়ে out-of-vocabulary শব্দের সমাধান

Word2Vec ও GloVe উভয়েই পুরো শব্দ প্রতি একটি অস্বচ্ছ vector দেয় — test-এর সময় একটি অদেখা শব্দে (একটি টাইপো, একটি নতুন পণ্যের নাম, একটি বিরল inflection) কেবল কোনো embedding থাকে না। **FastText** (Bojanowski et al., 2017) প্রতিটি শব্দকে তার **character n-gram** embedding-গুলোর যোগফল হিসেবে উপস্থাপন করে (যেমন `"running"` → `<ru, run, unn, nni, nin, ing, ng>` প্লাস পুরো শব্দটি)। আগে কখনো না দেখা শব্দটিও known শব্দের সাথে ভাগ করা যেকোনো subword অংশ যোগ করে embedding পেতে পারে — আর ভাগ করা morphology ("run", "runs", "running", "runner") স্বাভাবিকভাবেই সম্পর্কিত শব্দগুলোর vector কাছাকাছি টানে। এই subword ধারণাটি, সাধারণীকৃত রূপে, **byte-pair encoding tokenizer** হিসেবে আবার দেখা যায়, যা [Phase 02: Tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md)-এ আলোচিত।

## 6. Embedding arithmetic — এবং এর সীমাবদ্ধতা

বিখ্যাত demo: `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`। Embedding space-এর দিকনির্দেশনা শেষ পর্যন্ত সম্পর্ক encode করে (gender, tense, capital-of-country), শুধু নৈকট্য নয়। এটি বড় corpus-এ প্রশিক্ষিত embedding-এর উপর একটি বাস্তব, পুনরুৎপাদনযোগ্য ঘটনা — তবে এটি ভঙ্গুরও, এবং ছোট corpus-এ এটিকে অতিরঞ্জিত করা সহজ, যেমনটি নিচের `example.py` সৎভাবে দেখাবে।

## 7. কেন static embedding এখনও যথেষ্ট নয়

Word2Vec/GloVe/FastText প্রতি শব্দে **একটি নির্দিষ্ট vector** দেয়, context যাই হোক — "bank" শব্দটি "river bank" এবং "bank account" — দুটিতেই একই embedding পায়। এটি একটি প্রকৃত সীমাবদ্ধতা: অর্থ context-নির্ভর, আর এই পদ্ধতিগুলো তা উপস্থাপন করতে পারে না। এটি ঠিক করা — একটি শব্দের representation নির্ভর করা *নির্দিষ্ট বাক্যের* উপর যেখানে এটি দেখা দেয় — ঠিক এটিই Transformer-এর self-attention অর্জন করে, তাই ক্ষেত্রটি এই static embedding থেকে BERT ও GPT-র মতো model-গুলোর উৎপাদিত **contextual embedding**-এ চলে গেছে। সেখান থেকেই [Introduction to Transformers](../05-Intro-to-Transformers/README.md) এবং [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) শুরু হয়।

## Video Script Outline

1. Motivation — one-hot-এর অন্ধবিন্দুর recap: "যদি space-এ অবস্থান কিছু অর্থ বহন করত?"
2. CBOW বনাম skip-gram, স্ক্রিনে sliding-window pair-extraction উদাহরণসহ
3. Negative sampling — কেন প্রতি ধাপে 50,000 শব্দে শুধু softmax করলেই চলে না
4. GloVe ও FastText, Word2Vec-এর বিপরীতে
5. `example.py`-র walkthrough — ছোট্ট skip-gram embedding train করুন, nearest neighbors পরীক্ষা করুন
6. Recap: dense vector "no generalization" সমস্যা সমাধান করে, কিন্তু তারা এখনও *static* — contextual embedding-এর পূর্বাভাস

## Further Reading

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* (Word2Vec)
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases and their Compositionality* (negative sampling)
- Pennington, Socher, Manning (2014), *GloVe: Global Vectors for Word Representation*
- Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (FastText)