# NLP-এর সাথে পরিচিতি

**পর্ব:** [পূর্বশর্ত](../README.md) · **টপিক ফোল্ডার:** `03-Intro-to-NLP`

## কেন এটি গুরুত্বপূর্ণ

Neural network শুধু সংখ্যাই বোঝে। যেকোনো language model — একটি খেলনাসদৃশ bigram model থেকে GPT-4-আকারের systems — text নিয়ে কিছু করার আগে, সেই text-কে vector-এ পরিণত হতে হয়। আধুনিক subword tokenization ও learned embedding-এর দখল নেওয়ার *আগে* NLP কীভাবে text-কে উপস্থাপন করত, এই lesson তা নিয়ে আলোচনা করে (এগুলোর বিস্তারিত আলোচনা যথাক্রমে [Phase 01: Word Embeddings](../../Phase-01-Language-Modeling-Foundations/02-Word-Embeddings/README.md) ও [Phase 02: Tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md)-তে আছে)। প্রথমে ক্লাসিকাল পদ্ধতিগুলো বোঝা গেলে স্পষ্ট হয়ে যায় *কেন* embedding-গুলো এত বড় সাফল্য ছিল।

## এই পাঠে কী শেখা হবে

- Text preprocessing-এর মৌলিক বিষয় (এবং কেন আজ LLM-গুলো এর বেশিরভাগই বাদ দেয়)
- word-level-এ tokenization
- Bag-of-Words (BoW)
- TF-IDF
- One-hot encoding এবং এর সীমাবদ্ধতা
- distributional hypothesis (এই ধারণাটিই কোর্সের পরে প্রতিটি embedding পদ্ধতিকে অনুপ্রাণিত করে)
- text vectors তুলনার জন্য cosine similarity
- N-grams

## 1. Text preprocessing

ক্লাসিকাল NLP pipelines সাধারণত এই কাজগুলো করত:

- **Lowercasing** — "Cat" ও "cat" একই token হয়ে যায়
- **Punctuation/stopword removal** — কম তথ্যসমৃদ্ধ শব্দ বাদ দেওয়া ("the", "is", "a")
- **Stemming/lemmatization** — "running"/"ran"/"runs"-কে একটি সাধারণ root-এ ("run") নামিয়ে আনা

আধুনিক LLM-গুলো বেশিরভাগ ক্ষেত্রেই এই pipeline **বাদ দেয়**: subword tokenizer (Phase 02) raw text থেকে সরাসরি ঘনঘন আসা subword units শেখার মাধ্যমে casing ও morphology-কে নিহিতভাবে সামলায়, আর যথেষ্ট বড় model-ের জন্য stopwords-ও দরকারি signal বহন করে। এই pipeline জানা এখনও জরুরি — ঐতিহাসিক কারণেও, এবং কারণ এটি এখনও lightweight/classical NLP systems-এ (search, spam filter ইত্যাদি) ব্যবহৃত হয়।

## 2. Word-level tokenization

সবচেয়ে সহজ সম্ভব tokenizer: whitespace/punctuation দিয়ে ভাগ করা।

```
"The cat sat on the mat." -> ["the", "cat", "sat", "on", "the", "mat", "."]
```

এতে যে সমস্যাগুলো তৈরি হয়: বিশাল vocabulary, অদেখা শব্দ সামলানোর কোনো উপায় নেই ("out-of-vocabulary"), এবং সম্পর্কিত শব্দগুলোর মধ্যে কোনো ভাগ করা গঠন নেই ("run", "running", "runner" তিনটি সম্পর্কহীন token)। এই সমস্যাগুলো সমাধানের জন্যই subword tokenization (Phase 02)-এর অস্তিত্ব।

## 3. Bag-of-Words (BoW)

একটি document-কে এমন একটি vector হিসেবে উপস্থাপন করা যাতে প্রতিটি vocabulary word কতবার এসেছে তা গোনা হয়, order ও grammar সম্পূর্ণ উপেক্ষা করে:

```
vocab = ["the", "cat", "sat", "mat", "dog"]
"the cat sat on the mat" -> [2, 1, 1, 1, 0]
```

সহজ এবং spam detection-এর মতো কাজে আশ্চর্যজনকভাবে কার্যকর, কিন্তু word order সম্পূর্ণ ফেলে দেয় — "dog bites man" ও "man bites dog"-এর অর্থ খুব আলাদা, অথচ এদের BoW vectors একই রকম হতে পারে।

## 4. TF-IDF

Term Frequency–Inverse Document Frequency *প্রতিটি* document-এ আসা শব্দকে (তথ্যহীন) কম গুরুত্ব দেয় এবং একটি document-এ ঘন ঘন আসা অথচ পুরো corpus-এ বিরল শব্দকে (তথ্যবহুল) বেশি গুরুত্ব দেয়:

```
TF(t, d)  = count of term t in document d
IDF(t)    = log( N / (1 + document_count(t)) )     # N = total number of documents
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

"the"-এর মতো শব্দ প্রতিটি document-এ থাকে → `IDF ≈ 0` → প্রায় কিছুই অবদান রাখে না। একটি বিরল, topic-নির্দিষ্ট শব্দ উচ্চ IDF পায় এবং যে document-গুলোতে এটি আছে, সেগুলোর vector-এ প্রাধান্য বিস্তার করে।

## 5. One-hot encoding এবং এর সীমাবদ্ধতা

একটি শব্দকে one-hot vector হিসেবে উপস্থাপন — সেই শব্দের index-এ একটি মাত্র 1 বাদে সব শূন্য — vector-এর আকার vocabulary-এর আকার করে দেয় (সম্ভাব্য কয়েক লক্ষ dimension), এবং সবচেয়ে গুরুত্বপূর্ণভাবে, **যেকোনো দুটি ভিন্ন one-hot vector-এর dot product সবসময় 0**। "cat" ও "dog" যে "cat" ও "astronomy"-এর চেয়ে বেশি সম্পর্কিত — এমন কোনো ধারণাই থাকে না। dense embedding ঠিক এই সীমাবদ্ধতাটিই সমাধান করে।

## 6. distributional hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth, 1957

যেসব শব্দ একই রকম context-এ ঘটে, তাদের অর্থও একই রকম হওয়ার প্রবণতা থাকে। এই একটি মাত্র ধারণা পরের দিকে আলোচিত প্রতিটি embedding পদ্ধতির ভিত্তি: [Word2Vec, GloVe, এবং FastText](../../Phase-01-Language-Modeling-Foundations/02-Word-Embeddings/README.md) — সবগুলোই word co-occurrence মডেল করে vector শেখে, সরাসরি এই hypothesis-কে কার্যকর করে।

## 7. Cosine similarity

text একবার vector হলে, "এই দুটি text কতটা একই রকম" মাপার মানক উপায় হলো তাদের vectors-এর মাঝের কোণের cosine, raw distance নয় (যা document length-এর প্রতি সংবেদনশীল):

```
cos_sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

সীমা -1 (বিপরীত) থেকে 1 (অভিন্ন দিক); 0 মানে সম্পর্কহীন। embedding-ভিত্তিক semantic search ও attention scores সবই যে operation-টির উপর গড়ে ওঠে, সেটিই এটি — vectors যখন raw counts-এর বদলে অর্থ উপস্থাপন করতে শুরু করবে, তখনই এটি আবার দেখতে পাবে।

## 8. N-grams

BoW-এর "word order নেই" সমস্যার একটি সহজ সমাধান: একক শব্দের বদলে `n` শব্দের ধারাবাহিক sequences গোনা। "the cat sat"-এর bigrams: `["the cat", "cat sat"]`। কিছুটা স্থানীয় order ধরে রাখে, কিন্তু বিনিময়ে vocabulary অনেক বড় ও sparser হয়ে যায়।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "কম্পিউটার শুধু সংখ্যাই দেখে; সেখানে পৌঁছানোর pre-embedding উপায় এখানে"
2. Preprocessing pipeline-এর walkthrough
3. একটি ছোট 3-document উদাহরণে BoW ও TF-IDF derivation
4. One-hot-এর dot-product সমস্যা, code-এ লাইভ
5. distributional hypothesis — পরবর্তী phase-এর Word2Vec-এর সেতু
6. `example.py`-এর walkthrough
7. সংক্ষিপ্ত পুনরালোচনা + পরবর্তী PyTorch Fundamentals-এর দিকে ইঙ্গিত

## আরও পড়ুন

- Jurafsky & Martin, *Speech and Language Processing*, Ch. 2 (regex/tokenization) ও Ch. 6 (vector semantics)
- scikit-learn docs: `CountVectorizer`, `TfidfVectorizer`
- Firth, J.R. (1957), *A synopsis of linguistic theory* — distributional hypothesis-এর উৎস