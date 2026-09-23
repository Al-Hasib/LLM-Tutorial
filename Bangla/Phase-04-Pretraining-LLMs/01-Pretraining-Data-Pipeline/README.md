# Pretraining Data Pipeline

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `01-Pretraining-Data-Pipeline`

## কেন এটি গুরুত্বপূর্ণ

এই Phase-এর আগের প্রতিটি পাঠে ধরে নেওয়া হয়েছিল যে পরিষ্কার, ব্যবহারের জন্য প্রস্তুত text ইতিমধ্যেই বিদ্যমান — [Phase 01-এর bigram model](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md), [BPE tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md) এবং [mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) — সবাই ছোট, হাতে-বাছাই করা corpus-এ প্রশিক্ষিত হয়েছিল। বাস্তব pretraining corpus তৈরি হয় ইন্টারনেটের শত শত টেরাবাইট কাঁচা text থেকে, আর সেই কাঁচা text-কে এমন রূপে আনা যাতে তাতে [Scaling-Laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md)-উপযোগী compute খরচ করা যায় — এটি নিজেই এক বিশাল engineering প্রচেষ্টা; LLM তৈরির সবচেয়ে কম গ্ল্যামারাস, অথচ সবচেয়ে ফলপ্রসূ অংশ বলা যায়। খারাপ data শুধু compute নষ্ট করে না; এটি মডেলকে সক্রিয়ভাবে খারাপ অভ্যাসও শেখায় (boilerplate নকল করা, হুবহু string মুখস্থ করা, বা আরও খারাপ — ঠিক সেই benchmark প্রশ্নগুলো মুখস্থ করে ফেলা, যেগুলো দিয়ে মডেলকে পরে মূল্যায়ন করা হয়)। এই পাঠটি চারটি পাঠের মধ্যে প্রথম, যা একসাথে ব্যাখ্যা করে [Lesson 5-এর capstone training run](../05-Pretraining-a-Small-LLM-From-Scratch/README.md)-এর *আগে* যা কিছু ঘটে।

## এই পাঠে কী শেখানো হয়

- pretraining text আসলে কোথা থেকে আসে (Common Crawl এবং অন্যান্য উৎস)
- কাঁচা, জটিল HTML থেকে ব্যবহারযোগ্য text বের করা
- Quality filtering: heuristic নিয়ম এবং প্রশিক্ষিত classifier
- Deduplication: exact (hashing) ও near-duplicate (MinHash/LSH) সনাক্তকরণ, এবং duplicate কেন সক্রিয়ভাবে ক্ষতিকর
- Data mixing: কেন উৎসের *অনুপাত* মোট token সংখ্যার মতোই গুরুত্বপূর্ণ

## 1. Text কোথা থেকে আসে

আধুনিক LLM pretrain করার জন্য কোনো একক উৎসে যথেষ্ট প্রাকৃতিক উচ্চ-মানের text নেই, তাই বাস্তব pipeline কয়েকটি উৎস মিশিয়ে কাজ করে:

- **Common Crawl** — ক্রলযোগ্য ওয়েবের একটি বড় অংশের পাবলিক, ক্রমাগত আপডেট হওয়া snapshot (পেটাবাইট-পরিমাণ কাঁচা HTML/text); প্রায় প্রতিটি বড় pretraining corpus-এর মেরুদণ্ড (GPT-3-এর dataset, The Pile, RefinedWeb, FineWeb ইত্যাদি)। এটি বিশাল, কিন্তু অত্যন্ত noisy — spam, স্বয়ংক্রিয়-উৎপন্ন পেজ, boilerplate এবং অ-ভাষা বিষয়বস্তু কাঁচা crawl-এ প্রাধান্য পায়।
- **Curated উচ্চ-মানের উৎস** — Wikipedia, বই (Project Gutenberg, Books3-ধরনের corpus), academic paper (arXiv, PubMed) এবং code (GitHub)। এগুলো কাঁচা byte-সংখ্যায় Common Crawl-এর চেয়ে অনেক ছোট, কিন্তু প্রতি token-এ অসামঞ্জস্যপূর্ণভাবে বেশি মূল্যবান: পরিচ্ছন্ন গদ্য, ঘন factual বিষয়বস্তু, এবং (code-এর ক্ষেত্রে) সম্পূর্ণ ভিন্ন, অত্যন্ত কাঠামোবদ্ধ modality, যা code নয় এমন কাজেও reasoning পরিমাপযোগ্যভাবে উন্নত করে।
- **ফোরাম ও সোশ্যাল/QA ডেটা** (Reddit, StackExchange) — কথোপকথন ও প্রশ্ন-উত্তরের কাঠামো, যা ওয়েবের গদ্য বা বই প্রদান করে না।

ইন্টারনেটে "প্রতিটি উৎস কতটা আছে" তার কাঁচা অনুপাত *নয়* সেই অনুপাত, যাতে প্রশিক্ষণ দিতে চান — দেখুন §5।

## 2. Text extraction ও cleaning

কাঁচা Common Crawl record হলো একটি HTML page, পরিচ্ছন্ন গদ্য নয়। পরের যেকোনো ধাপ শুরু করার আগে, আপনাকে শুধু article/content-এর text-টা বের করে আনতে হবে:

- **Boilerplate removal** — নেভিগেশন মেনু, header/footer, cookie banner, বিজ্ঞাপন এবং সাইডবার link বাদ দিয়ে শুধু মূল content block রাখা। `trafilatura` ও `jusText`-এর মতো টুল HTML DOM-এর উপর heuristic দিয়ে এটা করে (text density, tag structure, প্রতি block-এ link density)।
- **Encoding ও language normalization** — character-encoding ভুল ঠিক করা, Unicode স্বাভাবিক করা, এবং language identification চালানো (যেমন `fastText`-এর language classifier) যাতে ডকুমেন্টগুলো per-language corpus-এ যায় অথবা যে ভাষা চান না সেগুলো বাদ দেওয়া যায়।
- **Markup stripping** — boilerplate removal-এ বাদ না পড়া অবশিষ্ট HTML tag, JavaScript এবং CSS সরিয়ে ফেলা।

শুধু এই ধাপটিই সাধারণত quality filtering শুরুর আগেই কাঁচা crawl-এর বেশিরভাগ byte ফেলে দেয়।

## 3. Quality filtering

Boilerplate removal-এর পরেও বেশিরভাগ অবশিষ্ট ডকুমেন্ট কম মূল্যের: SEO spam, স্বয়ংক্রিয়-উৎপন্ন product listing, keyword-stuffed পেজ, অথবা শুধুই খুব ছোট fragment। দুটি পরিপূরক filtering কৌশল একসাথে ব্যবহার করা হয়:

**Heuristic filter** — সস্তা, নিয়ম-ভিত্তিক চেক যা প্রতিটি ডকুমেন্টে প্রয়োগ করা হয়; উদাহরণস্বরূপ (CCNet, Gopher/MassiveText এবং RefinedWeb-এর ব্যবহৃত recipe অনুসরণ করে):

- ন্যূনতম word/token সংখ্যার নিচে থাকা ডকুমেন্ট বাতিল (দরকারি signal বহন করার জন্য খুব ছোট)
- অস্বাভাবিকভাবে উচ্চ symbol-to-word অনুপাতের ডকুমেন্ট বাতিল (অতিরিক্ত punctuation, emoji, বা markup অবশিষ্টাংশ spam বা extraction ব্যর্থতা নির্দেশ করে)
- অতিরিক্ত line- বা n-gram-স্তরের পুনরাবৃত্তি থাকা ডকুমেন্ট বাতিল (template পেজ, extraction-এ বেঁচে যাওয়া স্বয়ংক্রিয়-উৎপন্ন boilerplate)
- ন্যূনতম stop-word অনুপাত বা যুক্তিসঙ্গত word-length distribution দাবি করুন (একটি মোটামুটি কিন্তু কার্যকর "এটা কি সত্যিকারের গদ্যের মতো দেখাচ্ছে" চেক)

**Classifier-ভিত্তিক filtering** — একটি হালকা classifier প্রশিক্ষণ দিন (ঐতিহাসিকভাবে একটি `fastText` linear classifier, কারণ এটি পুরো crawl-এর উপর চালানোর পক্ষে যথেষ্ট সস্তা) যা "উচ্চ-মানের" text-কে "সাধারণ ওয়েব text" থেকে আলাদা করবে; পরিচিত-ভালো উৎসগুলোকে (Wikipedia, curated বই, OpenWebText-ধরনের মানুষের-কিউরেট করা link) positive example এবং এলোমেলো crawl ডকুমেন্টকে negative example হিসেবে ব্যবহার করে। এরপর প্রতিটি crawl ডকুমেন্ট একটি quality score পায়, এবং pipeline শুধু threshold-এর উপরের ডকুমেন্টগুলোই রাখে। এটি heuristic-এর চেয়ে স্পষ্টভাবে বেশি শক্তিশালী, কারণ এটি এমন সূক্ষ্ম stylistic signal ধরতে পারে যা কোনো হাতে-লেখা নিয়ম ধরে না — তবে এর জন্য labeled reference data লাগে, এবং এটি ততটাই ভালো, যতটা বোঝা "উচ্চ-মান" সেই reference set-এ বেক করা থাকে।

## 4. Deduplication

ওয়েব অসাধারণভাবে পুনরাবৃত্তিমূলক: একই article কয়েক ডজন news mirror-এ syndicate হয়, boilerplate আইনি text (privacy policy, license header) লক্ষ লক্ষ পেজে প্রায়-হুবহু হাজির থাকে, আর ফোরাম থ্রেড বারবার quote-and-requote হয়। প্রশিক্ষণ set-এ রেখে দিলে duplicate তিনটি স্বতন্ত্র সমস্যা তৈরি করে:

- **Memorization** — প্রশিক্ষণের সময় মডেল একই string বহুবার দেখে, ফলে generalization-এর বদলে সেই string হুবহু মুখস্থ করে পরে পুনরুৎপাদন করার সম্ভাবনা অনেক বেড়ে যায় (এটি বাস্তব privacy ও copyright উদ্বেগ)।
- **নষ্ট compute** — প্রতিটি duplicate token-এর তবুও একটি forward/backward pass খরচ হয়; একই বিষয়বস্তুতে দশবার প্রশিক্ষণ দশ গুণ শেখার signal দেয়, তার চেয়ে অনেক কম।
- **Benchmark contamination** — কোনো test benchmark-এর প্রশ্নের duplicate (বা near-duplicate) যদি ওই benchmark-এর কোনো crawl করা কপির মাধ্যমে প্রশিক্ষণ set-এ ঢুকে পড়ে, তাহলে evaluation score অর্থহীন হয়ে যায় (মডেল সমস্যাটা *সমাধান* করেনি, সে *উত্তরের চাবিকাঠি মুখস্থ* করেছে)।

দুটি dedup কৌশল ভিন্ন ভিন্ন granularity-তে কাজ করে:

**Exact deduplication** — প্রতিটি ডকুমেন্টকে (বা সূক্ষ্ম granularity-র জন্য প্রতিটি line-কে) দ্রুত hash function দিয়ে hash করুন (যেমন SHA-1) এবং হুবহু hash collision বাদ দিন। সস্তা এবং সম্পূর্ণ নির্ভুল, কিন্তু শুধুমাত্র byte-অভিন্ন duplicate-ই ধরে — এটি অনেক বেশি সাধারণ *near*-duplicate-এর ঘটনা মিস করে: একই article যাতে আলাদা একটা বিজ্ঞাপন ঢোকানো হয়েছে, একটা paragraph সম্পাদনা করে আবার প্রকাশ করা সংস্করণ, অথবা শুধু product name বদলানো টেমপ্লেট পেজ।

**MinHash + LSH দিয়ে near-duplicate সনাক্তকরণ** — near-duplicate ধরতে প্রতিটি ডকুমেন্টকে ওভারল্যাপিং word (বা character) **shingle**-এর সেট (n-gram) হিসেবে উপস্থাপন করুন, এবং দুটি ডকুমেন্টের shingle set-এর মধ্যে **Jaccard similarity** আনুমানিক করুন:

```
Jaccard(A, B) = |A intersect B| / |A union B|
```

বিলিয়ন-ডকুমেন্টের corpus-এ প্রতিটি জোড়া ডকুমেন্টের জন্য এটি হুবহু গণনা করা অসম্ভব (corpus-এর আকারে চতুর্মাত্রিক, quadratic)। **MinHash** এটিকে সম্ভব করে: একটি ডকুমেন্টের shingle set-এ অনেকগুলো স্বাধীন hash function প্রয়োগ করুন, প্রতিটি hash function-এর জন্য শুধু *ন্যূনতম* hashed মানটিকে সেই function-এর জন্য ডকুমেন্টের "signature" হিসেবে রাখুন, এবং `k`টি স্বাধীন hash function-এর জন্য পুনরাবৃত্তি করে একটি `k`-উপাদানের signature vector তৈরি করুন। যে মূল গাণিতিক ধর্মটি একে কার্যকর করে:

```
P( minhash_h(A) == minhash_h(B) ) = Jaccard(A, B)
```

একটি একক এলোমেলো hash function `h`-এর জন্য। অর্থাৎ দুটি ডকুমেন্টের `k`-উপাদানের MinHash signature-এর *ম্যাচিং অবস্থানের ভগ্নাংশ* তাদের প্রকৃত Jaccard similarity-র একটি unbiased estimator — দুটি ছোট signature থেকে গণনা করা, দুটি সম্পূর্ণ shingle set থেকে নয়। প্রোডাকশনে এটিকে আরও দ্রুত করা হয় **Locality-Sensitive Hashing (LSH)** দিয়ে: MinHash signature-এর band অনুযায়ী ডকুমেন্টগুলোকে bucket করুন, যাতে শুধু এমন ডকুমেন্টেই তুলনা করা হয় যারা কোনো একটি band-এ ইতিমধ্যেই একমত — সম্পূর্ণ quadratic pairwise তুলনা এড়িয়ে যায়। `example.py` সরাসরি MinHash signature ও Jaccard estimation বাস্তবায়ন করে, এবং দেখায় যে exact-hash dedup যে near-duplicate pair-কে ধরতে ব্যর্থ হয়, MinHash সেটিকে সঠিকভাবে চিহ্নিত করে।

## 5. Data mixing

বেশ কিছু উৎস থেকে cleaning ও deduplication-এর পরে বড় একটি pool পাওয়া গেলে, শেষ সিদ্ধান্তটি হলো: **প্রশিক্ষণ মিশ্রণের কত ভগ্নাংশ প্রতিটি উৎস থেকে আসা উচিত?** এটি মোট token সংখ্যার মতোই গুরুত্বপূর্ণ, দুটি কারণে:

- **উৎসভেদে value density ব্যাপকভাবে ভিন্ন।** Wikipedia বা peer-reviewed paper-এর একটি token গড়ে এলোমেলো ফোরাম মন্তব্যের একটি token-এর চেয়ে বেশি দরকারি signal বহন করে — তাই বেশিরভাগ বাস্তব pipeline উচ্চ-মানের curated উৎসগুলিকে (বই, Wikipedia, code) *যতটা ন্যায্য তার চেয়ে বেশি ওজন* দেয় (pool-এ কাঁচা অংশের চেয়ে বেশি নমুনা নেয়) এবং অনেক বড় কিন্তু noisier web-crawl অংশকে কাঁচা byte-সংখ্যার তুলনায় *কম ওজন* দেয়।
- **বৈচিত্র্য সংকীর্ণ দক্ষতা ঠেকায়।** শুধু ওয়েব গদ্যে প্রশিক্ষিত মডেল code এবং আনুষ্ঠানিক reasoning-এ দুর্বল; ইচ্ছাকৃতভাবে একটি code ভগ্নাংশ যোগ করলে (GPT-3, PaLM এবং LLaMA — তিনটিই করেছে) এমন কাজে পরিমাপযোগ্যভাবে উন্নতি হয় যার code লেখার সাথে কোনো সম্পর্ক নেই — দৃশ্যত কারণ code দীর্ঘ-পরিসরের কাঠামোগত ও যৌক্তিক নির্ভরতা শেখায় যা transfer হয়।

প্রকাশিত মিশ্রণগুলো এটিকে মূর্ত করে: GPT-3-এর প্রশিক্ষণ মিশ্রণ তার সবচেয়ে বড় কাঁচা উপাদানটিকে (একটি filtered Common Crawl, কাঁচা token-এর `~82%`) downweight করে আসল প্রশিক্ষণ মিশ্রণের প্রায় 60%-এ নামিয়ে এনেছিল, উচ্চ-মানের উৎসগুলোকে (WebText2, Books1/2, Wikipedia) তাদের আকারের তুলনায় কয়েক গুণ বেশি upsample করে। সঠিক অনুপাতগুলোকে নিজেদের মধ্যে গুরুত্বপূর্ণ tunable hyperparameter হিসেবেই ধরা হয় — প্রায় প্রতিটি নতুন model family-এর জন্য পরীক্ষামূলকভাবে নতুন করে আবিষ্কৃত হয়; কোনো সর্বজনীনভাবে "সঠিক" মিশ্রণ নেই, শুধু একটি আছে যেটি কোনো নির্দিষ্ট মডেলের টার্গেট capability-র জন্য কাজ করতে যাচাই করা হয়েছে।

## Video Script Outline

1. Motivation — "একটি LLM ততটাই ভালো, যতটা ভালো তার প্রশিক্ষণ set তৈরির pipeline"
2. কাঁচা text কোথা থেকে আসে: Common Crawl আর curated উৎস
3. Extraction ও cleaning: কাঁচা HTML-কে সাধারণ গদ্যে রূপান্তর
4. Heuristic বনাম classifier-ভিত্তিক quality filtering, কংক্রিট rejection উদাহরণসহ
5. Exact dedup (hashing) বনাম near-dedup (MinHash + Jaccard), এবং কেন duplicate ক্ষতি করে
6. Data mixing: কেন অনুপাত, শুধু ভলিউম নয়, চূড়ান্ত মিশ্রণ নির্ধারণ করে
7. `example.py`-এর walkthrough — একটি toy document set-এ heuristic filter, exact-hash dedup এবং MinHash near-dedup
8. Recap + Lesson 2-এর প্রিভিউ: পরিষ্কার data পেলে আসলে কী objective-তে প্রশিক্ষণ দেবেন?

## Further Reading

- Wenzek et al. (2020), *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*
- Rae et al. (2021), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (MassiveText pipeline এবং এর filtering/dedup recipe)
- Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3 — Appendix A-তে এর data mixture ও quality-classifier filtering বিস্তারিত আছে)
- Broder (1997), *On the Resemblance and Containment of Documents* (মূল MinHash paper)
- Penedo et al. (2023/2024), *The RefinedWeb Dataset* এবং *FineWeb* (আধুনিক, সম্পূর্ণ ডকুমেন্টেড ওপেন ওয়েব-স্কেল filtering pipeline)