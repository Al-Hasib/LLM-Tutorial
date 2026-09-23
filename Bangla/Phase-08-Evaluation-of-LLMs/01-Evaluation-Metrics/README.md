# Evaluation Metrics

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `01-Evaluation-Metrics`

## কেন এটি গুরুত্বপূর্ণ

[Phase 01 Lesson 1](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md#5-perplexity)-এ perplexity-কে একটি language model স্কোর করার "একমাত্র" উপায় হিসেবে পরিচয় করানো হয়েছিল, যেখানে এটি raw bigram count থেকে হিসাব করা হয়েছিল। এখন থেকে আপনি যেকোনো paper-এ যে মডেলটির কথাই পড়বেন, সেটি এখনও perplexity রিপোর্ট করবে — শুধু পার্থক্য হলো, সেটি count table-এর বদলে neural network-এর probability থেকে হিসাব করা হয়। কিন্তু perplexity কেবল মাপে যে মডেলটি *held-out*, *training-distribution*-এর টেক্সট কতটা ভালোভাবে ভবিষ্যদ্বাণী করে; এটি মোটেও বলে না যে *generate* করা একটি টেক্সট (একটি translation, একটি summary, একটি উত্তর) সত্যিই ভালো কি না। এই ফাঁকটিই এই পাঠ পূরণ করে: ক্লাসিক automatic metric — BLEU, ROUGE, exact match, token F1 — যেগুলো বিশেষভাবে *generate* হওয়া আউটপুটকে একটি reference-এর বিরুদ্ধে স্কোর করার জন্য তৈরি, overlapping word গণনা করে। এগুলো সস্তা, নির্ধারক (deterministic) এবং পুনরুৎপাদনযোগ্য — ঠিক এই কারণেই দুই দশক ধরে NLP evaluation-এ এগুলোই আধিপত্য করেছে — এবং ঠিক এই কারণেই, এই পাঠের শেষে আপনি দেখবেন, একটি সম্পূর্ণ সঠিক উত্তর যেটি শুধু ভিন্ন শব্দ ব্যবহার করেছে সেটি দিয়ে এগুলোকে বোকা বানানো যায়। এই সৎ সীমাবদ্ধতাই [Lesson 3: LLM-as-a-Judge](../03-LLM-as-a-Judge/README.md) থাকার পুরো কারণ: একবার আপনি দেখবেন যে *কেন* n-gram overlap paraphrase-এর ক্ষেত্রে ভেঙে পড়ে, তখন surface overlap-এর বদলে অর্থ বিচার করার জন্য একটি দ্বিতীয় LLM-কে judge হিসেবে ব্যবহার করা আর কোনো অদ্ভুত ধারণা নয়, বরং প্রয়োজনীয় বলে মনে হবে।

## এই পাঠে যা যা আছে

- Perplexity — n-gram table-এর বদলে neural মডেলের জন্য সংক্ষিপ্ত পুনরালোচনা
- BLEU: brevity penalty-সহ n-gram precision — ক্লাসিক machine-translation metric
- ROUGE-N এবং ROUGE-L: recall-মুখী overlap metric — ক্লাসিক summarization metric
- Exact Match এবং token-level F1: আদর্শ extractive-QA metric
- সব overlap-নির্ভর metric-এর অভিন্ন সৎ দুর্বলতা, বাস্তব সংখ্যা দিয়ে প্রদর্শিত

## 1. Perplexity — সংক্ষিপ্ত পুনরালোচনা

Phase 01-এর সংজ্ঞাটি মনে করিয়ে দেওয়া যাক:

```
PPL = exp( -(1/T) * sum_t log P(w_t | w_1, ..., w_{t-1}) )
```

কোনো LLM-এর ক্ষেত্রে ধারণাগতভাবে কিছুই বদলায় না — `P(w_t | ...)` এখন smoothed count ratio-এর বদলে Transformer-এর softmax থেকে উৎপন্ন হয়, এবং `T` মাপা হয় *subword token*-এ ([Phase 02 Lesson 1](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md)), সম্পূর্ণ শব্দে নয়। Perplexity দরকারি এবং সস্তা (এটির কোনো reference *output* লাগে না — শুধু এমন held-out টেক্সট লাগে যেটিতে মডেলকে প্রশিক্ষণ দেওয়া হয়নি), কিন্তু এটি কেবল মডেলের নিজস্ব next-token distribution-কে স্কোর করে, যে distribution টি প্রাকৃতিকভাবে ঘটে যাওয়া টেক্সটের উপর তৈরি। এটি কোনো *generate* করা translation, summary বা উত্তরকে একটি *reference*-এর বিরুদ্ধে স্কোর করতে পারে না — সেটির জন্য দরকার এমন একটি metric যা দুটি টেক্সটকে সরাসরি তুলনা করে। এই পাঠের বাকি অংশ সেটিই নিয়ে।

## 2. BLEU (machine translation)

BLEU (**BiLingual Evaluation Understudy**, Papineni et al., 2002) একটি candidate translation-কে স্কোর করে এই ভেবে যে তার n-gram বিষয়বস্তুর কতটুকু এক বা একাধিক human reference translation-এ-ও বিদ্যমান — এটি মূলত একটি **precision** metric: "মডেল যে n-gram তৈরি করেছে, তার মধ্যে কতগুলো সত্যিই সঠিক ছিল?"

**Modified n-gram precision।** সাদামাটা precision-কে গেম করা যায় একটি সঠিক শব্দ বারবার রিপিট করে (যেমন reference `"the cat sat down"`-এর বিরুদ্ধে candidate `"the the the the"` 100% unigram precision পায়)। BLEU এটা ঠিক করে প্রতিটি n-gram-এর count-কে *clip* করার মাধ্যমে — candidate-এ প্রতিটি n-gram-এর count-কে ক্যাপ করা হয় যেকোনো একটি reference-এ দেখা তার সর্বোচ্চ count-এ:

```
p_n = ( sum over n-grams in candidate of min(count_candidate(ngram), count_reference(ngram)) )
      / ( total count of n-grams in candidate )
```

**Brevity penalty।** শুধু precision-ই ছোট উত্তরকেও পুরস্কৃত করে — একটি candidate যে কেবল "the" আউটপুট করে, সেই একক শব্দে নিখুঁত precision পায়। BLEU এর প্রতিকার করে একটি **brevity penalty (BP)** দিয়ে, যা reference-এর চেয়ে ছোট candidate-কে শাস্তি দেয়:

```
BP = 1                       if c > r
BP = exp(1 - r/c)            if c <= r
```

যেখানে `c` হলো candidate-এর দৈর্ঘ্য এবং `r` হলো (নিকটতম) reference-এর দৈর্ঘ্য। reference-এর ঠিক সমান দৈর্ঘ্যের candidate কোনো penalty পায় না; reference-এর অর্ধেক দৈর্ঘ্যের candidate পায় `BP = exp(1 - 2) = exp(-1) ~= 0.37`।

**সম্পূর্ণ BLEU score।** কয়েকটি order-এর (সাধারণত n=1..4) n-gram-এর precision-কে weighted geometric mean হিসেবে একত্রিত করুন, তারপর brevity penalty প্রয়োগ করুন:

```
BLEU = BP * exp( sum_{n=1}^{N} w_n * log(p_n) )
```

সাধারণত `w_n = 1/N` ধরা হয় (unigram থেকে 4-gram পর্যন্ত সমান গুরুত্ব)। BLEU-এর মান 0 থেকে 1-এর মধ্যে (প্রায়ই x100 আকারে রিপোর্ট করা হয়)।

## 3. ROUGE (summarization)

ROUGE (**Recall-Oriented Understudy for Gisting Evaluation**, Lin, 2004) BLEU-এর জোরকে উল্টে দেয়: যেহেতু একটি ভালো summary-এর উচিত reference-এর বিষয়বস্তু *ধারণ* করা, তাই ROUGE মূলত একটি **recall** metric: "reference-এর n-gram-গুলোর মধ্যে কতগুলো candidate পুনরুৎপাদন করেছে?" (আধুনিক ব্যবহারে সাধারণত উভয় দিক মিলিয়ে একটি F-measure রিপোর্ট করা হয়, কিন্তু মূল প্রেরণা ছিল recall।)

**ROUGE-N** (n-gram recall):

```
ROUGE-N = ( sum over n-grams in reference of min(count_candidate(ngram), count_reference(ngram)) )
          / ( total count of n-grams in reference )
```

BLEU-এর precision-এর মতোই একই clipped-overlap numerator, কিন্তু divisor হলো candidate-এর বদলে *reference*-এর n-gram count — এটিই একে precision-এর বদলে recall বানায়।

**ROUGE-L** (longest common subsequence): নির্দিষ্ট দৈর্ঘ্যের n-gram-এর বদলে, candidate এবং reference-এর মধ্যে দীর্ঘতম সাধারণ অনুক্রম (longest common subsequence, LCS) খুঁজে বের করা হয় — একটি common subsequence-কে ধারাবাহিক (contiguous) হতে হয় না, তাই এটি শব্দ সন্নিবেশ বা পুনর্বিন্যাস সহ্য করতে পারে, যেখানে n-gram match ভেঙে যেত। দেওয়া আছে `LCS(X, Y)` — candidate `X` এবং reference `Y`-এর মধ্যে longest common subsequence-এর দৈর্ঘ্য:

```
R_lcs = LCS(X, Y) / len(Y)             # recall
P_lcs = LCS(X, Y) / len(X)             # precision
F_lcs = (1 + beta^2) * R_lcs * P_lcs / (R_lcs + beta^2 * P_lcs)
```

`beta` recall/precision-এর মধ্যে trade-off নিয়ন্ত্রণ করে (`beta` বড় হলে recall-এর পক্ষে যায়, যা ROUGE-এর মূল recall-মুখী মনোভাবের সাথে মেলে; `beta=1` দেয় harmonic mean, অর্থাৎ একটি সাধারণ F1)।

## 4. Exact Match এবং token F1 (extractive QA)

extractive QA-এর জন্য (উত্তরটি হচ্ছে passage থেকে কপি করা একটি literal span, যেমন SQuAD-এ), metric-গুলো সহজ ও কঠোর:

**Exact Match (EM):** হালকা normalization-এর পরে (lowercase করা, punctuation/article বাদ দেওয়া), ভবিষ্যদ্বাণীকৃত string যদি reference উত্তরটির সাথে অক্ষরে-অক্ষরে হুবহু মেলে তবে 1, নইলে 0। একটি dataset জুড়ে average করলে EM-ই হয়ে ওঠে accuracy।

**Token F1:** prediction এবং (normalize করা, whitespace-এ tokenize করা) reference উত্তরটিকে token-এর bag হিসেবে ধরে হিসাব করে:

```
overlap    = number of tokens shared between prediction and reference (as a multiset intersection)
precision  = overlap / len(prediction tokens)
recall     = overlap / len(reference tokens)
F1         = 2 * precision * recall / (precision + recall)
```

Token F1 এমন আংশিক credit দেয় যা EM পারে না: reference `"Eiffel Tower"`-এর বিরুদ্ধে `"the Eiffel Tower"` ভবিষ্যদ্বাণী করলে পাওয়া যায় `EM=0` কিন্তু `F1` শূন্যের অনেক উপরে — কারণ prediction-এর 3টি token-এর 2টিই সঠিক।

## 5. অভিন্ন দুর্বলতা: surface overlap-ই অর্থ নয়

উপরের প্রতিটি metric — BLEU, ROUGE, EM, token F1 — **lexical overlap** মাপে: একই শব্দ/n-gram কতগুলো একই জায়গায় দেখা যায়। এদের কারওই *অর্থ* সম্পর্কে কোনো ধারণা নেই। এর সরাসরি দুটি পরিণতি, এবং দুটোই `example.py`-তে বাস্তব সংখ্যা দিয়ে দেখানো হয়েছে:

- **বৈধ paraphrase-কে শাস্তি দেওয়া হয়।** একটি candidate যে একই কথা ভিন্ন শব্দে বলে (সঠিক synonym, পুনর্বিন্যস্ত clause, ভিন্ন কিন্তু সমানভাবে বৈধ শব্দচয়ন), BLEU/ROUGE-তে *প্রায় শূন্য* স্কোর করতে পারে — যদিও একজন মানব পাঠক একে নিখুঁত উত্তর বলবেন — কারণ পুরস্কৃত করার মতো কোনো n-gram overlap নেই।
- **Overlap-কে গেম করা যায়।** একটি candidate যে reference-এর (বা উৎস passage-এর) বড় verbatim অংশ হুবহু কপি করে, সেটি কৃত্রিমভাবে উচ্চ স্কোর পায় — সেটি সুসংগত (coherent), বিশ্বস্ত (faithful) বা সম্পূর্ণ উত্তর হিসেবে সঠিক কি না তো দূরের কথা — সরাসরি এই metric-গুলোর বিরুদ্ধে optimize করা (যেমন model selection-এর সময় অথবা training reward হিসেবে) একটি সিস্টেমকে প্রকৃত মানোন্নয়নের বদলে reference-সদৃশ শব্দচয়ন পুনরাবৃত্তি করার দিকে ঠেলে দিতে পারে।

এটি কোনো ছোটখাটো পাদটীকা নয় — এটিই প্রধান কারণ যে গুরুতর LLM evaluation open-ended generation-এর জন্য semantic-embedding-ভিত্তিক metric, model-ভিত্তিক স্কোরিং এবং শেষ পর্যন্ত [LLM-as-a-Judge](../03-LLM-as-a-Judge/README.md)-এর দিকে সরে এসেছে, আর BLEU/ROUGE/EM/F1 প্রধানত সস্তা ও পুনরুৎপাদনযোগ্য sanity check হিসেবেই দরকারি থেকে গেছে — যেসব ক্ষেত্রে surface overlap এবং semantic correctness বেশি ঘনিষ্ঠভাবে যুক্ত, এমন সংকীর্ণ ও সীমাবদ্ধ কাজগুলোর জন্য (টাইট reference phrasing সহ translation, দ্ব্যর্থহীন gold span সহ extractive QA)।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — perplexity মডেলের নিজস্ব distribution-কে স্কোর করে, কিন্তু একটি generate করা উত্তরকে reference-এর বিরুদ্ধে স্কোর করতে পারে না; তাহলে কী পারে?
2. BLEU: n-gram precision, কেন clipping দরকার, brevity penalty এবং কেন এটি বিদ্যমান
3. ROUGE: recall-এর দিকে উল্টে যাওয়া, ROUGE-N বনাম ROUGE-L (LCS), এবং কেন LCS পুনর্বিন্যাস সহ্য করে
4. extractive QA-এর জন্য Exact Match এবং token F1, এবং কেন F1 এমন আংশিক credit দেয় যা EM পারে না
5. `example.py`-এর ভেতরে-বাইরে — এগুলোর সবগুলোই শূন্য থেকে implement করুন, toy sentence জোড়ায় যাচাই করুন
6. সৎ প্রদর্শন: একটি ভালো paraphrase BLEU/ROUGE-তে খারাপ স্কোর পায়, অথচ সম্পূর্ণ সঠিক
7. একই toy model/sentence-এর উপর পাশাপাশি perplexity বনাম এই metric-গুলো
8. Recap: কেন surface-overlap metric-এর এই ফাঁক সরাসরি LLM-as-a-Judge-কে (পরবর্তী lesson) অনুপ্রাণিত করে

## আরও পড়ুন

- Papineni, Roukos, Ward, Zhu (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation* — machine translation মূল্যায়নের মূল BLEU paper
- Lin (2004), *ROUGE: A Package for Automatic Evaluation of Summaries* — summarization মূল্যায়নের জন্য ROUGE family-এর উৎস
- Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Comprehension of Text* — EM/F1 QA evaluation নিয়মের উৎপত্তি
- Callison-Burch, Osborne, Koehn (2006), *Re-evaluating the Role of BLEU in Machine Translation Research* — BLEU-এর দুর্বলতার একটি প্রাথমিক, প্রভাবশালী সমালোচনা