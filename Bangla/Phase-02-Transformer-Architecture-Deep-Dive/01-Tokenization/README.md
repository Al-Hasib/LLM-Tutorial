# Tokenization

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `01-Tokenization`

## কেন এই বিষয়টি গুরুত্বপূর্ণ

এ পর্যন্ত প্রতিটি লেসন নিঃশব্দে ধরে নিয়েছে যে text পরিষ্কার "শব্দ" আকারে আসে। বাস্তব tokenizer মোটেই সেভাবে কাজ করে না — আর কোনো model text-কে token-এ ভাঙার জন্য যে সঠিক অ্যালগরিদম ব্যবহার করে, তা নিচের প্রতিটি বিষয়কে গভীরভাবে প্রভাবিত করে: vocabulary size, sequence length, বানান ভুল ও বিরল ভাষা কতটা ভালোভাবে সামলানো যায়, এমনকি অক্ষর গণনা বা পাটিগণিত করাও। এই লেসনে আমরা subword tokenization অ্যালগরিদম — Byte-Pair Encoding — সম্পূর্ণ স্ক্র্যাচ থেকে তৈরি করব; এটি সেই একই অ্যালগরিদম যা এই কোর্সের প্রতিটি model আসলে ব্যবহার করে। ফলে প্রতিটি LLM-এর forward pass-এর প্রথম ধাপটি আর কোনো black box থাকবে না।

## এই লেসনে যা যা শেখানো হবে

- Word-level বনাম character-level বনাম subword-level tokenization
- Byte-Pair Encoding (BPE): অ্যালগরিদমটি, ধাপে ধাপে
- WordPiece ও SentencePiece: BPE-এর ঘনিষ্ঠ cousin
- Byte-level BPE (tiktoken / GPT-স্টাইল): raw byte-কে tokenize করা, কোনো unknown-token সমস্যাই নেই
- Special token
- Vocabulary-size-এর trade-off

## ১. গ্রানুলারিটির তিনটি স্তর

- **Word-level** (Phase 00-01 জুড়ে আমরা যা নিঃশব্দে ব্যবহার করেছি): বিশাল vocabulary, প্রতিটি অদেখা শব্দই `<unk>`, "run"/"running"/"runner"-এর মধ্যে কোনো ভাগাভাগি করা গঠন নেই — [Phase 00: Introduction to NLP §2](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#2-word-level-tokenization)-এ দেখানো ব্যর্থতার ধরনটিই।
- **Character-level**: ছোট vocabulary (~১০০টি সিম্বল), *যেকোনো* string উপস্থাপন করা যায়, কিন্তু sequence খুব লম্বা হয়ে যায় (১০ শব্দের একটি বাক্য ৬০টি অক্ষর হতে পারে), আর অর্থ শেখার আগেই model-কে বানান শিখতে হয়।
- **Subword-level**: প্রতিটি আধুনিক LLM-এর বাস্তব, কার্যকর মধ্যম পথ। পরিচিত শব্দ পুরো token-ই থাকে ("the", "cat"); বিরল/অদেখা শব্দ অর্থপূর্ণ অংশে ভাগ হয়ে যায় ("unhappiness" → "un" + "happi" + "ness")। দুই দিকের সেরাটাই: সাধারণ text-এর জন্য ছোট sequence, আর যেকোনো নতুন বিষয়ের জন্য সাবলীল অবনতি।

## ২. Byte-Pair Encoding (BPE)

মূলত এটি একটি data-compression অ্যালগরিদম (Gage, 1994), যাকে NLP-র জন্য নতুন করে ব্যবহার করেন Sennrich et al. (2016)। Training অ্যালগরিদমটি:

1. একক অক্ষর (বা byte)-এর vocabulary দিয়ে শুরু করুন। Training corpus-এর প্রতিটি শব্দকে এই সিম্বলগুলোর একটি sequence হিসেবে উপস্থাপন করুন।
2. পুরো corpus জুড়ে প্রতিটি পার্শ্ববর্তী সিম্বল-জোড়া (adjacent pair) গণনা করুন।
3. **সবচেয়ে ঘন ঘন আসা pair** খুঁজে বের করুন, এবং সেটিকে একটি নতুন সিম্বলে merge করে vocabulary-তে যোগ করুন।
4. নির্দিষ্ট সংখ্যক merge-এর জন্য ধাপ ২-৩ পুনরাবৃত্তি করুন (এই merge-সংখ্যাটিই মূল hyperparameter — বেশি merge = বড় vocabulary, ছোট sequence)।

```
Toy example, corpus = "low lower lowest":
symbols: l o w   l o w e r   l o w e s t
merge 1: most frequent pair is (l, o) -> new symbol "lo"
         lo w   lo w e r   lo w e s t
merge 2: most frequent pair is (lo, w) -> new symbol "low"
         low   low e r   low e s t
... and so on
```

ফলাফল: ডেটা থেকে সরাসরি শেখা subword unit-এর একটি vocabulary — কোনো ভাষাবিদ এগুলো হাতে ডিজাইন করেননি; এগুলো ফ্রিকোয়েন্সি পরিসংখ্যান থেকেই নিজে থেকে বেরিয়ে এসেছে।

## ৩. নতুন text এনকোড করা

একবার merge শিখে নিলে (এবং সেগুলোর ক্রম রেকর্ড করা হলে), নতুন কোনো text এনকোড করা মানে: অক্ষর থেকে শুরু করে, শেখা merge-গুলো *যে ক্রমে শেখা হয়েছিল সেই ক্রমেই* যেখানেই প্রযোজ্য সেখানেই greedily প্রয়োগ করা। Training-এ কখনোই না দেখা কোনো শব্দ তখনও সফলভাবে এনকোড হয়, যতক্ষণ তার অংশগুলো (সবচেয়ে খারাপ অবস্থায় একক অক্ষর পর্যন্ত) দেখা হয়ে থাকে — word-level tokenization-এর মতো কোনো **out-of-vocabulary সমস্যা নেই**।

## ৪. WordPiece ও SentencePiece

- **WordPiece** (BERT-এ ব্যবহৃত): BPE-এর প্রায় হুবহু অনুরূপ, কিন্তু *সবচেয়ে ঘন ঘন* pair-এর বদলে এটি এমন pair-কে merge করে, যা একটি সরল language model-এর অধীনে training corpus-এর likelihood সবচেয়ে বেশি বাড়ায় — সূক্ষ্মভাবে ভিন্ন, বেশি "তথ্য-তাত্ত্বিক" (information-theoretic) merge মানদণ্ড।
- **SentencePiece** (T5, LLaMA ও আরও অনেক model-এ ব্যবহৃত): কোনো ভিন্ন অ্যালগরিদম নয়, বরং একটি tokenizer *framework* — এটি ভেতরে BPE বা unigram-language-model অ্যালগরিদম চালাতে পারে; তবে সবচেয়ে গুরুত্বপূর্ণ বিষয় হলো, এটি input-কে raw, আগে থেকে pretokenize-না করা একটি স্ট্রিম হিসেবে দেখে (এমনকি whitespace-ও নিয়মিত সিম্বল হিসেবে এনকোড করে, সাধারণত `▁`), ফলে এটি ভাষা-নিরপেক্ষ — কোনো ধারণাই নেই যে শব্দগুলোকে স্পেস দিয়ে আলাদা করা হয়েছে, যা চীনা বা জাপানির মতো ভাষার জন্য অত্যন্ত গুরুত্বপূর্ণ।

## ৫. Byte-level BPE (GPT-পরিবারের model-গুলো আসলে যা ব্যবহার করে)

Character-level BPE-তেও একটি দুর্বল জায়গা থেকে যায়: এমন একটি সত্যিকারের নতুন Unicode character (একটি emoji, একটি অস্পষ্ট লিপি) যা training-এ কখনোই দেখা যায়নি, সেটিও এনকোড করা যায় না। GPT-2 (Radford et al., 2019) Unicode character-এর বদলে **raw UTF-8 byte**-র ওপর BPE চালিয়ে এই সমস্যাটি চিরতরে সমাধান করেছে। সম্ভাব্য byte value মাত্র ২৫৬টি, তাই **যেকোনো ভাষায় বা emoji-তে, প্রতিটি সম্ভাব্য string প্রথম merge থেকেই উপস্থাপনযোগ্য** — এই কারণেই GPT-পরিবারের model-গুলো কখনো `<unk>` token নির্গত করে না। OpenAI-র `tiktoken` লাইব্রেরিটি এই স্কিমেরই একটি দ্রুত বাস্তবায়ন, এবং প্রতিটি GPT-3.5/GPT-4-পরিবারের model-ই এটি ব্যবহার করে।

## ৬. Special token

বাস্তব tokenizer model-এর প্রয়োজনীয় non-text সংকেতের জন্য কয়েকটি ID সংরক্ষিত রাখে:

| Token                | উদ্দেশ্য                                                   |
| -------------------- | --------------------------------------------------------- |
| `<bos>` / `<s>`  | Sequence-এর শুরু                                     |
| `<eos>` / `</s>` | Sequence-এর শেষ — প্রায়শই generation-কে থামানোর সংকেত    |
| `<pad>`            | batching-এর জন্য ছোট sequence-কে একটি সাধারণ দৈর্ঘ্যে padding করা |
| `<unk>`            | অজানা সিম্বল (byte-level BPE-তে বিরল/অনুপস্থিত)          |

## ৭. Vocabulary-size-এর trade-off

- **বড় vocabulary** → বেশি পরিচিত বহু-অক্ষরের অংশ একক token হয়ে যায় → প্রতি বাক্যে ছোট sequence → সস্তা self-attention (মনে রাখুন [Phase 01: Introduction to Transformers §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity)-এর `O(T²)`) — তবে embedding table ও শেষের softmax layer দুটোই বড় হয়।
- **ছোট vocabulary** → লম্বা sequence, সস্তা embedding/output layer, কিন্তু প্রকৃত text-এর প্রতি এককে বেশি computation।

বাস্তব model-গুলো tens of thousands পরিসরে বসতি স্থাপন করে (GPT-2: ~৫০K, অনেক আধুনিক LLM: ১০০K-২৫০K) — এটি empirically টিউন করা ভারসাম্য।

## Video Script Outline

1. Motivation — "আপনার prompt-এর সাথে ঘটে যাওয়া একদম প্রথম জিনিসটি, আর এটি আপনার ধারণার মতো নয়"
2. OOV সমস্যাকে concrete করে দেখানো সহ word বনাম character বনাম subword
3. BPE-এর merge অ্যালগরিদম, ছোট একটি corpus-এ হাতে কাজ করে দেখানো
4. ঘনিষ্ঠ cousin হিসেবে WordPiece ও SentencePiece
5. Byte-level BPE — কেন GPT model-গুলো কখনো "unknown token" বলে না
6. `example.py`-এর ওয়াকথ্রু — স্ক্র্যাচ থেকে একটি BPE tokenizer-কে training, একটি অদেখা শব্দ এনকোড করা
7. রিক্যাপ: vocabulary size-এর trade-off, আর একটি পূর্বাভাস যে self-attention (পরবর্তী লেসন) এই ধাপ থেকে বের হওয়া token-গুলো নিয়েই কাজ করে

## Further Reading

- Sennrich, Haddow, Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* (NLP-র জন্য BPE)
- Kudo & Richardson (2018), *SentencePiece: A simple and language independent subword tokenizer*
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2 — byte-level BPE-র প্রবর্তন), Section 2.2
- OpenAI `tiktoken` লাইব্রেরি (github.com/openai/tiktoken)