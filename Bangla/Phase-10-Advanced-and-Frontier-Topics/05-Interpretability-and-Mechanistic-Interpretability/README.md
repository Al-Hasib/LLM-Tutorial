# Interpretability and Mechanistic Interpretability

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `05-Interpretability-and-Mechanistic-Interpretability`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 4: Model Merging and Editing](../04-Model-Merging-and-Editing/README.md) একটি প্রশিক্ষিত model-কে এমন একটি black box হিসেবে আচরণ করেছিল যাকে তার বাহ্যিক আচরণের ভিত্তিতে জোড়া লাগানো, গড় করা বা প্যাচ করা যায়। এই lesson-টি সেই সবকিছুর নিচে থাকা কঠিনতর প্রশ্নটি করে: **নেটওয়ার্কের ভেতরে আসলে কী ঘটছে যা প্রথম স্থানেই সেই আচরণ তৈরি করে?** এই কোর্সের পূর্ববর্তী প্রতিটি lesson-ই বর্ণনা করেছে একটি Transformer কী হিসাব করে — [scaled dot-product attention](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md), residual stream, feed-forward block — কিন্তু এ পর্যন্ত প্রায় কিছুই জিজ্ঞেস করেনি *কেন* একটি নির্দিষ্ট প্রশিক্ষিত weight সেট একটি নির্দিষ্ট আচরণ তৈরি করে, বা *কোথায়* বিলিয়ন parameter-এর ভেতরে কোনো নির্দিষ্ট fact, দক্ষতা বা bias আসলে বাস করে। সেটিই interpretability-র বিষয়, আর তার সবচেয়ে উচ্চাকাঙ্ক্ষী শাখা, **mechanistic interpretability**, যা neural network-কে সেভাবে reverse-engineer করার চেষ্টা করে যেভাবে তুমি একটি compiled binary-কে reverse-engineer করবে: এটি যে কাজ করে তা শুধু পর্যবেক্ষণ করা নয়, বরং এর weight-এ বাস্তবায়িত প্রকৃত algorithm-টি উদ্ধার করা।

এটি কেবল বুদ্ধিবৃত্তিক কৌতূহল নয়। [Phase 06 Lesson 1: The Alignment Problem](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md) প্রতিষ্ঠা করেছে যে আমরা model-দের তাদের *বাহ্যিক* আচরণ গঠন করে প্রশিক্ষণ দিই (next-token prediction, তারপর preference-based fine-tuning), কখনো তাদের *অভ্যন্তরীণ* computation-কে সরাসরি উল্লেখ বা যাচাই না করে। একটি model তুমি পরীক্ষা করার কথা ভাবো এমন প্রতিটি prompt-এ aligned দেখাতে পারে, অথচ অভ্যন্তরীণভাবে ভিন্ন কিছু অনুসরণ করছে — প্রতারণামূলক আচরণ, প্রকৃত ধারণার জায়গায় দাঁড়ানো একটি ভুয়ো (spurious) সম্পর্ক, এমন একটি capability যা শুধু distribution shift-এ সক্রিয় হয়। ক্ষেত্রটিতে আসলে *যাচাই* করার — শুধু আচরণগতভাবে পরীক্ষা নয় — সবচেয়ে কাছের জিনিস হলো interpretability। ক্রমবর্ধমান সক্ষম সিস্টেম-গুলোকে বিশ্বাস করার জন্য একটি গুরুতর দীর্ঘমেয়াদী যুক্তির ভিত্তিতে এটি বসে আছে।

আর এটি, যথাযথভাবেই, পুরো কোর্সের শেষ lesson। এর আগের প্রতিটি phase ছিল LLM তৈরি, প্রশিক্ষণ, alignment, prompting, মূল্যায়ন এবং deployment নিয়ে; এই lesson-টি হলো ক্ষেত্রটির সেই শিল্পবস্তুকে (artifact) আসলে *বোঝার* প্রচেষ্টা নিয়ে, যা এই সবকিছু তৈরি করে — শেষ করার জন্য একটি উপযুক্ত স্থান, কারণ এটি আজ ক্ষেত্রের অন্যতম সক্রিয়, অমীমাংসিত frontier।

## এই lesson-এ কী কী আচ্ছাদিত হবে

- Probing classifier: একটি প্রদত্ত layer-এ একটি ধারণা linear-ভাবে ডিকোডযোগ্য (decodable) কিনা তা পরীক্ষা করতে frozen internal activation-এর উপর একটি সরল classifier প্রশিক্ষণ দেওয়া — আর এতে যা প্রমাণিত হয় তার সৎ সীমাবদ্ধতা
- attention pattern বিশ্লেষণ একটি সরাসরি, পরিদর্শনযোগ্য interpretability signal হিসেবে, আর induction head একটি বাস্তব, সু-নথিভুক্ত mechanistic circuit হিসেবে
- Sparse autoencoder (SAE): superimpose-করা, polysemantic activation-কে আরও বড় sparse, অধিকতর monosemantic feature-এর সেটে বিশ্লেষণ (decompose) করা
- একটি probing-classifier পরীক্ষা এবং সিন্থেটিক superimposed feature-এর উপর প্রশিক্ষিত একটি toy sparse autoencoder — উভয়েরই একটি হাতে-কলমে প্রদর্শন
- সম্পূর্ণ 11-phase পাঠক্রম জুড়ে পেছনে ফিরে দেখা, এবং ক্ষেত্রটি এখান থেকে কোথায় যায় তার দিকে তাকানো

## 1. কেন "এটি কাজ করে" মানে এই নয় "আমরা এটি বুঝি"

[Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)-এর পদ্ধতি দিয়ে প্রশিক্ষিত একটি model — embeddings, attention, feed-forward block, next-token cross-entropy — শেষ পর্যন্ত কেউ হাতে লেখে নি এমন সংখ্যার একটি বড় matrix হয়ে দাঁড়ায়। আমরা *architecture*-টি সুনির্দিষ্টভাবে জানি (প্রতিটি অপারেশন স্পষ্ট, differentiable এবং code-এ সম্পূর্ণরূপে নির্দিষ্ট), কিন্তু সাধারণভাবে আমরা জানি না সেই নির্দিষ্ট প্রশিক্ষিত weight-গুলো *কোন algorithm* বাস্তবায়ন করে। এটি deep learning-এর কেন্দ্রীয় অদ্ভুততা: source code-টি সরল ও পরিচিত; ফলে-আসা program-টি অস্বচ্ছ এবং বিশাল সংখ্যক learned floating-point parameter পরিদর্শন করে, ঘটনার পরে, অভিজ্ঞতাভিত্তিকভাবে (empirically) আবিষ্কার করতে হয়। সেই খাঁজ বন্ধ করার প্রচেষ্টাই interpretability গবেষণা, উচ্চাকাঙ্ক্ষার ভিন্ন স্তরে:

- **Behavioral/black-box interpretability** — model-কে শুধু input ও output দিয়ে পরীক্ষা করা (এটি [Phase 08: Evaluation](../../Phase-08-Evaluation-of-LLMs/README.md)-এর বেশিরভাগ অংশ)। দরকারি, কিন্তু তোমাকে বলে model *কী* করে, *কীভাবে* করে তা নয়।
- **Representational interpretability** — model-এর অভ্যন্তরীণ activation-গুলো (hidden state) দেখা, যেগুলো তৈরি করেছে এমন computation-কে সম্পূর্ণ reverse-engineer না করেই। পরের অংশে আলোচিত probing classifier-গুলো এখানে বাস করে।
- **Mechanistic interpretability** — আরও এগিয়ে গিয়ে প্রকৃত উপ-computation-টি (একটি "circuit": একসাথে সংযুক্ত attention head এবং/অথবা neuron-এর একটি ছোট সেট) উদ্ধার করার চেষ্টা করা, যা একটি নির্দিষ্ট capability-র জন্য দায়ী, এমন পর্যাপ্ত বিস্তারিত সহ যে তুমি, নীতিগতভাবে, শুধু weight-গুলো থেকেই এটি পুনরায় বের করতে পারো। নিচের induction head এবং sparse autoencoder এই স্তরের দুটি সবচেয়ে পরিণত ফলাফল।

## 2. Probing classifier

"এই layer-এর representation-এ কি concept X আছে?" জিজ্ঞেস করার সবচেয়ে সরল উপায় হলো এটিকে বের করার চেষ্টা করা: প্রশিক্ষিত model-টিকে সম্পূর্ণ freeze করো (এতে কোনো gradient update নেই), এর মধ্য দিয়ে input চালাও, কিছু নির্বাচিত layer-এ hidden activation vector-টি টেনে বের করো, আর একটি ছোট **নতুন** classifier — সাধারণত একটি একক linear layer — প্রশিক্ষণ দাও শুধু সেই activation থেকে একটি human-labeled concept পূর্বাভাস করতে:

```
frozen_model(input) -> hidden activations h  (no gradients flow into the frozen model)
probe = Linear(d_model -> num_classes)
loss  = CrossEntropy( probe(h.detach()), concept_label )
```

শুধু `probe`-এর weight-ই প্রশিক্ষণ পায়। যদি একটি *linear* probe উচ্চ নির্ভুলতা অর্জন করে, তাহলে এটি প্রমাণ যে ধারণাটি সেই layer-এর activation space-এ একটি **linearly decodable direction** হিসেবে উপস্থাপিত — "activation-গুলোর কিছু nonlinear function নীতিগতভাবে এটি উদ্ধার করতে পারত" বলার চেয়ে অনেক শক্তিশালী ও আরও নির্দিষ্ট দাবি (একটি যথেষ্ট শক্তিশালী nonlinear probe প্রায় যেকোনো layer থেকে প্রায় সবকিছুই decode করতে পারে, এবং সেই কারণেই ক্ষেত্রটি বিশেষভাবে linear probe-তে মিলিত হয়েছে: linear separability একটি representation-এর একটি অর্থবহ, অ-তুচ্ছ বৈশিষ্ট্য, এটি শুধু probe-এর ক্ষমতা সম্পর্কে একটি বিবৃতি নয়)।

**গুরুত্বপূর্ণ সতর্কতা — এখানেও সম্পর্ক (correlation) মানেই কার্যকারণ (causation) নয়।** উচ্চ probing নির্ভুলতা তোমাকে বলে যে ধারণাটি সেই layer-এর activation-এ *উপস্থিত এবং linear-ভাবে পাঠযোগ্য*। এটি নিজে থেকে তোমাকে বলে না যে model-এর *নিজস্ব downstream computation আসলেই সেই তথ্য ব্যবহার করে*। একটি model-এর পক্ষে সম্পূর্ণ সম্ভব অন্য কিছু হিসাব করার একটি আনুষঙ্গিক উপজাত হিসেবে একটি feature হিসাব করা, সেই feature-টিকে linear-ভাবে ডিকোডযোগ্য অবস্থায় বসিয়ে রাখা, আর কখনো এটিকে model-এর output-কে প্রভাবিত করে এমন কোনো জায়গায় route না করা — তবুও একটি probe উচ্চ নির্ভুলতার সাথে এটি খুঁজে পাবে। এটি সরল probing-এর একটি ব্যাপক-উদ্ধৃত failure mode (Belinkov, 2022; Hewitt & Liang, 2019 — এমন probe-গুলো যা প্রকাশ করার বদলে মুখস্থ করা শেখে): একটি ভালো probe *representation* প্রমাণ করে, *causal ব্যবহার* নয়। আরও দৃঢ় causal দাবির জন্য হস্তক্ষেপ (intervention) দরকার — যেমন activation patching (আনুমানিক feature-টি শূন্য করে দেওয়া বা অদলবদল করা এবং যাচাই করা যে model-এর output সত্যিই বদলায় কিনা) — যা correlational probing-এর চেয়ে এক ধাপ এগিয়ে, আর এর কারণেই mechanistic interpretability circuit-স্তরের causal পরীক্ষাকে সোনার মান (gold standard) গণ্য করে, probing-কে একটি দরকারি কিন্তু দুর্বল প্রথম-পাস সংকেত হিসেবে রেখে।

`example.py` এই পরীক্ষার একটি বাস্তব সংস্করণ চালায়: একটি ছোট model একটি কাজে প্রশিক্ষণ পায়, তারপর তার frozen অভ্যন্তরীণ activation-এর উপর একটি linear probe প্রশিক্ষণ পায় একটি *ভিন্ন*, সম্পর্কিত ধারণা উদ্ধার করতে, যা model-কে কখনো সরাসরি পূর্বাভাস করতে বলা হয়নি — সৎভাবে তুলনা করা হয় raw input-এ প্রশিক্ষিত একটি probe এবং একটি অপ্রশিক্ষিত (untrained) নেটওয়ার্কের activation-এ প্রশিক্ষিত একটি probe-এর সাথে।

## 3. attention pattern একটি সরাসরি interpretability signal হিসেবে

নেটওয়ার্কের বেশিরভাগ অভ্যন্তরীণ অংশের চেয়ে ভিন্ন, attention weight-গুলো প্রায় "বিনামূল্যেই" অস্বাভাবিকভাবে interpretable, কারণ [Phase 02-এর Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) ইতিমধ্যেই তাদের একটি সুনির্দিষ্ট অর্থ দিয়েছে: `weights = softmax(QKᵀ / √d_k)` হলো, সারি-সারি, একটি probability distribution, যা বলে "এই অবস্থানটি কোন আগের অবস্থানগুলো থেকে তথ্য টানে।" তুমি আক্ষরিকভাবে সেই `(T, T)` matrix-টিকে একটি heatmap হিসেবে প্লট করতে পারো এবং, একটি নির্দিষ্ট প্রশিক্ষিত head-এর জন্য, পড়ে নিতে পারো একটি নির্দিষ্ট input-এর জন্য এটি কোন token-গুলোতে মনোযোগ দেয় — কোনো reverse-engineering দরকার নেই, কারণ বস্তুটিই নির্মাণগতভাবে ইতিমধ্যেই position-এর উপর একটি distribution।

অনেক প্রশিক্ষিত model এবং input জুড়ে এটি নিয়মতান্ত্রিকভাবে করলে, গবেষকেরা পেয়েছেন যে পৃথক head-গুলো প্রায়ই চেনা, পুনর্ব্যবহারযোগ্য pattern-এ বিশেষায়িত হয়। সর্বোত্তম-নথিভুক্ত উদাহরণ হলো **induction head** (Olsson et al., 2022): একটি ছোট circuit, সাধারণত ভিন্ন layer-এ কাজ করা *দুটি* attention head দিয়ে গঠিত, যা একটি সরল কিন্তু শক্তিশালী algorithm বাস্তবায়ন করে:

```
Given a sequence: ... [A][B] ... [A] <- current position
Induction head predicts: [B]   (i.e., "last time I saw A, it was followed by B; predict B again")
```

যান্ত্রিকভাবে, একটি আগের layer-এর "previous-token head" প্রথমে প্রতি position-এর *আগের* token সম্পর্কে তথ্য সেই position-এর residual stream-এ কপি করে; পরে একটি "induction head" তারপর *বর্তমান* token-টির অতীত উপস্থিতিগুলো খোঁজে এবং তাদের ঠিক পরের position-এ মনোযোগ দেয়, গতবার যা অনুসরণ করেছিল তা সামনে কপি করে। নিট ফলাফল হলো pattern-সম্পূর্ণকরণ: যদি `"Doctor Smith ... Doctor"` একবার দেখা যায়, তাহলে model পরের বার `"Doctor"` দেখলে আবার `"Smith"` পূর্বাভাস করে — এমনকি এমন নাম বা বানানো token-এর জন্যও যাতে এটি কখনো প্রশিক্ষণ পায়নি, কারণ circuit-টি নির্দিষ্ট শব্দ মুখস্থ করার বদলে একটি সাধারণ কপি করার algorithm বাস্তবায়ন করে। এটি **in-context learning**-এর mechanistic ব্যাখ্যার একটি বড় অংশ: Transformer-এর নিজস্ব context window-এর আগের অংশ থেকে একটি pattern তোলা এবং পরে সেটি পুনরাবৃত্তি করার ক্ষমতার একটি উল্লেখযোগ্য ভগ্নাংশ আক্ষরিকভাবেই এই একটি circuit-এর মাধ্যমে সম্পাদিত হয়, আর বিভিন্ন model architecture ও আকারে প্রশিক্ষণের একটি মোটামুটি অনুমানযোগ্য বিন্দুতে induction head-গুলো উদিত হতে দেখা গেছে — এটি আজ পর্যন্ত mechanistic interpretability-র অন্যতম পরিচ্ছন্ন, সবচেয়ে পুনরুৎপাদনযোগ্য আবিষ্কার।

## 4. Superposition এবং sparse autoencoder

Probing এবং attention বিশ্লেষণ দুটোই model বড় হওয়ার সাথে একই দেয়ালে গিয়ে পড়ে: **superposition**। একটি model-এর "প্রতি ধারণায় একটি direction" ভঙ্গিতে উপস্থাপন করার মতো neuron বা residual-stream মাত্রা আছে তার চেয়ে অনেক বেশি ভিন্ন ভিন্ন ধারণা থাকতে পারে, যা এটি উপস্থাপন করতে চাইতে পারে। Toy-model কাজ (Elhage et al., 2022, *Toy Models of Superposition*) দেখায় যে যখন feature-গুলো sparse (প্রতিটি শুধু input-এর একটি ছোট ভগ্নাংশ-এ প্রাসঙ্গিক) এবং নেটওয়ার্কটি under-complete (feature-এর চেয়ে কম মাত্রা) হয়, তখন নেটওয়ার্ক একাধিক feature-কে *একই* ছোট মাত্রার সেটে ওভারল্যাপিং, non-orthogonal direction-এ প্যাক করতে শেখে — ব্যক্তিগতভাবে যত ধারণার জায়গা আছে তার চেয়ে অনেক বেশি ধারণা উপস্থাপনের বিনিময়ে তাদের মধ্যে সামান্য হস্তক্ষেপ (interference) সহ্য করে। বাস্তব পরিণতি: একটি একক neuron, বা hidden layer-এর একটি একক direction, প্রায়ই একটি পরিচ্ছন্ন মানব-ধারণার সাথে **মিলে না** — এটি "polysemantic," যা ঘটনাক্রমে একটি direction-ভাগ করা কয়েকটি সম্পর্কহীন জিনিসের জন্য সক্রিয় হয়। সেজন্যই বাস্তব model-এ নিষ্পাপ "neuron 417-এর মানে কী" পরিদর্শন এত ঘন ঘন বিভ্রান্তিকর, আপাত-সম্পর্কহীন সক্রিয়করণ উদাহরণের একটি মিশেল বের করে আনে।

**Sparse autoencoder (SAE)**-গুলো এই প্যাকিং খোলার (undo) জন্য বর্তমান শীর্ষস্থানীয় টুল। ধারণাটি (Anthropic-এর "Towards Monosemanticity" কাজে LLM অভ্যন্তরীণ অংশের জন্য আনুষ্ঠানিক রূপ দেওয়া, Bricken et al., 2023) হলো একটি model-এর activation-এর উপর একটি ছোট autoencoder প্রশিক্ষণ দেওয়া, তবে একটি মানক autoencoder-এর সাপেক্ষে দুটি ইচ্ছাকৃত মোড় (twist) সহ:

```
h = ReLU(W_enc x + b_enc)              # encoder: activation -> sparse code (h is OVERCOMPLETE: dim(h) >> dim(x))
x_hat = W_dec h + b_dec                # decoder: sparse code -> reconstructed activation

L = || x - x_hat ||^2  +  lambda * || h ||_1
    \_____reconstruction_____/    \___sparsity penalty___/
```

- **Undercomplete নয়, overcomplete।** একটি মানক autoencoder সংকুচিত করে (hidden dim input-এর চেয়ে ছোট, যা একটি bottleneck চাপিয়ে দেয়)। একটি SAE বিপরীত করে: এর hidden dimension ইচ্ছাকৃতভাবে input activation dimension-এর চেয়ে *বড়* করা হয় — যা model-কে `d` superimposed input dimension-কে অনেক বড় সংখ্যক প্রার্থী feature direction-এ unpack করার জায়গা দেয়, যতটা কাঁচা activation width একা linear-ভাবে উপস্থাপন করতে পারত তার চেয়ে বেশি।
- **weight-এর উপর নয়, hidden code-এর উপর L1 penalty।** `lambda * ||h||_1` টার্মটি সরাসরি শাস্তি দেয় যেকোনো প্রদত্ত input-এর জন্য কতগুলো hidden unit সক্রিয় (non-zero) — weight-গুলো কত বড় তা নয়। এটি model-কে এমন representation-এর দিকে ঠেলে দেয় যেখানে যেকোনো একক input-এর জন্য উপলব্ধ (অনেক) hidden unit-এর মধ্যে শুধু একটি ছোট মুষ্টিভর্তি সক্রিয় হয় — যা ঠিক সেই sparse-এবং-বড়-অভিধান (sparse-and-large-dictionary) চিত্র, যা superposition তত্ত্ব পূর্বাভাস দেয় যে মূল activation-গুলোর নিচে আছে।

যদি এটি কাজ করে, ফলে-আসা sparse hidden unit-গুলোর প্রতিটি অনেক বেশি **monosemantic** হওয়ার প্রবণতা রাখে: কাঁচা neuron-এর যে জট পাকানো মিশ্রণ থাকে তার বদলে একটি অপেক্ষাকৃত সুসংগত, প্রায়ই মানব-ব্যাখ্যাযোগ্য ধারণার জন্য সক্রিয় হওয়া। এটি Section 2-এর correlation-vs-causation সতর্কতাটি দূর করে না — একটি SAE feature এখনও একটি *representational* আবিষ্কার, আর এটি কার্যকারণগতভাবে ভারবহনকারী (causally load-bearing) তা দেখাতে এখনও একটি হস্তক্ষেপ দরকার (যেমন, feature-টিকে আটকে দিয়ে (clamp) যাচাই করা যে model-এর আচরণ ভবিষ্যদ্বাণীকৃত উপায়ে বদলায় কিনা) — তবে এটি interpretability গবেষকদের সেই causal পরীক্ষাগুলো চালানোর জন্য কাঁচা neuron-এর চেয়ে অনেক বড়, পরিষ্কার প্রার্থী ধারণার সেট দেয়।

`example.py` এটির একটি ছোট, সৎ সংস্করণ তৈরি করে: সিন্থেটিক activation vector, যা কম raw dimension-এ প্যাক করা (superimpose করা) "সত্যিকারের" অন্তর্নিহিত feature-এর একটি পরিচিত, বড় সেটের sparse সংমিশ্রণ হিসেবে নির্মিত; তাদের উপর কয়েকটি L1 শক্তিতে প্রশিক্ষিত একটি SAE; আর এর ফলে-আসা বাস্তব reconstruction-vs-sparsity trade-off-এর একটি রিপোর্ট — সহ প্রতিটি শক্তিতে কোন unit-গুলো "মৃত" (never activate) হয়ে যায়।

## পেছনে ফিরে তাকানো, সামনে তাকানো

এটি একটি 11-phase, 59-বিষয়ের কোর্সের শেষ lesson, তাই পুরো চাপ-বৃত্তটি (arc) একবার, ক্রমানুসারে, পেছনে হেঁটে দেখা মূল্যবান:

- **[Phase 00: Prerequisites](../../Phase-00-Prerequisites/README.md)** তার পরে যা কিছু আসে তার জন্য অনুমিত গণিত, neural network, NLP এবং PyTorch ভিত্তি তৈরি করেছে।
- **[Phase 01: Language Modeling Foundations](../../Phase-01-Language-Modeling-Foundations/README.md)** ভাষা মডেল আসলে কী তা সংজ্ঞায়িত করেছে, word embedding-এর পরিচয় দিয়েছে, RNN/LSTM/GRU নিয়ে হেঁটেছে এবং *কেন* তাদের sequential recurrence bottleneck হয়ে উঠল, আর সমাধান হিসেবে attention-এর পূর্বাভাস দিয়েছে।
- **[Phase 02: Transformer Architecture Deep Dive](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md)** সেই পূর্বাভাসকে কঠোর করেছে: tokenization, scaled dot-product এবং multi-head self-attention, positional encoding, সম্পূর্ণ encoder-decoder architecture, আর residual/LayerNorm/FFN — একটি বাস্তব, প্রশিক্ষণযোগ্য mini-GPT-তে একত্রিত।
- **[Phase 03: LLM Architectures and Types](../../Phase-03-LLM-Architectures-and-Types/README.md)** সেই একটি architecture-কে নিয়ে তার পরিবার-বৃক্ষ দেখিয়েছে: decoder-only (GPT), encoder-only (BERT), encoder-decoder (T5/BART), Mixture of Experts, scaling law, আর long-context কৌশল।
- **[Phase 04: Pretraining LLMs](../../Phase-04-Pretraining-LLMs/README.md)** আচ্ছাদন করেছে ওই মাপের একটি model আসলে এত ডেটায় কীভাবে প্রশিক্ষণ পায়: data pipeline, pretraining objective, distributed training, আর mixed-precision optimization।
- **[Phase 05: Fine-tuning LLMs](../../Phase-05-Finetuning-LLMs/README.md)** একটি কাঁচা pretrained base model-কে task-অভিযোজনযোগ্য ও নির্দেশনা-শিক্ষণযোগ্য কিছুতে রূপান্তর করেছে: full fine-tuning বনাম PEFT, LoRA/QLoRA, prompt/prefix tuning ও adapter, আর instruction tuning (SFT)।
- **[Phase 06: Alignment and RLHF](../../Phase-06-Alignment-and-RLHF/README.md)** SFT খোলা রেখে যায় এমন খাঁজ বন্ধ করেছে: alignment সমস্যাটি নিজেই, reward modeling, PPO-সহ RLHF, DPO, RLAIF/Constitutional AI, এবং safety/bias/toxicity প্রশমন।
- **[Phase 07: Prompt Engineering and In-Context Learning](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/README.md)** — এই lesson-এর induction-head circuit-কে তার নিচের প্রকৃত mechanism-গুলোর একটি হিসেবে রেখে — আচ্ছাদন করেছে শুধু input-এর মাধ্যমে একটি প্রশিক্ষিত model থেকে সর্বোচ্চ সুবিধা কীভাবে পাওয়া যায়: zero/few-shot prompting, chain-of-thought, tree-of-thought/ReAct, automatic prompt optimization, আর structured output/function calling।
- **[Phase 08: Evaluation of LLMs](../../Phase-08-Evaluation-of-LLMs/README.md)** জিজ্ঞেস করেছে উপরের কোনো কিছু কাজ করেছে কিনা তা আসলে কীভাবে জানবে: metric, মানক benchmark, LLM-as-a-judge, hallucination/factuality মূল্যায়ন, আর human evaluation পদ্ধতি।
- **[Phase 09: Deployment and Inference Optimization](../../Phase-09-Deployment-and-Inference-Optimization/README.md)** আচ্ছাদন করেছে বাস্তব জগতে একটি প্রশিক্ষিত model দক্ষভাবে চালানো: quantization, KV-cache ও speculative decoding, serving framework, distillation/pruning, আর cost/latency optimization।
- **[Phase 10: Advanced and Frontier Topics](../README.md)** (এই phase) বর্তমানে যা সক্রিয়ভাবে গবেষণা করা হচ্ছে তা দিয়ে শেষ হয়েছে: multimodal LLM, advanced Mixture of Experts, state space model (Mamba), model merging and editing, আর — এই lesson — interpretability।

ক্ষেত্রটি এখান থেকে কোথায় যায়? সততার সাথে: কেউ সম্পূর্ণভাবে জানে না, আর এটিই একে একটি স্থিরীকৃত বিষয়ের বদলে frontier বলার বক্তব্য। কয়েকটি উন্মুক্ত সূত্র, যার পৃষ্ঠতল এই lesson-টি কেবল আঁচড় দেয়:

- **বাস্তব স্কেলে interpretability।** এই lesson-এর সবকিছু কয়েক মুঠো মাত্রা-সহ toy model-এ প্রদর্শিত। কয়েক দশ বিলিয়ন parameter এবং তদনুরূপ বিশাল সংখ্যক প্রার্থী feature-সহ model-এ probing, circuit analysis এবং SAE-কে স্কেল করা একটি সক্রিয়, ব্যয়বহুল, এবং শুধু আংশিকভাবে সমাধান করা প্রকৌশল ও বিজ্ঞানের সমস্যা — বর্তমান frontier SAE কাজ production-স্কেল model থেকে লক্ষ লক্ষ feature বের করে, আর তারপরও সম্ভবত আসলেই যা ঘটছে তার একটি ভগ্নাংশই ধারণ করে।
- **আরও সক্ষম ও দক্ষ architecture।** [Phase 10 Lesson 3](../03-State-Space-Models-Mamba/README.md)-এর state space model এবং [Lesson 2](../02-Mixture-of-Experts-Advanced/README.md)-এর sparse MoE routing হলো Transformer-এর compute/memory স্কেলিংকে তার গুণগত মান ছাড় না দিয়ে হারানোর বেশ কয়েকটি চলমান প্রচেষ্টার মধ্যে দুটি — এটি কোনোভাবেই সমাপ্ত আলোচনার কাছাকাছি নয়।
- **ক্রমবর্ধমান সক্ষম সিস্টেম-গুলোকে align করা।** [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md)-এর কৌশলগুলো মূলত বর্তমান frontier সিস্টেমের চেয়ে অনেক কম সক্ষম model-এ বিকশিত হয়েছিল; capability বাড়তে থাকলে preference-based fine-tuning একা যথেষ্ট থাকে কিনা — না হলে এটিকে এই lesson-এ বর্ণিত ধরনের অভ্যন্তরীণ, mechanistic যাচাইয়ের সাথে জুড়তে হবে কিনা — তা একটি উন্মুক্ত ও সক্রিয়ভাবে বিতর্কিত প্রশ্ন, সমাধান হওয়া নয়।
- **Multimodality এবং next-token text-এর বাইরে।** [Phase 10 Lesson 1](../01-Multimodal-LLMs/README.md) এমন model-গুলির একটি প্রাথমিক অধ্যায় যা image, audio এবং অন্যান্য modality-র উপর একটি add-on হিসেবে নয় বরং জন্মগতভাবে যুক্তি করে; এই কোর্সে যা আচ্ছাদিত হয়েছে (attention, scaling law, alignment কৌশল, interpretability টুল) তার কতটা পরিচ্ছন্নভাবে স্থানান্তরিত হয় আর কতটা সারগর্ভভাবে নতুন করে ভাবতে হয় — তা এখনও প্রকাশ্যে, বাস্তব সময়ে, কাজ করা হচ্ছে।

সেই উন্মুক্ত-অন্তসত্তা (open-endedness), সততার সাথে, 2026-এ একটি LLM কোর্স শেষ করার উপযুক্ত সুর: "এই হলো সম্পূর্ণ চিত্র" নয়, বরং "এই হলো আমরা কীভাবে এখানে পৌঁছলাম তার একটি বাস্তব, কার্যকর বোঝাপড়া, যা ক্ষেত্রটির পরের ফলাফল পড়তে এবং সত্যিই অনুসরণ করতে যথেষ্ট মজবুত।"

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. মোটিভেশন — "model weight জোড়া লাগানো" (Lesson 4) থেকে "weight যা হিসাব করে তা বোঝা" (এই lesson), এবং alignment-এর জন্য এটি কেন গুরুত্বপূর্ণ
2. interpretability-র তিনটি স্তর: behavioral, representational, mechanistic
3. Probing classifier — setup, উচ্চ নির্ভুলতা কী প্রমাণ করে ও করে না, causal-use সতর্কতা
4. attention pattern এবং induction head — in-context learning-এর পেছনে একটি বাস্তব, সু-নথিভুক্ত circuit
5. Superposition — কেন একটি neuron খুব কমই একটি ধারণা বোঝায়
6. Sparse autoencoder — overcomplete-plus-L1 কৌশল, আর Anthropic-এর monosemanticity ফলাফল
7. `example.py`-এর walkthrough — probing পরীক্ষার বাস্তব সংখ্যা, তারপর L1 শক্তি জুড়ে SAE-এর বাস্তব reconstruction/sparsity trade-off
8. বিদায়: prerequisites থেকে frontier গবেষণা পর্যন্ত সম্পূর্ণ 11-phase যাত্রার পুনরালোচনা, আর ক্ষেত্রটি এখান থেকে কোথায় যায়

## আরও পড়ার জন্য

- Alain & Bengio (2016), *Understanding Intermediate Layers Using Linear Classifier Probes*
- Olsson et al. (2022), *In-context Learning and Induction Heads* (Anthropic)
- Elhage et al. (2022), *Toy Models of Superposition* (Anthropic) — SAE-গুলো যে তাত্ত্বিক চিত্র খোলার জন্য নির্মিত
- Bricken et al. (2023), *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (Anthropic)