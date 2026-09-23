# Model Merging and Editing

**Phase:** [Advanced and Frontier Topics](../README.md) · **Topic folder:** `04-Model-Merging-and-Editing`

## কেন এটি গুরুত্বপূর্ণ

[Phase 05: Full Fine-tuning vs PEFT](../../Phase-05-Finetuning-LLMs/01-Full-Finetuning-vs-PEFT/README.md) এবং [LoRA and QLoRA](../../Phase-05-Finetuning-LLMs/02-LoRA-and-QLoRA/README.md) ক্ষেত্রটি আসলে কীভাবে কাজ করে তার একটি সরল সত্য প্রতিষ্ঠা করেছে: বিপুল সংখ্যক মানুষ *একই* open-weight base model-কে নিয়ে বিভিন্ন দিকে fine-tune করে — একটি দল coding-এর জন্য একে বিশেষায়িত করে, আরেকটি বিদেশি ভাষার জন্য, আরেকটি customer-support-এর সুরের জন্য, আরেকটি math reasoning-এর জন্য। সেই fine-tuning-গুলোর প্রতিটি একটি সম্পূর্ণ weight সেট তৈরি করে, যা যে base model থেকে শুরু হয়েছিল তার একই parameter space-এ বাস করে এবং শুধু সেই fine-tuning run-এ প্রয়োগ করা gradient update-গুলোর দ্বারাই তার থেকে পৃথক হয়।

এই lesson-টি সেই পর্যবেক্ষণ থেকে স্বাভাবিকভাবে উঠে আসা প্রশ্নটি করে: যদি দুটি fine-tuned model কেবল "base model + কিছু weight পরিবর্তন" হয়, তাহলে কি সেই weight পরিবর্তনগুলিকে নিজেদের অধিকারে বস্তু হিসেবে গণ্য করা যায় — যোগ করা, বিয়োগ করা, মিশ্রিত করা, বা অস্ত্রোপচারের মতো (surgically) মুছে ফেলা — আর কখনো আবার gradient descent চালাতে হয় না? আশ্চর্যজনক পরীক্ষালব্ধ উত্তর হলো হ্যাঁ, একটি কাজের উপযোগী মাত্রা পর্যন্ত। **Task arithmetic** তোমাকে বেশ কয়েকটি fine-tuned specialization-কে একটি model-এ সংযুক্ত করতে দেয়, আক্ষরিক অর্থেই তাদের weight delta-গুলো একসাথে যোগ করে। **SLERP** তোমাকে দুটি fine-tuned model-কে এমন একটি model-এ মিশ্রিত করতে দেয় যা উভয়ের বৈশিষ্ট্য উত্তরাধিকার করে নেয়, সরল averaging-এর ব্যর্থতার ধরনগুলো ছাড়াই। **Model editing** (ROME) আরও এগিয়ে যায়, দেখায় যে এমনকি একটি প্রশিক্ষিত model-এ সঞ্চিত একটি একক, নির্দিষ্ট তথ্যও সনাক্ত করা যায় এবং weight-এ সরাসরি মুছে-লেখা যায়, কোনো কিছুতে retrain না করেই। তিনটি কৌশলই প্রশিক্ষিত weight-কে একটি অস্বচ্ছ, অপরিবর্তনীয় (immutable) চূড়ান্ত পণ্য হিসেবে নয়, বরং একটি হেরফেরযোগ্য (manipulable) শিল্পবস্তু হিসেবে আচরণ করে — এমন একটি সুর যা পরের ও চূড়ান্ত lesson, [Interpretability and Mechanistic Interpretability](../05-Interpretability-and-Mechanistic-Interpretability/README.md)-এ বিপরীত দিক থেকে আবার ভেসে উঠবে, যেখানে প্রশ্ন "কীভাবে আমরা edit করব যা একটি model জানে" নয় বরং "কীভাবে আমরা বুঝতে পারি এটি কী জানে এবং কীভাবে এটি হিসাব করে"। এটি [State Space Models (Mamba)](../03-State-Space-Models-Mamba/README.md)-এর সাথেও একটি লুপ বন্ধ করে: merging এবং editing হলো architecture-নিরপেক্ষ (agnostic) ধারণা — তারা খাঁটি এই সত্যের উপর কাজ করে যে একটি model *হলো* সংখ্যার একটি বড় vector, সেই সংখ্যাগুলো attention বাস্তবায়ন করুক বা একটি recurrent state-space scan।

## এই lesson-এ কী কী আচ্ছাদিত হবে

- **Task arithmetic** (Ilharco et al., 2022): fine-tuned এবং base weight-এর পার্থক্য হিসেবে একটি "task vector" সংজ্ঞায়িত করা, আর সেই vector-গুলো যোগ/বিয়োগ করে capability সংযুক্ত বা অপসারণ করা
- **SLERP** (Spherical Linear intERPolation): কেন দুটি weight vector-এর মধ্যে great-circle চাপ বরাবর interpolate করলে তাদের norm সংরক্ষিত থাকে, সরল linear averaging-এর মতো নয়
- **ROME দিয়ে model editing** (Meng et al., 2022): causal tracing-এর মাধ্যমে model-এর MLP layer-এর ভেতরে একটি নির্দিষ্ট factual association সনাক্ত করা, আর একটি একক লক্ষ্যবস্তু rank-one weight update দিয়ে তা মুছে-লেখা — কোনো retraining নেই
- সৎ সীমাবদ্ধতা: কেন কাজগুলোর দ্বন্দ্ব বা স্কেল বাড়লে merging-এর মান ক্ষয় হয়, আর কেন localized edit-এর অ-স্থানীয় (non-local) পার্শ্বপ্রতিক্রিয়া থাকতে পারে

## 1. Task Arithmetic: weight delta-কে প্রথম-শ্রেণির বস্তু হিসেবে

ধরো তুমি `theta_base` parameter-সহ একটি shared base model থেকে শুরু করো (যেমন কোনো instruction tuning-এর আগের একটি pretrained LLM)। তুমি এটিকে task `i`-তে fine-tune করো — ধরো, একটি coding dataset — এবং একটি নতুন weight সেট `theta_finetuned_i` পেয়ে যাও। Ilharco et al. (2022) সেই fine-tune-এর জন্য **task vector**-কে কেবল elementwise পার্থক্য হিসেবে সংজ্ঞায়িত করেন:

```
tau_i = theta_finetuned_i - theta_base
```

`tau_i` হলো ঠিক সেই same space-এ বসবাসকারী একটি vector যেখানে weight-গুলো নিজেরা থাকে (প্রতি parameter-এ একটি সংখ্যা)। এটি additive ভাবে ধারণ করে "task `i`-তে fine-tuning base model-এর সম্পর্কে যা পরিবর্তন করেছে।" Paper-টির কেন্দ্রীয় পরীক্ষালব্ধ আবিষ্কার হলো, এই vector-গুলো আশ্চর্যজনকভাবে কাজের উপযোগী মাত্রা পর্যন্ত স্বাধীন, রচনাযোগ্য (composable) edit-এর মতো আচরণ করে:

```
theta_merged = theta_base + sum_i( lambda_i * tau_i )
```

- **task vector-গুলো একসাথে যোগ করা** (`theta_base + tau_1 + tau_2`) সাধারণত একটি একক merged model তৈরি করে যা *দুটি* কাজেই — task 1 এবং task 2 — দক্ষ, যদিও কখনো দুটিতে একসাথে প্রশিক্ষণ দেওয়া হয়নি, এবং merge-এর সময় কোনো কাজের মূল training data-ও লাগে না (শুধু দুটি checkpoint দরকার)।
- **একটি task vector-কে নেগেট করা** (`theta_base - tau_toxic`) সেই fine-tune যে আচরণ তৈরি করেছিল তা *দমন* করে — যেমন toxic text-এ fine-tune করে পাওয়া একটি task vector বিয়োগ করলে ফলে-আসা model-এ toxic generation পরিমাপযোগ্যভাবে কমে যায়, আলাদা কোনো detoxification প্রশিক্ষণ run ছাড়াই।
- স্কেলিং সহগ `lambda_i`-গুলো (সাধারণত একটি একক shared `lambda`, যা একটি ছোট validation set-এ টিউন করা হয়) নিয়ন্ত্রণ করে প্রতিটি task vector-এর edit কত জোরালোভাবে প্রয়োগ করা হয়; `lambda_i = 1` হলো সরল ডিফল্ট, কিন্তু বাস্তবে এটি প্রায়ই *সেরা* পছন্দ নয় — যেমন `example.py` পরীক্ষালব্ধভাবে দেখায়, merged-model-এর মান এই সহগটির প্রতি সংবেদনশীল হতে পারে, যেখানে কিছু মধ্যবর্তী মান "সাহায্য করার মতো যথেষ্ট edit"-এর সাথে "model-এর অন্যান্য capability নষ্ট করার মতো বেশি নয়"-এর মধ্যে ভারসাম্য রাখে।

এটি আদৌ কেন কাজ করে? স্বজ্ঞাতভাবে, কোনো downstream task-এ fine-tuning সাধারণত weight-গুলোকে base model থেকে সামান্য দূরত্বে নিয়ে যায়, একটি task-নির্দিষ্ট দিক বরাবর। যদি দুটি কাজের দিক প্রায় orthogonal হয়, তাহলে দুটি update যোগ করলেও কোনো একটি প্রায় বিঘ্নিত হয় না — প্রতিটি কাজের দিক অপরটির উপস্থিতিতে প্রায় অপরিবর্তিত থাকে। যদি দুটি কাজ weight-কে *পরস্পরবিরোধী* দিকে টানে (যেমন দুটি fine-tune-ই একই attention head-কে ভিন্ন, অসামঞ্জস্যপূর্ণ উদ্দেশ্যে পুনর্ব্যবহার করতে চায়), তাহলে যোগফল আংশিকভাবে বাতিল হয়ে যায়, আর merge-এর মান ক্ষয় হয় — এটিই ঠিক সেই failure mode, যা পরবর্তী merging কৌশল যেমন **TIES-Merging** (Yadav et al., 2023) সরাসরি লক্ষ্য করে, task vector-গুলো যোগ করার আগে তাদের মধ্যে sign দ্বন্দ্ব স্পষ্টভাবে সমাধান করে।

## 2. SLERP: সরলরেখা নয়, গোলকের (sphere) বরাবর interpolate করা

Task arithmetic vector-গুলো একসাথে যোগ করে। একটি ভিন্ন, সমানভাবে প্রচলিত merging পদক্ষেপ হলো দুই model-এর weight-এর মধ্যে *interpolate* করা — উদাহরণস্বরূপ, একই base-এর দুই fine-tune-এর "মাঝামাঝি 50% পথে" বসে থাকা একটি model তৈরি করা। সরল পদ্ধতিটি হলো **linear interpolation (LERP)**:

```
lerp(p0, p1, t) = (1 - t) * p0 + t * p1
```

LERP-এর একটি সূক্ষ্ম সমস্যা আছে। দুইটি ভিন্ন কিন্তু সম্পর্কিত fine-tune-এর উচ্চ-মাত্রিক (high-dimensional) weight vector সাধারণত *একইরকম norm* (মাত্রা) রাখে কিন্তু *ভিন্ন দিকে* ইঙ্গিত করে — জ্যামিতিকভাবে, তারা উৎপত্তিস্থল থেকে প্রায় একই দূরত্বে থাকে কিন্তু কিছু অ-তুচ্ছ কোণে পৃথক থাকে। দৈর্ঘ্যে মিল কিন্তু দিকে ভিন্ন এমন দুটি vector-কে গড় করলে এমন একটি vector পাওয়া যায় যা যেকোনো input-এর চেয়ে **ছোট** — হুবহু সেইভাবে যেভাবে 90 ডিগ্রি পৃথক দুটি unit vector-এর গড়ের দৈর্ঘ্য `~0.71` হয়, `1` নয়। model weight-এর ক্ষেত্রে প্রয়োগ করলে এর অর্থ হলো, `t` যতই প্রান্তবিন্দু থেকে দূরে সরে, LERP ততই নিয়মতান্ত্রিকভাবে **weight-এর কার্যকর মাত্রাকে সংকুচিত (shrink) করে**, আর সংকোচনটি সবচেয়ে খারাপ `t = 0.5`-এ — যা নীরবে activation-কে ছোট করে এবং merged model-এর আচরণকে এমনভাবে ক্ষয় করে যার সাথে কোনো fine-tune-এর *বিষয়বস্তুর* সম্পর্ক নেই; এটি সম্পূর্ণ vector averaging-এর একটি উপজাত।

**Spherical Linear interpolation (SLERP)** এটি ঠিক করে, দুটি vector-যে hypersphere-এ প্রায় অবস্থান করে তার উপর তাদের সংযোগকারী great-circle চাপ বরাবর interpolate করে — তাদের মধ্যকার সরল জ্যা (chord) বরাবর নয়:

```
omega = arccos( (p0 . p1) / (|p0| * |p1|) )      # angle between p0 and p1

slerp(p0, p1, t) = (sin((1-t) * omega) / sin(omega)) * p0
                  + (sin(t * omega)     / sin(omega)) * p1
```

`omega` হলো দুটি weight vector-এর মধ্যে কোণ, যা সাধারণ vector জ্যামিতির মতোই তাদের dot product থেকে হিসাব করা হয়। `sin((1-t)*omega)/sin(omega)` এবং `sin(t*omega)/sin(omega)` দুটি সহগ কোনো সরল `(1-t, t)` বিভাজন নয় — এদের এমনভাবে বেছে নেওয়া হয় যাতে interpolated vector-এর মাত্রা চাপ বরাবর `|p0|` ও `|p1|`-এর মধ্যে মসৃণভাবে পরিবর্তিত হয়, দুটোর নিচেই ডুবে যাওয়ার বদলে। যখন `t = 0`, সূত্রটি `p0`-তে পরিণত হয়; যখন `t = 1`, `p1`-তে পরিণত হয়; এর মাঝের প্রতিটি `t`-এ ফলাফল দুটোকে সংযোগকারী sphere-এর উপর (বা খুব কাছে) থাকে, তার অভ্যন্তরভেদ করে কাটার বদলে। `example.py` দুটি সত্যিই ভিন্ন weight vector-এর মধ্যে LERP এবং SLERP দুটোই হিসাব করে এবং 0 থেকে 1 পর্যন্ত প্রতিটি `t`-এ ফলে-আসা norm ছাপে, যাতে ডুব-বনাম-না-ডুব পার্থক্যটি একটি দাবি নয় বরং সরাসরি পর্যবেক্ষণযোগ্য সংখ্যা হয়।

বাস্তবে SLERP-কে পুরো flattened parameter vector-এ একসাথে নয় বরং per-tensor (বা per-layer) প্রয়োগ করা হয়, কারণ ভিন্ন layer-এর অত্যন্ত ভিন্ন scale এবং "দিক" থাকতে পারে — `mergekit`-এর মতো merging টুল layer-wise SLERP-কে একই base architecture-এর দুটি fine-tune মেশানোর একটি ডিফল্ট কৌশল হিসেবে জনপ্রিয় করেছে, যা প্রায়ই সরল weight averaging (যে কৌশলটি **Model Soups**, Wortsman et al., 2022-এর পেছনে) থেকে ভালো ফল দেয় — সুনির্দিষ্টভাবে কারণ এতে norm-collapse সমস্যাটি নেই।

## 3. Model Editing: ROME এবং surgical fact edit

Task arithmetic এবং SLERP দুটোই *সম্পূর্ণ* fine-tuned checkpoint-এর উপর কাজ করে — editing-এর একক হলো "এই training run যা যা পরিবর্তন করেছে।" **ROME** (Rank-One Model Editing; Meng et al., 2022) বিপরীত চরমে কাজ করে: এটি model-এর output করা একটি **একক fact**-কে edit করে, weight-এর একটি ক্ষুদ্র, সুনির্দিষ্টভাবে চিহ্নিত অংশ পরিবর্তন করে, কোনো gradient-descent প্রশিক্ষণ ছাড়াই।

পদ্ধতিটির দুটি ধাপ আছে:

**1. Causal tracing — *কোথায়* একটি fact থাকে তা খুঁজে বের করা।** Meng et al. `"The Eiffel Tower is located in the city of ___"`-এর মতো একটি prompt model-এর মধ্য দিয়ে চালান, তারপর পৃথক hidden state-গুলো (ভিন্ন layer ও token position-এ) নিয়মতান্ত্রিকভাবে corrupt ও restore করেন এবং পর্যবেক্ষণ করেন প্রতিটি restore সঠিক উত্তর `"Paris"`-কে কতটা ফিরিয়ে আনে। এটি এমন একটি causal map তৈরি করে যা দেখায় যে এই ধরনের associative fact-এর factual recall অসমভাবে অল্প কয়েকটি **mid-layer MLP module**-এর মাধ্যমে সম্পাদিত হয়, বিশেষত fact-টির subject-এর token position-এ (`"Eiffel Tower"`)। এটি editing-এর লক্ষ্যকে একটি নির্দিষ্ট layer-এর একটি নির্দিষ্ট weight matrix-এ স্থানীয়করণ (localize) করে — "নেটওয়ার্কের কোথাও" নয়, একটি কংক্রিট ঠিকানা।

**2. Rank-one update — সেই ঠিকানায় fact-টি মুছে-লেখা।** ROME প্রাসঙ্গিক MLP-এর down-projection weight matrix `W`-কে একটি associative key-value store হিসেবে মডেল করে: এটি একটি "key" vector `k`-কে (মোটামুটিভাবে, subject-এর একটি অভ্যন্তরীণ উপস্থাপনা, `"Eiffel Tower"`) একটি "value" vector `v`-তে (মোটামুটিভাবে, একটি অভ্যন্তরীণ উপস্থাপনা যা object-এ decode হয়, `"Paris"`) ম্যাপ করে। একটি নতুন association ঢোকাতে — ধরো, model-কে edit করে এর বদলে `"Rome"` উত্তর দেওয়ানো — ROME একটি নতুন value vector `v_new`-এর জন্য সমাধান করে যা কাঙ্ক্ষিত output তৈরি করবে, তারপর `W`-তে একটি ন্যূনতম, লক্ষ্যবস্তু **rank-one update** প্রয়োগ করে:

```
W_new = W + (v_new - v_old) * k^T / (k^T * k)
```

এটি একটি একক outer-product update (`rank 1`, কারণ এটি একটি কলাম vector এবং একটি সারি vector-এর গুণফল), যা একটি নির্দিষ্ট least-squares অর্থে `W`-এর *সবচেয়ে ছোট* পরিবর্তন হিসেবে বেছে নেওয়া হয় এবং যা key `k`-কে নতুন value `v_new`-এর দিকে redirect করে অন্য key-গুলোকে যতটা সম্ভব কম বিঘ্নিত করে। গুরুত্বপূর্ণভাবে, এখানে কোনো backpropagation নেই, বহু ধাপে optimize করা কোনো loss function নেই, আর edit করা ওই একক fact ছাড়া কোনো training data-ও নেই — এটি একটি weight matrix-এ প্রয়োগ করা closed-form linear algebra। Edit-টি অবিলম্বে কার্যকর হয় এবং, paper-এর মূল্যায়নে, edited fact-টির paraphrase-গুলিতে ("Which city is the Eiffel Tower in?") যথেষ্ট ভালোভাবে generalize করে, অন্যদিকে বেশিরভাগ অপ্রাসঙ্গিক fact অক্ষত রাখে — তবে এই বাক্যে "বেশিরভাগ" শব্দটি সত্যিই কাজ করে: পরবর্তী কাজ (যেমন sequential এবং mass editing-এর উপর) দেখেছে যে ROME-শৈলীর edit-গুলো আরও edit স্তূপীকৃত হওয়ার সাথে সাথে অন্যান্য সঞ্চিত জ্ঞানকে ক্ষয় করতে পারে, এবং একটি একক localized edit-এর এখনও সংশ্লিষ্ট fact-গুলোর উপর অ-স্থানীয় তরঙ্গপ্রভাব (ripple effect) থাকতে পারে। এত সূক্ষ্ম পর্যায়ের model editing একটি সম্পূর্ণভাবে সমাধান হওয়া সমস্যা নয়, বরং একটি উন্মুক্ত, সক্রিয়ভাবে গবেষণাধীন সমস্যা — এবং এটি ঠিক সেই ধরনের প্রশ্ন, যার উত্তর দিতে পরের lesson-এর interpretability টুল তৈরি করা হয়েছে: *কেন* এই নির্দিষ্ট matrix-এ একটি rank-one update এই একক fact-কে বদলে দেয়, আর এটি আমাদের কী বলে যে fact-টি প্রথম স্থানে কীভাবে উপস্থাপিত ছিল?

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. মোটিভেশন — অনেক মানুষ একই base model-কে ভিন্নভাবে fine-tune করে; কি আমরা retraining ছাড়াই ফলাফল সংযুক্ত বা edit করতে পারি?
2. Task vector: `tau = theta_finetuned - theta_base`, এবং কেন এদের সাধারণ vector-এর মতো যোগ-বিয়োগ করা যায়
3. Task arithmetic কর্মে — দুটি fine-tune merging করা, এবং সৎ সতর্কতা যে স্কেলিং সহগ `lambda` গুরুত্বপূর্ণ
4. LERP-এর norm-সংকোচন সমস্যা, জ্যামিতিকভাবে দেখানো (একই দৈর্ঘ্য কিন্তু ভিন্ন দিকের দুটি vector-কে গড় করলে যেকোনোটির চেয়ে ছোট হয়)
5. SLERP-এর সমাধান: চাপ বরাবর interpolate করা, সঠিক সূত্রসহ, এবং `example.py` থেকে মাপা norm-সংরক্ষণ ফলাফল
6. ROME: একটি নির্দিষ্ট MLP layer-এ fact সনাক্ত করতে causal tracing, তারপর rank-one closed-form update যা এটিকে মুছে-লেখে
7. তিনটি কৌশল জুড়েই সৎ সীমাবদ্ধতা — পরস্পরবিরোধী task vector, per-layer merge সিদ্ধান্ত, edit-এর তরঙ্গপ্রভাব
8. পুরো phase-এর পুনরালোচনা, এবং চূড়ান্ত lesson-এর হাতে তুলে দেওয়া: interpretability, অর্থাৎ "আমরা কীভাবে জানি এই weight-গুলোর ভেতরে প্রথম থেকেই আসলে কী আছে?"

## আরও পড়ার জন্য

- Ilharco et al. (2022), *Editing Models with Task Arithmetic*
- Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME)
- Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging Models*
- Wortsman et al. (2022), *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time*