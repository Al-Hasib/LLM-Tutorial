# LLM-as-a-Judge

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `03-LLM-as-a-Judge`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 1](../01-Evaluation-Metrics/README.md)-এ একটি অস্বস্তিকর, কংক্রিট ফলাফল দিয়ে শেষ হয়েছিল: একটি সত্যিই সঠিক paraphrase BLEU/ROUGE-তে *খারাপ* স্কোর করেছিল, অথচ একটি সাবলীল উত্তর যেটি একটি মূল তথ্যই ভুল পেয়েছিল সেটি ভালো স্কোর করেছিল — কারণ এই metric-গুলো কেবল shared n-gram গণনা করে এবং তাদের অর্থ (meaning) সম্পর্কে কোনো ধারণা নেই। [Lesson 2](../02-Standard-Benchmarks/README.md)-এ ফিল্ডের অন্য প্রধান টুল — মানসম্মত benchmark — দেখানো হয়েছিল, যেটি কেবল কারণেই কাজ করে যে প্রতিটি প্রশ্নের একটি দ্ব্যর্থহীন সঠিক উত্তর থাকে যার বিরুদ্ধে log-probability যাচাই করা যায়। দুটি টুলের কোনোটিরই open-ended generation-কে গ্রেড করার উপায় নেই: "এই customer-এর প্রশ্নের একটি সহায়ক, যুক্তিসঙ্গত উত্তর লিখো" — এর সাথে overlap করার মতো কোনো reference string নেই এবং স্কোর করার মতো কোনো একক সঠিক token নেই। **LLM-as-a-Judge** হলো এই ফাঁকের প্রতি ফিল্ডের উত্তর: একটি শক্তিশালী LLM-কে নিজেই ব্যবহার করা একটি candidate response পড়তে ও গ্রেড করতে — অর্থ, সহায়কতা, সঠিকতার জন্য — ঠিক যেমন একজন মানব evaluator করতেন, কিন্তু খরচ ও সময়ের সামান্য ভগ্নাংশে। এটি preference-ভিত্তিক training-এর সরাসরি evaluation backbone-ও, যেটির সাথে আপনি [Phase 06 Lesson 2 (Reward Modeling)](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md)-এ পরিচিত হবেন: একটি reward model *হলো* একটি শেখা judge, যেটিকে প্রশিক্ষণ দেওয়া হয় একই ধরনের pairwise preference data-তে, যে ডেটা একটি LLM judge (অথবা একজন মানুষ, দেখুন [Lesson 5](../05-Human-Evaluation-Methodologies/README.md)) তৈরি করে। এই lesson-এ আলোচনা হয় judge-এর কাছে রায় চাওয়ার দুটি উপায়, সেই systematic bias-গুলো যেগুলো একটি naive judge-এর রায়কে অবিশ্বাসযোগ্য করে তোলে, এবং সেই mitigations যেগুলো একে বাস্তবে ব্যবহারযোগ্য করে তোলে।

## এই পাঠে যা যা আছে

- দুটি judge protocol: pairwise comparison ("কোন response টি ভালো") এবং absolute rubric scoring ("এই criteria-গুলিতে response-টিকে 1-10 স্কোর দাও")
- Position bias — যে নির্দিষ্ট response-টি judge প্রথমে (বা দ্বিতীয়বার) দেখে সেটির পক্ষে যাওয়ার প্রবণতা
- Verbosity bias — গুণমান নির্বিশেষে দৈর্ঘ্যকে পুরস্কৃত করার judge-এর প্রবণতা
- Self-preference bias — নিজের model family-র শৈলীর সাথে মিলে যাওয়া আউটপুটগুলোর পক্ষপাত করার judge-এর প্রবণতা
- প্রমিত mitigations: swap-and-average ordering, score-এ length নিয়ন্ত্রণ, বিভিন্ন judge-এর একটি panel ব্যবহার
- একটি শূন্য থেকে তৈরি simulation যা পরিমাপ করে যে এই bias-গুলো মাপা win rate-কে ঠিক কতটা বিকৃত করে, এবং প্রতিটি mitigation কতটুকু তা পুনরুদ্ধার (recover) করে

## 1. দুটি judge protocol

**Pairwise comparison।** Judge-কে একটি prompt এবং দুটি candidate response দেখান (দুটি ভিন্ন মডেলের, অথবা একই মডেলের দুটি ভিন্ন decoding run-এর), এবং জিজ্ঞেস করুন কোনটি ভালো — ঐচ্ছিকভাবে একটি "tie" অনুমোদনসহ। আউটপুট হলো একটি একক আপেক্ষিক preference। অনেক prompt-জুড়ে aggregate করলে pairwise judgment দুটি সিস্টেমের মধ্যে একটি **win rate** তৈরি করে, এবং অনেক সিস্টেম-জুড়ে aggregate করলে সেগুলো [Lesson 5](../05-Human-Evaluation-Methodologies/README.md#2-elo-rating-from-pairwise-comparisons)-এ শূন্য থেকে তৈরি করা Elo-শৈলীর ranking-কে খাওয়াতে পারে। Chatbot Arena-র মতো পাবলিক arena-র পেছনে এই protocol-টিই আছে।

**Absolute (rubric) scoring।** Judge-কে একবারে একটি response দেখান, সাথে একটি rubric যা বর্ণনা করে কী গ্রেড করতে হবে (যেমন "helpfulness, correctness এবং clarity — প্রতিটিকে 1-5 স্কোর দাও"), এবং প্রতি অক্ষে একটি score চান। এটি একটি আপেক্ষিক preference-এর বদলে প্রতি response-এ একটি absolute সংখ্যা তৈরি করে — যখন আপনার "কোনটি ভালো" ছাড়াও প্রতি-response ডায়াগনস্টিকস দরকার তখন এটি দরকারি, তবে সাধারণভাবে এটি একটি সরাসরি A-বনাম-B তুলনার চেয়ে বেশি noisy এবং judge-এর নিজের বারবার কলের মধ্যেও কম ধারাবাহিক — কারণ "10-এর মধ্যে 7 কতটা ভালো" প্রশ্নটির এমন কোনো অ্যাঙ্কর নেই যেটি "A কি B-এর চেয়ে ভালো" প্রশ্নটির আছে।

দুটি protocol-ই একই অন্তর্নিহিত অনুমান ভাগ করে: একটি যথেষ্ট শক্তিশালী LLM-এর রায় একটি যত্নশীল মানুষ যা বলতেন তার সাথে ভালোভাবে সম্পর্কিত। aggregate-এ এই অনুমানটি মোটামুটি ভালোভাবে ধরে (Zheng et al., 2023 তাদের MT-Bench সেটআপে GPT-4-এর রায় এবং মানব পছন্দের মধ্যে ~80%+ চুক্তির রিপোর্ট করেন) কিন্তু নির্দিষ্ট, systematic এবং — যা গুরুত্বপূর্ণ — *ভবিষ্যদ্বাণীযোগ্য* উপায়ে ভেঙে পড়ে — এই lesson-এর বাকি অংশের পুরো বিষয়বস্তু সেটিই।

## 2. Position bias

ঠিক একই pair-এর response-এর ক্ষেত্রে, prompt-এ কোন response-টি **প্রথম** আর কোনটি **দ্বিতীয়** দেখানো হয় তার উপর নির্ভর করে একটি pairwise judge-এর রায় উল্টে যেতে পারে — গুণমান নির্বিশেষে। এটিই **position bias**, এবং বাস্তবে এটি এত বড় যে প্রাথমিক GPT-4-as-judge গবেষণাগুলোতে দেখা গেছে, অভিন্ন pair-এর দুটি ordering-জুড়ে judge নিজের সাথে নিজে কেবল 60-80% সময় একমত হয়েছিল (Wang et al., 2023; Zheng et al., 2023)। কংক্রিটভাবে: যদি একটি naive evaluation harness সর্বদা নতুন/candidate মডেলের আউটপুটকে প্রথম স্লটে এবং baseline-এর আউটপুটকে দ্বিতীয় স্লটে রাখে (একটি eval script লেখার সময় এটা করা খুব স্বাভাবিক), তাহলে যেকোনো অশূন্য position bias সরাসরি এবং নীরবে candidate-এর মাপা win rate স্ফীত করে — candidate সত্যিই ভালো কি না তা নির্বিশেষে। `example.py` এই ব্যর্থতাটিকেই বাস্তব সংখ্যা দিয়ে পুনরুৎপাদন করে।

## 3. Verbosity bias

একটি judge — অনেক মানব পাঠকের মতো — দৈর্ঘ্যকে পূর্ণাঙ্গতার সাথে জড়িয়ে ফেলে, এবং অতিরিক্ত দৈর্ঘ্য কোনো সঠিকতা বা উপযোগিতা যোগ না করলেও লম্বা response-কে পুরস্কৃত করে (Zheng et al., 2023; Dubois et al., 2024)। এটি বিশেষ করে বিপজ্জনক কারণ এটি সহজে শোষণযোগ্য: যেকোনো মডেলকে কেবল *বেশি লিখতে* fine-tune বা prompt করা হলে তা judge-এর scorecard-এ উন্নত দেখাতে পারে, যদিও মূল কাজে এর কোনো উন্নতি হয়নি। Length-controlled evaluation সেটআপ (যেমন AlpacaEval-এর "length-controlled win rate," Dubois et al., 2024) বিশেষভাবে এই প্রভাবটিকে মাপা score থেকে বের করে দেওয়ার জন্য তৈরি।

## 4. Self-preference bias

একটি judge নিজের model family-র শৈলী, শব্দচয়ন-রীতি বা reasoning প্যাটার্নের সাথে সাদৃশ্যপূর্ণ আউটপুটকে আরও অনুকূলভাবে মূল্যায়ন করে — এমনকি যখন একটি স্বাধীন মানুষ সেই পছন্দটি ভাগ না করে (Zheng et al., 2023 একে "self-enhancement bias" বলেন, যেমন GPT-4 judge হিসাবে GPT-4-শৈলীর আউটপুটকে সামান্য অনুকূল করে)। তিনটি bias-এর মধ্যে এটি সবচেয়ে কপট, কারণ একক-judge গবেষণা থেকে এটি অদৃশ্য: সবকিছু সামঞ্জস্যপূর্ণ এবং আত্মবিশ্বাসী দেখায়, শুধু তা ধারাবাহিকভাবে একটি family-র ঘরোয়া শৈলীর দিকে ঝুঁকে থাকে।

## 5. Mitigations

উপরের প্রতিটি bias-এর একটি অনুরূপ, বাস্তব সমাধান আছে — সবগুলো `example.py`-তে বাস্তব আগে-পরে সংখ্যা দিয়ে প্রদর্শিত:

- **Position bias → দুটি ordering-ই মূল্যায়ন করুন।** প্রতি pair-এ তুলনাটি দুবার চালান — একবার A প্রথমে, একবার B প্রথমে — এবং (ক) দুটি ফলে আসা preference score-এর average নিন (এতে যেকোনো *ধ্রুবক* position bonus বীজগণিতীয়ভাবে বাতিল হয়ে যায়, কারণ দুটি run-এ এটি ভিন্ন response-এর সাথে যোগ হয়) অথবা (খ) কেবল সেসব রায় গ্রহণ করুন যেখানে judge দুটি ordering-জুড়ে নিজের সাথে একমত, আর মতানৈক্যকে tie হিসেবে গণ্য করুন। যেকোনো পদ্ধতিতে judge কলের খরচ ঠিক 2x হবে এবং *aggregate* win rate-এর উপর bias-এর প্রভাব দূর হবে।
- **Verbosity bias → length নিয়ন্ত্রণ করুন।** হয় judge-কে স্পষ্টভাবে length উপেক্ষা করার নির্দেশ দিন, অথবা — আরও শক্তভাবে — judge-এর নিজস্ব প্রকাশিত length পছন্দ মাপুন (যেমন অনেক গ্রেড করা pair-জুড়ে judge score-কে response length-এর বিরুদ্ধে regress করা) এবং তুলনার আগে প্রতিটি score থেকে সেই আনুমানিক প্রভাব বাদ দিন। এটিই AlpacaEval-এর length-controlled win rate-এর পেছনের ধারণা।
- **Self-preference bias → বিভিন্ন judge-এর একটি panel ব্যবহার করুন।** কোনো একক judge নিরপেক্ষ নয়, কিন্তু যদি *ভিন্ন* model family-র কয়েকজন judge-এর প্রত্যেকের নিজস্ব, মোটামুটি স্বাধীন ঘরোয়া-শৈলী পছন্দ থাকে, তাহলে তাদের রায়ের average (একটি "jury") aggregate-এ প্রতিটি individual judge-এর idiosyncratic bias-কে দমন করে — যেকোনো noisy, ভিন্নভাবে-পক্ষপাতী estimator-এর average করার একই পরিসংখ্যানিক যুক্তি।

এই mitigation-গুলোর কোনোটিই judge-কে নিখুঁত করে না — এগুলো একটি *systematic, পরিমাপযোগ্য* বিকৃতিকে কমায়, তারা অনুপস্থিত বিচার-ক্ষমতা যোগ করে না। একটি judge (LLM বা মানুষ) উপরের সব bias-সংশোধনের পরেও একটি response-এর প্রকৃত গুণমান সম্পর্কে ভুল রায় দিতে পারে; এই সমাধানগুলো কেবল *তুলনাটি কীভাবে সাজানো হয়েছিল* তা থেকে আসা ত্রুটিগুলো দূর করে, judge-এর অন্তর্নিহিত দক্ষতার ত্রুটি নয়।

## ভিডিও স্ক্রিপ্টের রূপরেখা

1. Motivation — Lesson 1-2 দেখিয়েছে ঠিক কী অনুপস্থিত: open-ended, reference-মুক্ত generation-কে স্কোর করার একটি উপায়
2. LLM-as-a-Judge: একটি শক্তিশালী মডেল ব্যবহার করে অন্য মডেলের আউটপুট গ্রেড করা — pairwise বনাম absolute scoring
3. Position bias: একই pair, একই গুণমান, স্লট ক্রম অনুযায়ী ভিন্ন রায়
4. Verbosity bias: প্রকৃত গুণমান নির্বিশেষে দৈর্ঘ্য পুরস্কৃত হয় — এবং কেন এটি শোষণযোগ্য
5. Self-preference bias: একটি judge নিজের family-র শৈলীকে সমর্থন করে — একক-judge গবেষণা থেকে অদৃশ্য
6. তিনটি mitigation: swap-and-average, length-control, বিভিন্ন judge panel
7. `example.py`-এর ভেতরে-বাইরে — একটি toy biased judge, পরিমাপকৃত bias প্রভাব, এবং পরিমাপকৃত mitigation পুনরুদ্ধার — সবই বাস্তব সংখ্যা সহ
8. Recap + [Lesson 5](../05-Human-Evaluation-Methodologies/README.md)-এর দিকে নির্দেশনা, যেখানে একই pairwise-preference ধারণা Elo rating-এ একত্রিত হয় — এবার মানুষের ভোট থেকে

## আরও পড়ুন

- Zheng, Chiang, Sheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — যে paper-টি LLM judge-এ position, verbosity এবং self-enhancement bias-কে নামকরণ ও পরিমাপ করেছে
- Wang, Yuan, Yao et al. (2023), *Large Language Models are not Fair Evaluators* — pairwise LLM judging-এ position bias-এর একটি কেন্দ্রীভূত গবেষণা, সাথে calibration সমাধান
- Dubois, Galambosi, Liang, Hashimoto (2024), *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*
- Zeng, Attia, Wu et al. (2024), *Evaluating Large Language Models at Evaluating Instruction Following* (LLMBar — একটি benchmark যা বিশেষভাবে দৈর্ঘ্য ও শৈলীর মতো superficial cue-এর প্রতি judge-এর দৃঢ়তা পরীক্ষা করার জন্য তৈরি)