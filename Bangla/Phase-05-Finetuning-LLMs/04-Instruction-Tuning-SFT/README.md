# Instruction Tuning (SFT)

**Phase:** [LLM ফাইন-টিউনিং](../README.md) · **টপিক ফোল্ডার:** `04-Instruction-Tuning-SFT`

## কেন এটি গুরুত্বপূর্ণ

[Phase 03 Lesson 1 §1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#1-gpt-1-pretrain-then-fine-tune)-এ আমরা এই ধারণাটির রূপরেখা দেখেছি: GPT-1 কাঁচা text-এ pretrain হয়েছিল, তারপর একটি নির্দিষ্ট downstream task-এর জন্য labeled data-তে fine-tune হয়েছিল। **Instruction tuning** (যাকে প্রায়ই **Supervised Fine-Tuning**, বা সংক্ষেপে **SFT** বলা হয়) হলো এই একই pretrain-then-fine-tune প্যাটার্নটির জেনারেলাইজড রূপ: একটি সংকীর্ণ task-এর দিকে fine-tune করার বদলে (যেমন sentiment classification বা entailment), একটি base model-কে একসাথে অনেক task-জুড়ে বিস্তৃত হাজার হাজার *(instruction, response)* জোড়ায় fine-tune করা হয়, যাতে এটি একটি সাধারণ আচরণ শেখে — **যে instruction-ই দেওয়া হোক, তা অনুসরণ করা এবং সাহায্যকারী উত্তর দেওয়া** — পরিসংখ্যানগতভাবে যা-ই চালিয়ে যাওয়া যায়, সে ধরনের এলোমেলো text সম্পূর্ণ করার বদলে। এই ধাপটিই একটি কাঁচা pretrained language model-কে এমন কিছুতে রূপান্তর করে, যা একজন assistant-এর মতো আচরণ করে, আর [Lesson 5-এর Hugging Face workflow](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) এবং [Lesson 6-এর case study](../06-Domain-Specific-Finetuning-Case-Study/README.md) — দুটোই সরাসরি এই training ধাপের ওপর ভিত্তি করে তৈরি।

## এই lesson-এ যা যা শেখানো হবে

- কেন instruction tuning pretraining-এর মতোই একই architecture ও objective ব্যবহার করে, শুধু ভিন্ন dataset দিয়ে
- Instruction data-র ফরম্যাট: role tags ও chat templates
- SFT loss: শুধু response token-গুলোতে cross-entropy, `ignore_index` দিয়ে prompt mask করা
- দুটোতেই "mask" ব্যবহৃত হলেও, এটি BERT-এর masked-language-modeling loss থেকে কীভাবে আলাদা
- কেন instruction-*data-র quality ও diversity* কাঁচা পরিমাণকে ছাড়িয়ে যায় (LIMA-র সিদ্ধান্ত)
- `example.py`-র স্ক্র্যাচ-থেকে-তৈরি masked fine-tuning run আসলে কী প্রদর্শন করে

## ১. একই model, একই objective, ভিন্ন data

Instruction tuning-এর জন্য [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction)-এর Transformer architecture বা next-token-prediction loss-এর কোনো কিছুই বদলায় না। একটি pretrained base model — [Phase 04](../../Phase-04-Pretraining-LLMs/README.md)-তে যেমন বর্ণনা করা হয়েছে, বিশাল কাঁচা, unlabeled text corpus-এ সেভাবে train করা — *যেকোনো* text-এর বিশ্বাসযোগ্য continuation তৈরি করতে জানে। এটি যা শেখেনি তা হলো একটি নির্দিষ্ট *আচরণ*: প্রশ্ন বা আদেশের মতো আকৃতির কিছু দেখলে, এর উচিত "বিশ্বাসযোগ্য continuation"-টি হবে সরাসরি, সাহায্যকারী একটি উত্তর — prompt-টি যে register-এর সাথে সাদৃশ্যপূর্ণ হোক না কেন শুধু আরও text নয় (একটি forum post, একটি Wikipedia stub, একটি legal disclaimer — কাঁচা base model-এর "assistant" আচরণের প্রতি কোনো পক্ষপাতই থাকে না)। Instruction tuning ঠিক সেটিই ঠিক করে, *(instruction, response)* জোড়া দিয়ে তৈরি একটি curated dataset-এ সাধারণ supervised training চালিয়ে যাওয়ার মাধ্যমে — [full fine-tuning বা PEFT](../01-Full-Finetuning-vs-PEFT/README.md) ব্যবহার করে — দুটোই SFT-এর সাথে সামঞ্জস্যপূর্ণ; instruction tuning-কে সংজ্ঞায়িত করে *loss* ও *data*, update mechanism নয়।

## ২. Instruction data-র ফরম্যাট: role tags ও chat templates

সবচেয়ে সরল instruction format হলো কেবল দুটি labeled field, যা আদি-পর্বের open instruction dataset-গুলো জনপ্রিয় করে তুলেছিল (যেমন Stanford Alpaca):

```
### Instruction:
Summarize the following paragraph in one sentence.

### Response:
<the desired summary>
```

Production chat model-গুলো একই ধারণার একটি সমৃদ্ধ সংস্করণ ব্যবহার করে — একটি **chat template**, যাতে স্পষ্ট **role tags** দিয়ে চিহ্নিত থাকে প্রতিটি turn কে "বলেছে":

```
<|system|>
You are a helpful assistant.
<|user|>
Summarize the following paragraph in one sentence.
<|assistant|>
<the desired summary>
```

বিশেষ tokens-গুলো (`<|system|>`, `<|user|>`, `<|assistant|>`, বা model-নির্দিষ্ট সমতুল্য) tokenizer-এর vocabulary-তে যোগ করা হয়, ফলে একটি একক model multi-turn conversation, system prompt আর tool output — সব একই token stream-এর ভেতরে সামলাতে পারে, যেটিকে Transformer ইতিমধ্যেই অভিন্নভাবে প্রসেস করে। Hugging Face tokenizers এটিকে `tokenizer.apply_chat_template(...)` নামে প্রকাশ করে, যা [Lesson 5](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md#2-data-instruction-formatting-and-chat-templates)-জুড়ে ব্যবহৃত। `example.py` সবচেয়ে সরল সম্ভব two-field সংস্করণটি (`Instruction: ... \nResponse: ...`) ব্যবহার করে, যাতে নিচের masking কৌশলটি অক্ষরে অক্ষরে দেখা সহজ থাকে।

## ৩. Loss: শুধু response token-এ cross-entropy

সম্পূর্ণ `(instruction + response)` text-এ যদি প্রতিটি position-এ সাধারণ next-token-prediction loss দিয়ে train করা হতো, তাহলে model-এর gradient update-গুলোর একটি বড় অংশ চলে যেত *instruction পুনরুৎপাদন* শেখায় — "Summarize the following paragraph..."-এর পরের শব্দটি পূর্বাভাস করা — যা কাঙ্ক্ষিত আচরণ নয়, এবং ভালোভাবে উত্তর দেওয়ার যে সংকেত সেটি আসলে শেখায়, তাকে পাতলা করে দেবে। বাস্তব SFT বাস্তবায়নে সর্বজনীন সমাধানটি: input-এর আকৃতির সাথে হুবহু মিলে যায় এমন একটি **label tensor** বানাও, কিন্তু prompt অংশটি বদলে দেওয়া হয় একটি **ignore value** দিয়ে (PyTorch-এর রীতি হলো `-100`, যা `nn.CrossEntropyLoss`-এর ডিফল্ট `ignore_index`), যেন ওই position-গুলোতে loss ও gradient দুটোই শূন্য হয়:

```
input:   Instruction : uppercase the word cat \n Response :   C   A   T  \n
labels:  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100  C   A   T  \n
```

কংক্রিটভাবে, একটি tokenized sequence-এ, position `i`-এর label token `i+1` পূর্বাভাস করার তত্ত্বাবধান করে; যখনই token `i+1` এখনও prompt-এর অংশ, তখন সেটি `-100` করা হয়, আর অন্যথায় প্রকৃত token id। একটি সূক্ষ্ম বিষয়: prompt/response সীমানার ঠিক যে position-টি — শেষ prompt token থেকে *প্রথম* response token-টি পূর্বাভাস করা — সেটি **তত্ত্বাবধান করা হয়**, কারণ "সম্পূর্ণ prompt দেওয়া হলে, উত্তরটির প্রথম token তৈরি করো" এটিই ঠিক সেই আচরণ, যা শেখানো হচ্ছে। `example.py`-র `build_sft_example` ঠিক এটিই বাস্তবায়ন করে, আর `show_masking_demo` প্রতিটি position-এর label প্রিন্ট করে, যেন mechanism-টি দৃশ্যমান হয় — শুধু দাবি করা হয় না।

## ৪. BERT-এর সেই একই "mask" নয়

[Phase 03 Lesson 2-এর Masked Language Modeling](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#2-masked-language-modeling-mlm)-এর সাথে এটিকে গুলিয়ে ফেলা সহজ, কারণ দুটোতেই masking mechanism আছে এবং দুটোই loss-কে position-গুলোর একটি উপসেটে সীমাবদ্ধ করে — কিন্তু তারা সম্পূর্ণ ভিন্ন জিনিস mask করে:

|                      | BERT-এর MLM                                                                                                    | SFT                                                                                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| কী mask করা হয়       | **Input** token-গুলো `[MASK]` (বা noise) দিয়ে বদলে দেওয়া হয়                                                | Input-এ কিছুই বদলায় না                                                                                                                                                                                     |
| Loss কী ignore করে   | Masked position-গুলো **ব্যতীত** সবকিছু                                                                          | **Prompt-এর ভেতরের** সবকিছু                                                                                                                                                                                |
| কেন                  | Model-কে সত্যিকারের কার্যকর bidirectional representation গড়তে বাধ্য করা, কারণ এটি জানে না কোন position-গুলো মূল্যায়ন করা হবে | Prompt পুনরুদ্ধারে gradient নষ্ট না করা; শুধু যে আচরণটি শেখানো হচ্ছে তার তত্ত্বাবধান                                                                                                                         |
| Attention-এর দিক      | Bidirectional (encoder)                                                                                        | Causal (decoder) — [Phase 02 Lesson 6 §1](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#1-the-model-token-positional-embedding-n-decoder-blocks-head)       |

SFT-এর model এখনও প্রতিটি position-এ সম্পূর্ণ, অপরিবর্তিত prompt-টিকে input হিসেবে *দেখে* (প্রকৃত token-গুলোর ওপর সাধারণ causal self-attention); শুধু **loss target**-টি mask করা, এবং সেটিও sequence-এর prompt অংশটুকুর ওপরই।

## ৫. Data-র quality ও diversity কাঁচা পরিমাণকে ছাড়িয়ে যায়

স্বাভাবিক প্রবৃত্তি হলো "আরও instruction উদাহরণ = একটি আরও ভালো assistant।" Zhou et al. (2023), **LIMA** paper-এ ("Less Is More for Alignment"), খুঁজে পান এটি সিদ্ধান্তকারী কারণ নয়: মাত্র **1,000টি সযত্নে নির্বাচিত, বৈচিত্র্যপূর্ণ, উচ্চ-মানের** instruction/response জোড়ায় একটি শক্তিশালী pretrained base model fine-tune করলে এমন output পাওয়া যায়, যা orders of magnitude বেশি instruction data-তে train করা model-গুলোর সাথে প্রতিযোগিতা করতে পারে। তাদের ব্যাখ্যা — **Superficial Alignment Hypothesis** — হলো, একটি base model-এর জ্ঞান ও দক্ষতা প্রায় সম্পূর্ণরূপেই শেখা হয় বৃহৎ আকারের *pretraining*-এর সময়; instruction tuning-এর কাজ তুলনামূলকভাবে ছোট: model-কে সাহায্যকারী প্রতিক্রিয়ার *স্টাইল ও ফরম্যাট* শেখানো, যা একটি ছোট, সযত্নে বৈচিত্র্যপূর্ণ উদাহরণ-সেট ইতিমধ্যেই পুঙ্খানুপুঙ্খভাবে প্রদর্শন করতে পারে। এটি instruction-tuning-data প্রশ্নটিকে "আমার কাছে কতটা আছে" থেকে পুনর্নির্মিত করে এই প্রশ্নে — "এটি কি ধারাবাহিকভাবে উচ্চ মানের সঙ্গে যথেষ্ট বিস্তৃত পরিসরের task, phrase-গঠন ও response-স্টাইল কভার করে" — একটি data-curation দৃষ্টিভঙ্গি, যা [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md#1-data-curation-for-a-narrow-domain) সরাসরি একটি সংকীর্ণ domain-এ প্রয়োগ করে।

## ৬. `example.py` কী করে

`example.py` উপরের নিখুঁত mechanism-টিই স্ক্র্যাচ থেকে তৈরি করে:

1. একটি ছোট decoder-only Transformer ([Lesson 2](../02-LoRA-and-QLoRA/README.md) ও [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)-এর সেই একই `MiniGPT` block) সাধারণ text-এ plain, unmasked next-token prediction দিয়ে pretrain করে — যা "একটি base model, যে ইতিমধ্যেই ভাষাটি জানে"-এর প্রতিনিধি।
2. একটি instruction/response উদাহরণের জন্য masked label tensor position-by-position প্রিন্ট করে, যাতে `-100` বনাম প্রকৃত targets সরাসরি দেখা যায়।
3. Pretrained (এখনও instruction-tuned নয়) model-টির একটি held-out instruction prompt-এ generation দেখায় — এটি generic-corpus স্টাইলে এলোমেলো বকে, আর কোনো অবস্থাতেই উত্তর-সদৃশ কিছু তৈরি করে না।
4. একই *model*-টিকে একটি ছোট toy instruction dataset-এ (দুই ধরনের task — একটি শব্দ "uppercase" করা ও "reverse" করা — training-এর কিছু শব্দের ওপর প্রয়োগ করা) response-only masked loss ব্যবহার করে আরও train করে, তারপর একই held-out prompt-এ generation আবার চালায় এবং training শব্দ ও *instruction tuning-এর সময় কখনো দেখা হয়নি* এমন শব্দের সেট — দুটোর উপরই exact-match accuracy মাপে।

Held-out সংখ্যাগুলো সততার সঙ্গে রিপোর্ট করা হয়, যা-ই হোক না কেন: এত ছোট একটি model ও dataset-এ, instruction tuning নির্ভরযোগ্যভাবে **ফরম্যাটটি** শেখায় (সংক্ষিপ্ত উত্তর দেওয়ার পর থামো, এলোমেলো বকা না করে উত্তর দাও) — এমনকি না-দেখা input-গুলোতেও — অন্যদিকে নির্দিষ্ট **task দক্ষতা** একেবারে নতুন শব্দে নিখুঁতভাবে generalization করতে চায় আরও/আরও বৈচিত্র্যপূর্ণ example, যা এই toy run দিতে পারে না — §5-এর সেই বক্তব্যের একটি ছোট, কংক্রিট প্রতিধ্বনি: আপনি কী ধরনের ও কতটা instruction data ব্যবহার করেন, সেটিই সরাসরি নির্ধারণ করে কী আসলে generalize করে।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "সেই নিখুঁত fine-tuning ধাপ, যা একটি কাঁচা base model-কে assistant-এর মতো আচরণ করা কিছুতে রূপান্তর করে"
2. Recap: GPT-1-এর pretrain-then-fine-tune, একটি task থেকে অনেকগুলো instruction/response জোড়ায় জেনারেলাইজড
3. Instruction data-র ফরম্যাট: Alpaca-ধাঁচের fields, তারপর role tags-সহ প্রকৃত chat templates
4. SFT loss: `ignore_index=-100` দিয়ে prompt mask করা, position-by-position হেঁটে দেখা
5. BERT-এর MLM masking-এর সাথে পার্থক্য — একই শব্দ "mask", ভিন্ন mechanism ও উদ্দেশ্য
6. LIMA: কেন 1,000টি ভালো উদাহরণ অনেক বড়, নিম্ন-মানের dataset-কে হারাতে পারে
7. `example.py`-এর ওয়াকথ্রু — pretrain করো, label mask-টি স্পষ্টভাবে দেখাও, instruction-tune করো, আর held-out শব্দের উপর before/after generation ও accuracy তুলনা করো
8. Recap + preview: Lesson 5 ঠিক এই একই training বাস্তব Hugging Face টুলিং (`SFTTrainer`) দিয়ে প্রকৃত model scale-এ করে

## আরও পড়ার জন্য

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1-এর মূল pretrain-then-fine-tune রেসিপি)
- Wei et al. (2021), *Finetuned Language Models Are Zero-Shot Learners* (FLAN — একসাথে অনেক task-জুড়ে instruction tuning)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — RLHF-এর আগে প্রথম ধাপ হিসেবে SFT; [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md)-তে প্রিভিউ করা)
- Taori et al. (2023), *Stanford Alpaca: An Instruction-Following LLaMA Model* (এই lesson-এর toy format হিসেবে ব্যবহৃত সরল `### Instruction:` / `### Response:` ফরম্যাট)
- Zhou et al. (2023), *LIMA: Less Is More for Alignment* (§5-এ মানের-ওপরে-পরিমাণের সিদ্ধান্তটি)