# Hugging Face দিয়ে Fine-tuning (PEFT + TRL)

**Phase:** [LLM ফাইন-টিউনিং](../README.md) · **টপিক ফোল্ডার:** `05-Finetuning-with-HuggingFace-PEFT-TRL`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এ যা যা mechanism স্ক্র্যাচ থেকে তৈরি করা হয়েছে — [PEFT-এর frozen-base ধারণা](../01-Full-Finetuning-vs-PEFT/README.md#4-the-peft-idea-freeze-almost-everything), [LoRA-র `A`/`B` low-rank update](../02-LoRA-and-QLoRA/README.md#1-the-core-idea-freeze-w-learn-a-low-rank-update) আর [SFT-এর response-masked cross-entropy loss](../04-Instruction-Tuning-SFT/README.md#3-the-loss-cross-entropy-on-response-tokens-only) — একটি প্রকৃত fine-tuning কাজ ঠিক এটিই চালায়, কেবল হাতে-লেখা PyTorch-এর বদলে অল্প কয়েকটি সু-পরীক্ষিত লাইব্রেরি কলের আড়ালে মোড়ানো। এই lesson-টি "আমি mechanism-টি বুঝি" থেকে "আমি এটিকে একটি বাস্তব pretrained model-এ চালাতে পারি"-তে পৌঁছানোর সেতু: Hugging Face-এর `transformers`, `peft` আর `trl` লাইব্রেরিগুলো, একসাথে ব্যবহার করলে, open-source LLM জগতে মানক fine-tuning stack-এর সবচেয়ে কাছের জিনিস — আর internals শেখানোর জন্য তৈরি কোনো কোর্সের বাইরে আপনি বাস্তবে এটিই হাত বাড়িয়ে ধরবেন। [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md) এরপর এই একই ধারণাগুলো — এখনও স্ক্র্যাচ থেকে, ফলাফলের পূর্ণ দৃশ্যমানতার জন্য — একটি কংক্রিট domain-adaptation case study-তে প্রয়োগ করে।

**এই environment সম্পর্কে একটি নোট:** এই কোর্সের runtime-এ `transformers`, `peft` আর `trl` install করা নেই (বাকি প্রতিটি lesson ইচ্ছাকৃতভাবে কাঁচা PyTorch-তেই থাকে)। তাই `example.py` সেগুলো আছে কি না পরীক্ষা করে, install command প্রিন্ট করে এবং অনুপস্থিত থাকলে পরিচ্ছন্নভাবে বেরিয়ে যায় — এখানে এটি চালালে ঠিক এটিই ঘটে — আর ফাইলটির প্রতিটি function-এ প্রকৃত, নির্ভুল লাইব্রেরি ব্যবহার রয়েছে, যা আপনি সরাসরি পড়তে পারেন, অথবা `pip install transformers peft trl`-এর পর সত্যিকারের চালাতে পারেন।

## এই lesson-এ যা যা শেখানো হবে

- `transformers.AutoModelForCausalLM` / `AutoTokenizer`: একটি প্রকৃত pretrained model ও তার tokenizer load করা
- `peft.LoraConfig` ও `get_peft_model`: একটি প্রকৃত model-এ LoRA প্রয়োগ, আর প্রতিটি config field কী নিয়ন্ত্রণ করে
- প্রতিটি `LoraConfig` field-কে Lesson 2-এর স্ক্র্যাচ-থেকে-তৈরি `LoRALinear`-এর সাথে ম্যাপ করা
- `trl.SFTTrainer`: Lesson 4-এর masked instruction-tuning loss মাত্র একটি `.train()` কল দিয়ে চালানো
- `SFTTrainer`-এর জন্য chat-template-ভিত্তিক data ফরম্যাটিং
- deployment-এর জন্য fine-tuned adapter merge ও সংরক্ষণ

## ১. একটি প্রকৃত pretrained model load করা

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
```

`AutoModelForCausalLM` ও `AutoTokenizer` হলো Hugging Face-এর "এই checkpoint-টির জন্য কোন model/tokenizer class প্রয়োজন তা নির্ধারণ করে লোড করো" ধরনের এন্ট্রি-পয়েন্ট — একই দুটি লাইন কাজ করবে, checkpoint নামটি একটি 100M-parameter model-এর দিকে ইঙ্গিত করুক বা 70B-এর। এটি-ই সেই base model, যেটি এই phase-এর প্রতিটি fine-tuning পদ্ধতি ধরে নেয়: [Phase 04](../../Phase-04-Pretraining-LLMs/README.md)-র রেসিপি দিয়ে pretrained, শুধু স্ক্র্যাচ থেকে train করার বদলে এখানে download করা।

## ২. `LoraConfig`, সরাসরি Lesson 2-এর `LoRALinear`-এ ম্যাপ করা

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

প্রতিটি field-ই [Lesson 2](../02-LoRA-and-QLoRA/README.md)-তে হাতে-তৈরি করা কিছুর একটি সরাসরি, নামধারী প্রতিরূপ:

| `LoraConfig` field | কী নিয়ন্ত্রণ করে                                                                                                                          | Lesson 2-এর সমতুল্য                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `r`                  | LoRA rank                                                                                                                                   | `W' = W + (alpha/r) * B @ A`-এর `r` ([section 1](../02-LoRA-and-QLoRA/README.md#1-the-core-idea-freeze-w-learn-a-low-rank-update))                                                                                                                                                                                                                        |
| `lora_alpha`         | Scaling-এর লব                                                                                                                                  | একই formula-র `alpha` — `lora_alpha / r`-ই সেই scaling factor, যা `LoRALinear.scaling` সরাসরি হিসাব করে                                                                                                                                                                                                                                                     |
| `lora_dropout`       | শুধু LoRA পথে (`x @ A.T @ B.T`) প্রয়োগ করা dropout, frozen base-তে নয়                                                                       | Lesson 2-এর ন্যূনতম সংস্করণে নেই — একটি regularizer, যা বাস্তব workflow-গুলো অতিরিক্ত যোগ করে                                                                                                                                                                                                                                                                 |
| `target_modules`     | *কোন* frozen `nn.Linear` layer-গুলো একটি `A`/`B` update দিয়ে মোড়ানো হবে                                                                    | `LoRALinear` কোন weight matrix-টিকে মোড়ে — বাস্তব config-গুলো সাধারণত শুধু attention-এর `q_proj`/`v_proj` (বা `k_proj`/`o_proj`, বা FFN projection) টার্গেট করে, প্রতিটি linear layer নয় — ঠিক কারণ [Lesson 2 section 2](../02-LoRA-and-QLoRA/README.md#2-why-the-parameter-count-drops-so-much)-এর parameter সাশ্রয় আপনি যতগুলো matrix বাদ দেবেন, তত সঙ্গুণিত হয় |
| `bias`               | bias terms-গুলোও train করা হবে কি না (`"none"`, `"all"`, বা `"lora_only"`)                                                                    | Lesson 2-এর `base.bias` frozen থাকা (`bias="none"`-ই প্রচলিত ডিফল্ট)                                                                                                                                                                                                                                                                                 |
| `task_type`          | `peft`-কে বলে এটি কোন model-এর "shape" মোড়ছে (causal LM বনাম sequence classification, ইত্যাদি), যেন এটি জানে কোথায় adapter সঠিকভাবে লাগাতে হবে | N/A — Lesson 2-এর `LoRALinear` কখনো মাত্র একটি `nn.Linear`-কেই মোড়ত, "task"-এর কোনো ধারণা ছাড়াই                                                                                                                                                                                                                                                            |

`get_peft_model` লোড করা model-টি ঘুরে দেখে, `target_modules`-এর সাথে নাম মিলে যায় এমন প্রতিটি `nn.Linear`-কে LoRA-মোড়ানো সংস্করণ দিয়ে বদলায় (Lesson 2-এর `LoRALinear.forward`-এর সাথে কার্যত অভিন্ন), আর বাকি সবকিছু স্বয়ংক্রিয়ভাবে freeze করে। `model.print_trainable_parameters()` [Lesson 2 section 2](../02-LoRA-and-QLoRA/README.md#2-why-the-parameter-count-drops-so-much)-তে formula দিয়ে হিসাব করা সেই exact trainable-vs-frozen breakdown প্রিন্ট করে — এবার একটি প্রকৃত, লোড করা model-এর জন্য।

## ৩. Data: instruction ফরম্যাটিং ও chat templates

```python
def to_text(example):
    example["text"] = (
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    )
    return example

dataset = dataset.map(to_text)
```

একটি বাস্তব workflow এই সরল Alpaca-ধাঁচের template ব্যবহার করতে পারে, অথবা model-এর প্রকৃত **chat template** `tokenizer.apply_chat_template(messages, tokenize=False)` দিয়ে — যা [Lesson 4 section 2](../04-Instruction-Tuning-SFT/README.md#2-instruction-data-formats-role-tags-and-chat-templates)-এ বর্ণিত model-নির্দিষ্ট role tags-গুলো স্বয়ংক্রিয়ভাবে বসিয়ে দেয়, `{"role": ..., "content": ...}` dict-গুলোর একটি তালিকা থেকে। যেভাবেই হোক, `dataset_text_field` দিয়ে `SFTTrainer`-কে বলা হয় dataset-ের কোন কলামে সম্পূর্ণ-ফরম্যাট করা text আছে।

## ৪. `SFTTrainer`: Lesson 4-এর masked loss, একটিমাত্র কলে

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./sft-lora-demo",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    dataset_text_field="text",
    max_seq_length=128,
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

`SFTTrainer.train()` সেই নিখুঁত loop-টি চালায়, যা [Lesson 4](../04-Instruction-Tuning-SFT/README.md) হাতে তৈরি করেছিল: প্রতিটি ফরম্যাট করা example tokenize করো, prompt অংশ জুড়ে `-100`-সহ একটি label tensor বানাও (`SFTTrainer` ফরম্যাটিং বা একটি স্পষ্ট `response_template` থেকে prompt/response বিভাজন শনাক্ত করে), শুধু response token-গুলোতে cross-entropy হিসাব করো, আর একটি optimizer-এর step নাও — সাথে batching, padding, gradient accumulation, checkpointing ও logging আপনার জন্য সামলে দেওয়া হয়। এটি `transformers.Trainer`-এর একটি পাতলা, সু-পরীক্ষিত মোড়ক, যা instruction-tuning-ধাঁচের data ও loss-এর জন্য বিশেষায়িত।

## ৫. Merge করা ও deploy করা

```python
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./final-merged-model")
```

`merge_and_unload()` হলো [Lesson 2 section 3](../02-LoRA-and-QLoRA/README.md#3-merging-at-inference-time-zero-extra-latency)-এর `merged_weight()` পদ্ধতির `peft`-র প্রকৃত বাস্তবায়ন — এটি প্রতিটি LoRA-মোড়ানো layer-এর জন্য `W + (alpha/r) * B @ A` হিসাব করে, ফলাফলটি একটি সাধারণ `nn.Linear`-এ লিখে দেয়, আর এমন একটি সাধারণ model ফেরত দেয়, যাতে inference-এর সময় `peft`-নির্দিষ্ট কোনো মোড়ক কোড থাকে না — মূল base model-এর তুলনায় zero অতিরিক্ত latency-তে।

## ৬. GPU বা এই লাইব্রেরিগুলো install না থাকলে কী হবে?

এই lesson-টি বোঝার জন্য এটিকে চালানো জরুরি নয় — উপরের প্রতিটি code block সাম্প্রতিক `transformers`/`peft`/`trl` রিলিজ-এর হুবহু প্রকৃত, কার্যকরী syntax, আর `example.py` সেগুলো প্রকৃত function-এর ভেতরে verbatim রাখে। লাইব্রেরিগুলো সত্যিই install না থাকলে (এই কোর্সের environment-এর মতো), `example.py` চালালে একটি install command প্রিন্ট হয় এবং ক্র্যাশ না করে পরিচ্ছন্নভাবে বেরিয়ে যায়। সেগুলো install করলে (`pip install transformers peft trl`) এবং পুনরায় চালালে প্রকৃত ছোট demo-টি শুরু থেকে শেষ পর্যন্ত চলে — `sshleifer/tiny-gpt2` দিয়ে একটি toy instruction set-এ train করা একটি প্রকৃত LoRA adapter-সহ, যেটি এমন একটি ছোট public checkpoint, যা বিশেষভাবে বেছে নেওয়া হয়েছে, কারণ এটি যথেষ্ট দ্রুত download ও train হয় একটি দ্রুত demo-র জন্য।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "এই phase-এ হাতে তৈরি সবকিছু, এবার সেই লাইব্রেরি কল আকারে, যা আপনি বাস্তবে ব্যবহার করবেন"
2. `AutoModelForCausalLM` / `AutoTokenizer`: একটি বাস্তব pretrained checkpoint load করা
3. `LoraConfig` field-ধরে-ফিল্ড, সরাসরি Lesson 2-এর `LoRALinear`-এ ম্যাপ করা
4. Data ফরম্যাটিং: Alpaca-ধাঁচের templates ও প্রকৃত chat templates
5. `SFTTrainer.train()`: Lesson 4-এর masked loss, একটিমাত্র কল হিসেবে চলছে
6. `merge_and_unload()`: Lesson 2-এর merge-for-free বৈশিষ্ট্য, এখন সত্যি করে
7. `example.py`-র মার্জিত ImportError-হ্যান্ডলিং-এর ওয়াকথ্রু, আর এই কোর্স আগে স্ক্র্যাচ-সংস্করণটি শেখায় কেন
8. Recap + preview: Lesson 6 এই পুরো stack-টি একটি কংক্রিট domain-adaptation case study-তে প্রয়োগ করে

## আরও পড়ার জন্য

- Hugging Face `transformers` documentation, `AutoModelForCausalLM` and `Trainer`
- Hugging Face `peft` documentation, `LoraConfig` and `get_peft_model`
- Hugging Face `trl` documentation, `SFTTrainer` and `SFTConfig`
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (`LoraConfig` যে পদ্ধতিটি বাস্তবায়ন করে — পূর্ণ ব্যুৎপত্তি [Lesson 2](../02-LoRA-and-QLoRA/README.md)-তে)
- von Werra et al. (2020), *TRL: Transformer Reinforcement Learning* (লাইব্রেরিটির উৎস — এখন SFT, DPO ও RLHF-ধাঁচের training কভার করে; DPO আবার দেখা হবে [Phase 06 Lesson 4](../../Phase-06-Alignment-and-RLHF/04-Direct-Preference-Optimization-DPO/README.md)-এ)