# Model Distillation এবং Pruning

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `05-Model-Distillation-and-Pruning`

## কেন এটি গুরুত্বপূর্ণ

[Quantization](../02-Quantization/README.md) *একই* weights-কে কম bits দিয়ে উপস্থাপন করে model ছোট করে। এই lesson সেই দুটি কৌশল কভার করে যা বদলে weights সরিয়ে (অথবা প্রথম থেকেই একটি ছোট model প্রশিক্ষণ দিয়ে) model ছোট করে — একই "ছোট ও সস্তা করো" সমস্যার উপর একটি পরিপূরক, প্রতিযোগী নয়, lever-এর সেট। আপনি ইতিমধ্যে নামের দ্বারা একটি বাস্তব distilled model দেখা করেছেন, [Phase 03 Lesson 2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#6-roberta-and-todays-encoder-only-landscape)-এ — DistilBERT — তবে তখনো সেই mechanism-টি দেখেননি যা একে কাজ করে। এই lesson সেই mechanism-টিকে স্ক্র্যাচ থেকে গড়ে তোলে।

## এই lesson যা কভার করে

- Knowledge distillation: একটি ছোট student-কে প্রশিক্ষণ দিয়ে একটি বড় teacher-এর soft outputs-কে মেলানো
- কেন soft targets hard labels-এর চেয়ে বেশি signal বহন করে ("dark knowledge")
- Structured বনাম unstructured pruning
- Magnitude-based pruning: সরলতম বাস্তব অ্যালগরিদম
- Accuracy-vs-sparsity trade-off curve

## 1. Knowledge distillation

Hinton, Vinyals, Dean (2015) একটি পর্যবেক্ষণ দিয়ে শুরু: একটি প্রশিক্ষিত "teacher" network-এর classes-এর উপর output distribution "কোন class সঠিক" এর চেয়ে বেশি তথ্য বহন করে। ধরা যাক একটি teacher একটি "2"-এর ছবি classifying করছে এবং "7"-কেও অ-তুচ্ছ সম্ভাবনা দিচ্ছে (তারা অনুরূপ উপরের stroke ভাগ করে) কিন্তু "9"-কে প্রায় কিছুই নয়: *ভুল* classes জুড়ে সেই আপেক্ষিক আত্মবিশ্বাস প্রকৃত, শেখা সাদৃশ্য কাঠামো প্রতিফলিত করে — Hinton et al. একে **dark knowledge** বলে — এবং একটি one-hot hard label ("উত্তর হলো 2") এটি সম্পূর্ণরূপে ফেলে দেয়।

Distillation একটি ছোট **student** network-কে শুধু তার শীর্ষ পূর্বাভাস নয়, teacher-এর সম্পূর্ণ output distribution মেলাতে প্রশিক্ষণ দেয়, বিতরণকে নরম করতে এবং সেই আপেক্ষিক কাঠামোর আরও বেশি প্রকাশ করতে একটি **temperature-scaled softmax** ব্যবহার করে:

```
soft_target = softmax(teacher_logits / T)      # T > 1 "softens" the distribution, revealing more of the runner-up classes
soft_prediction = softmax(student_logits / T)
distillation_loss = KL_divergence(soft_target, soft_prediction)
total_loss = alpha * distillation_loss * T^2 + (1 - alpha) * hard_label_cross_entropy
```

উচ্চতর temperature `T` বিতরণকে আরও চ্যাপ্টা করে (`T -> infinity`-তে, প্রতিটি class অভিন্ন সম্ভাবনার কাছে পৌঁছায়), teacher-এর আপেক্ষিক-আত্মবিশ্বাস কাঠামোর বেশি প্রকাশ করে — খুব-কম-সম্ভাব্যতা classes-এ শব্দকে প্রশস্ত করার খরচে — `T` একটি tunable hyperparameter, কোনো নির্দিষ্ট ধ্রুবক নয়। `T^2` ফ্যাক্টরটি distillation loss-এর gradient magnitude-কে hard-label term-এর সাথে তুলনীয় থাকার জন্য পুনঃস্কেল করে, যেহেতু logits-কে `T` দিয়ে স্কেল করলেও gradients প্রায় `T` দিয়ে স্কেল হয়।

## 2. Structured বনাম unstructured pruning

Pruning প্রথম থেকেই একটি ছোট প্রশিক্ষণ দেওয়ার বদলে একটি ইতিমধ্যে-প্রশিক্ষিত network থেকে parameters সরিয়ে দেয়:

- **Structured pruning** সম্পূর্ণ, architectural-অর্থবহ ইউনিট সরিয়ে দেয় — সম্পূর্ণ neurons, attention heads, বা layers। ফলাফল একটি ছোট *dense* network যা কোনো বিশেষ সমর্থন ছাড়াই সাধারণ hardware-এ চলে, কিন্তু coarse-grained অপসারণ (একবারে একটি সম্পূর্ণ head বা layer) প্রতি-অপসারিত-parameter-এ সূক্ষ্ম-grained pruning-এর চেয়ে বেশি accuracy খরচ করে tend করে।
- **Unstructured pruning** একটি weight matrix-এর যেকোনো জায়গায়, position নির্বিশেষে, স্বতন্ত্র weights শূন্য করে দেয়। এটি একটি প্রদত্ত accuracy loss-এর জন্য অনেক বেশি সংকোচন অর্জন করে, কারণ এটি ঠিক সবচেয়ে কম-দরকারী স্বতন্ত্র সংযোগগুলো অপসারণ করতে পারে — কিন্তু একটি matrix যেটি বেশিরভাগই যত্রতত্র ছড়ানো zeros, তবু *sparse-matrix-aware* hardware/kernels দরকার হয় যাতে আসলেই দ্রুত বা memory-তে ছোট চলে; এটিকে dense সংরক্ষণ করা সাশ্রয় সম্পূর্ণরূপে নষ্ট করে।

## 3. Magnitude pruning

সরলতম বাস্তব unstructured-pruning অ্যালগরিদম, এবং এখনো একটি শক্তিশালী baseline: প্রতিটি weight-কে তার absolute value দ্বারা rank করুন, এবং ক্ষুদ্রতম ভগ্নাংশটি শূন্য করুন (the **sparsity level**, যেমন সমস্ত weights-এর 80% ঠিক শূন্যে সেট)। অন্তর্দৃষ্টি হলো শূন্যের কাছাকাছি একটি weight ইতিমধ্যে network-এর output-এ সামান্যই অবদান রাখে, তাই এটি সরানো আচরণকে সবচেয়ে কম ব্যাহত করবে — একটি অন্তর্দৃষ্টি যা বাস্তবে আশ্চর্যজনকভাবে ভালোভাবে ধরে, বিশেষ করে মধ্যপন্থী sparsity levels-এ, যদিও sparsity যথেষ্ট উঁচুতে উঠলে accuracy অনিবার্যভাবে ক্ষয় হয় — যে weights-গুলো *সত্যিই* গুরুত্বপূর্ণ সেগুলো সরানো শুরু হলে।

## 4. Accuracy-vs-sparsity trade-off

কোনো ফ্রি lunch নেই: কিছু sparsity প্রায় বিনামূল্যে (অপ্রয়োজনীয় capacity ন্যূনতম accuracy খরচে সরানো হয়), কিন্তু একটি বিন্দুর পরে sparsity-র প্রতিটি অতিরিক্ত শতাংশ measurably বেশি accuracy খরচ করে, এবং curve-টিতে সাধারণত একটি "knee" থাকে — একটি sparsity level যার বাইরে accuracy ধীরে ধীরে নয়, তীক্ষ্ণভাবে অবনত হয়। `example.py` §2 এই curve-টিকে একটি বাস্তব প্রশিক্ষিত toy network-এ, কয়েকটি sparsity levels-এ, অনুমিত আকৃতির বদলে প্রকৃত সংখ্যা-সহ সরাসরি মাপে।

## Video Script Outline

1. Motivation — "পরিমাণ হ্রাসের চেয়ে weights সরিয়ে ছোট করা, অথবা প্রথম থেকেই ছোট প্রশিক্ষণ"
2. Dark knowledge: কেন একটি ভুল-class সম্ভাবনা এখনো এমন signal বহন করে যা একটি hard label ফেলে দেয়
3. Distillation loss, temperature scaling, এবং `T^2` rescaling বিবরণ
4. Structured বনাম unstructured pruning, এবং sparse-hardware caveat
5. Magnitude pruning-এর অ্যালগরিদম, এক বাক্যে
6. `example.py` §1-এর walkthrough — একটি teacher প্রশিক্ষণ, তারপর একটি student দুভাবে (hard labels alone বনাম distillation), প্রকৃত held-out accuracy তুলনা
7. `example.py` §2-এর walkthrough — একটি প্রশিক্ষিত network-কে কয়েকটি sparsity levels জুড়ে magnitude-prune, প্রকৃত accuracy-vs-sparsity curve প্লট
8. Recap + pointer [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)-এর দিকে, যেখানে একটি ছোট distilled/pruned model একটি routing cascade-র "সস্তা" স্তর হয়ে ওঠে

## Further Reading

- Hinton, Vinyals, Dean (2015), *Distilling the Knowledge in a Neural Network*
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
- Han, Mao, Dally (2015), *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* (magnitude pruning at scale, combined with quantization)
- Frankle & Carbin (2019), *The Lottery Ticket Hypothesis* (why some sparse subnetworks train just as well as the full dense network)