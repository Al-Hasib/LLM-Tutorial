# Neural Networks-এর মৌলিক বিষয়

**পর্ব:** [পূর্বশর্ত](../README.md) · **টপিক ফোল্ডার:** `02-Neural-Networks-Basics`

## কেন এটি গুরুত্বপূর্ণ

Transformer "শুধুই" একটি খুব গভীর, খুব নির্দিষ্ট বিন্যাস — যার building blockগুলো তুমি এখানেই শিখতে যাচ্ছ: weighted sum, non-linear activation এবং gradient descent। যদি বুঝতে পারো একটি 2-layer network হাতে-কলমে কীভাবে XOR শেখে, তাহলে — যান্ত্রিকভাবে — বুঝেই গেছো কীভাবে একটি 100-billion-parameter model next token পূর্বাভাস দিতে শেখে। Scale বদলায়; mechanism বদলায় না।

## এই পাঠে কী শেখা হবে

- perceptron: neural network-এর পারমাণবিক একক
- Activation functions এবং কেন non-linearity আলোচনার অবকাশহীন
- multi-layer perceptron (MLP)-এর forward pass
- Loss functions: MSE ও cross-entropy
- Backpropagation: chain rule, layer-পর-পর প্রয়োগ
- Gradient descent এবং এর variants

## 1. perceptron

একটি perceptron তার inputs-এর weighted sum হিসাব করে, একটি bias যোগ করে এবং একটি activation function প্রয়োগ করে:

```
z = w · x + b
a = φ(z)
```

- `x` — input vector
- `w` — শেখা weight vector
- `b` — শেখা bias (scalar)
- `φ` — activation function
- `a` — neuron-এর আউটপুট ("activation")

অনেকগুলো perceptron পাশাপাশি সাজালে পাবে একটি **layer**; layer সাজালে পাবে একটি **multi-layer perceptron (MLP)**।

## 2. Activation functions

non-linear `φ` ছাড়া layer স্তূপীকরণ অর্থহীন — linear functions-এর composition শেষ পর্যন্ত একটি linear function-ই, তাই activation ছাড়া একটি 1000-layer network একটি single layer-এর চেয়ে বেশি অভিব্যক্তিপূর্ণ নয়। সাধারণ পছন্দগুলো:

| ফাংশন    | ফর্মুলা                                        | নোট                                                                                                      |
| -------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Sigmoid  | `σ(z) = 1 / (1 + e⁻ᶻ)`                         | `(0, 1)`-এ চেপে আনে; ঐতিহাসিকভাবে জনপ্রিয়, বড় \|z\|-এর জন্য saturate হয়ে gradient নষ্ট করে |
| Tanh     | `tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)`           | `(-1, 1)`-এ চেপে আনে; zero-centered, তবুও saturate করে                                            |
| ReLU     | `max(0, z)`                                     | সস্তা, `z > 0`-এর জন্য saturate করে না; বেশিরভাগ deep net-এর জন্য default                                 |
| GELU     | `z · Φ(z)` (Φ = Gaussian CDF)                   | Smooth ReLU ভ্যারিয়েন্ট; Phase 02-তে তুমি যে প্রতিটি Transformer feed-forward block বানাবে তার ভেতরে ব্যবহৃত |

## 3. MLP-এর ভেতর দিয়ে forward pass

`1..L` layer-বিশিষ্ট network-এর জন্য:

```
a⁽⁰⁾ = x
z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = φ(z⁽ˡ⁾)
ŷ    = a⁽ᴸ⁾
```

প্রতিটি layer-এর weight matrix `W⁽ˡ⁾`-এর shape হলো `(units_in_layer_l, units_in_layer_l-1)` — [Python and Math Refresher](../01-Python-and-Math-Refresher/README.md)-এ দেখানো matrix multiplication-এর shape নিয়মগুলোর হুবহু পুনরাবৃত্তি।

## 4. Loss functions

- **Mean Squared Error (regression)**: `L = (1/n) Σ (ŷᵢ − yᵢ)²`
- **Cross-entropy (classification)**: `L = −Σ yᵢ log(ŷᵢ)` — প্রতিটি LLM যে next-token loss ব্যবহার করে সেটিই, শুধু কয়েকটি class-এর বদলে vocabulary-আকারের আউটপুটের উপর।

## 5. Backpropagation

Backprop হলো chain rule ([Python and Math Refresher, §3](../01-Python-and-Math-Refresher/README.md#3-calculus)) — যা loss থেকে পেছনের দিকে পদ্ধতিগতভাবে প্রতিটি weight-এ প্রয়োগ করা হয়:

1. forward pass হিসাব করো, প্রতিটি মধ্যবর্তী `z⁽ˡ⁾` ও `a⁽ˡ⁾` cache করে রাখো।
2. আউটপুটে `∂L/∂a⁽ᴸ⁾` হিসাব করো।
3. `L` থেকে নিচের দিকে `1` পর্যন্ত প্রতিটি layer `l`-এর জন্য:
   - `∂L/∂z⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ ⊙ φ'(z⁽ˡ⁾)` (⊙ = elementwise product)
   - `∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · a⁽ˡ⁻¹⁾ᵗ`
   - `∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾`
   - `∂L/∂a⁽ˡ⁻¹⁾ = W⁽ˡ⁾ᵗ · ∂L/∂z⁽ˡ⁾` (ত্রুটিটিকে আরও এক layer পেছনে প্রেরণ)

প্রতিটি deep learning framework (PyTorch-ও সহ — দেখুন [PyTorch Fundamentals](../04-PyTorch-Fundamentals/README.md)) **autograd**-এর মাধ্যমে ঠিক এই recursion-টিকেই স্বয়ংক্রিয় করে, তাই এই lesson-এর পরে আর কখনো হাতে derivation করতে হবে না — তবে কমপক্ষে একবার raw code-এ এটি ঘটতে দেখা উচিত।

## 6. Gradient descent

প্রতিটি parameter `θ`-এর জন্য `∂L/∂θ` পেলে আমরা তা ঢাল বরাবর নিচের দিকে ঠেলে দিই:

```
θ ← θ − η · ∂L/∂θ
```

`η` (eta) হলো **learning rate**। কোর্সের পরে তুমি যে variants আবার দেখবে:

- **Batch GD**: প্রতি update-এ পুরো dataset ব্যবহার (ধীর, স্থিতিশীল)
- **Stochastic GD (SGD)**: প্রতি update-এ একটি মাত্র example (দ্রুত, noisy)
- **Mini-batch SGD**: প্রতি update-এ একটি ছোট batch — বাস্তবে যা ব্যবহৃত হয়, LLM-ও সহ
- **Momentum / Adam**: past gradients-এর চলমান গড় জমা করে convergence মসৃণ ও দ্রুততর করা (LLM প্রশিক্ষণের জন্য Adam/AdamW-ই default optimizer — [Phase 04: Mixed Precision and Optimization](../../Phase-04-Pretraining-LLMs/04-Mixed-Precision-and-Optimization/README.md)-তে আলোচিত)

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "একটি LLM layer হলো কেবল এটিই, পুনরাবৃত্ত ও স্কেল করা"
2. perceptron → activation functions, প্রতিটি activation-এর shape-এর plot সহ
3. MLP-এ স্তূপীকরণ, forward-pass equations-এর মধ্য দিয়ে হাঁটা
4. হোয়াইটবোর্ডে একটি ছোট 2-layer network-এর জন্য backprop derivation
5. `example.py`-এর walkthrough — XOR-এ scratch থেকে তৈরি NumPy MLP train করা, loss curve নেমে যাওয়া দেখা
6. সংক্ষিপ্ত পুনরালোচনা + পরবর্তী Introduction to NLP-এর দিকে ইঙ্গিত

## আরও পড়ুন

- Michael Nielsen, *Neural Networks and Deep Learning* (ফ্রি অনলাইন বই)
- 3Blue1Brown — *Neural Networks* ভিডিও সিরিজ (বিশেষত backpropagation পর্বগুলো)
- Andrej Karpathy — *The spelled-out intro to neural networks and backpropagation: building micrograd*
- Goodfellow, Bengio, Courville — *Deep Learning*, Ch. 6