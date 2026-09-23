# Python ও গণিত রিফ্রেশার

**পর্ব:** [পূর্বশর্ত](../README.md) · **টপিক ফোল্ডার:** `01-Python-and-Math-Refresher`

## কেন এটি গুরুত্বপূর্ণ

এই কোর্সের পরের প্রতিটি lesson মূলত vector, matrix, derivative ও probability-কে code-এর ভেতর দিয়ে চালনা করার ব্যাপার। LLM-কে বুঝতে তোমার math-এ ডিগ্রি লাগবে না, তবে linear algebra, calculus ও probability সম্পর্কে যথেষ্ট *intuition* দরকার — যেন কোনো paper-এর equation পড়েই সাথে সাথে কল্পনা করতে পারো কোন আকারের data কোথায় প্রবাহিত হচ্ছে। এই lesson ঠিক সেই toolbox-ই তৈরি করে, সাথে এই repo-র প্রতিটি `example.py`-তে বারবার ব্যবহৃত সেই কয়েকটি Python idiom-ও শেখায়।

## এই পাঠে কী শেখা হবে

- পুরো কোর্স জুড়ে ব্যবহৃত Python idioms (comprehension, unpacking, class)
- Linear algebra: vector, matrix, dot product, matrix multiplication, norm
- Calculus: derivative, partial derivative, gradient, chain rule
- Probability: distribution, expectation, softmax, log-likelihood
- বাকি সিরিজের জন্য একটি ভাগ করা notation cheat sheet

## 1. Python idioms যা বারবার দেখতে পাবে

```python
# List/dict comprehensions instead of manual loops
squares = [x ** 2 for x in range(5)]
vocab_to_id = {word: i for i, word in enumerate(["the", "cat", "sat"])}

# enumerate + zip for parallel iteration
for i, (word, idx) in enumerate(zip(["a", "b"], [0, 1])):
    ...

# Unpacking
batch_size, seq_len, hidden_dim = tensor.shape

# A minimal class, the shape every nn.Module will follow later
class Layer:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError
```

উপরের `__call__` → `forward` প্যাটার্নটি ঠিক এভাবেই `torch.nn.Module` কাজ করে — আমরা এখন থেকেই [PyTorch Fundamentals](../04-PyTorch-Fundamentals/README.md)-এর জন্য মানসিক মডেলটি গড়ে তুলছি।

## 2. Linear algebra (লিনিয়ার অ্যালজেব্রা)

LLM-এর মূলে রয়েছে বিপুল সংখ্যক matrix multiplication, যার মধ্যে ছড়িয়ে আছে কিছু non-linearity। এই ছোট toolkit-টিতে তোমাকে সাবলীল হতে হবে:

- **Scalar**: একটি একক সংখ্যা।
- **Vector** `v ∈ ℝⁿ`: সংখ্যার একটি সাজানো তালিকা (যেমন: একটি word embedding)।
- **Matrix** `A ∈ ℝᵈˣⁿ`: rows × columns (যেমন: একটি weight matrix, যা `n`-dimension ইনপুটকে `d`-dimension আউটপুটে ম্যাপ করে)।
- **Tensor**: 3+ dimension-এ generalization (যেমন: `[batch, sequence, hidden]`)।

**একই দৈর্ঘ্যের দুটি vector-এর Dot product:**

```
a · b = Σᵢ aᵢ bᵢ
```

এই একটি মাত্র operation-ই attention-এর প্রাণকেন্দ্র: "এই দুটি vector কতটা একই রকম" তা পরিমাপ করা হয় dot product দিয়ে।

**Matrix multiplication:** `A (d×n) @ B (n×m) = C (d×m)`। ভেতরের dimensions অবশ্যই মিলতে হবে — এটি প্রতিটি neural network-এ সবচেয়ে সাধারণ bug-এর উৎস, তাই value যাচাইয়ের আগে shape যাচাই করার অভ্যাস গড়ে তোলো।

**Transpose** `Aᵗ` rows ও columns অদলবদল করে — shape মেলানোর জন্য এটি প্রতিনিয়ত ব্যবহৃত হয় (যেমন attention-এ `Q @ Kᵗ`)।

**Norm** একটি vector-এর "আকার" পরিমাপ করে:

- L2 (Euclidean): `‖v‖₂ = √(Σ vᵢ²)`
- L1: `‖v‖₁ = Σ |vᵢ|`

## 3. Calculus (ক্যালকুলাস)

Neural network প্রশিক্ষণ মানে হলো: একটি loss হিসাব করা, তারপর বের করা প্রতিটি parameter কোন দিক বরাবর সরালে loss ছোট হবে। "কোন দিক" — এই প্রশ্নের উত্তর সম্পূর্ণ derivative দিয়ে দেওয়া হয়।

- **Derivative** `f'(x)`: `x` বিন্দুতে `f`-এর তাৎক্ষণিক পরিবর্তনের হার।
- **Partial derivative** `∂f/∂x`: অন্যগুলো স্থির রেখে *একটি* variable-এর সাপেক্ষে পরিবর্তনের হার — এটি প্রয়োজন, কারণ একটি model-এ millions/billions parameter থাকে।
- **Gradient** `∇f`: সব partial derivative-এর vector — এটি সবচেয়ে খাড়া বৃদ্ধির দিকে নির্দেশ করে, তাই training এর *বিপরীত* দিকে চলে (gradient **descent**)।
- **Chain rule**: যদি `y = f(g(x))` হয়, তবে `dy/dx = f'(g(x)) · g'(x)`। Backpropagation (পরের lesson) আসলে পুরো network জুড়ে layer-পর-পর স্বয়ংক্রিয়ভাবে প্রয়োগ করা chain rule ছাড়া আর কিছুই নয়।

## 4. Probability (সম্ভাবনা)

- কোনো outcomes সেটের উপর **probability distribution** প্রতিটি outcome-কে একটি non-negative weight দেয়, যেগুলোর যোগফল 1।
- **Expectation** `E[X] = Σ x · P(x)`: likelihood দিয়ে weighted গড় outcome।
- **Softmax** যেকোনো real-valued score ("logits")-কে probability distribution-এ রূপান্তর করে:

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

প্রতিটি LLM তার forward pass শেষ করে vocabulary-র উপর softmax দিয়ে — তখন "logits" পরিণত হয় "প্রতিটি সম্ভাব্য next token-এর probability"।

- **Log-likelihood / cross-entropy**: model সঠিক token-কে যে probability দেয় তা সর্বোচ্চ করার বদলে আমরা (সমতুল্য ও সংখ্যাগতভাবে আরও স্থিতিশীল উপায়ে) সেই probability-র negative log-কে সর্বনিম্ন করি। এই কোর্সে তুমি যে প্রায় প্রতিটি LLM train করবে, তার loss function-ই এটি।

## Notation cheat sheet (পুরো repo জুড়ে ব্যবহৃত)

| প্রতীক      | অর্থ                             |
| ----------- | -------------------------------- |
| `x`, `v`    | scalar / vector                  |
| `W`, `A`    | matrix                           |
| `Wᵗ`        | `W`-এর transpose                 |
| `d_model`   | hidden/embedding dimension       |
| `n`, `T`    | sequence length                  |
| `∇`         | gradient                         |
| `θ`         | model parameters (weights)       |
| `L`         | loss                             |

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "an LLM হলো স্কেলে linear algebra + calculus + probability মাত্র," প্রতিটি অংশ কেন গুরুত্বপূর্ণ তার পূর্বাভাস
2. Shape diagram সহ linear algebra walkthrough (vectors → matrices → dot products)
3. Calculus: derivative → gradient → chain rule, সরাসরি "training কীভাবে কাজ করে"-এর সাথে সম্পর্কিত
4. Probability: softmax ও log-likelihood, সরাসরি "model কীভাবে next-token অনুমান আউটপুট করে"-এর সাথে সম্পর্কিত
5. `example.py`-এর walkthrough — প্রতিটি concept বাস্তব code হিসেবে চালানো
6. সংক্ষিপ্ত পুনরালোচনা + স্ক্রিনে notation cheat sheet + পরবর্তী Neural Networks Basics-এর দিকে ইঙ্গিত

## আরও পড়ুন

- 3Blue1Brown — *Essence of Linear Algebra* ও *Essence of Calculus* ভিডিও সিরিজ
- Gilbert Strang, *Introduction to Linear Algebra*
- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (ফ্রি PDF)
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Ch. 1–2 (probability রিফ্রেশার)