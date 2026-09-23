# PyTorch-এর মৌলিক বিষয়

**পর্ব:** [পূর্বশর্ত](../README.md) · **টপিক ফোল্ডার:** `04-PyTorch-Fundamentals`

## কেন এটি গুরুত্বপূর্ণ

এই কোর্সের বাকি প্রতিটি `example.py` — হাতে তৈরি mini-GPT থেকে Hugging Face দিয়ে fine-tuning পর্যন্ত — সব PyTorch-এ লেখা। তুমি এইমাত্র [Neural Networks Basics](../02-Neural-Networks-Basics/README.md)-এ হাতে-কলমে একটি MLP-এর forward pass ও backward pass বানিয়েছিলে; এই lesson দেখায় ঠিক একই হিসাব, কিন্তু এমন একটি framework দিয়ে যা backward pass *তোমার জন্য*-ই হিসাব করে (autograd), এবং forward pass-কে প্যাকেজ করে তোলে পুনঃব্যবহারযোগ্য, রচনাযোগ্য modules-এ। একবার tensors, autograd ও `nn.Module`-এ সাবলীল হলে, যেকোনো model-এর source code পড়া — প্রকৃত Transformer-ও বাদ যায় না — শূন্য থেকে শুরু করার বদলে pattern চেনার ব্যাপার হয়ে দাঁড়ায়।

## এই পাঠে কী শেখা হবে

- Tensors: তৈরি, shape, dtype, device, indexing, broadcasting
- Autograd: `requires_grad`, `.backward()`, `.grad`, computational graph
- `nn.Module`: পুনঃব্যবহারযোগ্য, রচনাযোগ্য model components গড়া
- `torch.nn` / `torch.optim` থেকে loss functions ও optimizers
- মানক PyTorch training loop টেমপ্লেট
- Device-agnostic code (CPU/GPU)
- সাধারণ gotchas

## 1. Tensors

`torch.Tensor` হলো NumPy-র `ndarray` প্লাস দুটি সুপারপাওয়ার: এটি GPU-তে থাকতে পারে, এবং automatic differentiation-এর জন্য তার উপর প্রয়োগ হওয়া operations ট্র্যাক করতে পারে।

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])       # from a Python list
z = torch.zeros(2, 3)                    # shape (2, 3) of zeros
r = torch.randn(2, 3)                    # random normal values
print(x.shape, x.dtype, x.device)        # torch.Size([3]) torch.float32 cpu
```

Shapes ও broadcasting নিয়ম NumPy-র মতোই — [Python and Math Refresher](../01-Python-and-Math-Refresher/README.md)-এর linear algebra অংশের সবকিছু সরাসরি প্রযোজ্য।

## 2. Autograd

একটি leaf tensor-এ `requires_grad=True` সেট করো, আর PyTorch তোমার হিসাব চলাকালীন নীরবে একটি **computational graph** তৈরি করে। `.backward()` কল করলে সেই graph-টি উল্টো দিকে হেঁটে প্রতিটি leaf-এ `.grad` পূরণ করে — এটি [Neural Networks Basics §5](../02-Neural-Networks-Basics/README.md#5-backpropagation)-এর chain-rule recursion-ই, সম্পূর্ণ স্বয়ংক্রিয়ভাবে:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # tensor(6.) == dy/dx == 2x at x=3, matches the manual math exactly
```

- `.detach()` — graph ছাড়াই tensor-এর values পাওয়া (gradient-কে আরও পেছনে প্রবাহিত হতে বাধা দেওয়া)
- `torch.no_grad():` — একটি context manager যা graph-নির্মাণ সম্পূর্ণ বন্ধ করে দেয় (inference/evaluation-এর সময় ব্যবহৃত, যেখানে gradient দরকার নেই এবং তাদের memory খরচও চাওয়া হয় না)

## 3. `nn.Module`

PyTorch-এর প্রতিটি layer ও প্রতিটি model `nn.Module`-এর subclass। `__init__`-এ learnable অংশগুলো আর `forward`-এ forward computation সংজ্ঞায়িত করো:

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x))))
```

`nn.Linear(n_in, n_out)` একটি weight matrix ও bias vector-কে একত্রিত করে — MLP lesson-এর `W`, `b`-ই — এবং সেগুলো register করে, যাতে module-এর `.parameters()` optimizer-এর জন্য সেগুলো স্বয়ংক্রিয়ভাবে খুঁজে পায়।

## 4. Loss functions এবং optimizers

```python
criterion = nn.MSELoss()                                 # or nn.CrossEntropyLoss() for classification / next-token prediction
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # AdamW is the default for LLM training later in the course
```

## 5. মানক training loop

এই repository-র প্রতিটি training script — model যত বড়ই হোক — ঠিক এই পাঁচ লাইনের কাঠামো অনুসরণ করে:

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()        # 1. clear old gradients
    output = model(x)            # 2. forward pass
    loss = criterion(output, y)  # 3. compute the loss
    loss.backward()              # 4. backward pass (autograd fills .grad)
    optimizer.step()             # 5. gradient descent update
```

এটিকে [Neural Networks Basics&#39; `example.py`](../02-Neural-Networks-Basics/example.py)-এ দেখা হাতে-লেখা `forward` / `backward` / gradient-update কলগুলোর সাথে লাইন-বাই-লাইন মিলিয়ে দেখো — এটি অভিন্ন algorithm, শুধু 3–5 ধাপগুলো স্বয়ংক্রিয়।

## 6. Device-agnostic code

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

এভাবে code লেখা মানে একই script ল্যাপটপের CPU-তে train হয়, এবং (কোনো পরিবর্তন ছাড়াই) কোর্সের পরে multi-GPU training cluster-এও।

## 7. সাধারণ gotchas

- **`optimizer.zero_grad()` ভুলে যাওয়া** — default-এ `.backward()`-এর কলগুলোর মধ্যে gradients জমা (যোগ) হতে থাকে, তাই এটি বাদ দিলে নীরবে training নষ্ট হয়ে যায়।
- **grad-প্রয়োজন এমন tensors-এ in-place ops** (যেমন `x += 1`) autograd graph ভেঙে দিতে পারে — forward pass-এর ভেতরে out-of-place ops (`x = x + 1`) পছন্দ করো।
- **অসাবধানে NumPy ও Tensors মেশানো** — ইচ্ছা করেই `.numpy()` / `torch.from_numpy()` ব্যবহার করো, আর মনে রাখো tensor-টিতে যদি এখনও `requires_grad` সত্য থাকে তবে `.numpy()` ব্যর্থ হয় (আগে `.detach()` কল করো)।
- **loss function-এ ভুল shape দেওয়া** — `nn.CrossEntropyLoss` প্রত্যাশা করে raw logits (softmax output নয়) এবং integer class indices, one-hot vectors নয়; framework বদলানোর শুরুতে এটি খুব সাধারণ bug।

## ভিডিও স্ক্রিপ্ট আউটলাইন

1. Motivation — "তুমি এইমাত্র হাতে করেছ; এখানে সেই framework যা এটিকে স্বয়ংক্রিয় করে"
2. Tensors: তৈরি, shape, device — REPL-এ লাইভ
3. Autograd: `x**2` উদাহরণ, Lesson 1-এর ম্যানুয়াল derivative-এর সাথে `x.grad` তুলনা
4. `nn.Module`: Lesson 2-এর XOR network পুনর্নির্মাণ, তবে `nn.Linear` + autograd দিয়ে
5. `example.py`-এর training loop-এর walkthrough, ম্যানুয়াল সংস্করণের সাথে পাশাপাশি
6. সংক্ষিপ্ত পুনরালোচনা: "পরের প্রতিটি lesson-এর code এ রকমই হবে" + Phase 01-এর দিকে ইঙ্গিত

## আরও পড়ুন

- অফিসিয়াল PyTorch টিউটোরিয়াল: *Deep Learning with PyTorch: A 60 Minute Blitz*
- PyTorch docs: `torch.autograd`, `torch.nn`, `torch.optim`
- Karpathy, *A Recipe for Training Neural Networks* (ব্যবহারিক debugging অভ্যাস, যা এখান থেকে ভবিষ্যতের প্রতিটি lesson পর্যন্ত প্রযোজ্য)