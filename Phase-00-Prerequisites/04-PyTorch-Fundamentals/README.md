# PyTorch Fundamentals

**Phase:** [Prerequisites](../README.md) · **Topic folder:** `04-PyTorch-Fundamentals`

## Why this matters

Every remaining `example.py` in this course — from a mini-GPT built by hand to fine-tuning with Hugging Face — is written in PyTorch. You just built an MLP's forward pass and backward pass by hand in [Neural Networks Basics](../02-Neural-Networks-Basics/README.md); this lesson shows you the exact same computation done by a framework that computes the backward pass *for you* (autograd), and packages the forward pass into reusable, composable modules. Once you're fluent in tensors, autograd, and `nn.Module`, reading any model's source code — including a real Transformer's — becomes a matter of pattern recognition rather than starting from zero.

## What this lesson covers

- Tensors: creation, shape, dtype, device, indexing, broadcasting
- Autograd: `requires_grad`, `.backward()`, `.grad`, the computational graph
- `nn.Module`: building reusable, composable model components
- Loss functions and optimizers from `torch.nn` / `torch.optim`
- The standard PyTorch training loop template
- Device-agnostic code (CPU/GPU)
- Common gotchas

## 1. Tensors

A `torch.Tensor` is NumPy's `ndarray` plus two superpowers: it can live on a GPU, and it can track the operations applied to it for automatic differentiation.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])       # from a Python list
z = torch.zeros(2, 3)                    # shape (2, 3) of zeros
r = torch.randn(2, 3)                    # random normal values
print(x.shape, x.dtype, x.device)        # torch.Size([3]) torch.float32 cpu
```

Shapes and broadcasting rules are identical to NumPy — everything from the [Python and Math Refresher](../01-Python-and-Math-Refresher/README.md) linear algebra section carries over directly.

## 2. Autograd

Set `requires_grad=True` on a leaf tensor, and PyTorch silently builds a **computational graph** as you compute with it. Calling `.backward()` walks that graph in reverse and fills in `.grad` on every leaf — this *is* the chain-rule recursion from [Neural Networks Basics §5](../02-Neural-Networks-Basics/README.md#5-backpropagation), fully automated:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # tensor(6.) == dy/dx == 2x at x=3, matches the manual math exactly
```

- `.detach()` — get a tensor's values without the graph (stop gradients from flowing further back)
- `torch.no_grad():` — a context manager that disables graph-building entirely (used during inference/evaluation, where you don't need gradients and don't want to pay their memory cost)

## 3. `nn.Module`

Every layer and every model in PyTorch subclasses `nn.Module`. You define the learnable pieces in `__init__` and the forward computation in `forward`:

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

`nn.Linear(n_in, n_out)` bundles a weight matrix and bias vector — exactly `W`, `b` from the MLP lesson — and registers them so the module's `.parameters()` finds them automatically for the optimizer.

## 4. Loss functions and optimizers

```python
criterion = nn.MSELoss()                                 # or nn.CrossEntropyLoss() for classification / next-token prediction
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # AdamW is the default for LLM training later in the course
```

## 5. The standard training loop

Every training script in this repository — no matter how large the model gets — follows this exact five-line skeleton:

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()        # 1. clear old gradients
    output = model(x)            # 2. forward pass
    loss = criterion(output, y)  # 3. compute the loss
    loss.backward()              # 4. backward pass (autograd fills .grad)
    optimizer.step()             # 5. gradient descent update
```

Compare this line-by-line to the manual `forward` / `backward` / gradient-update calls in [Neural Networks Basics' `example.py`](../02-Neural-Networks-Basics/example.py) — it's the identical algorithm, just with steps 3-5 automated.

## 6. Device-agnostic code

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

Writing code this way means the exact same script trains on a laptop CPU and (unchanged) on a multi-GPU training cluster later in the course.

## 7. Common gotchas

- **Forgetting `optimizer.zero_grad()`** — gradients accumulate (add up) across calls to `.backward()` by default, so skipping this silently corrupts training.
- **In-place ops on tensors that require grad** (e.g. `x += 1`) can break the autograd graph — prefer out-of-place ops (`x = x + 1`) inside a forward pass.
- **Mixing NumPy and Tensors carelessly** — use `.numpy()` / `torch.from_numpy()` deliberately, and remember `.numpy()` fails on a tensor that still `requires_grad` (call `.detach()` first).
- **Wrong shape going into a loss function** — `nn.CrossEntropyLoss` expects raw logits (not softmax output) and integer class indices, not one-hot vectors; a very common bug when first switching frameworks.

## Video Script Outline

1. Motivation — "you just did this by hand; here's the framework that automates it"
2. Tensors: creation, shape, device — live in a REPL
3. Autograd: the `x**2` example, compare `x.grad` to the manual derivative from Lesson 1
4. `nn.Module`: rebuild the XOR network from Lesson 2, but with `nn.Linear` + autograd
5. Walkthrough of the training loop in `example.py`, side-by-side with the manual version
6. Recap: "every later lesson's code will look like this" + pointer to Phase 01

## Further Reading

- Official PyTorch tutorial: *Deep Learning with PyTorch: A 60 Minute Blitz*
- PyTorch docs: `torch.autograd`, `torch.nn`, `torch.optim`
- Karpathy, *A Recipe for Training Neural Networks* (practical debugging habits that apply from here through every future lesson)
