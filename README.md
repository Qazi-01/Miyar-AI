# MiyarAI

An educational deep learning framework written entirely from scratch in Python.

MiyarAI is built to demonstrate how modern deep learning frameworks work internally without relying on PyTorch or TensorFlow internals. Every component is implemented from first principles using pure Python, making the code easy to read, understand, and extend.

> **Current Version:** v0.2.0

---

# Features

## Tensor Core

- Custom `Tensor` class
- Shape inference
- Shape validation
- Data type inference
- Tensor cloning
- Tensor reshaping
- Matrix transpose
- Pythonic indexing
- Pythonic iteration

## Math Engine

- Tensor addition
- Tensor subtraction
- Tensor multiplication
- Tensor division
- Scalar arithmetic
- Broadcasting
- Matrix multiplication
- Sum reduction
- Mean reduction
- Maximum reduction
- Minimum reduction

## Tensor Manipulation

- `flatten()`
- `squeeze()`
- `unsqueeze()`
- `repeat()`
- `tile()`
- `stack()`
- `concat()`
- `split()`

## Indexing

- Integer indexing
- Slice indexing
- Tuple indexing
- Basic advanced indexing

---

# Installation

Clone the repository

```bash
git clone https://github.com/Qazi-01/Miyar-AI.git
cd Miyar-AI
```

Install in editable mode

```bash
pip install -e .
```

---

# Requirements

- Python 3.12+

---

# Quick Example

```python
from miyarai import Tensor

a = Tensor([
    [1, 2],
    [3, 4]
])

b = Tensor([
    [5, 6],
    [7, 8]
])

print(a + b)

print(a @ b)

print(a.sum())

print(a.mean())

print(a.flatten())

print(a[0])

print(a[:, 1])
```

Output

```text
Tensor([[6, 8], [10, 12]])
Tensor([[19, 22], [43, 50]])
10
2.5
Tensor([1, 2, 3, 4])
Tensor([1, 2])
Tensor([2, 4])
```

---

# Project Structure

```text
Miyar-AI/
│
├── examples/
├── tests/
├── miyarai/
│   ├── __init__.py
│   └── tensor/
│       ├── __init__.py
│       └── tensor.py
│
├── README.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

# Roadmap

## ✅ v0.1 - Tensor Core

- [x] Tensor class
- [x] Shape inference
- [x] Shape validation
- [x] Data type inference
- [x] Clone
- [x] Reshape
- [x] Transpose

---

## ✅ v0.2 - Math Engine

- [x] Tensor addition
- [x] Tensor subtraction
- [x] Tensor multiplication
- [x] Tensor division
- [x] Scalar operations
- [x] Broadcasting
- [x] Matrix multiplication
- [x] Sum
- [x] Mean
- [x] Max
- [x] Min
- [x] Flatten
- [x] Squeeze
- [x] Unsqueeze
- [x] Repeat
- [x] Tile
- [x] Stack
- [x] Concat
- [x] Split
- [x] Integer indexing
- [x] Slice indexing
- [x] Tuple indexing
- [x] Basic advanced indexing

---

##  v0.3 - Automatic Differentiation

- [ ] Computational graph
- [ ] Gradient storage
- [ ] Backpropagation
- [ ] Chain rule
- [ ] Gradient accumulation

---

##  v0.4 - Neural Network API

- [ ] Module
- [ ] Parameter
- [ ] Linear layer
- [ ] Sequential
- [ ] Activation functions

---

##  v0.5 - Optimizers

- [ ] SGD
- [ ] Momentum
- [ ] Adam
- [ ] Learning rate schedulers

---

# Running Tests

Run the complete test suite

```bash
python -m unittest discover -s tests
```

Current status

```text
203 tests passing
```

---

# Dependencies

MiyarAI currently depends only on the Python Standard Library.

No external numerical libraries such as NumPy, PyTorch, or TensorFlow are used internally.

---

# Philosophy

MiyarAI prioritizes clarity over performance.

Every algorithm is implemented manually so readers can understand how modern deep learning frameworks operate internally before relying on optimized libraries.

---

# Contributing

Contributions, bug reports, and feature requests are welcome.

---

# License

Licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

