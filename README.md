# MiyarAI

An educational deep learning framework written entirely from scratch in Python.

MiyarAI is built to demonstrate how modern deep learning frameworks work internally without relying on PyTorch or TensorFlow internals. Every component is implemented from first principles using Python.

> **Current Version:** v0.1.0 

---

## Features

Current features include:

- Custom Tensor class
- Shape inference
- Shape validation
- Data type inference
- Tensor reshaping
- Matrix transpose
- Tensor cloning
- Pythonic indexing and iteration

More features will be added in future releases.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Qazi-01/Miyar-AI.git
cd Miyar-AI
```

Install in editable mode:

```bash
pip install -e .
```

---

## Requirements

- Python 3.12 or newer

---

## Quick Example

```python
from miyarai import Tensor

x = Tensor([
    [1, 2],
    [3, 4]
])

print(x)

print(x.shape)

print(x.transpose())
```

---

## Project Structure

```
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

## Roadmap

### v0.1 — Tensor Core

- [x] Tensor class
- [x] Shape inference
- [x] Shape validation
- [x] Tensor cloning
- [x] Reshape
- [x] Transpose

### v0.2 — Math Engine

- [ ] Tensor addition
- [ ] Tensor subtraction
- [ ] Tensor multiplication
- [ ] Tensor division
- [ ] Matrix multiplication
- [ ] Sum
- [ ] Mean

### v0.3 — Computational Graph

- [ ] Graph nodes
- [ ] Operation tracking
- [ ] Gradient storage

### v0.4 — Automatic Differentiation

- [ ] Backpropagation
- [ ] Chain rule
- [ ] Gradient accumulation

### v0.5 — Neural Networks

- [ ] Module
- [ ] Parameter
- [ ] Linear
- [ ] Sequential

---

## Running Tests

```bash
python -m unittest discover tests
```

---

## Dependencies

MiyarAI v0.1.0 uses only the Python standard library.
No third-party dependencies are required.

---

## Contributing

Contributions, issues and feature requests are welcome.

---

## License

This project is licensed under the GNU General Public License v3.0.