# MiyarAI Roadmap

This document outlines the long-term development plan for MiyarAI.

---

# v0.1 — Tensor Core ✅

Status: Released

- Tensor class
- Shape inference
- Shape validation
- Tensor cloning
- Reshape
- Transpose
- Iteration
- Equality

---

# v0.2 — Math Engine ✅

Status: Released

## Arithmetic
- Tensor addition
- Tensor subtraction
- Tensor multiplication
- Tensor division

## Statistics
- sum()
- mean()
- max()
- min()

## Linear Algebra
- Matrix multiplication

## Broadcasting
- NumPy-style broadcasting

## Tensor Manipulation
- flatten()
- squeeze()
- unsqueeze()
- stack()
- concat()
- split()
- repeat()
- tile()

## Indexing
- __getitem__()
- __setitem__()
- Slice indexing
- Tuple indexing
- Advanced indexing

---

# v0.3 — Computational Graph 🚧

Status: Planned

## Phase 1
- requires_grad
- grad
- parents
- operation metadata
- backward placeholder

## Phase 2
- Record tensor operations

## Phase 3
- Leaf tensors

## Phase 4
- Graph traversal

## Phase 5
- Graph inspection

---

# v0.4 — Automatic Differentiation

- Backward propagation
- Chain rule
- Gradient accumulation
- Gradient clearing

---

# v0.5 — Neural Network API

- Module
- Parameter
- Sequential
- Linear
- Activation functions

---

# v0.6 — Optimizers

- SGD
- Momentum
- Adam
- Weight decay

---

# v0.7 — Loss Functions

- MSE
- Cross Entropy
- Binary Cross Entropy
- L1 Loss

---

# v0.8 — Data Pipeline

- Dataset
- TensorDataset
- DataLoader
- Batch iteration
- Shuffle

---

# v0.9 — Training Engine

- Training loop
- Evaluation loop
- Metrics
- Model saving
- Model loading

---

# v1.0 — Stable Release

- Public API freeze
- Documentation
- Tutorials
- Performance improvements
- Complete test suite

