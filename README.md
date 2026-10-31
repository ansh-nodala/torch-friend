# ⚡ torch-tools

**High-Performance C++20, OpenMP, and CUDA Extensions for PyTorch**

[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/torch-tools/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`torch-tools` is a native C++/LibTorch operator library built to accelerate modern Deep Learning and Transformer workloads in PyTorch. Instead of relying on memory-bound Python operations, `torch-tools` executes fused kernels directly in C++20 (using OpenMP multithreading and SIMD vectorization on CPU) and custom CUDA kernels on NVIDIA GPUs.

> **📢 Project Status:** **Rotary Position Embedding (RoPE)** is 100% complete, tested, and production-ready! `torch-tools` is an actively growing ecosystem—many more custom C++/CUDA neural network primitives, fused normalization layers, and PyTorch utilities are on the way.

---

## ✨ Key Highlights

- **Zero-Overhead C++20 & LibTorch Backend:** Directly binds to PyTorch tensors via `pybind11` without unnecessary memory copies.
- **Full Autograd Support:** Custom C++ forward and backward passes (`torch::autograd::Function`) for seamless end-to-end training.
- **Multi-Threaded CPU Execution:** Parallelized with OpenMP and compiled with `-O3 -march=native -ffast-math` for maximum SIMD throughput on CPU.
- **Custom CUDA Kernels:** Optional hardware-accelerated GPU kernels for high-throughput training and inference.
- **Multi-Precision Dispatch:** Native support for `float32`, `float16`, and `bfloat16` tensors.
- **KV-Cache & GQA Ready:** Built-in `seq_offset` support for single-token autoregressive decoding and `forward_qk` for Grouped-Query Attention (GQA).

---

## 📦 Installation

### 1. Install via PyPI (Recommended)

You can install `torch-tools` directly using `pip`:

```bash
pip install torch-tools
```

### 2. Install from Source

Ensure you have `torch>=2.0`, a C++20 compatible compiler (`gcc>=11` or `clang`), and `cmake>=3.18` installed on your system.

**CPU-Only Build:**
```bash
git clone https://github.com/ANSH-ind/torch-friend.git
cd torch-friend
pip install . --no-build-isolation
```

**CUDA-Enabled Build (for NVIDIA GPUs):**
```bash
git clone https://github.com/ANSH-ind/torch-friend.git
cd torch-friend
USE_CUDA=ON pip install . --no-build-isolation
```

**Editable Development Install:**
```bash
pip install -e . --no-build-isolation
```

---

## 🧮 Mathematical Formulation (RoPE)

For a given head dimension $d$ and token position $m$, Rotary Position Embedding pairs adjacent feature coordinates $(x_{2i}, x_{2i+1})$ for $i \in \{0, 1, \dots, d/2 - 1\}$ and applies a 2D rotation in the complex plane with frequency $\theta_i = \text{base}^{-2i/d}$:

$$
\begin{pmatrix}
\tilde{x}_{m, 2i} \\
\tilde{x}_{m, 2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{m, 2i} \\
x_{m, 2i+1}
\end{pmatrix}
$$

During the backward pass, the exact analytical gradient with respect to the input is computed by applying the transpose (inverse) rotation by $-m\theta_i$:

$$
\begin{pmatrix}
\frac{\partial \mathcal{L}}{\partial x_{m, 2i}} \\
\frac{\partial \mathcal{L}}{\partial x_{m, 2i+1}}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & \sin(m\theta_i) \\
-\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
\frac{\partial \mathcal{L}}{\partial \tilde{x}_{m, 2i}} \\
\frac{\partial \mathcal{L}}{\partial \tilde{x}_{m, 2i+1}}
\end{pmatrix}
$$

---

## 🚀 Usage Guide

### 1. Standard `nn.Module` Interface (`RotaryEmbedding`)

`RotaryEmbedding` is a drop-in `torch.nn.Module` that automatically computes, caches, and dynamically extends cosine and sine tables across devices and data types.

```python
import torch
from torch_tools import RotaryEmbedding

# Initialize RoPE (head_dim must be an even integer)
rope = RotaryEmbedding(
    head_dim=64,
    max_seq_len=4096,
    base=10000.0,
    seq_dim=1,  # Use 1 for [B, S, H, D] layout, or 2 for [B, H, S, D] layout
)

# Input tensor: [Batch=2, SeqLen=128, Heads=8, HeadDim=64]
x = torch.randn(2, 128, 8, 64, requires_grad=True)

# Forward pass (C++ / CUDA kernel)
out = rope(x)

# Backward pass (C++ custom autograd engine)
loss = out.sum()
loss.backward()

print("Module:", rope)
print("Output shape:", out.shape)
print("Input gradient norm:", x.grad.norm().item())
```

### 2. Simultaneous Query & Key Rotation (`forward_qk`)

In Transformer attention layers (including Multi-Head Attention and Grouped-Query Attention where Query and Key head counts differ), use `forward_qk` to rotate both tensors cleanly:

```python
import torch
from torch_tools import RotaryEmbedding

rope = RotaryEmbedding(head_dim=64, max_seq_len=4096, seq_dim=1)

# Grouped-Query Attention (GQA): 8 Query heads, 2 Key/Value heads
q = torch.randn(2, 64, 8, 64, requires_grad=True)
k = torch.randn(2, 64, 2, 64, requires_grad=True)

q_rot, k_rot = rope.forward_qk(q, k)

print("Rotated Q shape:", q_rot.shape)
print("Rotated K shape:", k_rot.shape)
```

### 3. Autoregressive Decoding with `seq_offset` (KV-Cache)

When generating tokens one-by-one during inference with a KV-cache, pass `seq_offset` so the newly generated token is rotated using its exact absolute position in the sequence:

```python
import torch
from torch_tools import RotaryEmbedding

rope = RotaryEmbedding(head_dim=64, max_seq_len=4096, seq_dim=1)

# Prefill phase: 128 prompt tokens (positions 0..127)
prompt_q = torch.randn(2, 128, 8, 64)
prompt_q_rot = rope(prompt_q, seq_offset=0)

# Decode step: 1 new token at position 128
next_token_q = torch.randn(2, 1, 8, 64)
next_token_q_rot = rope(next_token_q, seq_offset=128)
```

### 4. Stateless Functional API (`apply_rope`)

If you manage your own custom precomputed `cos` and `sin` frequency caches (e.g., for YaRN, NTK-aware scaling, or custom position IDs) of shape $[S, d/2]$, use `apply_rope`:

```python
import torch
from torch_tools import apply_rope

batch, seq_len, heads, head_dim = 2, 32, 4, 64
x = torch.randn(batch, seq_len, heads, head_dim, requires_grad=True)

# Precomputed cos and sin tables of shape [seq_len, head_dim // 2]
cos = torch.ones(seq_len, head_dim // 2)
sin = torch.zeros(seq_len, head_dim // 2)

out = apply_rope(x, cos, sin, seq_dim=1)
out.sum().backward()
```

---

## 🗺️ Roadmap: Upcoming Tools in `torch-tools`

`torch-tools` is designed to be a complete, high-speed C++/CUDA operator suite for PyTorch. While **RoPE** is our first completed module, several new operators are currently under development and will be released in upcoming versions:

- [x] **Rotary Position Embedding (RoPE)** — *Completed & Production-Ready (CPU OpenMP + CUDA, Custom Autograd, GQA & KV-Cache support)*
- [ ] **Fused RMSNorm** — *Ultra-fast Root Mean Square Layer Normalization with fused residual addition in C++/CUDA*
- [ ] **Fused SwiGLU / GeGLU Activations** — *Memory-efficient gated linear unit kernels with custom backward passes*
- [ ] **Grouped-Query Attention (GQA) Utilities** — *Zero-copy KV head broadcasting and fused attention helpers*
- [ ] **Low-Bit & Ternary Quantization Kernels** — *High-speed bit-packed linear operations for efficient LLM inference*
- [ ] **Fused Cross-Entropy Loss** — *Memory-saving logits-to-loss C++/CUDA computation*

---

## 🛠️ Building Distributable Wheels Locally

To build a standalone `.whl` package inside the `dist/` folder for deployment or uploading to PyPI via `twine`:

```bash
pip wheel . --no-deps --no-build-isolation -w dist/
```

---

## 👤 Author & License

- **Author:** Ansh Raj ([@ANSH-ind](https://github.com/ANSH-ind))
- **Contact:** anshraj0000000001@gmail.com
- **Copyright:** © 2026 Techgears - All Rights Reserved.
- **License:** Released under the [MIT License](LICENSE).
