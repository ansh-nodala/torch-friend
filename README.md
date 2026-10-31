# torch-friends

**High-Performance C++20, OpenMP, and CUDA Extensions for PyTorch**

[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/torch-friends/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`torch-friends` is a native C++/LibTorch operator library built by Ansh Raj to accelerate deep learning and Transformer workloads in PyTorch. Instead of relying on memory-bound Python loops and dynamic dispatch overheads, `torch-friends` executes fused kernels directly in C++20 (leveraging OpenMP multithreading and SIMD vectorization on CPUs) and custom CUDA kernels on NVIDIA GPUs.

> **Project Status:** Rotary Position Embedding (RoPE) is 100% complete, fully tested, and production-ready. `torch-friends` is an actively growing systems engineering project — bringing more custom C++/CUDA neural network primitives and low-level optimizations to PyTorch.

---

## About the Author

Built from first principles by **Ansh Raj**, a systems developer and machine learning practitioner from Bihar, India. `torch-friends` was created to bridge the gap between high-level PyTorch modeling and bare-metal hardware execution speed.

- **GitHub:** [@ANSH-ind](https://github.com/ANSH-ind)
- **Portfolio:** [anshstudios.pages.dev/portfolio](https://anshstudios.pages.dev/portfolio)
- **Contact:** anshraj0000000001@gmail.com

---

## Key Highlights

- **Zero-Overhead C++20 & LibTorch Backend** — Directly binds to PyTorch tensors via `pybind11` without unnecessary memory copies or slow Python overheads.
- **Full Autograd Integration** — Custom C++ forward and backward passes (`torch::autograd::Function`) with analytical gradient calculation for seamless end-to-end model training.
- **Multi-Threaded CPU Execution** — Parallelized with OpenMP and compiled with aggressive optimization flags (`-O3 -march=native -ffast-math`) for maximum SIMD throughput on CPU.
- **Custom CUDA Kernels** — Optional hardware-accelerated GPU execution paths for high-throughput training and inference.
- **Multi-Precision Support** — Native data-type agnosticism supporting `float32`, `float16`, and `bfloat16`.
- **KV-Cache & GQA Ready** — Built-in `seq_offset` support for single-token autoregressive decoding and `forward_qk` for Grouped-Query Attention (GQA).

---

## Installation

### 1. Install via PyPI (Recommended)

```bash
pip install torch-friends
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

> **Note:** The PyPI package is published as `torch-friends`, but the GitHub repository is named `torch-friend`. The Python import name in both cases is `torch_tools`.

---

## Mathematical Formulation (RoPE)

For a given head dimension $d$ and token position $m$, Rotary Position Embedding pairs adjacent feature coordinates $(x_{2i}, x_{2i+1})$ for $i \in \{0, 1, \dots, d/2 - 1\}$ and applies a 2D rotation in the complex plane with frequency:

$$
\theta_i = \text{base}^{-2i/d}
$$

The forward rotation is:

$$
\begin{pmatrix}
\tilde{x}_{m,\,2i} \\
\tilde{x}_{m,\,2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{m,\,2i} \\
x_{m,\,2i+1}
\end{pmatrix}
$$

During the backward pass, the exact analytical gradient with respect to the input is computed by applying the inverse rotation by $-m\theta_i$:

$$
\begin{pmatrix}
\dfrac{\partial \mathcal{L}}{\partial x_{m,\,2i}} \\[6pt]
\dfrac{\partial \mathcal{L}}{\partial x_{m,\,2i+1}}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & \sin(m\theta_i) \\
-\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
\dfrac{\partial \mathcal{L}}{\partial \tilde{x}_{m,\,2i}} \\[6pt]
\dfrac{\partial \mathcal{L}}{\partial \tilde{x}_{m,\,2i+1}}
\end{pmatrix}
$$

Since the rotation matrix is orthogonal, its inverse equals its transpose — so the backward pass reuses the same $\cos$ / $\sin$ tables as the forward pass, just with the sign of the $\sin$ term flipped.

---

## Usage Guide

*Note: Even though the repository is named `torch-friend`, the Python package imports cleanly as `torch_tools`.*

### 1. Standard `nn.Module` Interface (`RotaryEmbedding`)

`RotaryEmbedding` is a drop-in `torch.nn.Module` that automatically computes, caches, and dynamically extends cosine and sine frequency tables across devices and data types.

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

In Transformer attention layers (including Multi-Head Attention and Grouped-Query Attention where Query and Key head counts differ), use `forward_qk` to rotate both tensors cleanly in one pass:

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

If you manage your own custom precomputed `cos` and `sin` frequency tables (e.g., for YaRN, NTK-aware scaling, or custom position IDs) of shape $[S, d/2]$, use `apply_rope`:

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

## Roadmap: Upcoming Tools in `torch-friends`

`torch-friends` is designed to expand into a complete, high-speed C++/CUDA operator suite for PyTorch. While **Rotary Position Embedding (RoPE)** is the first completed core module, several new primitives are currently under development and will be released in upcoming versions:

- [x] **Rotary Position Embedding (RoPE)** — *Completed & Production-Ready (CPU OpenMP + CUDA, Custom Autograd, GQA & KV-Cache support)*
- [ ] **Fused RMSNorm** — *Ultra-fast Root Mean Square Layer Normalization with fused residual addition in C++/CUDA*
- [ ] **Fused SwiGLU / GeGLU Activations** — *Memory-efficient gated linear unit kernels with custom backward passes*
- [ ] **Grouped-Query Attention (GQA) Utilities** — *Zero-copy KV head broadcasting and fused attention helpers*
- [ ] **Low-Bit & Ternary Quantization Kernels** — *High-speed bit-packed linear operations for efficient LLM inference*
- [ ] **Fused Cross-Entropy Loss** — *Memory-saving logits-to-loss C++/CUDA computation*

---

## Building Distributable Wheels Locally

To build a standalone `.whl` package inside the `dist/` folder for deployment or uploading to PyPI via `twine`:

```bash
pip wheel . --no-deps --no-build-isolation -w dist/
```

---

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Techgears - All Rights Reserved.
