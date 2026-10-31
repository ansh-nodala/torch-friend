/**
 * @file rope.hpp
 * @brief Ultra-optimized Rotary Position Embedding (RoPE) for LibTorch (CPU & CUDA).
 * @author Ansh Raj
 * @version 1.0.0
 * @date 2026
 * 
 * @details
 * This header defines the RotaryEmbedding module and low-level functional kernels
 * for applying Rotary Position Embeddings (Su et al., 2021). 
 * 
 * Architecture Highlights:
 * - Zero-overhead compile-time fallback for CPU-only machines (no CUDA headers required).
 * - Pre-computed Cosine and Sine frequency caching to avoid redundant trigonometric ops.
 * - Contiguous memory layout enforcement for AVX2/AVX-512 (CPU) and coalesced memory access (CUDA).
 * - Full Autograd support via custom `torch::autograd::Function` for backward pass fusion.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <tuple>
#include <string>

namespace torch_tools {
namespace kernels {

/**
 * @brief CPU-optimized forward kernel for Rotary Position Embedding.
 * 
 * @details Uses OpenMP multi-threading and SIMD-friendly contiguous pointer arithmetic
 *          to apply rotary embeddings in-place or out-of-place on CPU tensors.
 * 
 * @param x Input tensor of shape `[batch_size, seq_len, num_heads, head_dim]` 
 *          or `[batch_size, num_heads, seq_len, head_dim]`.
 * @param cos Precomputed cosine cache tensor of shape `[max_seq_len, head_dim / 2]`.
 * @param sin Precomputed sine cache tensor of shape `[max_seq_len, head_dim / 2]`.
 * @param seq_dim The dimension index representing sequence length (default: 1).
 * @return torch::Tensor Rotated output tensor of the same shape and dtype as `x`.
 */
torch::Tensor rope_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim = 1
);

/**
 * @brief CPU-optimized backward kernel for Rotary Position Embedding.
 * 
 * @details Computes the gradient with respect to input `x` by applying the transpose
 *          of the 2D rotation matrix (inverting the sign of the sine component).
 * 
 * @param grad_output Upstream gradient tensor matching the shape of forward output.
 * @param cos Precomputed cosine cache tensor of shape `[max_seq_len, head_dim / 2]`.
 * @param sin Precomputed sine cache tensor of shape `[max_seq_len, head_dim / 2]`.
 * @param seq_dim The dimension index representing sequence length (default: 1).
 * @return torch::Tensor Gradient with respect to input `x` (`grad_x`).
 */
torch::Tensor rope_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim = 1
);

#ifdef USE_CUDA
/**
 * @brief CUDA-accelerated forward kernel launcher for Rotary Position Embedding.
 * 
 * @note Only compiled when `USE_CUDA` macro is defined via CMake/compiler flags.
 * 
 * @param x Input tensor residing on a CUDA device.
 * @param cos Precomputed cosine cache tensor on the same CUDA device.
 * @param sin Precomputed sine cache tensor on the same CUDA device.
 * @param seq_dim The dimension index representing sequence length (default: 1).
 * @return torch::Tensor Rotated output tensor on CUDA device.
 */
torch::Tensor rope_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim = 1
);

/**
 * @brief CUDA-accelerated backward kernel launcher for Rotary Position Embedding.
 * 
 * @note Only compiled when `USE_CUDA` macro is defined via CMake/compiler flags.
 * 
 * @param grad_output Upstream gradient tensor residing on a CUDA device.
 * @param cos Precomputed cosine cache tensor on the same CUDA device.
 * @param sin Precomputed sine cache tensor on the same CUDA device.
 * @param seq_dim The dimension index representing sequence length (default: 1).
 * @return torch::Tensor Gradient with respect to input `x` on CUDA device.
 */
torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim = 1
);
#endif // USE_CUDA

} // namespace kernels

/**
 * @class RoPEFunction
 * @brief Custom LibTorch Autograd Function linking forward and backward passes.
 * 
 * @details Dispatches execution automatically to `rope_forward_cpu` or `rope_forward_cuda`
 *          based on the input tensor's runtime device without breaking the computation graph.
 */
class RoPEFunction : public torch::autograd::Function<RoPEFunction> {
public:
    /**
     * @brief Forward pass for custom autograd node.
     * 
     * @param ctx Autograd context to save non-tensor data and cached tensors for backward pass.
     * @param x Input query or key tensor (4D).
     * @param cos Precomputed cosine frequency tensor.
     * @param sin Precomputed sine frequency tensor.
     * @param seq_dim Dimension index corresponding to sequence length `S`.
     * @return torch::Tensor Rotated tensor with positional information encoded.
     */
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x,
        const torch::Tensor& cos,
        const torch::Tensor& sin,
        int64_t seq_dim = 1
    );

    /**
     * @brief Backward pass for custom autograd node.
     * 
     * @param ctx Autograd context containing saved variables from the forward pass.
     * @param grad_outputs Vector of upstream gradients (contains `dL/dy`).
     * @return torch::autograd::variable_list List of gradients `{dL/dx, Undefined, Undefined, Undefined}`.
     */
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    );
};

/**
 * @struct RotaryEmbeddingOptions
 * @brief Configuration options for initializing the `RotaryEmbedding` module.
 */
struct TORCH_API RotaryEmbeddingOptions {
    /**
     * @brief Construct options with required head dimension.
     * @param head_dim Feature dimension of each attention head (must be an even number).
     */
    explicit RotaryEmbeddingOptions(int64_t head_dim) : head_dim_(head_dim) {}

    /// @brief Dimension of each attention head (d_k). Must be divisible by 2.
    TORCH_ARG(int64_t, head_dim);

    /// @brief Maximum sequence length supported by the precomputed cache (default: 4096).
    TORCH_ARG(int64_t, max_seq_len) = 4096;

    /// @brief Base geometric period (theta) for rotary frequencies (default: 10000.0).
    TORCH_ARG(double, base) = 10000.0;

    /// @brief Default sequence dimension index in the 4D input tensor (default: 1 for [B, S, H, D]).
    TORCH_ARG(int64_t, seq_dim) = 1;

    /// @brief Data type for frequency cache storage (default: torch::kFloat32).
    TORCH_ARG(torch::Dtype, dtype) = torch::kFloat32;

    /// @brief Target compute device for the frequency cache (default: torch::kCPU).
    TORCH_ARG(torch::Device, device) = torch::kCPU;
};

/**
 * @class RotaryEmbeddingImpl
 * @brief High-performance Neural Network Module for Rotary Position Embedding (RoPE).
 * 
 * @details Manages frequency cache buffers (`cos_cached_` and `sin_cached_`) registered as
 *          non-persistent buffers. Automatically rebuilds cache if an input sequence exceeds
 *          `max_seq_len` at runtime.
 */
class TORCH_API RotaryEmbeddingImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a RotaryEmbeddingImpl using explicit parameters.
     * 
     * @param head_dim Dimension of each attention head (must be even).
     * @param max_seq_len Initial maximum sequence length to cache (default: 4096).
     * @param base Frequency base theta (default: 10000.0).
     * @param seq_dim Sequence length dimension in 4D tensor (1 for `[B, S, H, D]`, 2 for `[B, H, S, D]`).
     */
    explicit RotaryEmbeddingImpl(
        int64_t head_dim,
        int64_t max_seq_len = 4096,
        double base = 10000.0,
        int64_t seq_dim = 1
    );

    /**
     * @brief Construct a RotaryEmbeddingImpl from a `RotaryEmbeddingOptions` struct.
     * @param options Configuration object containing all module hyperparameters.
     */
    explicit RotaryEmbeddingImpl(const RotaryEmbeddingOptions& options);

    /**
     * @brief Apply Rotary Position Embedding to a single 4D tensor (Query or Key).
     * 
     * @param x Input tensor of shape `[B, S, H, D]` (if `seq_dim == 1`) or `[B, H, S, D]` (if `seq_dim == 2`).
     * @param seq_offset Starting positional index offset for KV-cache decoding inference (default: 0).
     * @return torch::Tensor Position-encoded tensor of identical shape, device, and dtype as `x`.
     */
    torch::Tensor forward(const torch::Tensor& x, int64_t seq_offset = 0);

    /**
     * @brief Apply Rotary Position Embedding simultaneously to both Query (Q) and Key (K) tensors.
     * 
     * @param q Query tensor (4D).
     * @param k Key tensor (4D).
     * @param seq_offset Starting positional index offset for autoregressive decoding (default: 0).
     * @return std::tuple<torch::Tensor, torch::Tensor> Pair of rotated `{q_rot, k_rot}` tensors.
     */
    std::tuple<torch::Tensor, torch::Tensor> forward_qk(
        const torch::Tensor& q,
        const torch::Tensor& k,
        int64_t seq_offset = 0
    );

    /**
     * @brief Precompute or dynamically extend the internal cosine and sine frequency buffers.
     * 
     * @param seq_len Target sequence length to cache.
     * @param device Target device (`CPU` or `CUDA`) to place the buffers on.
     * @param dtype Target floating-point data type (`Float32`, `Float16`, `BFloat16`).
     */
    void update_cos_sin_cache(
        int64_t seq_len,
        torch::Device device,
        torch::Dtype dtype
    );

    /**
     * @brief Pretty-print module hyperparameters when inspecting the model architecture.
     * @param stream Output stream to write module details into.
     */
    void pretty_print(std::ostream& stream) const override;

    /// @brief Stored configuration options.
    RotaryEmbeddingOptions options;

private:
    /// @brief Cached cosine frequencies of shape `[max_seq_len, head_dim / 2]`.
    torch::Tensor cos_cached_;

    /// @brief Cached sine frequencies of shape `[max_seq_len, head_dim / 2]`.
    torch::Tensor sin_cached_;

    /// @brief Currently cached maximum sequence length.
    int64_t current_cached_seq_len_{0};
};

/// @brief LibTorch ModuleHolder wrapper generating `torch_tools::RotaryEmbedding`.
TORCH_MODULE(RotaryEmbedding);

/**
 * @brief Functional interface to directly apply RoPE using precomputed `cos` and `sin` caches.
 * 
 * @param x Input 4D tensor (`[B, S, H, D]` or `[B, H, S, D]`).
 * @param cos Precomputed cosine tensor of shape `[S, D/2]`.
 * @param sin Precomputed sine tensor of shape `[S, D/2]`.
 * @param seq_dim Dimension index corresponding to sequence length `S` (1 or 2).
 * @return torch::Tensor Rotated output tensor with full autograd tracking.
 */
inline torch::Tensor apply_rope(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim = 1
) {
    return RoPEFunction::apply(x, cos, sin, seq_dim);
}

} // namespace torch_tools
