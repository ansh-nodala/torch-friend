/**
 * @file rope.cpp
 * @brief Ultra-optimized CPU implementation of Rotary Position Embedding (RoPE) for LibTorch.
 * @author Ansh Raj
 * @version 1.0.1
 * @date 2026
 */

#include "torch_tools/rope.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torch_tools {
namespace kernels {

template <typename scalar_t>
void rope_forward_cpu_kernel(
    const scalar_t* __restrict__ x_ptr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    scalar_t* __restrict__ out_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_dim
) {
    const int64_t half_dim = head_dim / 2;

    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const scalar_t* c_ptr = cos_ptr + s * half_dim;
            const scalar_t* s_ptr = sin_ptr + s * half_dim;

            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t base_idx = (seq_dim == 1)
                    ? (b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim)
                    : (b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim);

                const scalar_t* x_row = x_ptr + base_idx;
                scalar_t* out_row = out_ptr + base_idx;

                const scalar_t* x1 = x_row;
                const scalar_t* x2 = x_row + half_dim;

                scalar_t* out1 = out_row;
                scalar_t* out2 = out_row + half_dim;

                for (int64_t i = 0; i < half_dim; ++i) {
                    float val_x1 = static_cast<float>(x1[i]);
                    float val_x2 = static_cast<float>(x2[i]);
                    float cos_val = static_cast<float>(c_ptr[i]);
                    float sin_val = static_cast<float>(s_ptr[i]);

                    out1[i] = static_cast<scalar_t>(val_x1 * cos_val - val_x2 * sin_val);
                    out2[i] = static_cast<scalar_t>(val_x2 * cos_val + val_x1 * sin_val);
                }
            }
        }
    }
}

torch::Tensor rope_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim
) {
    TORCH_CHECK(x.device().is_cpu(), "rope_forward_cpu expects input tensor on CPU");
    TORCH_CHECK(cos.device().is_cpu(), "rope_forward_cpu expects cos tensor on CPU");
    TORCH_CHECK(sin.device().is_cpu(), "rope_forward_cpu expects sin tensor on CPU");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional [B, S, H, D] or [B, H, S, D]");

    auto x_contig = x.contiguous();
    auto cos_contig = cos.contiguous();
    auto sin_contig = sin.contiguous();

    auto out = torch::empty_like(x_contig);

    int64_t batch_size = x_contig.size(0);
    int64_t seq_len = x_contig.size(seq_dim);
    int64_t num_heads = (seq_dim == 1) ? x_contig.size(2) : x_contig.size(1);
    int64_t head_dim = x_contig.size(3);

    TORCH_CHECK(head_dim % 2 == 0, "Head dimension must be an even number for RoPE");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        x_contig.scalar_type(), "rope_forward_cpu", [&] {
            rope_forward_cpu_kernel<scalar_t>(
                x_contig.data_ptr<scalar_t>(),
                cos_contig.data_ptr<scalar_t>(),
                sin_contig.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                seq_dim
            );
        }
    );

    return out;
}

template <typename scalar_t>
void rope_backward_cpu_kernel(
    const scalar_t* __restrict__ grad_out_ptr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    scalar_t* __restrict__ grad_x_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_dim
) {
    const int64_t half_dim = head_dim / 2;

    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const scalar_t* c_ptr = cos_ptr + s * half_dim;
            const scalar_t* s_ptr = sin_ptr + s * half_dim;

            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t base_idx = (seq_dim == 1)
                    ? (b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim)
                    : (b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim);

                const scalar_t* gout_row = grad_out_ptr + base_idx;
                scalar_t* ginc_row = grad_x_ptr + base_idx;

                const scalar_t* gout1 = gout_row;
                const scalar_t* gout2 = gout_row + half_dim;

                scalar_t* ginc1 = ginc_row;
                scalar_t* ginc2 = ginc_row + half_dim;

                for (int64_t i = 0; i < half_dim; ++i) {
                    float val_g1 = static_cast<float>(gout1[i]);
                    float val_g2 = static_cast<float>(gout2[i]);
                    float cos_val = static_cast<float>(c_ptr[i]);
                    float sin_val = static_cast<float>(s_ptr[i]);

                    ginc1[i] = static_cast<scalar_t>(val_g1 * cos_val + val_g2 * sin_val);
                    ginc2[i] = static_cast<scalar_t>(val_g2 * cos_val - val_g1 * sin_val);
                }
            }
        }
    }
}

torch::Tensor rope_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim
) {
    auto grad_out_contig = grad_output.contiguous();
    auto cos_contig = cos.contiguous();
    auto sin_contig = sin.contiguous();

    auto grad_x = torch::empty_like(grad_out_contig);

    int64_t batch_size = grad_out_contig.size(0);
    int64_t seq_len = grad_out_contig.size(seq_dim);
    int64_t num_heads = (seq_dim == 1) ? grad_out_contig.size(2) : grad_out_contig.size(1);
    int64_t head_dim = grad_out_contig.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        grad_out_contig.scalar_type(), "rope_backward_cpu", [&] {
            rope_backward_cpu_kernel<scalar_t>(
                grad_out_contig.data_ptr<scalar_t>(),
                cos_contig.data_ptr<scalar_t>(),
                sin_contig.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                seq_dim
            );
        }
    );

    return grad_x;
}

} // namespace kernels

// ============================================================================
// Autograd Function Implementation
// ============================================================================

torch::Tensor RoPEFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim
) {
    ctx->save_for_backward({cos, sin});
    ctx->saved_data["seq_dim"] = seq_dim;

    if (x.device().is_cuda()) {
#ifdef USE_CUDA
        return kernels::rope_forward_cuda(x, cos, sin, seq_dim);
#else
        TORCH_CHECK(false, "RoPE compiled without CUDA support, but input tensor is on CUDA device.");
#endif
    } else {
        return kernels::rope_forward_cpu(x, cos, sin, seq_dim);
    }
}

torch::autograd::variable_list RoPEFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto cos = saved[0];
    auto sin = saved[1];
    int64_t seq_dim = ctx->saved_data["seq_dim"].toInt();

    auto grad_output = grad_outputs[0];
    torch::Tensor grad_x;

    if (grad_output.device().is_cuda()) {
#ifdef USE_CUDA
        grad_x = kernels::rope_backward_cuda(grad_output, cos, sin, seq_dim);
#else
        TORCH_CHECK(false, "RoPE compiled without CUDA support, but gradient tensor is on CUDA device.");
#endif
    } else {
        grad_x = kernels::rope_backward_cpu(grad_output, cos, sin, seq_dim);
    }

    return {grad_x, torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

// ============================================================================
// RotaryEmbeddingImpl Module Implementation
// ============================================================================

RotaryEmbeddingImpl::RotaryEmbeddingImpl(
    int64_t head_dim,
    int64_t max_seq_len,
    double base,
    int64_t seq_dim
) : options(head_dim) {
    options.max_seq_len(max_seq_len);
    options.base(base);
    options.seq_dim(seq_dim);

    update_cos_sin_cache(max_seq_len, torch::kCPU, torch::kFloat32);
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(const RotaryEmbeddingOptions& options)
    : options(options) {
    update_cos_sin_cache(options.max_seq_len(), options.device(), options.dtype());
}

void RotaryEmbeddingImpl::update_cos_sin_cache(
    int64_t seq_len,
    torch::Device device,
    torch::Dtype dtype
) {
    if (seq_len <= current_cached_seq_len_ && 
        cos_cached_.device() == device && 
        cos_cached_.dtype() == dtype) {
        return;
    }

    current_cached_seq_len_ = std::max(seq_len, options.max_seq_len());

    auto inv_freq = torch::pow(
        options.base(),
        -torch::arange(0, options.head_dim(), 2, torch::dtype(torch::kFloat32).device(device)) / static_cast<double>(options.head_dim())
    );

    auto t = torch::arange(0, current_cached_seq_len_, torch::dtype(torch::kFloat32).device(device));
    auto freqs = torch::outer(t, inv_freq);

    cos_cached_ = torch::cos(freqs).to(dtype);
    sin_cached_ = torch::sin(freqs).to(dtype);

    // Fixed: register_buffer takes exactly 2 arguments in LibTorch C++ API
    register_buffer("cos_cached", cos_cached_);
    register_buffer("sin_cached", sin_cached_);
}

torch::Tensor RotaryEmbeddingImpl::forward(const torch::Tensor& x, int64_t seq_offset) {
    int64_t seq_len = x.size(options.seq_dim()) + seq_offset;

    if (seq_len > current_cached_seq_len_) {
        update_cos_sin_cache(seq_len * 2, x.device(), x.scalar_type());
    }

    auto cos = cos_cached_.slice(/*dim=*/0, /*start=*/seq_offset, /*end=*/seq_len);
    auto sin = sin_cached_.slice(/*dim=*/0, /*start=*/seq_offset, /*end=*/seq_len);

    if (cos.device() != x.device() || cos.dtype() != x.scalar_type()) {
        cos = cos.to(x.device(), x.scalar_type());
        sin = sin.to(x.device(), x.scalar_type());
    }

    return apply_rope(x, cos, sin, options.seq_dim());
}

std::tuple<torch::Tensor, torch::Tensor> RotaryEmbeddingImpl::forward_qk(
    const torch::Tensor& q,
    const torch::Tensor& k,
    int64_t seq_offset
) {
    return std::make_tuple(forward(q, seq_offset), forward(k, seq_offset));
}

void RotaryEmbeddingImpl::pretty_print(std::ostream& stream) const {
    stream << "torch_tools::RotaryEmbedding("
           << "head_dim=" << options.head_dim()
           << ", max_seq_len=" << options.max_seq_len()
           << ", base=" << options.base()
           << ", seq_dim=" << options.seq_dim() << ")";
}

} // namespace torch_tools
