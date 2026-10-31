/**
 * @file rope_cuda.cu
 * @brief CUDA-accelerated kernels for Rotary Position Embedding (RoPE).
 * @author Ansh Raj
 * @version 1.0.0
 * @date 2026
 * 
 * @details
 * Implements high-performance fused CUDA kernels for forward and backward passes
 * utilizing grid-stride loops, half-precision support (FP16/BF16), and coalesced memory access.
 */

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

namespace torch_tools {
namespace kernels {

/**
 * @brief CUDA kernel for Rotary Position Embedding forward pass.
 * 
 * @tparam scalar_t Floating point type (float, double, at::Half, at::BFloat16).
 */
template <typename scalar_t>
__global__ void rope_forward_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    scalar_t* __restrict__ out,
    int64_t total_elements,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_dim
) {
    const int64_t half_dim = head_dim / 2;
    const int64_t total_pairs = batch_size * seq_len * num_heads * half_dim;

    // Grid-stride loop for complete GPU core saturation
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_pairs; idx += blockDim.x * gridDim.x) {
        int64_t i = idx % half_dim;
        int64_t rem = idx / half_dim;
        int64_t h = rem % num_heads;
        rem /= num_heads;
        int64_t s = rem % seq_len;
        int64_t b = rem / seq_len;

        // Calculate tensor memory offset based on sequence dimension layout
        int64_t base_idx = (seq_dim == 1)
            ? (b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim)
            : (b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim);

        const scalar_t* x_row = x + base_idx;
        scalar_t* out_row = out + base_idx;

        scalar_t x1 = x_row[i];
        scalar_t x2 = x_row[i + half_dim];

        // Fetch precomputed cos and sin for current sequence position 's'
        const scalar_t* c_ptr = cos_cache + s * half_dim;
        const scalar_t* s_ptr = sin_cache + s * half_dim;

        scalar_t cos_val = c_ptr[i];
        scalar_t sin_val = s_ptr[i];

        // Apply rotary transformation: [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        out_row[i]              = static_cast<scalar_t>(static_cast<float>(x1) * static_cast<float>(cos_val) - static_cast<float>(x2) * static_cast<float>(sin_val));
        out_row[i + half_dim]   = static_cast<scalar_t>(static_cast<float>(x2) * static_cast<float>(cos_val) + static_cast<float>(x1) * static_cast<float>(sin_val));
    }
}

torch::Tensor rope_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim
) {
    TORCH_CHECK(x.device().is_cuda(), "rope_forward_cuda expects input tensor on CUDA");
    
    auto x_contig = x.contiguous();
    auto cos_contig = cos.contiguous();
    auto sin_contig = sin.contiguous();
    auto out = torch::empty_like(x_contig);

    int64_t batch_size = x_contig.size(0);
    int64_t seq_len = x_contig.size(seq_dim);
    int64_t num_heads = (seq_dim == 1) ? x_contig.size(2) : x_contig.size(1);
    int64_t head_dim = x_contig.size(3);
    int64_t half_dim = head_dim / 2;
    int64_t total_pairs = batch_size * seq_len * num_heads * half_dim;

    const int block_size = 256;
    const int num_blocks = static_cast<int>((total_pairs + block_size - 1) / block_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        x_contig.scalar_type(), "rope_forward_cuda", [&] {
            rope_forward_cuda_kernel<scalar_t><<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                cos_contig.data_ptr<scalar_t>(),
                sin_contig.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                total_pairs,
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

/**
 * @brief CUDA kernel for Rotary Position Embedding backward pass (transposed rotation).
 */
template <typename scalar_t>
__global__ void rope_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    scalar_t* __restrict__ grad_x,
    int64_t total_elements,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_dim
) {
    const int64_t half_dim = head_dim / 2;
    const int64_t total_pairs = batch_size * seq_len * num_heads * half_dim;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_pairs; idx += blockDim.x * gridDim.x) {
        int64_t i = idx % half_dim;
        int64_t rem = idx / half_dim;
        int64_t h = rem % num_heads;
        rem /= num_heads;
        int64_t s = rem % seq_len;
        int64_t b = rem / seq_len;

        int64_t base_idx = (seq_dim == 1)
            ? (b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim)
            : (b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim);

        const scalar_t* gout_row = grad_out + base_idx;
        scalar_t* ginc_row = grad_x + base_idx;

        scalar_t g1 = gout_row[i];
        scalar_t g2 = gout_row[i + half_dim];

        const scalar_t* c_ptr = cos_cache + s * half_dim;
        const scalar_t* s_ptr = sin_cache + s * half_dim;

        scalar_t cos_val = c_ptr[i];
        scalar_t sin_val = s_ptr[i];

        // Transposed rotation for gradient backward pass: [g1 * cos + g2 * sin, g2 * cos - g1 * sin]
        ginc_row[i]             = static_cast<scalar_t>(static_cast<float>(g1) * static_cast<float>(cos_val) + static_cast<float>(g2) * static_cast<float>(sin_val));
        ginc_row[i + half_dim]  = static_cast<scalar_t>(static_cast<float>(g2) * static_cast<float>(cos_val) - static_cast<float>(g1) * static_cast<float>(sin_val));
    }
}

torch::Tensor rope_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t seq_dim
) {
    TORCH_CHECK(grad_output.device().is_cuda(), "rope_backward_cuda expects gradient tensor on CUDA");

    auto grad_out_contig = grad_output.contiguous();
    auto cos_contig = cos.contiguous();
    auto sin_contig = sin.contiguous();
    auto grad_x = torch::empty_like(grad_out_contig);

    int64_t batch_size = grad_out_contig.size(0);
    int64_t seq_len = grad_out_contig.size(seq_dim);
    int64_t num_heads = (seq_dim == 1) ? grad_out_contig.size(2) : grad_out_contig.size(1);
    int64_t head_dim = grad_out_contig.size(3);
    int64_t half_dim = head_dim / 2;
    int64_t total_pairs = batch_size * seq_len * num_heads * half_dim;

    const int block_size = 256;
    const int num_blocks = static_cast<int>((total_pairs + block_size - 1) / block_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16, at::ScalarType::Half,
        grad_out_contig.scalar_type(), "rope_backward_cuda", [&] {
            rope_backward_cuda_kernel<scalar_t><<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_out_contig.data_ptr<scalar_t>(),
                cos_contig.data_ptr<scalar_t>(),
                sin_contig.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                total_pairs,
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
} // namespace torch_tools
