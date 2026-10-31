/**
 * @file bind.cpp
 * @brief Pybind11 Python bindings for torch_tools (RoPE module and functions).
 * @author Ansh Raj
 * @version 1.0.1
 * @date 2026
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include "torch_tools/rope.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Ultra-optimized C++ / LibTorch implementations of Rotary Position Embedding (RoPE)";

    // Bind RotaryEmbeddingOptions struct using explicit lambdas to avoid overloaded method ambiguity
    py::class_<torch_tools::RotaryEmbeddingOptions>(m, "RotaryEmbeddingOptions")
        .def(py::init<int64_t>(), py::arg("head_dim"))
        .def_property("head_dim", 
            [](const torch_tools::RotaryEmbeddingOptions& self) { return self.head_dim(); },
            [](torch_tools::RotaryEmbeddingOptions& self, int64_t val) { self.head_dim(val); })
        .def_property("max_seq_len", 
            [](const torch_tools::RotaryEmbeddingOptions& self) { return self.max_seq_len(); },
            [](torch_tools::RotaryEmbeddingOptions& self, int64_t val) { self.max_seq_len(val); })
        .def_property("base", 
            [](const torch_tools::RotaryEmbeddingOptions& self) { return self.base(); },
            [](torch_tools::RotaryEmbeddingOptions& self, double val) { self.base(val); })
        .def_property("seq_dim", 
            [](const torch_tools::RotaryEmbeddingOptions& self) { return self.seq_dim(); },
            [](torch_tools::RotaryEmbeddingOptions& self, int64_t val) { self.seq_dim(val); });

    // Bind RotaryEmbeddingImpl Module
    py::class_<torch_tools::RotaryEmbeddingImpl, std::shared_ptr<torch_tools::RotaryEmbeddingImpl>>(m, "RotaryEmbeddingImpl")
        .def(py::init<int64_t, int64_t, double, int64_t>(),
             py::arg("head_dim"),
             py::arg("max_seq_len") = 4096,
             py::arg("base") = 10000.0,
             py::arg("seq_dim") = 1)
        .def("forward", &torch_tools::RotaryEmbeddingImpl::forward,
             py::arg("x"),
             py::arg("seq_offset") = 0)
        .def("forward_qk", &torch_tools::RotaryEmbeddingImpl::forward_qk,
             py::arg("q"),
             py::arg("k"),
             py::arg("seq_offset") = 0)
        .def("update_cos_sin_cache", &torch_tools::RotaryEmbeddingImpl::update_cos_sin_cache,
             py::arg("seq_len"),
             py::arg("device"),
             py::arg("dtype"))
        .def_readwrite("options", &torch_tools::RotaryEmbeddingImpl::options);

    // Bind RotaryEmbedding ModuleHolder wrapper
    py::class_<torch_tools::RotaryEmbedding>(m, "RotaryEmbedding")
        .def(py::init<int64_t, int64_t, double, int64_t>(),
             py::arg("head_dim"),
             py::arg("max_seq_len") = 4096,
             py::arg("base") = 10000.0,
             py::arg("seq_dim") = 1)
        .def("__call__", [](torch_tools::RotaryEmbedding& self, const torch::Tensor& x, int64_t seq_offset) {
            return self->forward(x, seq_offset);
        }, py::arg("x"), py::arg("seq_offset") = 0)
        .def("forward", [](torch_tools::RotaryEmbedding& self, const torch::Tensor& x, int64_t seq_offset) {
            return self->forward(x, seq_offset);
        }, py::arg("x"), py::arg("seq_offset") = 0)
        .def("forward_qk", [](torch_tools::RotaryEmbedding& self, const torch::Tensor& q, const torch::Tensor& k, int64_t seq_offset) {
            return self->forward_qk(q, k, seq_offset);
        }, py::arg("q"), py::arg("k"), py::arg("seq_offset") = 0);

    // Expose functional apply_rope interface
    m.def("apply_rope", &torch_tools::apply_rope,
          py::arg("x"),
          py::arg("cos"),
          py::arg("sin"),
          py::arg("seq_dim") = 1);
}
