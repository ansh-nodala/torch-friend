from typing import Tuple
import torch
import torch.nn as nn
from . import torch_tools_cpp as _C


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_dim: int = 1,
) -> torch.Tensor:
    return _C.apply_rope(x, cos, sin, seq_dim)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        seq_dim: int = 1,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.seq_dim = seq_dim
        self._cpp_mod = _C.RotaryEmbedding(head_dim, max_seq_len, base, seq_dim)

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        return self._cpp_mod.forward(x, seq_offset)

    def forward_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cpp_mod.forward_qk(q, k, seq_offset)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, "
            f"base={self.base}, seq_dim={self.seq_dim}"
        )
