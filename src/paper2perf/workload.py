"""Trusted GEMM materialization. Model-authored Python is never imported."""
from .schemas import Gemm


def materialize(contract: Gemm) -> str:
    # All interpolated fields are validated integers or an enum.
    return f'''import torch
from torch import nn

class Workload(nn.Module):
    def __init__(self, K={contract.K}, N={contract.N}):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((K, N)))

    def forward(self, x):
        return torch.mm(x, self.weight)
'''


def gemm_class():
    import torch

    class GemmModule(torch.nn.Module):
        def __init__(self, K, N):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty((K, N)))

        def forward(self, x):
            return torch.mm(x, self.weight)

    return GemmModule


def validate_numerically(contract: Gemm):
    import torch
    # Small deterministic probe checks template semantics, independent of simulated shape.
    generator = torch.Generator().manual_seed(17)
    a = torch.randn((3, 5), generator=generator)
    b = torch.randn((5, 4), generator=generator)
    module = gemm_class()(5, 4)
    with torch.no_grad():
        module.weight.copy_(b)
    observed = module(a)
    expected = (a[:, :, None] * b[None, :, :]).sum(dim=1)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
    return {"passed": True, "operator": "gemm", "shape": [contract.M, contract.K, contract.N],
            "note": "Trusted torch.mm template; small FP32 semantic probe, not a latency measurement"}
