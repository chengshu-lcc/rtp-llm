#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 micro-bench: old GDN WY path vs fused _fused_merge_recompute_kernel.

Old path (3 kernels):
    chunk_scaled_dot_kkt_fwd  ->  solve_tril (= solve_tril_16x16 + merge_16x16_to_64x64_inverse)
                              ->  recompute_w_u_fwd

New path (2 kernels, N3):
    chunk_scaled_dot_kkt_fwd  ->  solve_tril_16x16_kernel  ->  fused_merge_recompute

Both consume identical k/v/beta/g_cumsum/A and produce w, u of identical shapes.
Asserts max abs diff in BF16 noise band, then prints per-kernel µs.

Shape: Qwen3.5-9B GDN prefill, T=15000, varlen single seq.
"""

import time
from typing import Tuple

import torch
import triton

from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.fused_merge_recompute import (
    fused_merge_recompute,
)
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.solve_tril import (
    solve_tril,
    solve_tril_16x16_kernel,
)
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


def _bench(fn, warmup: int = 5, iters: int = 50) -> float:
    """Returns mean wall-clock µs across `iters` after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def old_path(k, v, beta, g_raw, cu_seqlens) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(g_raw, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, g_cumsum=g, A=Ai, cu_seqlens=cu_seqlens
    )
    return g, w, u


def new_path(k, v, beta, g_raw, cu_seqlens) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = chunk_local_cumsum(g_raw, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    B, T, Hv = g.shape
    chunk_indices_16 = (
        prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    )
    NT_16 = len(chunk_indices_16) if cu_seqlens is not None else triton.cdiv(T, 16)
    Ai16 = torch.empty(B, T, Hv, 16, device=A.device, dtype=torch.float32)
    solve_tril_16x16_kernel[(NT_16, B * Hv)](
        A=A,
        Ad=Ai16,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_16,
        T=T,
        H=Hv,
        BT=64,
        num_warps=1,
        num_stages=4,
    )
    w, u = fused_merge_recompute(
        k=k, v=v, beta=beta, g_cumsum=g, A=A, Ai16=Ai16,
        chunk_size=64, cu_seqlens=cu_seqlens,
    )
    return g, w, u


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3.5-9B GDN per-rank shape (TP=2): Hv=16, Hg=2, K=V=128, T=15000 single seq
    T = 15000
    Hv = 16
    Hg = 2
    K = 128
    V = 128
    B = 1

    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, Hv, V, device=device, dtype=dtype) * 0.1
    beta = torch.rand(B, T, Hv, device=device, dtype=dtype).sigmoid()
    g_raw = torch.nn.functional.logsigmoid(
        torch.randn(B, T, Hv, device=device, dtype=torch.float32)
    ).to(dtype)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)

    # 1. Numerical parity
    g_old, w_old, u_old = old_path(k, v, beta, g_raw, cu_seqlens)
    g_new, w_new, u_new = new_path(k, v, beta, g_raw, cu_seqlens)

    assert torch.equal(g_old, g_new), "g_cumsum differs (should be identical)"

    def _stats(name, a, b):
        diff = (a.float() - b.float()).abs()
        print(
            f"  {name}: max_abs={diff.max().item():.4e}  "
            f"mean_abs={diff.mean().item():.4e}  "
            f"ref_max={b.float().abs().max().item():.4e}"
        )
        return diff.max().item()

    print("Numerical parity (old vs new):")
    max_w = _stats("w", w_new, w_old)
    max_u = _stats("u", u_new, u_old)
    bf16_tol = 5e-2
    assert max_w < bf16_tol, f"w max abs diff {max_w} >= {bf16_tol}"
    assert max_u < bf16_tol, f"u max abs diff {max_u} >= {bf16_tol}"
    print("  ✓ within BF16 tolerance\n")

    # 2. Per-path wall-clock (kernel launch + compute, host-side)
    print("Wall-clock per call (50 iters, after 5 warmup):")
    t_old = _bench(lambda: old_path(k, v, beta, g_raw, cu_seqlens))
    t_new = _bench(lambda: new_path(k, v, beta, g_raw, cu_seqlens))
    print(f"  old path (3 kernels post-cumsum): {t_old:7.2f} µs")
    print(f"  new path (2 kernels post-cumsum): {t_new:7.2f} µs")
    print(f"  delta: {t_old - t_new:+.2f} µs ({(t_old - t_new) / t_old * 100:+.1f}%)")


if __name__ == "__main__":
    main()
