"""Micro-benchmark of aiter add+rmsnorm variants on Qwen3.5-9B prefill shape.

Validates Phase 2 of N1 optimization:
  - Confirms aiter.rmsnorm2d_fwd_with_add_ck (path C) is numerically correct vs
    the current aiter.ops.triton.normalization.rmsnorm.rmsnorm2d_fwd_with_add (path A).
  - Confirms aiter.rmsnorm2d_fwd_with_add (path B) is BROKEN in this aiter pin
    (output values ~15000 instead of ~-0.6) — do not use without further investigation.
  - Reports per-call latency on MI308X.

Run:
    LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH \
      /opt/conda310/bin/python3.10 micro_bench.py
"""
import time
import torch
import aiter
from aiter.ops.triton.normalization.rmsnorm import (
    rmsnorm2d_fwd_with_add as triton_fused,
)

T, D = 15367, 4096
DTYPE = torch.bfloat16
EPS = 1e-6
N_WARMUP = 20
N_ITERS = 200


def bench(label, fn, ref_out, ref_res):
    out = torch.empty_like(ref_out)
    res_out = torch.empty_like(ref_res)
    fn(out, res_out)
    torch.cuda.synchronize()
    diff = (out.float() - ref_out.float()).abs()
    rdiff = (res_out.float() - ref_res.float()).abs()
    correct = diff.max().item() < 0.1
    tag = "✅" if correct else "❌ WRONG"
    print(f"  {tag}  out diff max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    print(f"        res diff max={rdiff.max().item():.3e} mean={rdiff.mean().item():.3e}")
    print(f"        sample: {out[0, :4].tolist()}")
    for _ in range(N_WARMUP):
        fn(out, res_out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn(out, res_out)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / N_ITERS * 1e6
    print(f"        latency: {us:.1f} us/call")
    return us, correct


def main():
    torch.manual_seed(0)
    x = torch.randn(T, D, dtype=DTYPE, device="cuda") * 0.1
    res = torch.randn(T, D, dtype=DTYPE, device="cuda") * 0.1
    w = torch.randn(D, dtype=DTYPE, device="cuda") * 0.05 + 1.0

    # Reference: current Triton kernel
    ref_out = torch.empty_like(x)
    ref_res = torch.empty_like(x)
    triton_fused(ref_out, x, res, ref_res, w, EPS)
    torch.cuda.synchronize()

    print(f"Shape: T={T}, D={D}, dtype={DTYPE}\n")

    print("[A] aiter.ops.triton.normalization.rmsnorm.rmsnorm2d_fwd_with_add (CURRENT)")
    bench("triton",
          lambda o, ro: triton_fused(o, x, res, ro, w, EPS),
          ref_out, ref_res)

    print("\n[B] aiter.rmsnorm2d_fwd_with_add (default dispatcher → module_rmsnorm_quant)")
    bench("dispatcher",
          lambda o, ro: aiter.rmsnorm2d_fwd_with_add(o, x, res, ro, w, EPS),
          ref_out, ref_res)

    print("\n[C] aiter.rmsnorm2d_fwd_with_add_ck (TARGET — drop-in CK)")
    bench("ck",
          lambda o, ro: aiter.rmsnorm2d_fwd_with_add_ck(o, x, res, ro, w, EPS),
          ref_out, ref_res)

    print("\n[D] aiter.fused_add_rms_norm_cu (in-place HIP)")
    xc = x.clone()
    resc = res.clone()
    def cu_fn(_o, _ro):
        xc.copy_(x); resc.copy_(res)
        aiter.fused_add_rms_norm_cu(xc, resc, w, EPS)
    cu_fn(None, None); torch.cuda.synchronize()
    diff = (xc.float() - ref_out.float()).abs()
    rdiff = (resc.float() - ref_res.float()).abs()
    print(f"  ✅  in-place x diff max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    print(f"        in-place res diff max={rdiff.max().item():.3e} mean={rdiff.mean().item():.3e}")
    for _ in range(N_WARMUP):
        xc.copy_(x); resc.copy_(res)
        aiter.fused_add_rms_norm_cu(xc, resc, w, EPS)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        aiter.fused_add_rms_norm_cu(xc, resc, w, EPS)
    torch.cuda.synchronize()
    print(f"        latency: {(time.perf_counter()-t0)/N_ITERS*1e6:.1f} us/call")


if __name__ == "__main__":
    main()
