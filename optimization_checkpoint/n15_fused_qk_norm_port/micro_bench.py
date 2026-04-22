"""
N15 Phase-2 micro-bench: compare baseline vs new V2 fused QK RMSNorm kernel.

  (a) baseline: rtp_llm_ops.fused_qk_rmsnorm    (warp_size=32, two-pass)
  (b) V2:       rtp_llm_ops.fused_qk_rmsnorm_v2 (wave64, single-pass, warp-per-head)

Real Qwen3.5-9B Full-Attn shape (TP=2 per rank):
  head_num=8, kv_head_num=2, size_per_head=256
  hidden_states.shape = [m, q_size + 2*kv_size] = [m, 3072]

Gates:
  - max abs diff vs fp32 reference < 5e-2 (BF16 noise)
  - V2 p50 at m=15000 < 200us (vs baseline ~322us in production trace)
"""
import sys
sys.path.insert(0, "/root/rtp-llm")

import torch
from rtp_llm.ops.compute_ops import rtp_llm_ops


# --- shape config (Qwen3.5-9B, TP=2) ---
HEAD_NUM = 8
KV_HEAD_NUM = 2
SIZE_PER_HEAD = 256
EPS = 1e-6
DTYPE = torch.bfloat16
DEVICE = "cuda"

Q_SIZE = HEAD_NUM * SIZE_PER_HEAD          # 2048
KV_SIZE = KV_HEAD_NUM * SIZE_PER_HEAD      # 512
HIDDEN = Q_SIZE + 2 * KV_SIZE              # 3072


def make_inputs(m):
    torch.manual_seed(0)
    h = (torch.randn(m, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.1)
    qw = torch.randn(SIZE_PER_HEAD, dtype=DTYPE, device=DEVICE) * 0.5 + 1.0
    kw = torch.randn(SIZE_PER_HEAD, dtype=DTYPE, device=DEVICE) * 0.5 + 1.0
    return h, qw, kw


# --- (a) baseline ---
def run_baseline(h, qw, kw):
    m, n = h.shape
    rtp_llm_ops.fused_qk_rmsnorm(
        h, qw, kw, EPS, HEAD_NUM, KV_HEAD_NUM, m, n, SIZE_PER_HEAD,
    )
    return h


# --- (b) V2 ---
def run_v2(h, qw, kw):
    m, n = h.shape
    rtp_llm_ops.fused_qk_rmsnorm_v2(
        h, qw, kw, EPS, HEAD_NUM, KV_HEAD_NUM, m, n, SIZE_PER_HEAD,
    )
    return h


# --- fp32 reference for ground truth ---
def run_fp32_reference(h, qw, kw):
    """Pure-PyTorch fp32 RMSNorm applied to q + k slices, v untouched."""
    h32 = h.float().clone()
    qw32 = qw.float()
    kw32 = kw.float()

    q = h32[:, :Q_SIZE].view(-1, HEAD_NUM, SIZE_PER_HEAD)
    var = (q * q).mean(dim=-1, keepdim=True)
    q_out = q * torch.rsqrt(var + EPS) * qw32
    h32[:, :Q_SIZE] = q_out.view(-1, Q_SIZE)

    k = h32[:, Q_SIZE:Q_SIZE + KV_SIZE].view(-1, KV_HEAD_NUM, SIZE_PER_HEAD)
    var = (k * k).mean(dim=-1, keepdim=True)
    k_out = k * torch.rsqrt(var + EPS) * kw32
    h32[:, Q_SIZE:Q_SIZE + KV_SIZE] = k_out.view(-1, KV_SIZE)

    return h32.to(DTYPE)


def bench(fn, h_template, qw, kw, n_warmup=10, n_iter=100):
    for _ in range(n_warmup):
        h = h_template.clone()
        fn(h, qw, kw)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        h = h_template.clone()
        starts[i].record()
        fn(h, qw, kw)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[n_iter // 2], times[n_iter // 10], times[(9 * n_iter) // 10]


def numerical_check(h_template, qw, kw):
    ref = run_fp32_reference(h_template, qw, kw)

    out_base = h_template.clone()
    run_baseline(out_base, qw, kw)

    out_v2 = h_template.clone()
    run_v2(out_v2, qw, kw)

    def slice_diff(a, b):
        q_d = (a[:, :Q_SIZE] - b[:, :Q_SIZE]).abs().max().item()
        k_d = (a[:, Q_SIZE:Q_SIZE + KV_SIZE] - b[:, Q_SIZE:Q_SIZE + KV_SIZE]).abs().max().item()
        v_d = (a[:, Q_SIZE + KV_SIZE:] - b[:, Q_SIZE + KV_SIZE:]).abs().max().item()
        return q_d, k_d, v_d

    bq, bk, bv = slice_diff(out_base, ref)
    vq, vk, vv = slice_diff(out_v2, ref)
    cq, ck, cv = slice_diff(out_base, out_v2)

    print(f"  numerical max-abs-diff:")
    print(f"    baseline vs fp32-ref : q={bq:.2e}  k={bk:.2e}  v={bv:.2e}")
    print(f"    V2       vs fp32-ref : q={vq:.2e}  k={vk:.2e}  v={vv:.2e}")
    print(f"    V2       vs baseline : q={cq:.2e}  k={ck:.2e}  v={cv:.2e}")
    return max(vq, vk)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: head_num={HEAD_NUM}, kv_head_num={KV_HEAD_NUM}, size_per_head={SIZE_PER_HEAD}")
    print(f"  q_size={Q_SIZE}, kv_size={KV_SIZE}, hidden={HIDDEN}")
    print()

    for m in (15000, 4096, 256, 7):
        print(f"=== m = {m} (hidden_states.shape = [{m}, {HIDDEN}]) ===")
        h, qw, kw = make_inputs(m)

        max_diff = numerical_check(h, qw, kw)
        gate_ok = max_diff < 5e-2
        print(f"  gate (max abs diff < 5e-2): {'PASS' if gate_ok else 'FAIL'} (got {max_diff:.2e})")

        a_p50, a_p10, a_p90 = bench(run_baseline, h, qw, kw)
        b_p50, b_p10, b_p90 = bench(run_v2, h, qw, kw)
        speedup = a_p50 / b_p50
        print(f"  (a) baseline  fused_qk_rmsnorm    : "
              f"p10={a_p10:>7.1f}us  p50={a_p50:>7.1f}us  p90={a_p90:>7.1f}us")
        print(f"  (b) V2        fused_qk_rmsnorm_v2 : "
              f"p10={b_p10:>7.1f}us  p50={b_p50:>7.1f}us  p90={b_p90:>7.1f}us  "
              f"(speedup vs a: {speedup:.2f}x)")
        print()


if __name__ == "__main__":
    main()
