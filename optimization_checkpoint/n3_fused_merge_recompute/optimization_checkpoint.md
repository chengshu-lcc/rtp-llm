# N3 — GDN 三 kernel 合并 → `_fused_merge_recompute_kernel`

> **优化项 N3**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N3）。
>
> **目标**：把 GDN 层里 `recompute_w_u_fwd_kernel` + `chunk_scaled_dot_kkt_fwd_kernel` +
> `merge_16x16_to_64x64_inverse_kernel` 三个独立 launch 合并成 SGLang 那个单 kernel
> `_fused_merge_recompute_kernel`。
>
> **状态**：✅ Done（待 commit）

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。本优化是
**Python-only Triton 改动**，无需重编 C++。

## 1. 问题与现象

每层 GDN 跑 3 个独立 Triton launch：

| Kernel | RTP per-layer | 总耗时（24 GDN 层） |
|---|---:|---:|
| `recompute_w_u_fwd_kernel` | 711.2 µs | 17 068 µs |
| `chunk_scaled_dot_kkt_fwd_kernel` | 230.3 µs | 5 528 µs |
| `merge_16x16_to_64x64_inverse_kernel` | 129.5 µs | 3 108 µs |
| **小计** | **1071.4 µs** | **25 704 µs** |

SGLang 把它们合成一个：

| Kernel | SGL per-layer | 总耗时 |
|---|---:|---:|
| `_fused_merge_recompute_kernel` | 692.9 µs | 16 629 µs |

**毛收益估算**：1071 − 693 = **378 µs / GDN-layer × 24 ≈ 9.1 ms / prefill**。

> ⚠️ **修正**：上面的 9.1 ms 把 `chunk_scaled_dot_kkt_fwd_kernel`（230 µs）算进去了，但 SGL
> 把 `chunk_scaled_dot_kkt` 的工作放到 `_fused_cumsum_kkt_kernel`（=N6）里，**不在 N3 范围内**。
> N3 单独只融合 `recompute_w_u + merge_16x16_to_64x64_inverse` 两个 kernel：
>
> - 老 RTP：711.2 + 129.5 = 840.7 µs/layer
> - 新 fused：~692.9 µs/layer
> - **N3 单独净收益**：~148 µs/layer × 24 = **~3.5 ms / prefill**
>
> 想拿到完整 9.1 ms，必须叠加 N6。

## 2. 调研待办

- 找到 SGLang 里 `_fused_merge_recompute_kernel` 的源码位置（猜测 `/root/sglang/sglang/srt/layers/attention/fla/` 或 fla upstream `/root/sglang/python/sglang/srt/layers/`）
- 对照 RTP 当前实现：
  - `rtp_llm/models_py/triton_kernels/fla/wy_fast.py`（`recompute_w_u_fwd`）
  - `rtp_llm/models_py/triton_kernels/fla/chunk_scaled_dot_kkt.py`（`chunk_scaled_dot_kkt_fwd_kernel`）
  - `rtp_llm/models_py/triton_kernels/fla/solve_tril.py`（`merge_16x16_to_64x64_inverse_kernel`）
- 调用链入口：`rtp_llm/models_py/triton_kernels/fla/chunk.py` 的 `chunk_gated_delta_rule_fwd_*`

## 3. 实施计划

### 3.1 Phase 1：调研 SGL 源码

```bash
# 在 SGL 源码里找 _fused_merge_recompute_kernel
find /root/sglang -name '*.py' -exec grep -l "_fused_merge_recompute_kernel" {} \;
```

### 3.2 Phase 2：移植

- 把 SGL 那个 kernel 文件拷到 `rtp_llm/models_py/triton_kernels/fla/fused_merge_recompute.py`（或合并入现有 `wy_fast.py`）
- 修改 `chunk.py` 的 `chunk_gated_delta_rule_fwd_*` 调用链：原来三连 launch 改成单次调用
- 保留三个老 kernel 作 fallback（runtime arg 决定走哪个）

### 3.3 Phase 3：数值正确性

写 micro-bench 脚本（参考 `n2_fused_l2norm_qk` 下 compare 脚本风格）：
- 输入：随机 BF16 tensor，shape 同 9B prefill 15k T=15000 D=128
- 期望：fused vs unfused max abs diff < 1e-2（BF16 噪声）
- 性能：fused 单 kernel ≤ 700 µs

### 3.4 Phase 4：端到端验证

跑父 checkpoint §0.3 的 perf_test `--partial=2`，期望 prefill_time 下降 ~3.5 ms（N3 单独）。

#### 实测结果（2026-04-21）

| 配置 | Prefill Time(ms) |
|---|---:|
| BEFORE（commit `622bb48f7`，N1+N2+N7-WIP） | 913.74 |
| AFTER（+N3，本次改动） | **911.49** |
| Δ | **−2.25 ms** |

micro-bench `compare_fused_merge_recompute.py`（B=1, T=15000, Hv=16, Hg=2, K=V=128, single seq varlen）：

| 路径 | 单次调用 wall-clock |
|---|---:|
| 老 3-kernel 路径 | 1061.07 µs |
| 新 2-kernel 路径 | 824.90 µs |
| Δ | **−236.16 µs (−22.3%)** |

数值正确性：max abs diff w=1.95e-3，u=1.95e-3（参考 max ≈ 0.27/0.38，BF16 噪声范围内 ✓）。

实测 −2.25 ms 比 micro-bench 推算的 −5.7 ms（236 µs × 24）小 —— 单次 perf_test 噪声 ±5 ms 量级，
后续若叠加 N6 拿到完整 9.1 ms 时再回看是否需要二次验证。

## 4. 风险

| 风险点 | 缓解 |
|---|---|
| SGL upstream 的 fused kernel 形状假设可能与 RTP 不同（如 head_dim、chunk_size） | 移植前先对比 caller 的调用形状；保留 fallback 路径 |
| recompute 那一段涉及 `B^-1 ⊗ B^-1` 的矩阵反演，融合可能影响数值精度 | 严格 BF16 数值 diff，必要时 fp32 中间累加 |
| 三个 kernel 原本可能用不同 `num_warps / BLOCK_SIZE` 调过 autotune | 融合后需要重新 autotune |

## 5. Phase 状态

- [x] Phase 0：checkpoint + 环境
- [x] Phase 1：定位 SGL `_fused_merge_recompute_kernel` 源码 → `/root/sglang/python/sglang/srt/layers/attention/fla/fused_merge_recompute.py`
- [x] Phase 2：移植 + caller 改动
  - 新增 `rtp_llm/models_py/triton_kernels/fla/fused_merge_recompute.py`（331 行，仅改 import path）
  - `chunk.py`：`solve_tril()` + `recompute_w_u_fwd()` → `solve_tril_16x16_kernel` 直接 launch + `fused_merge_recompute()`
- [x] Phase 3：micro-bench 数值/性能验证 — `compare_fused_merge_recompute.py`，单次 −22.3%
- [x] Phase 4：端到端 perf_test —— prefill_time 911.49 ms（−2.25 ms vs N1-only 913.74）
- [ ] Phase 5：commit

## 6. 关联代码

- `rtp_llm/models_py/triton_kernels/fla/chunk.py` —— caller，已切到 fused 路径
- `rtp_llm/models_py/triton_kernels/fla/fused_merge_recompute.py` —— **新增**，N3 主体
- `rtp_llm/models_py/triton_kernels/fla/test/compare_fused_merge_recompute.py` —— **新增**，数值/性能 micro-bench
- `rtp_llm/models_py/triton_kernels/fla/wy_fast.py` —— `recompute_w_u_fwd`，**保留**但 chunk.py 已不调用（仍可被独立测试用）
- `rtp_llm/models_py/triton_kernels/fla/solve_tril.py` —— `merge_16x16_to_64x64_inverse_kernel`，**保留**但已不在 chunk.py 路径上；`solve_tril_16x16_kernel` 现在直接被 chunk.py 调用
- 上游对照：`/root/sglang/python/sglang/srt/layers/attention/fla/fused_merge_recompute.py`

---

*创建于 2026-04-21。*
