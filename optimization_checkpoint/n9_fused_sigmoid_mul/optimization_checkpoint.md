# N9 — Full-Attn 输出 sigmoid + mul 融合 → `_fused_sigmoid_mul_kernel`

> **优化项 N9**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N9）。
>
> **目标**：把 RTP 当前用的 `vectorized_elementwise_kernel<8, sigmoid_kernel_cuda>` + 后续独立
> mul 换成 SGL 的单 kernel `_fused_sigmoid_mul_kernel`。
>
> **状态**：⏳ TODO（marginal，~0.1 ms）

---

## 0. 公共环境

参见父 checkpoint。Python-only。

## 1. 问题与现象

| 实现 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP `vectorized_elementwise_kernel<8, sigmoid_kernel_cuda>` | 144.5 µs | 8 | 1 156 µs |
| SGL `_fused_sigmoid_mul_kernel` | 130.2 µs | 8 | 1 042 µs |

收益：**~0.1 ms / prefill**（marginal，但属于"顺手做"，额外好处是少一次 launch overhead）。

## 2. 实施计划

工作区已有 compare 脚本：
- `rtp_llm/models_py/triton_kernels/fla/test/compare_fused_vs_unfused_sigmoid_gating.py`
- `rtp_llm/models_py/triton_kernels/fla/test/compare_rtp_vs_sglang_fused_sigmoid_gating.py`

### 2.1 调研

- 找 RTP 当前的 caller（grep `sigmoid_kernel_cuda` / `torch.sigmoid` 在 full-attn 输出 gating 处）
- 找 SGL 的 `_fused_sigmoid_mul_kernel` 实现，对照 compare 脚本里已经写过的版本

### 2.2 移植

- 把 fused kernel 落到 `rtp_llm/models_py/triton_kernels/fla/sigmoid_mul.py`
- caller 替换为单次调用

## 3. 风险

无明显风险，是 elementwise 操作，数值精度等价。

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：调研 + 对照 compare 脚本
- [ ] Phase 2：移植
- [ ] Phase 3：验证 + commit

## 5. 关联代码

- `rtp_llm/models_py/triton_kernels/fla/test/compare_*sigmoid_gating.py`（工作区已有，但未追踪）

---

*创建于 2026-04-21。*
