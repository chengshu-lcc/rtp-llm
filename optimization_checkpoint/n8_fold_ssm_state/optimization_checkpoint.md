# N8 — Fold `store_ssm_state_to_block_map` + `load_initial_state_from_block_map` 进 chunk_o 尾段

> **优化项 N8**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N8）。
>
> **目标**：把 `store_ssm_state_to_block_map` + `load_initial_state_from_block_map` 两个小 kernel
> 折进 `chunk_fwd_kernel_o` 的 epilogue（参考 SGL 的 `_scatter_fused_proj_kernel` 模式）。
>
> **状态**：⏳ TODO（小项，~0.5 ms 收益）

---

## 0. 公共环境

参见父 checkpoint。Python-only。

## 1. 问题与现象

| 项 | RTP per-layer | SGL per-layer |
|---|---:|---:|
| `store_ssm_state_to_block_map` + `load_initial_state_from_block_map` | 22.6 µs（合计） | 0（被吸进 `_scatter_fused_proj_kernel`） |

收益：22.6 µs / GDN-layer × 24 ≈ **~0.5 ms / prefill**。

## 2. 实施计划

### 2.1 调研

```bash
grep -rn "store_ssm_state_to_block_map\|load_initial_state_from_block_map" rtp_llm/models_py/triton_kernels/
grep -rn "_scatter_fused_proj_kernel" /root/sglang/
```

### 2.2 移植

把这两个小 kernel 的 store/load 写进 `chunk_fwd_kernel_o` 的 epilogue。注意：
- 这两个 kernel 当前可能跨 stream（fork-join）；融合后单 stream 即可
- block_map 索引要预计算好，融合后的 store/load 就只是按下标写

## 3. 风险

| 风险点 | 缓解 |
|---|---|
| `chunk_fwd_kernel_o` 已经被 N11 重 autotune 过（如果 N11 先做） | 融合后再 autotune 一次 |
| 两个 kernel 可能依赖不同的 block_map metadata，融合需要重组 | 看 caller，把 metadata 提前计算 |

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：调研代码
- [ ] Phase 2：融合 + 验证
- [ ] Phase 3：commit

---

*创建于 2026-04-21。*
