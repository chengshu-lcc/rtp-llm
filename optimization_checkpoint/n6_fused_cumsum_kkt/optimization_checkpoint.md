# N6 — GDN `chunk_scaled_dot_kkt + chunk_local_cumsum` 融合

> **优化项 N6**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N6）。
>
> **目标**：把 GDN 层的 `chunk_scaled_dot_kkt_fwd_kernel` + `chunk_local_cumsum_scalar_kernel`
> 融合成 SGL 的 `_fused_cumsum_kkt_kernel`。
>
> **状态**：⏳ TODO（依赖 N3；如果 N3 已经把 `chunk_scaled_dot_kkt` 吸进去了，本项净收益只剩 ~0.6 ms）

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。Python-only，无需重编。

## 1. 问题与现象

| 实现 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP `chunk_scaled_dot_kkt_fwd_kernel` | 230.3 µs | 24 | 5 528 µs |
| RTP `chunk_local_cumsum_scalar_kernel` | 25 µs | 24 | 600 µs |
| **小计** | **255.1 µs** | 24 | 6 128 µs |
| SGL `_fused_cumsum_kkt_kernel` | 203.7 µs | 24 | 4 889 µs |

毛收益：51 µs / GDN-layer × 24 = **~1.2 ms / prefill**

**重要**：N3（`_fused_merge_recompute_kernel`）已经把 `chunk_scaled_dot_kkt` 那 230 µs 吸进
合并 kernel 里了。如果 N3 已落地，N6 的净增量只有 `chunk_local_cumsum_scalar` 那 25 µs × 24
= ~0.6 ms，价值大幅缩水。

> **执行顺序建议**：先做 N3，N3 落完再决定 N6 是否还值得做。

## 2. 实施计划（如果不做 N3 而单独做 N6）

### 2.1 Phase 1：定位 SGL `_fused_cumsum_kkt_kernel` 源码

```bash
find /root/sglang -name '*.py' -exec grep -l "_fused_cumsum_kkt_kernel" {} \;
```

### 2.2 Phase 2：移植

参考 N3 / N5 的移植模式。caller 入口：`rtp_llm/models_py/triton_kernels/fla/chunk.py`。

## 3. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：等 N3 决策
- [ ] Phase 2：（如果还做）SGL 源码定位
- [ ] Phase 3：移植 + 验证

---

*创建于 2026-04-21。*
