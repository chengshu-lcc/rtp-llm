# N5 — GDN `_causal_conv1d_fwd_kernel` → `_causal_conv1d_fwd_split_kernel`

> **优化项 N5**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N5）。
>
> **目标**：把 GDN 层的 `_causal_conv1d_fwd_kernel` 换成 SGL 的 `_causal_conv1d_fwd_split_kernel`
> 变体（drop-in，相同算法但 load/compute 交错更好）。
>
> **状态**：⏳ TODO

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。Python-only Triton
改动，无需重编。

## 1. 问题与现象

| 实现 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP `_causal_conv1d_fwd_kernel` | 542.2 µs | 24 GDN 层 | 13 013 µs |
| SGL `_causal_conv1d_fwd_split_kernel` | 459.5 µs | 24 GDN 层 | 11 028 µs |

收益：83 µs / GDN-layer × 24 = **~2.0 ms / prefill**。

split 变体的核心思想：把 conv1d 沿 channel 维拆成多个独立 program，每个 program 处理一段
channel，让 DRAM load 和 conv compute 在不同 program 之间流水化。

## 2. 实施计划

### 2.1 Phase 1：定位 SGL 源码

```bash
find /root/sglang -name '*.py' -exec grep -l "_causal_conv1d_fwd_split_kernel" {} \;
```

预期：`sglang/srt/layers/attention/fla/causal_conv1d.py` 或类似路径。

### 2.2 Phase 2：定位 RTP 当前实现

```bash
grep -rn "_causal_conv1d_fwd_kernel\b" rtp_llm/models_py/triton_kernels/
```

预期在 `rtp_llm/models_py/triton_kernels/fla/causal_conv1d.py` 或 `mamba/`。

### 2.3 Phase 3：移植

drop-in 替换 kernel + 更新 caller 的 dispatcher（split 版有额外的 channel-tile 参数）。

### 2.4 Phase 4：数值/性能验证

- micro-bench：input shape 同 9B prefill 15k 的 conv1d 调用形状（D=4 conv 维 × C=channel）
  → max diff < 1e-2 BF16 噪声内
- 端到端：perf_test prefill_time 下降 ~2 ms

## 3. 风险

| 风险点 | 缓解 |
|---|---|
| split 变体对 channel 数有约束（需要被 tile size 整除） | 写 fallback：channel 不整除时回到旧 kernel |
| autotune 默认 config 在 RTP 这边未必最优 | 用 SGL 的 default config 起步，再用 `triton.autotune` 调一遍 |

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：定位 SGL `_causal_conv1d_fwd_split_kernel` 源码
- [ ] Phase 2：定位 RTP 当前实现 + caller
- [ ] Phase 3：移植 + dispatcher 更新
- [ ] Phase 4：micro-bench 数值/性能验证
- [ ] Phase 5：端到端 perf_test
- [ ] Phase 6：commit

---

*创建于 2026-04-21。*
