# N10 — GDN `_layer_norm_fwd_1pass_kernel` 重新 autotune

> **优化项 N10**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N10）。
>
> **目标**：同一份 Triton kernel `_layer_norm_fwd_1pass_kernel`，SGL 比 RTP 快 33%；估计是
> autotune config 没命中。重新跑 autotune，落最优 config 进 cache。
>
> **状态**：⏳ TODO（autotune cache rebuild）

---

## 0. 公共环境

参见父 checkpoint。Python-only。

## 1. 问题与现象

| 端 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP `_layer_norm_fwd_1pass_kernel` | 246.6 µs | 24 | 5 918 µs |
| SGL（**同 kernel name**）| 166.4 µs | 24 | 3 994 µs |

收益：80 µs / GDN-layer × 24 = **~1.9 ms / prefill**

> kernel name 完全一致 → 同一份 Triton 源码 → 区别只能在 autotune 选的 `BLOCK_SIZE` /
> `num_warps` / `num_stages`。

## 2. 实施计划

### 2.1 Phase 1：定位 autotune 配置

```bash
grep -rn "_layer_norm_fwd_1pass_kernel\|@triton.autotune" rtp_llm/models_py/triton_kernels/ \
  | head -20
```

找到 kernel 定义所在文件 + autotune `configs` 列表。

### 2.2 Phase 2：对照 SGL 的 config

```bash
grep -rn "_layer_norm_fwd_1pass_kernel" /root/sglang/
```

读 SGL 那份的 autotune configs，看哪几个 config 是 RTP 没列的。

### 2.3 Phase 3：扩 config + 实测

把 SGL 的 config 加到 RTP 的 autotune 列表（保留原列表 + 新增 SGL 的几条），跑端到端
prefill；Triton 的 autotune cache 会 lazy 选最快那条。

### 2.4 Phase 4：验证

跑父 checkpoint §0.3 的 perf_test，看 `_layer_norm_fwd_1pass_kernel` 的 p50 是否降到 ~166 µs；
端到端 prefill_time 应下降 ~1.9 ms。

## 3. 风险

| 风险点 | 缓解 |
|---|---|
| 加了 config 后 Triton 第一次跑要 autotune（编译多份 PTX，~10s 量级延迟） | 接受 warmup 成本；autotune cache 会持久化 |
| RTP 当前 config 在小 shape 上更快，加 SGL config 后 dispatcher 选错 | 用 `key` 参数把 config 按 shape 隔离 |

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：定位 kernel 定义 + 当前 autotune
- [ ] Phase 2：对照 SGL 的 config
- [ ] Phase 3：扩 config + 实测
- [ ] Phase 4：端到端验证 + commit

---

*创建于 2026-04-21。*
