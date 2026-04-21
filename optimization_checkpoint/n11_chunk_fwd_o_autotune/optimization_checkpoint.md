# N11 — GDN `chunk_fwd_kernel_o` 重新 autotune

> **优化项 N11**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N11）。
>
> **目标**：同一份 Triton kernel `chunk_fwd_kernel_o`，SGL 比 RTP 快 24%；估计是 autotune
> config 没命中 15k-token GDN shape 的最优配置。
>
> **状态**：⏳ TODO（autotune cache rebuild）

---

## 0. 公共环境

参见父 checkpoint。Python-only。

## 1. 问题与现象

| 端 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP `chunk_fwd_kernel_o` | 844.2 µs | 24 | 20 261 µs |
| SGL（**同 kernel name**）| 643.7 µs | 24 | 15 449 µs |

收益：**~4.8 ms / prefill** —— 是 N11 类自动调优中最大的一项，仅次于 N7 / N1 / N2 / N4 / N3。

> 注意：父 checkpoint §2 已经标注 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 是 RTP
> 反而更快的项（+3.9 ms）；调 N11 的时候不要顺手把同 .py 文件里的相邻 kernel 也重 autotune
> 给搞慢了。

## 2. 实施计划

### 2.1 Phase 1：定位 autotune 配置

```bash
grep -rn "chunk_fwd_kernel_o\b" rtp_llm/models_py/triton_kernels/
```

预期在 `rtp_llm/models_py/triton_kernels/fla/chunk.py` 或 `fla/chunk_fwd_o.py`。

### 2.2 Phase 2：对照 SGL config

```bash
grep -rn "chunk_fwd_kernel_o" /root/sglang/
```

### 2.3 Phase 3：扩 config + 验证

同 N10 的方法：把 SGL 的 config 合进 RTP 的 autotune configs 列表，让 Triton runtime 选最快那条。

> Tip：可以直接用 `triton.testing.do_bench` 在固定 input shape (T=15000, D=128, GDN heads…)
> 上 micro-bench 各 config，**手动**选最优 config 写死，避免 lazy autotune 影响首次延迟。

### 2.4 Phase 4：端到端验证

跑父 checkpoint §0.3 的 perf_test。期望：
- `chunk_fwd_kernel_o` p50 → ~644 µs
- prefill_time 下降 ~4.8 ms

## 3. 风险

| 风险点 | 缓解 |
|---|---|
| 同文件里 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 也用相邻 autotune，可能被一起影响 | 严格区分 kernel 名，不要扩到不该改的 kernel |
| SGL 的 config 在 RTP 当前 Triton 版本上不被支持 | 检查 num_stages 等参数是否需要适配 |
| 加 config 后第一次 prefill 要 autotune（warmup 慢一拍） | 接受 |

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：定位 kernel + 当前 autotune
- [ ] Phase 2：对照 SGL config
- [ ] Phase 3：扩 config / 写死最优 config
- [ ] Phase 4：端到端验证 + commit

---

*创建于 2026-04-21。*
