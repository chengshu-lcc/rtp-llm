# N4 — AllReduce → aiter `allreduce_prototype_twoshot`

> **优化项 N4**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N4）。
>
> **目标**：把 RTP 当前用的 NCCL `ncclDevKernel_Generic_4` 换成 SGL 用的 aiter
> `allreduce_prototype_twoshot<bf16, FP-codec>`。
>
> **状态**：⏳ TODO（应该是个 env-flip + 验证）

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。

> **关键**：父 checkpoint §0.1 当前启动命令里有 `ROCM_DISABLE_CUSTOM_AG=True FT_DISABLE_CUSTOM_AR=True`，
> 这两个就是关掉 aiter 自定义 AR / AG 的开关。本优化基本就是把这两个改成 False 后跑回归。

## 1. 问题与现象

| 实现 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP NCCL `ncclDevKernel_Generic_4` | 2 791.0 µs | 68 | 189 791 µs |
| SGL aiter `allreduce_prototype_twoshot<bf16, FP-codec>` | 2 755.3 µs | 65 | 179 097 µs |

收益：~36 µs/call × ~65 calls = **~2.3 ms / prefill**（per-layer 比较）。算上调用次数差异，
总收益 ≈ **10.7 ms**。

> 注：单 call 差异只有 1.3%，但累积 65 次后变成 10.7 ms。在 8 GPU 配置下 aiter twoshot 优势
> 更明显（NCCL 调度开销随 rank 数线性增长，twoshot 是固定 2 跳）。

## 2. 实施计划

### 2.1 Phase 1：环境变量翻转

修改父 checkpoint §0.1 的启动命令：

```diff
-ROCM_DISABLE_CUSTOM_AG=True FT_DISABLE_CUSTOM_AR=True \
+ROCM_DISABLE_CUSTOM_AG=False FT_DISABLE_CUSTOM_AR=False \
```

或直接 `unset ROCM_DISABLE_CUSTOM_AG FT_DISABLE_CUSTOM_AR`（两个 flag 默认值需要查
`rtp_llm/server/server_args/parallel_group_args.py`）。

### 2.2 Phase 2：审计代码

```bash
grep -rn "ROCM_DISABLE_CUSTOM_AG\|FT_DISABLE_CUSTOM_AR\|allreduce_prototype" rtp_llm/ \
  | grep -v test
```

确认：
- 关掉 disable flag 后会走 `aiter.allreduce_prototype_twoshot` 这条路径
- 默认 codec 是 `FP-codec`（与 SGL 对照）
- TP=2 / TP=4 / TP=8 都覆盖

### 2.3 Phase 3：正确性回归

aiter 的 custom AR 是有数值精度风险的（especially fp16/bf16 + accumulation order）。需要：
- 跑数学一致性检查：fixed-seed 推理 → 对比 logits（NCCL vs aiter twoshot）→ 期望 BF16 噪声内
- 跑端到端 inference 抽样：回答内容是否一致（hellaswag / mmlu 一两个 sample）

### 2.4 Phase 4：性能验证

跑父 checkpoint §0.3 的 perf_test `--partial=2`，期望 prefill_time 下降 ~10 ms。

## 3. 风险

| 风险点 | 缓解 |
|---|---|
| 这两个 flag 在父 checkpoint 启动命令里默认是 True，关掉后 trace 里没出现 aiter twoshot？ | 排查 caller：是不是某个 rank 数下 aiter 不可用、回退 NCCL；查 `aiter` 的 dispatch 逻辑 |
| aiter custom AR 在 prefill big shape (T=15k×D=4096) 上 numerically 不稳 | 对比 NCCL 的 fp32 reduce vs aiter 的 fp16/bf16 reduce 的 max diff；如有可观测 drift 需要换 codec |
| 8 GPU 拓扑下 twoshot 路由可能要额外 init | 看 aiter 文档，必要时设 `AITER_TWOSHOT_TOPOLOGY` 等 env |

## 4. Phase 状态

- [ ] Phase 0：checkpoint
- [ ] Phase 1：env flip 实验
- [ ] Phase 2：代码审计（确认默认值与 aiter dispatch）
- [ ] Phase 3：数值正确性回归
- [ ] Phase 4：端到端 perf_test
- [ ] Phase 5：commit（如果是改 env，commit 启动脚本即可）

## 5. 关联代码

- `rtp_llm/server/server_args/parallel_group_args.py` —— `ROCM_DISABLE_CUSTOM_AG / FT_DISABLE_CUSTOM_AR` 默认值
- `rtp_llm/cpp/devices/rocm_impl/` —— AR 实现入口
- aiter 实现：`/opt/conda310/lib/python3.10/site-packages/aiter/`（grep `allreduce_prototype_twoshot`）

---

*创建于 2026-04-21。*
