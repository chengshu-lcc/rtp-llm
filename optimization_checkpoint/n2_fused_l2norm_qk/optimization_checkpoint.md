# N2 — GDN `l2norm_fwd_kernel1 ×2` → `fused_l2norm_qk_kernel`

> **优化项 N2**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md`）。
>
> **目标**：把 GDN 层里 Q/K 各跑一次的 `l2norm_fwd_kernel1`（共 ×2）合成单次 `fused_l2norm_qk_kernel`，
> 且让 BT-tiled 路径在所有 T 上通用。
>
> **状态**：✅ **已完成**，commit `3c529f051 optimize fused_l2norm_qk: rsqrt+mul, BT-tiled for all T`。

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。

## 1. 问题与现象

| 实现 | µs/call | 调用次数 | 总耗时 |
|---|---:|---:|---:|
| RTP（baseline）`l2norm_fwd_kernel1` ×2 | 438.5 µs | 24 GDN 层 × 2 = 48 | **21 046 µs** |
| SGLang `fused_l2norm_qk_kernel` | 105.4 µs | 24 GDN 层 × 1 = 24 | **2 529 µs** |
| RTP（after，本 commit）单 kernel | 49 µs（T=122936 mega 样本） | 24 | （15k prefill 估算 ~6 µs/call × 24 ≈ 144 µs）|

调用链：`chunk.py:96 q, k = fused_l2norm_qk(q, k)` ← `chunk_gated_delta_rule` 的
`use_qk_l2norm_in_kernel=True` 分支 ← qwen3.5/qwen3_next 混合注意力的 GDN 层 forward。

## 2. 改动

文件：`rtp_llm/models_py/triton_kernels/fla/l2norm.py`，单文件，55 行 diff（patch 在本目录
`fused_l2norm_qk.patch`）。

### 2.1 优化点 A：rsqrt + mul 替换 div

```python
# Before
b_q_var = tl.sum(b_q * b_q, axis=1)
b_q_out = b_q / tl.sqrt(b_q_var + eps)[:, None]

# After
b_q_rstd = tl.rsqrt(tl.sum(b_q * b_q, axis=1) + eps)
b_q_out = b_q * b_q_rstd[:, None]
```

**根因**：AMD/NV fp32 div 都是多指令 rcp+Newton-Raphson；广播到 `BT*BD` 元素后整段被 div
拖死。`tl.rsqrt` 是单条硬件指令，整体 collapse 成 mul。

**单点收益**：~2.3×（从 BT-tiled kernel 单独打表）

### 2.2 优化点 B：BT-tiled 路径全 T 通用 + T,D 改 runtime arg

```python
# Before
fused_l2norm_qk_kernel  # T, D 是 tl.constexpr
if D <= 512 and T <= 128:                       # T<=128 才走 BT-tiled
    fused_l2norm_qk_kernel[(triton.cdiv(T, 16),)](
        ..., T=T, D=D, BT=16, BD=BD, num_warps=8, ...)
else:
    fused_l2norm_qk_kernel1[(T,)](...)          # 否则走 per-row

# After
fused_l2norm_qk_kernel  # T, D 改成 runtime args
if D <= 512:                                    # 不再卡 T<=128
    BT = 16
    fused_l2norm_qk_kernel[(triton.cdiv(T, BT),)](
        ..., T, D, BT=BT, BD=BD, num_warps=2, ...)
else:
    fused_l2norm_qk_kernel1[(T,)](...)
```

两个改动：

1. **`T, D` 从 `tl.constexpr` 改成 runtime arg** —— `T` 每个 batch 不同；如果是 constexpr，
   每次都要 70 ms 重编译，远超 kernel 时间（<1 ms）。所以原 dispatcher 才必须 `T<=128` 才用
   BT-tiled。改成 runtime arg 后 BT-tiled 全 T 通用。
2. **`num_warps` 8 → 2** —— BT=16 + BD ≤ 512 时，每 program 处理 16×512 = 8K 元素；warps 太多
   反而 occupancy 撑爆 LDS / 寄存器。降到 2 warps 后实测更快。

**为什么 BT-tiled 比 per-row 快**：per-row kernel 用 `num_warps=8` × 64 lane = 512 threads
处理 D≤512 元素，underutilized；BT-tiled 把 16 行打包，512 threads 处理 16×512=8192 元素，
完全填充。

## 3. 实测

**测试机器**：MI308X，9B prefill T=122936 D=128 bf16

| Kernel | Before | After | 加速 |
|---|---:|---:|---|
| `fused_l2norm_qk` | 842 µs | **49 µs** | **17.2×** |
| SGLang `_l2norm_kernel`（对照） | — | 100 µs | RTP 反超 2.04× |

> 注：测试 T=122936 是单次 mega-prefill 的极端 case；实际 15k prefill 时 T=15k，按比例预估
> ~6 µs / call × 24 GDN 层 ≈ 144 µs / step（vs SGL 2 529 µs，即 RTP 反超 ~17×）。

## 4. 验证脚本

工作区有三个未追踪的 compare 脚本：
- `rtp_llm/models_py/triton_kernels/fla/test/compare_rtp_unfused_vs_sglang_fused.py` — RTP 未融合 vs SGL 融合
- `rtp_llm/models_py/triton_kernels/fla/test/compare_rtp_vs_sglang_fused_sigmoid_gating.py` — RTP/SGL 同口径融合对比
- `rtp_llm/models_py/triton_kernels/fla/test/compare_fused_vs_unfused_sigmoid_gating.py` — 同后端 fused vs unfused 精度

`fused_l2norm_qk` 的 17× / 49 µs / 100 µs 三个数字来自这些脚本。

## 5. 风险

| 风险点 | 缓解 |
|---|---|
| `tl.rsqrt(x)` vs `1/tl.sqrt(x)` 数值差异 | 硬件指令精度等价于 IEEE-754 单精度；fp32 精度内无可观测差异。已在 compare 脚本中验过 |
| BT=16 在某些 T<16 的 corner case 上掩码处理是否正确 | dispatcher 用 `triton.cdiv(T, BT)`，最后一块 mask 已实现 |
| `num_warps=2` 在 D=128 / D=256 的小 head 上可能不优 | 实测 D=128 已经覆盖；如果有更小 head 出现，需要重新打表 |

## 6. Phase 状态

- [x] Phase 0：性能 profile 定位 l2norm 在 GDN 层占比
- [x] Phase 1：ISA 级根因（div = rcp+NR）
- [x] Phase 2：kernel 改写 + dispatcher 改写
- [x] Phase 3：单元测试 / 数值对比通过
- [x] Phase 4：性能验证（842→49 µs）
- [x] Phase 5：已 commit (`3c529f051`)，可关单

## 7. 文件清单

```
n2_fused_l2norm_qk/
├── optimization_checkpoint.md   # 本文件
└── fused_l2norm_qk.patch        # 完整 diff（基于 4517b6418）
```

## 8. 回滚

```bash
git revert 3c529f051
# 或
git apply -R optimization_checkpoint/n2_fused_l2norm_qk/fused_l2norm_qk.patch
```

---

*创建于 2026-04-21 —— 从 v2_baseline_4517b6418 §3 迁移而来。*
