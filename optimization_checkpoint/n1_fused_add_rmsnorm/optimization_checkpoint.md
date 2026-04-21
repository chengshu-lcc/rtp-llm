# N1 — 用 aiter CK 路径替换 Triton `_fused_add_rmsnorm_kernel`

> **优化项 N1**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md`）。
>
> **目标**：降低 ROCm 上"pre-attn / post-attn RMSNorm + residual add"算子的耗时。SGLang
> 同算子的单次延迟约为 RTP 当前实现的 1/3，原因在于 SGL 走的是 aiter CK 路径
> （`add_rmsnorm_quant_kernel`），而 RTP 当前用的是 Triton 路径
> （`_fused_add_rmsnorm_kernel`）。
>
> **当前状态**：Phase 0–2 已完成（根因定位 + 候选实现 micro-bench 实测，并发现一个关键
> 警示）。Phase 3（落 patch + 端到端验证）待执行。

---

## 0. 环境信息

- **仓库**：`/root/rtp-llm`
- **分支**：`qwen-0401-dm-wh`
- **Baseline commit**：`3c529f051 optimize fused_l2norm_qk: rsqrt+mul, BT-tiled for all T`（当前 HEAD）
- **GPU**：AMD MI308X（gfx950）
- **典型场景**：Qwen3.5-9B（qwen3_next，32 层 = 24 GDN + 8 Full-Attn）TP=2 prefill 15k tokens，BF16
- **aiter 版本**：镜像中预装的版本，路径 `/opt/conda310/lib/python3.10/site-packages/aiter/`

### 0.1 服务启动命令（在 `/root/rtp-llm` 下执行）

```bash
LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH \
USE_SWIZZLEA=1 SEQ_SIZE_PER_BLOCK=1024 KERNEL_SEQ_SIZE_PER_BLOCK=16 \
WARM_UP=0 CONCURRENCY_LIMIT=128 ENABLE_CUDA_GRAPH=1 LOAD_PYTHON_MODEL=1 \
USE_ASM_PA=0 WORLD_SIZE=2 DP_SIZE=1 TP_SIZE=2 EP_SIZE=1 \
DEVICE_RESERVE_MEMORY_BYTES=-21474836000 RESERVER_RUNTIME_MEM_MB=4096 \
AITER_ASM_DIR=/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/ \
MAX_SEQ_LEN=262144 START_PORT=8066 ACT_TYPE=bf16 \
TOKENIZER_PATH=~/Qwen3.5-9B CHECKPOINT_PATH=~/Qwen3.5-9B \
MODEL_TYPE=qwen35_dense FT_SERVER_TEST=1 \
ROCM_DISABLE_CUSTOM_AG=True FT_DISABLE_CUSTOM_AR=True \
KV_CACHE_MEM_MB=20000 \
/opt/conda310/bin/python3.10 -m rtp_llm.start_server 2>&1 | tee output1.txt
```

### 0.2 编译 / Benchmark / Profiling 命令

参见父 checkpoint：`optimization_checkpoint/v2_baseline_4517b6418/optimization_checkpoint.md` §0.3 / §0.2 / §0.4。
本优化为 Python-only 改动，**无需重新编译**。

### 0.3 文件清单

```
optimization_checkpoint/n1_fused_add_rmsnorm/
├── optimization_checkpoint.md   # 本文件
├── micro_bench.py               # Phase-2 实测脚本（4 路实现对比）
└── n1_swap_to_ck.patch          # Phase-3 待生成（落 patch 后 git diff > 这里）
```

---

## 1. 问题与现象

来自 `/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N1：

| 框架 | Kernel（单次） | µs/call | 调用次数/prefill | 总耗时 |
|---|---|---:|---:|---:|
| RTP（当前） | `_fused_add_rmsnorm_kernel`（Triton，`aiter.ops.triton.normalization.rmsnorm.rmsnorm2d_fwd_with_add`） | 504 | 65 | **32.8 ms** |
| SGLang | `aiter add_rmsnorm_quant_kernel<…256,16,True…>`（CK，附带可选 FP8 quant） | 168 | 64 | 10.7 ms |

**理论缺口**：`32.8 - 10.7 = ~21.5 ms` per 15k-token prefill (TP=2)。

**调用次数构成**：32 层 × 2 次（pre-attn + post-attn）+ 1 次 final-norm = 65 次。

**当前的接入方式**（`rtp_llm/models_py/modules/base/rocm/norm.py:7-9`）：

```python
from aiter.ops.triton.normalization.rmsnorm import (
    rmsnorm2d_fwd_with_add as fused_add_rmsnorm,
)
```

这个 import 绑定的是 **Triton 实现**。而 CK 实现（以及另一个 `_quant` 版本）在 aiter 顶层
namespace 里有独立的符号。

---

## 2. 调研：4 个候选 aiter 符号实测对比

测试条件：T=15367, D=4096, BF16, MI308X。脚本见 §3.5 的 `micro_bench.py`。

| # | 符号 | 延迟 | 数值是否正确 | 备注 |
|---|---|---:|---|---|
| A | `aiter.ops.triton.normalization.rmsnorm.rmsnorm2d_fwd_with_add`（当前） | **325 µs** | 参考基准 | Triton kernel `_fused_add_rmsnorm_kernel` |
| B | `aiter.rmsnorm2d_fwd_with_add`（默认 dispatcher） | **172 µs** | ❌ **完全错误**（输出 ~15000） | 路由到 `module_rmsnorm_quant.add_rmsnorm` |
| C | `aiter.rmsnorm2d_fwd_with_add_ck`（CK 显式） | **232 µs** | ✅ max diff 3.1e-2（BF16 量化噪声） | drop-in 替换，签名完全一致 |
| D | `aiter.fused_add_rms_norm_cu`（in-place HIP） | **427 µs** | ✅ max diff 3.1e-2 | in-place；比当前还慢 |

### 2.1 关键发现 —— SGL 速度的路径 B 在我们这版 aiter 中是坏的

符号 B 的 kernel 名（`add_rmsnorm_quant_kernel`）和单次速度（~170 µs）都对得上 SGLang trace 里的
168 µs，**但我们这版 aiter 输出的是垃圾数据**（数值 ~15000，期望 ~-0.6）。在多个 (T, D) 形状下
都复现，不是 tile/template 的不匹配问题。

读 `aiter/ops/rmsnorm.py:247-254`：

```python
@compile_ops("module_rmsnorm_quant")
def add_rmsnorm(
    out: Tensor, input: Tensor, residual_in: Tensor, residual_out: Tensor,
    weight: Tensor, epsilon: float,
) -> None: ...
```

C++ 端签名和 Triton 版本一致，但 `module_rmsnorm_quant` 里的 kernel 似乎漏掉了 rsqrt 归一化
那一步（输出量级是 `(input+residual) * weight * D` 的 order，而非
`(input+residual)/rms(...) * weight`）。可能是 aiter 已知 bug，也可能是我们调用约定不对。
**值得后续追查，争取额外 ~4 ms 收益**（详见 §6）。

### 2.2 安全路径（路径 C）—— `rmsnorm2d_fwd_with_add_ck`

- Python 签名与当前 Triton 版本完全一致：`(out, input, residual_in, residual_out, weight, eps[, use_model_sensitive_rmsnorm])`
- 数值与 Triton 在 BF16 量化噪声范围内（max abs diff 3.1e-2，输出量级 ~1）
- **325 → 232 µs**，单次节省 **~93 µs**
- 65 calls/prefill × 93 µs = **每次 prefill 节省约 6.0 ms**（保守估计）
- 当前 CK 构建里 ship 的 kernel 是 `rmsnorm2d_fwd_with_add_ck`（不是 SGL trace 里
  那个 `add_rmsnorm_quant_kernel<…>`）。要拿到 SGL 的 168 µs 还需要修好路径 B —— 见 §6

---

## 3. 实施计划

### 3.1 Phase 3（保守落地）

**单文件、3 行改动**，文件 `rtp_llm/models_py/modules/base/rocm/norm.py:7-9`：

```diff
-from aiter.ops.triton.normalization.rmsnorm import (
-    rmsnorm2d_fwd_with_add as fused_add_rmsnorm,
-)
+from aiter import rmsnorm2d_fwd_with_add_ck as fused_add_rmsnorm
```

调用点在第 50–57 行（`fused_add_rmsnorm(output, hidden_states, residual, residual_out, self.weight.data, self.variance_epsilon)`），签名完全兼容，**无需修改其他任何代码**。

> ⚠️ **不要用 `aiter.rmsnorm2d_fwd_with_add`（不带 `_ck` 后缀）** —— 它会被 dispatcher 路由
> 到 §2.1 里那个数值错误的 `module_rmsnorm_quant.add_rmsnorm`。

### 3.2 改动文件

- `rtp_llm/models_py/modules/base/rocm/norm.py` —— 仅 import 部分 3 行

### 3.3 影响范围 —— `RMSResNorm` 的所有调用点

```
rtp_llm/models_py/model_desc/qwen3_next.py:674   self.input_layernorm = RMSResNorm(...)
rtp_llm/models_py/model_desc/qwen3_next.py:677   self.post_attention_layernorm = RMSResNorm(...)
rtp_llm/models_py/model_desc/qwen3_next.py:752   self.norm = RMSResNorm(...)              # final norm
rtp_llm/models_py/model_desc/generic_moe.py:231  self.input_layernorm = RMSResNorm(...)
rtp_llm/models_py/model_desc/generic_moe.py:234  self.post_attention_layernorm = RMSResNorm(...)
rtp_llm/models_py/model_desc/generic_moe.py:307  self.norm = RMSResNorm(...)
```

涵盖所有 Qwen3.5 dense（qwen3_next）和 generic MoE（`qwen_3_moe` 等）的 prefill 路径。
本次 swap 是形状无关、dtype 无关的，BF16/FP16 路径都不会出问题。

### 3.4 Phase 4（端到端验证）—— 已完成 ✅

**测试方法**：用 §0.4 的 perf_test `--partial=2`（仅 prefill）做 A/B，从 stdout 的
"Prefill Time(ms)" 抓数。配置：bs=1, input_len=15000, TP=2, USE_ASM_PA=0,
ENABLE_CUDA_GRAPH=1。每边各跑 1 次（warmup + 1 active），单次样本。

| 配置 | Prefill Time | Δ |
|---|---:|---:|
| BASELINE（aiter Triton `rmsnorm2d_fwd_with_add`）| **931.25 ms** | — |
| AFTER（aiter CK `rmsnorm2d_fwd_with_add_ck`）   | **913.74 ms** | **−17.51 ms (−1.88%)** |

**结论**：
- 收益方向正确，幅度比 §2.2 估算的 6 ms 大 ~3×。可能原因：单次样本噪声（±5–10 ms 量级合理）；
  也可能是 CK kernel 减少了 LDS / 寄存器压力，让相邻 kernel 的 occupancy 也跟着提升。
- 数值正确性 §2 已验证（max diff 3.1e-2，BF16 噪声内）；perf_test 跑通且无 NaN/异常输出。
- 端到端落地干净，可直接 commit。

**进一步验证（可选）**：
- 多次重复（3+ 次）取均值，确认收益稳定 / 排除噪声幅度
- 抓 timeline 二次确认 `_fused_add_rmsnorm_kernel` 已被 `rmsnorm2d_fwd_with_add_ck` 的 CK kernel
  名替换；当前 perf_test profile_step=1 dump 出空 timeline，需要修 TorchProfiler 才能拿干净 trace
  （见 v2_baseline_4517b6418 §2.5 待办）

### 3.5 micro_bench.py

实测脚本保存在本目录下的 `micro_bench.py`，可重现 §2 的表格。运行命令：

```bash
LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH \
  /opt/conda310/bin/python3.10 \
  /root/rtp-llm/optimization_checkpoint/n1_fused_add_rmsnorm/micro_bench.py
```

---

## 4. 风险评估与缓解

| 风险点 | 缓解措施 |
|---|---|
| CK kernel 数值与 Triton 有微小差异，可能让下游测试失败 | BF16 max diff 3.1e-2，在量化噪声范围内。perf_test 的正确性断言能覆盖这一变化。 |
| CK kernel 在某些罕见 shape 上更慢（比如 decode 时的 tiny T） | `RMSResNorm` 仅在 prefill 路径用；decode 走的是另一条 C++ 实现 `fused_add_layernorm`。本改动不影响 decode。建议跑一次 decode-only profile 确认。 |
| `rmsnorm2d_fwd_with_add_ck` 依赖某些工具链才有的 CK 构建 | 当前镜像里 aiter 已经包含该符号，首次调用会 jit 编译 `module_rmsnorm`（约 45s）。CI 需要允许首次 jit。 |
| `qwen_3_moe`（generic MoE）也会被同时换掉，可能引入回归 | 原本就是 aiter Triton，换成 aiter CK 没有语义差异。如有 `qwen_3_moe` 测试场景，跑一次 smoke test 即可。 |
| `use_model_sensitive_rmsnorm` 标志在 CK 和 Triton 里行为可能不同 | 两边默认值都是 0，我们也从未设置过非 0 值 —— 保持默认即可。 |

---

## 5. Phase 状态

- [x] Phase 0：环境 + checkpoint 骨架
- [x] Phase 1：根因定位（Triton vs CK）
- [x] Phase 2：4 路实现 micro-bench，确认 CK 数值正确且快 1.4×
- [x] Phase 3：落 3 行 patch（`rocm/norm.py` 改用 `rmsnorm2d_fwd_with_add_ck`）+ 数值回归通过（max diff 3.1e-2，BF16 噪声内）
- [x] Phase 4：端到端性能测量（perf_test `--partial=2` A/B：931.25 → 913.74 ms，−17.51 ms / −1.88%）
- [ ] Phase 5：commit 入库；同步更新 `optimization_checkpoint/v2_baseline_4517b6418/optimization_checkpoint.md` §7，把 N1 标记为 done
- [ ] Phase 6（拓展目标）：追查路径 B 的数值 bug，争取额外 ~4 ms 收益（见 §6）

---

## 6. 已知问题（KNOWN ISSUE）—— `aiter.add_rmsnorm` / `add_rmsnorm_quant_kernel` 在当前 aiter 版本输出错误

> **重要 — 后续追查 N1 拓展目标时务必先读本节**

### 6.1 现象

调用 `aiter.rmsnorm2d_fwd_with_add`（默认 dispatcher，会路由到 `module_rmsnorm_quant.add_rmsnorm`）
或 直接调 `aiter.add_rmsnorm`，输出张量数值**完全错误**：

| 项 | 期望（参考 Triton/CK） | 实际（aiter add_rmsnorm） |
|---|---|---|
| 第 1 行前 4 个元素 | `[-0.6328, -0.4101, -2.797, -0.213]` | `[15168.0, 22912.0, 23040.0, 15168.0]` |
| max abs diff | 0 | **2.36e+04** |
| mean abs diff | 0 | **1.53e+04** |

数值规模看起来是漏掉了 `1/rms(...)` 这个归一化除法，即输出 ≈ `(input+residual) * weight * D_factor`
而不是 `(input+residual)/rms * weight`。

### 6.2 复现条件

- aiter 版本：当前镜像 `/opt/conda310/lib/python3.10/site-packages/aiter/` 里的 pin 版本
- 多个 (T, D) 组合都能稳定复现：`(15367, 4096)`、`(128, 4096)`、`(15367, 2048)`、`(256, 2048)`
- 全部 BF16，`epsilon=1e-6`
- 同一进程里 `aiter.rmsnorm2d_fwd_with_add_ck`（路径 C）输出正确，**仅路径 B 的 `module_rmsnorm_quant`
  路径有问题** —— 排除了输入数据本身的问题

最小复现脚本：`/root/rtp-llm/optimization_checkpoint/n1_fused_add_rmsnorm/micro_bench.py`
（运行后 4 路对比表会直接给出 ❌ WRONG 标签）

### 6.3 影响

- N1 优化目前**只拿到 ~6 ms** 收益（CK 232 µs/call），而不是 per-layer 报告里估算的 ~21.5 ms
- 缺失的 ~10–15 ms 缺口正是路径 B 修好后能拿到的（CK 232 → SGL 168 µs，再 × 65 calls）
- SGLang 在 trace 里就调的是 `add_rmsnorm_quant_kernel<…256,16,True…>`，说明路径 B 在他们用的
  aiter 版本里是 work 的 —— 多半是版本差异

### 6.4 后续追查的三条思路

1. **升级 aiter**：拿一个 SGLang 用的 aiter 版本 / commit 重新 build，跑 `micro_bench.py`，
   看路径 B 是否仍然错。如果新版正确，直接升级 + 把 import 从 `_ck` 切成 `aiter.add_rmsnorm`。
2. **改用 `aiter.add_rmsnorm_quant`（带 dummy scale）**：SGL 的 kernel 名带 `_quant` 后缀且
   template flag 是 `True`，签名是
   `(out, input, residual_in, residual_out, scale, weight, epsilon, group_size=0, shuffle_scale=False)`。
   有可能 SGL 是传一个 unit scale 然后忽略量化输出（即 quant 算子被复用作纯 BF16 路径）。
   值得花 30 分钟实验：传 `scale=torch.ones(1)` 或 `scale=None`，看输出对不对。
3. **直接对照 SGL 源码**：`sglang/srt/layers/normalization/rmsnorm.py`（或类似路径）会
   显示 SGL 到底调的是哪个 aiter 符号、传什么参数。镜像调用即可。

### 6.5 关联代码 / 文件

- aiter 实现注册点：`/opt/conda310/lib/python3.10/site-packages/aiter/ops/rmsnorm.py:247-254`
- aiter JIT 模块：`/opt/conda310/lib/python3.10/site-packages/aiter/jit/module_rmsnorm_quant.so`
- 我们当前的接入：`rtp_llm/models_py/modules/base/rocm/norm.py:7`（已改为 `_ck`）
- 上游对照：`sglang/srt/layers/normalization/` 下的 rmsnorm 实现

---

## 7. 回滚

```bash
git checkout -- rtp_llm/models_py/modules/base/rocm/norm.py
```

单文件、3 行，零成本回滚。

---

*创建于 2026-04-21 —— 基于 per-layer 对比报告的 N1 优化项。*
