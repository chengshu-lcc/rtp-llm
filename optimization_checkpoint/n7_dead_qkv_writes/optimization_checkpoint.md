# N7 — 消除 prefill `add_fusedQKV_bias_transpose_prefill_kernel_v1` 的 dead writes

> **优化项 N7**，源自 per-layer 对比报告
> （`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` §2 N7）。
>
> **目标**：在 ROCm prefill `contextAttention` 路径上消除 `add_fusedQKV_bias_transpose_prefill_kernel_v1`
> 写到 `q_output / k_output / v_output` 的 dead bytes，以及前置 `bufMemset` 的 dead fills。
>
> **状态**：⚠️ **WIP** —— commit `6075b5b8b WIP: dead_qkv_writes v2 + DEBUG LOG` 已落库，但
> timeline 显示 patch 未生效（FillFunctor 数量不变、v1 kernel duration 不变），原因待 DEBUG LOG 二次定位。

---

## 0. 公共环境

参见父 checkpoint：`optimization_checkpoint/optimization_checkpoint.md` §0。本优化是 C++ 改动，
**需要重编**：

```bash
USE_BAZEL_VERSION=6.4.0 bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100
```

> 顶层 target `//rtp_llm:rtp_llm` 是聚合 target，会同时 build `librtp_compute_ops.so` /
> `libth_transformer.so` 等服务实际加载的 .so，并把 `librocm_impl` 的改动 link 进去。
> 单 build `//rtp_llm/cpp/devices/rocm_impl:rocm_impl` 只产生静态 lo，**不会更新最终 .so**。

## 1. 问题与现象

### 1.1 Trace 文件
- SGLang 对照：`/root/chengshu_code/rtp_profile/sgl-9b-15k-nocache.json`
- RTP 当前：`/root/chengshu_code/rtp_profile/rtp-9b-15k-0420.json`

### 1.2 关键发现

两边都跑同一个 ck_tile FMHA 内核，模板参数完全一致。差异在 pre-FMHA 的 wrapper 代码：

| 阶段 | RTP 现状 | SGLang | 备注 |
|---|---|---|---|
| Q+K RmsNorm | 331 µs | ~120 µs | RTP 用 `fusedQkRmsNorm`，SGL 用 aiter `add_rmsnorm_quant ×2` |
| RoPE | (在 v1 里) | mrope 223 µs | |
| Scatter to paged cache | (在 v1 里) | scatter 127 µs | |
| Gather to ragged | 不需要 | gather 102 µs | |
| **dead writes (store_q+store_kv)** | **~480 µs** | **0** | 见下方根因 |
| **dead fills (3× bufMemset on q/k/v_output)** | **~98 µs** | **0** | 配套问题 |
| FmhaFwd | 10.5 ms | 10.5 ms | 一致 |

### 1.3 根因（`ROCmAttentionOp.cc::contextAttention`，base commit `4517b6418`）

`runCKFmha` 分支（`skip_add_bias_transpose || max_prefix_prompt_length<=0`）从 `params.input`
直接读 Q/K/V：

```c++
// 第 896-901 行附近
fmha_runner_->runCKFmha(
    ... params.input.data(), ...
    ... params.input.dataWithOffset(hidden_units), ...
    ... params.input.dataWithOffset(hidden_units + hidden_units_kv), ...);
```

但 `add_fusedQKV_bias_transpose_prefill_kernel_v1` 的 store flags 是无条件 true：

```c++
// 第 724-727 行
bool store_qkv   = !use_mtp_pa_;        // ← runCKFmha 必须 (写回 params.input)
bool store_q     = true;                 // ← runCKFmha 路径上 dead
bool store_kv    = !use_mtp_pa_;         // ← runCKFmha 路径上 dead
bool store_cache = ...has_value();       // ← decode 必须
```

`q_output / k_output / v_output`（第 615-627 行）按 padded `[B,H,T,D]` / `[B,Hk,T+prefix,D]`
分配，本身被 alloc 时 **zero-init 一次**（trace 中那 3 个 `FillFunctor`，~98 µs），然后被 v1
kernel 全部覆写（~480 µs 写入），但 `runCKFmha` 完全不读。15k×16×128×2B ≈ 60 MB / layer 的 dead bytes。

### 1.4 Decode 路径（已确认干净，**不在本 patch 范围**）

`ROCmAttentionOp.cc:1172-1273`，`USE_ASM_PA=0` 走 `invokeAddFusedQKVBiasTransposeDecodeV1`：
- `store_qkv=false`、`store_q=true`、`store_kv=false`、`store_cache=true`
- `k_buf/v_buf` 直接传 `nullptr`（第 1235-1236 行）
- `q_output[B,H,D]` 在第 1272 行被 `runAiterPA(params, this, *q_output)` 真实读取
- 没有 dead writes、没有多余 fill

## 2. 修复方案

让 `q_output / k_output / v_output` 按需分配；让 `store_q / store_kv` 与 buffer 是否 alive 严格匹配。

```c++
const bool needs_padded_qkv =
    prefix_prompt_param.max_prefix_prompt_length > 0 && !use_mtp_pa_;

// q_output: mtp_pa 走 [token,H,D] 给 runTritonPA/runHipPA；prefix-prompt 走 [B,H,T,D] 给 gather
BufferPtr q_output, k_output, v_output;
if (use_mtp_pa_)            { q_output = allocateBuffer({...{token_num, head_num, size_per_head}...}); }
else if (needs_padded_qkv)  { q_output = allocateBuffer({...{batch_size, head_num, seq_len, size_per_head}...}); }
if (needs_padded_qkv) {
    k_output = allocateBuffer({...{batch_size, kv_head_num, seq_len_with_prefix, size_per_head}...});
    v_output = allocateBuffer({...{batch_size, kv_head_num, seq_len_with_prefix, size_per_head}...});
}

bool store_q     = use_mtp_pa_ || needs_padded_qkv;
bool store_kv    = needs_padded_qkv;
void* q_buf_data = q_output ? q_output->data() : nullptr;
void* k_buf_data = k_output ? k_output->data() : nullptr;
void* v_buf_data = v_output ? v_output->data() : nullptr;
```

3 个 `invokeAddFusedQKVBiasTranspose*` 的调用点用 `q_buf_data/k_buf_data/v_buf_data` 替换原来
的 `q_output->data()/k_output->data()/v_output->data()`。kernel 内部由 `if (store_q)` /
`if (store_kv)` gate，store flags 已对齐为 false 时 nullptr 不会被解引用。

`printBufferData(*q_output, ...)` 等行加 `if (q_output)` 守卫，避免 trace 模式下 nullptr deref。

完整 diff 见本目录 `dead_qkv_writes_v3.patch`，且 patch 还包含 DEBUG LOG（写
`/tmp/dead_qkv_debug.log`）。

## 3. 实测（来自 commit `6075b5b8b` 后的验证）

**测试场景**：Qwen3.5-9B TP=2，prefill 15k tokens，`USE_ASM_PA=0`，`ENABLE_CUDA_GRAPH=1`，单 sample。

**Profiling 工具**：父 checkpoint §0.3 perf_test。timeline 在 `/root/wenhua_code/rtp_profiling/full-attn/`：

| 文件 | 含义 | 大小 |
|---|---|---|
| `rtp-9b-15k-0420.json` | RTP baseline (无 patch) | 5.7 MB |
| `rtp-9b-15k-0421-opi.json` | RTP after (含 v2 patch，**未含 DEBUG LOG**) | 5.7 MB |
| `sgl-9b-15k-nocache.json` | SGLang baseline | 277 MB |

**端到端** benchmark.py（concurrency=1）：

| 指标 | Baseline | After | Δ |
|---|---:|---:|---:|
| prefill_time | 2575.67 ms | 2565.22 ms | −10.45 ms (−0.41%，噪声) |
| decode_time/token | 6.025 ms | 6.055 ms | +0.030 ms (+0.50%，噪声) |

**Per-kernel** 对比：

| Kernel | Baseline | After | Δ |
|---|---|---|---|
| `add_fusedQKV_bias_transpose_prefill_kernel_v1` | n=8, p50=**968 µs** | n=8, p50=**986 µs** | **+18 µs（噪声）** |
| `FillFunctor<c10::BFloat16>` (zero-init) | **n=52** | **n=52** | **0** |

**结论**：FillFunctor 数量完全一致 → `q_output / k_output / v_output` 还在被分配 →
**`needs_padded_qkv` 在每次 prefill call 都是 `true`**（与"nocache 时应为 false"的假设矛盾）。
**patch 等于不生效。**

## 4. 已排除的可能性

- ✅ patch 编进 .so：`librtp_compute_ops.so` mtime newer than `.cc` mtime
- ✅ service 加载正确 .so：`from librtp_compute_ops import *` 直接 dlopen
  `bazel-bin/librtp_compute_ops.so`，`nm` 确认 `T _ZN7rtp_llm10ROCmDevice16contextAttention...`
  在该 .so 里
- ✅ v1 kernel 内有 `if (store_q) / if (store_kv)` gate（`unfused_attention_kernels.cu` line
  2389/2409/2588/2609 等）—— 不是 kernel 实现的问题

## 5. 待验证假设（接着干这里入手）

**假设**：在当前启动配置下（`USE_ASM_PA=0` + `LOAD_PYTHON_MODEL=1` + `ENABLE_CUDA_GRAPH=1`
+ nocache 输入），`prefix_prompt_param.max_prefix_prompt_length` 在每次 prefill 调用都 > 0，
导致 `needs_padded_qkv=true`，patch 等于不生效。

**验证方法**：v3 patch 已埋 DEBUG LOG（写 `/tmp/dead_qkv_debug.log`）。在新机器：

```bash
cd /root/rtp-llm
git apply --check optimization_checkpoint/n7_dead_qkv_writes/dead_qkv_writes_v3.patch
# (如果 6075b5b8b 已合入 HEAD，会报 "already applied"；那就直接 build)

USE_BAZEL_VERSION=6.4.0 bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100
rm -f /tmp/dead_qkv_debug.log
# 跑父 checkpoint §0.3 的 perf_test 命令（partial=2，prefill only）
head -10 /tmp/dead_qkv_debug.log
```

**预期输出**：

```
[DEAD_QKV_DEBUG] needs_padded_qkv=? max_prefix_prompt_length=? use_mtp_pa_=? prefix_prompt_lengths_ptr=0x... batch_size=1 seq_len=15000 token_num=15000
```

**判读**：

- `needs_padded_qkv=0`（false）但 timeline 里 v1 kernel 没变 → kernel 内 if-gate 不省时
  （仅省写入字节，runtime 仍跑 RoPE/Q 计算），patch 思路要换；可能要走 P1 拆
  `add_rmsnorm_quant ×2` + 独立 mrope + scatter（参考 SGLang `sglang/srt/layers/attention/`）。
- `needs_padded_qkv=1`（true）+ `max_prefix_prompt_length>0` → 追溯 `GptModel.cc:182` 加同款
  LOG 打 `attention_inputs.max_prefix_length` 与 `prefix_lengths` 内容；怀疑 perf_test
  warmup 后命中 reuse cache，导致 active 这次 prefill 进来就有 prefix。如果是这个原因，patch
  假设要变成"`prefill_lengths` 整段全 0 时跳过"，而不是"`max_prefix_prompt_length=0` 时跳过"。

## 6. 风险 / 已审计依赖

| 风险点 | 处理 |
|---|---|
| `q_output` 在 mtp_pa 路径被 `runTritonPA / runHipPA` 读 (line 858/860) | ✅ patch 保留 mtp_pa 路径的 alloc + store_q=true |
| `q_output / k_output / v_output` 在 prefix-prompt 路径被 `invokeLoadPrefixKVCache*` 写、被 `invokeGatherSequencesCombined` 读 | ✅ `needs_padded_qkv` 与 line 687 的 prefix gate 完全一致 |
| `runCKFmhaV2` (prefix path else 分支) 内部 `q_output->updateShape(...)` + `gemm({*q_output, ...})` | ✅ 此 else 分支只在 `prefix_prompt > 0` 时进入，q_output 保证 alive |
| `writeCacheStore` 是否依赖 k_output/v_output | ✅ 不依赖；它读 `kv_cache_buffer`（paged cache） |
| v1 kernel 在 `store_q=false` 时是否会 deref `q_buf` | ✅ 已读 `unfused_attention_kernels.cu:3697` 的 `if (store_q)` gate |
| `printBufferData` 在 trace 模式下解引用 nullptr | ✅ 已加 `if (q_output)` 等守卫 |
| `selfAttentionwrapper`（!use_aiter_pa decode 路径）依赖 | ✅ 不在 contextAttention 范围 |

## 7. Phase 状态

- [x] Phase 0：checkpoint + 环境
- [x] Phase 1：根因定位
- [x] Phase 2：方案设计
- [x] Phase 3：apply + build（已 commit `6075b5b8b`）
- [x] Phase 4：正确性（perf_test 通过，无 NaN）
- [x] Phase 5：性能验证 — **❌ patch 未生效**
- [ ] Phase 6：DEBUG LOG 验证 + 决策（**接着这里干**）

## 8. 文件清单

```
n7_dead_qkv_writes/
├── optimization_checkpoint.md      # 本文件
└── dead_qkv_writes_v3.patch        # 完整 diff（基于 4517b6418，含 DEBUG LOG）
```

## 9. 回滚

```bash
git revert 6075b5b8b
# 或
git apply -R optimization_checkpoint/n7_dead_qkv_writes/dead_qkv_writes_v3.patch
```

---

*创建于 2026-04-21 —— 从老 top-level checkpoint + v2_baseline_4517b6418 §2 迁移而来。*
