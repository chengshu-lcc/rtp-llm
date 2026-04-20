# RTP-LLM ROCm Prefill Attention Pre-FMHA Dead-Write Elimination

> 实验目标：在 ROCm 后端的 contextAttention 路径上消除 `add_fusedQKV_bias_transpose_prefill_kernel_v1`
> 写到 `q_output`/`k_output`/`v_output` 的 dead bytes，以及前置 `bufMemset` 的 dead fills。
> 影响范围：**仅 prefill；decode 路径已经是干净的，不动**。

---

## 0. 环境信息

- **本地仓库**: `/Users/hxy/Desktop/hxy/rtp-llm`
- **服务器仓库**: `/root/rtp-llm`
- **当前分支**: `qwen-0401-dm-wh`
- **Patch 基准 commit**: `a0b8191c8 add fused_l2_norm`
- **HEAD（baseline 用）**: `0c4c39879 opti`（仅入库 checkpoint 文件，未改 cc，等价于 a0b8191c8）
- **目标 GPU**: AMD MI355X (gfx950)
- **典型场景**: Qwen3.5-9B TP2，prefill 15k tokens，`USE_ASM_PA=1`（prefill）+ `USE_ASM_PA=0`（decode）
- **构建模式**: bazel build (Python 端无需重编)
- **patch 文件**: `optimization_checkpoint/dead_qkv_writes.patch`

### 0.1 服务启动命令（在 `/root/rtp-llm` 下执行）

```bash
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

> 注意：`USE_ASM_PA=0` → decode 走 aiter PA（非 asm），prefill 仍是 ck_tile FMHA。

### 0.2 Benchmark 命令（在 `/root/wenhua_code/rtp-test` 下执行）

```bash
START_PORT=8066 TOKENIZER_PATH=~/Qwen3.5-9B/ CHECKPOINT_PATH=~/Qwen3.5-9B/ \
DP_SIZE=1 python benchmark.py
```

`benchmark.py` 默认配置：`test_lengths=[15000]`、`concurrencies=[1]`、`output_length=500`。
单次运行输出：avg/max prefill_time、avg/max decode_time_per_token、avg/max total_time。

### 0.3 编译命令（在 `/root/rtp-llm` 下执行）

```bash
bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100
```

> Target `//rtp_llm:rtp_llm` 是顶层聚合 target，会同时 build `librtp_compute_ops.so` / `libth_transformer.so` 等服务实际加载的 .so，并把 `librocm_impl` 的改动 link 进去。
> 单 build `//rtp_llm/cpp/devices/rocm_impl:rocm_impl` 只产生静态 lo，不会更新最终 .so，**改动不会生效**。

### 0.4 Profiling 命令（perf_test，自动管理 server 生命周期）

`benchmark.py + gen_timeline=True` 会让 server 卡死。改用 perf_test：MagaServerManager 自动 spawn + stop server，即使 timeline 触发卡死框架会强制 kill，timeline 文件依然生成。**不需要 cherry-pick patch**（cherry-pick 那个只加 shape 信息；perf_test 默认走 server 端 `gen_timeline` 路径，timeline 默认就含算子）。

```bash
bazelisk test //rtp_llm/test/perf_test:perf_test --config=rocm \
  --test_arg=--ckpt_path=/root/Qwen3.5-9B \
  --test_arg=--tokenizer_path=/root/Qwen3.5-9B \
  --test_arg=--model_type=qwen35_dense \
  --test_arg=--tp_size=2 --test_arg=--dp_size=1 \
  --test_arg=--batch_size="1" --test_arg=--input_len="15000" \
  --test_arg=--partial=2 \
  --test_env=USE_SWIZZLEA=1 \
  --test_env=SEQ_SIZE_PER_BLOCK=1024 --test_env=KERNEL_SEQ_SIZE_PER_BLOCK=16 \
  --test_env=USE_ASM_PA=0 --test_env=LOAD_PYTHON_MODEL=1 \
  --test_env=ENABLE_CUDA_GRAPH=1 --test_env=WARM_UP=0 --test_env=ACT_TYPE=bf16 \
  --test_env=ROCM_DISABLE_CUSTOM_AG=True --test_env=FT_DISABLE_CUSTOM_AR=True \
  --test_env=KV_CACHE_MEM_MB=20000 \
  --test_env=DEVICE_RESERVE_MEMORY_BYTES=-21474836000 \
  --test_env=RESERVER_RUNTIME_MEM_MB=4096 \
  --test_env=AITER_ASM_DIR=/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/ \
  --test_env=MAX_SEQ_LEN=262144 \
  --test_output=streamed --nocache_test_results --sandbox_debug
```

`--partial=2` = 仅跑 Prefill（`1` = 仅 Decode）。timeline 输出 `profiler_wr*.json`，sandbox 退出前会挂在 sandbox tmp 路径下，需要根据 stdout 里 `gen timeline file:` 的提示拉到 `/root/rtp-llm/`。

---

## 1. 问题与证据（来自本地 trace 分析）

### Trace 文件
- SGLang 对照：`/Users/hxy/Desktop/hxy/wenhua_code/sglang_profiling/sgl-9b-tp2-15k-hippa-qr/sgl-9b-15k-nocache.json`
- RTP 当前：`/Users/hxy/Desktop/hxy/wenhua_code/rtp_profiling/rtp-9b-15k-0420.json`

### 关键发现

两边都跑同一个 ck_tile FMHA 内核：
```
ck_tile::FmhaFwdKernel<BlockFmhaPipelineQRKSVS<..., sequence<128,128,32,256,32,256>, sequence<4,1,1>, ...>>
```
模板参数完全一致，FMHA 自身无差异。差异在 pre-FMHA 的 wrapper 代码。

#### Per-layer 时序对比（prefill 1 layer）

| 阶段 | RTP 现状 | SGLang | 备注 |
|---|---|---|---|
| Q+K RmsNorm | 331 µs | ~120 µs | RTP 用 `fusedQkRmsNorm`，SGLang 用 aiter `add_rmsnorm_quant` ×2 |
| RoPE | (在 v1 里) | mrope 223 µs | |
| Scatter to paged cache | (在 v1 里) | scatter 127 µs | |
| Gather to ragged | 不需要 | gather 102 µs | |
| **dead writes (store_q+store_kv)** | **~480 µs** | **0** | 见下方根因 |
| **dead fills (3× bufMemset on q/k/v_output)** | **~98 µs** | **0** | 配套问题 |
| FmhaFwd | 10.5 ms | 10.5 ms | 一致 |

### 根因（`ROCmAttentionOp.cc::contextAttention`，base commit a0b8191c8）

`runCKFmha` 分支（`skip_add_bias_transpose || max_prefix_prompt_length<=0`）从 `params.input` 直接读 Q/K/V：
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

`q_output`/`k_output`/`v_output`（第 615-627 行）按 padded `[B,H,T,D]` / `[B,Hk,T+prefix,D]` 分配，
本身被 alloc 时 **zero-init 一次**（trace 中那 3 个 `FillFunctor`，~98 µs），然后被 v1 kernel 全部覆写
（~480 µs 写入），但 `runCKFmha` 完全不读。15k×16×128×2B ≈ 60 MB / layer 的 dead bytes。

### Decode 路径（已确认干净，**不在本 patch 范围内**）

`ROCmAttentionOp.cc:1172-1273`，`USE_ASM_PA=0` 走 `invokeAddFusedQKVBiasTransposeDecodeV1`：
- `store_qkv=false`、`store_q=true`、`store_kv=false`、`store_cache=true`
- `k_buf/v_buf` 直接传 `nullptr`（第 1235-1236 行）
- `q_output[B,H,D]` 在第 1272 行被 `runAiterPA(params, this, *q_output)` 真实读取
- 没有 dead writes，没有多余 fill

---

## 2. 修复方案

让 `q_output`/`k_output`/`v_output` 按需分配；让 `store_q`/`store_kv` 与 buffer 是否 alive 严格匹配。

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

3 个 `invokeAddFusedQKVBiasTranspose*` 的调用点用 `q_buf_data/k_buf_data/v_buf_data` 替换原来的
`q_output->data()/k_output->data()/v_output->data()`。kernel 内部由 `if (store_q)` / `if (store_kv)`
gate，store flags 已对齐为 false 时 nullptr 不会被解引用。

`printBufferData(*q_output, ...)` 等行加 `if (q_output)` 守卫，避免 trace 模式下 nullptr deref。

完整 diff 见 `dead_qkv_writes.patch`。

---

## 3. 应用步骤（在服务器上）

```bash
cd <RTP_REPO_ROOT>
git fetch
git checkout qwen-0401-dm-wh   # 或对应分支
# 确认基准 commit
git log --oneline -3   # 期望最上面是 a0b8191c8 add fused_l2_norm
                       # 如果分支已经前进，先评估冲突再 apply

git apply --check optimization_checkpoint/dead_qkv_writes.patch   # 干跑校验
git apply optimization_checkpoint/dead_qkv_writes.patch
```

构建：
```bash
bazel build //rtp_llm/cpp/devices/rocm_impl:rocm_impl 2>&1 | tee build.log
# 或者更直接的 target，按你日常的构建方式
```

---

## 4. 验证

### 4.1 单测 / 编译

- `bazel build` 必须通过，无 warning 增加
- 现有 attention 单测全过：
  ```bash
  bazel test //rtp_llm/cpp/devices/rocm_impl:tests 2>&1 | tail -30
  ```

### 4.2 端到端正确性

Qwen3.5-9B 推理输出应**完全一致**（patch 不改变数学逻辑，只是不写没人读的 buffer）：
```bash
# 用 baseline (a0b8191c8) 跑一组 fixed-seed 推理，记录输出
python -m rtp_llm.start_server --model_type=qwen3_next --checkpoint_path=...

# 用 patch 后版本跑同一组，diff 输出，期望 0 byte 差异
```

### 4.3 性能验证（关键）

**A/B 测试，一次只改 patch 一个变量**。
重跑用户提供的 trace 命令（与 `rtp-9b-15k-0420.json` 同样的 batch / seq / TP 配置）：

```bash
# 假设你之前的 profiling 命令保存在某 script 里，复用即可，不要拼新参数
bash <你之前的 profile script> --output rtp-9b-15k-after-patch.json
```

期望在 trace 里看到：
1. **`add_fusedQKV_bias_transpose_prefill_kernel_v1` 的 duration 下降** ~960 µs → ~480 µs
2. **3 个 `FillFunctor` (q_output/k_output/v_output 的 zero-init) 消失**
3. **每层省 ~580 µs；45 GDN/full-attn 层 → 全 prefill 省 ~25 ms**
4. FMHA kernel 时间不变（10.5 ms 一致）

可以用之前的 timeline 解析脚本对比：
```python
# 对比 patch 前后 fusedQKV kernel 的中位 duration
import json, statistics
for label,p in [('before','rtp-9b-15k-0420.json'), ('after','rtp-9b-15k-after-patch.json')]:
    d = json.load(open(p))
    events = d['traceEvents'] if isinstance(d, dict) else d
    durs = [e['dur'] for e in events if 'dur' in e
            and 'add_fusedQKV_bias_transpose_prefill_kernel_v1' in e.get('name','')]
    print(label, 'count=', len(durs), 'p50=', statistics.median(durs))
```

---

## 5. 风险 / 已审计的依赖

| 风险点 | 处理 |
|---|---|
| `q_output` 在 mtp_pa 路径被 `runTritonPA/runHipPA` 读 (line 858/860) | ✅ patch 保留 mtp_pa 路径的 alloc + store_q=true |
| `q_output/k_output/v_output` 在 prefix-prompt 路径被 `invokeLoadPrefixKVCache*` 写、被 `invokeGatherSequencesCombined` 读 | ✅ `needs_padded_qkv` 与 line 687 的 prefix gate 完全一致 |
| `runCKFmhaV2` (prefix path else 分支) 内部 line 1004 `q_output->updateShape(...)` + line 1005 `gemm({*q_output, ...})` | ✅ 此 else 分支只在 `prefix_prompt > 0` 时进入，q_output 保证 alive；line 1001-1003 的 `RTP_LLM_CHECK` 也已经验证 |
| `writeCacheStore` (line 853) 是否依赖 k_output/v_output | ✅ 不依赖；它读 `kv_cache_buffer`（paged cache），与 store_cache 路径独立 |
| v1 kernel 在 `store_q=false` 时是否会 deref `q_buf` | ✅ 已读 `unfused_attention_kernels.cu:3697` 的 `if (store_q)` gate |
| `printBufferData` 在 trace 模式下解引用 nullptr | ✅ 已加 `if (q_output)` 等守卫 |
| `selfAttentionwrapper`（!use_aiter_pa decode 路径）依赖 | ✅ 不在 contextAttention 范围，未触及 |

---

## 6. 回滚

```bash
git apply -R optimization_checkpoint/dead_qkv_writes.patch
# 或
git checkout -- rtp_llm/cpp/devices/rocm_impl/ROCmAttentionOp.cc
```

---

## 7. 后续优化候选（本 checkpoint 不实施，仅记录）

按 ROI 排序：

| # | 优化 | 预计省 | 难度 |
|---|---|---|---|
| 2 | 锁定并消除 fmha setup 的 HtoD/DtoH 抖动（trace 里那 ~14 µs 块） | 30 µs/层 + 流水 | 中（需 backtrace 定位 `.item()` 或 `torch::tensor(scalar)`） |
| 3 | 将 v1 kernel 的 store_cache 拆成 SGLang 风格 Triton scatter | 边际，可能负收益 | 中 |
| 4 | 改 ragged QKV layout，省掉 prefix path 的 gather pass | 140 µs/层（仅 prefix 场景）| 高 |

---

## 8. Phase 状态

按 Playbook：
- [x] Phase 0: checkpoint 创建 + 环境记录
- [x] Phase 1: 问题分析 + 根因定位（trace 比对 + 代码 review）
- [x] Phase 2: 方案确认（patch 已生成）
- [x] Phase 3: 服务器 apply + 编译（v1 patch hunk 偏移 → 改用手动 Edit + git diff 重新生成 v2 patch）
- [x] Phase 4: 正确性验证（perf_test 通过，无 NaN/异常）
- [x] Phase 5: 性能验证（trace 对比） — **❌ patch 未生效**
- [ ] Phase 6: 决策 — **进入根因二次分析（§9）**

---

## 9. 当前进展 + 移机继续

### 9.1 patch 移植说明

| 文件 | 状态 |
|---|---|
| `dead_qkv_writes.patch` | **历史 v1**，基于 commit `a0b8191c8` 生成；当前 HEAD 上 hunk header 行号偏移 3 行，`git apply` 会报 "corrupt patch at line 24"。**不要再用这个。** |
| `dead_qkv_writes_v2.patch` | **当前实际改动**（基于 HEAD `0c4c39879`），含 dead-write elimination + DEBUG LOG（写 `/tmp/dead_qkv_debug.log`）。在新机器上 `git apply --check optimization_checkpoint/dead_qkv_writes_v2.patch` 应通过。 |

### 9.2 实测数据（关键结论）

**测试场景**：Qwen3.5-9B TP=2，prefill 15k tokens，`USE_ASM_PA=0`，`ENABLE_CUDA_GRAPH=1`，单 sample。

**Profiling 工具**：用 §0.4 的 perf_test，**不是** benchmark.py（后者 + `gen_timeline=True` 会让 service 卡死）。

**timeline 文件**（用户已 dump 在 `/root/wenhua_code/rtp_profiling/full-attn/`）：

| 文件 | 含义 | 大小 |
|---|---|---|
| `rtp-9b-15k-0420.json` | RTP baseline (HEAD `0c4c39879`，无 patch) | 5.7 MB |
| `rtp-9b-15k-0421-opi.json` | RTP after (HEAD + dead_qkv_writes_v2 但 **未含 DEBUG LOG**) | 5.7 MB |
| `sgl-9b-15k-nocache.json` | SGLang baseline (相同模型 / 序列) | 277 MB |

**Benchmark.py 端到端时间**（concurrency=1，单次）：

| 指标 | Baseline | After (patched) | Δ |
|---|---|---|---|
| prefill_time | 2575.67 ms | 2565.22 ms | −10.45 ms (−0.41%，噪声) |
| decode_time/token | 6.025 ms | 6.055 ms | +0.030 ms (+0.50%，噪声) |

**Per-kernel 对比**（来自 timeline）：

| Kernel | Baseline | After | Δ |
|---|---|---|---|
| `add_fusedQKV_bias_transpose_prefill_kernel_v1` | n=8, p50=**968 µs** | n=8, p50=**986 µs** | **+18 µs（噪声）** |
| `FillFunctor<c10::BFloat16>` (zero-init) | **n=52** | **n=52** | **0** |

**结论**：FillFunctor 数量完全一致 → `q_output/k_output/v_output` 还在被分配 → **`needs_padded_qkv` 在每次 prefill call 都是 `true`**（与"nocache 时应为 false"的假设矛盾）。

### 9.3 已排除的可能性

- ✅ patch 编进 .so：`librtp_compute_ops.so` mtime newer than `.cc` mtime
- ✅ service 加载正确 .so：`from librtp_compute_ops import *` 直接 dlopen `bazel-bin/librtp_compute_ops.so`，nm 确认 `T _ZN7rtp_llm10ROCmDevice16contextAttention...` 在该 .so 里
- ✅ v1 kernel 内有 `if (store_q) / if (store_kv)` gate（unfused_attention_kernels.cu line 2389/2409/2588/2609 等）— 不是 kernel 实现的问题

### 9.4 待验证假设（接着干这里入手）

**假设**：在当前启动配置下（`USE_ASM_PA=0` + `LOAD_PYTHON_MODEL=1` + `ENABLE_CUDA_GRAPH=1` + nocache 输入），`prefix_prompt_param.max_prefix_prompt_length` 在每次 prefill 调用都 > 0，导致 `needs_padded_qkv=true`，patch 等于不生效。

**验证方法**：v2 patch 已加 DEBUG LOG（写 `/tmp/dead_qkv_debug.log`）。在新机器：

```bash
cd <repo>
git apply optimization_checkpoint/dead_qkv_writes_v2.patch
USE_BAZEL_VERSION=6.4.0 bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100
rm -f /tmp/dead_qkv_debug.log
# 跑 §0.4 的 perf_test 命令（partial=2，prefill only）
cat /tmp/dead_qkv_debug.log | head -10
```

**预期看到的输出格式**：
```
[DEAD_QKV_DEBUG] needs_padded_qkv=? max_prefix_prompt_length=? use_mtp_pa_=? prefix_prompt_lengths_ptr=0x... batch_size=1 seq_len=15000 token_num=15000
```

**判读**：
- 若 `needs_padded_qkv=0`（false）但 timeline 里 v1 kernel 没变 → kernel 内 if-gate 不省时（仅省写入字节，runtime 仍跑 RoPE/Q 计算），patch 思路要换
- 若 `needs_padded_qkv=1`（true） + `max_prefix_prompt_length>0` → 追溯到 `GptModel.cc:182` 看为何 `max_prefix_length>0`；这是更可能的 case

### 9.5 接着干清单（按建议顺序）

1. **(a) 拿到 DEBUG LOG**（最优先，10 分钟搞定）：apply v2 patch → 编译 → perf_test → 读 `/tmp/dead_qkv_debug.log`。决定下一步走 9.4 的哪一支。
2. **(b) 对照 SGLang 277 MB timeline**：用 `analyze_framework_compare.py` 思路（rtp-llm-profiler skill 的 references/scripts/）做 RTP vs SGL per-layer 对比，找出 SGL 在 prefill wrapper 这段省下的 211 µs（Q+K RmsNorm 那条）和 102 µs（gather）从哪来 — 用户 SGLang 源码在 `/root/sglang/`，结合代码读 SGL 的 prefill attention 实现（典型路径：`sglang/srt/layers/attention/`）反推 RTP 还能学什么。
3. **(c) 若 (a) 验证 `needs_padded_qkv` 一直 true**：去 `GptModel.cc:182` 加同款 LOG 打 `attention_inputs.max_prefix_length` 和 `prefix_lengths` 内容，找出"为什么 nocache prefill 也有 prefix_length"。怀疑方向：第二次以后的 prefill request 可能命中了 reuse cache（`REUSE_CACHE` 默认值需要确认；perf_test 里有 1 次 warmup + 1 次 active，warmup 后 KV cache 已缓存，active 这一次进来就有 prefix）。如果是这个原因，patch 假设要变成"prefill_lengths 整段全 0 时跳过"，而不是"max_prefix_prompt_length=0 时跳过"。
4. **(d) 若 (a) 显示 `needs_padded_qkv=false` 但 timeline 没变**：v1 kernel 的 RoPE / vec load 是主开销，`if (store_q/store_kv)` 跳过的只是 store。要换思路，可能要复用 SGLang 的 add_rmsnorm_quant×2 + 独立 mrope + scatter 的拆分方案（即原 P1，用户之前说不做，但如果 P0 失效可重新评估）。

### 9.6 移机后第一组命令（cheatsheet）

```bash
# 0. 确认环境
cd <repo_root>
git log --oneline -3   # 期望 HEAD 是 0c4c39879 或包含 dead_qkv_writes_v2 的 commit

# 1. apply patch（如果是 fresh repo）
git apply --check optimization_checkpoint/dead_qkv_writes_v2.patch
git apply optimization_checkpoint/dead_qkv_writes_v2.patch

# 2. 编译
USE_BAZEL_VERSION=6.4.0 bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100

# 3. kill 旧 server（很重要，每次启动前）
pkill -9 -f rtp_llm; sleep 3

# 4. 跑 perf_test 抓 timeline + LOG
rm -f /tmp/dead_qkv_debug.log
# 把 §0.4 完整命令贴这里
# ...
cat /tmp/dead_qkv_debug.log | head -10
ls /tmp/perf_test_outputs2/profiler_wr*.json 2>/dev/null   # 如果 sandbox 输出在这里
```

---

服务器上完成 (a) 之后回头更新此 §9.4 / §9.5。
