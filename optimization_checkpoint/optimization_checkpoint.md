# RTP-LLM ROCm Prefill 优化总 Checkpoint（Qwen3.5-9B / qwen3_next）

> **作用**：本文件是所有 prefill 优化项的**索引**。每一项 N1–N11 在 `optimization_checkpoint/n*_<short_name>/` 下都有独立的 sub-checkpoint。
>
> **来源**：`/root/chengshu_code/rtp_profile/analysis/9b_15k/per_layer_optimization_items.md` 中的 RTP vs SGLang per-layer 对比，外加历史 Work A / Work B。
>
> **目标**：把 RTP prefill 在 9B/15k/TP=2 场景下的 GPU 时间从 **976 ms** 降到 SGLang 的 **907 ms** 量级（~70 ms gap）。

---

## 0. 公共环境

- **仓库**：`/root/rtp-llm`
- **当前分支**：`qwen-0401-dm-wh`
- **基线 commit**：`4517b6418 opti`
- **当前 HEAD**：`622bb48f7 n1: swap fused_add_rmsnorm to aiter CK kernel`
  - = `4517b6418` + `6075b5b8b`（N7/Work A WIP）+ `3c529f051`（N2/Work B done）+ `622bb48f7`（N1 done）
- **GPU**：AMD MI355X / MI308X（gfx950）
- **典型场景**：Qwen3.5-9B（qwen3_next，32 层 = 24 GDN + 8 Full-Attn）TP=2，prefill 15k tokens，BF16，`USE_ASM_PA=0`

### 0.1 服务启动命令

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

### 0.2 编译命令

```bash
USE_BAZEL_VERSION=6.4.0 bazelisk build --config=rocm //rtp_llm:rtp_llm --jobs=100
```

> 顶层 target `//rtp_llm:rtp_llm` 会聚合 `librtp_compute_ops.so` / `libth_transformer.so`。Python-only 改动（N1/N2 等走 Python 路径的）无需重编。

### 0.3 Profiling — perf_test 抓 timeline

```bash
USE_BAZEL_VERSION=6.4.0 bazelisk test //rtp_llm/test/perf_test:perf_test --config=rocm \
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
  --test_env=LD_LIBRARY_PATH=/opt/conda310/lib \
  --test_output=streamed --nocache_test_results --sandbox_debug
```

`--partial=2` = 仅 prefill；从 stdout `Prefill Result` 表抓 `Prefill Time(ms)`。timeline 输出 `profiler_wr*.json`，sandbox 退出前在 sandbox tmp 路径下，按 stdout 提示拉到 `/root/rtp-llm/`。

### 0.4 Benchmark 命令（`/root/wenhua_code/rtp-test`）

```bash
START_PORT=8066 TOKENIZER_PATH=~/Qwen3.5-9B/ CHECKPOINT_PATH=~/Qwen3.5-9B/ \
DP_SIZE=1 python benchmark.py
```

`benchmark.py` 默认 `test_lengths=[15000]`、`concurrencies=[1]`、`output_length=500`。

---

## 1. 优化项总览（N1–N11）

> 收益估算来自 `per_layer_optimization_items.md` §2，单位是单次 prefill 节省的 GPU 时间（per-rank, TP=2）。已实测的会同时给出 perf_test prefill_time 实测值。

| # | 项目 | 状态 | 估算 | 实测 | Sub-checkpoint |
|---|---|---|---:|---:|---|
| **N1** | Pre/Post-attn `_fused_add_rmsnorm_kernel` → aiter CK | ✅ Done (commit `622bb48f7`) | ~6 ms 安全路径 / ~21 ms 完整路径 | **−17.5 ms** (931→914 ms) | [n1_fused_add_rmsnorm/](./n1_fused_add_rmsnorm/optimization_checkpoint.md) |
| **N2** | GDN `l2norm_fwd_kernel1` ×2 → fused `fused_l2norm_qk_kernel` | ✅ Done (commit `3c529f051`) | ~18.5 ms | 单 kernel 842→49 µs (17×) | [n2_fused_l2norm_qk/](./n2_fused_l2norm_qk/optimization_checkpoint.md) |
| **N3** | GDN 三 kernel 合并 → `_fused_merge_recompute_kernel` | ✅ Done | ~3.5 ms (单独) / ~9.1 ms (+N6) | **−2.25 ms** (914→911 ms) | [n3_fused_merge_recompute/](./n3_fused_merge_recompute/optimization_checkpoint.md) |
| **N4** | AllReduce → aiter `allreduce_prototype_twoshot` | ⏳ TODO（环境变量 flip + 验证） | ~10.7 ms | — | [n4_aiter_allreduce_twoshot/](./n4_aiter_allreduce_twoshot/optimization_checkpoint.md) |
| **N5** | GDN `_causal_conv1d_fwd_kernel` → `_split_kernel` | ⏳ TODO | ~2.0 ms | — | [n5_causal_conv1d_split/](./n5_causal_conv1d_split/optimization_checkpoint.md) |
| **N6** | GDN `chunk_scaled_dot_kkt + chunk_local_cumsum` 融合 | ⏳ TODO（依赖 N3） | ~0.6 ms (净) | — | [n6_fused_cumsum_kkt/](./n6_fused_cumsum_kkt/optimization_checkpoint.md) |
| **N7** | Full-Attn `add_fusedQKV_bias_transpose_prefill_kernel_v1` 去 dead writes | ❌ PATCH 落死代码（`LOAD_PYTHON_MODEL=1` 走 `FusedRopeKVCacheOp.cc` 不走 `ROCmAttentionOp.cc`）；建议 revert `6075b5b8b`，按 sub-checkpoint §10 重做 | ~25 ms（v1 kernel 总耗时上限）/ 实际可省取决于路径 (c) | 0 | [n7_dead_qkv_writes/](./n7_dead_qkv_writes/optimization_checkpoint.md) |
| **N8** | Fold `store_ssm_state_to_block_map` + `load_initial_state_from_block_map` 进 chunk_o 尾段 | ⏳ TODO | ~0.5 ms | — | [n8_fold_ssm_state/](./n8_fold_ssm_state/optimization_checkpoint.md) |
| **N9** | Full-Attn `sigmoid_kernel + mul` → fused `_fused_sigmoid_mul_kernel` | ⏳ TODO | ~0.1 ms | — | [n9_fused_sigmoid_mul/](./n9_fused_sigmoid_mul/optimization_checkpoint.md) |
| **N10** | GDN `_layer_norm_fwd_1pass_kernel` 重新 autotune | ⏳ TODO（autotune cache rebuild） | ~1.9 ms | — | [n10_layernorm_autotune/](./n10_layernorm_autotune/optimization_checkpoint.md) |
| **N11** | GDN `chunk_fwd_kernel_o` 重新 autotune | ⏳ TODO（autotune cache rebuild） | ~4.8 ms | — | [n11_chunk_fwd_o_autotune/](./n11_chunk_fwd_o_autotune/optimization_checkpoint.md) |
| **N12** | 消除 Python-side `tensor.contiguous() / .to()` 复制（attention path）| ✅ Done (Plan A+B+C, 2026-04-21) | ~2.5–3.5 ms | **−3 ms**（A 单独贡献，B+C 在波动内 ≈ 0） | C++ RoPE op 直出 packed K/V `[total_kv_tokens, Hkv, D]`（替代 padded `[B, Hkv, T+prefix, D]`），Python 侧砍掉 per-batch unpad+`.contiguous()`+`torch.cat`；同时让 `FMHAParams` 直接复用 `attn_inputs.cu_seqlens` 取消 HtoD；删除调试 print |
| **N14** | RTP 用 5 种 GEMM tile shape，SGL 只用 1 种 | ⏳ RESEARCH（架构性） | ~3–5 ms | — | 详见 addendum §B.3。`MT256x96x64`/`MT256x160x32`/`MT192x128x64`/`MT256x80x64`/`MT32x224x128` vs SGL 单一 `MT256x96x64`。需改 rocBLAS dispatch 或权重 layout |

**估算合计**：N1–N12 ≈ **76 ms**（N9 修正 0.1→0.55 ms + 新增 N12 ~3 ms），仍约等于 RTP vs SGL 70 ms gap（含小幅重叠）。N14 留作 P3 研究项。

**目前已实拿**：
- N1 实测 −17.5 ms（perf_test prefill_time 931→914 ms）
- N2 单 kernel 实测 17× 加速（端到端待测）
- N3 实测 −2.25 ms（perf_test prefill_time 914→911 ms）；估算与实测都偏小，N3+N6 一起做才能拿满 9.1 ms
- N12 实测 −3 ms（perf_test prefill_time 911→908 ms，3 次中位数）；A/B/C 拆分：A (packed K/V) 贡献 ~2.9 ms，B+C 在 ±2 ms 单次波动内不可分辨
- 其它 0 / 待测 / WIP

---

## 2. RTP 相对 SGLang 已经更快的项 —— 务必不要回归

| Kernel | RTP | SGLang | RTP 优势 |
|---|---:|---:|---:|
| `act_and_mul_kernel<bf16>` (silu+mul) | 261 µs/call | 325 µs/call | **+1.9 ms / prefill** |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (GDN) | 1270 µs/layer | 1434 µs/layer | **+3.9 ms / prefill** |
| `ck_tile::FmhaFwd` (Full-Attn) | 10535 µs/layer | 11308 µs/layer | **+6.2 ms / prefill** |

→ 在做 N3/N5/N11 等 GDN 改动时，**不要**顺手把这三个 kernel 也换掉。同样 ck_tile / Triton kernel，RTP 这边的 build/autotune 是赢的。

---

## 3. 后续推进顺序（按 ROI 排序）

| 优先级 | 项目 | 难度 | 预期收益 | 备注 |
|---|---|---|---:|---|
| P0-1 | N4 aiter twoshot allreduce | 低（env flip + 跑回归） | ~10.7 ms | 需要 8GPU 上验证更明显 |
| P0-2 | N11 重 autotune `chunk_fwd_kernel_o` | 低 | ~4.8 ms | 同 Triton kernel，差 config |
| P1-3 | N7 重新选向（§10 candidate (c) paged prefill 给 nocache） | 高 | ~25 ms（仅 (c)/(d) 候选） | 死代码 patch 已确认无效；需 micro-bench paged prefill 在 nocache 长 query 上的精度+速度 |
| P1-4 | N5 移植 `_causal_conv1d_fwd_split_kernel` | 中 | ~2.0 ms | drop-in |
| ~~P1-5~~ | ~~**N12** 消 Python `.contiguous()/.to()` 复制~~ | ~~中~~ | ✅ Done −3 ms | Plan A (packed K/V from C++ RoPE op) 贡献 ~2.9 ms；B+C 在波动内 |
| P2-6 | N10 重 autotune `_layer_norm_fwd_1pass_kernel` | 低 | ~1.9 ms | 同 Triton kernel |
| P2-7 | N6 / N8 / N9 | 低/小 | 合计 ~1.6 ms（N9 修正 0.55 ms） | 顺手做 |
| P3-8 | **N14** 收敛 GEMM tile shapes | 高（架构性） | ~3–5 ms | 5 shapes → 1，需改 rocBLAS dispatch |

---

## 4. 文件清单

```
optimization_checkpoint/
├── optimization_checkpoint.md            # 本文件（总索引）
├── n1_fused_add_rmsnorm/                 # ✅ Done
│   ├── optimization_checkpoint.md
│   └── micro_bench.py
├── n2_fused_l2norm_qk/                   # ✅ Done
│   ├── optimization_checkpoint.md
│   └── fused_l2norm_qk.patch
├── n3_fused_merge_recompute/             # ⏳ TODO
│   └── optimization_checkpoint.md
├── n4_aiter_allreduce_twoshot/           # ⏳ TODO
│   └── optimization_checkpoint.md
├── n5_causal_conv1d_split/               # ⏳ TODO
│   └── optimization_checkpoint.md
├── n6_fused_cumsum_kkt/                  # ⏳ TODO
│   └── optimization_checkpoint.md
├── n7_dead_qkv_writes/                   # ⚠️ WIP
│   ├── optimization_checkpoint.md
│   └── dead_qkv_writes_v3.patch
├── n8_fold_ssm_state/                    # ⏳ TODO
│   └── optimization_checkpoint.md
├── n9_fused_sigmoid_mul/                 # ⏳ TODO
│   └── optimization_checkpoint.md
├── n10_layernorm_autotune/               # ⏳ TODO
│   └── optimization_checkpoint.md
└── n11_chunk_fwd_o_autotune/             # ⏳ TODO
    └── optimization_checkpoint.md
```

> `v2_baseline_4517b6418/`、`bench_*.txt`、`dead_qkv_writes{,_v2}.patch` 是迁移前的临时档案，内容已散到 n2/n7 等子目录，可在确认无遗漏后删除。

---

## 5. 总回滚

```bash
# 一次性回到新 baseline
git reset --hard 4517b6418

# 或单独 revert
git revert 622bb48f7    # N1
git revert 3c529f051    # N2
git revert 6075b5b8b    # N7（WIP）
```

---

*创建于 2026-04-21 —— 基于 per-layer 对比报告，统一管理 N1–N11 全部优化项。*
