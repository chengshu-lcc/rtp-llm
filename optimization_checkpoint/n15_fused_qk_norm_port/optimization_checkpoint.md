# N15 — Port SGL PR 21440 warp-per-head QK-RMSNorm kernel to ROCm

> **优化项 N15**，N13 KILL 后的复活：取 Full-Attn `fusedQkRmsNorm` 那 ~1.6 ms 缺口。
>
> **路线**：参考 SGL PR #21440（CUDA diffusion `qknorm_rope.cuh`）的 warp-per-head + 2D-stride 设计，移植成 HIP，**支持直接吃 fused QKV 切片不需要 `.contiguous()`**。
>
> **预估收益**：单次 322 µs → ~120 µs，× 8 Full-Attn 层 = **~1.6 ms / prefill**
>
> **状态**：⏳ WIP（P0 完成，P1 进行中）

---

## 0. 环境

- **仓库**：`/root/rtp-llm`
- **分支**：`qwen-0401-dm-wh`
- **当前 HEAD**：`7085e757a n3: end-to-end precision verification vs fp32 recurrent reference`（含 N1+N2+N3+N12，prefill 908 ms）
- **GPU**：AMD MI308X / MI355X（gfx950）
- **场景**：Qwen3.5-9B（qwen3_next，TP=2 → per-rank head_num=8, kv_head_num=2, head_dim=256）prefill 15k tokens BF16 `LOAD_PYTHON_MODEL=1`
- **Baseline kernel**：`rtp_llm/cpp/kernels/rocm/fused_qk_rmsnorm.cu` 中的 `fusedQkRmsNorm`（OLD 路径，`norm_size=256` 不命中 `Opt` 分支）
  - 当前实测 322 µs/call × 8 层 = **2576 µs / prefill**
  - 慢的原因：`warp_size=32`（仅用半个 wave64）+ 两 pass（load → sum → load → norm）+ 150k blocks × 32 thread

---

## 1. 设计来源

参考 [SGL PR #21440](https://github.com/sgl-project/sglang/pull/21440)，文件 `python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh`（246 行 CUDA）。
关键设计点（**忽略 RoPE 部分**，只取 norm 部分）：

| 设计点 | PR 21440 | 我们移植后 |
|---|---|---|
| Layout | `[N, H, D]` + 三 stride | 直接吃 RTP 现有 fused QKV `[m, q_size+2*kv_size]` 切片 |
| Weight | `[head_dim]` per-head | 同 |
| Reduce | warp 内 `warp::reduce_sum` | HIP `__shfl_xor` 5 轮（wave64） |
| Q+K | 单 launch，`num_qk_heads = qH + kvH`，K 用负 offset 技巧 | 同（保留） |
| Pass | 单 pass + reg cache | 同 |
| Vector load | `AlignedVector<Packed, kVecSize>` | 自定义 `aligned_vec_t` HIP 等价 |
| 启动配置 | `kThreadsPerBlock=256` (8 warps × 32 thread) | `kThreadsPerBlock=256` (4 warps × 64 thread) |

**为何不用 PR 23186（aiter `fused_qk_rmsnorm`）**：
- 已 N13 §8 验证：JIT build 失败 + 即使能 build，reshape copy 吃光收益 + decode 5-7× 回归

**为何不直接拷 SGL 的 .cuh**：
- CUDA-only：依赖 `__grid_constant__` / PDL launch / `__shfl_xor_sync` mask 语义
- 依赖 SGL 内部框架（`sgl_kernel/{tensor,runtime,type,utils,vec,warp}.h` 共 6 个 header）
- PyPI `sgl-kernel` wheel 无 ROCm 版本

→ 拷 246 行 .cuh + 写 ~30 行 HIP shim header（替换 6 个 SGL helper）= 一个独立文件

---

## 2. 分阶段执行计划与状态

| Phase | 内容 | 状态 | Gate | 实测 |
|---|---|---|---|---|
| **P0** | 建 `n15_fused_qk_norm_port/` 目录 + checkpoint 骨架 + 保存 PR 21440 diff | ✅ Done | — | — |
| **P1** | 写 HIP kernel 草稿（warp-per-head + 2D stride，参考 PR 21440），standalone build 验证可编译 | ✅ Done | hipcc 编译过 ✓ | bazel `rocm_fused_qk_rmsnorm` 8s 通过 |
| **P2** | Standalone micro-bench：数值对比 + 单次延迟 | ✅ Done | atol < 5e-2 vs fp32-ref ✓ + p50 ≤ 200 µs (vs baseline 322 µs) | m=15000: **291.9 → 157.9 µs (1.85×)**；m=7 decode 无回归 (12.4 → 12.3 µs)；diff = BF16 噪声 |
| **P3** | C++ 绑定 + Python 类切换（env flag gated）+ 单测回归 | ✅ Done | `rocm_fusedqkrmsnorm_test.py` 通过 | binding/.pyi 完成、`norm.py` env flag 切换、size_per_head=128 (fallback) + 256 (V2) 数值都通过 |
| **P4** | perf_test 端到端 prefill | ⏳ TODO | 3 次中位数 −1 ms+ | — |
| **P5** | decode 单测验不回归（N13 死因） | ⏳ TODO | decode 延迟无显著回归 | — |
| **P6** | Commit + 更新主 checkpoint | ⏳ TODO | — | — |

每个 Phase 完成后**必须**更新本文件 §6 进度日志，记录：实测数据、遇到的问题、决策。

---

## 3. 文件清单（计划）

```
optimization_checkpoint/n15_fused_qk_norm_port/
├── optimization_checkpoint.md        # 本文件
├── sgl_pr_21440.diff                # PR 21440 完整 diff（备查）
├── micro_bench.py                   # P2 micro-bench 脚本（TODO）
└── bench_results.txt                # P2 bench 输出（TODO）

rtp_llm/cpp/kernels/rocm/
├── fused_qk_rmsnorm.cu              # 原 baseline（不动）
├── fused_qk_rmsnorm.h               # 原 baseline header（不动）
├── fused_qk_rmsnorm_v2.cu           # 新 kernel（P1 创建）
├── fused_qk_rmsnorm_v2.h            # 新 header（P1 创建）
└── qk_norm_hip_shim.h               # SGL helper → HIP 等价（P1 创建）
```

---

## 4. 风险与监控点

| 风险 | 缓解 | 来源 |
|---|---|---|
| HIP `__shfl_xor` 在 wave64 上语义和 CUDA 不一致 | 写 micro-test 单独验证 reduce 正确性 | PR 21440 用 `__shfl_xor_sync(mask,..)`，HIP 没有 mask 概念 |
| head_dim=256, wave=64 → kElemsPerThread=4，vec_size=2，可能 vector load 退化到 4-byte | bench 时检查 ISA：`v_buffer_load_dwordx4` 计数应 = 1/lane | — |
| Decode 路径回归（N13 死因） | P5 单独跑 m∈{1,7,128} 微 bench | N13 §8.2 |
| 数值差异 vs baseline 超出 atol | 对照 fp32 reference + 现有 `rocm_fusedqkrmsnorm_test.py` | — |
| `__grid_constant__` 删除后 param 走普通 ABI，extra register pressure | 看 ISA `s_load_dwordx16` 序列，必要时手 `__constant__` | — |

---

## 5. 回滚

```bash
# 全部新文件位于独立目录/独立文件，删掉即可：
rm -rf optimization_checkpoint/n15_fused_qk_norm_port/
rm rtp_llm/cpp/kernels/rocm/fused_qk_rmsnorm_v2.{cu,h}
rm rtp_llm/cpp/kernels/rocm/qk_norm_hip_shim.h
# 反掉 bazel BUILD + Python wrapper 改动（P3 之后）
git diff HEAD --name-only | xargs git checkout
```

---

## 6. 进度日志（每 phase 完成必填）

### P0 — Done — 2026-04-22

- 创建目录 `optimization_checkpoint/n15_fused_qk_norm_port/`
- 写本 checkpoint 骨架
- 保存 SGL PR 21440 完整 diff 到 `sgl_pr_21440.diff` 备查
- 创建 6 个 TaskCreate（任务 #43-#48）跟踪 P0-P5
- **下一步**：P1 写 HIP kernel `fused_qk_rmsnorm_v2.cu` 草稿

### P1 — Done — 2026-04-22

- 创建 `rtp_llm/cpp/kernels/rocm/fused_qk_rmsnorm_v2.{cu,h}`
  - 接口与 baseline `invokeFusedQkRmsNorm` 完全一致 → drop-in replacement，但加了两个 fallback 条件：
    - `q_bias != nullptr || k_bias != nullptr` → 走 baseline（V2 暂不支持 bias）
    - `norm_size != 256` → 走 baseline（V2 当前只特化 HEAD_DIM=256，命中 Qwen3.5-9B Full-Attn 形状）
  - BF16 路径用 `bf16_4_t`（8 byte / thread）vec load + 单 pass + wave64 `__shfl_xor` reduce
  - 通用 fp32/half 路径用标量 4-elem reg cache，单 pass
  - 一个 wave 处理一个 (token, head) work item，4 wave / block，BLOCK_SIZE=256
  - Grid blocks = `ceil((m * (qH + kvH)) / 4)`：Qwen3.5-9B prefill 实例（m=15000, qH=8, kvH=2）= 37500 blocks（baseline 150000，少 4×）
- 改 `rtp_llm/cpp/kernels/BUILD` `rocm_fused_qk_rmsnorm` 规则，加入 `fused_qk_rmsnorm_v2.{cu,h}`
- `bazelisk build --config=rocm //rtp_llm/cpp/kernels:rocm_fused_qk_rmsnorm` 通过（8s，全新 source 编译只触发该 .cu）
- **下一步**：P2 写 standalone micro-bench（数值 + 单次延迟）→ 通过 gate 后再做 P3 binding 接入

### P2 — Done — 2026-04-22

- 写 `optimization_checkpoint/n15_fused_qk_norm_port/micro_bench.py`
- 添加 sibling 绑定 `rtp_llm_ops.fused_qk_rmsnorm_v2`（接口与 baseline 完全相同）
  - `rtp_llm/models_py/bindings/common/FusedQKRmsNorm.{h,cc}`：新增 `FusedQKRMSNormV2` C++ wrapper（USING_ROCM gated）
  - `rtp_llm/models_py/bindings/rocm/RegisterBaseBindings.hpp`：注册到 pybind module
- `bazelisk build //:rtp_compute_ops` 增量 67 秒通过
- micro-bench 4 个形状 × 2 实现 + fp32 参考的对比（结果存 `bench_results.txt`）：

| m | baseline (µs) | V2 (µs) | speedup | num diff vs fp32 |
|---:|---:|---:|---:|---:|
| **15000 (prefill)** | 291.9 | **157.9** | **1.85×** | 1.56e-2 ✓ |
| 4096 | 89.6 | 55.5 | 1.61× | 1.56e-2 ✓ |
| 256 | 17.4 | 12.8 | 1.36× | 3.91e-3 ✓ |
| **7 (decode)** | 12.4 | **12.3** | **1.00× (no regression!)** | 0 ✓ |

- **Gate 1（数值）**: V2 vs fp32-ref max abs diff = 1.56e-2 < 5e-2，且和 baseline 的 diff 完全一致（同一 BF16 量化噪声）✓
- **Gate 2（性能）**: V2 p50 = 157.9 µs < 200 µs gate ✓
- **Gate 3（decode 不回归）**: m=7 时 V2 等价于 baseline（N13 死因），✓
- **预估端到端节省**: (291.9 − 157.9) µs × 8 层 ≈ **1.07 ms / prefill**

- **下一步**：P3 在 `norm.py` 加 env flag (`USE_FUSED_QK_RMSNORM_V2=1`) 切换 + 跑 `rocm_fusedqkrmsnorm_test.py`

### P3 — Done — 2026-04-22

- 改 `rtp_llm/models_py/modules/base/rocm/norm.py:176` — `FusedQKRMSNorm.forward` 加 env flag 切换：
  - 默认 `USE_FUSED_QK_RMSNORM_V2=1` 走 V2，`=0` fallback baseline
  - `_USE_V2` 在 class body 读 env，import 后不可变（生产合理：env 在启动脚本里设）
- 改 `rtp_llm/ops/librtp_compute_ops/rtp_llm_ops.pyi` 加 `fused_qk_rmsnorm_v2` 类型签名
- 数值回归：
  - `python -m unittest rtp_llm.models_py.modules.base.rocm.test.rocm_fusedqkrmsnorm_test` → OK（覆盖 size_per_head=128，V2 fallback 路径）
  - 自定 smoke 在 size_per_head=256（V2 active）：tokens ∈ {7, 256, 4096, 15000} × dtype ∈ {bf16, fp16}，全部 PASS（atol=1e-2, rtol=1e-2）
- **下一步**：P4 跑 perf_test 端到端验证 `Prefill Time(ms)` 收益

### P4 — TODO

（待填）

### P5 — TODO

（待填）

### P6 — TODO

（待填）

---

*创建于 2026-04-22；继 N13 KILL 后第二次尝试 Full-Attn QK-norm 优化。*
