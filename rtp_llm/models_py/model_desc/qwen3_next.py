import logging
import os
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.utils.debug import cudagraph_debug_kernel
from rtp_llm.ops import (
    AttentionConfigs,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

# Global flag and directory for precision debug dumping.
# Set environment variable QWEN3_NEXT_DUMP_DIR to enable dumping,
# e.g. export QWEN3_NEXT_DUMP_DIR=/tmp/qwen3_next_dump
#
# When running with TP > 1, each rank writes to its own sub-directory
# (rank_0/, rank_1/, ...) to avoid file write conflicts. For precision
# comparison you typically only need rank_0's output since all dump
# points are placed *after* all_reduce and contain identical values
# across ranks.
_DUMP_DIR = os.environ.get("QWEN3_NEXT_DUMP_DIR", "")
_DUMP_ENABLED = bool(_DUMP_DIR)
_DUMP_STEP_COUNTER = 0
_DUMP_RANK: Optional[int] = None
_DUMP_WEIGHTS_DONE = False
_IS_PREFILL_GDN = False


def _get_dump_rank() -> int:
    """Get the current process rank for TP-safe dump directory isolation."""
    global _DUMP_RANK
    if _DUMP_RANK is not None:
        return _DUMP_RANK
    rank = 0
    try:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    except Exception:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    _DUMP_RANK = rank
    return rank


def _set_prefill_gdn(is_prefill: bool):
    """Set the global prefill GDN flag for activation logging."""
    global _IS_PREFILL_GDN
    _IS_PREFILL_GDN = is_prefill


def _dump_tensor(tensor: torch.Tensor, name: str):

    if not _DUMP_ENABLED:
        return
    rank = _get_dump_rank()
    step_dir = os.path.join(_DUMP_DIR, f"rank_{rank}", f"step_{_DUMP_STEP_COUNTER}")
    os.makedirs(step_dir, exist_ok=True)
    file_path = os.path.join(step_dir, f"{name}.pt")
    torch.save(tensor.detach().cpu(), file_path)


def _increment_dump_step():
    """Advance the global dump step counter (call once per forward pass)."""
    global _DUMP_STEP_COUNTER
    # if _DUMP_ENABLED:
    _DUMP_STEP_COUNTER += 1


def _dump_weights_once(weights_dict: Dict[str, torch.Tensor], prefix: str):
    """Dump all weight tensors once (at first forward pass) for cross-framework comparison.

    Weights are saved under _DUMP_DIR/rank_R/weights/ to separate them from
    per-step activation dumps.
    """
    global _DUMP_WEIGHTS_DONE
    if not _DUMP_ENABLED or _DUMP_WEIGHTS_DONE:
        return
    _DUMP_WEIGHTS_DONE = True
    rank = _get_dump_rank()
    weight_dir = os.path.join(_DUMP_DIR, f"rank_{rank}", "weights")
    os.makedirs(weight_dir, exist_ok=True)
    for name, tensor in weights_dict.items():
        safe_name = name.replace("/", "_").replace(".", "_")
        file_path = os.path.join(weight_dir, f"{prefix}_{safe_name}.pt")
        torch.save(tensor.detach().cpu(), file_path)


def _dump_config_once(config_dict: Dict[str, Any], prefix: str):
    """Dump model configuration parameters once for cross-framework comparison."""
    if not _DUMP_ENABLED:
        return
    rank = _get_dump_rank()
    config_dir = os.path.join(_DUMP_DIR, f"rank_{rank}", "config")
    os.makedirs(config_dir, exist_ok=True)
    file_path = os.path.join(config_dir, f"{prefix}_config.pt")
    torch.save(config_dict, file_path)


def _is_target_verify(attention_inputs: PyAttentionInputs) -> bool:
    """Check if the current forward pass is in target verify mode."""
    # current impl will judge prefill with prefix as target verify, which cause problem is causal_conv1d_update, which needs more block id
    # so disable it temp, we should mark it in structure
    return False
    return (
        attention_inputs.input_lengths is not None
        and attention_inputs.prefix_lengths is not None
        and attention_inputs.prefix_lengths.size(0) > 0
        and torch.all(
            attention_inputs.input_lengths == attention_inputs.input_lengths[0]
        ).item()
        and torch.max(attention_inputs.input_lengths).item() < 10
        and torch.min(attention_inputs.prefix_lengths).item() > 0
    )


class Qwen3NextMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


class Qwen3NextGatedDeltaNetBase(torch.nn.Module):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.layer_idx = layer_idx
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // parallelism_config.tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )
        self.num_key_value_heads: int = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        _, block_size = kv_cache_tensor.view(kv_cache_tensor.shape[0], -1).shape
        assert (
            block_size >= self.ssm_state_size + self.conv_state_size
        ), "block_size is too small, please check seq_size_per_block"
        conv_states = torch.as_strided(
            kv_cache_tensor,
            (kv_cache_tensor.shape[0], self.linear_conv_kernel_dim - 1, self.qkv_size),
            (kv_cache_tensor.stride()[0], self.qkv_size, 1),
            storage_offset=self.ssm_state_size + kv_cache_tensor.storage_offset(),
        )
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        # maybe should support smsm cahe with difference dtype(fp32/bf16/fp16)
        _, block_size = kv_cache_tensor.view(kv_cache_tensor.shape[0], -1).shape
        assert (
            block_size >= self.ssm_state_size + self.conv_state_size
        ), "block_size is too small, please check seq_size_per_block"
        ssm_states = torch.as_strided(
            kv_cache_tensor,
            (
                kv_cache_tensor.shape[0],
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            (
                kv_cache_tensor.stride()[0],
                self.head_k_dim * self.head_v_dim,
                self.head_k_dim,
                1,
            ),
            storage_offset=kv_cache_tensor.storage_offset(),
        )
        return ssm_states


class Qwen3NextGatedDeltaNetPrefill(Qwen3NextGatedDeltaNetBase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ):
        super().__init__(linear_attn_config, parallelism_config, weights, layer_idx)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
    ) -> torch.Tensor:
        # cu_seqlen_without_padding = attn_inputs.cu_seqlens[
        #     : attn_inputs.input_lengths.size(0) + 1
        # ]
        cu_seqlen_without_padding = attn_inputs.cu_seqlens
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_d,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        lp = f"layer_{self.layer_idx}_gdn"
        _dump_tensor(self.alog, f"{lp}_prefill_fla_alog")
        _dump_tensor(self.dt_bias, f"{lp}_prefill_fla_dt_bias")
        _dump_tensor(a, f"{lp}_prefill_fla_a_input")
        _dump_tensor(b, f"{lp}_prefill_fla_b_input")
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        _dump_tensor(g, f"{lp}_prefill_fla_g")
        _dump_tensor(beta, f"{lp}_prefill_fla_beta")
        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        # cu_seqlens_without_padding = attn_inputs.cu_seqlens[: context_batch_size + 1]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens
        lp = f"layer_{self.layer_idx}_gdn"
        _dump_tensor(cu_seqlens_without_padding, f"{lp}_prefill_fla_cu_seqlens")
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            )

            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_d,
                attn_inputs.kv_cache_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        lp = f"layer_{self.layer_idx}_gdn"
        _dump_tensor(query, f"{lp}_prefill_fla_query_before_view")
        _dump_tensor(key, f"{lp}_prefill_fla_key_before_view")
        _dump_tensor(value, f"{lp}_prefill_fla_value_before_view")
        query = query.view(1, query.shape[0], self.local_num_k_heads, self.head_k_dim)
        key = key.view(1, key.shape[0], self.local_num_k_heads, self.head_k_dim)
        value = value.view(1, value.shape[0], self.local_num_v_heads, self.head_v_dim)
        _dump_tensor(query, f"{lp}_prefill_fla_query")
        _dump_tensor(key, f"{lp}_prefill_fla_key")
        _dump_tensor(value, f"{lp}_prefill_fla_value")
        attn_out, h, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
        )
        if ssm_states is not None:
            store_ssm_state_to_block_map(
                h,
                final_state.to(h.dtype),
                attn_inputs.prefix_lengths_d,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )
        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        lp = f"layer_{self.layer_idx}_gdn"
        _dump_tensor(mixed_qkv, f"{lp}_prefill_conv1d_input")
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        _dump_tensor(mixed_qkv, f"{lp}_prefill_conv1d_output")
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, attn_inputs
        )
        _dump_tensor(attn_out, f"{lp}_prefill_fla_output")
        if kv_cache is not None:
            # write kvcache to cache store
            compute_ops.write_cache_store(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id_host,
                attn_inputs.cache_store_inputs,
                kv_cache,
            )
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ):
        super().__init__(linear_attn_config, parallelism_config, weights, layer_idx)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        # (batch, dim) -> # (batch, dim, 1)
        batch, seq = self._get_bs_from_attenion_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
        )
        out = out.transpose(1, 2).reshape(origin_shape)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attenion_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # asserr head_k_dim == head_v_dim
        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads,
                self.local_num_k_heads,
                self.local_num_v_heads,
            ],
            dim=2,
        )
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)

        # contiguous will be applyed when call fused_recurrent_gated_delta_rule
        g = g.view(batch, seq, self.local_num_v_heads)
        beta = beta.view(batch, seq, self.local_num_v_heads)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states,
            inplace_final_state=True,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            use_qk_l2norm_in_kernel=True,
        )
        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )
        attn_out = self._fla(
            mixed_qkv,
            b,
            a,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )

        return attn_out

    def _get_bs_from_attenion_input(
        self,
        mixed_qkv: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), f"prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} <=0 when target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), f"token: {token} is not divisible by prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} when target verify"
        b, s = attention_inputs.prefix_lengths.size(
            0
        ), token // attention_inputs.prefix_lengths.size(0)
        return b, s


class Qwen3NextAttention(CausalAttention):
    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
    ):
        super().__init__(
            attn_config, parallelism_config, weights, layernorm_eps, quant_config
        )
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.attn_gate_w, W.attn_gate_s, None, quant_config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        attn_out = super().forward(hidden_states, fmha_impl, kv_cache, gate)
        return attn_out


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.quant_config = quant_config
        self.layer_idx = layer_idx
        # in_proj_qkvz is bf16 / fp8
        self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, None, quant_config
        )
        # in_proj_ba is bf16
        self.in_proj_ba = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_ba_w, None, None, quant_config
        )
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        self.local_num_k_heads = (
            linear_attn_config.linear_num_key_heads // parallelism_config.tp_size
        )
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(
            linear_attn_config, parallelism_config, weights, layer_idx
        )
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(
            linear_attn_config, parallelism_config, weights, layer_idx
        )
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_out_w, W.linear_attn_out_s, None, quant_config
        )

        # Dump config parameters once
        _dump_config_once(
            {
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "local_num_k_heads": self.local_num_k_heads,
                "local_num_v_heads": self.local_num_v_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "linear_conv_kernel_dim": linear_attn_config.linear_conv_kernel_dim,
                "tp_size": parallelism_config.tp_size,
            },
            prefix="gdn",
        )

    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        split_arg_list_qkvz = [
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads,
            self.head_v_dim * self.local_num_v_heads,
        ]

        mixed_qkv, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=1)
        b, a = torch.split(
            mixed_ba, [self.local_num_v_heads, self.local_num_v_heads], dim=1
        )
        # reshape to [token, v_head_num, v_head_dim]
        # b,a should be contiguous for fused_gdn_gating
        return mixed_qkv, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
        ), "prefill_conv1d_meta is required for prefill"
        _set_prefill_gdn(attention_inputs.is_prefill)
        lp = f"layer_{layer_idx}_gdn"
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        # tttt = projected_states_qkvz.detach().float()
        # logging.info(
        #     f"[projected_states_qkvz_{_DUMP_STEP_COUNTER}_layer{layer_idx}] {tttt}: shape={list(tttt.shape)}, dtype={tttt.dtype}, "
        #     f"min={tttt.min().item():.6f}, max={tttt.max().item():.6f}, "
        #     f"mean={tttt.mean().item():.6f}, std={tttt.std().item():.6f}"
        # )

        projected_states_ba = self.in_proj_ba(hidden_states)
        _dump_tensor(projected_states_qkvz, f"{lp}_proj_qkvz")
        _dump_tensor(projected_states_ba, f"{lp}_proj_ba")
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        # Dump individual q, k, v split from mixed_qkv for cross-framework comparison
        _q_size = self.head_k_dim * self.local_num_k_heads
        _k_size = self.head_k_dim * self.local_num_k_heads
        _v_size = self.head_v_dim * self.local_num_v_heads
        _q, _k, _v = torch.split(mixed_qkv, [_q_size, _k_size, _v_size], dim=1)
        _dump_tensor(_q, f"{lp}_q")
        _dump_tensor(_k, f"{lp}_k")
        _dump_tensor(_v, f"{lp}_v")
        _dump_tensor(mixed_qkv, f"{lp}_mixed_qkv")
        _dump_tensor(z, f"{lp}_z")
        _dump_tensor(b, f"{lp}_b")
        _dump_tensor(a, f"{lp}_a")
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            attn_output = self.prefill_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        else:
            attn_output = self.decode_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        _dump_tensor(attn_output, f"{lp}_fla_output")
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        _dump_tensor(attn_output, f"{lp}_norm_output")
        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)
        attn_output = self.out_proj(attn_output)
        _dump_tensor(attn_output, f"{lp}_out_proj_output")
        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        _dump_tensor(attn_output, f"{lp}_final_output")
        _set_prefill_gdn(False)
        return attn_output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = Qwen3NextGatedDeltaNet(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = Qwen3NextAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
            )

        if config.moe_style == 2:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
            )
        elif config.moe_style == 0:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, config.quant_config
            )

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        layer_prefix = f"layer_{self.layer_idx}"
        _dump_tensor(hidden_states, f"{layer_prefix}_input")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        attn_kwargs = dict(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )
        if isinstance(self.self_attn, Qwen3NextGatedDeltaNet):
            attn_kwargs["layer_idx"] = self.layer_idx
        hidden_states = self.self_attn(**attn_kwargs)
        hidden_states = residual + hidden_states
        _dump_tensor(hidden_states, f"{layer_prefix}_after_attn")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        _dump_tensor(hidden_states, f"{layer_prefix}_output")

        return hidden_states


class Qwen3NextModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    @staticmethod
    def _select_block_map_for_layer(
        attention_inputs: PyAttentionInputs, layer_idx: int
    ) -> None:
        if attention_inputs.kv_cache_block_id_device_by_group is None:
            return

        gid = 0
        if attention_inputs.kv_cache_layer_to_group is not None:
            gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())

        if attention_inputs.kv_cache_block_id_device_by_group is not None and len(
            attention_inputs.kv_cache_block_id_device_by_group
        ):
            attention_inputs.kv_cache_block_id_device = (
                attention_inputs.kv_cache_block_id_device_by_group[gid]
            )

        if attention_inputs.kv_cache_block_id_host_by_group is not None and len(
            attention_inputs.kv_cache_block_id_host_by_group
        ):
            attention_inputs.kv_cache_block_id_host = (
                attention_inputs.kv_cache_block_id_host_by_group[gid]
            )
        return gid

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        _dump_tensor(input_ids, "input_ids")
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        prefill_conv1d_meta = None
        if attention_inputs.is_prefill:
            # cu_seqlen_without_padding = attention_inputs.cu_seqlens[
            #     : attention_inputs.input_lengths.size(0) + 1
            # ]
            cu_seqlen_without_padding = attention_inputs.cu_seqlens
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
            )
        # hack temp
        is_target_verify = _is_target_verify(attention_inputs)
        if is_target_verify:
            attention_inputs.sequence_lengths_plus_1_d = (
                attention_inputs.prefix_lengths + 1
            ).to(hidden_states.device)
        attn_meta = Qwen3NextMetadata(prefill_conv1d_meta, is_target_verify)

        # qwen3_next model has only one full group (group 0): use fmha_impl from input param
        # if there is a model with more than 1 full groups,
        # we should prepare fmha_impl for each full group/ fix later

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        _dump_tensor(hidden_states, "embedding_output")

        for i, decoder_layer in enumerate(self.layers):
            # Switch to correct block_map for this layer in hybrid attention mode
            gid = self._select_block_map_for_layer(attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )

        hidden_states = self.norm(hidden_states)
        _dump_tensor(hidden_states, "final_norm_output")
        _increment_dump_step()
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
