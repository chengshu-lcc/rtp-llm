"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation"""

import logging
from typing import Optional

import aiter
import torch
from aiter import hipb_mm, hipb_create_extension
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile
from functools import lru_cache

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
from rtp_llm.ops import HWKernelConfig

class RocmFp8PTPCLinear(LinearBase):
    """ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Handle FP8_PER_CHANNEL_COMPRESSED and FP8_PER_CHANNEL_QUARK"""
        if weight_scales is None or quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK")

    @staticmethod
    @lru_cache(maxsize=1)
    def _init_hipblas():
        hipb_create_extension()

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        # Reshape weight from [k, n] to [n, k] as done in C++ code
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias
        self.use_hipb_mm = (
            hw_kernel_config is not None and hw_kernel_config.use_swizzleA
        )

    def _forward_hipb_mm(self, input_fp8: torch.Tensor, input_scales: torch.Tensor,
                         output_dtype: torch.dtype) -> torch.Tensor:
        """FP8 GEMM via hipBLASLt (hipb_mm with bpreshuffle)."""
        self._init_hipblas()
        # hipb_mm expects weight as (k, n), so transpose from stored (n, k)
        # scaleB needs to be transposed for per-channel layout
        return hipb_mm(
            input_fp8,
            self.weight.t(),
            solution_index=-1,
            bias=None,
            out_dtype=output_dtype,
            scaleA=input_scales,
            scaleB=self.weight_scales.t(),
            scaleOut=None,
            bpreshuffle=True,
        )

    def _forward_aiter(self, input_fp8: torch.Tensor, input_scales: torch.Tensor,
                       output_dtype: torch.dtype) -> torch.Tensor:
        """FP8 GEMM via aiter CK kernels (gemm_a8w8_bpreshuffle)."""
        hidden_dim = input_fp8.shape[-1]
        if hidden_dim < 192:
            num_tokens = input_fp8.shape[0]
            output = torch.empty(
                (num_tokens, self.output_size), dtype=output_dtype,
                device=input_fp8.device,
            )
            gemm_a8w8_bpreshuffle_cktile(
                input_fp8, self.weight, input_scales, self.weight_scales, output
            )
            return output
        return aiter.gemm_a8w8_bpreshuffle(
            input_fp8, self.weight, input_scales, self.weight_scales,
            None, output_dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_dtype = input.dtype
        # Convert to BF16 if needed
        input_bf16 = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)

        quantization_eps = 1e-10
        input_fp8, input_scales = rocm_per_token_quant_fp8(
            input_bf16, eps=quantization_eps,
        )
        input_scales = input_scales.to(torch.float32)

        if self.use_hipb_mm:
            output = self._forward_hipb_mm(input_fp8, input_scales, input_bf16.dtype)
        else:
            output = self._forward_aiter(input_fp8, input_scales, input_bf16.dtype)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if output.dtype != original_dtype:
            output = output.to(original_dtype)

        return output