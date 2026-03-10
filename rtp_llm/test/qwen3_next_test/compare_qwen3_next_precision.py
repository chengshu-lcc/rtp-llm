#!/usr/bin/env python3
"""Qwen3-Next precision comparison: rtp-llm vs HuggingFace.

Loads a Quark-quantized Qwen3-Next model via HuggingFace Transformers
(following https://huggingface.co/docs/transformers/quantization/quark),
runs a single forward pass, dumps per-layer hidden states, and compares
them against tensors previously dumped by rtp-llm.

Usage
-----
1. Run rtp-llm with dump enabled::

       export QWEN3_NEXT_DUMP_DIR=/tmp/rtp_llm_dump
       # launch rtp-llm inference ...

2. Run this script::

       python tools/compare_qwen3_next_precision.py \\
           --model_path /path/to/checkpoint \\
           --rtp_dump_dir /tmp/rtp_llm_dump/rank_0/step_0 \\
           --hf_dump_dir /tmp/hf_dump \\
           --input_text "Hello" \\
           --num_layers 4
"""

import argparse
import glob
import logging
import os
import sys
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------


def compute_diff_metrics(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> Dict[str, float]:
    """Compute max_abs_diff, mean_abs_diff, cosine_sim, relative_diff."""
    float_a = tensor_a.float()
    float_b = tensor_b.float()
    abs_diff = (float_a - float_b).abs()

    cosine_sim = torch.nn.functional.cosine_similarity(
        float_a.flatten().unsqueeze(0),
        float_b.flatten().unsqueeze(0),
    ).item()

    ref_mean = float_b.abs().mean().item()
    mean_abs = abs_diff.mean().item()

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": mean_abs,
        "cosine_sim": cosine_sim,
        "relative_diff": mean_abs / ref_mean if ref_mean > 1e-10 else float("inf"),
    }


def print_diff_report(
    name: str, metrics: Dict[str, float], threshold: float = 1e-3
) -> bool:
    """Print one comparison line. Returns True when diff exceeds threshold."""
    is_bad = metrics["max_abs_diff"] > threshold or metrics["cosine_sim"] < 0.999
    status = "❌ MISMATCH" if is_bad else "✅ OK"
    logger.info(
        f"  {status}  {name:50s}  "
        f"max_abs={metrics['max_abs_diff']:.6e}  "
        f"mean_abs={metrics['mean_abs_diff']:.6e}  "
        f"cos_sim={metrics['cosine_sim']:.8f}  "
        f"rel_diff={metrics['relative_diff']:.6e}"
    )
    return is_bad


# ---------------------------------------------------------------------------
# HuggingFace model: load → forward → dump
# ---------------------------------------------------------------------------


def find_inner_model(model):
    """Locate the inner transformer that owns ``.layers`` and ``.embed_tokens``."""
    for attr_chain in [
        ["model", "language_model"],
        ["model"],
    ]:
        obj = model
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj
    raise RuntimeError(
        "Cannot locate decoder layers. "
        "Tried model.model.language_model.layers and model.model.layers."
    )


def run_hf_model_and_dump(
    model_path: str,
    input_text: str,
    dump_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    num_layers: Optional[int] = None,
) -> None:
    """Load a Quark-quantized HF model following the official Quark integration
    guide (``from_pretrained`` with ``device_map="auto"``), run one forward
    pass, and dump per-layer hidden states to *dump_dir*.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---- optionally truncate to first N layers ----
    config_overrides: dict = {}
    if num_layers is not None:
        logger.info(f"Truncating model to first {num_layers} layers")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.num_hidden_layers = num_layers
        if hasattr(config, "layer_types") and config.layer_types is not None:
            config.layer_types = config.layer_types[:num_layers]
        config_overrides["config"] = config

    # ---- load model (Quark official way) ----
    # Reference: https://huggingface.co/docs/transformers/quantization/quark
    #   model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    logger.info(f"Loading model (dtype={dtype}, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **config_overrides,
    )
    model.eval()

    # ---- tokenize ----
    logger.info(f"Tokenizing: {input_text!r}")
    first_device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(first_device)
    input_ids = inputs["input_ids"]
    logger.info(f"  input_ids shape={input_ids.shape}, ids={input_ids.tolist()}")

    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        input_ids.detach().cpu().squeeze(0),
        os.path.join(dump_dir, "input_ids.pt"),
    )

    # ---- locate decoder layers ----
    inner_model = find_inner_model(model)
    decoder_layers = inner_model.layers
    logger.info(f"Found {len(decoder_layers)} decoder layers")

    # ---- register forward hooks ----
    captured: Dict[str, torch.Tensor] = OrderedDict()

    def make_layer_hook(layer_idx: int):
        def hook(module, args, output):
            layer_input = args[0] if isinstance(args, tuple) else args
            layer_output = output[0] if isinstance(output, tuple) else output
            captured[f"layer_{layer_idx}_input"] = layer_input.detach().cpu()
            captured[f"layer_{layer_idx}_output"] = layer_output.detach().cpu()

        return hook

    hooks = []
    for idx, layer in enumerate(decoder_layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))

    embed_module = getattr(inner_model, "embed_tokens", None)
    if embed_module is not None:
        hooks.append(
            embed_module.register_forward_hook(
                lambda _mod, _args, out: captured.update(
                    {"embedding_output": out.detach().cpu()}
                )
            )
        )

    norm_module = getattr(inner_model, "norm", None)
    if norm_module is not None:
        hooks.append(
            norm_module.register_forward_hook(
                lambda _mod, _args, out: captured.update(
                    {"final_norm_output": out.detach().cpu()}
                )
            )
        )

    # ---- forward ----
    logger.info("Running forward pass ...")
    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    # ---- save ----
    logger.info(f"Saving {len(captured)} tensors to {dump_dir}")
    for name, tensor in captured.items():
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        path = os.path.join(dump_dir, f"{name}.pt")
        torch.save(tensor, path)
        logger.info(f"  {name}: shape={tensor.shape} dtype={tensor.dtype}")

    logger.info("HF dump complete.")


# ---------------------------------------------------------------------------
# Compare rtp-llm dump vs HF dump
# ---------------------------------------------------------------------------


def compare_dumps(
    rtp_dump_dir: str,
    hf_dump_dir: str,
    threshold: float = 1e-3,
) -> Tuple[int, int]:
    """Compare ``.pt`` files. Returns ``(matched, mismatched)``."""
    rtp_files = sorted(glob.glob(os.path.join(rtp_dump_dir, "*.pt")))
    hf_basenames = {
        os.path.basename(f) for f in glob.glob(os.path.join(hf_dump_dir, "*.pt"))
    }

    if not rtp_files:
        logger.error(f"No .pt files in {rtp_dump_dir}")
        return 0, 0

    logger.info(f"rtp-llm files: {len(rtp_files)},  HF files: {len(hf_basenames)}")

    # ---- verify input_ids ----
    rtp_ids_path = os.path.join(rtp_dump_dir, "input_ids.pt")
    hf_ids_path = os.path.join(hf_dump_dir, "input_ids.pt")
    if os.path.exists(rtp_ids_path) and os.path.exists(hf_ids_path):
        rtp_ids = torch.load(rtp_ids_path, map_location="cpu", weights_only=True)
        hf_ids = torch.load(hf_ids_path, map_location="cpu", weights_only=True)
        if torch.equal(rtp_ids, hf_ids):
            logger.info(f"✅ input_ids match ({rtp_ids.shape[0]} tokens)")
        else:
            logger.error(
                f"❌ input_ids MISMATCH!\n"
                f"   rtp-llm: {rtp_ids.tolist()}\n"
                f"   HF:      {hf_ids.tolist()}"
            )

    # ---- header ----
    logger.info("=" * 120)
    logger.info(
        f"  {'Status':12s}  {'Tensor':50s}  "
        f"{'max_abs':>14s}  {'mean_abs':>14s}  {'cos_sim':>14s}  {'rel_diff':>14s}"
    )
    logger.info("-" * 120)

    num_matched = 0
    num_mismatched = 0
    first_mismatch: Optional[str] = None

    for rtp_file in rtp_files:
        basename = os.path.basename(rtp_file)
        name = basename.replace(".pt", "")

        if basename not in hf_basenames:
            logger.warning(f"  ⚠️  SKIP   {name:50s}  (no HF match)")
            continue

        rtp_tensor = torch.load(rtp_file, map_location="cpu", weights_only=True)
        hf_tensor = torch.load(
            os.path.join(hf_dump_dir, basename), map_location="cpu", weights_only=True
        )

        if rtp_tensor.shape != hf_tensor.shape:
            min_seq = min(rtp_tensor.shape[0], hf_tensor.shape[0])
            logger.warning(
                f"  ⚠️  Shape mismatch {name}: "
                f"rtp={rtp_tensor.shape} vs hf={hf_tensor.shape}, "
                f"comparing first {min_seq} tokens"
            )
            rtp_tensor = rtp_tensor[:min_seq]
            hf_tensor = hf_tensor[:min_seq]

        metrics = compute_diff_metrics(rtp_tensor, hf_tensor)
        is_bad = print_diff_report(name, metrics, threshold)

        if is_bad:
            num_mismatched += 1
            if first_mismatch is None:
                first_mismatch = name
        else:
            num_matched += 1

    logger.info("=" * 120)
    logger.info(
        f"Summary: {num_matched} matched, {num_mismatched} mismatched "
        f"(threshold={threshold})"
    )
    if first_mismatch:
        logger.info(f"First mismatch: {first_mismatch}")
    else:
        logger.info("All tensors matched! 🎉")

    return num_matched, num_mismatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-Next precision: rtp-llm vs HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace / Quark checkpoint directory",
    )
    parser.add_argument(
        "--rtp_dump_dir",
        type=str,
        required=True,
        help="rtp-llm dump directory (e.g. .../rank_0/step_0)",
    )
    parser.add_argument(
        "--hf_dump_dir",
        type=str,
        default="/tmp/hf_qwen3_next_dump",
        help="Directory to save HF dumps (default: /tmp/hf_qwen3_next_dump)",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="Hello, how are you?",
        help="Input text for forward pass",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Max abs diff threshold for flagging mismatches",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for HF model",
    )
    parser.add_argument(
        "--skip_hf_run",
        action="store_true",
        help="Skip HF model run, use existing dump in --hf_dump_dir",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Only load first N decoder layers (default: all)",
    )

    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if not args.skip_hf_run:
        run_hf_model_and_dump(
            model_path=args.model_path,
            input_text=args.input_text,
            dump_dir=args.hf_dump_dir,
            dtype=dtype_map[args.dtype],
            num_layers=args.num_layers,
        )
    else:
        logger.info(f"Skipping HF run, using existing dump at {args.hf_dump_dir}")

    logger.info("")
    logger.info("=" * 120)
    logger.info("  PRECISION COMPARISON: rtp-llm vs HuggingFace")
    logger.info("=" * 120)

    num_matched, num_mismatched = compare_dumps(
        rtp_dump_dir=args.rtp_dump_dir,
        hf_dump_dir=args.hf_dump_dir,
        threshold=args.threshold,
    )

    sys.exit(1 if num_mismatched > 0 else 0)


if __name__ == "__main__":
    main()
