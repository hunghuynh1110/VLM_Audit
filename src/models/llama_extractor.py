"""
Llama-3.2-Vision logit extractor.

Uses model.forward() — NOT model.generate() — so we read raw next-token
logits before any sampling, bypassing the model's safety-filtered output
distribution.

Dev (local):  meta-llama/Llama-3.2-11B-Vision-Instruct (4-bit, ~7 GB)
HPC:          meta-llama/Llama-3.2-90B-Vision-Instruct  (full / 8-bit)

Token resolution, the forward passes, the multi-GPU safety bootstrap and the
post-load self-test live in HFVisionExtractor, shared with the Qwen extractor.
"""

from __future__ import annotations

import sys
from typing import Literal, Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from src.config import CFG
from src.models.hf_vision_extractor import HFVisionExtractor

Quantization = Literal["none", "4bit", "8bit"]

_MODEL_IDS = {
    "llama_dev": CFG["models"]["llama_dev"],
    "llama":     CFG["models"]["llama"],
}


class LlamaExtractor(HFVisionExtractor):
    """
    Wraps MllamaForConditionalGeneration for single-pass logit extraction.

    Args:
        variant:        "llama_dev" (11B, local) or "llama" (90B, HPC).
        device:         "auto" lets accelerate shard across available hardware.
        quantization:   "none" (fp16), "4bit", or "8bit".  Quantised modes
                        require bitsandbytes.  4-bit ≈ 0.5 byte/param,
                        8-bit ≈ 1 byte/param — pick to fit your VRAM.
        weights_path:   local staged weights dir; overrides the hub id for
                        loading only, leaving model_id intact for provenance.
    """

    # Vision tower + projector + lm_head are excluded from quantisation by
    # default. Two reasons:
    #   1. bnb int8 crashes on Mllama's 4-D vision tensors (jobs 27087751,
    #      27086002 -- bitsandbytes/backends/cuda/ops.py lines 145 and 34).
    #   2. lm_head produces the logits we measure. Quantising it injects
    #      quantisation noise directly into the dependent variable.
    DEFAULT_SKIP_MODULES = ["vision_model", "multi_modal_projector", "lm_head"]

    def __init__(
        self,
        variant: str = "llama_dev",
        device: str = "auto",
        quantization: Quantization = "none",
        skip_modules: Optional[list[str]] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        if variant not in _MODEL_IDS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_MODEL_IDS)}")
        if quantization not in ("none", "4bit", "8bit"):
            raise ValueError(f"Unknown quantization '{quantization}'. Choose from none, 4bit, 8bit.")

        model_id = _MODEL_IDS[variant]
        # model_id stays the canonical hub string: it is the provenance column
        # and the resume key in both runners, so it must not change just
        # because the bytes came off /scratch instead of the hub cache.
        self.model_id = model_id
        # weights_path points at a staged local copy (scripts/bunya_stage_90b.sh).
        # safetensors mmap() over the NFS-mounted QRISdata cache is pathological,
        # so on Bunya we always load from GPFS scratch.
        self.weights_source = weights_path or model_id
        self.quantization = quantization
        # Pass skip_modules=[] to deliberately quantise everything.
        self.skip_modules = (
            self.DEFAULT_SKIP_MODULES if skip_modules is None else skip_modules
        )

        self.maybe_enable_p2p_workaround()

        import logging
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.logging.add_handler(logging.StreamHandler(sys.stdout))
        print(f"[LlamaExtractor] loading processor for {self.weights_source} ...")
        self.processor = AutoProcessor.from_pretrained(self.weights_source)
        print(f"[LlamaExtractor] processor loaded")

        print(f"[LlamaExtractor] loading weights ({quantization}) ...")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.weights_source,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **self._quant_kwargs(quantization),
        )
        self.model.eval()
        print(f"[LlamaExtractor] weights loaded, device_map="
              f"{getattr(self.model, 'hf_device_map', device)}")

        self._finalise_setup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        prompt: str,
        image: Optional[Image.Image],
        assistant_prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Build the processor inputs expected by MllamaForConditionalGeneration.

        The Llama-3.2-Vision processor expects the prompt wrapped in the
        instruct chat template with an <image> placeholder when an image is
        present.
        """
        content = ([{"type": "image"}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # apply_chat_template closes the user turn and opens an empty assistant
        # turn, so anything trailing the user message ("Rating: ") is NOT what
        # the model continues -- it starts a fresh reply, and the next token is
        # whatever a reply opens with. Appending the prefix here puts it inside
        # the assistant turn, making the token we want genuinely next.
        # Verified: without this, Phase 2's digits held 0.43% of the
        # distribution (job 27105400).
        if assistant_prefix:
            text = text + assistant_prefix

        inputs = self.processor(
            text=text,
            images=[image] if image is not None else None,
            return_tensors="pt",
        )
        return self._to_model_device(dict(inputs))
