"""
Qwen2-VL logit extractor.

Uses model.forward() — NOT model.generate() — so we read raw next-token logits
before any sampling, the same measurement the Llama extractor makes.

HPC: Qwen/Qwen2-VL-72B-Instruct (136 GB bf16, 38 shards)

Two things differ from Mllama and both matter for the measurement:

1. Dynamic resolution. Qwen2-VL turns an image into a variable number of visual
   tokens depending on its pixel count, so the same prompt can cost wildly
   different context lengths across conditions. min_pixels/max_pixels are pinned
   here so every image in the study is encoded to the same token budget --
   otherwise the vision conditions would differ in sequence length as well as in
   content, and that is a confound.

2. A different tokeniser. Llama-3 splits " 1" into [space, "1"], which is why
   the Phase 2 prompt ends in a trailing space. Qwen2's BPE does not necessarily
   agree, so the trailing-space assumption must be re-validated for this model
   before trusting Phase 2 digits -- watch captured_mass, which is how both
   earlier surface-form bugs were caught.
"""

from __future__ import annotations

import sys
from typing import Literal, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.config import CFG
from src.models.hf_vision_extractor import HFVisionExtractor

Quantization = Literal["none", "4bit", "8bit"]

_MODEL_IDS = {"qwen": CFG["models"]["qwen"]}

# Qwen2-VL counts visual tokens in 28x28 patches. 256..1280 patches is Qwen's
# own documented balanced range: enough detail for a 1024x1024 stimulus without
# letting one image dominate the context.
_PATCH = 28 * 28
DEFAULT_MIN_PIXELS = 256 * _PATCH
DEFAULT_MAX_PIXELS = 1280 * _PATCH


class QwenExtractor(HFVisionExtractor):
    """
    Wraps Qwen2VLForConditionalGeneration for single-pass logit extraction.

    Args:
        variant:      "qwen" (72B).
        device:       "auto" lets accelerate shard across available hardware.
        quantization: "none" (bf16), "4bit", or "8bit".
        weights_path: local staged weights dir; overrides the hub id for
                      loading only, leaving model_id intact for provenance.
    """

    # "visual" is Qwen2-VL's vision tower; lm_head is the dependent variable.
    DEFAULT_SKIP_MODULES = ["visual", "lm_head"]

    def __init__(
        self,
        variant: str = "qwen",
        device: str = "auto",
        quantization: Quantization = "none",
        skip_modules: Optional[list[str]] = None,
        weights_path: Optional[str] = None,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        if variant not in _MODEL_IDS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_MODEL_IDS)}")
        if quantization not in ("none", "4bit", "8bit"):
            raise ValueError(f"Unknown quantization '{quantization}'. Choose from none, 4bit, 8bit.")

        model_id = _MODEL_IDS[variant]
        # model_id stays the canonical hub string: it is the provenance column
        # and the resume key in both runners.
        self.model_id = model_id
        self.weights_source = weights_path or model_id
        self.quantization = quantization
        self.skip_modules = (
            self.DEFAULT_SKIP_MODULES if skip_modules is None else skip_modules
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.maybe_enable_p2p_workaround()

        import logging
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.logging.add_handler(logging.StreamHandler(sys.stdout))

        print(f"[QwenExtractor] loading processor for {self.weights_source} ...")
        self.processor = AutoProcessor.from_pretrained(
            self.weights_source, min_pixels=min_pixels, max_pixels=max_pixels
        )
        print(f"[QwenExtractor] processor loaded "
              f"(min_pixels={min_pixels}, max_pixels={max_pixels})")

        print(f"[QwenExtractor] loading weights ({quantization}) ...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.weights_source,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **self._quant_kwargs(quantization),
        )
        self.model.eval()
        print(f"[QwenExtractor] weights loaded, device_map="
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
        Build the processor inputs expected by Qwen2VLForConditionalGeneration.

        apply_chat_template inserts Qwen's <|vision_start|><|image_pad|>
        <|vision_end|> placeholders when the content carries an image part, and
        the processor then expands the pad token to the right number of visual
        tokens for the image's resolution.
        """
        content = ([{"type": "image"}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # add_generation_prompt closes the user turn and opens an empty
        # assistant turn, so anything trailing the user message is NOT what the
        # model continues -- it starts a fresh reply. Appending the prefix here
        # puts it inside the assistant turn, making the token we want genuinely
        # next. Same fix as the Mllama path; verified there in job 27105400.
        if assistant_prefix:
            text = text + assistant_prefix

        inputs = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            return_tensors="pt",
        )
        return self._to_model_device(dict(inputs))
