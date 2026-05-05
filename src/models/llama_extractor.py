"""
Llama-3.2-Vision logit extractor.

Uses model.forward() — NOT model.generate() — so we read raw next-token
logits before any sampling, bypassing the model's safety-filtered output
distribution.

Dev (local):  meta-llama/Llama-3.2-11B-Vision-Instruct (4-bit, ~7 GB)
HPC:          meta-llama/Llama-3.2-90B-Vision-Instruct  (full / 8-bit)
"""

from __future__ import annotations

import sys
import warnings
from typing import Literal, Optional

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from src.config import CFG
from src.models.base_extractor import BaseExtractor

Quantization = Literal["none", "4bit", "8bit"]

_MODEL_IDS = {
    "llama_dev": CFG["models"]["llama_dev"],
    "llama":     CFG["models"]["llama"],
}


class LlamaExtractor(BaseExtractor):
    """
    Wraps MllamaForConditionalGeneration for single-pass logit extraction.

    Args:
        variant:        "llama_dev" (11B, local) or "llama" (90B, HPC).
        device:         "auto" lets accelerate shard across available hardware.
        quantization:   "none" (fp16), "4bit", or "8bit".  Quantised modes
                        require bitsandbytes.  4-bit ≈ 0.5 byte/param,
                        8-bit ≈ 1 byte/param — pick to fit your VRAM.
    """

    def __init__(
        self,
        variant: str = "llama_dev",
        device: str = "auto",
        quantization: Quantization = "none",
    ) -> None:
        if variant not in _MODEL_IDS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_MODEL_IDS)}")
        if quantization not in ("none", "4bit", "8bit"):
            raise ValueError(f"Unknown quantization '{quantization}'. Choose from none, 4bit, 8bit.")

        model_id = _MODEL_IDS[variant]
        self.model_id = model_id
        self.quantization = quantization

        import logging
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.logging.add_handler(logging.StreamHandler(sys.stdout))
        print(f"[LlamaExtractor] loading processor for {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[LlamaExtractor] processor loaded")

        quant_kwargs: dict = {}
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        print(f"[LlamaExtractor] loading weights ({quantization}) ...")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            **quant_kwargs,
        )
        self.model.eval()
        print(f"[LlamaExtractor] weights loaded, device_map={self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else device}")

        # Cache token IDs to avoid repeated tokeniser lookups
        self._token_id_cache: dict[str, int] = {}
        # Storage for the most recent cross-attention output
        self._last_attention: dict | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_token_id(self, token: str) -> int:
        """
        Map a single-word token string to its vocabulary ID.

        Strips leading whitespace variants and picks the token ID that the
        model actually uses at the start of a continuation (i.e. ' yes' not
        'yes' when the tokeniser adds a space prefix).
        """
        if token in self._token_id_cache:
            return self._token_id_cache[token]

        vocab = self.processor.tokenizer.get_vocab()

        # Try common surface forms: plain, space-prefixed (Ġ / ▁ style)
        candidates = [token, " " + token, "▁" + token, "Ġ" + token]
        for surface in candidates:
            if surface in vocab:
                tid = vocab[surface]
                self._token_id_cache[token] = tid
                return tid

        # Fall back: encode as a continuation and take the first token ID
        ids = self.processor.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            warnings.warn(
                f"Token '{token}' encodes to {len(ids)} sub-tokens; "
                "using the first. Consider using single-token target strings."
            )
        tid = ids[0]
        self._token_id_cache[token] = tid
        return tid

    def _build_inputs(
        self, prompt: str, image: Optional[Image.Image]
    ) -> dict[str, torch.Tensor]:
        """
        Build the processor inputs expected by MllamaForConditionalGeneration.

        The Llama-3.2-Vision processor expects the prompt wrapped in the
        instruct chat template with an <image> placeholder when an image is
        present.
        """
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image] if image is not None else None,
            return_tensors="pt",
        )
        # Move all tensors to the model's device
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    def extract_logits(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict[str, float]:
        """
        Forward pass → next-token logits for each target token.

        Returns raw (pre-softmax) logits.  Use self.softmax_probs() to
        convert to probabilities.
        """
        token_ids = [self._resolve_token_id(t) for t in target_tokens]
        inputs = self._build_inputs(prompt, image)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # outputs.logits: (batch=1, seq_len, vocab_size)
        # We want the logits at the last input position (next-token prediction)
        last_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
        self._last_attention = None  # populated only by get_attention_weights()

        return {
            tok: last_logits[tid].item()
            for tok, tid in zip(target_tokens, token_ids)
        }

    def get_attention_weights(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict:
        """
        Run extract_logits and return the cross-attention tensors.

        Returns a dict with key "cross_attentions" — a tuple of per-layer
        tensors (batch, heads, text_len, image_patches).
        """
        self.extract_logits(prompt, image, target_tokens)
        return {"cross_attentions": self._last_attention}
