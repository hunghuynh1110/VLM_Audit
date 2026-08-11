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

        # Must happen BEFORE from_pretrained: accelerate moves tensors between
        # devices while dispatching the model, and on Bunya a direct cuda->cuda
        # copy silently delivers zeros. Left unpatched, any device_map spanning
        # more than one GPU yields a hidden state of zeros and therefore an
        # exactly uniform softmax, with no error raised. See
        # src/models/p2p_workaround.py.
        if torch.cuda.device_count() > 1:
            from src.models import p2p_workaround
            if p2p_workaround.is_affected():
                print("[LlamaExtractor] WARNING: cross-GPU copies on this node are "
                      "broken; staging them through host memory")
                p2p_workaround.enable_host_staged_cross_device_copies()

        import logging
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.logging.add_handler(logging.StreamHandler(sys.stdout))
        print(f"[LlamaExtractor] loading processor for {self.weights_source} ...")
        self.processor = AutoProcessor.from_pretrained(self.weights_source)
        print(f"[LlamaExtractor] processor loaded")

        quant_kwargs: dict = {}
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            # llm_int8_skip_modules is misleadingly named: transformers maps it
            # to modules_to_not_convert, which governs 4-bit as well as 8-bit.
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=self.skip_modules,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=self.skip_modules,
            )

        print(f"[LlamaExtractor] loading weights ({quantization}) ...")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.weights_source,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **quant_kwargs,
        )
        self.model.eval()
        print(f"[LlamaExtractor] weights loaded, device_map={self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else device}")

        # Cache token IDs to avoid repeated tokeniser lookups
        self._token_id_cache: dict[str, int] = {}
        # Storage for the most recent cross-attention output
        self._last_attention: dict | None = None

        self._assert_not_degenerate()

    def _assert_not_degenerate(self) -> None:
        """
        One forward pass to prove the model actually computes something.

        The zeroed-cross-GPU-copy fault produced constant logits and a perfectly
        uniform softmax while raising nothing and reporting plausible aggregate
        metrics -- two full production runs, 90 minutes of H100 time, and a long
        investigation before anyone noticed. Constant logits are never a valid
        state, so check for them here and fail in seconds rather than writing
        thousands of rows of noise.

        Text-only, so it costs one short forward and does not depend on images.
        """
        inputs = self._build_inputs("What colour is the sky?", None)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        logits = outputs.logits[0, -1, :].float()
        std = logits.std().item()
        n_nonfinite = int((~torch.isfinite(logits)).sum().item())
        print(f"[LlamaExtractor] self-test: logits std={std:.6g} "
              f"non-finite={n_nonfinite}")

        if n_nonfinite or std < 1e-4:
            raise RuntimeError(
                f"Model self-test failed: next-token logits are degenerate "
                f"(std={std:.6g}, non-finite={n_nonfinite}). A std of 0 means "
                f"constant logits, i.e. a uniform softmax over the whole "
                f"vocabulary, and the measurement would be meaningless. On "
                f"Bunya the usual cause is silently zeroed cross-GPU copies -- "
                f"see src/models/p2p_workaround.py."
            )

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

    def extract_probs(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
        assistant_prefix: str = "",
    ) -> dict[str, float]:
        """
        Full-vocabulary softmax probability for each target token.

        Unlike softmax_probs(extract_logits(...)), these probabilities are
        comparable in absolute terms and can be summed across surface forms.

        assistant_prefix is inserted at the start of the model's own turn, so
        the target tokens are genuinely the next token rather than the opening
        word of a fresh reply.
        """
        token_ids = [self._resolve_token_id(t) for t in target_tokens]
        inputs = self._build_inputs(prompt, image, assistant_prefix)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        last_logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(last_logits, dim=-1)
        return {tok: probs[tid].item() for tok, tid in zip(target_tokens, token_ids)}

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
