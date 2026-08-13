"""
Shared machinery for HuggingFace vision-language logit extractors.

Everything here is model-agnostic: target-token resolution, the two forward
passes, the multi-GPU safety bootstrap and the post-load self-test. A concrete
extractor supplies only the two things that genuinely differ between
architectures -- how the weights are loaded, and how a (prompt, image) pair is
turned into processor inputs.

Kept separate from BaseExtractor because that interface is also implemented by
the GPT-4o extractor, which has no local weights, no processor and no logits.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from PIL import Image

from src.models.base_extractor import BaseExtractor


class HFVisionExtractor(BaseExtractor):
    """Common behaviour for locally-hosted HF vision-language models."""

    # Modules held out of quantisation. lm_head produces the logits we measure,
    # so quantising it injects noise straight into the dependent variable; the
    # vision tower is excluded because bitsandbytes' int8 kernel crashes on the
    # 4-D vision tensors. Subclasses override with their own module names.
    DEFAULT_SKIP_MODULES: list[str] = ["lm_head"]

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    @staticmethod
    def maybe_enable_p2p_workaround() -> None:
        """
        Guard against Bunya's silently-zeroed cross-GPU copies.

        Must run BEFORE from_pretrained: accelerate moves tensors between
        devices while dispatching the model. Left unpatched, any device_map
        spanning more than one GPU yields a hidden state of zeros and therefore
        an exactly uniform softmax, with no error raised.
        """
        if torch.cuda.device_count() <= 1:
            return
        from src.models import p2p_workaround
        if p2p_workaround.is_affected():
            print("[extractor] WARNING: cross-GPU copies on this node are "
                  "broken; staging them through host memory")
            p2p_workaround.enable_host_staged_cross_device_copies()

    def _quant_kwargs(self, quantization: str) -> dict:
        """BitsAndBytesConfig for the requested mode, honouring skip_modules."""
        if quantization == "none":
            return {}
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            # llm_int8_skip_modules is misleadingly named: transformers maps it
            # to modules_to_not_convert, which governs 4-bit as well as 8-bit.
            cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=self.skip_modules,
            )
        else:
            cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=self.skip_modules,
            )
        return {"quantization_config": cfg}

    def _finalise_setup(self) -> None:
        """Call at the end of a subclass __init__, once model+processor exist."""
        self._token_id_cache: dict[str, int] = {}
        self._last_attention: dict | None = None
        self._assert_not_degenerate()

    def _assert_not_degenerate(self) -> None:
        """
        One forward pass to prove the model actually computes something.

        The zeroed-cross-GPU-copy fault produced constant logits and a perfectly
        uniform softmax while raising nothing and reporting plausible aggregate
        metrics -- two full production runs before anyone noticed. Constant
        logits are never a valid state, so check for them here and fail in
        seconds rather than writing thousands of rows of noise.

        Text-only, so it costs one short forward and does not depend on images.
        """
        inputs = self._build_inputs("What colour is the sky?", None)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        logits = outputs.logits[0, -1, :].float()
        std = logits.std().item()
        n_nonfinite = int((~torch.isfinite(logits)).sum().item())
        print(f"[extractor] self-test: logits std={std:.6g} "
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
    # internal helpers
    # ------------------------------------------------------------------

    def _resolve_token_id(self, token: str) -> int:
        """
        Map a single-word token string to its vocabulary ID.

        Tries the plain form first, then the space-prefixed variants each
        tokeniser family spells differently, so the same target string works
        across models whose BPE marks a leading space as 'Ġ' or '▁'.
        """
        if token in self._token_id_cache:
            return self._token_id_cache[token]

        vocab = self.processor.tokenizer.get_vocab()

        for surface in (token, " " + token, "▁" + token, "Ġ" + token):
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

    def _to_model_device(self, inputs: dict) -> dict:
        device = next(self.model.parameters()).device
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}

    def _build_inputs(
        self,
        prompt: str,
        image: Optional[Image.Image],
        assistant_prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    def extract_logits(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict[str, float]:
        """Forward pass → raw (pre-softmax) next-token logits per target token."""
        token_ids = [self._resolve_token_id(t) for t in target_tokens]
        inputs = self._build_inputs(prompt, image)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        last_logits = outputs.logits[0, -1, :]
        self._last_attention = None  # populated only by get_attention_weights()

        return {tok: last_logits[tid].item()
                for tok, tid in zip(target_tokens, token_ids)}

    def extract_probs(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
        assistant_prefix: str = "",
    ) -> dict[str, float]:
        """
        Full-vocabulary softmax probability for each target token.

        Unlike softmax_probs(extract_logits(...)), these are comparable in
        absolute terms and can be summed across surface forms to give
        captured_mass.

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
        return {tok: probs[tid].item()
                for tok, tid in zip(target_tokens, token_ids)}

    def get_attention_weights(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict:
        """Run extract_logits and return the stored cross-attention tensors."""
        self.extract_logits(prompt, image, target_tokens)
        return {"cross_attentions": self._last_attention}
