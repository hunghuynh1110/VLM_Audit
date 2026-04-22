"""
Abstract base class for logit extractors.

All model-specific extractors implement extract_logits(), which returns
raw (unnormalised) logit scores for a set of target tokens.  Callers
normalise with softmax when they need probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseExtractor(ABC):
    """
    Minimal interface shared by every model extractor (Llama, Qwen, GPT-4o).

    extract_logits is intentionally token-agnostic so the same extractor
    handles all three use cases:
      • Phase 1  — target_tokens=["yes", "no"]
      • Step 3b  — target_tokens=["male", "female", "uncertain"]
      • Phase 2  — target_tokens=["1", "2", "3", "4", "5", "6", "7"]
    """

    @abstractmethod
    def extract_logits(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict[str, float]:
        """
        Run a single forward pass and return the logit for each target token.

        Args:
            prompt:        Full text prompt including any instruction suffix.
            image:         PIL image, or None for TEXT_ONLY condition.
            target_tokens: Tokens whose next-token logits to extract.
                           Each string must resolve to a single vocabulary token.

        Returns:
            dict mapping each target token to its raw (pre-softmax) logit.
            e.g. {"yes": 12.3, "no": 8.1}
        """

    @abstractmethod
    def get_attention_weights(
        self,
        prompt: str,
        image: Optional[Image.Image],
        target_tokens: list[str],
    ) -> dict:
        """
        Return cross-attention weights for the last forward pass.

        Used to generate per-token heatmaps over the image patches.
        Structure is model-specific; callers should access via the
        concrete subclass if they need the raw tensors.
        """

    def softmax_probs(self, logits: dict[str, float]) -> dict[str, float]:
        """Convert raw logits to softmax probabilities over the given tokens."""
        import math
        max_z = max(logits.values())
        exps = {tok: math.exp(z - max_z) for tok, z in logits.items()}
        total = sum(exps.values())
        return {tok: e / total for tok, e in exps.items()}
