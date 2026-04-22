"""
Empirical validation of the humanoid silhouette stimulus.

Probes Llama-3.2-Vision with two questions:
  1. Gender — "male", "female", "uncertain"
  2. Race   — "white", "black", "asian", "uncertain"

Decision thresholds (per PIPELINE_SPEC §silhouette-validation):
  • Gender: |P(male) - P(female)| > 0.2  → flag
  • Race:   any single race token P > 0.5 → flag

Output: outputs/validation/silhouette_validation.json

Usage:
    .venv/bin/python3 scripts/validate_silhouette.py
    .venv/bin/python3 scripts/validate_silhouette.py --load-in-4bit
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import CFG
from src.data.image_generators import generate_humanoid_silhouette
from src.models.llama_extractor import LlamaExtractor

OUTPUT_PATH = CFG["paths"]["outputs"] / "validation" / "silhouette_validation.json"

GENDER_TOKENS = ["male", "female", "uncertain"]
RACE_TOKENS   = ["white", "black", "asian", "uncertain"]

GENDER_PROMPT = (
    "What gender does this person appear to be? "
    "Answer with one word: male, female, or uncertain."
)
RACE_PROMPT = (
    "What race does this person appear to be? "
    "Answer with one word: white, black, asian, or uncertain."
)

GENDER_THRESHOLD = 0.2   # |P(male) - P(female)| > this → flag
RACE_THRESHOLD   = 0.5   # any single race > this → flag


def _flag_gender(probs: dict[str, float]) -> bool:
    return abs(probs["male"] - probs["female"]) > GENDER_THRESHOLD


def _flag_race(probs: dict[str, float]) -> bool:
    race_tokens = [t for t in RACE_TOKENS if t != "uncertain"]
    return any(probs[t] > RACE_THRESHOLD for t in race_tokens)


def run_validation(extractor: LlamaExtractor) -> dict:
    silhouette = generate_humanoid_silhouette()

    gender_logits = extractor.extract_logits(GENDER_PROMPT, silhouette, GENDER_TOKENS)
    gender_probs  = extractor.softmax_probs(gender_logits)

    race_logits = extractor.extract_logits(RACE_PROMPT, silhouette, RACE_TOKENS)
    race_probs  = extractor.softmax_probs(race_logits)

    gender_flagged = _flag_gender(gender_probs)
    race_flagged   = _flag_race(race_probs)

    result = {
        "gender": {
            "logits":       gender_logits,
            "probs":        gender_probs,
            "flagged":      gender_flagged,
            "threshold":    GENDER_THRESHOLD,
            "metric":       abs(gender_probs["male"] - gender_probs["female"]),
        },
        "race": {
            "logits":    race_logits,
            "probs":     race_probs,
            "flagged":   race_flagged,
            "threshold": RACE_THRESHOLD,
            "metric":    max(race_probs[t] for t in RACE_TOKENS if t != "uncertain"),
        },
        "overall_pass": not gender_flagged and not race_flagged,
        "recommendation": (
            "Keep SILHOUETTE condition"
            if not gender_flagged and not race_flagged
            else "Switch to GRAY_PATCH — silhouette encodes detectable demographic signal"
        ),
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate humanoid silhouette stimulus neutrality")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (requires bitsandbytes; recommended for local Mac run)")
    args = parser.parse_args()

    print("Loading Llama-3.2-11B-Vision-Instruct …")
    extractor = LlamaExtractor(variant="llama_dev", load_in_4bit=args.load_in_4bit)

    print("Running gender probe …")
    print("Running race probe …")
    result = run_validation(extractor)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"\nGender  flagged: {result['gender']['flagged']}  "
          f"(|P(male)-P(female)| = {result['gender']['metric']:.3f})")
    print(f"Race    flagged: {result['race']['flagged']}  "
          f"(max race P = {result['race']['metric']:.3f})")
    print(f"\n{'✅' if result['overall_pass'] else '❌'}  {result['recommendation']}")


if __name__ == "__main__":
    main()
