"""
Export all Phase 1 prompts to CSV for inspection and reproducibility.
Saves to data/prompts/phase1_prompts.csv
"""

import csv
from pathlib import Path
from src.data.asi_items import get_all_prompts

OUTPUT = Path("data/prompts/phase1_prompts.csv")

def main():
    prompts = get_all_prompts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "subscale", "polarity", "structure", "condition", "prompt"])
        for r in prompts:
            writer.writerow([
                r.item_id,
                r.subscale,
                r.polarity,
                r.structure.value,
                r.condition.value,
                r.prompt,
            ])

    print(f"Saved {len(prompts)} prompts to {OUTPUT}")

if __name__ == "__main__":
    main()
