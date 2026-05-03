"""Download Llama-3.2-90B-Vision-Instruct weights to HF cache (set via HF_HOME)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from huggingface_hub import snapshot_download
from src.config import CFG

MODEL_ID = CFG["models"]["llama"]

print(f"Downloading {MODEL_ID} ...")
path = snapshot_download(MODEL_ID, max_workers=4)
print(f"Done. Cached at: {path}")
