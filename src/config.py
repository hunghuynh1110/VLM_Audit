"""Loads config.yaml and resolves all paths relative to the project root."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = PROJECT_ROOT / val
    return cfg

CFG = load()
