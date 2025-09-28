import gzip
import json
import pickle
from pathlib import Path


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from chunks.json."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pickled_chunks(pickle_file: Path):
    """Load chunks from pickle file."""
    with gzip.open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data["chunks"]


def minmax_normalize(x):
    """Min-max normalization."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)
