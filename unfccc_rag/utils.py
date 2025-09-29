import gzip
import json
import logging
import pickle
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_chunks(path: str | Path) -> dict[str, Any]:
    """Load chunks from chunks.json."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Chunks data is not a dictionary")
    return data


def save_chunks(
    chunks: dict[str, Any],
    output_path: str | Path,
    ensure_ascii: bool = True,
):
    """Save chunks to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=ensure_ascii)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_pickled_chunks(pickle_file: Path):
    """Load chunks from pickle file."""
    with gzip.open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data["chunks"]


def minmax_normalize(x):
    """Min-max normalization."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def clean_text(s: str) -> str:
    """Collapse whitespace and strip leading/trailing."""
    return re.sub(r"\s+", " ", s or "").strip()


def get_chunk_id() -> str:
    """Get a unique chunk ID."""
    return str(uuid.uuid4())
