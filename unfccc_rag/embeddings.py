# Run with: uv run python embedding_with_keyword_search.py
import gzip
import json
import logging
import os
import pickle
import re
from pathlib import Path

import click
from sentence_transformers import SentenceTransformer

from unfccc_rag.utils import load_chunks

logger = logging.getLogger(__name__)

chunks_file = "../data/correct_chunks.json"


def load_sanitized_chunks(json_file):
    """Load a JSON file exported from your DB and return a list of text chunks."""

    def clean_text(s: str) -> str:
        # collapse whitespace and strip leading/trailing
        return re.sub(r"\s+", " ", s or "").strip()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    ids = []
    for item in data:
        text_val = item.get("text") or item.get("document")
        if text_val:
            documents.append(clean_text(text_val))
        id_val = item.get("id")
        ids.append(id_val)
    return documents, ids


@click.command()
@click.option(
    "--chunks-file",
    type=str,
    required=True,
    help="Path to the chunks file.",
)
@click.option(
    "--out-file",
    type=str,
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--compressed",
    type=bool,
    default=True,
    help="Whether to compress the output file.",
)
def serialize_embeddings(chunks_file: str, out_file: str, compressed: bool) -> None:
    """Serialize embeddings to a pickle file."""
    # Download from the Hub
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model = SentenceTransformer("google/embeddinggemma-300m")

    documents, ids = load_sanitized_chunks(chunks_file)
    logger.info(f"Total chunks loaded: {len(documents)}")
    logger.info(f"Total ids loaded: {len(ids)}")

    doc_embeddings = model.encode_document(documents, show_progress_bar=True)

    chunks = load_chunks(Path(chunks_file))

    chunk_by_id = {c["id"]: c for c in chunks}

    for emb, id_ in zip(doc_embeddings, ids):
        if id_ in chunk_by_id:
            chunk_by_id[id_]["embedding"] = emb  # ndarray or list—both OK for pickle

    updated_chunks = [chunk_by_id[c["id"]] for c in chunks]
    payload = {"chunks": updated_chunks}

    if compressed or out_file.endswith(".gz"):
        with gzip.open(out_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(out_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
