# Run with: uv run python embedding_with_keyword_search.py
import logging
import os
from itertools import islice
from pathlib import Path
from typing import Any

import click
import numpy as np
from sentence_transformers import SentenceTransformer

from unfccc_rag.utils import clean_text, load_chunks, save_chunks

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--chunks-file",
    type=Path,
    required=True,
    help="Path to the chunks file.",
)
@click.option(
    "--output-file",
    type=Path,
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--model-name",
    type=str,
    default="google/embeddinggemma-300m",
    help="Name of the model to use for the embeddings.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="The batch size to use for the embeddings.",
)
@click.option(
    "--normalize-embeddings",
    type=bool,
    default=True,
    help="Whether to normalize the embeddings.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite the output file if it already exists.",
)
@click.option(
    "--torch-device",
    type=str,
    default=None,
    help="The device to use to compute the embeddings.",
)
def serialize_embeddings(
    chunks_file: Path,
    output_file: Path,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    overwrite: bool,
    torch_device: str | None,
) -> None:
    """Serialize embeddings to a pickle file."""
    if output_file.exists() and not overwrite:
        raise click.ClickException(
            f"Output file {output_file} already exists. Use --overwrite to overwrite."
        )

    # Download from the Hub
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model = SentenceTransformer(model_name)
    chunks = dict(islice(load_chunks(chunks_file).items(), 50))

    def get_sanitized_chunks(chunks: dict[str, Any]) -> list[str]:
        return [clean_text(c["text"]) for c in chunks.values()]

    doc_embeddings = model.encode_document(
        sentences=get_sanitized_chunks(chunks),
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        device=torch_device,
    )

    for doc_embedding, chunk_id in zip(doc_embeddings, chunks, strict=True):
        assert not np.isnan(doc_embedding).any(), "NaN embeddings found"
        chunks[chunk_id]["embedding"] = doc_embedding.tolist()
        chunks[chunk_id]["embedding_model"] = model_name

    save_chunks(chunks, output_file)
