import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Callable

import numpy as np
import torch
from pydantic import Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from unfccc_rag.utils import load_chunks, minmax_normalize

logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

torch_device = torch.device("cpu")
if sys.platform == "darwin" and torch.backends.mps.is_available():
    torch_device = torch.device("mps")
else:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")


def load_retrieval_engine(
    embeddings_file: Path,
    model_name: str = "google/embeddinggemma-300m",
    top_k: int = 5,
    alpha: float = 0.9,
    silent: bool = False,
) -> Callable[[str], dict[str, Any]]:
    """Load the retrieval engine.

    Args:
        embeddings_file (Path): The path to the embeddings file.
        model_name (str, optional): The name of the embedding model to use.
            Defaults to "google/embeddinggemma-300m".
        top_k (int, optional): The number of top chunks to retrieve.
            Defaults to 5.
        alpha (float, optional): The weight of the embedding similarity compared to the TF-IDF similarity.
            Allows to have an hybrid approach balancing between semantic and lexical similarity.
            - alpha = 1 -> only semantic similarity
            - alpha = 0 -> only lexical similarity
            - alpha = 0.5 -> balanced approach
            Defaults to 0.9.
        silent (bool, optional): Whether to suppress the logging.
            Defaults to False.

    Returns:
        Callable[[str], str]: A callable that takes a query and returns a string
            containing the document chunks and their respective proximity scores.
    """
    # Initialize the embedding model and load chunks
    if not silent:
        logger.info("Loading embedding model...")
    model = SentenceTransformer(model_name)

    # Load chunks and prepare search
    chunks = load_chunks(embeddings_file)
    chunks_text = [chunk["text"] for chunk in chunks.values()]
    chunk_ids = list(chunks.keys())

    # Prepare TF-IDF vectorizer
    if not silent:
        logger.info("Preparing TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(chunks_text)

    # Get the embeddings for all chunks
    d_emb = torch.tensor(
        np.array(
            [chunk.pop("embedding") for chunk in chunks.values()],
            dtype=np.float32,
        ),
        device=torch_device,
    )

    def search_documents(
        query: Annotated[
            str, Field(description="The query used to retrieve relevant chunks from.")
        ],
    ) -> Annotated[
        dict[str, Any],
        Field(description="Document chunks sorted by similarity score."),
    ]:
        """Retrieves relevant chunks from the UNFCCC climate change negotiations documents."""
        # Embedding similarity
        q_emb = model.encode_query(
            sentences=query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            precision="float32",
            show_progress_bar=False,
            device=torch_device,  # type: ignore
        )
        embed_sims = model.similarity(q_emb, d_emb).to("cpu").numpy().squeeze()

        # TF-IDF similarity
        q_tfidf = tfidf.transform([query])
        tfidf_sims = cosine_similarity(q_tfidf, tfidf_matrix).squeeze()

        embed_s = minmax_normalize(embed_sims)
        tfidf_s = minmax_normalize(tfidf_sims)
        scores = alpha * embed_s + (1 - alpha) * tfidf_s

        # Get top results
        top_indices = np.argsort(-scores)[:top_k]

        wanted_keys = ["text", "file_name"]
        top_chunks = {
            chunk_ids[i]: dict(
                filter(lambda kv: kv[0] in wanted_keys, chunks[chunk_ids[i]].items())
            )
            | {"similarity_score": round(float(scores[i]), 3)}
            for i in top_indices
        }

        # Return top chunks with their scores
        return top_chunks

    return search_documents
