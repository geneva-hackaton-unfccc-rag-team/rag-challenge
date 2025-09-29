import logging
import sys
from pathlib import Path
from typing import Annotated, Callable

import numpy as np
import torch
from pydantic import Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from unfccc_rag.utils import load_chunks, minmax_normalize

logger = logging.getLogger(__name__)

torch_device = torch.device("cpu")
if sys.platform == "darwin" and torch.backends.mps.is_available():
    torch_device = torch.device("mps")
else:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")


def load_retrieval_engine(
    embeddings_file: Path,
    model_name: str = "google/embeddinggemma-300m",
) -> Callable[[str], str]:
    """Load the retrieval engine.

    Args:
        embeddings_file (Path): The path to the embeddings file.
        model_name (str, optional): The name of the embedding model to use.
            Defaults to "google/embeddinggemma-300m".

    Returns:
        Callable[[str], str]: A callable that takes a query and returns a string
            containing the document chunks and their respective proximity scores.
    """
    # Initialize the embedding model and load chunks
    logger.info("Loading embedding model...")
    model = SentenceTransformer(model_name)

    # Load chunks and prepare search
    chunks = load_chunks(embeddings_file)
    chunks_text = [chunk["text"] for chunk in chunks.values()]
    chunk_ids = list(chunks.keys())

    # Prepare TF-IDF vectorizer
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

    def search_documents(query: str, top_k: int = 5, alpha: float = 0.9):
        """Search for relevant documents using hybrid approach."""
        # Embedding similarity
        q_emb = model.encode_query(
            sentences=query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            precision="float32",
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

        # Return top chunks with their scores
        return [(chunks[chunk_ids[i]], scores[i]) for i in top_indices]

    def rag_query(
        user_query: Annotated[
            str, Field(description="The query used to retrieve relevant chunks from.")
        ],
    ) -> Annotated[
        str,
        Field(
            description="A string containing the document chunks and their respective proximity scores."
        ),
    ]:
        """Retrieves relevant chunks from the UNFCCC climate change negotiations documents."""
        relevant_chunks = search_documents(user_query, top_k=5)
        # Format the response
        response = "# Sources:"
        # Add sources
        for i, (chunk, score) in enumerate(relevant_chunks):
            source_file = chunk.get("source_file", "Unknown")
            text = chunk["text"]
            response += f"""
```
Id: {i}
Score: {score:.3f}
Source file: {source_file}
Content: {text}
```
"""
        return response

    return rag_query
