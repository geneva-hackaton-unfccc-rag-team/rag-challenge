# %%
import gzip
import logging
import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from pydantic import Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parent.parent

# Initialize the embedding model and load chunks
logger.info("Loading embedding model...")
model = SentenceTransformer("google/embeddinggemma-300m")


def load_chunks(pickle_file: Path):
    """Load chunks from pickle file."""
    with gzip.open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data["chunks"]


# Load chunks and prepare search
chunks_file = REPO_ROOT / "data/data.pkl.gz"
chunks = load_chunks(chunks_file)
chunks_text = [chunk["text"] for chunk in chunks]

# Prepare TF-IDF vectorizer
logger.info("Preparing TF-IDF vectorizer...")
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(chunks_text)

# Get the embeddings for all chunks
d_emb = torch.tensor(np.array([chunk.pop("embedding") for chunk in chunks]))


# %%
def minmax_normalize(x):
    """Min-max normalization."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def search_documents(query: str, top_k: int = 5, alpha: float = 0.9):
    """Search for relevant documents using hybrid approach."""
    # Embedding similarity
    q_emb = model.encode_query(query)
    embed_sims = model.similarity(q_emb, d_emb).numpy().squeeze()

    # TF-IDF similarity
    q_tfidf = tfidf.transform([query])
    tfidf_sims = cosine_similarity(q_tfidf, tfidf_matrix).squeeze()
    # Combine scores
    # print(embed_sims.shape)
    # return
    embed_s = minmax_normalize(embed_sims)
    tfidf_s = minmax_normalize(tfidf_sims)
    scores = alpha * embed_s + (1 - alpha) * tfidf_s
    # scores = embed_sims  # Using only embedding similarity for simplicity
    # scores = tfidf_sims

    # Get top results
    top_indices = np.argsort(-scores)[:top_k]

    # Return top chunks with their scores
    return [(chunks[i], scores[i]) for i in top_indices]


def unfccc_rag_query(
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


mcp = FastMCP(
    name="rag-mcp",
    instructions="This server provides a tool to retrieve documents from the UNFCCC climate change negotiations.",
    tools=[Tool.from_function(unfccc_rag_query)],
)


def start():
    # Get fresh config to ensure we have latest environment variables
    logger.info("Starting UNFCCC RAG MCP Server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    start()
