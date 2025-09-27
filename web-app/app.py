# %%
import gradio as gr
import numpy as np
import gzip
import pickle
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Initialize the embedding model and load chunks
print("Loading embedding model...")
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
print("Preparing TF-IDF vectorizer...")
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


def rag_query(message: str, history):
    relevant_chunks = search_documents(message, top_k=5)
    # Format the response
    response = "**Answer:**\n\n"

    # Create a simple answer by combining top chunks
    # answer_parts = []
    # for i, (chunk, score) in enumerate(relevant_chunks[:3], 1):
    #     # Take first sentence or up to 100 characters
    #     text = chunk["text"]
    #     first_sentence = text.split(".")[0] + "." if "." in text else text[:100] + "..."
    #     answer_parts.append(first_sentence)

    # response += " ".join(answer_parts)

    # Add sources
    response += "\n\n**Sources:**\n"
    for i, (chunk, score) in enumerate(relevant_chunks, 1):
        source_file = chunk.get("source_file", "Unknown")
        text = chunk["text"]
        response += "```\n"
        text_preview = text[:150] + "..." if len(text) > 150 else text
        response += f"\n{i}. **Score: {score:.3f}** | {source_file}\n"
        response += f"Preview:\n{text_preview}\n"
        response += "```\n"

    yield response

# search_documents("awd", 5, 0.2)

# %%
demo = gr.ChatInterface(
    rag_query,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
    title="🌍 UN Climate Documents RAG System",
    description="Ask questions about UN climate change decisions and get answers with sources",
)

if __name__ == "__main__":
    demo.launch()
