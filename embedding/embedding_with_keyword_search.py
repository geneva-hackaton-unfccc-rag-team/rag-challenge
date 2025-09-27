# Run with: uv run python embedding_with_keyword_search.py 
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
import re
from tqdm import tqdm
import pickle

import json
import pickle
import gzip


chunks_file = "../data/correct_chunks.json"

def load_chunks(json_file):
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
    
# Download from the Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which decision addresses matters relating to the implementation of the Paris Agreement?,Decision1/CMA.1"
q_emb = model.encode_query(query)

documents,ids = load_chunks(chunks_file)
print("Total chunks loaded: ", len(documents))
print("Total ids loaded: ", len(ids))

doc_embeddings = model.encode_document(documents,show_progress_bar=True)
# print(doc_embeddings)

def save_embeddings_pkl(
    chunks_file: str,
    doc_embeddings,
    ids,
    out_file: str = "chunks_with_embeddings.pkl",
    compressed: bool = False,
):
    """
    Save chunks + embeddings to a pickle file.
    Keeps embeddings as NumPy arrays (pickle handles them fine).
    """
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

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


def load_embeddings_pkl(path: str):
    """Load the payload back (works for .pkl or .pkl.gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)

save_embeddings_pkl(chunks_file, doc_embeddings, ids, "data.pkl.gz", compressed=True)  # gzipped
# data = load_embeddings_pkl("data.pkl.gz")
         
print(f"Embeddings saved in: {"data.pkl.gz"}")
breakpoint()





































# Example call right after you print results:
# save_results_pkl("results.pkl", rank, hybrid, embed_s, tfidf_s, documents, source_files=None, top_k=50)



embed_sims = model.similarity(q_emb, d_emb).cpu().numpy().ravel()  # shape: (4,)

# ---- Simple keyword search via TF-IDF ----
# bi-grams help with phrases like "Red Planet"
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
D = tfidf.fit_transform(documents)
q = tfidf.transform([query])
tfidf_sims = cosine_similarity(q, D).ravel()  # shape: (4,)

# ---- Combine (min-max normalize then weighted sum) ----
def minmax(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-8)

embed_s = minmax(embed_sims)
tfidf_s = minmax(tfidf_sims)

alpha = 0.9  # tune: 0.3→favor keywords, 0.7→favor embeddings
hybrid = alpha * embed_s + (1 - alpha) * tfidf_s

# ---- Rank and show ----
rank = np.argsort(-hybrid)
for i, idx in enumerate(rank, 1):
    print(f"{i}. score={hybrid[idx]:.3f}  |  embed={embed_s[idx]:.3f}  |  tfidf={tfidf_s[idx]:.3f}")
    print(f"   {documents[idx]}")

