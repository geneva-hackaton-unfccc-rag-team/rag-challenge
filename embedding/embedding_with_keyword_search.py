# pip install scikit-learn if you don't have it:
# uv pip install scikit-learn

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
import re

from tqdm import tqdm


def load_chunks(json_file):
    """Load a JSON file exported from your DB and return a list of text chunks."""

    def clean_text(s: str) -> str:
        # collapse whitespace and strip leading/trailing
        return re.sub(r"\s+", " ", s or "").strip()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        # depending on export, the key might be "text" or "document"
        text_val = item.get("text") or item.get("document")
        if text_val:
            documents.append(clean_text(text_val))

    return documents
    

# Download from the Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which decision addresses matters relating to the implementation of the Paris Agreement?,Decision1/CMA.1"

documents = load_chunks("../data/chunks.json")
# print(len(documents))

q_emb = model.encode_query(query)
d_emb = model.encode_document(documents)
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
