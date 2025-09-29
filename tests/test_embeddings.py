from sentence_transformers import SentenceTransformer


def test_embedding_ranks_mars_highest() -> None:
    model = SentenceTransformer("google/embeddinggemma-300m")

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    query_embeddings = model.encode_query(query)
    document_embeddings = model.encode_document(documents)

    # Compute similarities to determine a ranking
    similarities = model.similarity(query_embeddings, document_embeddings)

    # similarities is expected to be shape (1, N). Get top index robustly.
    top_idx = int(similarities.reshape(-1).argmax().item())

    # The Mars document should be the most similar to the Red Planet query
    assert top_idx == 1
