from vector_utils import text_to_vector, cosine_similarity


class VectorRetriever:
    """
    An in-memory vector retriever that indexes document chunks and handles Top-K retrieval.
    """

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary
        self.chunks = []

    def index_chunks(self, chunks: list[dict]):
        """
        Stores document chunks with their numerical vector embeddings.
        """
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Vectorizes a query string, computes cosine similarity against all chunks,
        and returns the Top-K highest scoring chunks.
        """
        query_vector = text_to_vector(query, self.vocabulary)
        scored_chunks = []

        for chunk in self.chunks:
            score = cosine_similarity(query_vector, chunk["embedding"])
            # Create a shallow copy of chunk dict including the similarity score
            chunk_result = chunk.copy()
            chunk_result["score"] = score
            scored_chunks.append(chunk_result)

        # Sort chunks in descending order of similarity score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Return only the Top-K results
        return scored_chunks[:top_k]


def calculate_precision_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """
    Calculates Precision@K metric:
    Precision@K = (Number of relevant documents retrieved in Top-K) / K
    """
    top_k_retrieved = retrieved_sources[:k]
    # Count how many retrieved sources are present in the ground-truth relevant set
    relevant_count = sum(1 for source in top_k_retrieved if source in relevant_sources)
    return relevant_count / k


def calculate_reciprocal_rank(retrieved_sources: list[str], target_source: str) -> float:
    """
    Calculates Reciprocal Rank (RR):
    RR = 1 / (Rank position of the first relevant document retrieved)
    If not found in retrieved list, returns 0.0.
    """
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == target_source:
            return 1.0 / rank
    return 0.0
