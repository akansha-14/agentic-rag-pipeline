from vector_utils import text_to_vector, cosine_similarity


class VectorRetriever:
    """
    An in-memory vector retriever that indexes document chunks and retrieves
    the most relevant chunks for a given query using cosine similarity.
    """

    def __init__(self, vocabulary: list[str]):
        """
        Initializes the retriever with a predefined vocabulary.
        
        :param vocabulary: Sorted list of unique vocabulary words used for vector dimensions.
        """
        self.vocabulary = vocabulary
        self.chunks = []

    def index_chunks(self, chunks: list[dict]):
        """
        Stores document chunks and their pre-computed vector embeddings.
        
        :param chunks: List of dictionaries, each containing 'chunk_id',
                       'source_document', 'text', and 'embedding'.
        """
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float | None = None) -> list[dict]:
        """
        Vectorizes a user query, computes cosine similarity against all indexed chunks,
        ranks them from highest to lowest similarity, and returns the Top-K results.

        Safely handles:
        - Empty or whitespace-only queries -> returns empty list.
        - Empty indexed chunks -> returns empty list.
        - top_k <= 0 -> returns empty list.
        - Queries with no vocabulary overlap -> all chunks receive similarity score 0.0 safely.

        :param query: The search query string.
        :param top_k: Number of highest-ranking chunks to return.
        :param score_threshold: Optional minimum similarity score threshold.
                                If provided, only chunks with score > score_threshold are returned.
        :return: List of top_k dictionaries with 'chunk_id', 'source_document', 'text', and 'score'.
        """
        # Edge Case 1: Empty or whitespace query
        if not query or not query.strip():
            return []

        # Edge Case 2: Empty index or invalid top_k
        if not self.chunks or top_k <= 0:
            return []

        # Convert query text into a numerical vector using the established vocabulary
        query_vector = text_to_vector(query, self.vocabulary)

        scored_chunks = []
        for chunk in self.chunks:
            # Calculate cosine similarity between query vector and chunk embedding
            score = cosine_similarity(query_vector, chunk["embedding"])

            # Build result dictionary containing only the required fields
            scored_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source_document": chunk["source_document"],
                "text": chunk["text"],
                "score": score
            })

        # Rank chunks from highest to lowest similarity score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            scored_chunks = [c for c in scored_chunks if c["score"] > score_threshold]

        # Return only the Top-K items
        return scored_chunks[:top_k]


def calculate_precision_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """
    Calculates the Precision@K metric:
    Precision@K = (Number of relevant documents found in Top-K results) / K

    :param retrieved_sources: List of source document names from retrieved chunks.
    :param relevant_sources: List of ground-truth relevant document names.
    :param k: The cutoff rank.
    :return: Precision score between 0.0 and 1.0.
    """
    # Guard against invalid K and empty ground truth
    if k <= 0 or not relevant_sources:
        return 0.0

    top_k_retrieved = retrieved_sources[:k]

    # Count how many of the top-k retrieved sources match relevant documents
    relevant_count = sum(1 for source in top_k_retrieved if source in relevant_sources)

    return relevant_count / k


def calculate_reciprocal_rank(retrieved_sources: list[str], target_source: str | None) -> float:
    """
    Calculates Reciprocal Rank (RR):
    RR = 1 / (Rank position of the first relevant document retrieved)
    If the target document is not found in the retrieved list, returns 0.0.

    :param retrieved_sources: List of source document names from retrieved chunks.
    :param target_source: Name of the target document expected.
    :return: Reciprocal rank score between 0.0 and 1.0.
    """
    if not target_source:
        return 0.0

    for rank, source in enumerate(retrieved_sources, start=1):
        if source == target_source:
            return 1.0 / rank

    return 0.0
