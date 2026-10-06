from pathlib import Path
from vector_utils import build_vocabulary, text_to_vector
from retriever import VectorRetriever, calculate_precision_at_k, calculate_reciprocal_rank

# Define documents folder location
DOCUMENTS_DIR = Path("documents")

# Chunking configuration
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def load_documents() -> dict[str, str]:
    """Reads all text files from the documents directory."""
    if not DOCUMENTS_DIR.exists() or not DOCUMENTS_DIR.is_dir():
        print(f"Error: Directory '{DOCUMENTS_DIR}' does not exist.")
        return {}

    file_paths = list(DOCUMENTS_DIR.glob("*.txt"))
    if not file_paths:
        print("No text documents found in the 'documents' directory.")
        return {}

    documents = {}
    for file_path in sorted(file_paths):
        with open(file_path, "r", encoding="utf-8") as file:
            documents[file_path.name] = file.read()

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping chunks using a sliding window."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        step = chunk_size - chunk_overlap
        if step <= 0:
            raise ValueError("CHUNK_SIZE must be strictly greater than CHUNK_OVERLAP.")

        start += step

    return chunks


def run_pipeline():
    # Step 1: Ingest Documents & Split into Chunks
    documents = load_documents()
    if not documents:
        return

    all_chunks = []
    for doc_name, content in documents.items():
        raw_chunks = chunk_text(content)
        for index, chunk in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"{doc_name}_chunk_{index + 1}",
                "source_document": doc_name,
                "text": chunk
            })

    # Step 2: Build Vocabulary & Generate Vector Embeddings
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    vocabulary = build_vocabulary(chunk_texts)

    for chunk in all_chunks:
        chunk["embedding"] = text_to_vector(chunk["text"], vocabulary)

    # Step 3: Initialize VectorRetriever and Index Chunks
    retriever = VectorRetriever(vocabulary=vocabulary)
    retriever.index_chunks(all_chunks)

    print(f"Successfully indexed {len(all_chunks)} chunk(s) into VectorRetriever.\n")

    # Step 4: Perform Top-K Retrieval Query
    query = "How do precision and recall evaluate retrieval performance?"
    print("=" * 60)
    print(f"Top-K Retrieval Query: \"{query}\"")
    print("=" * 60 + "\n")

    top_k_results = retriever.retrieve(query, top_k=2)

    for rank, result in enumerate(top_k_results, start=1):
        print(f"Rank {rank} | Similarity Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source Document: {result['source_document']}")
        print("-" * 60)
        print(f"Content:\n\"{result['text']}\"")
        print("=" * 60 + "\n")

    # Step 5: Run Retrieval Evaluation Benchmark (Precision@K and MRR)
    print("=" * 60)
    print("Running Retrieval Evaluation Benchmark")
    print("=" * 60 + "\n")

    test_queries = [
        {
            "query": "What is semantic retrieval and keyword search?",
            "expected_source": "semantic_and_keyword_retrieval.txt"
        },
        {
            "query": "What metrics measure retrieval precision and recall?",
            "expected_source": "retrieval_evaluation_metrics.txt"
        },
        {
            "query": "How do vectors and embeddings measure geometric distance?",
            "expected_source": "embeddings_and_vector_similarity.txt"
        }
    ]

    reciprocal_ranks = []
    precision_scores = []
    K = 2

    for item in test_queries:
        q_text = item["query"]
        target_doc = item["expected_source"]

        results = retriever.retrieve(q_text, top_k=K)
        retrieved_sources = [r["source_document"] for r in results]

        p_at_k = calculate_precision_at_k(retrieved_sources, [target_doc], k=K)
        rr = calculate_reciprocal_rank(retrieved_sources, target_doc)

        precision_scores.append(p_at_k)
        reciprocal_ranks.append(rr)

        print(f"Query: \"{q_text}\"")
        print(f"Expected Target: {target_doc}")
        print(f"Retrieved Top-{K} Sources: {retrieved_sources}")
        print(f"Precision@{K}: {p_at_k:.2f} | Reciprocal Rank: {rr:.2f}")
        print("-" * 60 + "\n")

    # Calculate Mean Reciprocal Rank (MRR) and Average Precision@K
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    mean_precision = sum(precision_scores) / len(precision_scores)

    print("=" * 60)
    print("Benchmark Results Summary:")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Average Precision@{K}:       {mean_precision:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
