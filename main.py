from pathlib import Path
from vector_utils import build_vocabulary, text_to_vector
from retriever import VectorRetriever, calculate_precision_at_k, calculate_reciprocal_rank
from context_builder import build_context

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

    print(f"Successfully indexed {len(all_chunks)} chunk(s) into VectorRetriever.")
    print(f"Vocabulary Size: {len(vocabulary)} unique terms across corpus.\n")

    # Step 4: Test 1 — Known Query Matching Relevant Documents
    test_query_1 = "How do precision and recall evaluate retrieval performance?"
    print("=" * 70)
    print(f"Test 1: Known Matching Query -> \"{test_query_1}\"")
    print("=" * 70)

    top_k_results_1 = retriever.retrieve(test_query_1, top_k=2)

    for rank, result in enumerate(top_k_results_1, start=1):
        print(f"Rank {rank} | Similarity Score: {result['score']:.4f}")
        print(f"Chunk ID:        {result['chunk_id']}")
        print(f"Source Document: {result['source_document']}")
        print(f"Content:         \"{result['text'].strip()}\"")
        print("-" * 70)

    # Step 5: Demonstrate Context Preparation (Milestone 4)
    print("=" * 70)
    print("Context Preparation (Milestone 4: Formatted Traceable Context)")
    print("=" * 70)
    prepared_context_1 = build_context(top_k_results_1)
    print(prepared_context_1)
    print("=" * 70 + "\n")

    # Step 6: Test 2 — Query with Zero Vocabulary Overlap
    # Note: Words like 'in' or 'and' exist in our corpus. To test true zero overlap,
    # we choose words that have 0 occurrences in any document or vocabulary.
    test_query_2 = "xylophone quantum dinosaur pterodactyl"
    print("=" * 70)
    print(f"Test 2: Zero-Vocabulary Overlap Query -> \"{test_query_2}\"")
    print("=" * 70)

    # Without threshold: cosine similarity safely produces 0.0000 without crashing
    results_no_threshold = retriever.retrieve(test_query_2, top_k=2)
    print("Results without threshold (all similarity scores evaluate to 0.0000):")
    for rank, result in enumerate(results_no_threshold, start=1):
        print(f"  Rank {rank} | Score: {result['score']:.4f} | Chunk ID: {result['chunk_id']}")

    # With score_threshold=0.0: discards irrelevant zero-similarity chunks
    results_with_threshold = retriever.retrieve(test_query_2, top_k=2, score_threshold=0.0)
    print(f"\nResults with score_threshold=0.0: {len(results_with_threshold)} chunks (Safely filtered out zero-score results)")
    
    # Demonstrate context preparation on empty retrieval results
    prepared_empty_context = build_context(results_with_threshold)
    print(f"Prepared Context for Zero-Match Query:\n-> \"{prepared_empty_context}\"")
    print("-" * 70 + "\n")

    # Step 6: Test 3 — Edge Case Safety (Empty Query & Empty Index)
    print("=" * 70)
    print("Test 3: Edge Case Safety (Empty Query & Empty Index)")
    print("=" * 70)
    empty_query_results = retriever.retrieve("   ", top_k=2)
    print(f"Query with whitespace-only '   ' -> Returned {len(empty_query_results)} results (Safe)")

    empty_retriever = VectorRetriever(vocabulary=vocabulary)
    empty_index_results = empty_retriever.retrieve("retrieval evaluation", top_k=2)
    print(f"Query against empty index       -> Returned {len(empty_index_results)} results (Safe)")
    print("-" * 70 + "\n")

    # Step 7: Retrieval Evaluation Benchmark (Precision@K and MRR)
    print("=" * 70)
    print("Running Retrieval Evaluation Benchmark Suite")
    print("=" * 70 + "\n")

    benchmark_queries = [
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
        },
        {
            "query": "xylophone quantum dinosaur pterodactyl",
            "expected_source": None  # Negative control: no vocabulary match
        }
    ]

    reciprocal_ranks = []
    precision_scores = []
    K = 2

    for item in benchmark_queries:
        q_text = item["query"]
        target_doc = item["expected_source"]

        # Retrieve top K with a threshold to only select matching chunks
        results = retriever.retrieve(q_text, top_k=K, score_threshold=0.0)
        retrieved_sources = [r["source_document"] for r in results]

        # Calculate metrics
        relevant_list = [target_doc] if target_doc else []
        p_at_k = calculate_precision_at_k(retrieved_sources, relevant_list, k=K)
        rr = calculate_reciprocal_rank(retrieved_sources, target_doc)

        # For queries with an expected target, track benchmark statistics
        if target_doc is not None:
            precision_scores.append(p_at_k)
            reciprocal_ranks.append(rr)

        print(f"Query:                  \"{q_text}\"")
        print(f"Expected Target:        {target_doc if target_doc else 'None (Out of vocabulary)'}")
        print(f"Retrieved Top-{K} Docs:   {retrieved_sources if retrieved_sources else '[] (No matching chunks)'}")
        print(f"Precision@{K}:           {p_at_k:.2f} | Reciprocal Rank: {rr:.2f}")
        print("-" * 70)

    # Calculate Mean Reciprocal Rank (MRR) and Average Precision@K across target queries
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    mean_precision = sum(precision_scores) / len(precision_scores)

    print("\n" + "=" * 70)
    print("Benchmark Results Summary (for Target Queries):")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f} (1.0 = target always ranked #1)")
    print(f"Average Precision@{K}:       {mean_precision:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
