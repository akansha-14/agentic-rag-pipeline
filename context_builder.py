"""
context_builder.py
------------------
Module responsible for Milestone 4: Context Preparation.
Formats retrieved chunks into a clean, structured context string with
traceable source document names and chunk IDs.
"""

NO_CONTEXT_MESSAGE = "No relevant context found."


def format_chunk(chunk: dict) -> str:
    """
    Formats a single chunk dictionary into a traceable text block.
    
    If fields like source_document or chunk_id are missing or None,
    it marks them explicitly as 'Unknown' rather than inventing data.
    
    :param chunk: Dictionary containing 'source_document', 'chunk_id', and 'text'.
    :return: Formatted string representing the chunk, or empty string if chunk is invalid.
    """
    if not isinstance(chunk, dict):
        return ""

    # Extract source document safely
    raw_source = chunk.get("source_document")
    if raw_source is not None and str(raw_source).strip():
        source_label = str(raw_source).strip()
    else:
        source_label = "Unknown Source"

    # Extract chunk ID safely
    raw_id = chunk.get("chunk_id")
    if raw_id is not None and str(raw_id).strip():
        id_label = str(raw_id).strip()
    else:
        id_label = "Unknown Chunk ID"

    # Extract text content safely
    raw_text = chunk.get("text")
    if raw_text is not None and str(raw_text).strip():
        content = str(raw_text).strip()
    else:
        content = "[No text content]"

    header = f"[{source_label} | Chunk ID: {id_label}]"
    return f"{header}\n{content}"


def build_context(chunks: list[dict] | None, empty_fallback: str = NO_CONTEXT_MESSAGE) -> str:
    """
    Constructs a unified, clean context string from retrieved chunks.

    Preserves provenance by attaching document and chunk IDs to each entry.
    Handles empty lists, None, or invalid items gracefully.

    :param chunks: List of chunk dictionaries from VectorRetriever.
    :param empty_fallback: Fallback string returned when no valid chunks exist.
    :return: A formatted context string ready for downstream prompt augmentation.
    """
    if not chunks:
        return empty_fallback

    formatted_entries = []
    for chunk in chunks:
        entry = format_chunk(chunk)
        if entry:
            formatted_entries.append(entry)

    if not formatted_entries:
        return empty_fallback

    # Join multiple chunks with a clear visual separator
    separator = "\n\n" + "-" * 50 + "\n\n"
    return separator.join(formatted_entries)
