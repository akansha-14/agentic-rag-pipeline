import math
import re


def tokenize(text: str) -> list[str]:
    """
    Converts text to lowercase and extracts words using regular expressions.
    """
    return re.findall(r"\w+", text.lower())


def build_vocabulary(texts: list[str]) -> list[str]:
    """
    Collects a sorted list of unique words across all input texts to form the vector dimensions.
    """
    vocab = set()
    for text in texts:
        words = tokenize(text)
        vocab.update(words)
    return sorted(list(vocab))


def text_to_vector(text: str, vocabulary: list[str]) -> list[float]:
    """
    Converts a text string into a numerical Term Frequency (TF) vector based on the vocabulary.
    
    Each index in the returned vector corresponds to a word in the vocabulary,
    and its value represents how many times that word appears in the text.
    """
    words = tokenize(text)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Construct the vector: count for each vocabulary word
    vector = [float(word_counts.get(vocab_word, 0)) for vocab_word in vocabulary]
    return vector


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculates Cosine Similarity between two numerical vectors.
    Formula: Cosine Similarity = (vec1 . vec2) / (||vec1|| * ||vec2||)
    
    Returns a score between 0.0 (orthogonal/no overlap) and 1.0 (identical direction).
    """
    # Dot product: sum of element-wise multiplication
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

    # Magnitude (Euclidean norm) of vector 1
    magnitude1 = math.sqrt(sum(v1**2 for v1 in vec1))

    # Magnitude (Euclidean norm) of vector 2
    magnitude2 = math.sqrt(sum(v2**2 for v2 in vec2))

    # Avoid division by zero if a vector is empty (all zeros)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
