import re
import numpy as np
from Name import fulltext_1, fulltext_2, fulltext_3
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def improved_chunk(fulltext, chunk_size=1200, overlap_sentences=2):
    """Sentence-aware chunking with sentence overlap"""

    sentences = re.split(r'(?<=[.!?])\s+', fulltext)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))

            # sentence-level overlap
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


# -------------------------------
# Chunking
# -------------------------------

text_chunks_1 = improved_chunk(fulltext_1)
text_chunks_2 = improved_chunk(fulltext_2)
text_chunks_3 = improved_chunk(fulltext_3)

# -------------------------------
# Embedding (BATCHED)
# -------------------------------

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_1 = sentence_model.encode(
    text_chunks_1,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True
)

embeddings_2 = sentence_model.encode(
    text_chunks_2,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True
)

embeddings_3 = sentence_model.encode(
    text_chunks_3,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True
)

# -------------------------------
# Normalize for cosine similarity
# -------------------------------

embeddings_1 = normalize(embeddings_1)
embeddings_2 = normalize(embeddings_2)
embeddings_3 = normalize(embeddings_3)

print(
    f"Shapes → "
    f"{embeddings_1.shape}, "
    f"{embeddings_2.shape}, "
    f"{embeddings_3.shape}"
)
