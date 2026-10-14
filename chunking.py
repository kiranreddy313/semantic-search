import re

CHUNK_SIZE_WORDS = 120
OVERLAP_WORDS = 20


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    text = clean_text(text)
    if not text:
        return []

    words = text.split(" ")
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks