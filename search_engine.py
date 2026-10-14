TOP_K = 5
MIN_SIMILARITY = 0.30

SEARCH_SQL = """
    SELECT source, page, chunk_index, content,
           1 - (embedding <=> %(query)s) AS similarity
    FROM chunks
    ORDER BY embedding <=> %(query)s
    LIMIT %(limit)s
"""


def search(conn, model, query, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    query_embedding = model.encode(query)
    rows = conn.execute(
        SEARCH_SQL, {"query": query_embedding, "limit": top_k}
    ).fetchall()

    return [
        {
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "content": content,
            "similarity": round(float(similarity), 3),
        }
        for source, page, chunk_index, content, similarity in rows
        if similarity >= min_similarity
    ]