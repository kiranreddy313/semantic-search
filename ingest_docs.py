from pathlib import Path

from chunking import chunk_text
from common import EMBEDDING_DIMENSIONS, get_connection, load_model
from loaders import SUPPORTED_EXTENSIONS, load_document

DOCS_FOLDER = Path("docs")

records = []
for path in sorted(DOCS_FOLDER.iterdir()):
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        continue

    file_chunks = 0
    for page_number, page_text in load_document(path):
        for chunk in chunk_text(page_text):
            records.append({
                "source": path.name,
                "page": page_number,
                "chunk_index": file_chunks,
                "content": chunk,
            })
            file_chunks += 1

    print(f"{path.name}: {file_chunks} chunks")

if not records:
    raise SystemExit("No text found. Add .txt, .md or text-based .pdf files to the docs folder.")

print(f"\nLoading model and embedding {len(records)} chunks...")
model = load_model()
embeddings = model.encode(
    [r["content"] for r in records],
    batch_size=32,
    show_progress_bar=True,
)

rows = [
    (r["source"], r["page"], r["chunk_index"], r["content"], embedding)
    for r, embedding in zip(records, embeddings)
]

with get_connection() as conn:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            page INTEGER,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
        )
    """)
    conn.execute("TRUNCATE chunks RESTART IDENTITY")
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, page, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s)",
            rows,
        )

print(f"\nStored {len(rows)} chunks from {len({r['source'] for r in records})} files.")