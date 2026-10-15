import sys
import time

import numpy as np

from common import get_connection

TOP_K = 10
NUM_QUERIES = 20
QUERY_SQL = "SELECT id FROM bench_items ORDER BY embedding <=> %s LIMIT %s"


def run_queries(conn, queries):
    results = []
    start = time.perf_counter()
    for query in queries:
        rows = conn.execute(QUERY_SQL, (query, TOP_K)).fetchall()
        results.append({row[0] for row in rows})
    average_ms = (time.perf_counter() - start) * 1000 / len(queries)
    return results, average_ms


rng = np.random.default_rng(42)
queries = (rng.random((NUM_QUERIES, 384)) - 0.5).astype(np.float32)
extra_settings = sys.argv[1:]

with get_connection() as conn:
    conn.execute("SET enable_indexscan = off")
    exact_results, exact_ms = run_queries(conn, queries)

    conn.execute("SET enable_indexscan = on")
    for setting in extra_settings:
        conn.execute(setting)
    index_results, index_ms = run_queries(conn, queries)

recall = np.mean([
    len(exact & found) / len(exact)
    for exact, found in zip(exact_results, index_results)
])

print(f"Settings:            {extra_settings or 'defaults'}")
print(f"Exact search:        {exact_ms:8.2f} ms per query")
print(f"With index (if any): {index_ms:8.2f} ms per query")
print(f"Speed-up:            {exact_ms / index_ms:8.1f}x")
print(f"Recall@{TOP_K}:           {recall:8.2f}")
