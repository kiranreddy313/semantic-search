from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "when can I visit your shop?"

documents = [
    "Forgot your login credentials? Here's how to recover them.",
    "Our store is open from 9am to 6pm, Monday to Saturday.",
    "To change your password, go to Settings and click Security.",
    "We ship to all 50 US states within 5 business days.",
    "The cat is sleeping on the sofa.",
]

query_embedding = model.encode(query)
document_embeddings = model.encode(documents)

scores = util.cos_sim(query_embedding, document_embeddings)[0]

results = sorted(zip(documents, scores.tolist()), key=lambda pair: pair[1], reverse=True)

print(f"Query: {query}\n")
for document, score in results:
    print(f"{score:.3f}  {document}")