from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "What are your store opening hours?",
    "The cat is sleeping on the sofa.",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("First 5 numbers of sentence 1:", embeddings[0][:5])