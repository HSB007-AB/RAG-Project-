from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# Some example sentences
sentences = [
    "What is artificial intelligence?",
    "AI is a field of computer science.",
    "What is the weather today?"
]


# Convert sentences into embeddings
embeddings = model.encode(sentences)


print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)