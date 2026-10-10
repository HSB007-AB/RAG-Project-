from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# 2. Load our existing vector store
vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

# 3. Turn the vector store into a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Ask a question
question = "What is quantization?"

# 5. Retrieve the 3 most relevant chunks
results = retriever.invoke(question)

# 6. Display the results
print(f"Found {len(results)} relevant chunks.\n")

for i, document in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)
    print()