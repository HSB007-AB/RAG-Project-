from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load the PDF
loader = PyPDFLoader("../Document Loaders/sample.pdf")
documents = loader.load()

print(f"Original pages: {len(documents)}")


# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# Split the documents
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# Show the first chunk
print("\nFirst chunk:")
print(chunks[0].page_content)


# Show metadata
print("\nMetadata:")
print(chunks[0].metadata)