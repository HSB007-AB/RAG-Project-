from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Document Loaders/sample.pdf")

documents = loader.load()

print(f"Number of pages: {len(documents)}")

print("\nFirst page:")
print(documents[0].page_content)