# PDF Uploader RAG

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using a Large Language Model (LLM).

## Overview

PDF Uploader RAG combines document processing, text splitting, embeddings, vector storage, and retrieval to help users find relevant information from PDF documents and generate context-aware answers.

## Features

- **PDF Document Loading:** Load and process PDF documents.
- **Text Splitting:** Divide documents into smaller chunks for efficient retrieval.
- **Embeddings:** Convert text into vector representations for semantic search.
- **Vector Storage:** Store document embeddings for retrieval.
- **Retrieval-Augmented Generation:** Retrieve relevant document content to help generate answers.
- **Question Answering:** Ask questions about the content of your documents.

## Project Structure

```text
RAG-Project/
├── Document Loaders/
│   └── sample.pdf
├── src/
│   ├── app.py
│   ├── embedding_test.py
│   ├── load_document.py
│   ├── rag.py
│   ├── retriever.py
│   ├── split_document.py
│   └── vector_store.py
├── .gitignore
└── README.md
```

## Tech Stack

- Python
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Text Embeddings
- Vector Databases

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HSB007-AB/RAG-Project-.git
cd RAG-Project-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the packages required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

If the application requires API keys, configure them in a local `.env` file. Never commit API keys or other secrets to GitHub.

### 5. Run the application

If `src/app.py` is the application entry point, run it using the command appropriate for your framework. For a Streamlit application:

```bash
streamlit run src/app.py
```

## Learning Objectives

This project demonstrates practical experience with document processing, embeddings, vector search, retrieval pipelines, and LLM-powered question answering.

## Future Improvements

- Add source citations to generated answers.
- Improve retrieval quality and document chunking.
- Support multiple PDF documents.
- Add evaluation for retrieval and answer quality.
- Deploy the application as a web app.

## Author

**HSB007-AB**

GitHub: [@HSB007-AB](https://github.com/HSB007-AB)