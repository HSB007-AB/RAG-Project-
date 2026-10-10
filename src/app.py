
import io
import uuid
import requests
import streamlit as st

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Local RAG Assistant")
st.caption("Upload a PDF and ask questions about its contents.")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

embeddings = load_embeddings()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Drag and drop your PDFs here",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs", disabled=not uploaded_files):
    with st.spinner("Reading and processing PDFs..."):
        try:
            documents = []

            for uploaded_file in uploaded_files:
                reader = PdfReader(
                    io.BytesIO(uploaded_file.getvalue())
                )

                for page_number, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_number
                                }
                            )
                        )

            if not documents:
                st.error("No readable text found in the PDFs.")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_documents(documents)

                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=f"pdf_{uuid.uuid4().hex}",
                )

                st.session_state.messages = []

                st.success(
                    f"Processed {len(uploaded_files)} PDF(s) "
                    f"into {len(chunks)} chunks."
                )

        except Exception as e:
            st.error(f"PDF processing failed: {e}")


if st.session_state.vector_store is None:
    st.info("Upload and process a PDF to start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about your PDFs...",
    disabled=st.session_state.vector_store is None
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vector_store = st.session_state.vector_store
                if vector_store is None:
                    raise RuntimeError("Process a PDF before asking a question.")

                results = vector_store.similarity_search_with_score(
                    question, k=2
                )

                if not results or results[0][1] >= 0.8:
                    answer = (
                        "I couldn't find enough relevant information "
                        "in the uploaded PDFs to answer that question."
                    )
                else:
                    context = "\n\n".join(
                        doc.page_content for doc, score in results
                    )

                    history = ""
                    for message in st.session_state.messages[-7:-1]:
                        role = (
                            "User" if message["role"] == "user"
                            else "Assistant"
                        )
                        history += f"{role}: {message['content']}\n"

                    prompt = f"""
You are a helpful assistant that answers questions
using the provided PDF context.

Instructions:
- Answer directly using the context.
- Use concise bullet points for explanations.
- Include a short **Summary** at the end when useful.
- Do not invent facts or introduce unsupported details.
- If the context does not contain the answer, say so.
- Use conversation history only to understand follow-up questions.
- Keep the answer clear and easy to understand.

Conversation history:
{history}

PDF context:
{context}

Current question:
{question}

Answer:
"""

                    response = requests.post(
                        "http://localhost:8080/v1/chat/completions",
                        json={
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a helpful assistant. "
                                        "Give only the final answer."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.1,
                            "max_tokens": 300,
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        },
                        timeout=120
                    )

                    response.raise_for_status()
                    answer = (
                        response.json()["choices"][0]["message"]
                        ["content"].strip()
                    )

            except requests.exceptions.ConnectionError:
                answer = "Could not connect to Qwen. Is the server running?"
            except requests.exceptions.Timeout:
                answer = "Qwen took too long to respond. Please try again."
            except requests.exceptions.RequestException as e:
                answer = f"Request failed: {e}"
            except Exception as e:
                answer = f"Something went wrong: {e}"

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })