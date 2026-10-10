
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

chat_history = []


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


prompt_template = PromptTemplate(
    input_variables=["context", "question", "history"],
    template="""
You are a helpful AI assistant.

Instructions:
- Answer the user's current question directly.
- Use simple, clear language.
- Keep answers concise, usually 1-3 sentences.
- Avoid repeating information already given.
- Use conversation history to understand follow-up questions.
- Use the provided context when it is relevant.
- If the answer is not in the context or history,
  say so rather than inventing facts.
- Do not include unnecessary introductions or summaries.

Conversation history:
{history}

Retrieved context:
{context}

Current question:
{question}

Answer:
"""
)


while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    documents_with_scores = vector_store.similarity_search_with_score(
        question,
        k=2
    )

    threshold = 0.8

    if documents_with_scores and documents_with_scores[0][1] >= threshold:
        print("I don't have enough relevant information to answer.")
        continue

    documents = [doc for doc, score in documents_with_scores]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    history_text = ""

    for chat in chat_history[-3:]:
        history_text += f"""
User: {chat["question"]}
Assistant: {chat["answer"]}
"""

    prompt = prompt_template.format(
        context=context,
        question=question,
        history=history_text
    )

    url = "http://localhost:8080/v1/chat/completions"

    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Give only the final answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    try:
        response = requests.post(
            url,
            json=data,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("Could not connect to Qwen. Is the server running?")
        continue

    except requests.exceptions.Timeout:
        print("Qwen took too long to respond.")
        continue

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        continue

    print(answer)

    chat_history.append({
        "question": question,
        "answer": answer
    })
