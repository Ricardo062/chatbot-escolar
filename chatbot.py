import os
from pathlib import Path
from typing import Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from PyPDF2 import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =====================
# Config
# =====================
BASE_DIR = Path(__file__).resolve().parent
FOLDER_PATH = BASE_DIR / "Docs"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY não está definida nas variáveis de ambiente.")

client = OpenAI(api_key=api_key)

app = Flask(__name__)
CORS(app)


# =====================
# Carregar documentos
# =====================
def load_documents(folder_path: str) -> list[str]:
    documents = []

    for file in Path(folder_path).iterdir():
        if file.is_file() and file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())

        elif file.is_file() and file.suffix.lower() == ".pdf":
            reader = PdfReader(str(file))
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            documents.append(text)

    return documents


docs = load_documents(FOLDER_PATH)
print(f"{len(docs)} documentos carregados")


# =====================
# Dividir texto em chunks
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = []
for doc in docs:
    if doc and doc.strip():
        chunks.extend(text_splitter.split_text(doc))

print(f"{len(chunks)} chunks criados")


# =====================
# Criar embeddings e vector store
# =====================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=api_key
)

vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
print("Vectorstore criado com sucesso.")


# =====================
# Memória por sessão
# =====================
sessions: Dict[str, List[Dict[str, str]]] = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


# =====================
# Reescrever pergunta com contexto
# =====================
def condense_question(user_input: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return user_input

    messages = [
        {
            "role": "system",
            "content": (
                "Reescreve a última pergunta do utilizador para que fique completa e clara, "
                "usando o histórico da conversa. "
                "Mantém exatamente o mesmo significado. "
                "Responde apenas com a pergunta reescrita."
            )
        },
        *history[-6:],
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content.strip()


# =====================
# Função RAG
# =====================
def ask_rag_chat(user_input: str, history: List[Dict[str, str]], k: int = 4) -> str:
    # 1. Transformar a pergunta numa versão completa, se necessário
    final_question = condense_question(user_input, history)

    # 2. Recuperar chunks relevantes
    retrieved_docs = vectorstore.similarity_search(final_question, k=k)
    context = "\n\n".join([d.page_content for d in retrieved_docs])

    # 3. Prompt
    prompt = f"""
Responde à pergunta APENAS com base na informação abaixo, em português de Portugal.
Se a resposta não estiver na informação, diz claramente que não sabes.

INFORMAÇÃO:
{context}

PERGUNTA:
{final_question}
""".strip()

    # 4. Chamada ao LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "És um assistente informativo sobre uma escola e sobre o seu plano curricular. "
                    "Fornece o máximo de informação possível. "
                    "No final da resposta, recomenda novas possíveis questões que possas responder."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    # 5. Guardar histórico
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})

    return answer


# =====================
# API endpoint
# =====================
@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id em falta"}), 400

    if not message:
        return jsonify({"error": "message em falta"}), 400

    history = get_history(session_id)
    answer = ask_rag_chat(message, history, k=4)

    return jsonify({"text": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
