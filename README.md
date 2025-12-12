# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot template with:

- **Backend**: FastAPI + LangChain + OpenAI + PostgreSQL (pgvector)
- **Frontend**: Streamlit chat UI
- Extras from day 1:
  - RAG pipeline (rewrite → retrieve → answer → suggest next questions)
  - Intent classification (RAG_QA vs CHITCHAT vs OTHER)
  - Follow-up question rewriting (follow-up → standalone)
  - Suggested follow-up questions after each answer
  - Centralized prompts module
  - Clean Pydantic models and API structure

---

## 🏗 Project Structure

```bash
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── chat.py
│   │   │   │   ├── ingest.py
│   │   │   │   └── health.py
│   │   │   └── __init__.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging_config.py
│   │   │   └── prompts.py
│   │   ├── rag/
│   │   │   ├── embeddings.py
│   │   │   ├── retriever.py
│   │   │   ├── pipeline.py
│   │   │   ├── intent.py
│   │   │   └── suggestions.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tests/
│   │   ├── test_chat.py
│   │   ├── test_rag.py
│   │   └── __init__.py
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   ├── components/
│   │   ├── chat_ui.py
│   │   └── sidebar.py
│   └── requirements.txt
│
├── scripts/
│   ├── dev_run_backend.sh
│   ├── dev_run_frontend.sh
│   └── init_db.py
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── .env.example
├── .gitignore
└── README.md
⚙️ Tech Stack
Backend

Python 3.12

FastAPI

LangChain + langchain-openai

PostgreSQL + pgvector (via langchain-postgres)

Pydantic v2 / pydantic-settings

Uvicorn

Frontend

Streamlit

Requests (to call the backend)

🚀 Getting Started (Local Dev)
1. Clone and set up env
bash
Copy code
git clone <your-repo-url>
cd rag-chatbot

cp .env.example .env
# edit .env and add your real OPENAI_API_KEY, etc.
Make sure Postgres is running and accessible from DATABASE_URL in .env.
If you use the provided docker-compose, that will be handled for you (see below).

2. Install backend dependencies
bash
Copy code
cd backend
pip install -r requirements.txt
3. Install frontend dependencies
bash
Copy code
cd ../frontend
pip install -r requirements.txt
4. Initialize the database (pgvector)
From project root:

bash
Copy code
python scripts/init_db.py
This will:

Ensure the vector extension exists

Initialize the PGVector collection (table) via LangChain

5. Run backend (FastAPI)
From project root:

bash
Copy code
./scripts/dev_run_backend.sh
Backend will run on: http://localhost:8000

6. Run frontend (Streamlit)
In a separate terminal, from project root:

bash
Copy code
./scripts/dev_run_frontend.sh
Frontend will run on: http://localhost:8501

🐳 Running with Docker
From the docker/ directory:

bash
Copy code
cd docker
cp ../.env.example ../.env    # if you haven't already
# edit ../.env and add your OPENAI_API_KEY

docker-compose up --build
Services:

db: Postgres + pgvector (pg16)

backend: FastAPI (http://localhost:8000)

frontend: Streamlit (http://localhost:8501)

📡 API Overview
Health
GET /v1/health
Returns basic status info (environment, vector store status stub).

Chat
POST /v1/chat

Request body (ChatRequest):

jsonc
Copy code
{
  "question": "What does the design doc say about the RAG pipeline?",
  "history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello, how can I help?" }
  ],
  "tone": "neutral",
  "style": "concise",
  "top_k": 5,
  "namespace": "kb-1"
}
Response body (ChatResponse):

jsonc
Copy code
{
  "answer": "The design doc explains that the RAG pipeline first rewrites follow-up questions...",
  "sources": [
    {
      "id": "chunk-1",
      "title": "design_doc.md",
      "snippet": "The RAG pipeline consists of question rewriting, retrieval, and answer generation...",
      "metadata": { "filename": "design_doc.md", "namespace": "kb-1" }
    }
  ],
  "suggested_questions": [
    "How is the retriever configured?",
    "What embeddings model is used?"
  ],
  "intent": "RAG_QA"
}
Ingest
POST /v1/ingest/text
Ingests raw text.

jsonc
Copy code
{
  "texts": ["Some text to ingest into the knowledge base."],
  "namespace": "kb-1",
  "metadata": { "source": "manual" }
}
POST /v1/ingest/file
Basic support for .txt/.md (can be extended for PDFs, etc.).

🧠 RAG Features
Intent classification (intent.py):
RAG_QA vs CHITCHAT vs OTHER using an LLM + INTENT_PROMPT.

Question rewriting (pipeline.py):
Follow-up → standalone using CONDENSE_QUESTION_PROMPT.

Retrieval (retriever.py):
Chunking (RecursiveCharacterTextSplitter) and similarity search via Postgres/pgvector.

Answer generation (pipeline.py + prompts.py):
ANSWER_PROMPT with context, tone, and style controls.

Suggested questions (suggestions.py):
Generated from question + answer + context, returned as a list.

🧪 Testing
From backend/:

bash
Copy code
pytest
Current tests:

test_chat.py: basic chat endpoint test with monkeypatched pipeline (no real OpenAI/DB calls).

test_rag.py: chunking and context building tests.

✅ Status
Backend core: implemented

Frontend core: implemented

Dev scripts: implemented

Docker + Postgres: implemented

RAG pipeline: implemented

Intent classification + suggestions: implemented
