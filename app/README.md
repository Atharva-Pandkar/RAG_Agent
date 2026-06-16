# 10-K RAG Chatbot (skeleton)

A minimal full-stack chatbot wrapping the RAG retrieval experiments in a
LangChain ReAct-style agent (`create_agent`), with a FastAPI backend and a
React (Vite) frontend.

## Backend

```
pip install -r app/backend/requirements.txt
cp app/backend/.env.example app/backend/.env   # fill in OPENAI_API_KEY
```

Run from the project root:

```
uvicorn app.backend.main:app --reload --port 8000
```

- `GET /health` - liveness check
- `POST /chat` - `{"messages": [{"role": "user", "content": "..."}]}` -> `{"response": "..."}`

The agent has one tool, `search_10k_filings`, backed by `RagPipeline`
(hybrid retrieval over `Experiments/corpora/fixed_size_1024_100.json` - the
best-performing config from the retrieval experiments). Swap the
`corpus_path` / `retrieval_strategy` in `app/backend/rag_tool.py` to try
other configs.

## Frontend

```
cd app/frontend
npm install
npm run dev
```

Dev server proxies `/api/*` to `http://localhost:8000/*` (see `vite.config.js`).

## Next steps

- Per-session conversation memory (currently stateless - frontend resends
  full history each turn)
- Stream responses
- Surface retrieved sources/citations in the UI
- Swap retrieval config / add reranking via `rag_tool.py`
