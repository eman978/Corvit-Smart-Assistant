# Corvit Smart Assistant

A production-grade Retrieval-Augmented Generation (RAG) web app that answers questions strictly from Corvit Systems Rawalpindi's documents using two Groq LLMs with automatic failover.

## Features

- 📄 PDF ingestion with semantic chunking
- 🔎 TF-IDF vector retrieval with cosine similarity (lightweight, no GPU)
- 🤖 Dual Groq LLM (primary + fallback, automatic switch on error/empty)
- 💬 ChatGPT-style chat with **per-conversation history and delete** controls
- 📚 Default Corvit knowledge base + on-the-fly user PDF upload
- 🎨 Modern Inter-typography UI in the official Corvit palette
- 🛡️ Strict context control — never hallucinates beyond the documents

## Run locally (PyCharm or terminal)

1. Install Python 3.11+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.example` and add your Groq keys.
4. Run:
   ```bash
   streamlit run app.py --server.port 5000
   ```
5. Open `http://localhost:5000`.

## Project layout

```
corvit_assistant/
├── app.py                 # Streamlit UI
├── modules/
│   ├── ingestion.py       # PDF text extraction + semantic chunking
│   ├── embeddings.py      # TF-IDF embedding model
│   ├── retriever.py       # Vector index + top-k retrieval
│   └── llm_handler.py     # Dual Groq client with failover
├── data/                  # Pre-loaded Corvit PDFs
├── assets/logo.png        # Official Corvit logo
├── .streamlit/config.toml # Server + theme
├── requirements.txt
└── .env.example
```

## Deployment

The app reads `PORT` from the environment if set; otherwise defaults to 5000.
You can deploy on any platform that supports Streamlit (Streamlit Cloud, Replit, Render, Fly.io, etc.).
