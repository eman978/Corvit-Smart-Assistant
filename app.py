"""Corvit Smart Assistant — Streamlit RAG application."""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from modules.ingestion import Chunk, ingest_many, ingest_pdf
from modules.llm_handler import DualLLMClient
from modules.retriever import VectorIndex, build_index, retrieve

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("corvit-assistant")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

st.set_page_config(
    page_title="Corvit Smart Assistant",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Styling — production-grade UI per spec
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          html, body, [class*="css"], .stApp, .stMarkdown, .stChatMessage {
              font-family: 'Inter', sans-serif !important;
              color: #0B0F19;
          }
          .stApp { background: #F8FAFC; }

          /* Top bar */
          .corvit-topbar {
              display: flex; align-items: center; justify-content: space-between;
              padding: 14px 22px; margin: -1rem -1rem 1.25rem -1rem;
              background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
              color: #F8FAFC; border-bottom: 1px solid #1E293B;
          }
          .corvit-topbar .brand { display: flex; align-items: center; gap: 14px; }
          .corvit-topbar .brand img { height: 40px; }
          .corvit-topbar .brand .title { font-weight: 700; font-size: 20px; letter-spacing: 0.2px; }
          .corvit-topbar .brand .sub { font-size: 12px; opacity: 0.7; margin-top: 2px; }
          .corvit-status {
              display: inline-flex; align-items: center; gap: 8px;
              padding: 6px 12px; border-radius: 999px;
              background: rgba(59, 130, 246, 0.12);
              color: #93C5FD; font-size: 12px; font-weight: 600;
              border: 1px solid rgba(59, 130, 246, 0.35);
          }
          .corvit-status .dot {
              width: 8px; height: 8px; border-radius: 50%;
              background: #22C55E; box-shadow: 0 0 8px #22C55E;
          }
          .corvit-status.fallback { background: rgba(245, 158, 11, 0.12); color: #FCD34D; border-color: rgba(245, 158, 11, 0.35); }
          .corvit-status.fallback .dot { background: #F59E0B; box-shadow: 0 0 8px #F59E0B; }
          .corvit-status.idle .dot { background: #64748B; box-shadow: none; }

          /* Sidebar */
          section[data-testid="stSidebar"] {
              background: #FFFFFF;
              border-right: 1px solid #E2E8F0;
          }
          section[data-testid="stSidebar"] h1,
          section[data-testid="stSidebar"] h2,
          section[data-testid="stSidebar"] h3 { color: #0F172A; }
          .doc-chip {
              display: flex; align-items: center; justify-content: space-between;
              padding: 10px 12px; border: 1px solid #E2E8F0; border-radius: 10px;
              margin-bottom: 8px; background: #F8FAFC;
              font-size: 13px;
          }
          .doc-chip .name { font-weight: 600; color: #0F172A; }
          .doc-chip .meta { color: #64748B; font-size: 11px; }

          /* Chat bubbles */
          [data-testid="stChatMessage"] {
              background: transparent !important;
              padding: 0 !important;
          }
          .bubble {
              padding: 14px 18px; border-radius: 14px; max-width: 100%;
              line-height: 1.55; font-size: 15px;
              box-shadow: 0 1px 2px rgba(15,23,42,0.04);
          }
          .bubble.user {
              background: #3B82F6; color: #FFFFFF;
              border-bottom-right-radius: 4px;
          }
          .bubble.assistant {
              background: #FFFFFF; color: #0B0F19;
              border: 1px solid #E2E8F0;
              border-bottom-left-radius: 4px;
          }
          .source-block {
              margin-top: 10px; padding: 10px 12px; border-radius: 10px;
              background: #F1F5F9; border: 1px solid #E2E8F0;
              font-size: 12px; color: #475569;
          }
          .source-block strong { color: #0F172A; }
          .meta-line { font-size: 11px; color: #94A3B8; margin-top: 6px; }

          /* Buttons */
          .stButton > button {
              border-radius: 10px; font-weight: 600; border: 1px solid #E2E8F0;
          }
          .stButton > button[kind="primary"] {
              background: #3B82F6; color: #FFFFFF; border-color: #3B82F6;
          }

          /* Chat input */
          [data-testid="stChatInput"] textarea {
              border-radius: 12px !important;
              font-family: 'Inter', sans-serif !important;
          }

          /* Hide streamlit chrome */
          #MainMenu, footer, header { visibility: hidden; }
          .block-container { padding-top: 0.5rem; max-width: 1200px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return ""


def render_topbar(status: str, model_label: str) -> None:
    logo = _logo_data_uri()
    status_class = ""
    label = "Idle"
    if status == "primary":
        status_class = ""
        label = f"LLM 1 active · {model_label}"
    elif status == "fallback":
        status_class = "fallback"
        label = f"LLM 2 active · {model_label}"
    else:
        status_class = "idle"
        label = "Standing by"

    st.markdown(
        f"""
        <div class="corvit-topbar">
          <div class="brand">
            {f'<img src="{logo}" alt="Corvit"/>' if logo else ''}
            <div>
              <div class="title">Corvit Smart Assistant</div>
              <div class="sub">Reaching End to End · RAG over your documents</div>
            </div>
          </div>
          <div class="corvit-status {status_class}">
            <span class="dot"></span><span>{label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm_client() -> DualLLMClient:
    return DualLLMClient()


@st.cache_resource(show_spinner="Reading the Corvit knowledge base…")
def build_default_index() -> tuple[VectorIndex, list[dict]]:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        return None, []  # type: ignore
    chunks = ingest_many(pdfs)
    index = build_index(chunks)
    docs = [
        {"name": p.name, "path": str(p), "chunks": sum(1 for c in chunks if c.source == p.name)}
        for p in pdfs
    ]
    return index, docs


def _build_uploaded_index(uploaded_files) -> tuple[VectorIndex, list[dict]]:
    all_chunks: list[Chunk] = []
    docs_meta: list[dict] = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        try:
            file_chunks = ingest_pdf(tmp_path)
            for ch in file_chunks:
                ch.source = uf.name
                ch.chunk_id = len(all_chunks)
                all_chunks.append(ch)
            docs_meta.append({"name": uf.name, "path": tmp_path, "chunks": len(file_chunks)})
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if not all_chunks:
        return None, []  # type: ignore
    return build_index(all_chunks), docs_meta


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_conv" not in st.session_state:
        new_id = _new_conversation()
        st.session_state.active_conv = new_id
    if "index_mode" not in st.session_state:
        st.session_state.index_mode = "default"  # 'default' | 'custom'
    if "custom_index" not in st.session_state:
        st.session_state.custom_index = None
    if "custom_docs" not in st.session_state:
        st.session_state.custom_docs = []
    if "last_provider" not in st.session_state:
        st.session_state.last_provider = "idle"
    if "last_model" not in st.session_state:
        st.session_state.last_model = ""


def _new_conversation() -> str:
    cid = uuid.uuid4().hex[:10]
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    st.session_state.conversations[cid] = {
        "id": cid,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    return cid


def _delete_conversation(cid: str) -> None:
    if cid in st.session_state.conversations:
        del st.session_state.conversations[cid]
    if not st.session_state.conversations:
        st.session_state.active_conv = _new_conversation()
    elif st.session_state.active_conv == cid:
        st.session_state.active_conv = next(iter(st.session_state.conversations))


def _active_conv() -> dict:
    return st.session_state.conversations[st.session_state.active_conv]


def _maybe_set_title(conv: dict, question: str) -> None:
    if conv["title"] == "New chat" and question:
        conv["title"] = (question[:48] + "…") if len(question) > 48 else question


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar(default_docs: list[dict]) -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        st.markdown("### Corvit Smart Assistant")
        st.caption("Ask questions strictly from the uploaded documents.")
        st.divider()

        # ---- Conversations ----
        st.markdown("#### 💬 Conversations")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("➕ New chat", use_container_width=True, type="primary"):
                new_id = _new_conversation()
                st.session_state.active_conv = new_id
                st.rerun()
        with c2:
            if st.button("🗑️", help="Clear ALL conversations", use_container_width=True):
                st.session_state.conversations = {}
                st.session_state.active_conv = _new_conversation()
                st.rerun()

        for cid, conv in list(st.session_state.conversations.items())[::-1]:
            is_active = cid == st.session_state.active_conv
            row = st.container()
            with row:
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    label = ("● " if is_active else "○ ") + conv["title"]
                    if st.button(label, key=f"open_{cid}", use_container_width=True):
                        st.session_state.active_conv = cid
                        st.rerun()
                with col_b:
                    if st.button("✕", key=f"del_{cid}", help="Delete this chat"):
                        _delete_conversation(cid)
                        st.rerun()

        st.divider()

        # ---- Documents ----
        st.markdown("#### 📚 Knowledge Base")
        mode = st.radio(
            "Source",
            options=["Default Corvit docs", "Upload my own PDFs"],
            index=0 if st.session_state.index_mode == "default" else 1,
            label_visibility="collapsed",
        )
        st.session_state.index_mode = "default" if mode.startswith("Default") else "custom"

        if st.session_state.index_mode == "default":
            for d in default_docs:
                st.markdown(
                    f"<div class='doc-chip'><div><div class='name'>📄 {d['name']}</div>"
                    f"<div class='meta'>{d['chunks']} chunks indexed</div></div>"
                    f"<span class='meta'>Ready</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            uploaded = st.file_uploader(
                "Upload PDF(s)",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded:
                with st.spinner("Indexing your documents…"):
                    idx, docs = _build_uploaded_index(uploaded)
                st.session_state.custom_index = idx
                st.session_state.custom_docs = docs
                if idx is not None:
                    st.success(f"Indexed {sum(d['chunks'] for d in docs)} chunks from {len(docs)} file(s).")
            for d in st.session_state.custom_docs:
                st.markdown(
                    f"<div class='doc-chip'><div><div class='name'>📄 {d['name']}</div>"
                    f"<div class='meta'>{d['chunks']} chunks indexed</div></div>"
                    f"<span class='meta'>Ready</span></div>",
                    unsafe_allow_html=True,
                )
            if not st.session_state.custom_docs:
                st.info("Upload one or more PDFs to start chatting with them.")

        st.divider()
        st.caption("⚙️ Powered by Groq LLMs with automatic failover.")


# ----------------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------------
def render_messages(conv: dict) -> None:
    if not conv["messages"]:
        st.markdown(
            """
            <div style="margin: 80px auto; max-width: 640px; text-align: center; color: #475569;">
              <div style="font-size: 36px; font-weight: 800; color: #0F172A;">Welcome to Corvit Smart Assistant</div>
              <p style="margin-top: 8px; font-size: 15px;">Ask anything about Corvit Systems Rawalpindi —
              courses, fees, NAVTTC certifications, admissions, schedules, contact details, and more.</p>
              <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 24px;">
                <span style="padding:8px 14px;background:#fff;border:1px solid #E2E8F0;border-radius:999px;font-size:13px;">What courses are offered?</span>
                <span style="padding:8px 14px;background:#fff;border:1px solid #E2E8F0;border-radius:999px;font-size:13px;">Free NAVTTC programs?</span>
                <span style="padding:8px 14px;background:#fff;border:1px solid #E2E8F0;border-radius:999px;font-size:13px;">Admission process?</span>
                <span style="padding:8px 14px;background:#fff;border:1px solid #E2E8F0;border-radius:999px;font-size:13px;">Contact &amp; address</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in conv["messages"]:
        with st.chat_message(msg["role"], avatar=str(LOGO_PATH) if msg["role"] == "assistant" and LOGO_PATH.exists() else None):
            cls = "user" if msg["role"] == "user" else "assistant"
            st.markdown(f"<div class='bubble {cls}'>{_escape(msg['content'])}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                src_html = "<div class='source-block'><strong>Sources used:</strong><br/>"
                for s in msg["sources"]:
                    src_html += f"• <em>{s['source']}</em> · page {s['page']} · score {s['score']:.2f}<br/>"
                src_html += "</div>"
                st.markdown(src_html, unsafe_allow_html=True)
            if msg.get("model"):
                provider_label = "LLM 1" if msg.get("provider") == "primary" else "LLM 2"
                st.markdown(
                    f"<div class='meta-line'>Answered by {provider_label} · {msg['model']}</div>",
                    unsafe_allow_html=True,
                )


def _escape(text: str) -> str:
    # Preserve simple line breaks; escape HTML
    import html
    return html.escape(text).replace("\n", "<br/>")


def get_active_index() -> VectorIndex | None:
    if st.session_state.index_mode == "custom":
        return st.session_state.custom_index
    idx, _ = build_default_index()
    return idx


def handle_query(question: str) -> None:
    conv = _active_conv()
    _maybe_set_title(conv, question)
    conv["messages"].append({"role": "user", "content": question})

    index = get_active_index()
    if index is None:
        conv["messages"].append({
            "role": "assistant",
            "content": "Please upload at least one PDF before asking questions.",
        })
        return

    with st.spinner("Searching the documents…"):
        results = retrieve(index, question, top_k=4)

    # Build context block
    if not results or all(r.score < 0.05 for r in results):
        # No relevant context — short-circuit per spec
        conv["messages"].append({
            "role": "assistant",
            "content": "The requested information is not available in the provided document.",
            "sources": [],
        })
        return

    context_parts = []
    for r in results:
        context_parts.append(
            f"[Source: {r.chunk.source} · page {r.chunk.page}]\n{r.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    history = [m for m in conv["messages"][:-1] if m.get("role") in {"user", "assistant"}]

    try:
        client = get_llm_client()
        with st.spinner("Thinking…"):
            response = client.chat(question=question, context=context, history=history)
    except Exception as exc:
        logger.exception("LLM call failed")
        conv["messages"].append({
            "role": "assistant",
            "content": f"⚠️ Sorry, I could not reach the language models. ({exc})",
        })
        return

    st.session_state.last_provider = response.provider_id
    st.session_state.last_model = response.model_used

    conv["messages"].append({
        "role": "assistant",
        "content": response.text,
        "sources": [
            {"source": r.chunk.source, "page": r.chunk.page, "score": r.score}
            for r in results
        ],
        "model": response.model_used,
        "provider": response.provider_id,
    })


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_state()

    # Default index (cached)
    default_index, default_docs = build_default_index()

    render_topbar(
        status=st.session_state.last_provider,
        model_label=st.session_state.last_model or "—",
    )

    render_sidebar(default_docs)

    conv = _active_conv()
    render_messages(conv)

    question = st.chat_input("Ask anything about Corvit Systems Rawalpindi…")
    if question:
        handle_query(question)
        st.rerun()


if __name__ == "__main__":
    main()
