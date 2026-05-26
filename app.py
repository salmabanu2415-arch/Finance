"""
Finance Policy Assistant - Streamlit Web UI
Provides a browser-based interface for querying finance policy documents.

Usage:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Enter your OpenAI API key or set OPENAI_API_KEY in .env",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.markdown("### RAG Parameters")
    chunk_size = st.slider("Chunk Size (tokens)", 400, 1200, 800, 50)
    chunk_overlap = st.slider("Chunk Overlap (tokens)", 50, 300, 150, 25)
    top_k = st.slider("Top-K Retrieved Chunks", 2, 8, 4, 1)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    st.markdown("### Sample Queries")
    sample_queries = [
        "What is the maximum reimbursement limit for domestic business travel?",
        "What approvals are required for vendor payments above $10,000?",
        "Can employees claim international travel expenses?",
        "What documents are mandatory for a procurement request?",
        "What is the meal expense reimbursement policy?",
        "How long does reimbursement processing take?",
        "What receipts are needed for expense claims under $25?",
        "What are the tax compliance requirements for vendor payments?",
    ]
    selected_sample = st.selectbox(
        "Try a sample query:", ["-- Select --"] + sample_queries
    )

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("💼 AI-Powered Finance Policy Assistant")
st.markdown(
    "Ask any question about company finance policies — reimbursements, travel, "
    "procurement, vendor payments, and more."
)

# ── Initialize RAG (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base from finance documents...")
def get_rag_system(chunk_size, chunk_overlap, top_k, temperature):
    """Build and cache the RAG system. Rebuilds if parameters change."""
    from src.rag_pipeline import DocumentIngestionPipeline, FinancePolicyRAG
    pipeline = DocumentIngestionPipeline(
        docs_dir="data/documents",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vector_store = pipeline.ingest()
    return FinancePolicyRAG(
        vector_store=vector_store,
        llm_model="gpt-4o-mini",
        top_k=top_k,
        temperature=temperature,
    )

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()

try:
    rag = get_rag_system(chunk_size, chunk_overlap, top_k, temperature)
except Exception as e:
    st.error(f"❌ Failed to initialize the RAG system: {e}")
    st.stop()

# ── Query input ───────────────────────────────────────────────────────────────
if selected_sample != "-- Select --":
    default_query = selected_sample
else:
    default_query = ""

query = st.text_area(
    "📝 Enter your finance policy question:",
    value=default_query,
    height=80,
    placeholder="e.g. What is the reimbursement limit for client travel?",
)

col1, col2 = st.columns([1, 6])
with col1:
    submit = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Clear", use_container_width=False)

if clear:
    st.rerun()

# ── Process query ─────────────────────────────────────────────────────────────
if submit and query.strip():
    with st.spinner("Retrieving relevant policies and generating answer..."):
        try:
            result = rag.query(query.strip())
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            st.stop()

    # Answer
    st.markdown("---")
    st.subheader("📋 Answer")
    st.markdown(result.answer)

    # Source References
    st.markdown("---")
    st.subheader("🔗 Retrieved Source References")
    for i, src in enumerate(result.sources, 1):
        with st.expander(f"Source {i}: {src['file']}" +
                         (f" — Page {src['page']}" if src["page"] != "N/A" else "")):
            st.markdown(f"**Snippet:**\n```\n{src['snippet']}\n```")

elif submit and not query.strip():
    st.warning("Please enter a question before clicking Ask.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Finance Policy Assistant · Powered by OpenAI GPT-4o Mini + text-embedding-3-small · "
    "Built with LangChain & FAISS"
)
