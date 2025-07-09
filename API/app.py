import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

import sys
import os

# Add the root project directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.rag_pipeline import load_local_llm
from src.embed_index import retrieve_top_k_chunks
from src.rag_pipeline import generate_answer, build_prompt

# Load models and data once using caching
@st.cache_resource
def load_models_and_data():
    # Base directory (project root)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Absolute paths
    chunk_path = os.path.join(BASE_DIR, "Data", "chunked_with_embeddings.csv")
    index_path = os.path.join(BASE_DIR, "Data", "complaint_chunks.index")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load and parse embeddings (limit rows for memory efficiency)
    chunk_df = pd.read_csv(chunk_path).head(10000)
    chunk_df['embedding'] = chunk_df['embedding'].apply(eval)

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load local LLM (CPU by default)
    llm = load_local_llm(model_name="facebook/opt-1.3b", device_map="cpu")

    return model, chunk_df, index, llm

# Unpack returned values
embedder, chunk_df, faiss_index, llm_pipeline = load_models_and_data()

# Page Config
st.set_page_config(page_title="CrediTrust AI Assistant", page_icon="🤖", layout="wide")

st.title("💬 CrediTrust AI Assistant")
st.write("Ask a question about customer complaints and see what the system finds.")

# Input form
with st.form(key="query_form"):
    user_question = st.text_area("Type your question here:", height=100)
    submit_button = st.form_submit_button("🔍 Ask")

# Handle submission
if submit_button and user_question:
    with st.spinner("Retrieving and answering..."):
        top_chunks = retrieve_top_k_chunks(user_question, embedder, faiss_index, chunk_df, k=5)
        context_chunks = [c["text"] for c in top_chunks]

        # Generate answer
        final_answer = generate_answer(context_chunks, user_question, llm_pipeline)

    # Display answer
    st.markdown("### 🤖 AI-Generated Answer")
    st.success(final_answer)

    # Display source context
    with st.expander("📄 Show Sources (Top 5 Retrieved Chunks)", expanded=True):
        for i, source in enumerate(context_chunks):
            st.markdown(f"**Source {i+1}:**\n> {source}")

# Clear button
if st.button("🧹 Clear"):
    st.experimental_rerun()
