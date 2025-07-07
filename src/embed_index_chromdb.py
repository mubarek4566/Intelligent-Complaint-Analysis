# src/embed_index.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load the embedding model
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Custom text chunking function
def custom_text_splitter(text, chunk_size=500, chunk_overlap=100):
    text = str(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Create chunked records for each narrative
def chunk_narratives(df, text_col="cleaned_narrative", id_col="Complaint ID", chunk_size=500, chunk_overlap=100):
    tqdm.pandas(desc="Splitting text into chunks")
    df["chunks"] = df[text_col].progress_apply(lambda x: custom_text_splitter(x, chunk_size, chunk_overlap))
    
    records = []
    for i, row in df.iterrows():
        for j, chunk in enumerate(row["chunks"]):
            records.append({
                "complaint_id": row[id_col],
                "chunk_id": f"{row[id_col]}_{j}",
                "text": chunk
            })
    return pd.DataFrame(records)

# Generate embeddings using batching for improved performance
def generate_embeddings(df, model, batch_size=64):
    texts = df['text'].tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype('float32')

# Create a ChromaDB collection for storing our embeddings along with metadata.
def build_chromadb_collection(collection_name="complaint_chunks", persist_directory="chromadb_store"):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet", 
        persist_directory=persist_directory
    ))
    # Optionally remove an existing collection with the same name.
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    return client, collection

# Add documents, their embeddings, and metadata to the ChromaDB collection.
def add_documents_to_collection(collection, df, embeddings):
    # Assemble lists of ids, documents, and metadata.
    ids = df['chunk_id'].tolist()
    documents = df['text'].tolist()
    metadatas = df[['complaint_id']].to_dict(orient='records')
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

# Primary function that processes data and creates the ChromaDB index
def process_and_index(df, model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=100, batch_size=64):
    model = load_model(model_name)
    # Generate text chunks
    chunk_df = chunk_narratives(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Generate embeddings for all chunks efficiently
    embeddings = generate_embeddings(chunk_df, model, batch_size=batch_size)
    chunk_df['embedding'] = embeddings.tolist()
    
    # Build and populate the ChromaDB collection
    client, collection = build_chromadb_collection()
    add_documents_to_collection(collection, chunk_df, embeddings)
    
    return chunk_df, client, collection
