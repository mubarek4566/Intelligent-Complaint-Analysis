# import libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..", "src")))

from rag_pipeline import retrieve_similar_chunks, load_embedding_model, load_chromadb_collection


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


def load_chromadb_collection(collection_name="complaint_chunks", persist_directory="chromadb_store"):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    ))
    collection = client.get_collection(name=collection_name)
    return collection

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def retrieve_similar_chunks(query, model, collection, top_k=5):
    # Step 1: Embed the query
    query_embedding = model.encode(query).tolist()

    # Step 2: Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 3: Return results
    return results

# Load the same model and collection used in Task 2
model = load_embedding_model()
collection = load_chromadb_collection()

# Sample user question
user_question = "How can I dispute an unauthorized transaction on my card?"

# Retrieve top 5 relevant complaint chunks
results = retrieve_similar_chunks(user_question, model, collection, top_k=5)

# Display results
for i, (doc, meta, score) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
    print(f"\nRank {i+1}:")
    print(f"Complaint ID: {meta['complaint_id']}")
    print(f"Score (L2 Distance): {score:.4f}")
    print(f"Text: {doc}")
