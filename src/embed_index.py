
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def custom_text_splitter(text, chunk_size=500, chunk_overlap=100):
    text = str(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

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

def generate_embeddings(df, model, batch_size=64):
    texts = df['text'].tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype('float32')

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_faiss_index(index, path="complaint_chunks.index"):
    faiss.write_index(index, path)

def process_and_index(df, model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=100):
    model = load_model(model_name)
    chunk_df = chunk_narratives(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = generate_embeddings(chunk_df, model)
    chunk_df['embedding'] = embeddings.tolist()
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    return chunk_df, index

def load_chunked_data_and_index(data_path="chunked_with_embeddings.csv", index_path="complaint_chunks.index"):
    import ast
    df = pd.read_csv(data_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    embeddings = np.vstack(df['embedding'].values).astype('float32')
    index = faiss.read_index(index_path)
    return df, index

def retrieve_top_k_chunks(question, model, index, chunk_df, k=5):
    """
    Retrieves top-k most relevant text chunks for a given question using FAISS.
    """
    # Embed the question
    query_vector = model.encode([question], convert_to_numpy=True).astype("float32")

    # Search in the FAISS index
    distances, indices = index.search(query_vector, k)

    # Retrieve matching chunks and metadata
    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < len(chunk_df):
            result = {
                "chunk_id": chunk_df.iloc[idx]["chunk_id"],
                "complaint_id": chunk_df.iloc[idx]["complaint_id"],
                "text": chunk_df.iloc[idx]["text"],
                "score": float(distances[0][i])
            }
            results.append(result)
    return results
