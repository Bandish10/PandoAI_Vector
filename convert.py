# convert.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def preprocess_data(df):
    texts = df.apply(lambda row: ' | '.join([str(v) for v in row.values if pd.notnull(v)]), axis=1)
    return texts

def compute_embeddings(model, texts):
    return model.encode(texts.tolist(), convert_to_numpy=True, show_progress_bar=True)

def build_faiss_index(embeddings):
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    # Use IndexFlatIP for cosine similarity with normalized vectors
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_embeddings(embeddings, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)

def save_faiss_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def should_recompute():
    excel_path = "Vector.xlsx"
    embed_path = "sapbert-mpnet/embeddings/embeddings.npy"
    
    if not os.path.exists(embed_path):
        return True
    
    excel_mtime = os.path.getmtime(excel_path)
    embed_mtime = os.path.getmtime(embed_path)
    
    return excel_mtime > embed_mtime

if __name__ == "__main__":
    if should_recompute():
        print("Dataset changed or embeddings missing. Recomputing...")
        
        # Load data
        excel_path = "Vector.xlsx"
        df = pd.read_excel(excel_path)

        # Load model
        model_dir = "sapbert-mpnet/model"
        model = SentenceTransformer(model_dir)

        # Preprocess and compute embeddings
        texts = preprocess_data(df)
        embeddings = compute_embeddings(model, texts)

        # Save embeddings
        embed_dir = "sapbert-mpnet/embeddings"
        embed_path = os.path.join(embed_dir, "embeddings.npy")
        save_embeddings(embeddings, embed_path)

        # Build and save FAISS index
        index_dir = "sapbert-mpnet/faiss_index"
        index_path = os.path.join(index_dir, "faiss_index.index")
        index, normalized_embeddings = build_faiss_index(embeddings)
        save_faiss_index(index, index_path)

        print("Precomputation complete!")
        print(f"Processed {len(df)} records")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Test similarity to verify normalization
        print("Testing similarity computation...")
        test_embedding = normalized_embeddings[0:1].copy()
        similarities = np.dot(normalized_embeddings, test_embedding.T).flatten()
        print(f"Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
        print(f"Expected range: -1.000 to 1.000")
        
        # Test with a few sample queries
        sample_texts = ["TKR", "Total Knee Replacement", "CABG", "Coronary Artery Bypass"]
        sample_embeddings = model.encode(sample_texts, convert_to_numpy=True)
        faiss.normalize_L2(sample_embeddings)
        
        print("\nSample similarity tests:")
        for i, text in enumerate(sample_texts):
            D, I = index.search(sample_embeddings[i:i+1], 3)
            print(f"'{text}' top 3 similarities: {[f'{s:.3f}' for s in D[0]]}")
    else:
        print("Embeddings are up to date. Skipping computation.")