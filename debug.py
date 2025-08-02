# debug_similarity.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def debug_similarity():
    # Load data and model
    excel_path = "Vector.xlsx"
    df = pd.read_excel(excel_path)
    
    model_dir = "sapbert-mpnet/model"
    model = SentenceTransformer(model_dir)
    
    # Load embeddings and index
    embed_path = "sapbert-mpnet/embeddings/embeddings.npy"
    index_path = "sapbert-mpnet/faiss_index/faiss_index.index"
    
    embeddings = np.load(embed_path)
    index = faiss.read_index(index_path)
    
    print(f"Loaded {len(embeddings)} embeddings of shape {embeddings.shape}")
    print(f"Index type: {type(index)}")
    
    # Test queries
    test_queries = ["TKR", "Total Knee Replacement", "CABG", "Kidney"]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        
        # Get query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Normalize
        faiss.normalize_L2(query_embedding)
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Search
        D, I = index.search(query_embedding, 5)
        
        print("Top 5 results:")
        for i in range(5):
            idx = I[0][i]
            similarity = D[0][i]
            desc = df.iloc[idx]['Description']
            print(f"  {similarity:.6f}: {desc}")
        
        # Manual cosine similarity calculation for verification
        manual_sim = np.dot(embeddings[I[0][0:1]], query_embedding.T)[0][0]
        print(f"Manual calculation for top result: {manual_sim:.6f}")

if __name__ == "__main__":
    debug_similarity()