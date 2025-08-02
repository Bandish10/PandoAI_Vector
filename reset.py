# reset_and_recompute.py

import os
import shutil

def reset_embeddings():
    """Delete existing embeddings and index to force recomputation"""
    
    # Remove embeddings directory
    embed_dir = "sapbert-mpnet/embeddings"
    if os.path.exists(embed_dir):
        shutil.rmtree(embed_dir)
        print(f"Deleted {embed_dir}")
    
    # Remove index directory
    index_dir = "sapbert-mpnet/faiss_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        print(f"Deleted {index_dir}")
    
    print("All embeddings and indices have been removed.")
    print("Run 'python convert.py' to recompute with proper normalization.")

if __name__ == "__main__":
    reset_embeddings()