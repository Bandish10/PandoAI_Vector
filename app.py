import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import altair as alt
import os

# ---- PATH SETUP ----
root_dir = "C:/Users/bandi/Desktop/PandoAI Cost/Vector/Web/sapbert-mpnet"
model_dir = os.path.join(root_dir, "model")
embed_dir = os.path.join(root_dir, "embeddings")
index_dir = os.path.join(root_dir, "faiss_index")
excel_path = "Vector.xlsx"

# ---- CACHING ----
@st.cache_resource
def load_model():
    return SentenceTransformer(model_dir)

@st.cache_data
def load_data():
    df = pd.read_excel(excel_path)
    index = faiss.read_index(os.path.join(index_dir, "faiss_index.index"))
    embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))
    return df, index

# ---- SEARCH FUNCTION ----
def search_faiss(model, index, df, query, k=5):
    embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(embedding, k)
    results = df.iloc[I[0]].copy()
    results["Similarity Score"] = 1 - D[0] / 2  # Approx. cosine similarity
    return results

# ---- UI ----
st.set_page_config(page_title="PandoAI - Medical Vector Search", layout="wide")
st.title("🧠 PandoAI - Medical Procedure Vector Search")

query = st.text_input("🔍 Enter a medical term, abbreviation, or procedure", placeholder="e.g. TKR, CABG, Lap Chole, PCI")
k = st.slider("Number of top results", min_value=3, max_value=15, value=5)

if query:
    model = load_model()
    df, index = load_data()

    with st.spinner("Searching..."):
        results = search_faiss(model, index, df, query, k)

    st.subheader("🔎 Top Matches")
    st.dataframe(results, use_container_width=True)

    # ---- CHART ----
    st.subheader("📊 Similarity Scores")
    chart = alt.Chart(results.reset_index()).mark_bar().encode(
        x=alt.X('Description:N', sort='-y', title="Procedure Description"),
        y=alt.Y('Similarity Score:Q'),
        tooltip=['Code', 'Description', 'Similarity Score']
    ).properties(width=800, height=300)
    st.altair_chart(chart, use_container_width=True)

    # ---- DOWNLOAD ----
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, "results.csv", "text/csv")
else:
    st.info("Enter a query above to start searching.")
