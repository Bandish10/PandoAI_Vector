# app.py

import streamlit as st
import os
import numpy as np
import pandas as pd
import faiss
import altair as alt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
import umap
import hdbscan

# Paths
root_dir = "sapbert-mpnet"
model_dir = os.path.join(root_dir, "model")
embed_dir = os.path.join(root_dir, "embeddings")
index_dir = os.path.join(root_dir, "faiss_index")
excel_path = "Vector.xlsx"

# ---- Load Model and Data ----
@st.cache_resource
def load_model():
    return SentenceTransformer(model_dir)

@st.cache_data
def load_data():
    df = pd.read_excel(excel_path)
    index = faiss.read_index(os.path.join(index_dir, "faiss_index.index"))
    return df, index

@st.cache_data
def load_embeddings(_model, df):
    embed_path = os.path.join(embed_dir, "embeddings.npy")
    if os.path.exists(embed_path):
        return np.load(embed_path)
    else:
        texts = df.apply(lambda row: ' | '.join([str(v) for v in row.values if pd.notnull(v)]), axis=1)
        embeddings = _model.encode(texts.tolist(), convert_to_numpy=True, show_progress_bar=True)
        os.makedirs(embed_dir, exist_ok=True)
        np.save(embed_path, embeddings)
        return embeddings

@st.cache_data
def compute_umap_hdbscan(embeddings, min_cluster_size, n_neighbors, min_dist):
    scaled = StandardScaler().fit_transform(embeddings)
    umap_embed = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42).fit_transform(scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10).fit(umap_embed)
    return umap_embed, clusterer

def create_intuitive_cluster_label(cluster_id, examples, df_subset):
    """Create more intuitive and readable cluster labels"""
    if cluster_id == -1:
        return "Outlier/Noise"
    
    if len(examples) == 0:
        return f"Cluster {cluster_id}"
    
    # Get descriptions for this cluster
    descriptions = df_subset[df_subset['Cluster'] == cluster_id]['Description'].tolist()
    
    if len(descriptions) == 0:
        return f"Cluster {cluster_id}"
    
    # Method 1: Find common keywords in the cluster
    all_words = []
    for desc in descriptions[:20]:  # Use top 20 for better representation
        words = str(desc).lower().split()
        all_words.extend(words)
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(all_words)
    
    # Remove common medical stop words and get top meaningful words
    stop_words = {'for', 'the', 'and', 'of', 'in', 'to', 'with', 'on', 'by', 'at', 'as', 'or', 'is', 'are', 'be', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'package', 'procedure', 'service', 'services'}
    
    meaningful_words = [(word, count) for word, count in word_counts.most_common(20) if len(word) > 2 and word not in stop_words]
    
    # Try to create a meaningful label
    if meaningful_words:
        # Take top 2-3 most common meaningful words
        top_words = [word for word, count in meaningful_words[:3]]
        label_base = " ".join(top_words[:2]).title()  # Max 2 words, capitalized
    else:
        # Fallback to first description
        first_desc = str(descriptions[0])
        words = first_desc.split()
        label_base = " ".join(words[:3]).title()  # First 3 words
    
    # Truncate if too long
    if len(label_base) > 25:
        label_base = label_base[:22] + "..."
    
    return f"C{cluster_id}: {label_base}"

def label_clusters(embeddings, df, clusterer):
    df['Cluster'] = clusterer.labels_
    cluster_names = {}
    
    # Get unique clusters
    unique_clusters = sorted([c for c in set(clusterer.labels_) if c != -1])  # Exclude outliers for numbering
    outlier_exists = -1 in set(clusterer.labels_)
    
    # Handle outliers first if they exist
    if outlier_exists:
        cluster_names[-1] = "Outlier/Noise"
    
    # Create intuitive labels for regular clusters
    for c in unique_clusters:
        indices = np.where(clusterer.labels_ == c)[0]
        examples = df.iloc[indices]['Description'].tolist()[:10]  # Use more examples for better labeling
        cluster_names[c] = create_intuitive_cluster_label(c, examples, df)
    
    df['Cluster Label'] = df['Cluster'].map(cluster_names)
    return df['Cluster Label'], cluster_names

def search_faiss(_model, index, df, query, k=5):
    # Get query embedding
    query_embedding = _model.encode([query], convert_to_numpy=True)
    
    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search using normalized vectors
    D, I = index.search(query_embedding, k)
    
    # Get results
    results = df.iloc[I[0]].copy()
    
    # D already contains cosine similarities (between -1 and 1) from FAISS
    results["Similarity Score"] = D[0]
    
    return results

# ---- Streamlit UI ----
st.set_page_config(page_title="PandoAI - Medical Vector Explorer", layout="wide")
st.title("🧠 PandoAI - Medical Vector Explorer")

# Dark mode theme with better contrast
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #0e1117;
    }
    [data-testid=stAppViewContainer] {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid=stMarkdownContainer] {
        color: #fafafa;
    }
    .stDataFrame {
        color: #000000;
    }
    /* Fix for dropdown text color */
    .stSelectbox > div > div {
        color: #ffffff !important;
        background-color: #262730 !important;
    }
    .stSelectbox label {
        color: #fafafa !important;
    }
    .stSlider label {
        color: #fafafa !important;
    }
    .stTextInput label {
        color: #fafafa !important;
    }
    /* Wider dropdowns */
    .stSelectbox div[data-baseweb="select"] {
        min-width: 300px;
    }
    /* Better cluster label display */
    .cluster-label {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

model = load_model()
df, index = load_data()
embeddings = load_embeddings(model, df)

# Initialize session state for clustering
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
    st.session_state.cluster_data = None
    st.session_state.cluster_labels = None

tab1, tab2 = st.tabs(["🔍 Search", "🧬 Clustering"])

with tab1:
    st.header("🔍 Search Procedures")
    query = st.text_input("Enter medical term or procedure:")
    k = st.slider("Number of top results", 3, 15, 5)
    if query:
        with st.spinner("Searching..."):
            results = search_faiss(model, index, df, query, k)
        st.subheader("🔎 Top Matches")
        st.dataframe(results)
        st.altair_chart(
            alt.Chart(results.reset_index()).mark_bar().encode(
                x=alt.X('Description:N', sort='-y'),
                y=alt.Y('Similarity Score:Q')
            ).properties(width=800),
            use_container_width=True
        )
        st.download_button(
            "📥 Download",
            results.to_csv(index=False).encode(),
            "results.csv",
            "text/csv"
        )

with tab2:
    st.header("🔗 Semantic Clustering")
    
    st.subheader("⚙️ Clustering Parameters")
    
    # Adjusted parameter ranges for better cluster control
    col1, col2, col3 = st.columns(3)
    with col1:
        min_cluster_size = st.slider("Min Cluster Size", 50, 300, 100, 
                                   help="Minimum points to form a cluster. Higher = fewer, larger clusters (2-20 target).")
    with col2:
        n_neighbors = st.slider("UMAP Neighbors", 20, 100, 50,
                              help="Neighborhood size for manifold learning. Higher = more global structure.")
    with col3:
        min_dist = st.slider("UMAP Min Distance", 0.0, 1.0, 0.5,
                           help="Minimum separation between embedded points. Higher = more separated clusters.")
    
    run_cluster = st.button("Run UMAP + HDBSCAN Clustering")
    
    if run_cluster:
        with st.spinner("Running clustering..."):
            scaled = StandardScaler().fit_transform(embeddings)
            umap_embed = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42).fit_transform(scaled)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=15).fit(umap_embed)
            
            df['X'], df['Y'] = umap_embed[:, 0], umap_embed[:, 1]
            df['Cluster Label'], cluster_map = label_clusters(embeddings, df, clusterer)
            
            # Store in session state
            st.session_state.clustering_done = True
            st.session_state.cluster_data = df.copy()
            st.session_state.cluster_labels = cluster_map
            st.session_state.umap_data = umap_embed
            st.session_state.clusterer = clusterer
            
            # Show cluster count
            unique_clusters = len([c for c in set(clusterer.labels_) if c != -1])
            outlier_count = sum(1 for c in clusterer.labels_ if c == -1)
            st.success(f"Clustering complete! Found {unique_clusters} clusters and {outlier_count} outliers.")

    # Display clustering results if available
    if st.session_state.clustering_done and st.session_state.cluster_data is not None:
        df_display = st.session_state.cluster_data
        
        st.subheader("📊 2D Cluster Visualization")
        
        # Create ordered legend for chart
        cluster_labels_ordered = sorted(df_display['Cluster Label'].unique(), 
                                      key=lambda x: (x.startswith('Outlier'), 
                                                   int(x.split(':')[0][1:]) if ':' in x and x.split(':')[0][1:].isdigit() else 999999))
        
        chart = alt.Chart(df_display).mark_circle(size=60).encode(
            x='X:Q',
            y='Y:Q',
            color=alt.Color('Cluster Label:N', sort=cluster_labels_ordered),
            tooltip=['Code', 'Description', 'Billing Group', 'Surgery Level', 'Cluster Label']
        ).properties(width=1000, height=500).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.subheader("🔍 Explore Cluster")
        
        # Create better ordered cluster selection
        cluster_options = sorted(df_display['Cluster Label'].unique(), 
                               key=lambda x: (x.startswith('Outlier'), 
                                            int(x.split(':')[0][1:]) if ':' in x and x.split(':')[0][1:].isdigit() else 999999))
        
        # Ensure wide dropdown
        selected = st.selectbox("Select Cluster", cluster_options, key="cluster_select")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            billing = st.selectbox("Billing Group", ["All"] + sorted(df_display['Billing Group'].dropna().unique().tolist()))
        with col2:
            surgery = st.selectbox("Surgery Level", ["All"] + sorted(df_display['Surgery Level'].dropna().unique().tolist()))
        with col3:
            keyword = st.text_input("Keyword filter:")
        
        # Filter data
        filtered = df_display[df_display['Cluster Label'] == selected]
        if billing != "All":
            filtered = filtered[filtered['Billing Group'] == billing]
        if surgery != "All":
            filtered = filtered[filtered['Surgery Level'] == surgery]
        if keyword:
            filtered = filtered[filtered['Description'].str.contains(keyword, case=False, na=False)]

        st.dataframe(filtered)
        st.download_button(
            "📥 Download Cluster View",
            filtered.to_csv(index=False).encode(),
            f"cluster_{selected.replace('/', '_').replace(':', '_').replace(' ', '_')}.csv",
            "text/csv",
            key="download_cluster"
        )
        st.write(f"Points in cluster: {len(filtered)}")
        if len(filtered) > 0:
            st.write("Top Billing Groups:", filtered['Billing Group'].value_counts().head(3))
            st.write("Top Surgery Levels:", filtered['Surgery Level'].value_counts().head(3))