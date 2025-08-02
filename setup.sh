#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Precompute embeddings and FAISS index
python convert.py

# Run Streamlit app
streamlit run app.py