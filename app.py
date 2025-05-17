# =============================
# 🛡️ Contournement du bug Streamlit + torch
# =============================
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# =============================
# 📦 Imports
# =============================
import streamlit as st
import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# =============================
# 1️⃣ - Chargement des modèles
# =============================
st.title("🔎 Search Documentation Recommender")

# Chargement du modèle SBERT
st.sidebar.subheader("Chargement du modèle...")
model = SentenceTransformer('all-MiniLM-L6-v2')
#with open('sbert_model.pkl', 'rb') as f:
    #model = pickle.load(f)
st.sidebar.success("Modèle SBERT chargé !")

# Chargement de l'index Faiss
st.sidebar.subheader("Chargement de l'index...")
index = faiss.read_index('faiss_index.index')
st.sidebar.success("Index Faiss chargé !")

# Chargement de ton fichier CSV
st.sidebar.subheader("Chargement des articles...")
documents = pd.read_csv('arxiv_articles_nettoye.csv')
st.sidebar.success(f"{len(documents)} articles chargés !")

# =============================
# 2️⃣ - Fonction de recherche
# =============================
def search_similar_documents(query, model, index, documents, top_k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'Titre': documents.iloc[idx]['Titre'],
            'Résumé': documents.iloc[idx]['Résumé'],
            'Lien PDF': documents.iloc[idx]['Lien PDF'],
            'Sujet': documents.iloc[idx]['Sujet'],
            'Distance': distances[0][i]
        })
    return results

# =============================
# 3️⃣ - Interface de recherche
# =============================
st.subheader("🔎 Recherche de documents")
query = st.text_input("Entrez un mot-clé, un sujet ou un résumé :")

if query:
    st.write("Recherche en cours...")
    results = search_similar_documents(query, model, index, documents, top_k=5)

    for res in results:
        st.markdown(f"### 📌 {res['Titre']}")
        st.write(f"**Sujet** : {res['Sujet']}")
        st.write(f"**Résumé** : {res['Résumé']}")
        st.markdown(f"[📄 Lire l'article complet]({res['Lien PDF']})", unsafe_allow_html=True)
        st.divider()

