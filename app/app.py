# ==========================================================
# 🎬 Content-Based Movie Recommendation System (Frontend)
# 💻 Day 6 — Streamlit Interface
# ==========================================================

import streamlit as st
import pandas as pd
import pickle
import os

# ==========================================================
# 📂 Path Setup (Dynamic Detection)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_movies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "tfidf_matrix_v2.pkl")
SIM_PATH = os.path.join(BASE_DIR, "src", "models", "cosine_similarity_v2.pkl")

# ==========================================================
# 📦 Load Data & Models
# ==========================================================
@st.cache_resource
def load_models_and_data():
    df = pd.read_csv(DATA_PATH)
    with open(MODEL_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(SIM_PATH, "rb") as f:
        cosine_sim = pickle.load(f)
    return df, tfidf_matrix, cosine_sim

df, tfidf_matrix, cosine_sim = load_models_and_data()

# ==========================================================
# 🎯 Recommendation Function
# ==========================================================
def recommend(movie_name):
    movie_name = movie_name.lower().strip()

    # Handle missing movie
    if movie_name not in df['title'].str.lower().values:
        return []

    # Find movie index
    idx = df[df['title'].str.lower() == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_movies = sim_scores[1:6]
    return [df.iloc[i[0]]['title'] for i in top_movies]

# ==========================================================
# 🖥️ Streamlit UI Layout
# ==========================================================
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Helvetica', sans-serif;
        }
        h1, h2, h3 {
            color: #FF4B4B;
        }
        .movie-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 0 5px rgba(255,255,255,0.1);
            text-align: center;
        }
        .movie-title {
            font-size: 16px;
            font-weight: 600;
            color: #FAFAFA;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Find movies similar in **genre**, **director**, or **description** — powered by TF-IDF + Cosine Similarity!")

# --- Input Section ---
movie_name = st.text_input("🔍 Enter a movie name:")

# --- Recommend Button ---
if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("⚠️ Please enter a valid movie name.")
    else:
        recommendations = recommend(movie_name)

        if len(recommendations) == 0:
            st.error(f"❌ Movie '{movie_name.title()}' not found in the database.")
        else:
            st.success(f"🎥 Movies similar to **{movie_name.title()}**:")
            st.markdown("---")

            # Display recommendations in card layout
            cols = st.columns(5)
            for i, movie in enumerate(recommendations):
                with cols[i % 5]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            🎞️ <div class="movie-title">{movie}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ==========================================================
# 🧩 Footer
# ==========================================================
st.markdown("---")
st.caption("🚀 Built with ❤️ using Streamlit | Dunith")
