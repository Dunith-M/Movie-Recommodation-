import pandas as pd
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_engine import recommend

# ==========================================================
# 🎬 DAY 5 — TESTING & TUNING (FINAL & PATH-AWARE VERSION)
# ==========================================================

# 🔍 Detect project root directory automatically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 📂 Auto-detect model directory (works for both /models and /src/models)
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, "models")

# 📁 Data paths (inside root/data)
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_movies.csv")

# ==========================================================
# 1️⃣ Load Data & Previous Model Components
# ==========================================================
print(f"📂 Using model directory: {MODEL_DIR}")

df = pd.read_csv(CLEANED_PATH)

# Validate pickle files
required_files = [
    "tfidf_vectorizer.pkl",
    "tfidf_matrix.pkl",
    "cosine_similarity.pkl"
]
for f in required_files:
    path = os.path.join(MODEL_DIR, f)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"❌ Missing or empty model file: {path}. "
                                "Please run recommendation_engine.py first.")

tfidf = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb"))
tfidf_matrix = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb"))
cosine_sim = pickle.load(open(os.path.join(MODEL_DIR, "cosine_similarity.pkl"), "rb"))

print("✅ All base components loaded successfully!")
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# ==========================================================
# 2️⃣ Qualitative Testing (before tuning)
# ==========================================================
test_movies = ["Inception", "Titanic", "Avengers: Endgame", "The Lion King", "The Godfather"]

print("\n🎬 Baseline Recommendations:")
for movie in test_movies:
    print("\n-------------------------------")
    recommend(movie)

# ==========================================================
# 3️⃣ Data Cleaning & Normalization
# ==========================================================
def clean_text(text):
    """Clean and normalize text data."""
    if pd.isnull(text):  # Handle NaN
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Ensure required columns exist
for col in ['genres', 'overview', 'director']:
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].fillna("")

# Apply cleaning
df['genres'] = df['genres'].apply(clean_text)
df['overview'] = df['overview'].apply(clean_text)
df['director'] = df['director'].apply(clean_text)

print("\n✅ Text cleaned and normalized successfully!")

# ==========================================================
# 4️⃣ Feature Weighting (before vectorization)
# ==========================================================
def weighted_features(row):
    """Apply weights to emphasize genres and directors."""
    genres = (row['genres'] + " ") * 3   # triple weight
    director = (row['director'] + " ") * 2
    overview = row['overview']
    return genres + director + overview

df['combined'] = df.apply(weighted_features, axis=1)
print("✅ Weighted features combined successfully!")

# ==========================================================
# 5️⃣ TF-IDF Vectorization (tuned parameters)
# ==========================================================
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=8000,      # richer vocabulary
    ngram_range=(1, 2),     # capture unigrams + bigrams
    min_df=2,               # ignore rare terms
    max_df=0.9              # ignore overly common ones
)

tfidf_matrix = tfidf.fit_transform(df['combined'])
print("\n✅ Tuned TF-IDF matrix created!")
print("Shape:", tfidf_matrix.shape)

# ==========================================================
# 6️⃣ Compute Cosine Similarity
# ==========================================================
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("✅ Cosine similarity matrix computed!")
print("Matrix shape:", cosine_sim.shape)

# ==========================================================
# 7️⃣ Save Optimized Models (v2 versions)
# ==========================================================
pickle.dump(tfidf, open(os.path.join(MODEL_DIR, "tfidf_vectorizer_v2.pkl"), "wb"))
pickle.dump(tfidf_matrix, open(os.path.join(MODEL_DIR, "tfidf_matrix_v2.pkl"), "wb"))
pickle.dump(cosine_sim, open(os.path.join(MODEL_DIR, "cosine_similarity_v2.pkl"), "wb"))

print(f"\n💾 Optimized models saved successfully in {MODEL_DIR} (v2 versions)")

# ==========================================================
# 8️⃣ Qualitative Testing (after tuning)
# ==========================================================
print("\n🎬 Testing improved recommendations...\n")
for movie in test_movies:
    print("\n-------------------------------")
    recommend(movie)

print("\n🏁 ✅ Day 5 completed — optimized, cleaned, and tuned successfully!")
