import pandas as pd
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# Detect project root directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths (adjusted for your structure)
MOVIES_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "cleaned_movies.csv")


# Load cleaned movie data
df = pd.read_csv(CLEANED_PATH)

# Load TF-IDF matrix and vectorizer
tfidf_matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

print("✅ Data and vectorizer loaded successfully!")
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)


# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("✅ Cosine similarity matrix created!")
print("Matrix shape:", cosine_sim.shape)


def recommend(movie_name):
    # Ensure lowercase matching
    movie_name = movie_name.lower()

    # Check if movie exists
    if movie_name not in df['title'].values:
        return f"❌ Movie '{movie_name}' not found in dataset."

    # Get the index of the movie
    idx = df[df['title'] == movie_name].index[0]

    # Get pairwise similarity scores for this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 most similar movies (excluding itself)
    top_movies = sim_scores[1:6]  # skip first (self)

    # Display recommendations
    print(f"\n🎬 Because you liked '{movie_name.title()}', you might also enjoy:\n")
    for i, (index, score) in enumerate(top_movies, start=1):
        print(f"{i}. {df.iloc[index]['title'].title()}  (Score: {round(score, 3)})")

    return [df.iloc[index]['title'] for index, _ in top_movies]


if __name__ == "__main__":
    movie = input("Enter a movie name: ")
    recommend(movie)
