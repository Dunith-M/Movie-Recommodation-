import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Detect project root directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths (adjusted for your structure)
MOVIES_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "cleaned_movies.csv")

# Load cleaned dataset
df = pd.read_csv(CLEANED_PATH)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())



# Combine all text features
def combine_features(row):
    return f"{row['genres']} {row.get('director', '')} {row['overview']}"

df['combined'] = df.apply(combine_features, axis=1)

print("\n✅ Combined text features created!")
print(df['combined'].head(3))


# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    stop_words='english',      # removes common English words like "the", "and", "is"
    max_features=5000          # limit vocabulary size (you can adjust)
)


# Fit the vectorizer on combined text
tfidf_matrix = tfidf.fit_transform(df['combined'])

print("\n✅ TF-IDF matrix created successfully!")
print("Shape of matrix:", tfidf_matrix.shape)


# Check the Vocabulary
print("\nSample feature names:")
print(tfidf.get_feature_names_out()[:20])


count_vec = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vec.fit_transform(df['combined'])

print("\nCount matrix shape:", count_matrix.shape)




# Save TF-IDF vectorizer and feature matrix
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))
pickle.dump(tfidf_matrix, open("models/tfidf_matrix.pkl", "wb"))

print("✅ TF-IDF vectorizer and matrix saved in /models/")