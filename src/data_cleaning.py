import os
import pandas as pd
import ast

# Detect project root directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths (adjusted for your structure)
MOVIES_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_credits.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "data", "cleaned_movies.csv")

# Load datasets
try:
    movies = pd.read_csv(MOVIES_PATH)
    credits = pd.read_csv(CREDITS_PATH)
    print("✅ Datasets loaded successfully!")
except FileNotFoundError:
    print("❌ Dataset not found! Check if the CSV files exist inside the 'data' folder.")
    print(f"Expected:\n{MOVIES_PATH}\n{CREDITS_PATH}")
    exit()

# Select only required columns
df = movies[['title', 'genres', 'overview']].copy()

#Merge with credits dataset on 'title'
credits = credits.rename(columns={'title': 'title', 'crew': 'crew'})
df = df.merge(credits[['title', 'crew']], on='title', how='left')
print("✅ Merged movies and credits data successfully!")

# Extract director names from crew JSON-like field
def get_director(crew_list):
    try:
        crew = ast.literal_eval(crew_list)
        for person in crew:
            if person.get('job') == 'Director':
                return person.get('name')
    except:
        return ''
    return ''

df['director'] = df['crew'].apply(get_director)

# Remove 'crew' column (not needed anymore)
df.drop(columns=['crew'], inplace=True)

# Handle missing data
df.fillna('', inplace=True)
print("\n✅ Filled NaN values successfully!")

# Remove duplicates
df.drop_duplicates(subset='title', inplace=True)
print("\n✅ Removed duplicates. Dataset shape:", df.shape)

# Convert all text columns to lowercase
for col in ['title', 'genres', 'overview', 'director']:
    df[col] = df[col].astype(str).str.lower()

print("\n✅ Text columns normalized to lowercase.")

# Save cleaned dataset
df.to_csv(CLEANED_PATH, index=False)
print(f"\n✅ Cleaned dataset saved successfully at: {CLEANED_PATH}")

# Display preview
print("\n🎬 Sample of cleaned data:")
print(df.head(3))
