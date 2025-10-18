# src/data_cleaning.py

import os
import pandas as pd


DATA_PATH = os.path.join("..","data", "tmdb_5000_movies.csv")


# Load the dataset
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")
    print(f"📊 Shape of dataset: {df.shape}")
    print("\n📋 First 3 rows:")
    print(df.head(3))
except FileNotFoundError:
    print("❌ Error: Dataset file not found!")
    print("Please make sure 'tmdb_5000_movies.csv' is inside the 'data/' folder.")
