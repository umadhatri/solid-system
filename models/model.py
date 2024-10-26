import pandas as pd
import ast
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

# Load datasets
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Check for ID consistency
missing_in_credits = set(movies['id']) - set(credits['id'])
missing_in_movies = set(credits['id']) - set(movies['id'])

print(f"IDs in movies not in credits: {missing_in_credits}")
print(f"IDs in credits not in movies: {missing_in_movies}")

# Keep only the common IDs
common_ids = set(movies['id']).intersection(set(credits['id']))

# Filter datasets to keep only rows with common IDs
movies_filtered = movies[movies['id'].isin(common_ids)]
credits_filtered = credits[credits['id'].isin(common_ids)]

# Merge the filtered datasets
data = pd.merge(movies_filtered, credits_filtered, on='id')

# Preprocess the data
def convert_to_list(text):
    try:
        return ast.literal_eval(text)
    except ValueError:
        return []

data['cast'] = data['cast'].apply(convert_to_list)
data['crew'] = data['crew'].apply(convert_to_list)

# Create a reader object for Surprise
reader = Reader(rating_scale=(1, 5))

# Load the data into Surprise's format
# Assuming you have a 'ratings.csv' with columns: userId, movieId, rating
ratings = pd.read_csv('data/ratings.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm for collaborative filtering
algo = SVD()

# Train the algorithm on the training set
algo.fit(trainset)

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(algo, f)