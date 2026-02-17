import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class ContentModel:
    def __init__(self, movies_path, tfidf_path, matrix_path):
        self.movies = pd.read_csv(movies_path)
        self.id_to_idx = {
            mid: i for i, mid in enumerate(self.movies["movieId"].values)
        }
        self.tfidf = joblib.load(tfidf_path)
        self.content_matrix = np.load(matrix_path)

    def build_user_profile(self, user_id, ratings_df):
        liked = ratings_df[
            (ratings_df.userId == user_id) & (ratings_df.rating >= 4)
        ]
        if liked.empty:
            return None
        idx = [self.id_to_idx[mid] for mid in liked.movieId if mid in self.id_to_idx]
        if not idx:
            return None
        return self.content_matrix[idx].mean(axis=0).reshape(1, -1)

    def score(self, user_id, movie_id, ratings_df):
        profile = self.build_user_profile(user_id, ratings_df)
        if profile is None:
            return 0.0
        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            return 0.0
        return cosine_similarity(
            profile,
            self.content_matrix[idx].reshape(1, -1)
        )[0][0]

    def explain(self, movie_id, top_k=5):
        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            return []
        vec = self.content_matrix[idx]
        terms = self.tfidf.get_feature_names_out()
        top = np.argsort(vec[:len(terms)])[-top_k:][::-1]
        return terms[top].tolist()
