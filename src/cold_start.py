import pandas as pd

class ColdStart:
    def __init__(self, movies_path):
        self.movies = pd.read_csv(movies_path)

    def recommend(self, genres, top_k=10):
        mask = self.movies["genres"].str.contains("|".join(genres), regex=True)
        return (
            self.movies[mask]
            .sort_values(["avg_rating", "popularity_score"], ascending=False)
            .head(top_k)[["title", "genres", "avg_rating"]]
        )