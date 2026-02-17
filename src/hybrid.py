from src.diversity import mmr_rerank

class HybridRecommender:
    def __init__(self, content_model, cf_model, ratings_df, alpha=0.7):
        self.content = content_model
        self.cf = cf_model
        self.ratings = ratings_df
        self.alpha = alpha

    def score(self, user_id, movie_id):
        cf_score = self.cf.predict(user_id, movie_id)
        content_score = self.content.score(user_id, movie_id, self.ratings)
        return self.alpha * cf_score + (1 - self.alpha) * content_score

    def recommend_top_n_diverse(self, user_id, movies_df, n=10, pre_k=50):
        rated = self.ratings[self.ratings.userId == user_id].movieId.values
        candidates = movies_df[~movies_df.movieId.isin(rated)]

        scored = [
            (row.movieId, self.score(user_id, row.movieId))
            for _, row in candidates.iterrows()
        ]

        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:pre_k]

        ids = [m for m, _ in scored]
        scores = [s for _, s in scored]

        reranked_ids = mmr_rerank(
            ids,
            scores,
            self.content.content_matrix,
            k=n
        )

        return movies_df[movies_df.movieId.isin(reranked_ids)]
