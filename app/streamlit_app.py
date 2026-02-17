import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.cf_model import CFModel
from src.content_model import ContentModel
from src.cold_start import ColdStart
from src.hybrid import HybridRecommender

MOVIES_PATH = "data/movies_final.csv"
RATINGS_PATH = "data/ratings_final.csv"

movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

cf = CFModel("models/svd_model.pkl")
content = ContentModel(
    MOVIES_PATH,
    "models/tfidf_vectorizer.pkl",
    "models/content_matrix.npy"
)
hybrid = HybridRecommender(content, cf, ratings, alpha=0.7)
cold = ColdStart(MOVIES_PATH)

st.title("Movie Recommendation Engine")

mode = st.radio("Select mode", ["Existing User", "Cold Start"])

if mode == "Existing User":
    user_id = st.number_input("User ID", min_value=1, step=1)
    top_n = st.slider("Top-N recommendations", 5, 20, 10)

    if st.button("Get Recommendations"):
        recs = hybrid.recommend_top_n_diverse(user_id, movies, n=top_n)

        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            st.subheader("Recommended Movies")
            for _, row in recs.iterrows():
                with st.expander(f"{row.title}"):
                    st.write(f"**Genres:** {row.genres}")
                    reasons = content.explain(row.movieId)
                    st.write("**Why recommended:**", ", ".join(reasons))

else:
    genres = st.multiselect(
        "Select preferred genres",
        sorted({g for gs in movies.genres.dropna() for g in gs.split("|")})
    )

    if st.button("Recommend") and genres:
        st.subheader("Cold-Start Recommendations")
        st.dataframe(cold.recommend(genres))
