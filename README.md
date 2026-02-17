# Movie Recommendation Engine (Advanced Hybrid System)

An industry-style **hybrid movie recommendation system** combining **Collaborative Filtering**, **Content-Based Filtering**, and **Diversity Re-ranking**, built using Python and deployed with Streamlit.

---

## Features

- **Collaborative Filtering (CF)** using Matrix Factorization (SVD)
- **Content-Based Filtering** using TF-IDF (genres + metadata)
- **Hybrid Recommendation Engine** (CF + Content scoring)
- **Cold-Start Handling** for new users using genre preferences
- **MMR Diversity Re-ranking** to reduce redundancy in Top-N results
- **Explainable Recommendations** (“Why recommended” keywords)
- **Interactive Streamlit Web App**

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, scikit-surprise, Streamlit
- **Models:** SVD (Collaborative Filtering), TF-IDF (Content)
- **Deployment:** Streamlit Cloud
- **Tools:** Google Colab, VS Code, GitHub

---

## Project Structure

Movie-Recommendation-Engine/
│
├── app/
│ └── streamlit_app.py # Streamlit UI
│
├── data/
│ ├── movies_final.csv
│ └── ratings_final.csv
│
├── models/
│ ├── svd_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── content_matrix.npy
│
├── src/
│ ├── cf_model.py # Collaborative Filtering
│ ├── content_model.py # Content-Based Model
│ ├── hybrid.py # Hybrid Recommender
│ ├── cold_start.py # Cold-start logic
│ ├── diversity.py # MMR diversity re-ranking
│ └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore

---

## System Architecture

1. **Collaborative Filtering:**  
   Predicts user–movie affinity using latent factors (SVD).

2. **Content-Based Filtering:**  
   Computes similarity using TF-IDF features derived from genres and metadata.

3. **Hybrid Ranking:**  
   Final score = weighted combination of CF score and content similarity.

4. **Diversity Re-ranking (MMR):**  
   Reduces similarity between recommended items to increase novelty.

---

## Cold-Start Strategy

- **New Users:** Genre-based preference selection
- **New Items:** Content-only similarity fallback

---

## Evaluation Metrics

Evaluated using user-stratified train/test split:

| Model | Precision@10 | Recall@10 | nDCG@10 | MAP@10 |
|------|-------------|-----------|---------|--------|
| CF Only | ✓ | ✓ | ✓ | ✓ |
| Content Only | ✓ | ✓ | ✓ | ✓ |
| **Hybrid** | **Best** | **Best** | **Best** | **Best** |

Hybrid model consistently outperformed individual approaches.

---

## Diversity Verification

- Applied **Maximal Marginal Relevance (MMR)**
- Reduced **intra-list similarity**
- Ensured relevance–novelty trade-off

---

## How to Run Locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Streamlit app
streamlit run app/streamlit_app.py

