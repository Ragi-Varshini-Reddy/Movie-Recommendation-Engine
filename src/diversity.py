import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_rerank(candidate_ids, scores, content_matrix, lambda_div=0.7, k=10):
    selected = []
    candidate_ids = list(candidate_ids)

    while len(selected) < k and candidate_ids:
        mmr_scores = []

        for i, mid in enumerate(candidate_ids):
            relevance = scores[i]

            if not selected:
                diversity_penalty = 0
            else:
                sims = cosine_similarity(
                    content_matrix[mid].reshape(1, -1),
                    content_matrix[selected]
                )
                diversity_penalty = sims.max()

            mmr = lambda_div * relevance - (1 - lambda_div) * diversity_penalty
            mmr_scores.append(mmr)

        idx = int(np.argmax(mmr_scores))
        selected.append(candidate_ids.pop(idx))
        scores.pop(idx)

    return selected
