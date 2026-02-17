import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_rerank(candidate_ids, scores, content_matrix, id_to_idx, lambda_div=0.7, k=10):
    selected = []
    candidate_ids = list(candidate_ids)
    scores = list(scores)

    k = min(k, len(candidate_ids))

    while len(selected) < k and candidate_ids:
        mmr_scores = []

        for i, mid in enumerate(candidate_ids):
            idx = id_to_idx.get(mid)
            if idx is None:
                mmr_scores.append(-np.inf)
                continue

            relevance = scores[i]

            if not selected:
                diversity_penalty = 0.0
            else:
                selected_idx = [id_to_idx[s] for s in selected if s in id_to_idx]
                sims = cosine_similarity(
                    content_matrix[idx].reshape(1, -1),
                    content_matrix[selected_idx]
                )
                diversity_penalty = sims.max()

            mmr = lambda_div * relevance - (1 - lambda_div) * diversity_penalty
            mmr_scores.append(mmr)

        best_idx = int(np.argmax(mmr_scores))
        selected.append(candidate_ids.pop(best_idx))
        scores.pop(best_idx)

    return selected
