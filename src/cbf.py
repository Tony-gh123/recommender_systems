import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def _item_norms(X_tfidf: sparse.csr_matrix) -> np.ndarray:
    """
    Compute the L2 norms of item vectors in the TF-IDF matrix.
    """
    # Compute the L2 norm of each item vector
    return np.sqrt(X_tfidf.multiply(X_tfidf).sum(axis=1)).A1 + 1e-12

def score_cbf_last_item_candidates(
    last_item_indices: np.ndarray,
    candidates: np.ndarray,
    X_tfidf: sparse.csr_matrix,
) -> np.ndarray:
    """
    Score candidate items based on cosine similarity to the last item in user history.
    """
    # Get the number of users and candidates
    n_users, n_cand = candidates.shape
    # Initialize the output score matrix with zeros
    out = np.zeros((n_users, n_cand), dtype=np.float32)

    # Compute the item norms
    item_norm = _item_norms(X_tfidf)
    # Identify valid last item indices
    valid = last_item_indices >= 0
    # Check if there are any valid indices
    if not valid.any():
        # Return the empty output if no valid indices
        return out

    # Iterate over valid user indices
    for row in np.where(valid)[0]:
        # Get the index of the last item for the user
        li = int(last_item_indices[row])
        # Get the candidate items for the user
        c = candidates[row]
        # Get the TF-IDF vector for the last item
        x = X_tfidf[li]
        # Compute the dot product between the last item and candidates
        dots = (x @ X_tfidf[c].T).toarray().ravel().astype(np.float32)
        # Normalize the scores by item norms
        out[row] = dots / (item_norm[li] * item_norm[c])

    # Return the computed scores
    return out

def score_cbf_last_item(
    users: np.ndarray,
    candidates: np.ndarray,
    user_hist: dict[int, list[int]],
    item2idx: dict[int, int],
    X_tfidf: sparse.csr_matrix,
):
    """
    Score candidate items based on cosine similarity to the last item in user history.
    """
    # Initialize a list for last item indices
    last_idx = []
    # Iterate over each user
    for u in users:
        # Get the history for the user
        hist = user_hist.get(int(u), [])
        # Check if the history is empty
        if len(hist) == 0:
            # Append -1 if history is empty
            last_idx.append(-1)
        else:
            # Get the index of the last item in history
            li = item2idx.get(hist[-1], -1)
            # Append the last item index
            last_idx.append(li)
    # Convert the list to a numpy array
    last_idx = np.array(last_idx, dtype=np.int32)
    # Compute and return the scores
    return score_cbf_last_item_candidates(last_idx, candidates, X_tfidf)
