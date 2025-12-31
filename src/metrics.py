import numpy as np

def eval_hr_ndcg(
    scores: np.ndarray,
    ks=(10, 20),
    pos_col: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Evaluate Hit Rate (HR), Normalized Discounted Cumulative Gain (NDCG), and Mean Reciprocal Rank (MRR)
    for the given scores.
    """
    # Check if a random number generator is provided
    if rng is None:
        # Create a default random number generator if none is provided
        rng = np.random.default_rng()
    
    # Add small random noise to scores to break ties randomly
    noise = rng.uniform(0, 1e-12, size=scores.shape)
    # Add the noise to the original scores
    scores_noisy = scores + noise
    
    # Extract the scores of the positive items
    pos_scores = scores_noisy[:, pos_col][:, None]
    # Calculate the rank of positive items by counting how many items have higher scores
    pos_ranks = (scores_noisy > pos_scores).sum(axis=1)
    
    # Initialize a dictionary to store metrics
    metrics = {}
    # Iterate over each k value
    for k in ks:
        # Check if the positive item is within the top k
        hits = (pos_ranks < k)
        # Calculate the Hit Rate at k
        metrics[f"HR@{k}"] = float(hits.mean())
        
        # Initialize an array for NDCG scores
        ndcg = np.zeros_like(hits, dtype=float)
        # Calculate NDCG for hits
        ndcg[hits] = 1.0 / np.log2(pos_ranks[hits] + 2)
        # Calculate the mean NDCG at k
        metrics[f"NDCG@{k}"] = float(ndcg.mean())
        
    # Calculate the Mean Reciprocal Rank
    metrics["MRR"] = float((1.0 / (pos_ranks + 1)).mean())
        
    # Return the computed metrics
    return metrics

def get_user_metrics(
    scores: np.ndarray,
    k: int = 10,
    pos_col: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Return raw per-user metric arrays for statistical testing.
    Returns a dict with keys 'HR', 'NDCG', 'MRR', each containing a 1D numpy array of length n_users.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Add noise to break ties
    noise = rng.uniform(0, 1e-12, size=scores.shape)
    scores_noisy = scores + noise
    
    # Rank positive items
    pos_scores = scores_noisy[:, pos_col][:, None]
    pos_ranks = (scores_noisy > pos_scores).sum(axis=1)
    
    # Calculate per-user metrics
    hits = (pos_ranks < k).astype(float)
    
    ndcg = np.zeros_like(hits, dtype=float)
    hits_mask = (pos_ranks < k)
    ndcg[hits_mask] = 1.0 / np.log2(pos_ranks[hits_mask] + 2)
    
    mrr = 1.0 / (pos_ranks + 1)
    
    return {
        "HR": hits,
        "NDCG": ndcg,
        "MRR": mrr
    }
