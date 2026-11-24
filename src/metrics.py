"""Information retrieval and ranking evaluation metrics.

Functions for calculating Precision@K, Mean Reciprocal Rank (MRR@K),
and normalized Discounted Cumulative Gain (nDCG@K) using human relevance judgments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def metrics_at_k(grades: Sequence[int], k: int = 5, include_ndcg_alias: bool = True) -> dict[str, float]:
    """Calculate Precision@K, MRR@K, and judged-pool nDCG@K for ranked predictions.

    Parameters
    ----------
    grades : Sequence[int]
        Relevance grades ordered by rank (all elements must be in {0, 1, 2}).
        0 = irrelevant, 1 = partially relevant, 2 = highly relevant.
    k : int, default 5
        Cutoff rank (number of top recommendations).
    include_ndcg_alias : bool, default True
        Whether to include both `ndcg@{k}_judged_pool` and `ndcg@{k}` keys.

    Returns
    -------
    dict[str, float]
        Dictionary with precision@k, mrr@k, and ndcg@k (and ndcg@k_judged_pool).
    """
    vals = np.asarray(grades, dtype=int)
    if len(vals) != k or not np.isin(vals, [0, 1, 2]).all():
        raise ValueError(f"Need {k} valid human relevance grades in {{0, 1, 2}}.")

    binary = vals >= 1
    precision = float(binary.mean())

    relevant_positions = np.flatnonzero(binary)
    mrr = float(1 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0

    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((2.0 ** vals - 1) / discounts))
    ideal = np.sort(vals)[::-1]
    idcg = float(np.sum((2.0 ** ideal - 1) / discounts))
    ndcg = dcg / idcg if idcg else 0.0

    res = {
        f"precision@{k}": precision,
        f"mrr@{k}": mrr,
        f"ndcg@{k}_judged_pool": ndcg,
    }
    if include_ndcg_alias:
        res[f"ndcg@{k}"] = ndcg

    return res
