"""Solve three signed lip gaps (% of neutral mouth width) with 1–3 opening keys."""
from itertools import product

import numpy as np


def bounded_weights(neutral, response, target):
    """Solve min ||response @ w - (target - neutral)||^2 with 0 <= w_i <= 1."""
    matrix = np.asarray(response, dtype=float)
    rhs = np.asarray(target, dtype=float) - neutral
    if matrix.ndim != 2 or matrix.shape[0] != 3 or not 1 <= matrix.shape[1] <= 3:
        raise ValueError("Expected three landmarks and one to three controls")
    if rhs.shape != (3,) or not (np.isfinite(rhs).all() and np.isfinite(matrix).all()):
        raise ValueError("Expected three finite gaps and a finite response")
    best_error = np.inf
    for state in product((-1, 0, 1), repeat=matrix.shape[1]):
        free = np.array(state) == -1
        weights = np.maximum(state, 0).astype(float)
        if free.any():
            weights[free] = np.linalg.lstsq(matrix[:, free], rhs - matrix @ weights, rcond=None)[0]
        if np.any(weights < -1e-10) or np.any(weights > 1 + 1e-10):
            continue
        weights = weights.clip(0, 1)
        error = np.linalg.norm(matrix @ weights - rhs)
        if error < best_error:
            best_error, best = error, weights
    return best
