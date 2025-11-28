from typing import Iterable, List, Tuple

import numpy as np


def get_pareto_front(f_scores: np.ndarray, r_scores: np.ndarray) -> np.ndarray:
    """Compute Pareto front indices for objectives: maximize F, minimize R.

    We transform to (F, -R) and keep points that are not dominated.

    Args:
        f_scores: Array of Forget sensitivity per neuron, shape (N,).
        r_scores: Array of Retain sensitivity per neuron, shape (N,).

    Returns:
        Indices of points on the Pareto front (nondominated set) as np.ndarray.
    """
    assert f_scores.shape == r_scores.shape
    population = np.vstack((f_scores, -r_scores)).T  # (N, 2)
    indices = np.arange(len(f_scores))

    order = np.lexsort((-population[:, 1], -population[:, 0]))
    pop_sorted = population[order]
    idx_sorted = indices[order]

    pareto: List[int] = []
    best_neg_r = -np.inf
    for i, point in enumerate(pop_sorted):
        if point[1] > best_neg_r:  # strictly improves -R
            pareto.append(idx_sorted[i])
            best_neg_r = point[1]
    return np.array(pareto)


def calculate_hypervolume(f_front: np.ndarray, r_front: np.ndarray, ref_point: Tuple[float, float] | None = None) -> float:
    """Compute 2D hypervolume for a Pareto front with objectives (maximize F, minimize R).

    Convert to (F, G) with G = 1 - R, so both are maximized. In 2D, HV equals the area
    dominated by the front with respect to a reference point.

    Args:
        f_front: F values on the front.
        r_front: R values on the front.
        ref_point: Optional reference point in (F, G) space; if None, use (min(F), min(G)).

    Returns:
        Hypervolume score (larger is better).
    """
    if len(f_front) == 0:
        return 0.0

    # Normalize to [0, 1] per front to avoid scale issues
    f_min, f_max = float(np.min(f_front)), float(np.max(f_front))
    r_min, r_max = float(np.min(r_front)), float(np.max(r_front))
    if f_max == f_min:
        return 0.0
    if r_max == r_min:
        # if all R equal, front has no curvature; still compute area from R
        pass

    f = (f_front - f_min) / (f_max - f_min + 1e-8)
    g = 1.0 - (r_front - r_min) / (r_max - r_min + 1e-8)  # G = 1 - R_norm

    if ref_point is None:
        ref_f = float(np.min(f))
        ref_g = float(np.min(g))
    else:
        ref_f, ref_g = ref_point

    # Sort by F ascending; accumulate area using the envelope of G (monotone increasing)
    order = np.argsort(f)
    f = f[order]
    g = g[order]

    area = 0.0
    prev_f = ref_f
    env_g = ref_g
    for i in range(len(f)):
        env_g = max(env_g, float(g[i]))
        width = float(f[i] - prev_f)
        height = float(env_g - ref_g)
        if width > 0 and height > 0:
            area += width * height
        prev_f = float(f[i])

    return float(area)


def find_knee_point(f_front: np.ndarray, r_front: np.ndarray) -> int:
    """Find knee point index on the front using max distance to chord method.

    Args:
        f_front: F values on the front.
        r_front: R values on the front.

    Returns:
        Index (in the front arrays) of the knee point.
    """
    n = len(f_front)
    if n < 3:
        return 0

    f_min, f_max = float(np.min(f_front)), float(np.max(f_front))
    r_min, r_max = float(np.min(r_front)), float(np.max(r_front))
    if f_max == f_min or r_max == r_min:
        return 0

    order = np.argsort(f_front)
    f = (f_front[order] - f_min) / (f_max - f_min + 1e-8)
    r = (r_front[order] - r_min) / (r_max - r_min + 1e-8)

    p0 = np.array([f[0], r[0]], dtype=np.float64)
    p1 = np.array([f[-1], r[-1]], dtype=np.float64)
    chord = p1 - p0
    chord_norm = np.linalg.norm(chord) + 1e-12

    dists = []
    for i in range(n):
        p = np.array([f[i], r[i]], dtype=np.float64)
        v = p - p0
        # area of parallelogram via 2D cross; height = area / base length
        dist = np.abs(np.cross(chord, v)) / chord_norm
        dists.append(dist)

    knee_sorted_idx = int(np.argmax(dists))
    # Map back to original front index
    front_indices = np.arange(n)[order]
    return int(front_indices[knee_sorted_idx])

