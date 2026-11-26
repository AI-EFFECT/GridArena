"""How the load is distributed across the grid within a sample (Phase 2, axis 2).

- Sizing (`rating`): a historical series gives only the *shape* of a
  connection point's load (its values divided by its mean |P| over the
  whole database -- a fixed sample of REFERENCE_TIMESTAMPS of its
  timestamps -- or, with scale_over `run`, over the run's own timestamps);
  the point's mean active power over the database is its transformer
  rating x a utilisation factor, and a run's level follows its time of day
  and season. With size_by `peak` the divisor is the series' 99th
  percentile of |P| instead (PEAK_PERCENTILE), so utilisation is the
  point's peak loading. Without it (`series`), values are used as
  recorded -- right for LV databases, whose per-point load is already the sum
  of as many series as the LV grid has nodes, but not for MV or Feeder data,
  where a 100 kVA point would draw as much as a 2 MVA one.
- Feeders per point (`by_kva`, Feeder data): a point sums several distinct
  feeder series, their count drawn from its transformer's kVA band
  (draw_feeder_counts); under `rating` the sum is sized as one series, by its
  own divisor (combination_scale), so utilisation keeps its meaning.
- Reactive power: missing Q (and every Q in `pf` mode) comes from a power
  factor drawn per point per reassignment block, inductive.
- Noise: each point's factor is spatial x local, with an optional extra
  factor on Q. All are log-normal with mean 1 (never negative); the spatial
  part is correlated along the cables, exp(-distance / correlation length),
  and independent between parts of the grid not connected to each other.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# How many of a database's timestamps the shape divisor (scale_over `database`) is averaged over.
REFERENCE_TIMESTAMPS = 2000
# The percentile of |P| a series' shape is divided by under size_by `peak`.
PEAK_PERCENTILE = 99.0

# Mean size of a point with no (active) transformer when no point has one:
# a mid-range secondary substation.
DEFAULT_POINT_KVA = 400.0


def resolve_point_sizing(setting: str, source_type: str) -> str:
    """'auto' means 'rating' for MV and Feeder databases, 'series' for LV."""
    if setting != "auto":
        return setting
    return "series" if source_type == "LV" else "rating"


def point_mean_load_w(points: Sequence[str], point_kva: Dict[str, float], utilisation: float) -> Dict[str, float]:
    """Mean (or, with size_by `peak`, peak) active power (W) per point: its
    active transformers' kVA x utilisation. Points without one get the
    average of the others."""
    return {p: kva * 1e3 * utilisation for p, kva in point_kva_or_fallback(points, point_kva).items()}


def reference_timestamps(timeline: Sequence, n: int = REFERENCE_TIMESTAMPS) -> List:
    """A fixed sample of n of the timeline's timestamps (all when fewer), the
    same for every run: drawn with a constant seed rather than evenly spaced,
    which could fall at one time of day when the spacing is a whole day."""
    if len(timeline) <= n:
        return list(timeline)
    idx = np.sort(np.random.default_rng(0).choice(len(timeline), size=n, replace=False))
    return [timeline[i] for i in idx]


def series_means(values: Dict[Tuple[str, object], Tuple[float, Optional[float]]], series_ids,
                 timestamps) -> Dict[str, float]:
    """Average magnitude, mean |P|, of each series over the given timestamps:
    the shape divisor. For a series that never goes negative this is its mean.
    For one that does -- e.g. a feeder with rooftop PV exporting at midday --
    dividing by the plain mean would blow the shape up as the mean nears zero
    (one real feeder averages 4 kW while swinging between -56 and +34 kW),
    so the magnitude is used. An all-zero series gets 0 (zero shape)."""
    return series_scales(values, series_ids, timestamps, "mean")


def series_scales(values: Dict[Tuple[str, object], Tuple[float, Optional[float]]], series_ids, timestamps,
                  size_by: str = "mean") -> Dict[str, float]:
    """The shape divisor of each series over the given timestamps: its mean
    |P| (`mean`, see series_means) or its PEAK_PERCENTILE of |P| (`peak`).
    An all-zero series gets 0 (zero shape)."""
    out = {}
    for s in series_ids:
        p = np.abs(np.array([values[(s, t)][0] for t in timestamps if (s, t) in values], dtype=float))
        if not p.size:
            out[s] = 0.0
        else:
            out[s] = float(np.percentile(p, PEAK_PERCENTILE)) if size_by == "peak" else float(p.mean())
    return out


def point_kva_or_fallback(points: Sequence[str], point_kva: Dict[str, float]) -> Dict[str, float]:
    """Each point's transformer kVA; points without one get the others' average (as point_mean_load_w)."""
    known = [point_kva[p] for p in points if point_kva.get(p, 0) > 0]
    fallback = float(np.mean(known)) if known else DEFAULT_POINT_KVA
    return {p: point_kva[p] if point_kva.get(p, 0) > 0 else fallback for p in points}


def draw_feeder_counts(points: Sequence[str], point_kva: Dict[str, float], bands, n_series: int,
                       rng: np.random.Generator) -> Dict[str, int]:
    """How many Feeder series each point sums (feeders_per_point `by_kva`):
    uniform in [min, max] of the first band whose max_kva covers the point's
    kVA (the last band above them all), capped at the database's series."""
    kva = point_kva_or_fallback(points, point_kva)
    out = {}
    for p in points:
        band = next((b for b in bands if kva[p] <= b.max_kva), bands[-1])
        out[p] = int(min(rng.integers(band.min, band.max + 1), n_series))
    return out


def series_matrix(values: Dict[Tuple[str, object], Tuple[float, Optional[float]]], series_ids: Sequence[str],
                  timestamps: Sequence) -> Tuple[Dict[str, int], np.ndarray]:
    """P of each series at each timestamp, NaN where missing: (row of each series, matrix)."""
    row = {s: i for i, s in enumerate(series_ids)}
    M = np.full((len(series_ids), len(timestamps)), np.nan)
    for j, t in enumerate(timestamps):
        for s in series_ids:
            v = values.get((s, t))
            if v is not None:
                M[row[s], j] = v[0]
    return row, M


def combination_scale(row: Dict[str, int], M: np.ndarray, series: Sequence[str], size_by: str = "mean") -> float:
    """The shape divisor (series_scales) of the sum of `series`: mean |P| or
    the PEAK_PERCENTILE of |P| of the summed series, over the timestamps where
    any of them has a value (a missing one counts 0, as in the scenarios)."""
    sub = M[[row[s] for s in series]]
    present = ~np.isnan(sub).all(axis=0)
    if not present.any():
        return 0.0
    p = np.abs(np.nansum(sub[:, present], axis=0))
    return float(np.percentile(p, PEAK_PERCENTILE)) if size_by == "peak" else float(p.mean())


def draw_power_factors(n_blocks: int, points: Sequence[str], pf_min: float, pf_max: float,
                       rng: np.random.Generator) -> List[Dict[str, float]]:
    """One inductive power factor per point per reassignment block."""
    return [dict(zip(points, rng.uniform(pf_min, pf_max, size=len(points)))) for _ in range(n_blocks)]


def tan_phi(pf: float) -> float:
    return float(np.tan(np.arccos(np.clip(pf, 1e-6, 1.0))))


def cable_distances_km(branches, node_ids: Sequence[str], points: Sequence[str]) -> np.ndarray:
    """Shortest path length (km) along in-service branches of the base
    topology between every pair of points; transformers count as zero length.
    Points in parts of the grid not connected to each other are at inf."""
    index = {n: i for i, n in enumerate(node_ids)}
    rows, cols, data = [], [], []
    for b in branches:
        if not b.in_service or b.from_node_id not in index or b.to_node_id not in index:
            continue
        length = 0.0 if b.kind == "transformer" else max(float(b.length_m or 0.0), 0.0) / 1000.0
        # csgraph treats an explicit 0 as "no edge"; a tiny length keeps zero-length branches.
        w = max(length, 1e-9)
        i, j = index[b.from_node_id], index[b.to_node_id]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]
    graph = coo_matrix((data, (rows, cols)), shape=(len(node_ids), len(node_ids))).tocsr()
    idx = [index[p] for p in points]
    d = dijkstra(graph, directed=False, indices=idx)
    return d[:, idx]


class LoadNoise:
    """Multiplicative noise per point and sample: (spatial x local) on P and
    Q, times an extra factor on Q. Factors are exp(N(-s^2/2, s^2)), so their
    mean is 1."""

    def __init__(self, n_points: int, local_sigma: float = 0.0, spatial_sigma: float = 0.0,
                 correlation_km: float = 2.0, distance_km: Optional[np.ndarray] = None, q_sigma: float = 0.0):
        self.n = n_points
        self.local_sigma = float(local_sigma)
        self.spatial_sigma = float(spatial_sigma)
        self.q_sigma = float(q_sigma)
        self._chol = None
        if self.spatial_sigma > 0 and n_points > 0:
            if distance_km is None:
                raise ValueError("Spatially correlated load noise needs the distances between points.")
            corr = np.exp(-np.asarray(distance_km, dtype=float) / float(correlation_km))  # inf -> 0
            np.fill_diagonal(corr, 1.0)
            cov = self.spatial_sigma ** 2 * corr
            self._chol = _cholesky_psd(cov)

    @property
    def active(self) -> bool:
        return self.local_sigma > 0 or self.spatial_sigma > 0 or self.q_sigma > 0

    def draw(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """(factor on P and Q, extra factor on Q), each of length n_points."""
        log_f = np.zeros(self.n)
        if self._chol is not None:
            log_f += self._chol @ rng.standard_normal(self.n) - 0.5 * self.spatial_sigma ** 2
        if self.local_sigma > 0:
            log_f += rng.normal(-0.5 * self.local_sigma ** 2, self.local_sigma, size=self.n)
        log_q = (rng.normal(-0.5 * self.q_sigma ** 2, self.q_sigma, size=self.n)
                 if self.q_sigma > 0 else np.zeros(self.n))
        return np.exp(log_f), np.exp(log_q)


def _cholesky_psd(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor, with a little jitter; falls back to a symmetric square
    root if rounding makes the matrix slightly indefinite."""
    try:
        return np.linalg.cholesky(cov + 1e-10 * np.eye(len(cov)))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        return v @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
