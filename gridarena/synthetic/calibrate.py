"""Calibration of the synthetic feeder generator on the stored MV grids.

Each stored grid (synthetic ones excluded) is measured -- size, section
length, depth, farthest distance, branching, connection-point share,
overhead share, transformer ratings -- and assigned an archetype by its
overhead share of cable length: `urban` below URBAN_MAX_OH, `rural` above
RURAL_MIN_OH, `mixed` between. Overhead means a cable with less than
UNDERGROUND_CAP_UF_PER_KM of charging capacitance (as in
pf_datagen/parameters.py; X/R when there is none).

Per archetype the generator then draws:
- node count from a log-normal fit (so sizes beyond the stored range, up to
  the generator's limit, remain possible);
- section length in two levels: a grid's median from a log-normal fit of the
  stored grids' medians, and each section around it with the stored grids'
  spread (pooling every section instead would let the large grids, with
  their shorter sections, set every grid's length);
- the overhead share from the archetype's grids (empirical);
- transformer ratings from the archetype's rating mix (empirical);
- the connection-point share as the archetype's median;
- the lateral and return probabilities of the growth (generator.py), found by
  simulation so the share of branching nodes and the depth relative to the
  node count match the archetype's medians.

`grid_metrics` is shared with the generator report, so generated and stored
grids are measured the same way.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from gridarena.pf_datagen.parameters import cable_family

URBAN_MAX_OH = 0.10
RURAL_MIN_OH = 0.50
ARCHETYPES = ("urban", "mixed", "rural")
METRICS = ("n_nodes", "section_m", "depth", "far_km", "branching", "cp_share", "overhead_share")
SYNTHETIC_PREFIX = "SYN-"


def archetype_of(overhead_share: float) -> str:
    if overhead_share < URBAN_MAX_OH:
        return "urban"
    return "rural" if overhead_share > RURAL_MIN_OH else "mixed"


def grid_metrics(node_types: Dict[str, str], edges: Sequence[tuple], n_cps: int) -> Optional[dict]:
    """edges: (from, to, length_m, overhead: bool) of the in-service, closed
    connections. None when the grid has no substation in its graph."""
    g = nx.Graph()
    g.add_nodes_from(node_types)
    for a, b, length, _ in edges:
        g.add_edge(a, b, w=length)
    subs = [n for n, t in node_types.items() if t == "substation" and n in g]
    if not subs or g.number_of_edges() == 0:
        return None
    hops = nx.multi_source_dijkstra_path_length(g, set(subs), weight=None)
    dist = nx.multi_source_dijkstra_path_length(g, set(subs), weight="w")
    total = sum(e[2] for e in edges)
    n = g.number_of_nodes()
    return {
        "n_nodes": n,
        "section_m": float(np.median([e[2] for e in edges])),
        "depth": max(hops.values()),
        "far_km": max(dist.values()) / 1000.0,
        "branching": sum(1 for _, d in g.degree() if d >= 3) / n,
        "cp_share": n_cps / n,
        "overhead_share": sum(e[2] for e in edges if e[3]) / total if total > 0 else 0.0,
    }


@dataclass
class ArchetypeFit:
    archetype: str
    n_grids: int
    log_nodes: tuple                    # (mean, std) of ln(node count)
    log_section: tuple                  # (mean, std) of ln(a grid's median section length, m)
    section_spread: float               # std of ln(section length / its grid's median)
    overhead_shares: List[float]
    cp_share: float
    branching: float
    depth_ratio: float                  # median depth (sections) / node count
    ratings: Dict[float, float]         # kVA -> share
    p_lateral: float = 0.0
    p_return: float = 0.0
    real: Dict[str, List[float]] = field(default_factory=dict)   # per metric, the grids' values

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ratings"] = {str(k): v for k, v in self.ratings.items()}
        d["real"] = {k: _pct(v) for k, v in self.real.items()}
        d["overhead_shares"] = _pct(self.overhead_shares)
        return d


def _pct(values) -> Optional[dict]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {f"p{p}": float(np.percentile(v, p)) for p in (10, 50, 90)}


def load_grids(cursor) -> List[dict]:
    """Every stored, non-synthetic grid, measured (grid_metrics), with its
    section lengths and transformer ratings."""
    cursor.execute('SELECT mv_grid_id FROM "MVGrid" WHERE mv_grid_id NOT LIKE %s', (SYNTHETIC_PREFIX + "%",))
    ids = [r[0] for r in cursor.fetchall()]
    if not ids:
        return []
    cursor.execute('SELECT mv_grid_id, "NodeId", "NodeType" FROM "MVNode" WHERE mv_grid_id = ANY(%s)', (ids,))
    types: Dict[str, Dict[str, str]] = {}
    for g, n, t in cursor.fetchall():
        types.setdefault(g, {})[n] = t
    cursor.execute(
        """
        SELECT c.mv_grid_id, c."FromNodeId", c."ToNodeId", c."Length", k."ImpReal", k."ImpImag", k."CapUfPerKm",
               c."Status", c."NormallyOpen"
        FROM "MVConnection" c JOIN "MVCable" k ON k."CableId" = c."CableId" AND k.mv_grid_id = c.mv_grid_id
        WHERE c.mv_grid_id = ANY(%s)
        """,
        (ids,),
    )
    edges: Dict[str, list] = {}
    for g, a, b, length, r, x, cap, status, n_open in cursor.fetchall():
        if (status or "active").lower() != "active" or n_open:
            continue
        overhead = cable_family(r or 0.0, x or 0.0, cap) == "overhead"
        edges.setdefault(g, []).append((a, b, float(length or 0.0), overhead))
    cursor.execute(
        """
        SELECT t.mv_grid_id, t."RatedPowerKVA" FROM "MVTransformer" t WHERE t.mv_grid_id = ANY(%s)
        """,
        (ids,),
    )
    ratings: Dict[str, list] = {}
    for g, kva in cursor.fetchall():
        ratings.setdefault(g, []).append(float(kva))
    cursor.execute('SELECT mv_grid_id, count(*) FROM "MVConnectionPoint" WHERE mv_grid_id = ANY(%s) GROUP BY 1',
                   (ids,))
    cps = dict(cursor.fetchall())
    out = []
    for g in ids:
        m = grid_metrics(types.get(g, {}), edges.get(g, []), cps.get(g, 0))
        if m is None or m["n_nodes"] < 5:
            continue
        out.append({"mv_grid_id": g, "metrics": m, "sections": [e[2] for e in edges[g] if e[2] > 0],
                    "ratings": ratings.get(g, [])})
    return out


def fit(grids: List[dict], seed: int = 0) -> Dict[str, ArchetypeFit]:
    """One fit per archetype that has grids."""
    from gridarena.synthetic.generator import calibrate_growth

    by: Dict[str, List[dict]] = {a: [] for a in ARCHETYPES}
    for g in grids:
        by[archetype_of(g["metrics"]["overhead_share"])].append(g)
    fits = {}
    for a, gs in by.items():
        if not gs:
            continue
        ln_n = np.log([g["metrics"]["n_nodes"] for g in gs])
        ln_med = np.log([g["metrics"]["section_m"] for g in gs])
        spread = np.concatenate([np.log(np.asarray(g["sections"]) / g["metrics"]["section_m"]) for g in gs])
        ratings = [r for g in gs for r in g["ratings"]]
        values, counts = np.unique(ratings, return_counts=True) if ratings else (np.array([400.0]), np.array([1]))
        f = ArchetypeFit(
            archetype=a, n_grids=len(gs),
            log_nodes=(float(ln_n.mean()), float(max(ln_n.std(), 0.05))),
            log_section=(float(ln_med.mean()), float(max(ln_med.std(), 0.05))),
            section_spread=float(max(spread.std(), 0.05)),
            overhead_shares=[g["metrics"]["overhead_share"] for g in gs],
            cp_share=float(np.median([g["metrics"]["cp_share"] for g in gs])),
            branching=float(np.median([g["metrics"]["branching"] for g in gs])),
            depth_ratio=float(np.median([g["metrics"]["depth"] / g["metrics"]["n_nodes"] for g in gs])),
            ratings={float(v): float(c / counts.sum()) for v, c in zip(values, counts)},
            real={k: [g["metrics"][k] for g in gs] for k in METRICS},
        )
        f.p_lateral, f.p_return = calibrate_growth(f, seed)
        fits[a] = f
    return fits


def calibrate(cursor, seed: int = 0) -> Dict[str, ArchetypeFit]:
    return fit(load_grids(cursor), seed)
