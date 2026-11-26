"""Tie-line suggestions for an MV grid (Phase 2 axis 4; a tool on the MV grid
page, not part of a run).

Most stored feeders have no switches, so reconfiguration has nothing to work
with. A normally-open tie is worth having where two nodes are close on the
ground but far apart along the cables (closing it would give one a much
shorter back-feed), or close and on different feeders. Candidates:

- MV nodes with coordinates, substations excluded, not already connected;
- less than `max_distance_m` apart (straight line), and either more than
  `min_cable_distance_m` apart along the cables or on different feeders.

Cable distance runs over every active connection, open ties included, so a
pair an existing tie already shortcuts isn't proposed. A feeder is the part of
the base topology below one cable leaving a substation.

Candidates are ranked by cable distance / straight-line distance (different
feeders first when there is no path), and picked greedily: a candidate is
skipped if either end lies within `max_distance_m` of an end of a tie already
picked, so the ties spread over the grid. Each tie gets the grid's most common
cable type and a length of LENGTH_FACTOR x the straight-line distance.
"""

from collections import Counter
from math import asin, ceil, cos, radians, sin, sqrt
from typing import Dict, List, Optional, Tuple

import networkx as nx

LENGTH_FACTOR = 1.2
EARTH_RADIUS_M = 6371000.0


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(radians, (*a, *b))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(h))


def _feeders(grid: dict) -> Dict[str, str]:
    """Node -> the first cable on its path from a substation, in the base topology."""
    subs = {n["node_id"] for n in grid["nodes"] if n.get("node_type") == "substation"}
    g = nx.Graph()
    for c in grid["connections"]:
        if (c.get("status") or "active").lower() == "active" and not c.get("normally_open"):
            g.add_edge(c["from_node_id"], c["to_node_id"], cid=c["connection_id"])
    out: Dict[str, str] = {}
    for s in subs:
        if s not in g:
            continue
        for nb in g.neighbors(s):
            if nb in subs:
                continue
            first = g.edges[s, nb]["cid"]
            for n in nx.node_connected_component(g.subgraph(set(g) - subs), nb):
                out.setdefault(n, first)
    return out


def suggest_ties(grid: dict, max_distance_m: float = 500.0, min_cable_distance_m: float = 2000.0,
                 max_ties: int = 5) -> dict:
    """Tie-line proposals for one grid (as get_mv_grid_detail returns it)."""
    subs = {n["node_id"] for n in grid["nodes"] if n.get("node_type") == "substation"}
    coords = {n["node_id"]: (n["coord_lat"], n["coord_lon"]) for n in grid["nodes"]
              if n.get("coord_lat") is not None and n.get("coord_lon") is not None and n["node_id"] not in subs}
    g = nx.Graph()
    g.add_nodes_from(n["node_id"] for n in grid["nodes"])
    connected = set()
    for c in grid["connections"]:
        a, b = c["from_node_id"], c["to_node_id"]
        connected.add(frozenset((a, b)))
        if (c.get("status") or "active").lower() == "active":
            w = g.edges[a, b]["w"] if g.has_edge(a, b) else float("inf")
            g.add_edge(a, b, w=min(w, float(c.get("length") or 0.0)))
    feeder = _feeders(grid)

    ids = sorted(coords)
    # Neighbours within max_distance_m (a coarse grid of cells keeps this near-linear).
    cell_deg = max_distance_m / 111000.0
    cells: Dict[Tuple[int, int], List[str]] = {}
    for n in ids:
        lat, lon = coords[n]
        cells.setdefault((int(lat // cell_deg), int(lon // cell_deg)), []).append(n)
    candidates = []
    dist_cache: Dict[str, Dict[str, float]] = {}
    for a in ids:
        lat, lon = coords[a]
        ci, cj = int(lat // cell_deg), int(lon // cell_deg)
        span = int(ceil(1 / max(cos(radians(lat)), 0.05)))      # a degree of longitude shrinks with latitude
        near = [b for di in (-1, 0, 1) for dj in range(-span, span + 1) for b in cells.get((ci + di, cj + dj), ())
                if b > a]
        for b in near:
            if frozenset((a, b)) in connected:
                continue
            d = haversine_m(coords[a], coords[b])
            if d >= max_distance_m or d <= 0:
                continue
            if a not in dist_cache:
                dist_cache[a] = nx.single_source_dijkstra_path_length(g, a, weight="w")
            cable = dist_cache[a].get(b)
            other_feeder = a in feeder and b in feeder and feeder[a] != feeder[b]
            if cable is not None and cable <= min_cable_distance_m and not other_feeder:
                continue
            candidates.append({
                "from_node_id": a, "to_node_id": b, "distance_m": d,
                "cable_distance_m": cable,
                "reason": "different_feeders" if other_feeder else "distant_along_cables",
                "_score": float("inf") if cable is None else cable / d,
            })
    candidates.sort(key=lambda c: (-c["_score"], c["distance_m"], c["from_node_id"], c["to_node_id"]))

    cable_counts = Counter(c["cable_id"] for c in grid["connections"] if c.get("cable_id"))
    cable_id: Optional[str] = cable_counts.most_common(1)[0][0] if cable_counts else None
    taken = {c["connection_id"] for c in grid["connections"]}
    picked: List[dict] = []
    ends: List[Tuple[float, float]] = []
    k = 1
    for c in candidates:
        if len(picked) >= max_ties:
            break
        pa, pb = coords[c["from_node_id"]], coords[c["to_node_id"]]
        if any(haversine_m(p, e) < max_distance_m for p in (pa, pb) for e in ends):
            continue
        while f"TIE{k:02d}" in taken:
            k += 1
        cid = f"TIE{k:02d}"
        taken.add(cid)
        c = {key: v for key, v in c.items() if key != "_score"}
        c.update(connection_id=cid, cable_id=cable_id, length=round(LENGTH_FACTOR * c["distance_m"], 1))
        picked.append(c)
        ends += [pa, pb]
    return {"suggestions": picked, "n_candidates": len(candidates), "cable_id": cable_id}


def validate_ties(grid: dict, ties) -> List[str]:
    """Errors that keep `ties` (schemas.MVTieLine) from being stored: unknown
    nodes or cable, an id already used, or a pair already connected."""
    errors = []
    nodes = {n["node_id"] for n in grid["nodes"]}
    cables = {c["cable_id"] for c in grid["cables"]}
    taken = {c["connection_id"] for c in grid["connections"]}
    pairs = {frozenset((c["from_node_id"], c["to_node_id"])) for c in grid["connections"]}
    for t in ties:
        for end in (t.from_node_id, t.to_node_id):
            if end not in nodes:
                errors.append(f"Tie '{t.connection_id}': node '{end}' not found.")
        if t.from_node_id == t.to_node_id:
            errors.append(f"Tie '{t.connection_id}': both ends are '{t.from_node_id}'.")
        if t.cable_id not in cables:
            errors.append(f"Tie '{t.connection_id}': cable '{t.cable_id}' not found.")
        if t.connection_id in taken:
            errors.append(f"Tie '{t.connection_id}': connection id already used.")
        pair = frozenset((t.from_node_id, t.to_node_id))
        if pair in pairs:
            errors.append(f"Tie '{t.connection_id}': '{t.from_node_id}' and '{t.to_node_id}' are already connected.")
        taken.add(t.connection_id)
        pairs.add(pair)
    return errors
