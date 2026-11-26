"""Random radial configurations of an MV grid (Phase 2 axis 4).

A configuration decides which switchable connections are closed. Connections
that can't be switched (and in-service transformers) are always closed, so
they are contracted first: every group of nodes they join becomes one vertex,
and all substations merge into one root. The switchable connections are the
edges of the resulting multigraph, and a radial configuration -- every node
fed, no loop -- is exactly a spanning tree of it. Trees are drawn uniformly
with Wilson's algorithm (loop-erased random walks), and their number is the
matrix-tree theorem's determinant.

Which connections are switchable (`switchable`):
- `marked`: switches and normally-open ties, as the grid stores them;
- `all_lines`: every MV line, as if each section had a disconnector.
Either way only connections whose status is active can be closed.

An edge that is a bridge of the multigraph is in every tree, so bridges are
contracted too: what remains are the loops the ties form, which keeps the
walks short on long radial feeders. An edge joining a vertex to itself (a
switch on a loop of unswitchable cables) is never closed in a radial
configuration; `allow_meshed` may close it, like any other open switch.

Only the part of the multigraph reachable from the root takes part; switchable
connections outside it (beyond an inactive cable, say) keep their base state.

A step voltage regulator regulates in one direction only, so a radial
configuration that would feed a regulator from its far side (one fed from
its from end in the base topology) is rejected and redrawn. Rejection keeps
the draw uniform over the configurations that remain; count() still counts
all spanning trees, so it is an upper bound when the grid has regulators.
"""

from dataclasses import dataclass
from math import exp, log
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np

MAX_DRAW_ATTEMPTS_PER_CONFIG = 200
EXACT_COUNT_LIMIT = 2 ** 50     # below it the determinant is rounded to the exact integer


@dataclass(frozen=True)
class Configuration:
    """A switching state relative to the base topology: `opened` holds
    connections closed in the base that this configuration opens, `closed`
    connections open in the base that it closes. Configuration 0 is the base."""
    config_idx: int
    opened: FrozenSet[str] = frozenset()
    closed: FrozenSet[str] = frozenset()
    meshed: bool = False


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def switchable_ids(branches, mode: str) -> Set[str]:
    """Connections a configuration may open or close."""
    out = set()
    for b in branches:
        if b.kind == "transformer" or not b.available:
            continue
        if mode == "all_lines" or b.is_switch or b.normally_open:
            out.add(b.connection_id)
    return out


class Reconfigurer:
    def __init__(self, branches, node_ids: Sequence[str], substation_node_ids: Sequence[str],
                 switchable: str = "marked"):
        self.switchable = switchable_ids(branches, switchable)
        self.base_closed = {b.connection_id for b in branches if b.in_service}
        self._ends = {b.connection_id: (b.from_node_id, b.to_node_id) for b in branches}
        self._subs = list(substation_node_ids)
        self._regulators = [b.connection_id for b in branches if b.kind == "regulated_line"]
        # Only regulators the base topology feeds from their from end are protected.
        self._regulators = [c for c in self._regulators if self._forward(self.base_closed, c) is not False]
        uf = _UnionFind()
        for nid in node_ids:
            uf.find(nid)
        subs = list(substation_node_ids)
        for s in subs[1:]:
            uf.union(subs[0], s)
        for b in branches:
            if b.in_service and b.connection_id not in self.switchable:
                uf.union(b.from_node_id, b.to_node_id)
        edges = [(b.connection_id, uf.find(b.from_node_id), uf.find(b.to_node_id))
                 for b in branches if b.connection_id in self.switchable]
        root = uf.find(subs[0])

        # The part reachable from the root through switchable connections.
        g = nx.MultiGraph()
        g.add_node(root)
        g.add_edges_from((u, v, cid) for cid, u, v in edges if u != v)
        reach = nx.node_connected_component(g, root)
        self.loop_only = [cid for cid, u, v in edges if u == v and u in reach]   # never closed when radial
        tree_edges = [(cid, u, v) for cid, u, v in edges if u != v and u in reach]

        # Bridges are in every spanning tree: contract them too.
        simple = nx.Graph()
        simple.add_nodes_from(reach)
        multiplicity: Dict[FrozenSet[str], int] = {}
        for _, u, v in tree_edges:
            simple.add_edge(u, v)
            k = frozenset((u, v))
            multiplicity[k] = multiplicity.get(k, 0) + 1
        bridges = {frozenset(e) for e in nx.bridges(simple) if multiplicity[frozenset(e)] == 1}
        self.forced = [cid for cid, u, v in tree_edges if frozenset((u, v)) in bridges]
        uf2 = _UnionFind()
        for _, u, v in tree_edges:
            if frozenset((u, v)) in bridges:
                uf2.union(u, v)
        self.free: List[Tuple[str, str, str]] = [
            (cid, uf2.find(u), uf2.find(v)) for cid, u, v in tree_edges if frozenset((u, v)) not in bridges
        ]
        self.root = uf2.find(root)
        self.vertices = sorted({self.root} | {x for _, u, v in self.free for x in (u, v)})
        self.adj: Dict[str, List[Tuple[str, str]]] = {x: [] for x in self.vertices}
        for cid, u, v in self.free:
            self.adj[u].append((cid, v))
            self.adj[v].append((cid, u))
        self.considered = {cid for cid, *_ in tree_edges} | set(self.loop_only)

    def _forward(self, in_service: Set[str], cid: str) -> Optional[bool]:
        """Whether in-service regulator `cid` is fed from its from end (None
        when it is out of service or not energised)."""
        if cid not in in_service:
            return None
        g = nx.Graph()
        g.add_edges_from(self._ends[c] for c in in_service)
        g.add_nodes_from(self._subs)
        dist = nx.multi_source_dijkstra_path_length(g, set(self._subs), weight=None) if self._subs else {}
        a, b = self._ends[cid]
        if a not in dist or b not in dist:
            return None
        return dist[a] < dist[b]

    def regulators_forward(self, closed_considered: Set[str]) -> bool:
        if not self._regulators:
            return True
        in_service = (self.base_closed - self.considered) | closed_considered | set(self.forced)
        return all(self._forward(in_service, c) is not False for c in self._regulators)

    # -- counting ------------------------------------------------------------

    def count(self) -> Union[int, float]:
        """Number of radial configurations (spanning trees of the contracted
        multigraph): an exact int, or a float when it exceeds 2^50."""
        n = len(self.vertices)
        if n <= 1:
            return 1
        idx = {x: i for i, x in enumerate(self.vertices)}
        lap = np.zeros((n, n))
        for _, u, v in self.free:
            i, j = idx[u], idx[v]
            lap[i, i] += 1
            lap[j, j] += 1
            lap[i, j] -= 1
            lap[j, i] -= 1
        r = idx[self.root]
        minor = np.delete(np.delete(lap, r, axis=0), r, axis=1)
        sign, logdet = np.linalg.slogdet(minor)
        if sign <= 0:
            return 0
        if logdet < log(EXACT_COUNT_LIMIT):
            return int(round(exp(logdet)))
        return float(exp(logdet)) if logdet < 700 else float("inf")

    def base_is_radial_tree(self) -> bool:
        """Whether the base topology is itself one of the drawable configurations."""
        closed = {cid for cid, *_ in self.free if cid in self.base_closed}
        if any(cid in self.base_closed for cid in self.loop_only):
            return False
        if len(closed) != len(self.vertices) - 1:
            return False
        g = nx.MultiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from((u, v) for cid, u, v in self.free if cid in closed)
        return nx.is_connected(g)

    # -- drawing -------------------------------------------------------------

    def draw_tree(self, rng: np.random.Generator) -> Set[str]:
        """A uniformly random spanning tree (Wilson's algorithm): the free
        switchable connections it closes."""
        in_tree = {self.root}
        next_edge: Dict[str, Tuple[str, str]] = {}
        closed: Set[str] = set()
        buf, pos = rng.random(4096), 0
        for start in self.vertices:
            u = start
            while u not in in_tree:
                if pos == len(buf):
                    buf, pos = rng.random(4096), 0
                options = self.adj[u]
                next_edge[u] = options[min(int(buf[pos] * len(options)), len(options) - 1)]
                pos += 1
                u = next_edge[u][1]
            u = start
            while u not in in_tree:
                in_tree.add(u)
                cid, u = next_edge[u]
                closed.add(cid)
        return closed

    def _configuration(self, idx: int, closed_considered: Set[str], meshed: bool) -> Configuration:
        closed_all = closed_considered | set(self.forced)
        opened = frozenset(cid for cid in self.considered if cid in self.base_closed and cid not in closed_all)
        newly = frozenset(cid for cid in closed_all if cid not in self.base_closed)
        return Configuration(idx, opened, newly, meshed)

    def draw(self, n_configs: int, rng: np.random.Generator, allow_meshed: bool = False,
             extra_closed: int = 1) -> List[Configuration]:
        """n_configs distinct configurations, none equal to the base, numbered
        1..n_configs. Radial ones are distinct spanning trees; with
        allow_meshed each closes `extra_closed` more open switches on top.
        Raises when the grid doesn't have that many."""
        if n_configs <= 0:
            return []
        base_key = frozenset(cid for cid in self.considered if cid in self.base_closed)
        if not allow_meshed:
            available = self.count() - (1 if self.base_is_radial_tree() else 0)
            if n_configs > available:
                hint = ("Add normally-open ties (the MV grid page can suggest them) or use switchable 'all_lines'."
                        if available < 1 else "Lower n_configs, or use switchable 'all_lines'.")
                raise ValueError(
                    f"Reconfiguration asked for {n_configs} configurations, but the grid has only "
                    f"{_fmt(available)} radial configuration(s) besides the base one. {hint}"
                )
        seen = {base_key}
        out: List[Configuration] = []
        for _ in range(MAX_DRAW_ATTEMPTS_PER_CONFIG * n_configs + 1000):
            if len(out) == n_configs:
                break
            closed = self.draw_tree(rng)
            if allow_meshed:
                open_now = sorted(cid for cid, *_ in self.free if cid not in closed) + sorted(self.loop_only)
                if len(open_now) < extra_closed:
                    raise ValueError(
                        f"Meshed reconfiguration asked to close {extra_closed} extra switch(es), but a radial "
                        f"configuration leaves only {len(open_now)} open."
                    )
                pick = rng.choice(len(open_now), size=extra_closed, replace=False)
                closed = closed | {open_now[int(k)] for k in pick}
            key = frozenset(closed | set(self.forced))
            if key in seen:
                continue
            seen.add(key)
            if not allow_meshed and not self.regulators_forward(closed):
                continue
            out.append(self._configuration(len(out) + 1, closed, allow_meshed))
        if len(out) < n_configs:
            raise ValueError(
                f"Reconfiguration found only {len(out)} of {n_configs} distinct configurations"
                + (" that feed every regulator from its from end" if self._regulators and not allow_meshed else "")
                + ". Lower n_configs."
            )
        return out


def _fmt(n) -> str:
    return f"{n:.3g}" if isinstance(n, float) else str(n)


def configuration_in_service(base_in_service: Set[str], config: Configuration) -> Set[str]:
    return (set(base_in_service) - config.opened) | config.closed


def draw_configurations(cfg, branches, node_ids, substation_node_ids, rng) -> Tuple[List[Configuration], Optional[
        Union[int, float]]]:
    """Configuration 0 (the base) plus cfg.n_configs drawn ones when
    reconfiguration is on. Also returns the radial count (None when off)."""
    base = Configuration(0)
    if cfg is None or not cfg.enabled:
        return [base], None
    rec = Reconfigurer(branches, node_ids, substation_node_ids, cfg.switchable)
    drawn = rec.draw(cfg.n_configs, rng, cfg.allow_meshed, cfg.extra_closed)
    return [base] + drawn, rec.count()
