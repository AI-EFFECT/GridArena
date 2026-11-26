"""Synthetic MV feeder generator (Phase 2, axis 7B): gridarena/synthetic/."""

from math import log

import networkx as nx
import numpy as np
import pytest

from gridarena.routers import mv_grid as mv_router
from gridarena.synthetic import planning
from gridarena.synthetic.batch import generate
from gridarena.synthetic.calibrate import ArchetypeFit, archetype_of, fit, grid_metrics
from gridarena.synthetic.generator import Options, _grow_feeder, branching_share, equip, grow
from gridarena.synthetic.planning import CableType, check

CABLES = {
    "UG": [CableType("SYN-UG-S", 0.206, 0.096, 325.0, 0.109), CableType("SYN-UG-L", 0.125, 0.090, 428.0, 0.118)],
    "OH": [CableType("SYN-OH-S", 0.699, 0.360, 193.0, 0.0101), CableType("SYN-OH-L", 0.243, 0.329, 357.0, 0.0111)],
}


def _fit(archetype="rural", **kw):
    base = dict(archetype=archetype, n_grids=20, log_nodes=(log(100), 0.3), log_section=(log(260), 0.2),
                section_spread=0.5, overhead_shares=[0.6, 0.7, 0.8], cp_share=0.44, branching=0.38,
                depth_ratio=0.26, ratings={100.0: 0.5, 160.0: 0.3, 250.0: 0.2}, p_lateral=0.39, p_return=0.2,
                real={k: [1.0, 2.0, 3.0] for k in ("n_nodes", "section_m", "depth", "far_km", "branching",
                                                     "cp_share", "overhead_share")})
    base.update(kw)
    return ArchetypeFit(**base)


def _graph(grid, closed_only=True):
    g = nx.Graph()
    g.add_nodes_from(n.node_id for n in grid.nodes)
    for c in grid.connections:
        if closed_only and c.normally_open:
            continue
        g.add_edge(c.from_node_id, c.to_node_id)
    return g


def test_archetypes_and_metrics():
    assert archetype_of(0.05) == "urban" and archetype_of(0.3) == "mixed" and archetype_of(0.7) == "rural"
    types = {"S": "substation", "A": "junction", "B": "connection_point", "C": "connection_point"}
    m = grid_metrics(types, [("S", "A", 100.0, False), ("A", "B", 300.0, True), ("A", "C", 200.0, False)], 2)
    assert m["n_nodes"] == 4 and m["depth"] == 2 and m["far_km"] == pytest.approx(0.4)
    assert m["branching"] == pytest.approx(0.25) and m["cp_share"] == 0.5
    assert m["overhead_share"] == pytest.approx(0.5) and m["section_m"] == 200.0
    assert grid_metrics({"A": "junction"}, [], 0) is None


def test_growth_starts_laterals_only_where_they_add_a_branch_point():
    parents, branches = _grow_feeder(200, 0.5, 0.3, np.random.default_rng(1))
    children = np.bincount([p for p in parents if p >= 0], minlength=200)
    assert all(p < k for k, p in enumerate(parents))                     # a tree grown in order
    assert parents[0] == -1 and parents.count(-1) == 1                    # one cable from the substation
    assert children.max() <= 2                                            # laterals never pile onto a junction
    assert 0.3 < branching_share(parents, 200) < 0.5


def test_generated_grids_are_valid_radial_and_connected():
    grids, report = generate({"rural": _fit()}, "rural", 6, seed=3, opts=Options(), cables=CABLES,
                             numbers={"urban": 1, "mixed": 1, "rural": 7})
    assert [g.mv_grid_id for g in grids] == [f"SYN-rural-{k:04d}" for k in range(7, 13)]
    for grid in grids:
        assert mv_router._validate_mv_grid(grid) == []
        g = _graph(grid)
        assert nx.is_tree(g)
        subs = [n.node_id for n in grid.nodes if n.node_type == "substation"]
        assert subs == ["SUB1"] and g.degree("SUB1") == 1
        cps = {cp.node_id for cp in grid.connection_points}
        leaves = {n for n in g if g.degree(n) == 1 and n != "SUB1"}
        assert leaves <= cps and len(grid.transformers) == len(cps)
        assert {t.rated_power_kva for t in grid.transformers} <= {100.0, 160.0, 250.0}
        assert all(-1 < n.coord_lat - 48.85 < 1 and n.coord_lon is not None for n in grid.nodes)
        assert '"seed": 3' in grid.description and grid.region == "synthetic"
    assert all(r["max_loading"] <= 1.0 and r["min_voltage_pu"] >= 0.95 for r in report["grids"])


def test_options_add_feeders_a_second_substation_and_open_ties():
    opts = Options(feeders_per_substation=3, second_substation=True, ties_between_feeders=2, n_nodes=(150, 200))
    grids, report = generate({"mixed": _fit("mixed", overhead_shares=[0.4])}, "mixed", 3, seed=5, opts=opts,
                             cables=CABLES, numbers={"urban": 1, "mixed": 1, "rural": 1})
    for grid in grids:
        g = _graph(grid)
        assert nx.is_forest(g) and nx.number_connected_components(g) == 2           # one tree per substation
        for sub in ("SUB1", "SUB2"):
            assert g.degree(sub) == 3
        ties = [c for c in grid.connections if c.normally_open]
        assert len(ties) == 3 and all(c.is_switch for c in ties)
        closed_switches = {c.to_node_id for c in grid.connections if c.is_switch and not c.normally_open}
        assert {e for c in ties for e in (c.from_node_id, c.to_node_id)} <= closed_switches
        assert sum(1 for c in grid.connections if c.is_switch and c.from_node_id.startswith("SUB")) == 6
        comp = {n: k for k, c in enumerate(nx.connected_components(g)) for n in c}
        # The first tie joins the two substations' feeders.
        assert comp[ties[0].from_node_id] != comp[ties[0].to_node_id]
        # Ties never use a node twice, and always join different feeders.
        ends = [e for c in ties for e in (c.from_node_id, c.to_node_id)]
        assert len(set(ends)) == len(ends)
        assert 150 <= len(grid.nodes) <= 200
        assert nx.is_connected(_graph(grid, closed_only=False))


def test_same_seed_same_grids():
    a, _ = generate({"rural": _fit()}, "rural", 2, 9, Options(), CABLES, {"rural": 1})
    b, _ = generate({"rural": _fit()}, "rural", 2, 9, Options(), CABLES, {"rural": 1})
    c, _ = generate({"rural": _fit()}, "rural", 2, 10, Options(), CABLES, {"rural": 1})
    assert [g.model_dump() for g in a] == [g.model_dump() for g in b]
    assert [g.model_dump() for g in a] != [g.model_dump() for g in c]


def test_statistics_stay_within_the_fitted_ranges():
    f = _fit()                          # nodes ~ ln N(ln 100, 0.3), sections ~ 260 m, the rest as fitted
    _, report = generate({"rural": f}, "rural", 40, 1, Options(), CABLES, {"rural": 1})
    gen = report["archetypes"]["rural"]["metrics"]
    assert 70 < gen["n_nodes"]["generated"]["p50"] < 140
    assert 180 < gen["section_m"]["generated"]["p50"] < 380
    assert 0.3 < gen["branching"]["generated"]["p50"] < 0.46
    assert 0.38 < gen["cp_share"]["generated"]["p50"] < 0.55
    assert 0.55 < gen["overhead_share"]["generated"]["p50"] < 0.85


def test_fit_measures_each_archetype():
    def grid(n, oh, sections):
        return {"metrics": {"n_nodes": n, "section_m": float(np.median(sections)), "depth": n // 3,
                            "far_km": 5.0, "branching": 0.3, "cp_share": 0.5, "overhead_share": oh},
                "sections": sections, "ratings": [160.0, 250.0]}
    fits = fit([grid(40, 0.0, [200.0, 300.0]), grid(60, 0.05, [250.0, 260.0]), grid(120, 0.8, [300.0, 320.0])])
    assert set(fits) == {"urban", "rural"} and fits["urban"].n_grids == 2
    assert fits["urban"].ratings == {160.0: 0.5, 250.0: 0.5}
    assert 0 <= fits["rural"].p_lateral <= 0.9 and 0 <= fits["rural"].p_return <= 1


def test_planning_upgrades_overloaded_sections_and_fails_when_it_cannot():
    # SUB - A - B, 1 km overhead sections; the load at B is a transformer rating, at 60 % design load.
    parent, length = [-1, 0, 1], [0.0, 1000.0, 1000.0]
    family = {1: "OH", 2: "OH"}
    res = check(parent, length, family, {2: 5000.0}, CABLES, 20.0)      # 3 MW at B: ~91 A, within 193 A
    assert res.passed and res.upgrades == 0
    res = check(parent, length, family, {2: 12000.0}, CABLES, 20.0)     # ~219 A: above 193 A, within 357 A
    assert res.passed and res.size == {1: 1, 2: 1} and res.max_loading <= 1.0
    res = check(parent, length, family, {2: 30000.0}, CABLES, 20.0)     # ~547 A: over even the larger cable
    assert not res.passed and "larger cable" in res.reason
    # A long overhead line: voltage drop fixed by upgrading along the path, else a failure.
    long_parent, long_len = [-1, 0, 1, 2], [0.0, 8000.0, 8000.0, 8000.0]
    fam3 = {1: "OH", 2: "OH", 3: "OH"}
    res = check(long_parent, long_len, fam3, {3: 3000.0}, CABLES, 20.0)
    assert res.passed and res.min_voltage >= 0.95 and sum(res.size.values()) >= 1
    res = check(long_parent, [0.0, 12000.0, 12000.0, 12000.0], fam3, {3: 3000.0}, CABLES, 20.0)
    assert not res.passed and "voltage drop" in res.reason
    assert planning.DESIGN_LOADING == 0.6 and planning.MAX_DROP == 0.05
