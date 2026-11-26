"""Generating a batch of synthetic grids: cable types, assembly into ordinary
MV grids (schemas.MVGrid) and the generator report.

Grids are named `SYN-<archetype>-NNNN`, numbered after the highest existing
one, with region "synthetic"; the description records the generator version,
seed and options. Each feeder's first section and the section into each tie
end are closed switches. Grid k of a batch draws from default_rng([seed, k]), so a
batch reproduces from its seed. Per family the smaller cable is the stored
type used by the most grids and the larger one the highest-rated stored type
(test grids' rare types can't take the smaller slot that way), copied as
`SYN-UG-S` / `SYN-UG-L` / `SYN-OH-S` / `SYN-OH-L`.
"""

import json
from typing import Dict, List, Optional

import numpy as np

import gridarena.schemas as schema
from gridarena.pf_datagen.parameters import cable_family
from gridarena.synthetic.calibrate import ARCHETYPES, METRICS, SYNTHETIC_PREFIX, _pct, grid_metrics
from gridarena.synthetic.generator import GENERATOR_VERSION, NOMINAL_KV, Options, draw_size, equip, grow, to_lat_lon
from gridarena.synthetic.planning import CableType, check

MAX_ATTEMPTS = 20
ATTEMPTS_PER_SIZE = 5
_FAMILY = {"underground": "UG", "overhead": "OH"}


def choose_cables(cursor) -> Dict[str, List[CableType]]:
    cursor.execute(
        """
        SELECT k."ImpReal", k."ImpImag", k."NomCurr", max(k."CapUfPerKm"), max(k."EmergCurr"),
               count(DISTINCT c.mv_grid_id)
        FROM "MVCable" k JOIN "MVConnection" c ON c."CableId" = k."CableId" AND c.mv_grid_id = k.mv_grid_id
        WHERE k.mv_grid_id NOT LIKE %s AND k."ImpReal" > 0 AND k."NomCurr" > 0
        GROUP BY 1, 2, 3 ORDER BY 6 DESC, 3
        """,
        (SYNTHETIC_PREFIX + "%",),
    )
    by: Dict[str, list] = {"UG": [], "OH": []}
    for r, x, nom, cap, emerg, _ in cursor.fetchall():
        by[_FAMILY[cable_family(r, x, cap)]].append((r, x, nom, cap, emerg))
    out = {}
    for fam, types in by.items():
        if not types:
            continue
        smaller = types[0]                                     # used by the most grids
        larger = max(types, key=lambda t: t[2])                # highest rating
        pair = [smaller] if larger[2] <= smaller[2] else [smaller, larger]
        out[fam] = [CableType(f"SYN-{fam}-{'SL'[k] if len(pair) > 1 else 'S'}", *t) for k, t in enumerate(pair)]
    if "UG" not in out and "OH" not in out:
        raise ValueError("No stored cable type to build synthetic grids from.")
    for fam, other in (("UG", "OH"), ("OH", "UG")):
        out.setdefault(fam, out[other])
    return out


def next_numbers(cursor) -> Dict[str, int]:
    cursor.execute('SELECT mv_grid_id FROM "MVGrid" WHERE mv_grid_id LIKE %s', (SYNTHETIC_PREFIX + "%",))
    out = {a: 1 for a in ARCHETYPES}
    for (gid,) in cursor.fetchall():
        parts = gid.split("-")
        if len(parts) == 3 and parts[1] in out and parts[2].isdigit():
            out[parts[1]] = max(out[parts[1]], int(parts[2]) + 1)
    return out


def _assemble(grid_id, archetype, sk, eq, sizes, cables, opts, seed, k) -> schema.MVGrid:
    n = len(sk.parent)
    name = {i: (f"SUB{i + 1}" if sk.parent[i] < 0 else f"N{i:04d}") for i in range(n)}
    coords = to_lat_lon(sk.xy, opts.reference)
    cps = set(eq["cps"])
    nodes = [schema.MVNode(node_id=name[i], coord_lat=round(coords[i][0], 7), coord_lon=round(coords[i][1], 7),
                           node_type="substation" if sk.parent[i] < 0 else
                           ("connection_point" if i in cps else "junction")) for i in range(n)]
    used = {}
    conns = []
    # Closed switches: each feeder's breaker (its first section) and the section
    # into each tie end, so a tie's loop has switches to open (reconfiguration).
    switches = {i for i in range(n) if sk.parent[i] >= 0 and sk.parent[i] in sk.substations}
    switches |= {e for a, b, _ in sk.ties for e in (a, b)}
    for i in range(n):
        if sk.parent[i] < 0:
            continue
        c = cables[eq["family"][i]][sizes[i]]
        used[c.cable_id] = c
        conns.append(schema.MVConnection(connection_id=f"L{i:04d}", from_node_id=name[sk.parent[i]],
                                         to_node_id=name[i], cable_id=c.cable_id, length=round(sk.length[i], 1),
                                         is_switch=i in switches))
    for t, (a, b, length) in enumerate(sk.ties, start=1):
        fams = [eq["family"][a], eq["family"][b]]
        c = cables["OH" if fams.count("OH") == 2 else "UG"][0]
        used[c.cable_id] = c
        conns.append(schema.MVConnection(connection_id=f"TIE{t:02d}", from_node_id=name[a], to_node_id=name[b],
                                         cable_id=c.cable_id, length=round(length, 1), is_switch=True,
                                         normally_open=True))
    cable_rows = [schema.MVCable(cable_id=c.cable_id, imp_real=c.r_ohm_per_km, imp_imag=c.x_ohm_per_km,
                                 nom_curr=c.nom_curr_a, cap_uf_per_km=c.cap_uf_per_km, emerg_curr=c.emerg_curr_a)
                  for c in used.values()]
    cp_rows = [schema.MVConnectionPoint(connection_point_id=f"CP{i:04d}", node_id=name[i]) for i in sorted(cps)]
    tr_rows = [schema.Transformer(transformer_id=f"T{i:04d}", connection_point_id=f"CP{i:04d}",
                                  rated_power_kva=eq["ratings"][i], primary_voltage_kv=NOMINAL_KV,
                                  secondary_voltage_v=230.0) for i in sorted(cps)]
    description = "Synthetic " + archetype + " feeder; " + json.dumps({
        "generator_version": GENERATOR_VERSION, "seed": seed, "index": k,
        "options": {"feeders_per_substation": opts.feeders_per_substation,
                    "second_substation": opts.second_substation,
                    "ties_between_feeders": opts.ties_between_feeders,
                    "n_nodes": list(opts.n_nodes) if opts.n_nodes else None,
                    "reference": list(opts.reference)},
    })
    return schema.MVGrid(mv_grid_id=grid_id, name=f"Synthetic {archetype} feeder", description=description,
                         nominal_voltage_kv=NOMINAL_KV, region="synthetic", nodes=nodes, cables=cable_rows,
                         connections=conns, connection_points=cp_rows, transformers=tr_rows)


def _metrics(grid: schema.MVGrid, cables: Dict[str, List[CableType]]) -> dict:
    oh_ids = {c.cable_id for c in cables.get("OH", [])} - {c.cable_id for c in cables.get("UG", [])}
    edges = [(c.from_node_id, c.to_node_id, c.length, c.cable_id in oh_ids)
             for c in grid.connections if not c.normally_open]
    return grid_metrics({nd.node_id: nd.node_type for nd in grid.nodes}, edges, len(grid.connection_points))


def generate(fits, archetype: str, count: int, seed: int, opts: Options, cables, numbers: Dict[str, int]):
    """`count` grids of `archetype` ('all': archetypes drawn in their stored
    proportions). Returns (grids, report)."""
    available = [a for a in ARCHETYPES if a in fits]
    if archetype != "all" and archetype not in fits:
        raise ValueError(f"No stored grid of archetype '{archetype}' to calibrate on.")
    weights = np.array([fits[a].n_grids for a in available], dtype=float)
    pick = np.random.default_rng([seed, 0]).choice(len(available), size=count, p=weights / weights.sum())
    numbers = dict(numbers)
    grids, per_grid, measured = [], [], {a: [] for a in available}
    for k in range(count):
        a = archetype if archetype != "all" else available[int(pick[k])]
        fit = fits[a]
        rng = np.random.default_rng([seed, k + 1])
        result = None
        resized = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # The size is kept for ATTEMPTS_PER_SIZE shapes (see generator.draw_size);
            # one no shape can carry -- too much load for a feeder -- is drawn again.
            if (attempt - 1) % ATTEMPTS_PER_SIZE == 0:
                size = draw_size(fit, opts, rng)
                resized += attempt > 1
            sk = grow(fit, opts, rng, size)
            eq = equip(sk, fit, rng)
            loads = {i: eq["ratings"][i] for i in eq["cps"]}
            result = check(sk.parent, sk.length, eq["family"], loads, cables, NOMINAL_KV)
            if result.passed:
                break
        if not result.passed:
            raise ValueError(f"Grid {k + 1} of the batch failed the planning check {MAX_ATTEMPTS} times "
                             f"(last: {result.reason}). Try fewer nodes or another seed.")
        grid_id = f"{SYNTHETIC_PREFIX}{a}-{numbers[a]:04d}"
        numbers[a] += 1
        grid = _assemble(grid_id, a, sk, eq, result.size, cables, opts, seed, k)
        m = _metrics(grid, cables)
        measured[a].append(m)
        grids.append(grid)
        per_grid.append({"mv_grid_id": grid_id, "archetype": a, "attempts": attempt, "sizes_redrawn": resized,
                         "upgrades": result.upgrades,
                         "max_loading": result.max_loading, "min_voltage_pu": result.min_voltage,
                         "n_ties": len(sk.ties), **m})
    report = {
        "generator_version": GENERATOR_VERSION, "seed": seed, "count": count,
        "archetypes": {a: {"generated": len(measured[a]),
                           "metrics": {k: {"generated": _pct([m[k] for m in measured[a]]),
                                           "stored": _pct(fits[a].real[k])} for k in METRICS}}
                       for a in available if measured[a]},
        "sizes_redrawn": sum(g["sizes_redrawn"] for g in per_grid),
        "grids": per_grid,
    }
    return grids, report
