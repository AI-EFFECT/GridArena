"""Planning check of a synthetic grid (Phase 2, axis 7B).

At design load -- every MV/LV transformer at DESIGN_LOADING of its rating, power
factor DESIGN_PF lagging -- each section must stay within its rating and every
node within MAX_DROP of the substation voltage (the busbar at 1 p.u.; the
transformers' own impedance is not part of the check). The sweep solver of
the PF pipeline (pf_datagen/solver.py) solves the MV network.

Sizing, from every section on its family's smaller cable:
1. sections above their rating move to their family's larger cable;
2. while a node is still too low, the smaller-cable section carrying the most
   current on the path from its substation to the lowest node moves up;
3. if neither helps -- an overloaded section already on the larger cable, or no
   smaller-cable section left on that path -- the grid fails the check and the
   caller draws a new one.
"""

from dataclasses import dataclass
from math import pi, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from gridarena.pf_datagen.solver import branch_flows_pu, build_radial_network, solve_radial

DESIGN_LOADING = 0.6
DESIGN_PF = 0.95
MAX_DROP = 0.05
S_BASE_VA = 1e6
FREQ_HZ = 50.0
MAX_UPGRADES = 2000


@dataclass
class CableType:
    cable_id: str
    r_ohm_per_km: float
    x_ohm_per_km: float
    nom_curr_a: float
    cap_uf_per_km: Optional[float] = None
    emerg_curr_a: Optional[float] = None


@dataclass
class PlanningResult:
    passed: bool
    size: Dict[int, int]                    # section (child node) -> 0 smaller / 1 larger cable
    upgrades: int
    max_loading: float
    min_voltage: float
    reason: Optional[str] = None


def check(parent: List[int], length_m: List[float], family: Dict[int, str], loads_kva: Dict[int, float],
          cables: Dict[str, List[CableType]], nominal_kv: float) -> PlanningResult:
    """parent: parent node of each node (-1 for substations); family: 'UG' /
    'OH' per section (keyed by its child node); loads_kva: transformer rating
    per connection-point node; cables: family -> [smaller, larger]."""
    n = len(parent)
    subs = [i for i in range(n) if parent[i] < 0]
    sections = [i for i in range(n) if parent[i] >= 0]
    size = {i: 0 for i in sections}
    z_base = (nominal_kv * 1e3) ** 2 / S_BASE_VA
    i_base = S_BASE_VA / (sqrt(3) * nominal_kv * 1e3)
    phi = np.arccos(DESIGN_PF)
    s = np.zeros(n, dtype=complex)
    for i, kva in loads_kva.items():
        s[i] = kva * 1e3 * DESIGN_LOADING * complex(DESIGN_PF, np.sin(phi)) / S_BASE_VA
    node_ids = [str(i) for i in range(n)]
    upgrades = 0
    while True:
        cab = {i: cables[family[i]][size[i]] for i in sections}
        z = {str(i): complex(cab[i].r_ohm_per_km, cab[i].x_ohm_per_km) * length_m[i] / 1000.0 / z_base
             for i in sections}
        ysh = {str(i): 0.5j * 2 * pi * FREQ_HZ * (cab[i].cap_uf_per_km or 0.0) * 1e-6 * length_m[i] / 1000.0 * z_base
               for i in sections}
        net = build_radial_network(node_ids, [str(i) for i in subs], [(str(i), str(parent[i]), str(i)) for i in sections],
                                   z, ysh)
        order = [int(x) for x in net.node_ids]
        v, converged, _ = solve_radial(net, s[order])
        if not converged or not np.all(np.isfinite(v)):
            return PlanningResult(False, size, upgrades, np.inf, 0.0, "the design load doesn't converge")
        v_by = dict(zip(order, v))
        amps = {}
        for i in sections:
            i_from, i_to, _, _ = branch_flows_pu(v_by[parent[i]], v_by[i], 1 / z[str(i)], ysh[str(i)], 1.0)
            amps[i] = max(abs(i_from), abs(i_to)) * i_base
        loading = {i: amps[i] / cab[i].nom_curr_a for i in sections}
        vmin_node = min(range(n), key=lambda k: abs(v_by[k]))
        vmin = abs(v_by[vmin_node])
        over = [i for i in sections if loading[i] > 1.0]
        if not over and vmin >= 1.0 - MAX_DROP:
            return PlanningResult(True, size, upgrades, max(loading.values(), default=0.0), vmin)
        if upgrades >= MAX_UPGRADES:
            return PlanningResult(False, size, upgrades, max(loading.values()), vmin, "too many upgrades")
        if over:
            stuck = [i for i in over if size[i] == 1 or len(cables[family[i]]) < 2]
            if stuck:
                return PlanningResult(False, size, upgrades, max(loading.values()), vmin,
                                      f"{len(stuck)} section(s) overloaded on the larger cable")
            for i in over:
                size[i] = 1
                upgrades += 1
            continue
        path, k = [], vmin_node
        while parent[k] >= 0:
            path.append(k)
            k = parent[k]
        candidates = [i for i in path if size[i] == 0 and len(cables[family[i]]) > 1]
        if not candidates:
            return PlanningResult(False, size, upgrades, max(loading.values()), vmin,
                                  f"voltage drop {100 * (1 - vmin):.1f} % with every section on the larger cable")
        size[max(candidates, key=lambda i: amps[i])] = 1
        upgrades += 1
