"""Sparse Y-bus assembler for the PF-datagen output record group.

Built in per unit, where buses on different voltage levels (MV nodes, the LV
side of MV/LV transformers) share one consistent system. Each in-service
branch contributes its pi-model / off-nominal-ratio stamp (see
solver.branch_flows_pu): series admittance y on the from side, ideal ratio n
at the to end, y_sh = j*B/2 to ground at each end:
Yff += y + y_sh, Ytt += y/n^2 + y_sh, Yft += -y/n, Ytf += -y/n. Capacitor
banks and reactors add their admittance to their node's diagonal entry.
"""

from typing import Dict, List, Optional, Tuple

from gridarena.pf_datagen.grid_loader import MVBranch


def build_ybus_entries(
    sub_branches: List[MVBranch], y_pu: Dict[str, complex],
    y_shunt_half_pu: Optional[Dict[str, complex]] = None,
    ratio: Optional[Dict[str, float]] = None,
    node_y_shunt_pu: Optional[Dict[str, complex]] = None,
) -> List[Tuple[str, str, float, float]]:
    """Returns sparse (i_node_id, j_node_id, G, B) rows in p.u. for one
    topology variant's energised sub-network. node_y_shunt_pu: admittance of
    shunt elements at the energised nodes."""
    y_shunt_half_pu = y_shunt_half_pu or {}
    ratio = ratio or {}
    y_matrix: Dict[Tuple[str, str], complex] = {}

    def _add(i, j, val):
        y_matrix[(i, j)] = y_matrix.get((i, j), 0j) + val

    for b in sub_branches:
        y = y_pu.get(b.connection_id)
        if y is None:
            continue
        y_sh = y_shunt_half_pu.get(b.connection_id, 0j)
        n = ratio.get(b.connection_id, 1.0)
        i, j = b.from_node_id, b.to_node_id
        _add(i, i, y + y_sh)
        _add(j, j, y / n ** 2 + y_sh)
        _add(i, j, -y / n)
        _add(j, i, -y / n)
    for nid, y in (node_y_shunt_pu or {}).items():
        _add(nid, nid, y)

    return [(i, j, y.real, y.imag) for (i, j), y in y_matrix.items() if y != 0]
