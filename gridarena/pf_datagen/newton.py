"""Newton-Raphson power flow for MV networks of any topology (the request's
solver = "newton").

The radial sweep (pf_datagen/solver.py) needs every energised part to be a
tree fed by one substation. This solver has no such limit: loops, closed tie
lines and parts fed by several substations all solve. It uses the same model
-- per unit on the run's bases, pi-model branches with the off-nominal ratio
at the to end, capacitor banks / reactors as node admittances, ZIP loads and
constant DER injections -- so on a radial grid both solvers give the same
voltages.

Formulation: polar Newton-Raphson on the Y-bus (sparse, or dense below
DENSE_MAX_NODES buses, where that is faster). Every substation is a
reference bus at the set-point (angle 0); every other bus is PQ, its
specified injection s_gen - demand(s_load, |V|) with the ZIP voltage terms in
the Jacobian (derivatives as in MATPOWER's dSbus_dV). Voltage-controlling DER
are handled by the shared control loop (solver.solve_with_controls), not as
PV buses, so both solvers settle controls the same way. A step voltage
regulator on a loop holds its starting tap rather than regulating: in a
closed loop between stiff sources any tap away from neutral drives a
circulating current around the loop (tens of MW on an MV feeder), which is
why regulators aren't operated that way (connections_on_loops).
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

from gridarena.pf_datagen.solver import DEFAULT_TOL, LoadModel, RootVoltage, initial_voltage

NEWTON_MAX_ITER = 30
DENSE_MAX_NODES = 400   # below this, dense linear algebra beats the sparse overhead


class MeshedNetwork:
    """The energised network as a Y-bus, for Newton-Raphson. node_ids has the
    roots (substations) first. Offers the same solve / controlled_node /
    sensitivity interface as solver.RadialNetwork."""

    def __init__(self, node_ids: Iterable[str], root: Union[str, Iterable[str]],
                 branches: Iterable[Tuple[str, str, str]], y_pu: Dict[str, complex],
                 y_shunt_half_pu: Optional[Dict[str, complex]] = None,
                 ratio: Optional[Dict[str, float]] = None,
                 node_y_shunt_pu: Optional[Dict[str, complex]] = None):
        roots = [root] if isinstance(root, str) else list(root)
        root_set = set(roots)
        self.node_ids: List[str] = roots + [n for n in node_ids if n not in root_set]
        self.n_roots = len(roots)
        index = {nid: i for i, nid in enumerate(self.node_ids)}
        branches = list(branches)
        for cid, a, b in branches:
            if a not in index or b not in index:
                raise ValueError(f"Connection '{cid}' links unknown node(s) '{a}'/'{b}'.")
        y_shunt_half_pu = y_shunt_half_pu or {}
        self.branch_ids = [cid for cid, _, _ in branches]
        self.f = np.array([index[a] for _, a, _ in branches], dtype=int)
        self.t = np.array([index[b] for _, _, b in branches], dtype=int)
        self.y = np.array([y_pu[cid] for cid in self.branch_ids], dtype=complex)
        self.y_sh = np.array([y_shunt_half_pu.get(cid, 0j) for cid in self.branch_ids], dtype=complex)
        self.base_ratio = {cid: (ratio or {}).get(cid, 1.0) for cid in self.branch_ids}
        self.to_index = {cid: int(t) for cid, t in zip(self.branch_ids, self.t)}
        n = len(self.node_ids)
        self.y_node = np.zeros(n, dtype=complex)
        for nid, y in (node_y_shunt_pu or {}).items():
            if nid not in index:
                raise ValueError(f"Shunt at unknown node '{nid}'.")
            self.y_node[index[nid]] += y
        self._check_supplied()

    def _check_supplied(self):
        """Every node must be reachable from a root."""
        n = len(self.node_ids)
        adj = sp.coo_matrix((np.ones(len(self.f)), (self.f, self.t)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)
        fed = set(labels[:self.n_roots])
        unfed = [self.node_ids[i] for i in range(n) if labels[i] not in fed]
        if unfed:
            raise ValueError(f"Node(s) not connected to any of {self.node_ids[:self.n_roots]}: {unfed[:5]}")

    def ybus(self, ratios: Optional[Dict[str, float]] = None) -> sp.csr_matrix:
        """Y-bus (p.u.) with the given branch ratios (default: the base ones)."""
        n_ratio = np.array([(ratios or {}).get(cid, self.base_ratio[cid]) for cid in self.branch_ids])
        yff = self.y + self.y_sh
        ytt = self.y / n_ratio ** 2 + self.y_sh
        yft = -self.y / n_ratio
        n = len(self.node_ids)
        rows = np.concatenate([self.f, self.t, self.f, self.t, np.arange(n)])
        cols = np.concatenate([self.f, self.t, self.t, self.f, np.arange(n)])
        vals = np.concatenate([yff, ytt, yft, yft, self.y_node])
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    def solve(self, s_pu: np.ndarray, v_root_pu: RootVoltage = 1.0 + 0j, ratios: Optional[Dict[str, float]] = None,
              load_model: Optional[LoadModel] = None, s_gen_pu: Optional[np.ndarray] = None,
              y_shunt_extra_pu: Optional[np.ndarray] = None,
              tol: float = DEFAULT_TOL, max_iter: int = NEWTON_MAX_ITER) -> Tuple[np.ndarray, bool, int]:
        """Newton-Raphson from a flat start. s_pu: demand at 1 p.u. (load
        convention), s_gen_pu: constant injection (generator convention), both
        in node_ids order; v_root_pu: one set-point for every substation or one
        per substation; y_shunt_extra_pu: node shunt admittance on top of the
        network's own. Returns (v_pu, converged, n_iter), converged when the
        largest power mismatch is below tol."""
        y = self.ybus(ratios)
        if y_shunt_extra_pu is not None:
            y = (y + sp.diags(y_shunt_extra_pu)).tocsr()
        v0 = initial_voltage(self.node_ids, range(self.n_roots), v_root_pu)
        return newton_raphson(y, self.n_roots, s_pu, v0, load_model, s_gen_pu, tol, max_iter)

    def controlled_node(self, connection_id: str) -> int:
        """Index of the node a regulator on this connection controls (its to end)."""
        return self.to_index[connection_id]

    def sensitivity(self, node_idx: List[int], ratios: Optional[Dict[str, float]] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """R and X of the driving-point / transfer impedance between the given
        nodes, with the substations held at fixed voltage: |V_i| moves by
        about sum_j (R_ij dP_j + X_ij dQ_j). On a radial feeder without
        shunts this is the common-path impedance the sweep uses."""
        if not node_idx:
            return np.zeros((0, 0)), np.zeros((0, 0))
        y = self.ybus(ratios)
        k = self.n_roots
        rhs = np.zeros((y.shape[0] - k, len(node_idx)), dtype=complex)
        for c, i in enumerate(node_idx):
            rhs[i - k, c] = 1.0
        y_nn = y[k:, k:]
        if y.shape[0] <= DENSE_MAX_NODES:
            z = np.linalg.solve(y_nn.toarray(), rhs)
        else:
            z = splu(y_nn.tocsc()).solve(rhs)
        z = z[[i - k for i in node_idx], :]
        return z.real, z.imag


def connections_on_loops(branches: Iterable[Tuple[str, str, str]], candidates: Iterable[str],
                         roots: Iterable[str] = ()) -> Set[str]:
    """The candidate connections that lie on a loop of the network formed by
    `branches` (connection_id, from, to): removing one leaves its two ends
    still connected. The substations (`roots`) count as one node, since they
    are tied together through the upstream grid -- a path from one
    substation to another is a loop too. Parallel connections count as one."""
    root_set = set(roots)

    def node(n):
        return "__substations__" if n in root_set else n

    branches = [(c, node(a), node(b)) for c, a, b in branches]
    out = set()
    for cid in candidates:
        g = nx.MultiGraph()
        g.add_edges_from((a, b) for c, a, b in branches if c != cid)
        ends = next((a, b) for c, a, b in branches if c == cid)
        if ends[0] in g and ends[1] in g and nx.has_path(g, *ends):
            out.add(cid)
    return out


def newton_raphson(y: sp.csr_matrix, n_roots: int, s_pu: np.ndarray, v0: np.ndarray,
                   load_model: Optional[LoadModel], s_gen_pu: Optional[np.ndarray],
                   tol: float, max_iter: int) -> Tuple[np.ndarray, bool, int]:
    """v0: starting voltages; the first n_roots (the substations) stay fixed."""
    n = y.shape[0]
    v = np.array(v0, dtype=complex)
    if n == n_roots:
        return v, True, 0
    s_gen = np.zeros(n, dtype=complex) if s_gen_pu is None else s_gen_pu
    zip_load = load_model is not None and not load_model.is_constant_power
    pq = np.arange(n_roots, n)

    def mismatch(v):
        vm = np.abs(v)
        demand = load_model.demand(s_pu, vm) if zip_load else s_pu
        return v * np.conj(y @ v) - (s_gen - demand)

    dense = n <= DENSE_MAX_NODES
    if dense:
        y = y.toarray()
    f = mismatch(v)
    for it in range(1, max_iter + 1):
        vm = np.abs(v)
        ibus = y @ v
        vn = v / vm
        if zip_load:
            (pz, pi, _), (qz, qi, _) = load_model.p, load_model.q
            d_demand = s_pu.real * (2 * pz * vm + pi) + 1j * s_pu.imag * (2 * qz * vm + qi)
        else:
            d_demand = np.zeros(n)
        rhs = -np.concatenate([f[pq].real, f[pq].imag])
        if dense:
            ds_dva = 1j * v[:, None] * np.conj(np.diag(ibus) - y * v[None, :])
            ds_dvm = v[:, None] * np.conj(y * vn[None, :]) + np.diag(np.conj(ibus) * vn + d_demand)
            ds_dva, ds_dvm = ds_dva[np.ix_(pq, pq)], ds_dvm[np.ix_(pq, pq)]
            jac = np.block([[ds_dva.real, ds_dvm.real], [ds_dva.imag, ds_dvm.imag]])
            try:
                dx = np.linalg.solve(jac, rhs)
            except np.linalg.LinAlgError:
                return np.full(n, np.nan + 0j), False, it
        else:
            dv_v = sp.diags(v)
            dv_n = sp.diags(vn)
            ds_dva = 1j * dv_v @ (sp.diags(ibus) - y @ dv_v).conj()
            ds_dvm = dv_v @ (y @ dv_n).conj() + sp.diags(np.conj(ibus) * vn + d_demand)
            ds_dva = sp.csr_matrix(ds_dva)[pq][:, pq]
            ds_dvm = sp.csr_matrix(ds_dvm)[pq][:, pq]
            jac = sp.bmat([[ds_dva.real, ds_dvm.real], [ds_dva.imag, ds_dvm.imag]], format="csc")
            dx = spsolve(jac, rhs)
        if not np.all(np.isfinite(dx)):
            return np.full(n, np.nan + 0j), False, it
        va = np.angle(v)
        va[pq] += dx[:len(pq)]
        vm = vm.copy()
        vm[pq] += dx[len(pq):]
        v = vm * np.exp(1j * va)
        if not np.all(np.isfinite(v)):
            return v, False, it
        f = mismatch(v)
        if np.max(np.abs(f[pq])) < tol:
            return v, True, it
    return v, False, max_iter
