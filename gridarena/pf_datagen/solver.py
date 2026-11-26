"""Per-unit backward/forward-sweep power flow for radial MV networks.

A separate solver from powerflow/powerflow_algorithm.py::full_pf, which LV
power flow, the digital twin and RL keep using unchanged. Differences:

- Branch orientation and processing order are derived from the graph (a
  breadth-first search from the substations), so results don't depend on the
  order connections are stored in or on their from/to direction. full_pf
  needs edges listed root-outward and pointing parent -> child. Several
  substations are fine as long as each energised component is fed by exactly
  one (a radial forest).
- Pi-model branches: series impedance plus half the branch's charging
  susceptance at each end (see grid_loader.branch_charging_susceptance).
- Off-nominal ratio branches (MV/LV transformer taps, step voltage
  regulators): series impedance on the from side, then an ideal ratio n at
  the to end, v_to = n * (v_from - z * i_from). Such branches must point
  away from the substation.
- Each sweep iteration is vectorised per tree level -- O(N) work instead of
  full_pf's O(E^2) nested edge loop.
- Everything is per unit on the run's bases (pf_datagen/per_unit.py), with
  each bus on its own voltage level's base. The network is balanced
  positive-sequence, so three-phase powers and line-to-line voltages are used
  directly; there is no S/3, V/sqrt(3) step.

It also computes the LinDistFlow approximation (Baran & Wu, 1989): lossless
flows and a voltage-squared drop that is linear in P and Q. That is the MV
counterpart of the DC power flow datakit reports as a baseline -- DC power
flow assumes R << X and flat voltage magnitudes, neither of which holds on
MV cables.

Loads use the load convention (positive P/Q = consumption). They are
constant power by default, or a ZIP mix (LoadModel) re-evaluated at each
iteration's voltage. Capacitor banks and reactors are constant admittances at
their node, alongside the branches' charging. Generators (DER) inject
constant P/Q within one sweep, in the generator convention (positive =
injected); their controls, like the regulators', are settled in an outer loop
(solve_with_controls).
"""

from collections import deque
from dataclasses import dataclass
from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 1000


@dataclass(frozen=True)
class LoadModel:
    """ZIP load model: the share of each load's P and of its Q that behaves
    as constant impedance (z, ~|V|^2), constant current (i, ~|V|) and
    constant power (p). The reference power is drawn at 1 p.u."""
    p: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    q: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def is_constant_power(self) -> bool:
        return self.p == (0.0, 0.0, 1.0) and self.q == (0.0, 0.0, 1.0)

    def demand(self, s_ref: np.ndarray, vm: np.ndarray) -> np.ndarray:
        """Power drawn at voltage magnitudes vm (p.u.) by loads whose
        reference (1 p.u.) power is s_ref."""
        if self.is_constant_power:
            return s_ref
        vm2 = vm * vm
        kp = self.p[0] * vm2 + self.p[1] * vm + self.p[2]
        kq = self.q[0] * vm2 + self.q[1] * vm + self.q[2]
        return s_ref.real * kp + 1j * s_ref.imag * kq


# A substation set-point: one value for every substation, or {substation id: value}.
RootVoltage = Union[complex, float, Dict[str, complex]]


def initial_voltage(node_ids: List[str], root_idx, v_root_pu: RootVoltage) -> np.ndarray:
    """Flat start: the substations (root_idx) at their set-point, every other
    node at it too -- or, with one value per substation, at their mean."""
    if isinstance(v_root_pu, dict):
        mean = np.mean(list(v_root_pu.values())) if v_root_pu else 1.0
        v = np.full(len(node_ids), mean, dtype=complex)
        for i in root_idx:
            v[i] = v_root_pu.get(node_ids[i], mean)
        return v
    return np.full(len(node_ids), v_root_pu, dtype=complex)


class MeshedNetworkError(Exception):
    """The energised network contains a loop, or a component fed by two
    substations; the radial sweep can't solve it."""


@dataclass
class RadialNetwork:
    node_ids: List[str]                 # the roots (substations) first
    parent: np.ndarray                  # parent index per node, -1 at a root
    levels: List[np.ndarray]            # node indices by depth; levels[0] are the roots
    z_pu: np.ndarray                    # series impedance of the branch to the parent (0 at a root)
    y_shunt_pu: np.ndarray              # total shunt admittance at each node
    ratio: np.ndarray                   # off-nominal ratio of the branch to the parent (1 at a root)
    branch_child: Dict[str, int]        # connection_id -> index of its child-side node

    def branch_sign(self, connection_id: str, from_node_id: str) -> int:
        """+1 if the connection's from end is the parent side, -1 if it is
        the child side -- to report parent->child quantities in the
        connection's own from/to convention."""
        return -1 if self.node_ids[self.branch_child[connection_id]] == from_node_id else 1

    def ratios_with(self, ratio_by_branch: Dict[str, float]) -> np.ndarray:
        """The network's ratio array with some branches' ratios replaced."""
        out = self.ratio.copy()
        for cid, n in ratio_by_branch.items():
            out[self.branch_child[cid]] = n
        return out

    # The interface solve_with_controls uses, shared with newton.MeshedNetwork.

    def solve(self, s_pu: np.ndarray, v_root_pu: RootVoltage = 1.0 + 0j, ratios: Optional[Dict[str, float]] = None,
              load_model: Optional["LoadModel"] = None, s_gen_pu: Optional[np.ndarray] = None,
              y_shunt_extra_pu: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, int]:
        """solve_radial with some branches' ratios replaced (e.g. regulator taps)."""
        return solve_radial(self, s_pu, v_root_pu, ratio=self.ratios_with(ratios or {}),
                            load_model=load_model, s_gen_pu=s_gen_pu, y_shunt_extra_pu=y_shunt_extra_pu)

    def controlled_node(self, connection_id: str) -> int:
        """Index of the node a regulator on this connection controls (its child end)."""
        return self.branch_child[connection_id]

    def sensitivity(self, node_idx: List[int], ratios: Optional[Dict[str, float]] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        return _path_matrix(self, node_idx)


def build_radial_network(
    node_ids: Iterable[str],
    root: Union[str, Iterable[str]],
    branches: Iterable[Tuple[str, str, str]],
    z_pu: Dict[str, complex],
    y_shunt_half_pu: Optional[Dict[str, complex]] = None,
    ratio: Optional[Dict[str, float]] = None,
    directional: AbstractSet[str] = frozenset(),
    node_y_shunt_pu: Optional[Dict[str, complex]] = None,
) -> RadialNetwork:
    """Orient a set of connections into a forest rooted at `root`: one
    substation node id, or several.

    branches: (connection_id, from_node_id, to_node_id), in any order and
    direction. z_pu / y_shunt_half_pu: per-unit series impedance and half the
    shunt admittance of each connection. ratio: off-nominal ratio at the to
    end of a connection (default 1). Connections in `directional` (those with
    a ratio, e.g. transformers and regulators) must point away from the root.
    node_y_shunt_pu: shunt admittance of elements at a node (capacitor banks,
    reactors), added to the branches' charging there.

    Raises MeshedNetworkError if the connections form a loop or join two
    substations, and ValueError if a node can't be reached from any root, a
    connection names an unknown node, or a directional connection points
    towards its root.
    """
    y_shunt_half_pu = y_shunt_half_pu or {}
    ratio = ratio or {}
    roots = [root] if isinstance(root, str) else list(root)
    root_set = set(roots)
    ordered = roots + [n for n in node_ids if n not in root_set]
    index = {nid: i for i, nid in enumerate(ordered)}
    n = len(ordered)

    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(n)]
    from_node: Dict[str, str] = {}
    y_shunt = np.zeros(n, dtype=complex)
    for cid, a, b in branches:
        if a not in index or b not in index:
            raise ValueError(f"Connection '{cid}' links unknown node(s) '{a}'/'{b}'.")
        ia, ib = index[a], index[b]
        adjacency[ia].append((ib, cid))
        adjacency[ib].append((ia, cid))
        from_node[cid] = a
        half = y_shunt_half_pu.get(cid, 0j)
        y_shunt[ia] += half
        y_shunt[ib] += half

    parent = np.full(n, -1, dtype=int)
    parent_branch: List[Optional[str]] = [None] * n
    depth = np.full(n, -1, dtype=int)
    source = np.full(n, -1, dtype=int)          # which root feeds each node
    n_roots = len(roots)
    depth[:n_roots] = 0
    source[:n_roots] = np.arange(n_roots)
    queue = deque(range(n_roots))
    while queue:
        u = queue.popleft()
        for v, cid in adjacency[u]:
            if cid == parent_branch[u]:
                continue
            if depth[v] != -1:
                if source[v] != source[u]:
                    first, second = sorted((roots[source[u]], roots[source[v]]))
                    raise MeshedNetworkError(
                        f"Connection '{cid}' joins the parts fed by substations '{first}' and '{second}'; "
                        "the radial sweep needs each component fed by one substation (use solver 'newton')."
                    )
                raise MeshedNetworkError(
                    f"Connection '{cid}' closes a loop between '{ordered[u]}' and '{ordered[v]}'; "
                    "the radial sweep needs a tree (use solver 'newton')."
                )
            depth[v] = depth[u] + 1
            parent[v] = u
            parent_branch[v] = cid
            source[v] = source[u]
            queue.append(v)

    unreached = [ordered[i] for i in range(n) if depth[i] == -1]
    if unreached:
        raise ValueError(f"Node(s) not connected to any of {roots}: {unreached[:5]}")

    z = np.zeros(n, dtype=complex)
    n_ratio = np.ones(n)
    branch_child = {}
    for i in range(n_roots, n):
        cid = parent_branch[i]
        if cid in directional and from_node[cid] != ordered[parent[i]]:
            raise ValueError(
                f"Connection '{cid}' has a transformer or regulator but points towards the substation; "
                "its from node must be on the substation side."
            )
        z[i] = z_pu[cid]
        n_ratio[i] = ratio.get(cid, 1.0)
        branch_child[cid] = i
    for nid, y in (node_y_shunt_pu or {}).items():
        if nid not in index:
            raise ValueError(f"Shunt at unknown node '{nid}'.")
        y_shunt[index[nid]] += y
    levels = [np.flatnonzero(depth == d) for d in range(int(depth.max()) + 1)]
    return RadialNetwork(ordered, parent, levels, z, y_shunt, n_ratio, branch_child)


def _subtree_sums(net: RadialNetwork, values: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Each node's value plus everything below it, scaled by the ratio of
    every branch crossed on the way up (leaves to root, one vectorised step
    per tree level). With values = currents this is the current each branch
    carries on its child side; times its ratio, on its parent side."""
    out = values.copy()
    for level in reversed(net.levels[1:]):
        np.add.at(out, net.parent[level], ratio[level] * out[level])
    return out


def solve_radial(
    net: RadialNetwork, s_pu: np.ndarray, v_root_pu: RootVoltage = 1.0 + 0j,
    tol: float = DEFAULT_TOL, max_iter: int = DEFAULT_MAX_ITER, ratio: Optional[np.ndarray] = None,
    load_model: Optional[LoadModel] = None, s_gen_pu: Optional[np.ndarray] = None,
    y_shunt_extra_pu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, int]:
    """Backward/forward sweep. s_pu: complex power drawn at each node (in
    net.node_ids order) at 1 p.u. -- the actual demand when the load model is
    constant power (the default), otherwise load_model.demand(s_pu, |v|);
    s_gen_pu: constant power injected at each node (generator convention);
    ratio: per-node branch ratios (default net.ratio); v_root_pu: one
    set-point for every substation or one per substation;
    y_shunt_extra_pu: shunt admittance added at each node on top of the
    network's own (e.g. capacitor steps switched in or out this sample).
    Returns (v_pu, converged, n_iter); a network with no branches is
    trivially solved in 0 iterations. Stops early, unconverged, if the
    voltages go non-finite."""
    ratio = net.ratio if ratio is None else ratio
    zip_load = load_model is not None and not load_model.is_constant_power
    v = initial_voltage(net.node_ids, net.levels[0], v_root_pu)
    if len(net.levels) == 1:            # substations only, no branches
        return v, True, 0
    y_shunt = net.y_shunt_pu if y_shunt_extra_pu is None else net.y_shunt_pu + y_shunt_extra_pu

    for it in range(1, max_iter + 1):
        # Current drawn at each node: its load plus its share of the shunts.
        s_now = load_model.demand(s_pu, np.abs(v)) if zip_load else s_pu
        if s_gen_pu is not None:
            s_now = s_now - s_gen_pu
        i_node = np.conj(s_now / v) + y_shunt * v
        # Backward: current through each node's branch on its child side.
        j = _subtree_sums(net, i_node, ratio)
        # Forward, root outwards: drop across the series impedance (carrying
        # the parent-side current n*j), then the ideal ratio.
        v_new = v.copy()
        for level in net.levels[1:]:
            n = ratio[level]
            v_new[level] = n * (v_new[net.parent[level]] - net.z_pu[level] * n * j[level])
        if not np.all(np.isfinite(v_new)):
            return v_new, False, it
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, True, it
        v = v_new
    return v, False, max_iter


@dataclass
class RegulatorControl:
    """A step voltage regulator on one branch: keeps the voltage at the
    branch's child (to) node within v_set_pu +/- deadband_pu / 2."""
    connection_id: str
    v_set_pu: float
    deadband_pu: float
    tap_step_pct: float
    tap_min: int
    tap_max: int
    tap_pos: int  # starting position


@dataclass
class ShuntControl:
    """A capacitor bank / reactor switching its own steps to keep its node's
    voltage inside [band_min, band_max] (p.u.): a step of y_step_pu more
    shunt admittance when the voltage is on the side that step corrects, one
    step per round. `step` is where it starts; `base_step` the step already
    built into the network's own shunt admittance."""
    shunt_id: str
    node_id: str
    y_step_pu: complex
    n_steps: int
    step: int
    base_step: int
    band_min: float
    band_max: float


@dataclass
class DERControl:
    """One generator (DER) in the control loop, per unit on the system base,
    generator convention (p, q > 0 = injected).

    mode:
    - "fixed_pf": p = p_available, q = tan_phi * p;
    - "q_v": p = p_available, q follows the Q(V) droop qv_curve = (v1, v2,
      v3, v4): +q_max at or below v1, 0 between v2 and v3, -q_max at or
      above v4, linear in between;
    - "p_v": p = p_available scaled down linearly from 100 % at pv_curve[0]
      to 0 at pv_curve[1] (curtailment), q = tan_phi * p;
    - "voltage": p = p_available, q holds the node at v_set_pu (a PV bus)
      until it reaches +/-q_max, where the node becomes PQ.
    q is always kept within +/-q_max."""
    der_id: str
    node_id: str
    mode: str
    p_available: float
    q_max: float
    tan_phi: float = 0.0
    qv_curve: Tuple[float, float, float, float] = (0.95, 0.99, 1.01, 1.05)
    pv_curve: Tuple[float, float] = (1.06, 1.09)
    v_set_pu: float = 1.0

    def clip_q(self, q: float) -> float:
        return min(max(q, -self.q_max), self.q_max)

    def q_of_v(self, vm: float) -> Tuple[float, float]:
        """Q(V) target and the magnitude of its slope dQ/dV at vm."""
        v1, v2, v3, v4 = self.qv_curve
        if vm <= v1:
            return self.q_max, 0.0
        if vm < v2:
            k = self.q_max / (v2 - v1)
            return k * (v2 - vm), k
        if vm <= v3:
            return 0.0, 0.0
        if vm < v4:
            k = self.q_max / (v4 - v3)
            return -k * (vm - v3), k
        return -self.q_max, 0.0

    def p_of_v(self, vm: float) -> Tuple[float, float]:
        """P(V) target and the magnitude of its slope dP/dV at vm."""
        v_start, v_end = self.pv_curve
        if vm <= v_start:
            return self.p_available, 0.0
        if vm >= v_end:
            return 0.0, 0.0
        k = self.p_available / (v_end - v_start)
        return k * (v_end - vm), abs(k)


@dataclass
class ControlResult:
    v: np.ndarray
    converged: bool
    n_iter: int                         # sweep iterations over all rounds
    rounds: int                         # outer control rounds solved
    settled: bool                       # False if controls were still moving at the round limit
    taps: Dict[str, int]                # regulator connection_id -> final position
    der_p: Dict[str, float]             # der_id -> final p (p.u.)
    der_q: Dict[str, float]             # der_id -> final q (p.u.)
    der_at_q_limit: Dict[str, bool]
    shunt_steps: Optional[Dict[str, int]] = None  # shunt_id -> final step of voltage-controlled banks
    y_shunt_extra: Optional[np.ndarray] = None   # node shunt admittance on top of the network's, as solved


DER_TOL = 1e-6        # p.u. change in a DER's P/Q, or voltage error at a PV bus, counted as settled
DER_MAX_ROUNDS = 100
TAP_EXEMPTIONS = 2    # sign flips after a tap move a DER may take undamped, per solve


def _path_matrix(net: RadialNetwork, node_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Common-path resistance and reactance between the given nodes and the
    root: entry (i, j) sums the series impedance of the branches both paths
    share. To first order, |V_i| moves by sum_j (R_ij dP_j + X_ij dQ_j) for
    injections dP, dQ (ratios ignored)."""
    paths = []
    for i in node_idx:
        path = set()
        while net.parent[i] >= 0:
            path.add(i)
            i = int(net.parent[i])
        paths.append(path)
    n = len(node_idx)
    r, x = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            common = list(paths[a] & paths[b])
            z = net.z_pu[common].sum() if common else 0j
            r[a, b] = r[b, a] = z.real
            x[a, b] = x[b, a] = z.imag
    return r, x


def _update_ders(ders: List[DERControl], vm: np.ndarray, p: Dict[str, float], q: Dict[str, float],
                 r_path: np.ndarray, x_path: np.ndarray, damping: Dict[str, Tuple[float, float, int]],
                 taps_moved: bool = False) -> bool:
    """One joint Newton step for the controlled DER, from the voltages vm at
    their nodes. Unknowns: dq for Q(V) and voltage-controlling units, dp for
    P(V) units (whose q follows as tan_phi * p). With the linearised voltage
    change dV = sum_j (R_kj dp_j + X_kj dq_j):

    - Q(V) / P(V): dq_k + slope_k dV_k = gap_k, so the curve is met at the
      new voltage (the curve falls with V, slope_k = |dQ/dV| or |dP/dV|);
    - voltage control: dV_k = v_set_k - vm_k, unless the unit is at a
      reactive limit and the set-point asks for more (then it holds there).

    A piecewise-linear curve can make the full step jump back and forth
    across a kink, so a unit whose error changes sign from one round to the
    next has its step halved, growing back by 25 % per round while it doesn't
    (damping: der_id -> (factor, last error, exemptions left), kept across
    rounds). A sign change right after a regulator tap moved (taps_moved)
    usually comes from the tap, not from overshoot, so it isn't damped -- at
    most TAP_EXEMPTIONS times per unit, so a tap and a unit can't chase each
    other for ever. Updates p and q in place, within each unit's limits.
    Returns True if any unit was still off its control law by more than
    DER_TOL."""
    active, unit, slope, rhs = [], [], [], []
    for k, d in enumerate(ders):
        if d.mode == "q_v":
            target, s = d.q_of_v(vm[k])
            gap = target - q[d.der_id]
        elif d.mode == "p_v":
            target, s = d.p_of_v(vm[k])
            gap = target - p[d.der_id]
        elif d.mode == "voltage":
            gap = d.v_set_pu - vm[k]
            if (q[d.der_id] >= d.q_max - 1e-12 and gap > 0) or (q[d.der_id] <= -d.q_max + 1e-12 and gap < 0):
                continue                                    # PQ at its limit
            active.append(k), unit.append(0.0), slope.append(1.0), rhs.append(gap)
            continue
        else:
            continue
        active.append(k), unit.append(1.0), slope.append(s), rhs.append(gap)
    if not active or max(abs(g) for g in rhs) <= DER_TOL:
        return False

    sens = np.empty((len(active), len(active)))
    for b, j in enumerate(active):
        dj = ders[j]
        col = r_path[:, j] + dj.tan_phi * x_path[:, j] if dj.mode == "p_v" else x_path[:, j]
        sens[:, b] = col[active]
    m = np.diag(unit) + np.diag(slope) @ sens
    step = np.linalg.lstsq(m, np.array(rhs), rcond=None)[0]
    for k, u, gap in zip(active, step, rhs):
        d = ders[k]
        factor, last, exempt = damping.get(d.der_id, (1.0, 0.0, TAP_EXEMPTIONS))
        flipped = gap * last < 0
        if flipped and taps_moved and exempt > 0:
            exempt -= 1
        elif flipped:
            factor = max(factor / 2, 1 / 64)
        else:
            factor = min(1.0, factor * 1.25)
        damping[d.der_id] = (factor, gap, exempt)
        u *= factor
        if d.mode == "p_v":
            lo, hi = min(0.0, d.p_available), max(0.0, d.p_available)
            p[d.der_id] = min(max(p[d.der_id] + u, lo), hi)
            q[d.der_id] = d.clip_q(d.tan_phi * p[d.der_id])
        else:
            q[d.der_id] = d.clip_q(q[d.der_id] + u)
    return True


def solve_with_controls(
    net, s_pu: np.ndarray, v_root_pu: RootVoltage = 1.0 + 0j,
    regulators: Optional[List[RegulatorControl]] = None, ders: Optional[List[DERControl]] = None,
    load_model: Optional[LoadModel] = None, max_rounds: Optional[int] = None,
    shunts: Optional[List[ShuntControl]] = None, y_shunt_extra_pu: Optional[np.ndarray] = None,
) -> ControlResult:
    """Solve with the generators' injections, then update every control and
    solve again, until nothing moves:

    - regulators: each round moves a tap straight to the position that would
      reach the set-point if the voltage scaled with the ratio (no move while
      inside the deadband or at a tap limit);
    - DER: one Newton step for all of them together (_update_ders), using
      the R/X sensitivities between their nodes (the common-path impedance
      on a radial network, the impedance matrix on a meshed one), so units
      on the same feeder don't fight each other;
    - shunts (voltage-controlled capacitor banks / reactors): one step per
      round towards their band, when their node's voltage is outside it.

    y_shunt_extra_pu is node shunt admittance on top of the network's own,
    fixed for the solve (e.g. capacitor steps drawn for the sample); the
    voltage-controlled shunts add theirs to it. net is a RadialNetwork (sweep) or newton.MeshedNetwork (Newton-Raphson);
    both offer solve / controlled_node / sensitivity. max_rounds defaults to
    the largest regulator tap range plus 5, or DER_MAX_ROUNDS with DER
    present (shunts count like regulators, by their step range). With no
    controls this is one plain solve."""
    regulators, ders, shunts = regulators or [], ders or [], shunts or []
    index = {nid: i for i, nid in enumerate(net.node_ids)}
    taps = {r.connection_id: r.tap_pos for r in regulators}
    steps = {sh.shunt_id: sh.step for sh in shunts}
    shunt_idx = [index[sh.node_id] for sh in shunts]
    if max_rounds is None:
        max_rounds = max([r.tap_max - r.tap_min for r in regulators] + [sh.n_steps for sh in shunts],
                         default=0) + 5
        if ders:
            max_rounds = max(max_rounds, DER_MAX_ROUNDS)

    def shunt_extra():
        if not shunts:
            return y_shunt_extra_pu
        extra = np.zeros(len(net.node_ids), dtype=complex) if y_shunt_extra_pu is None else y_shunt_extra_pu.copy()
        for sh, i in zip(shunts, shunt_idx):
            extra[i] += (steps[sh.shunt_id] - sh.base_step) * sh.y_step_pu
        return extra

    p = {d.der_id: d.p_available for d in ders}
    q = {d.der_id: (d.clip_q(d.tan_phi * d.p_available) if d.mode in ("fixed_pf", "p_v") else 0.0) for d in ders}
    der_idx = [index[d.node_id] for d in ders]
    r_path, x_path = net.sensitivity(der_idx) if ders else (np.zeros((0, 0)), np.zeros((0, 0)))
    damping: Dict[str, Tuple[float, float, int]] = {}

    def at_limit(d):
        return abs(q[d.der_id]) >= d.q_max - 1e-12

    def result(v, converged, settled, extra):
        return ControlResult(v, converged, total_iter, rounds, settled, taps, p, q,
                             {d.der_id: at_limit(d) for d in ders}, dict(steps), extra)

    total_iter, rounds = 0, 0
    taps_moved_last = False
    v = np.zeros(len(net.node_ids), dtype=complex)
    extra = shunt_extra()
    for _ in range(max_rounds):
        rounds += 1
        current = {r.connection_id: 1.0 + taps[r.connection_id] * r.tap_step_pct / 100.0 for r in regulators}
        s_gen = None
        if ders:
            s_gen = np.zeros(len(net.node_ids), dtype=complex)
            for d, i in zip(ders, der_idx):
                s_gen[i] += complex(p[d.der_id], q[d.der_id])
        extra = shunt_extra()
        v, converged, n_iter = net.solve(s_pu, v_root_pu, ratios=current, load_model=load_model, s_gen_pu=s_gen,
                                         y_shunt_extra_pu=extra)
        total_iter += n_iter
        if not converged:
            return result(v, False, False, extra)

        moved = False
        for r in regulators:
            vm = abs(v[net.controlled_node(r.connection_id)])
            if abs(vm - r.v_set_pu) <= r.deadband_pu / 2:
                continue
            wanted = round(((current[r.connection_id] * r.v_set_pu / vm) - 1.0) * 100.0 / r.tap_step_pct)
            wanted = min(max(wanted, r.tap_min), r.tap_max)
            if wanted != taps[r.connection_id]:
                taps[r.connection_id] = wanted
                moved = True

        # A step adds susceptance of the sign that raises (capacitor) or lowers
        # (reactor) the voltage: step towards the band, one step per round.
        for sh, i in zip(shunts, shunt_idx):
            vm = abs(v[i])
            raises = 1 if sh.y_step_pu.imag > 0 else -1
            want = raises if vm < sh.band_min else (-raises if vm > sh.band_max else 0)
            new_step = min(max(steps[sh.shunt_id] + want, 0), sh.n_steps)
            if new_step != steps[sh.shunt_id]:
                steps[sh.shunt_id] = new_step
                moved = True

        # The voltages solved this round already reflect the taps moved last round.
        if ders and _update_ders(ders, np.abs(v[der_idx]), p, q, r_path, x_path, damping,
                                 taps_moved=taps_moved_last):
            taps_moved_last = moved
            moved = True
        else:
            taps_moved_last = moved

        if not moved:
            return result(v, True, True, extra)
    return result(v, True, False, extra)


def solve_with_regulators(
    net, s_pu: np.ndarray, regulators: List[RegulatorControl],
    v_root_pu: complex = 1.0 + 0j, max_rounds: Optional[int] = None,
    load_model: Optional[LoadModel] = None,
) -> Tuple[np.ndarray, bool, int, Dict[str, int], bool]:
    """solve_with_controls with regulators only. Returns (v_pu, converged,
    n_iter_total, tap_positions, settled)."""
    res = solve_with_controls(net, s_pu, v_root_pu, regulators=regulators,
                              load_model=load_model, max_rounds=max_rounds)
    return res.v, res.converged, res.n_iter, res.taps, res.settled


def branch_flows_pu(v_from: complex, v_to: complex, y: complex, y_shunt_half: complex = 0j,
                    ratio: float = 1.0) -> Tuple[complex, complex, complex, complex]:
    """Per-unit flows through one branch: series admittance y on the from
    side, ideal ratio n at the to end, y_shunt_half to ground at each end.
    Returns (i_from, i_to, s_from, s_to): current and power entering the
    branch at each end (s_to is negative when power is delivered there).
    The equivalent admittance matrix is Yff = y + y_sh, Yft = Ytf = -y/n,
    Ytt = y/n^2 + y_sh."""
    i_series = (v_from - v_to / ratio) * y
    i_from = i_series + v_from * y_shunt_half
    i_to = -i_series / ratio + v_to * y_shunt_half
    return i_from, i_to, v_from * np.conj(i_from), v_to * np.conj(i_to)


def lindistflow(
    net: RadialNetwork, s_pu: np.ndarray, v_root_pu: RootVoltage = 1.0 + 0j, ratio: Optional[np.ndarray] = None,
    y_shunt_extra_pu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """LinDistFlow approximation. s_pu is the demand as drawn (for ZIP loads,
    at the solved voltage). Shunts are taken at 1 p.u. (a capacitive branch
    end draws -B/2 of reactive power, a capacitor bank -B); a ratio branch scales the
    voltage squared by n^2. Returns (vm_pu, s_branch_pu): approximate voltage
    magnitudes, and the lossless power entering each node's branch at its
    parent end (0 at the root)."""
    ratio = net.ratio if ratio is None else ratio
    y_shunt = net.y_shunt_pu if y_shunt_extra_pu is None else net.y_shunt_pu + y_shunt_extra_pu
    s_node = s_pu + np.conj(y_shunt)               # S_shunt = |V|^2 conj(y) at |V| = 1
    s_branch = _subtree_sums(net, s_node, np.ones(len(net.node_ids)))   # ideal ratios pass power through
    w = np.abs(initial_voltage(net.node_ids, net.levels[0], v_root_pu)) ** 2
    for level in net.levels[1:]:
        z, sb = net.z_pu[level], s_branch[level]
        w[level] = ratio[level] ** 2 * (w[net.parent[level]] - 2 * (z.real * sb.real + z.imag * sb.imag))
    s_branch[net.levels[0]] = 0
    return np.sqrt(np.maximum(w, 0.0)), s_branch
