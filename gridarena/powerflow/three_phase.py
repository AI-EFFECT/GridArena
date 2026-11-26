"""Three-phase four-wire power flow for radial LV grids.

Each node can draw a different power on each phase (R, S, T), each cable has
its own per-phase impedance, and the substation (root) imposes its measured
phase-to-neutral voltages. Loads are connected phase-to-neutral, so an
unbalanced load sends a return current through the neutral conductor, whose
voltage drop shifts the neutral potential along the feeder and with it every
customer's phase-to-neutral voltage. The neutral is grounded at the root
only and has impedance ``neutral_ratio`` x the mean phase impedance of each
branch; with ``neutral_ratio = 0`` (an ideal neutral) the three phases
decouple into independent single-phase problems.

There are no mutual (phase-to-phase) impedances: the grid data only holds
per-phase self impedances.

Method: fixed-point iteration on the load currents (the "BIBC/BCBV" form of the
backward/forward sweep). For a radial grid rooted at ``root`` and a branch
indexed by its downstream node c, the path matrix P[k, c] = 1 when branch c
lies on the path from the root to node k. Branch currents are I = P^T J (J =
load currents) and the voltage drop from the root to every node is
P diag(Z) P^T J, so each iteration is a few matrix products, done for every
timestamp at once.

Conventions (as in the Measurements table): powers in W / VAr, load convention
(positive = consumption, negative = generation); voltages in V; angles in
degrees only at the edges (inputs and outputs here are complex).
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

PHASES = ("R", "S", "T")
NOMINAL_ANGLES_DEG = (0.0, -120.0, 120.0)


class NotRadialError(ValueError):
    """The connections contain a loop."""


@dataclass
class RadialGrid:
    """A radial grid prepared for solving: only the nodes reachable from the root."""
    nodes: np.ndarray        # indices (into the caller's node numbering) of the reachable nodes
    unreachable: list        # indices of nodes not connected to the root
    d_phase: np.ndarray      # (3, n, n) complex: voltage-drop matrices P diag(Z_p) P^T
    d_neutral: np.ndarray    # (n, n) complex: same for the neutral conductor


def build_radial_grid(n_nodes, connections, z_phase, neutral_ratio=1.0, root=0) -> RadialGrid:
    """Orient the connections away from ``root`` and build the drop matrices.

    connections: (from, to) node index pairs, in any order and direction.
    z_phase: (n_connections, 3) complex series impedance per phase, in ohm.
    Parallel connections between the same two nodes are combined. Raises
    NotRadialError if the connections contain a loop.
    """
    z_phase = np.asarray(z_phase, dtype=complex).reshape(-1, 3)

    # Combine parallel connections: Z = 1 / sum(1 / Z_i), per phase.
    combined = {}
    for (a, b), z in zip(connections, z_phase):
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in combined:
            prev = combined[key]
            with np.errstate(divide="ignore", invalid="ignore"):
                combined[key] = np.where((prev == 0) | (z == 0), 0, prev * z / (prev + z))
        else:
            combined[key] = z

    neighbours = defaultdict(list)
    for (a, b), z in combined.items():
        neighbours[a].append((b, z))
        neighbours[b].append((a, z))

    # Breadth-first from the root: parent and branch impedance of every reached node.
    parent = {root: None}
    branch_z = {}
    order = [root]
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for other, z in neighbours[node]:
            if other == parent[node]:
                continue
            if other in parent:
                raise NotRadialError(
                    f"The grid has a loop (nodes {node} and {other} are both reached from the root); "
                    "the power flow needs a radial grid."
                )
            parent[other] = node
            branch_z[other] = z
            order.append(other)
            queue.append(other)

    nodes = np.array(order)
    position = {node: i for i, node in enumerate(order)}
    n = len(order)

    # P[k, c] = 1 when branch c (into node c) is on the root -> k path.
    path = np.zeros((n, n))
    for k, node in enumerate(order):
        cursor = node
        while parent[cursor] is not None:
            path[k, position[cursor]] = 1.0
            cursor = parent[cursor]

    z = np.zeros((n, 3), dtype=complex)
    for node, zb in branch_z.items():
        z[position[node]] = zb
    z_n = neutral_ratio * z.mean(axis=1)

    d_phase = np.stack([(path * z[:, p]) @ path.T for p in range(3)])
    d_neutral = (path * z_n) @ path.T
    unreachable = sorted(set(range(n_nodes)) - set(order))
    return RadialGrid(nodes=nodes, unreachable=unreachable, d_phase=d_phase, d_neutral=d_neutral)


@dataclass
class ThreePhaseResult:
    v_phase: np.ndarray      # (T, n_nodes, 3) complex phase-to-neutral voltage; NaN where unreachable
    v_neutral: np.ndarray    # (T, n_nodes) complex neutral-to-ground voltage; NaN where unreachable
    converged: np.ndarray    # (T,) bool
    iterations: np.ndarray   # (T,) int


def solve(grid: RadialGrid, s_load, v_root, n_nodes=None, tol=1e-6, max_iter=200) -> ThreePhaseResult:
    """Solve the power flow for one or many timestamps.

    s_load: (T, n_nodes, 3) complex power drawn per node and phase (VA, load
            convention), in the caller's node numbering. The root's own load is
            ignored (it is the source).
    v_root: (T, 3) complex phase-to-neutral voltage imposed at the root.
    tol:    convergence threshold on the largest voltage change between two
            iterations, in V.
    """
    s_load = np.asarray(s_load, dtype=complex)
    v_root = np.asarray(v_root, dtype=complex)
    n_steps = s_load.shape[0]
    n_nodes = s_load.shape[1] if n_nodes is None else n_nodes

    s = s_load[:, grid.nodes, :]                       # reachable nodes, root first
    u = np.repeat(v_root[:, None, :], len(grid.nodes), axis=1)
    converged = np.zeros(n_steps, dtype=bool)
    iterations = np.zeros(n_steps, dtype=int)

    for it in range(1, max_iter + 1):
        active = ~converged
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(s[active] == 0, 0, np.conj(s[active] / u[active]))
        drop = np.stack([j[:, :, p] @ grid.d_phase[p].T for p in range(3)], axis=-1)
        v_neutral = j.sum(axis=-1) @ grid.d_neutral.T
        u_new = v_root[active][:, None, :] - drop - v_neutral[:, :, None]

        change = np.abs(u_new - u[active]).max(axis=(1, 2))
        u[active] = u_new
        iterations[active] = it
        done = np.flatnonzero(active)[change < tol]
        converged[done] = True
        if converged.all():
            break

    # Neutral voltage of the final state.
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(s == 0, 0, np.conj(s / u))
    v_neutral = j.sum(axis=-1) @ grid.d_neutral.T

    v_phase_full = np.full((n_steps, n_nodes, 3), np.nan, dtype=complex)
    v_neutral_full = np.full((n_steps, n_nodes), np.nan, dtype=complex)
    v_phase_full[:, grid.nodes, :] = u
    v_neutral_full[:, grid.nodes] = v_neutral
    return ThreePhaseResult(v_phase_full, v_neutral_full, converged, iterations)


def balanced_inputs(s_load, v_root):
    """Turn per-phase inputs into their balanced equivalent: every node's total
    power split equally over the three phases, and a symmetric root voltage
    (mean magnitude of the root phases, R-phase angle, 120 degrees apart)."""
    s_load = np.asarray(s_load, dtype=complex)
    v_root = np.asarray(v_root, dtype=complex)
    s_bal = np.repeat(s_load.sum(axis=-1, keepdims=True) / 3.0, 3, axis=-1)
    mag = np.abs(v_root).mean(axis=-1, keepdims=True)
    angle_r = np.angle(v_root[:, :1])
    v_bal = mag * np.exp(1j * (angle_r + np.radians(NOMINAL_ANGLES_DEG)))
    return s_bal, v_bal


def solve_single_phase(grid: RadialGrid, s_phase, phase, v_ref, tol=1e-6, max_iter=200):
    """Solve one snapshot on one phase, the others unloaded.

    Meant for a grid built with ``neutral_ratio=0``, where the phases are
    independent, so this is the classic per-phase power flow with that phase's
    own cable impedances. s_phase: (n_nodes,) complex power drawn (VA, load
    convention); v_ref: reference voltage magnitude at the root. Returns the
    (n_nodes,) complex voltages (NaN where unreachable) and whether it converged.
    """
    k = PHASES.index(phase)
    s_phase = np.asarray(s_phase, dtype=complex)
    s = np.zeros((1, len(s_phase), 3), dtype=complex)
    s[0, :, k] = s_phase
    v_root = v_ref * np.exp(1j * np.radians(NOMINAL_ANGLES_DEG))[None, :]
    result = solve(grid, s, v_root, tol=tol, max_iter=max_iter)
    return result.v_phase[0, :, k], bool(result.converged[0])
