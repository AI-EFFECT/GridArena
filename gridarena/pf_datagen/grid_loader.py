"""MV-specific grid topology loader for the PF data-generation pipeline.

Mirrors gridarena/powerflow/get_grid_info.py's LV accessor pattern against the
MV tables ("MVNode"/"MVConnection"/"MVCable"/"MVTransformer"/"MVRegulator"/"MVShunt"/"MVDER")
instead of the LV ones, plus a substation-rooted connectivity helper used by
topology perturbation.

Besides the MV connections ("line" branches, or "regulated_line" when a step
voltage regulator sits on them), every MV/LV transformer becomes a
"transformer" branch from its connection-point node to a new LV-side bus
"<transformer_id>_LV", and the load assigned to that connection point moves
to the LV side (split by rating when a point has several transformers).
Capacitor banks and reactors ("MVShunt") are constant admittances at their
MV node; generators ("MVDER") are injections at theirs (pf_datagen/der.py).
"""

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

SYSTEM_FREQUENCY_HZ = 50.0

# Typical MV/LV transformer short-circuit impedance when imp_real/imp_imag
# aren't given: uk 4 % up to 630 kVA, 6 % above, with X/R = 5.
DEFAULT_UK_SMALL, DEFAULT_UK_LARGE, DEFAULT_UK_THRESHOLD_KVA = 0.04, 0.06, 630.0
DEFAULT_X_OVER_R = 5.0
DEFAULT_TAP_STEP_PCT, DEFAULT_TAP_MIN, DEFAULT_TAP_MAX = 2.5, -2, 2
DEFAULT_SECONDARY_V = 230.0
LV_BUS_SUFFIX = "_LV"


@dataclass
class MVBranch:
    connection_id: str
    from_node_id: str
    to_node_id: str
    cable_id: Optional[str]
    length_m: float
    r_ohm_per_km: float
    x_ohm_per_km: float
    nom_curr_a: Optional[float]            # line: cable rating; transformer: rated current on the MV side
    cap_uf_per_km: Optional[float] = None  # per-phase charging capacitance; None = not modelled
    emerg_curr_a: Optional[float] = None   # emergency rating; None = no emergency check
    kind: str = "line"                     # "line" | "regulated_line" | "transformer"
    z_ohm: Optional[complex] = None        # transformer: series impedance per phase, referred to the MV side
    rated_kva: Optional[float] = None      # transformer rating
    in_service: bool = True                # base status: False for an inactive connection or transformer,
                                           # or a normally-open switch
    is_switch: bool = False                # can be opened / closed (tie lines are normally-open switches)
    normally_open: bool = False
    available: bool = True                 # status active: a normally-open switch can then be closed
    tap_pos: Optional[int] = None          # transformer: fixed position; regulator: starting position
    tap_step_pct: Optional[float] = None
    tap_min: Optional[int] = None
    tap_max: Optional[int] = None
    reg_v_set_pu: Optional[float] = None   # regulator: target voltage at the to node
    reg_deadband_pu: Optional[float] = None
    regulator_id: Optional[str] = None

    @property
    def ratio(self) -> float:
        """Off-nominal voltage ratio n at the branch's to end (v_to = n * ...);
        a positive tap position raises the to-side voltage."""
        if self.tap_pos is None or not self.tap_step_pct:
            return 1.0
        return tap_ratio(self.tap_pos, self.tap_step_pct)


def tap_ratio(tap_pos: int, tap_step_pct: float) -> float:
    return 1.0 + tap_pos * tap_step_pct / 100.0


@dataclass
class MVShunt:
    """Capacitor bank (q_step_kvar > 0) or shunt reactor (< 0): `step` of its
    n_steps steps switched in, each q_step_kvar of three-phase reactive power
    at 1 p.u. voltage."""
    shunt_id: str
    node_id: str
    q_step_kvar: float
    n_steps: int
    step: int
    in_service: bool = True

    @property
    def q_var(self) -> float:
        """Reactive power it generates at 1 p.u. (VAr; negative for a reactor),
        0 when out of service."""
        return self.q_step_kvar * 1e3 * self.step if self.in_service else 0.0


@dataclass
class MVDER:
    """An MV-connected generator, battery or EV hub (see
    schemas/mv_grid_schema.py::DER and pf_datagen/der.py)."""
    der_id: str
    node_id: str
    der_type: str
    s_rated_kva: float
    p_max_kw: float
    profile: str                       # "solar" | "constant"
    profile_value: float
    control_mode: str                  # "fixed_pf" | "q_v" | "p_v" | "voltage"
    cos_phi: float = 1.0
    excitation: str = "under"
    q_limit_pu: float = 0.33
    qv_curve: Tuple[float, float, float, float] = (0.95, 0.99, 1.01, 1.05)
    pv_curve: Tuple[float, float] = (1.06, 1.09)
    v_set_pu: float = 1.02
    in_service: bool = True
    lat: Optional[float] = None        # the node's coordinates (solar profile)
    lon: Optional[float] = None


@dataclass
class MVGridTopology:
    mv_grid_id: str
    nominal_voltage_kv: float
    substation_node_ids: List[str]           # every substation is an ideal source (REF)
    node_id_to_index: Dict[str, int]
    branches: List[MVBranch]  # lines by connection id, then transformers; orientation/order for
                              # solving comes from pf_datagen/solver.py, not from this order
    node_vn_kv: Dict[str, float] = field(default_factory=dict)  # line-to-line base; missing = MV nominal
    load_transfer: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    # connection-point node -> [(LV bus, share)]: where its load is moved to
    shunts: List[MVShunt] = field(default_factory=list)
    ders: List[MVDER] = field(default_factory=list)
    node_coords: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    # MV node -> (lat, lon); LV buses have none

    def vn_kv(self, node_id: str) -> float:
        return self.node_vn_kv.get(node_id, self.nominal_voltage_kv)

    @property
    def outage_candidates(self) -> List[MVBranch]:
        """Branches topology perturbation may switch off by default: the MV
        connections in service in the base topology. Transformers join them
        with `outages.transformers`, and closing open switches is
        reconfiguration (pf_datagen/reconfiguration.py)."""
        return [b for b in self.branches if b.kind != "transformer" and b.in_service]

    @property
    def base_out_of_service(self) -> Set[str]:
        return {b.connection_id for b in self.branches if not b.in_service}


def transformer_impedance_pu(rated_kva: float, imp_real: Optional[float],
                             imp_imag: Optional[float]) -> complex:
    """Series impedance in p.u. on the transformer's own rating. A missing
    value is taken from the typical default (uk 4 % up to 630 kVA, 6 % above,
    X/R = 5)."""
    uk = DEFAULT_UK_SMALL if rated_kva <= DEFAULT_UK_THRESHOLD_KVA else DEFAULT_UK_LARGE
    r_default = uk / sqrt(1 + DEFAULT_X_OVER_R ** 2)
    r = r_default if imp_real is None else imp_real
    x = r_default * DEFAULT_X_OVER_R if imp_imag is None else imp_imag
    return complex(r, x)


def build_transformer_branch(
    transformer_id: str, cp_node_id: str, rated_kva: float, primary_kv: float,
    secondary_v: Optional[float], imp_real: Optional[float], imp_imag: Optional[float],
    status: Optional[str], tap_pos: Optional[int], tap_step_pct: Optional[float],
    tap_min: Optional[int], tap_max: Optional[int],
) -> Tuple[MVBranch, str, float]:
    """An MV/LV transformer as a branch from its connection-point node to its
    LV-side bus. Returns (branch, lv_bus_id, lv_vn_kv) -- lv_vn_kv is the
    line-to-line LV voltage, sqrt(3) x secondary_voltage_v (phase-to-neutral)."""
    lv_bus = f"{transformer_id}{LV_BUS_SUFFIX}"
    z_pu_own = transformer_impedance_pu(rated_kva, imp_real, imp_imag)
    z_ohm = z_pu_own * (primary_kv * 1e3) ** 2 / (rated_kva * 1e3)
    branch = MVBranch(
        connection_id=transformer_id, from_node_id=cp_node_id, to_node_id=lv_bus, cable_id=None,
        length_m=0.0, r_ohm_per_km=0.0, x_ohm_per_km=0.0,
        nom_curr_a=rated_kva * 1e3 / (sqrt(3) * primary_kv * 1e3),
        kind="transformer", z_ohm=z_ohm, rated_kva=rated_kva,
        in_service=(status or "active").lower() == "active",
        available=(status or "active").lower() == "active",
        tap_pos=0 if tap_pos is None else tap_pos,
        tap_step_pct=DEFAULT_TAP_STEP_PCT if tap_step_pct is None else tap_step_pct,
        tap_min=DEFAULT_TAP_MIN if tap_min is None else tap_min,
        tap_max=DEFAULT_TAP_MAX if tap_max is None else tap_max,
    )
    return branch, lv_bus, sqrt(3) * (secondary_v or DEFAULT_SECONDARY_V) / 1e3


def load_transfer_shares(transformers: List[MVBranch]) -> Dict[str, List[Tuple[str, float]]]:
    """Connection-point node -> [(LV bus, share of the point's load)], shares
    proportional to transformer rating."""
    by_cp: Dict[str, List[MVBranch]] = {}
    for t in transformers:
        by_cp.setdefault(t.from_node_id, []).append(t)
    out = {}
    for cp, ts in by_cp.items():
        total = sum(t.rated_kva for t in ts)
        out[cp] = [(t.to_node_id, t.rated_kva / total) for t in ts]
    return out


def get_mv_grid_topology(mv_grid_id: str, cursor) -> MVGridTopology:
    """Load an MV grid's nodes/branches from the database.

    Raises if the grid has no nodes, no connections, or no node with
    NodeType = 'substation'. Every substation is an ideal source at the
    run's set-point (a reference bus, the MV-side analogue of LV's "PT"
    node).
    """
    cursor.execute('SELECT nominal_voltage_kv FROM "MVGrid" WHERE mv_grid_id = %s', (mv_grid_id,))
    row = cursor.fetchone()
    if row is None:
        raise Exception(f"MV grid '{mv_grid_id}' not found.")
    nominal_voltage_kv = row[0]

    cursor.execute('SELECT "NodeId", "NodeType", "CoordLat", "CoordLon" FROM "MVNode" WHERE mv_grid_id = %s',
                   (mv_grid_id,))
    node_rows = cursor.fetchall()
    if not node_rows:
        raise Exception(f"No nodes found for MV grid '{mv_grid_id}'.")

    substations = sorted(nid for nid, ntype, *_ in node_rows if ntype == "substation")
    if not substations:
        raise Exception(
            f"MV grid '{mv_grid_id}' has no node with NodeType='substation' to act as the reference bus."
        )
    node_ids = substations + [nid for nid, *_ in node_rows if nid not in set(substations)]
    node_coords = {nid: (lat, lon) for nid, _, lat, lon in node_rows}

    cursor.execute(
        """
        SELECT c."ConnectionId", c."FromNodeId", c."ToNodeId", c."CableId", c."Length",
               cb."ImpReal", cb."ImpImag", cb."NomCurr", cb."CapUfPerKm", cb."EmergCurr",
               c."Status", c."IsSwitch", c."NormallyOpen"
        FROM "MVConnection" c
        JOIN "MVCable" cb ON cb."CableId" = c."CableId" AND cb.mv_grid_id = c.mv_grid_id
        WHERE c.mv_grid_id = %s
        ORDER BY c."ConnectionId"
        """,
        (mv_grid_id,),
    )
    conn_rows = cursor.fetchall()
    if not conn_rows:
        raise Exception(f"No connections found for MV grid '{mv_grid_id}'.")

    branches = [
        MVBranch(
            connection_id=cid, from_node_id=fn, to_node_id=tn, cable_id=cable_id,
            length_m=length, r_ohm_per_km=r, x_ohm_per_km=x, nom_curr_a=nom_curr,
            cap_uf_per_km=cap, emerg_curr_a=emerg,
            in_service=(status or "active").lower() == "active" and not normally_open,
            is_switch=bool(is_switch), normally_open=bool(normally_open),
            available=(status or "active").lower() == "active",
        )
        for cid, fn, tn, cable_id, length, r, x, nom_curr, cap, emerg, status, is_switch, normally_open
        in conn_rows
    ]

    cursor.execute(
        """
        SELECT "ConnectionId", "RegulatorId", "VSetPu", "DeadbandPu", "TapStepPct", "TapMin", "TapMax", "TapPos"
        FROM "MVRegulator" WHERE mv_grid_id = %s
        """,
        (mv_grid_id,),
    )
    regulators = {r[0]: r[1:] for r in cursor.fetchall()}
    for b in branches:
        if b.connection_id in regulators:
            reg_id, v_set, deadband, step, tmin, tmax, tpos = regulators[b.connection_id]
            b.kind, b.regulator_id = "regulated_line", reg_id
            b.reg_v_set_pu, b.reg_deadband_pu = v_set, deadband
            b.tap_step_pct, b.tap_min, b.tap_max, b.tap_pos = step, tmin, tmax, tpos

    cursor.execute(
        """
        SELECT t."TransformerId", cp."NodeId", t."RatedPowerKVA", t."PrimaryVoltageKV", t."SecondaryVoltageV",
               t."ImpReal", t."ImpImag", t."Status", t."TapPos", t."TapStepPct", t."TapMin", t."TapMax"
        FROM "MVTransformer" t
        JOIN "MVConnectionPoint" cp
            ON cp."ConnectionPointId" = t."ConnectionPointId" AND cp.mv_grid_id = t.mv_grid_id
        WHERE t.mv_grid_id = %s
        ORDER BY t."TransformerId"
        """,
        (mv_grid_id,),
    )
    taken = set(node_ids) | {b.connection_id for b in branches}
    node_vn_kv: Dict[str, float] = {}
    transformers = []
    for row in cursor.fetchall():
        branch, lv_bus, lv_kv = build_transformer_branch(*row)
        if branch.connection_id in taken or lv_bus in taken:
            raise Exception(
                f"Transformer '{branch.connection_id}' clashes with an existing node or connection id "
                f"(it is modelled as connection '{branch.connection_id}' to bus '{lv_bus}')."
            )
        taken |= {branch.connection_id, lv_bus}
        transformers.append(branch)
        node_ids.append(lv_bus)
        node_vn_kv[lv_bus] = lv_kv

    cursor.execute(
        """
        SELECT "ShuntId", "NodeId", "QStepKvar", "NSteps", "Step", "Status"
        FROM "MVShunt" WHERE mv_grid_id = %s ORDER BY "ShuntId"
        """,
        (mv_grid_id,),
    )
    shunts = [
        MVShunt(sid, nid, q_step, n_steps, step, in_service=(status or "active").lower() == "active")
        for sid, nid, q_step, n_steps, step, status in cursor.fetchall()
    ]

    cursor.execute(
        """
        SELECT d."DerId", d."NodeId", d."DerType", d."SRatedKva", d."PMaxKw", d."Profile", d."ProfileValue",
               d."ControlMode", d."CosPhi", d."Excitation", d."QLimitPu", d."QvV1", d."QvV2", d."QvV3", d."QvV4",
               d."PvVStart", d."PvVEnd", d."VSetPu", d."Status", n."CoordLat", n."CoordLon"
        FROM "MVDER" d
        JOIN "MVNode" n ON n."NodeId" = d."NodeId" AND n.mv_grid_id = d.mv_grid_id
        WHERE d.mv_grid_id = %s
        ORDER BY d."DerId"
        """,
        (mv_grid_id,),
    )
    ders = []
    for (der_id, nid, der_type, s_rated, p_max, profile, profile_value, mode, cos_phi, excitation, q_limit,
         v1, v2, v3, v4, pv_start, pv_end, v_set, status, lat, lon) in cursor.fetchall():
        if nid in substations:
            raise Exception(f"DER '{der_id}' sits on substation node '{nid}'; the busbar is an ideal source.")
        ders.append(MVDER(
            der_id, nid, der_type, s_rated, p_max, profile, profile_value, mode, cos_phi, excitation, q_limit,
            (v1, v2, v3, v4), (pv_start, pv_end), v_set,
            in_service=(status or "active").lower() == "active", lat=lat, lon=lon,
        ))

    return MVGridTopology(
        mv_grid_id=mv_grid_id,
        nominal_voltage_kv=nominal_voltage_kv,
        substation_node_ids=substations,
        node_id_to_index={nid: idx for idx, nid in enumerate(node_ids)},
        branches=branches + transformers,
        node_vn_kv=node_vn_kv,
        load_transfer=load_transfer_shares(transformers),
        shunts=shunts,
        ders=ders,
        node_coords=node_coords,
    )


def branch_admittance(branch: MVBranch, r_override: Optional[float] = None,
                       x_override: Optional[float] = None) -> complex:
    """Series Y = 1/Z for one branch, per phase, referred to the MV side.
    Lines: R/X are ohm/km and length is metres, Z = (R + jX) * L_km (same
    convention as powerflow/get_grid_info.py::get_grid_admitances).
    Transformers: their z_ohm (the R/X overrides don't apply)."""
    if branch.kind == "transformer":
        return 1 / branch.z_ohm
    r = branch.r_ohm_per_km if r_override is None else r_override
    x = branch.x_ohm_per_km if x_override is None else x_override
    z = complex(r, x) * (branch.length_m / 1000.0)
    if z == 0:
        raise Exception(f"Zero impedance for connection '{branch.connection_id}'.")
    return 1 / z


def branch_charging_susceptance(branch: MVBranch) -> float:
    """Total per-phase shunt susceptance B (S) of a branch from its cable's
    charging capacitance: B = 2*pi*f*C, C = cap_uf_per_km * 1e-6 * length_km.
    0 when the cable has no capacitance, and for transformers. The pi-model
    puts B/2 at each end."""
    if branch.kind == "transformer" or not branch.cap_uf_per_km:
        return 0.0
    return 2 * pi * SYSTEM_FREQUENCY_HZ * branch.cap_uf_per_km * 1e-6 * (branch.length_m / 1000.0)


def shunt_admittance_pu(shunts: List[MVShunt], s_base_va: float) -> Dict[str, complex]:
    """Node -> total admittance (p.u.) of its in-service shunts. A shunt
    generating Q at 1 p.u. is y = jQ/S_base (it draws |V|^2 conj(y) = -jQ|V|^2)."""
    out: Dict[str, complex] = {}
    for sh in shunts:
        if sh.q_var:
            out[sh.node_id] = out.get(sh.node_id, 0j) + 1j * sh.q_var / s_base_va
    return out


def substation_connected_component(
    node_ids: List[str], branches: List[MVBranch], active_connection_ids: Set[str],
    substation_node_ids: Iterable[str],
) -> Set[str]:
    """Return the set of node ids reachable from any substation using only
    the given active (in-service) branches. Used both to validate the base grid at
    load time and to determine which nodes survive a topology perturbation
    (see pf_datagen/perturbation.py for why disconnected nodes are dropped
    rather than the whole variant being rejected)."""
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for b in branches:
        if b.connection_id in active_connection_ids:
            G.add_edge(b.from_node_id, b.to_node_id)
    out: Set[str] = set()
    for sub in substation_node_ids:
        if sub in G and sub not in out:
            out |= nx.node_connected_component(G, sub)
    return out
