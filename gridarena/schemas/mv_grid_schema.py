"""Pydantic schemas for Medium Voltage grids, transformers, and LV grid abstractions."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MVNode(BaseModel):
    node_id: str
    coord_lat: Optional[float] = None
    coord_lon: Optional[float] = None
    node_type: Optional[str] = None


class MVCable(BaseModel):
    """Per-phase cable type. imp_real/imp_imag are ohm/km and nom_curr is the
    continuous current rating (A). cap_uf_per_km is the shunt (charging)
    capacitance per phase in uF/km at 50 Hz -- omit it to model the cable
    without charging. emerg_curr is the short-term emergency rating (A) used
    for post-outage checks -- omit it to skip that check."""
    cable_id: str
    imp_real: float
    imp_imag: float
    nom_curr: float
    cap_uf_per_km: Optional[float] = Field(default=None, ge=0)
    emerg_curr: Optional[float] = Field(default=None, gt=0)


class MVConnection(BaseModel):
    """A cable between two MV nodes (length in metres). status other than
    "active" takes it out of service. is_switch marks a connection that can
    be opened and closed; normally_open (switches only) means it is open in
    the base topology -- how a tie line between feeders is described."""
    connection_id: str
    from_node_id: str
    to_node_id: str
    cable_id: str
    length: float
    status: Optional[str] = "active"
    is_switch: bool = False
    normally_open: bool = False

    @model_validator(mode="after")
    def _open_only_if_switch(self):
        if self.normally_open and not self.is_switch:
            raise ValueError(f"Connection '{self.connection_id}': normally_open needs is_switch.")
        return self


class MVConnectionPoint(BaseModel):
    connection_point_id: str
    node_id: str
    name: Optional[str] = None


TAP_STEP_PCT_DEFAULT = 2.5
TAP_MIN_DEFAULT = -2
TAP_MAX_DEFAULT = 2


class Transformer(BaseModel):
    """MV/LV transformer. imp_real/imp_imag are its series resistance and
    reactance in p.u. on its own rating (rated_power_kva, primary voltage) --
    empty means typical values are assumed (see pf_datagen/grid_loader.py).
    secondary_voltage_v is the LV phase-to-neutral voltage.

    Fixed off-load tap: tap_pos steps of tap_step_pct, within
    [tap_min, tap_max]; a positive position raises the LV-side voltage.
    Empty fields mean position 0, 2.5 % steps, -2..+2."""
    transformer_id: str
    connection_point_id: str
    lv_grid_id: Optional[str] = None
    rated_power_kva: float
    primary_voltage_kv: float
    secondary_voltage_v: float = 230.0
    imp_real: Optional[float] = None
    imp_imag: Optional[float] = None
    name: Optional[str] = None
    status: Optional[str] = "active"
    tap_pos: Optional[int] = None
    tap_step_pct: Optional[float] = Field(default=None, gt=0, le=10)
    tap_min: Optional[int] = None
    tap_max: Optional[int] = None

    @model_validator(mode="after")
    def _tap_in_range(self):
        lo = TAP_MIN_DEFAULT if self.tap_min is None else self.tap_min
        hi = TAP_MAX_DEFAULT if self.tap_max is None else self.tap_max
        pos = 0 if self.tap_pos is None else self.tap_pos
        if lo > hi or not lo <= pos <= hi:
            raise ValueError(
                f"Transformer '{self.transformer_id}': tap_pos {pos} must lie within [tap_min {lo}, tap_max {hi}]."
            )
        return self


class Regulator(BaseModel):
    """Step voltage regulator on an MV connection. It sits at the connection's
    to end and steps its tap (tap_step_pct per position, within
    [tap_min, tap_max]; positive raises the to-side voltage) to keep the
    to-node voltage within v_set_pu +/- deadband_pu / 2. tap_pos is the
    starting position. Defaults match a common +/-10 %, 32-step regulator."""
    regulator_id: str
    connection_id: str
    v_set_pu: float = Field(default=1.0, gt=0.8, lt=1.2)
    deadband_pu: float = Field(default=0.02, gt=0, lt=0.2)
    tap_step_pct: float = Field(default=0.625, gt=0, le=10)
    tap_min: int = -16
    tap_max: int = 16
    tap_pos: int = 0

    @model_validator(mode="after")
    def _tap_in_range(self):
        if self.tap_min > self.tap_max or not self.tap_min <= self.tap_pos <= self.tap_max:
            raise ValueError(
                f"Regulator '{self.regulator_id}': tap_pos {self.tap_pos} must lie within "
                f"[tap_min {self.tap_min}, tap_max {self.tap_max}]."
            )
        return self


class Shunt(BaseModel):
    """Capacitor bank (q_step_kvar > 0) or shunt reactor (q_step_kvar < 0) at
    an MV node: n_steps switchable steps of q_step_kvar each (three-phase
    reactive power at nominal voltage), `step` of them switched in. Modelled
    as a constant admittance, so its reactive power scales with V^2."""
    shunt_id: str
    node_id: str
    q_step_kvar: float
    n_steps: int = Field(default=1, ge=1)
    step: int = Field(default=1, ge=0)
    status: Optional[str] = "active"
    name: Optional[str] = None

    @model_validator(mode="after")
    def _valid(self):
        if self.q_step_kvar == 0:
            raise ValueError(f"Shunt '{self.shunt_id}': q_step_kvar must be non-zero.")
        if self.step > self.n_steps:
            raise ValueError(f"Shunt '{self.shunt_id}': step {self.step} exceeds n_steps {self.n_steps}.")
        return self


class DER(BaseModel):
    """An MV-connected generator, battery or EV hub at an MV node (not the
    substation). Generator convention: positive P/Q = injected.

    Available power comes from `profile`: "solar" (clear-sky PV output at the
    node's coordinates times a daily clearness factor; the default for
    type "pv") or "constant" (`profile_value` x p_max_kw in every sample,
    a share in [-1, 1]; negative consumes). p_max_kw defaults to s_rated_kva.

    control_mode:
    - "fixed_pf": constant cos_phi; `excitation` "under" absorbs reactive
      power (the usual setting against voltage rise), "over" injects it;
    - "q_v": Q(V) droop through qv_curve = [v1, v2, v3, v4] (p.u.): full
      capacitive Q at or below v1, none between v2 and v3, full inductive Q
      at or above v4 (defaults follow VDE-AR-N 4110);
    - "p_v": active power curtailed linearly from 100 % at pv_v_start to 0
      at pv_v_end, with cos_phi as in fixed_pf;
    - "voltage": holds its node at v_set_pu (a PV bus) within its reactive
      limit, then holds the limit.
    Outside fixed_pf, |Q| is limited to q_limit_pu x s_rated_kva (default
    0.33, cos phi 0.95 at full output); always within the rating circle."""
    der_id: str
    node_id: str
    type: Literal["pv", "wind", "chp", "battery", "ev_hub", "other"]
    s_rated_kva: float = Field(gt=0)
    p_max_kw: Optional[float] = Field(default=None, gt=0)
    profile: Optional[Literal["solar", "constant"]] = None
    profile_value: float = Field(default=1.0, ge=-1, le=1)
    control_mode: Literal["fixed_pf", "q_v", "p_v", "voltage"] = "fixed_pf"
    cos_phi: float = Field(default=1.0, gt=0, le=1)
    excitation: Literal["under", "over"] = "under"
    q_limit_pu: float = Field(default=0.33, ge=0, le=1)
    qv_curve: List[float] = Field(default_factory=lambda: [0.95, 0.99, 1.01, 1.05], min_length=4, max_length=4)
    pv_v_start: float = Field(default=1.06, gt=0.8, lt=1.3)
    pv_v_end: float = Field(default=1.09, gt=0.8, lt=1.3)
    v_set_pu: float = Field(default=1.02, ge=0.9, le=1.1)
    status: Optional[str] = "active"
    name: Optional[str] = None

    @model_validator(mode="after")
    def _resolve_and_check(self):
        if self.profile is None:
            self.profile = "solar" if self.type == "pv" else "constant"
        if self.p_max_kw is None:
            self.p_max_kw = self.s_rated_kva
        v = self.qv_curve
        if not (0.8 < v[0] < v[1] <= v[2] < v[3] < 1.3):
            raise ValueError(f"DER '{self.der_id}': qv_curve must satisfy 0.8 < v1 < v2 <= v3 < v4 < 1.3.")
        if self.pv_v_start >= self.pv_v_end:
            raise ValueError(f"DER '{self.der_id}': pv_v_start must be below pv_v_end.")
        return self


class MVGrid(BaseModel):
    mv_grid_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    nominal_voltage_kv: float
    region: Optional[str] = None
    nodes: List[MVNode]
    cables: Optional[List[MVCable]] = None
    connections: Optional[List[MVConnection]] = None
    connection_points: Optional[List[MVConnectionPoint]] = None
    transformers: Optional[List[Transformer]] = None
    regulators: Optional[List[Regulator]] = None
    shunts: Optional[List[Shunt]] = None
    ders: Optional[List[DER]] = None


# ── Configuration edits (PATCH /mv_grid/{id}/configuration) ──
# Each edit names one element and only the operating settings that change;
# omitted fields keep their stored value. The merged element is checked
# against the full schema above before anything is written.

_Status = Literal["active", "inactive"]


class _Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConnectionEdit(_Edit):
    connection_id: str
    status: Optional[_Status] = None
    normally_open: Optional[bool] = None


class TransformerEdit(_Edit):
    transformer_id: str
    status: Optional[_Status] = None
    tap_pos: Optional[int] = None


class RegulatorEdit(_Edit):
    regulator_id: str
    v_set_pu: Optional[float] = None
    deadband_pu: Optional[float] = None
    tap_pos: Optional[int] = None


class ShuntEdit(_Edit):
    shunt_id: str
    step: Optional[int] = None
    status: Optional[_Status] = None


class DEREdit(_Edit):
    der_id: str
    status: Optional[_Status] = None
    control_mode: Optional[Literal["fixed_pf", "q_v", "p_v", "voltage"]] = None
    cos_phi: Optional[float] = None
    excitation: Optional[Literal["under", "over"]] = None
    profile_value: Optional[float] = None
    v_set_pu: Optional[float] = None
    q_limit_pu: Optional[float] = None


class MVConfigurationUpdate(_Edit):
    connections: List[ConnectionEdit] = []
    transformers: List[TransformerEdit] = []
    regulators: List[RegulatorEdit] = []
    shunts: List[ShuntEdit] = []
    ders: List[DEREdit] = []


class MVTieLine(BaseModel):
    """A normally-open tie to add between two MV nodes (length in metres),
    e.g. one of GET /mv-grid/{id}/tie-suggestions."""
    connection_id: str = Field(min_length=1)
    from_node_id: str
    to_node_id: str
    cable_id: str
    length: float = Field(gt=0)


class MVTiesRequest(BaseModel):
    ties: List[MVTieLine] = Field(min_length=1)


class SyntheticGridRequest(BaseModel):
    """Generate synthetic MV grids (gridarena/synthetic/): `count` grids of an
    archetype, calibrated on the stored grids ('all': archetypes in their
    stored proportions)."""
    archetype: Literal["urban", "mixed", "rural", "all"] = "all"
    count: int = Field(default=5, ge=1, le=100)
    seed: Optional[int] = Field(default=None, description="Omit for a fresh seed (returned).")
    feeders_per_substation: int = Field(default=1, ge=1, le=4)
    second_substation: bool = Field(default=False, description="A second substation, joined by a normally-open tie.")
    ties_between_feeders: int = Field(default=0, ge=0, le=3)
    n_nodes_min: Optional[int] = Field(default=None, ge=5, le=500)
    n_nodes_max: Optional[int] = Field(default=None, ge=5, le=500)
    reference_lat: float = Field(default=48.85, ge=-80, le=80)
    reference_lon: float = Field(default=2.2, ge=-180, le=180)

    @model_validator(mode="after")
    def _node_range(self):
        if (self.n_nodes_min is None) != (self.n_nodes_max is None):
            raise ValueError("Give both n_nodes_min and n_nodes_max, or neither.")
        if self.n_nodes_min is not None and self.n_nodes_min > self.n_nodes_max:
            raise ValueError("n_nodes_min must not exceed n_nodes_max.")
        return self


class ConnectLVRequest(BaseModel):
    lv_grid_id: str
    transformer_id: Optional[str] = None
    rated_power_kva: Optional[float] = None
    primary_voltage_kv: Optional[float] = None
    secondary_voltage_v: Optional[float] = 230.0
    imp_real: Optional[float] = None
    imp_imag: Optional[float] = None
    name: Optional[str] = None


class LVAbstraction(BaseModel):
    lv_grid_id: str
    display_name: Optional[str] = None
    connection_point_id: str
    transformer_id: str
    n_nodes: int = 0
    pv_pairs: Optional[List[dict]] = None
