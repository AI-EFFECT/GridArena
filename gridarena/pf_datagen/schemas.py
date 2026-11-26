"""Request/response schemas for the MV power-flow training-data generation pipeline."""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, model_validator


class VoltageLimits(BaseModel):
    """Bus voltage band (p.u. of the line-to-line nominal voltage) each solved
    bus is checked against. Defaults are EN 50160's +/-10 %, which Portugal
    applies to AT, MT and BT (ERSE Regulamento da Qualidade de Servico,
    Nov 2024, art. 25.o n.o 3: NP EN 50160; only MAT has its own +/-5 % band,
    in the MPQS). DSOs often plan MV tighter (e.g. +/-5 %)."""
    vm_min: float = Field(default=0.90, gt=0, le=1)
    vm_max: float = Field(default=1.10, ge=1, lt=2)

    @model_validator(mode="after")
    def _ordered(self):
        if self.vm_min >= self.vm_max:
            raise ValueError("voltage_limits.vm_min must be below vm_max.")
        return self


class SlackVoltageRange(BaseModel):
    """Substation busbar set-point drawn uniformly per sample from [min, max] (p.u.)."""
    min: float = Field(gt=0.8, lt=1.2)
    max: float = Field(gt=0.8, lt=1.2)

    @model_validator(mode="after")
    def _ordered(self):
        if self.min >= self.max:
            raise ValueError("slack_voltage_pu.min must be below max.")
        return self


class ZIPFractions(BaseModel):
    """Shares of a load's P (or Q) that behave as constant impedance (z),
    constant current (i) and constant power (p); they must sum to 1."""
    z: float = Field(default=0.0, ge=0, le=1)
    i: float = Field(default=0.0, ge=0, le=1)
    p: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self):
        if abs(self.z + self.i + self.p - 1.0) > 1e-6:
            raise ValueError("ZIP fractions z + i + p must sum to 1.")
        return self

    def as_tuple(self):
        return (self.z, self.i, self.p)


class LoadModel(BaseModel):
    """Voltage dependence of every load in the run, separately for P and Q.
    The historical P/Q is the demand at 1 p.u.; the default is constant power."""
    p: ZIPFractions = ZIPFractions()
    q: ZIPFractions = ZIPFractions()


class ClearnessRange(BaseModel):
    """Range the daily clearness factor of the PV profile is drawn from
    (1 = clear sky); min == max gives a fixed factor."""
    min: float = Field(default=0.3, ge=0, le=1)
    max: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self):
        if self.min > self.max:
            raise ValueError("pv_clearness.min must not exceed max.")
        return self


class TopologyPerturbationConfig(BaseModel):
    type: Literal["none", "n_minus_k", "random"] = "none"
    k: int = Field(
        default=1, ge=1,
        description="Max simultaneous branch outages. No fixed upper bound here -- "
                    "for type == 'n_minus_k' the enumeration itself is capped by "
                    "perturbation.MAX_N_MINUS_K_COMBOS, which raises a clear error "
                    "if the requested k/branch-count combination is infeasible.",
    )
    n_variants: int = Field(default=10, ge=1, le=2000, description="Used only when type == 'random'.")


class AdmittancePerturbationConfig(BaseModel):
    enabled: bool = False
    sigma: float = Field(default=0.2, gt=0, lt=1, description="Relative R/X perturbation, e.g. 0.2 = +/-20%.")


class ReconfigurationConfig(BaseModel):
    """Random radial configurations (pf_datagen/reconfiguration.py): each
    closes a different spanning set of switchable connections, drawn
    uniformly. The base configuration is always variant 0's."""
    enabled: bool = False
    n_configs: int = Field(default=10, ge=1, le=1000,
                           description="Configurations drawn besides the base one; all distinct.")
    switchable: Literal["marked", "all_lines"] = Field(
        default="marked",
        description="'marked': only switches and normally-open ties. 'all_lines': every MV line can open, "
                    "as if each section had a disconnector.",
    )
    allow_meshed: bool = Field(default=False,
                               description="Close extra_closed more open switches on top (needs solver 'newton').")
    extra_closed: int = Field(default=1, ge=1, le=20)


class OutagesConfig(BaseModel):
    """Equipment outages. `transformers` makes MV/LV transformers outage
    candidates of topology_perturbation; the probabilities are per unit and
    per sample (pf_datagen/outages.py)."""
    transformers: bool = False
    der_trip_p: float = Field(default=0.0, ge=0, le=1)
    shunt_out_p: float = Field(default=0.0, ge=0, le=1)
    regulator_bypass_p: float = Field(default=0.0, ge=0, le=1)


class TemperatureConfig(BaseModel):
    """Conductor temperature (degC): one per sample in [min, max], plus a per-cable offset within +/- spread."""
    min: float = Field(default=20.0, ge=-30, le=120)
    max: float = Field(default=70.0, ge=-30, le=120)
    spread: float = Field(default=5.0, ge=0, le=30)

    @model_validator(mode="after")
    def _ordered(self):
        if self.min > self.max:
            raise ValueError("parameters.temperature.min must not exceed max.")
        return self


class CatalogueCable(BaseModel):
    """A cable type a swap may pick (per phase, per km)."""
    cable_id: str = Field(min_length=1)
    r_ohm_per_km: float = Field(gt=0)
    x_ohm_per_km: float = Field(ge=0)
    nom_curr_a: float = Field(gt=0)
    cap_uf_per_km: Optional[float] = Field(default=None, ge=0)
    emerg_curr_a: Optional[float] = Field(default=None, gt=0)


class ParametersConfig(BaseModel):
    """Branch parameter uncertainty (pf_datagen/parameters.py). 0 / null
    turns an option off; all are off by default."""
    length_sigma: float = Field(default=0.0, ge=0, lt=0.5,
                                description="Line length (so R, X and charging) x 1 +/- sigma; e.g. 0.05.")
    temperature: Optional[TemperatureConfig] = Field(
        default=None, description="Conductor temperature per sample; changes R only. Null = 20 degC (stored R).")
    swap_p: float = Field(default=0.0, ge=0, le=1,
                          description="Probability a line's cable type is replaced within its family; e.g. 0.1.")
    transformer_uk_sigma: float = Field(default=0.0, ge=0, lt=0.5,
                                        description="MV/LV transformer impedance magnitude x 1 +/- sigma; e.g. 0.1.")
    transformer_xr_sigma: float = Field(default=0.0, ge=0, lt=0.9,
                                        description="MV/LV transformer X/R x 1 +/- sigma; e.g. 0.2.")
    draw: Literal["per_variant", "per_sample"] = Field(
        default="per_variant",
        description="When length, swaps and transformers are drawn. Temperature is always per sample.",
    )
    catalogue: List[CatalogueCable] = Field(
        default_factory=list,
        description="Cable types added to the database's for swaps (deduplicated by value).",
    )


class DesignConfig(BaseModel):
    """How the per-sample axes pair up (pf_datagen/design.py): `independent`
    (each axis stratified on its own), `lhs` (one Latin hypercube, low
    correlation between axes) or `factorial` (every combination of bins filled
    equally)."""
    mode: Literal["independent", "lhs", "factorial"] = "independent"


class TargetMixConfig(BaseModel):
    """A minimum share of samples with a violation (pf_datagen/target_mix.py):
    a pilot measures the violation rate per stratum, then the rest of the
    samples oversample the strata that produce them. Null = off."""
    min_violation_share: Optional[float] = Field(default=None, gt=0, lt=1)
    violations: Literal["any", "voltage", "thermal"] = "any"
    pilot_share: float = Field(default=0.2, ge=0.05, le=0.5)


NoiseProfile = Literal["clean", "easy", "medium", "hard"]


class MeasurementsConfig(BaseModel):
    """Measurement view (pf_datagen/measurements.py): next to each clean
    solution, the readings of a meter layout drawn once per run. Profiles are
    benchmarks/general_noise's (relative noise and rare outliers)."""
    enabled: bool = False
    cp_coverage: float = Field(default=0.30, ge=0, le=1,
                               description="Share of connection points metered (V, and P, Q through the transformer).")
    cable_coverage: float = Field(default=0.10, ge=0, le=1,
                                  description="Share of the lines other than switches and feeder heads metered "
                                              "(P, Q, I); every switch is.")
    der_min_mva: float = Field(default=1.0, ge=0, description="DER of at least this rating are metered (P, Q, V).")
    profile: NoiseProfile = "medium"
    substation_profile: NoiseProfile = Field(
        default="easy", description="Noise of the substation busbar voltage and the feeder heads.")
    missing_p: float = Field(default=0.02, ge=0, lt=1, description="Probability a reading is missing, per sample.")
    pseudo_sigma: float = Field(default=0.30, ge=0, le=2,
                                description="Relative error of the P, Q pseudo-measurements at unmetered "
                                            "connection points.")


LOAD_LEVEL_DEFAULT_RANGE = {"loadability": (0.1, 1.2), "multiplier": (0.5, 1.5)}


class LoadLevelConfig(BaseModel):
    """Load level per sample (pf_datagen/load_level.py). `multiplier`: every
    load x a factor drawn from [min, max]. `loadability`: each sample rescaled
    so its gross demand is a drawn fraction of P_max, the most demand the grid
    carries within its limits. Empty min/max take the mode's default range."""
    mode: Literal["off", "multiplier", "loadability"] = "off"
    min: Optional[float] = Field(default=None, ge=0, le=10)
    max: Optional[float] = Field(default=None, gt=0, le=10)
    sampling: Literal["uniform", "stratified"] = "stratified"
    bins: int = Field(default=10, ge=1, le=1000)

    @model_validator(mode="after")
    def _range(self):
        if self.mode != "off":
            lo, hi = LOAD_LEVEL_DEFAULT_RANGE[self.mode]
            self.min = lo if self.min is None else self.min
            self.max = hi if self.max is None else self.max
            if self.min > self.max:
                raise ValueError("load_level.min must not exceed max.")
        return self


class FeederBand(BaseModel):
    """How many Feeder series a connection point sums, by its transformer
    rating: a point of up to `max_kva` (and above the previous band's) draws
    its count uniformly from [min, max]."""
    max_kva: float = Field(gt=0)
    min: int = Field(ge=1, le=1000)
    max: int = Field(ge=1, le=1000)

    @model_validator(mode="after")
    def _range(self):
        if self.min > self.max:
            raise ValueError("feeder_bands: min must not exceed max.")
        return self


# Feeders per point by transformer kVA (load_shape.feeders_per_point `by_kva`). Sized on
# feeder_measurements, whose LV feeders peak at about 40-120 kW: a 630 kVA point at 0.8 peak
# utilisation (~500 kW) takes 4-8 of them. Above the last band, the last band applies.
DEFAULT_FEEDER_BANDS = [
    {"max_kva": 100, "min": 1, "max": 2}, {"max_kva": 160, "min": 1, "max": 3},
    {"max_kva": 250, "min": 2, "max": 4}, {"max_kva": 400, "min": 3, "max": 6},
    {"max_kva": 630, "min": 4, "max": 8}, {"max_kva": 1000, "min": 6, "max": 12},
    {"max_kva": 2000, "min": 10, "max": 20}, {"max_kva": 3150, "min": 16, "max": 32},
]


class LoadShapeConfig(BaseModel):
    """How the load is spread over the grid (pf_datagen/load_shape.py).

    point_sizing: `series` uses the historical values as recorded; `rating`
    takes only each series' shape and sizes the point by its transformer
    rating x utilisation (MV and Feeder data then populate every connection
    point); `auto` = `rating` for MV / Feeder databases, `series` for LV.
    reactive: `series` keeps historical Q and draws a power factor where Q is
    missing; `pf` always draws it (inductive, per point per reassignment
    block). spatial_sigma / correlation_km: noise correlated along the cables
    (log-normal, mean 1); q_sigma: extra noise on Q only.
    scale_over (`rating`): what a series' shape is divided by -- its mean |P|
    over the whole `database` (a fixed sample of its timestamps, the same for
    every run), so a window keeps the level of its time of day and season;
    or over the `run`'s own timestamps (before 2026-11), which sets every
    window to the same average level.
    size_by (`rating`): what `utilisation` measures -- the point's `mean`
    |P| (average loading, the default), or its `peak`, the 99th percentile
    of |P|, so 99 % of the time |P| <= utilisation x kVA and a series that
    exports a lot is bounded by the rating too.
    feeders_per_point (Feeder databases): `one` series per point (distinct
    across the grid while the database has enough); or `by_kva`, a count per
    point drawn from its transformer's band in `feeder_bands`, its series
    drawn without replacement, the point's load their sum (under `rating`
    sizing the sum is sized as one series: divided by its own mean or peak)."""
    point_sizing: Literal["auto", "series", "rating"] = "auto"
    utilisation: float = Field(default=0.4, gt=0, le=2)
    reactive: Literal["series", "pf"] = "series"
    pf_min: float = Field(default=0.90, gt=0, le=1)
    pf_max: float = Field(default=0.99, gt=0, le=1)
    spatial_sigma: float = Field(default=0.0, ge=0, lt=2)
    correlation_km: float = Field(default=2.0, gt=0, le=1000)
    q_sigma: float = Field(default=0.0, ge=0, lt=2)
    scale_over: Literal["database", "run"] = "database"
    size_by: Literal["mean", "peak"] = "mean"
    feeders_per_point: Literal["one", "by_kva"] = "one"
    feeder_bands: List[FeederBand] = Field(default_factory=lambda: [FeederBand(**b) for b in DEFAULT_FEEDER_BANDS],
                                           min_length=1)

    @model_validator(mode="after")
    def _pf_range(self):
        if self.pf_min > self.pf_max:
            raise ValueError("load_shape.pf_min must not exceed pf_max.")
        kva = [b.max_kva for b in self.feeder_bands]
        if kva != sorted(kva) or len(set(kva)) != len(kva):
            raise ValueError("load_shape.feeder_bands must be in increasing max_kva order.")
        return self


# Where a group's plants go: any eligible node, weighted by transformer kVA,
# around a random centre, or towards the feeder ends.
DER_PLACEMENTS = ("random", "load", "clustered", "feeder_end")
DER_MODES = ("fixed_pf", "q_v", "p_v", "voltage")
DER_CONTROLS = ("mixed",) + DER_MODES
# The control-mode mix of a DER type: a group with control `mixed` draws each
# plant's mode from its type's mix, and resample_grid_settings redraws the
# grid's own units from theirs (a type without one: fixed cos phi).
DEFAULT_MODE_MIX: Dict[str, Dict[str, float]] = {
    "pv": {"q_v": 0.4, "fixed_pf": 0.4, "p_v": 0.2},
    "wind": {"q_v": 0.5, "fixed_pf": 0.5},
    "chp": {"fixed_pf": 0.7, "voltage": 0.3},
    "hydro": {"fixed_pf": 0.7, "voltage": 0.3},
    "battery": {"fixed_pf": 1.0},
}
# Fields of the earlier DER formats, named in the error a generating request
# that still uses them gets (penetration sizing, location weights, mode mixes
# and the single generation database).
_REMOVED_DER_FIELDS = ("historical", "penetration_min", "penetration_max", "sampling", "bins", "units_min",
                       "units_max", "min_unit_share", "locations", "modes", "types", "profile_source")


class DERGroup(BaseModel):
    """count_min..count_max generated plants of `capacity_mw` each, following one
    historical generation database (pf_datagen/der_profiles.py), placed by
    `placement` and controlled by `control`.

    interpretation: `aggregated` -- the database is the summed output of many
    plants (e.g. E-REDES' injection by technology); each plant k of the group
    is a synthetic plant following its shape f(t) = P(t) / max(P over the
    reference year), with a fixed performance factor and its own AR(1)
    variation (der.generation):
    P_k(t) = clip(capacity * f(t) * a_k * (1 + eps_k(t)), 0, capacity).
    `individual` -- each series of the database is one plant, used as
    recorded (W) and rated at its maximum; the count and capacity_mw don't apply.

    The number of plants is drawn once per run, uniformly in [count_min,
    count_max] from the run's seed: in a group of grids each grid draws its
    own. `count` sets both (a fixed number). count_min 0 lets a grid have
    none of the group.

    placement: `random` (uniform over the eligible nodes), `load` (weighted by
    the node's transformer kVA), `clustered` (around a random centre,
    exp(-cable distance / der.cluster_km)) or `feeder_end` (weighted by cable
    distance from the substation). control: one mode for every plant of the
    group, or `mixed` -- each plant's drawn from its type's mix
    (DEFAULT_MODE_MIX)."""
    database_id: str = Field(min_length=1, description="Historical database of a generation type (not 'load').")
    interpretation: Literal["aggregated", "individual"] = "aggregated"
    count_min: int = Field(default=1, ge=0, le=1000, description="Aggregated: fewest plants per grid.")
    count_max: int = Field(default=1, ge=0, le=1000, description="Aggregated: most plants per grid.")
    capacity_mw: Optional[float] = Field(default=None, gt=0, le=10000,
                                         description="Aggregated: each plant's capacity (MW).")
    placement: Literal["random", "load", "clustered", "feeder_end"] = "random"
    control: Literal["mixed", "fixed_pf", "q_v", "p_v", "voltage"] = "mixed"

    @model_validator(mode="before")
    @classmethod
    def _fixed_count(cls, data):
        # `count`: a fixed number, i.e. count_min = count_max.
        if isinstance(data, dict) and "count" in data:
            data = dict(data)
            n = data.pop("count")
            data.setdefault("count_min", n)
            data.setdefault("count_max", n)
        return data

    @model_validator(mode="after")
    def _valid(self):
        if self.count_min > self.count_max:
            raise ValueError(f"der.groups ({self.database_id}): count_min must not exceed count_max.")
        if self.interpretation == "aggregated" and self.capacity_mw is None:
            raise ValueError(f"der.groups ({self.database_id}): an aggregated group needs capacity_mw, each "
                             "plant's capacity.")
        return self


class DERGenerationConfig(BaseModel):
    """How every group's generation database is read and turned into plants
    (pf_datagen/der_profiles.py).

    Time: `generation_start` (in the databases' clock, `timezone`) is where
    the generation window starts, matched to the load's first timestamp;
    `keep_time_of_day` rounds that shift to whole days so day and night line
    up. Without it the generation is read at the same instants as the load.
    One shift for every database, so all technologies stay on one calendar.
    The sparser of the load and the generation is interpolated linearly --
    unless `load_aggregation_s` is set: then both are averaged over its
    intervals (a generation coarser than them still interpolated); a
    timestamp in a gap of the generation data takes the same clock time on
    the nearest day with data (never zero).

    Aggregated plants: performance factor a_k ~ U[performance_min,
    performance_max], drawn once per plant; eps_k AR(1) with stationary std
    `sigma` and lag-one correlation `rho` per 15 minutes. Synthetic plants
    derived from an aggregate pattern, not measurements of real ones."""
    generation_start: Optional[datetime] = Field(
        default=None, description="Start of the generation window (database clock); none: the load's instants.")
    keep_time_of_day: bool = Field(
        default=True, description="Round the window shift to whole days, so day and night line up.")
    timezone: str = Field(default="UTC", description="Clock of the databases' timestamps (IANA name or 'UTC').")
    reference_year: Optional[int] = Field(
        default=None, ge=1900, le=2200,
        description="Aggregated: the year whose maximum normalises a profile; none: each database's "
                    "best-covered year.")
    performance_min: float = Field(default=0.90, gt=0, le=1.5)
    performance_max: float = Field(default=1.00, gt=0, le=1.5)
    rho: float = Field(default=0.85, ge=0, lt=1, description="AR(1) correlation between 15-minute steps.")
    sigma: float = Field(default=0.08, ge=0, lt=1, description="Stationary std of the relative variation.")

    @model_validator(mode="after")
    def _valid(self):
        if self.performance_min > self.performance_max:
            raise ValueError("der.generation.performance_min must not exceed performance_max.")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"der.generation: unknown time zone '{self.timezone}'.")
        return self


class DERConfig(BaseModel):
    """DER per sample (pf_datagen/der_sampling.py). `source`: `grid` uses
    only the grid's own DER (no generated units); `add` adds the generated
    plants to the grid's DER; `replace` uses generated plants only.

    Generated plants are listed in `groups` (count x capacity, or every
    series of an individual-plant database), each following a historical
    generation database; `generation` holds the settings they share. The
    installed penetration (capacity / P_max) is therefore fixed by the groups
    and only reported. cos phi and voltage set-points of the drawn modes come
    from the ranges here; with resample_grid_settings the grid's own units
    are redrawn from their type's mode mix (DEFAULT_MODE_MIX)."""
    source: Literal["grid", "add", "replace"] = "grid"
    groups: List[DERGroup] = Field(
        default_factory=list, description="The generated plants; required unless source 'grid'.")
    generation: DERGenerationConfig = Field(default_factory=DERGenerationConfig)
    cluster_km: float = Field(default=2.0, gt=0, le=1000)
    cos_phi_min: float = Field(default=0.95, gt=0, le=1)
    cos_phi_max: float = Field(default=1.0, gt=0, le=1)
    v_set_min: float = Field(default=0.99, ge=0.9, le=1.1)
    v_set_max: float = Field(default=1.03, ge=0.9, le=1.1)
    resample_grid_settings: bool = False
    layout: Optional[Literal["per_sample", "per_run"]] = Field(
        default=None,
        description="`per_sample`: the plants are placed (and their modes drawn) anew for every sample. "
                    "`per_run`: the first sample's placement for the whole run, as a time series needs. "
                    "Omitted: follows the timeline -- per_run for 'window', per_sample for 'spread'.",
    )

    @model_validator(mode="before")
    @classmethod
    def _removed_fields(cls, data):
        # A request that generates DER in an earlier format: say what replaced it.
        if isinstance(data, dict) and data.get("source", "grid") != "grid" and not data.get("groups"):
            old = [k for k in _REMOVED_DER_FIELDS if k in data]
            if old:
                raise ValueError(
                    f"der.{', der.'.join(old)} no longer exist: generated DER are listed in der.groups -- "
                    "[{database_id, interpretation, count_min, count_max, capacity_mw, placement, control}] -- "
                    "with the shared "
                    "time and plant settings in der.generation.")
        return data

    @model_validator(mode="after")
    def _valid(self):
        if self.source != "grid" and not self.groups:
            raise ValueError(f"der.source '{self.source}' generates DER: list them in der.groups (each a "
                             "generation database_id, and for aggregated data count_min, count_max and "
                             "capacity_mw).")
        for lo, hi, name in ((self.cos_phi_min, self.cos_phi_max, "cos_phi"),
                             (self.v_set_min, self.v_set_max, "v_set")):
            if lo > hi:
                raise ValueError(f"der.{name}_min must not exceed {name}_max.")
        return self

    @property
    def database_ids(self) -> List[str]:
        """The generation databases the groups follow, in first-use order (none for `grid`)."""
        if self.source == "grid":
            return []
        return list(dict.fromkeys(g.database_id for g in self.groups))

    def aggregated(self, database_id: str) -> bool:
        """Some group reads the database as aggregated (so its profile needs a reference maximum)."""
        return any(g.database_id == database_id and g.interpretation == "aggregated" for g in self.groups)


class SetpointsConfig(BaseModel):
    """Operating set-points per sample (pf_datagen/setpoints.py).

    slack_mode: None keeps slack_voltage_pu's behaviour (fixed, or a range
    redrawn for every output); `fixed` uses its value; `uniform` draws one
    value per sample (shared by its topology variants) from its range,
    uniform or stratified; `load_dependent` follows the load level:
    v = slack_v0 + (slack_v1 - slack_v0) x level, +/- slack_noise.
    per_substation: an independent draw for each substation.
    shunt_steps: `fixed` (the grid's step), `random` (0..n_steps per bank and
    sample) or `voltage` (each bank switches steps to keep its node inside
    [shunt_band_min, shunt_band_max]). regulators_resample: draw each
    regulator's set-point, deadband and starting tap per sample.
    transformer_taps_resample: draw each MV/LV transformer's off-load tap per
    topology variant, within its range -- or, with transformer_tap_steps N,
    within +/-N steps of nominal (tap 0) and its range."""
    slack_mode: Optional[Literal["fixed", "uniform", "load_dependent"]] = None
    slack_sampling: Literal["uniform", "stratified"] = "stratified"
    slack_bins: int = Field(default=10, ge=1, le=1000)
    slack_v0: float = Field(default=1.00, gt=0.8, lt=1.2)
    slack_v1: float = Field(default=1.04, gt=0.8, lt=1.2)
    slack_noise: float = Field(default=0.01, ge=0, lt=0.2)
    per_substation: bool = False
    shunt_steps: Literal["fixed", "random", "voltage"] = "fixed"
    shunt_band_min: float = Field(default=0.97, gt=0.8, lt=1.2)
    shunt_band_max: float = Field(default=1.03, gt=0.8, lt=1.2)
    regulators_resample: bool = False
    reg_v_set_min: float = Field(default=0.98, gt=0.8, lt=1.2)
    reg_v_set_max: float = Field(default=1.03, gt=0.8, lt=1.2)
    reg_deadband_min: float = Field(default=0.015, gt=0, lt=0.2)
    reg_deadband_max: float = Field(default=0.025, gt=0, lt=0.2)
    transformer_taps_resample: bool = False
    transformer_tap_steps: Optional[int] = Field(
        default=None, ge=0, le=50,
        description="Redrawn taps within +/- this many steps of nominal; none: the transformer's full range.")

    @model_validator(mode="after")
    def _valid(self):
        for lo, hi, name in ((self.shunt_band_min, self.shunt_band_max, "shunt_band"),
                             (self.reg_v_set_min, self.reg_v_set_max, "reg_v_set"),
                             (self.reg_deadband_min, self.reg_deadband_max, "reg_deadband")):
            if lo >= hi:
                raise ValueError(f"setpoints.{name}_min must be below {name}_max.")
        return self


class PFDataGenRequest(BaseModel):
    historical_database_id: str = Field(
        description="Historical database (see gridarena/routers/historical.py) to source P/Q injections from. "
                    "Its series must all share one grid_level (MV, LV, or Feeder) -- that determines how "
                    "injections are assigned to MV connection points.",
    )
    reassignment_period_timesteps: int = Field(
        default=100, ge=1,
        description="How many consecutive scenarios (in timestamp order) share one random "
                    "series-to-point assignment before it's redrawn, e.g. 100 = reassign every "
                    "100 scenarios.",
    )
    scenario_count: int = Field(default=1000, ge=1, le=200000, description="Total timesteps to use as scenarios.")
    timeline: Literal["spread", "window"] = Field(
        default="spread",
        description="'spread': timestamps evenly spread over the database's timeline. 'window': consecutive "
                    "timestamps (for forecasting datasets; turn the per-sample random settings off).",
    )
    timeline_start: Optional[datetime] = Field(
        default=None, description="Start of a 'window' timeline; omitted: drawn from the seed.")
    load_aggregation_s: Optional[int] = Field(
        default=None, ge=60, le=86400,
        description="Average the load over clock-aligned intervals of this many seconds before it becomes "
                    "samples (e.g. 900: :00, :15, :30, :45, each labelled with its start). The run then steps "
                    "at this interval, and generation databases with a finer step are averaged over the same "
                    "intervals. Omitted: the load database's own step. Must divide a day.")
    load_noise_sigma: float = Field(
        default=0.0, ge=0, lt=1,
        description="Local load noise: an independent log-normal factor (mean 1) per load point and sample, "
                    "0 = off. Spatially correlated noise is in load_shape.",
    )
    load_level: LoadLevelConfig = Field(
        default=LoadLevelConfig(),
        description="Load level per sample: off, a multiplier on the historical load, or a fraction of the "
                    "grid's loadability limit P_max.",
    )
    load_shape: LoadShapeConfig = Field(
        default=LoadShapeConfig(),
        description="Point sizing, reactive power and spatially correlated noise of the load.",
    )
    der: DERConfig = Field(
        default=DERConfig(),
        description="Generated DER: plant groups (count x capacity, a generation database, placement and "
                    "control each). `source: grid` (default) uses only the grid's own DER.",
    )
    setpoints: SetpointsConfig = Field(
        default=SetpointsConfig(),
        description="Busbar set-point mode, capacitor bank steps, regulator and transformer tap draws.",
    )
    topology_perturbation: TopologyPerturbationConfig = TopologyPerturbationConfig()
    reconfiguration: ReconfigurationConfig = Field(
        default=ReconfigurationConfig(),
        description="Random radial configurations; each topology variant is a configuration plus an outage set.",
    )
    outages: OutagesConfig = Field(
        default=OutagesConfig(),
        description="Transformer outages, and DER trips, capacitor banks out and regulators bypassed per sample.",
    )
    design: DesignConfig = Field(
        default=DesignConfig(),
        description="How load level and busbar set-point pair up across samples.",
    )
    target_mix: TargetMixConfig = Field(
        default=TargetMixConfig(),
        description="Minimum share of samples with a violation, reached by oversampling after a pilot.",
    )
    variants_per_sample: Union[Literal["all"], int] = Field(
        default="all",
        description="'all': every sample on every topology variant. A number m: each sample on m variants, "
                    "spread evenly over the variants.",
    )
    measurements: MeasurementsConfig = Field(
        default=MeasurementsConfig(),
        description="Noisy, partial measurements next to each clean solution.",
    )
    admittance_perturbation: AdmittancePerturbationConfig = AdmittancePerturbationConfig()
    parameters: ParametersConfig = Field(
        default=ParametersConfig(),
        description="Line length, conductor temperature, cable swaps and transformer impedance uncertainty.",
    )
    voltage_limits: VoltageLimits = VoltageLimits()
    slack_voltage_pu: Union[float, SlackVoltageRange] = Field(
        default=1.0,
        description="Substation busbar voltage set-point (p.u. of nominal): a fixed value, or {min, max} to draw "
                    "it uniformly per sample. The busbar is an ideal source -- no upstream grid is modelled.",
    )
    load_model: LoadModel = Field(
        default=LoadModel(),
        description="ZIP load model: {p: {z, i, p}, q: {z, i, p}}. Default constant power.",
    )
    solver: Literal["sweep", "newton"] = Field(
        default="sweep",
        description="'sweep': backward/forward sweep, radial only (each energised part fed by one substation). "
                    "'newton': Newton-Raphson, any topology (loops, closed ties, several substations).",
    )
    timezone: str = Field(
        default="Europe/Lisbon",
        description="IANA time zone of the historical timestamps (local clock time); places solar noon "
                    "for DER with a solar profile.",
    )
    pv_clearness: ClearnessRange = Field(
        default=ClearnessRange(),
        description="Daily clearness factor of the solar profile, drawn uniformly per day from [min, max].",
    )
    s_base_mva: float = Field(
        default=1.0, gt=0, le=1000,
        description="Three-phase power base (MVA) for the per-unit output columns. The voltage base is "
                    "always the grid's line-to-line nominal voltage.",
    )
    chunk_commit_size: int = Field(default=100, ge=1, le=5000)
    seed: Optional[int] = Field(default=None, description="Base RNG seed. Omit for a fresh random seed (logged).")

    @model_validator(mode="after")
    def _valid_timezone(self):
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"Unknown time zone '{self.timezone}' (expected an IANA name, e.g. 'Europe/Lisbon').")
        return self

    @model_validator(mode="after")
    def _der_layout(self):
        # Omitted: the plants stay put over a consecutive window, move for spread samples.
        if self.der.layout is None:
            self.der = self.der.model_copy(
                update={"layout": "per_run" if self.timeline == "window" else "per_sample"})
        return self

    @model_validator(mode="after")
    def _topology_options(self):
        if self.reconfiguration.enabled and self.reconfiguration.allow_meshed and self.solver != "newton":
            raise ValueError("reconfiguration.allow_meshed closes loops: it needs solver 'newton'.")
        if self.variants_per_sample != "all" and self.variants_per_sample < 1:
            raise ValueError("variants_per_sample must be 'all' or at least 1.")
        if self.target_mix.min_violation_share is not None and self.load_level.mode == "off":
            raise ValueError("target_mix needs a load level (load_level.mode 'multiplier' or 'loadability').")
        if self.timeline_start is not None and self.timeline != "window":
            raise ValueError("timeline_start needs timeline 'window'.")
        if self.load_aggregation_s is not None and 86400 % self.load_aggregation_s:
            raise ValueError("load_aggregation_s must divide a day (e.g. 300, 900, 1800, 3600), so the "
                             "intervals line up with the clock.")
        return self

    @model_validator(mode="after")
    def _slack_in_range(self):
        v = self.slack_voltage_pu
        if not isinstance(v, SlackVoltageRange) and not 0.8 < v < 1.2:
            raise ValueError("slack_voltage_pu must lie between 0.8 and 1.2 p.u.")
        mode = self.setpoints.slack_mode
        if mode == "uniform" and not isinstance(v, SlackVoltageRange):
            raise ValueError("setpoints.slack_mode 'uniform' needs slack_voltage_pu as {min, max}.")
        if mode == "fixed" and isinstance(v, SlackVoltageRange):
            raise ValueError("setpoints.slack_mode 'fixed' needs slack_voltage_pu as one value.")
        return self


class PFDataGenStartResponse(BaseModel):
    run_id: str
    status: str
    message: str


class PFDataGenGroupRequest(BaseModel):
    """One run per grid, started together as a group (pf_datagen/run_manager.py)."""
    mv_grid_ids: Union[Literal["all"], List[str]] = Field(
        description="The MV grids to run, or 'all'.")
    request: PFDataGenRequest
    min_nodes: int = Field(default=10, ge=0, description="Grids with fewer nodes are skipped.")
    min_connection_points: int = Field(default=3, ge=0,
                                       description="Grids with fewer connection points are skipped.")

    @model_validator(mode="after")
    def _some_grids(self):
        if isinstance(self.mv_grid_ids, list) and not self.mv_grid_ids:
            raise ValueError("mv_grid_ids must list at least one grid, or be 'all'.")
        return self


class PFDataGenRunStatus(BaseModel):
    run_id: str
    mv_grid_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    config: Optional[dict] = None
    total_scenarios: int = 0
    completed_scenarios: int = 0
    failed_scenarios: int = 0
    n_topology_variants: int = 1
    seed_used: Optional[int] = None
    metrics: Optional[dict] = None
    group_id: Optional[str] = None
