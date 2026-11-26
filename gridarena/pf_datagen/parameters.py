"""Branch parameter uncertainty (Phase 2 axis 5).

Each option changes the branches a sample is solved on, never the grid
stored in the database:

- `length_sigma`: each line's length x a factor drawn uniformly in 1 +/- sigma,
  so its R, X and charging scale together;
- `swap_p`: a line's cable type is replaced, with that probability, by another
  type of the same family from the cable catalogue -- ratings included;
- `transformer_uk_sigma` / `transformer_xr_sigma`: an MV/LV transformer's
  impedance magnitude x 1 +/- sigma_uk and its X/R x 1 +/- sigma_xr;
- `temperature`: one conductor temperature per sample, drawn uniformly in
  [min, max], plus a per-cable offset within +/- spread. It changes R only,
  R_T = R_20 (1 + ALPHA (T - 20)): stored R is taken to be at 20 degC, and
  one coefficient serves aluminium and copper alike.

Length, swaps and transformers are drawn per topology variant (from
[seed, 8, 0, v]) or, with `draw: per_sample`, per sample (from [seed, 8, 1, i],
shared by its variants); temperature always per sample ([seed, 8, 2, i]).
Each option has its own sub-stream, so turning one on doesn't change the
others' draws.

Cable families. Underground cables have far more charging capacitance than
overhead lines (about 0.1-0.3 uF/km against about 0.01), so a type with at
least UNDERGROUND_CAP_UF_PER_KM is underground; without a capacitance, X/R
decides (underground below FAMILY_XR_THRESHOLD). The catalogue holds the
cable types in the database, deduplicated by value, plus any the request
uploads -- no invented values.
"""

from dataclasses import dataclass, replace
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from gridarena.pf_datagen.sampling import STREAM_PARAMETERS, sample_stream

ALPHA_PER_C = 0.004
REFERENCE_TEMP_C = 20.0
UNDERGROUND_CAP_UF_PER_KM = 0.05
FAMILY_XR_THRESHOLD = 0.49

_LENGTH, _SWAP, _TRANSFORMER = 0, 1, 2


def cable_family(r_ohm_per_km: float, x_ohm_per_km: float, cap_uf_per_km: Optional[float]) -> str:
    if cap_uf_per_km is not None and cap_uf_per_km > 0:
        return "underground" if cap_uf_per_km >= UNDERGROUND_CAP_UF_PER_KM else "overhead"
    if r_ohm_per_km <= 0:
        return "underground"
    return "underground" if x_ohm_per_km / r_ohm_per_km < FAMILY_XR_THRESHOLD else "overhead"


@dataclass(frozen=True)
class CableType:
    cable_id: str
    r_ohm_per_km: float
    x_ohm_per_km: float
    nom_curr_a: Optional[float]
    cap_uf_per_km: Optional[float] = None
    emerg_curr_a: Optional[float] = None

    @property
    def family(self) -> str:
        return cable_family(self.r_ohm_per_km, self.x_ohm_per_km, self.cap_uf_per_km)

    @property
    def value_key(self) -> tuple:
        return (round(self.r_ohm_per_km, 6), round(self.x_ohm_per_km, 6), self.nom_curr_a)


def load_catalogue(cursor) -> List[CableType]:
    """The MV cable types of every grid in the database, one per distinct
    (R, X, rating), named after the first cable id carrying it; a type stored
    with and without capacitance or emergency rating keeps the values given."""
    cursor.execute(
        """
        SELECT min("CableId"), "ImpReal", "ImpImag", "NomCurr", max("CapUfPerKm"), max("EmergCurr")
        FROM "MVCable" WHERE "ImpReal" IS NOT NULL AND "ImpImag" IS NOT NULL
        GROUP BY "ImpReal", "ImpImag", "NomCurr"
        ORDER BY 1
        """
    )
    return [CableType(cid, r, x, nom, cap, emerg) for cid, r, x, nom, cap, emerg in cursor.fetchall()]


def build_catalogue(db_types: Iterable[CableType], uploaded: Iterable = ()) -> List[CableType]:
    """Database types plus uploaded ones (schemas.CatalogueCable), deduplicated
    by value; an uploaded type replaces a database one with the same values."""
    by_key: Dict[tuple, CableType] = {t.value_key: t for t in db_types}
    for u in uploaded:
        t = CableType(u.cable_id, u.r_ohm_per_km, u.x_ohm_per_km, u.nom_curr_a, u.cap_uf_per_km, u.emerg_curr_a)
        by_key[t.value_key] = t
    return sorted(by_key.values(), key=lambda t: t.cable_id)


@dataclass
class StaticDraw:
    """Length factors, cable swaps and transformer factors for one variant (or sample)."""
    length_factor: Dict[str, float]
    swaps: Dict[str, CableType]
    transformer: Dict[str, Tuple[float, float]]     # connection id -> (uk factor, X/R factor)


@dataclass
class TemperatureDraw:
    temp_c: float                      # the sample's conductor temperature
    cable_temp_c: Dict[str, float]     # per line, with its offset


class ParameterSampler:
    def __init__(self, cfg, seed: int, branches, catalogue: Optional[List[CableType]] = None):
        self.cfg, self.seed = cfg, seed
        self.lines = [b for b in branches if b.kind != "transformer"]
        self.transformers = [b for b in branches if b.kind == "transformer"]
        self.catalogue = catalogue or []
        self.by_family: Dict[str, List[CableType]] = {}
        for t in self.catalogue:
            self.by_family.setdefault(t.family, []).append(t)

    @property
    def static_on(self) -> bool:
        c = self.cfg
        return c.length_sigma > 0 or c.swap_p > 0 or c.transformer_uk_sigma > 0 or c.transformer_xr_sigma > 0

    @property
    def temperature_on(self) -> bool:
        return self.cfg.temperature is not None

    @property
    def per_sample(self) -> bool:
        """Whether the branches change from one sample to the next."""
        return self.temperature_on or (self.static_on and self.cfg.draw == "per_sample")

    def static(self, variant_idx: int, sample_idx: int) -> Optional[StaticDraw]:
        if not self.static_on:
            return None
        key = (1, sample_idx) if self.cfg.draw == "per_sample" else (0, variant_idx)
        c = self.cfg
        length, swaps, trafo = {}, {}, {}
        if c.length_sigma > 0:
            rng = sample_stream(self.seed, STREAM_PARAMETERS, *key, _LENGTH)
            f = rng.uniform(1 - c.length_sigma, 1 + c.length_sigma, size=len(self.lines))
            length = {b.connection_id: float(x) for b, x in zip(self.lines, f)}
        if c.swap_p > 0:
            rng = sample_stream(self.seed, STREAM_PARAMETERS, *key, _SWAP)
            u = rng.random((len(self.lines), 2))
            for b, (u_swap, u_pick) in zip(self.lines, u):
                if u_swap >= c.swap_p:
                    continue
                own = (round(b.r_ohm_per_km, 6), round(b.x_ohm_per_km, 6), b.nom_curr_a)
                options = [t for t in self.by_family.get(
                    cable_family(b.r_ohm_per_km, b.x_ohm_per_km, b.cap_uf_per_km), []) if t.value_key != own]
                if options:
                    swaps[b.connection_id] = options[min(int(u_pick * len(options)), len(options) - 1)]
        if c.transformer_uk_sigma > 0 or c.transformer_xr_sigma > 0:
            rng = sample_stream(self.seed, STREAM_PARAMETERS, *key, _TRANSFORMER)
            u = rng.uniform(-1.0, 1.0, size=(len(self.transformers), 2))
            trafo = {b.connection_id: (1 + c.transformer_uk_sigma * a, 1 + c.transformer_xr_sigma * x)
                     for b, (a, x) in zip(self.transformers, u)}
        return StaticDraw(length, swaps, trafo)

    def temperature(self, sample_idx: int) -> Optional[TemperatureDraw]:
        t = self.cfg.temperature
        if t is None:
            return None
        rng = sample_stream(self.seed, STREAM_PARAMETERS, 2, sample_idx)
        temp = float(rng.uniform(t.min, t.max))
        offsets = rng.uniform(-t.spread, t.spread, size=len(self.lines)) if t.spread > 0 else np.zeros(len(self.lines))
        return TemperatureDraw(temp, {b.connection_id: temp + float(o) for b, o in zip(self.lines, offsets)})


def effective_branches(branches, static: Optional[StaticDraw], temperature: Optional[TemperatureDraw],
                       overrides: Optional[Dict[str, Tuple[float, float]]] = None):
    """The branches as solved: the existing admittance perturbation's per-km
    R/X (`overrides`, relative to the stored values), then a swapped cable
    type, the length factor, the conductor temperature and the transformer
    factors. Returns (branches, length_factor by line or None, n_swaps)."""
    overrides = overrides or {}
    out = []
    n_swaps = 0
    for b in branches:
        cid = b.connection_id
        if b.kind == "transformer":
            if static is not None and cid in static.transformer and b.z_ohm is not None:
                f_uk, f_xr = static.transformer[cid]
                z = b.z_ohm
                mag = abs(z) * f_uk
                if z.real > 0:
                    xr = z.imag / z.real * f_xr
                    r = mag / sqrt(1 + xr * xr)
                    b = replace(b, z_ohm=complex(r, r * xr))
                else:
                    b = replace(b, z_ohm=z * f_uk)
            out.append(b)
            continue
        r_ov, x_ov = overrides.get(cid, (None, None))
        swap = static.swaps.get(cid) if static is not None else None
        if swap is not None:
            n_swaps += 1
            # The perturbation keeps its relative size on the new type.
            fr = r_ov / b.r_ohm_per_km if r_ov is not None and b.r_ohm_per_km else 1.0
            fx = x_ov / b.x_ohm_per_km if x_ov is not None and b.x_ohm_per_km else 1.0
            r, x = swap.r_ohm_per_km * fr, swap.x_ohm_per_km * fx
            b = replace(b, cable_id=swap.cable_id, nom_curr_a=swap.nom_curr_a, cap_uf_per_km=swap.cap_uf_per_km,
                        emerg_curr_a=swap.emerg_curr_a)
        else:
            r = b.r_ohm_per_km if r_ov is None else r_ov
            x = b.x_ohm_per_km if x_ov is None else x_ov
        if temperature is not None:
            r *= 1 + ALPHA_PER_C * (temperature.cable_temp_c[cid] - REFERENCE_TEMP_C)
        length = b.length_m
        if static is not None and static.length_factor:
            length *= static.length_factor[cid]
        out.append(replace(b, r_ohm_per_km=r, x_ohm_per_km=x, length_m=length))
    length_factor = static.length_factor if static is not None and static.length_factor else None
    return out, length_factor, n_swaps
