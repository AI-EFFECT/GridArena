"""MV-connected generators (DER) for the PF data-generation pipeline: the
power each one has available in a sample, and the control set-up the solver
needs (pf_datagen/solver.py::DERControl).

Profiles:
- "solar": p_max x clear-sky irradiance share x the day's clearness. The
  clear-sky global horizontal irradiance is Haurwitz's model,
  GHI = 1098 cos(z) exp(-0.057 / cos(z)) W/m^2, with the solar zenith z from
  the NOAA solar-position equations at the DER node's coordinates; the share
  is GHI / 1000 W/m^2 (STC), capped at 1. Clearness is one factor per local
  calendar day, shared by every PV unit of the run and drawn uniformly from
  the request's pv_clearness range. Timestamps are local clock time in the
  request's time zone.
- "constant": profile_value x p_max in every sample (a share in [-1, 1]; a
  negative value consumes, e.g. a charging battery or an EV hub).

Available power is capped at the rating s_rated. Reactive power stays inside
the rating's circle, sqrt(S^2 - P^2); outside fixed cos phi it is also
limited to q_limit_pu x S_rated (default 0.33, i.e. cos phi 0.95 at full
output). Generator convention throughout: positive P/Q = injected.
"""

from datetime import date, datetime, timezone
from math import acos, cos, exp, pi, radians, sin, sqrt, tan
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from gridarena.pf_datagen.solver import DERControl

STC_IRRADIANCE = 1000.0  # W/m^2


def solar_cos_zenith(ts_utc: datetime, lat_deg: float, lon_deg: float) -> float:
    """cos of the solar zenith angle (NOAA general solar position equations)."""
    day = ts_utc.timetuple().tm_yday
    hours = ts_utc.hour + ts_utc.minute / 60 + ts_utc.second / 3600
    g = 2 * pi / 365 * (day - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * cos(g) - 0.032077 * sin(g)
                       - 0.014615 * cos(2 * g) - 0.040849 * sin(2 * g))
    decl = (0.006918 - 0.399912 * cos(g) + 0.070257 * sin(g) - 0.006758 * cos(2 * g)
            + 0.000907 * sin(2 * g) - 0.002697 * cos(3 * g) + 0.00148 * sin(3 * g))
    true_solar_min = hours * 60 + eqtime + 4 * lon_deg
    hour_angle = radians(true_solar_min / 4 - 180)
    lat = radians(lat_deg)
    return sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hour_angle)


def clear_sky_share(ts_local: datetime, tz: str, lat_deg: float, lon_deg: float) -> float:
    """Clear-sky GHI as a share of STC irradiance (0 at night, at most 1)."""
    ts_utc = ts_local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)
    cz = solar_cos_zenith(ts_utc, lat_deg, lon_deg)
    if cz <= 0:
        return 0.0
    return min(1.0, 1098.0 * cz * exp(-0.057 / cz) / STC_IRRADIANCE)


def daily_clearness(timestamps: Iterable[datetime], lo: float, hi: float,
                    rng: np.random.Generator) -> Dict[date, float]:
    """One clearness factor per local calendar day, drawn in date order so
    the draws depend only on the days covered."""
    days = sorted({ts.date() for ts in timestamps})
    return {d: float(rng.uniform(lo, hi)) if hi > lo else float(lo) for d in days}


def available_p_w(der, ts_local: datetime, tz: str, clearness: Dict[date, float]) -> float:
    """Active power (W) the DER can deliver in this sample, from its profile."""
    p_max = der.p_max_kw * 1e3
    if der.profile == "solar":
        if der.lat is None or der.lon is None:
            raise Exception(f"DER '{der.der_id}': a solar profile needs coordinates on node '{der.node_id}'.")
        p = p_max * clear_sky_share(ts_local, tz, der.lat, der.lon) * clearness[ts_local.date()]
    else:
        p = p_max * der.profile_value
    s_rated = der.s_rated_kva * 1e3
    return min(max(p, -s_rated), s_rated)


def der_control(der, p_available_w: float, s_base_va: float) -> DERControl:
    """The solver's view of one DER for one sample (p.u. on s_base_va)."""
    s_rated = der.s_rated_kva * 1e3
    q_circle = sqrt(max(s_rated ** 2 - p_available_w ** 2, 0.0))
    q_max = q_circle if der.control_mode == "fixed_pf" else min(q_circle, der.q_limit_pu * s_rated)
    tan_phi = tan(acos(der.cos_phi)) * (-1.0 if der.excitation == "under" else 1.0)
    return DERControl(
        der_id=der.der_id, node_id=der.node_id, mode=der.control_mode,
        p_available=p_available_w / s_base_va, q_max=q_max / s_base_va, tan_phi=tan_phi,
        qv_curve=der.qv_curve, pv_curve=der.pv_curve, v_set_pu=der.v_set_pu,
    )


def parse_timestamp(ts: str) -> datetime:
    """Scenario timestamps are str(datetime) of the historical timeline."""
    return datetime.fromisoformat(ts)


def run_clearness(scenarios: List[dict], clearness_lo: float, clearness_hi: float,
                  rng: np.random.Generator) -> Dict[date, float]:
    """The run's daily clearness factors, one per local calendar day of its
    timestamps -- shared by the grid's PV and generated PV units."""
    return daily_clearness([parse_timestamp(sc["timestamp"]) for sc in scenarios], clearness_lo, clearness_hi, rng)


def p_available_by_scenario(ders: List, scenarios: List[dict], tz: str, clearness_lo: float,
                            clearness_hi: float, rng: np.random.Generator,
                            clearness: Optional[Dict[date, float]] = None) -> List[Dict[str, float]]:
    """For each scenario, der_id -> available active power (W). `clearness`:
    the run's daily factors if already drawn (run_clearness); otherwise they
    are drawn here when a DER has a solar profile."""
    if not ders:
        return [{} for _ in scenarios]
    stamps = [parse_timestamp(sc["timestamp"]) for sc in scenarios]
    if clearness is None:
        clearness = daily_clearness(stamps, clearness_lo, clearness_hi, rng) \
            if any(d.profile == "solar" for d in ders) else {}
    return [{d.der_id: available_p_w(d, ts, tz, clearness) for d in ders} for ts in stamps]
