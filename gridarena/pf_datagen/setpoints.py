"""Operating set-points per sample (Phase 2, axis 6).

- Busbar set-point (`setpoints.slack_mode`):
  - None: slack_voltage_pu as before -- a fixed value, or a range redrawn for
    every output (sample x variant) from stream [seed, 1], so existing runs
    reproduce;
  - `fixed`: slack_voltage_pu's value;
  - `uniform`: one value per sample, shared by its topology variants, drawn
    from slack_voltage_pu's range, uniform or stratified;
  - `load_dependent`: v = slack_v0 + (slack_v1 - slack_v0) x load level,
    +/- slack_noise -- operators (and line-drop compensation) raise the
    busbar with the load. The level is the sample's load level (axis 1) or,
    without one, its demand relative to the run's peak.
  With per_substation, each substation gets its own draw (a dict).
- Capacitor bank / reactor steps (`shunt_steps`): `random` draws 0..n_steps
  per bank and sample; `voltage` is handled by the control loop.
- Regulators (`regulators_resample`): set-point, deadband and starting tap
  drawn per sample.
- MV/LV transformer off-load taps (`transformer_taps_resample`): drawn per
  topology variant within each transformer's range, or with
  `transformer_tap_steps` N within +/-N steps of nominal (tap 0).
"""

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from gridarena.pf_datagen.sampling import (
    STREAM_SETPOINTS, STREAM_TRANSFORMER_TAPS, draw_values, sample_stream,
)

SLACK_CLIP = (0.81, 1.19)
_PER_SUBSTATION_OFFSET = 1_000_000_000      # keeps per-substation streams apart from per-sample ones


def _clip(v):
    return np.clip(v, *SLACK_CLIP)


class SetpointSampler:
    def __init__(self, cfg, slack, substations: Sequence[str], n_samples: int, seed: int,
                 load_levels: Optional[List[Optional[float]]] = None,
                 gross_demand: Optional[Sequence[float]] = None):
        self.cfg, self.slack, self.seed = cfg, slack, seed
        self.subs = list(substations)
        self.legacy_rng = np.random.default_rng([seed, 1])
        self.values: Optional[np.ndarray] = None      # (n_samples, n_subs or 1)
        mode = cfg.slack_mode
        cols = len(self.subs) if cfg.per_substation else 1
        if mode == "fixed":
            self.values = np.full((n_samples, cols), float(slack))
        elif mode == "uniform":
            self.values = np.column_stack([
                draw_values(n_samples, slack.min, slack.max, cfg.slack_sampling, cfg.slack_bins,
                            np.random.default_rng([seed, STREAM_SETPOINTS, _PER_SUBSTATION_OFFSET + j]))
                for j in range(cols)
            ])
        elif mode == "load_dependent":
            level = self._levels(n_samples, load_levels, gross_demand)
            self._rel = self._levels(n_samples, None, gross_demand)
            self._noise = noise = np.random.default_rng([seed, STREAM_SETPOINTS]).uniform(
                -cfg.slack_noise, cfg.slack_noise, size=(n_samples, cols)) if cfg.slack_noise > 0 \
                else np.zeros((n_samples, cols))
            self.values = _clip(cfg.slack_v0 + (cfg.slack_v1 - cfg.slack_v0) * level[:, None] + noise)

    def set_levels(self, ids: Sequence[int], levels: Sequence[Optional[float]]) -> None:
        """Load levels redrawn for samples `ids` (target mix): `load_dependent`
        set-points follow them, with the same noise."""
        if self.cfg.slack_mode != "load_dependent":
            return
        cfg = self.cfg
        for i, lv in zip(ids, levels):
            level = self._rel[i] if lv is None else lv
            self.values[i] = _clip(cfg.slack_v0 + (cfg.slack_v1 - cfg.slack_v0) * level + self._noise[i])

    @staticmethod
    def _levels(n, load_levels, gross_demand) -> np.ndarray:
        """Each sample's load level; samples without one (level off, or a
        near-empty sample) use their demand relative to the run's peak."""
        g = np.asarray(gross_demand if gross_demand is not None else np.zeros(n), dtype=float)
        rel = g / g.max() if g.size and g.max() > 0 else np.zeros(n)
        lv = load_levels or [None] * n
        return np.array([rel[i] if lv[i] is None else lv[i] for i in range(n)], dtype=float)

    # -- busbar ----------------------------------------------------------------

    def slack_for_output(self, i: int) -> Union[float, Dict[str, float]]:
        """Set-point for one output of sample i: one value, or one per substation."""
        if self.values is None:                       # slack_mode None: slack_voltage_pu as before
            if hasattr(self.slack, "min"):
                if self.cfg.per_substation:
                    return {s: float(self.legacy_rng.uniform(self.slack.min, self.slack.max)) for s in self.subs}
                return float(self.legacy_rng.uniform(self.slack.min, self.slack.max))
            return float(self.slack)
        row = self.values[i]
        if self.cfg.per_substation:
            return {s: float(v) for s, v in zip(self.subs, row)}
        return float(row[0])

    @staticmethod
    def mean(v) -> float:
        return float(np.mean(list(v.values()))) if isinstance(v, dict) else float(v)

    # -- capacitor banks, regulators, transformer taps ----------------------------

    def shunt_steps(self, i: int, shunts) -> Optional[Dict[str, int]]:
        """`random` mode: steps switched in per bank for sample i (None otherwise)."""
        if self.cfg.shunt_steps != "random":
            return None
        rng = sample_stream(self.seed, STREAM_SETPOINTS, i)
        return {sh.shunt_id: int(rng.integers(0, sh.n_steps + 1)) for sh in shunts}

    def regulators(self, i: int, regulators: List) -> List:
        """Regulators with their set-point, deadband and starting tap drawn for sample i."""
        if not self.cfg.regulators_resample or not regulators:
            return regulators
        cfg = self.cfg
        rng = np.random.default_rng([self.seed, STREAM_SETPOINTS, i, 1])
        return [replace(r, v_set_pu=float(rng.uniform(cfg.reg_v_set_min, cfg.reg_v_set_max)),
                        deadband_pu=float(rng.uniform(cfg.reg_deadband_min, cfg.reg_deadband_max)),
                        tap_pos=int(rng.integers(r.tap_min, r.tap_max + 1)))
                for r in regulators]

    def transformer_taps(self, variant_idx: int, transformers) -> Dict[str, int]:
        """Off-load tap per transformer for one topology variant (empty if not resampled)."""
        if not self.cfg.transformer_taps_resample:
            return {}
        rng = sample_stream(self.seed, STREAM_TRANSFORMER_TAPS, variant_idx)
        n = self.cfg.transformer_tap_steps
        out = {}
        for t in transformers:
            lo, hi = t.tap_min, t.tap_max
            if n is not None:
                # Within +/-n of nominal, kept inside the range (a range beside 0: its end nearest 0).
                lo, hi = max(lo, -n), min(hi, n)
                if lo > hi:
                    lo = hi = t.tap_min if t.tap_min > 0 else t.tap_max
            out[t.connection_id] = int(rng.integers(lo, hi + 1))
        return out
