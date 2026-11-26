"""Drawing a per-sample value from a range (Phase 2, proposal section 5.1).

`uniform` draws independently, which over-samples nothing on average but
leaves gaps and clusters in short runs. `stratified` splits the range into
equal bins and spreads the samples evenly over them (bin counts differ by at
most one), with a uniform draw inside each bin, in a shuffled order so the
bins don't follow the timeline. Each Phase 2 axis draws from its own
random stream (see `stream`), so turning one axis on doesn't change another's
draws for the same seed.
"""

from typing import Literal

import numpy as np

# Random streams per axis: numpy.random.default_rng([seed, k]), or [seed, k, i]
# for a draw specific to sample (or variant) i. k = 1 (busbar set-point as
# configured by slack_voltage_pu) and 2 (PV clearness) predate Phase 2; the
# main generator (default_rng(seed)) keeps series assignment and topology
# variants.
STREAM_LOAD_LEVEL = 3
STREAM_LOAD_SHAPE = 4
STREAM_DER_LAYOUT = 5            # [seed, 5, i]: sample i's placement and control modes of the generated plants
STREAM_DER_AVAILABILITY = 6      # unused, reserved: the modelled generated-DER output (removed) drew from it
STREAM_OUTAGES = 7               # [seed, 7, i]: DER trips, shunts out, regulators bypassed in sample i
STREAM_PARAMETERS = 8            # [seed, 8, 0, v] per variant, [seed, 8, 1, i] per sample, [seed, 8, 2, i]: temperature
STREAM_SETPOINTS = 9             # busbar set-points per sample; [seed, 9, i]: capacitor steps, regulators
STREAM_RECONFIGURATION = 10      # configurations; [seed, 10, c]: outage sets of configuration c >= 1
STREAM_MEASUREMENTS = 11         # [seed, 11, 0 / 1]: meter layout; [seed, 11, 2 / 3 / 4, i, v]: noise, missing, pseudo
STREAM_TRANSFORMER_TAPS = 12     # [seed, 12, v]: off-load taps of topology variant v
STREAM_GRID_DER_SETTINGS = 13    # [seed, 13, i]: the grid's own DER settings, redrawn for sample i
STREAM_VARIANT_CHOICE = 14       # which variants each sample is solved on (variants_per_sample)
STREAM_DESIGN = 15               # joint design (lhs / factorial) over level and busbar set-point
STREAM_TARGET_MIX = 16           # [seed, 16, 0]: pilot samples; [seed, 16, 1]: the rest's strata and values
STREAM_WINDOW = 17               # start of a `window` timeline
STREAM_DER_HISTORICAL = 18       # [seed, 18, k]: aggregated plant k's performance factor and variation
STREAM_DER_COUNT = 19            # [seed, 19, g]: group g's number of plants in this run (grid)


def stream(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng([seed, k])


def sample_stream(seed: int, k: int, *i: int) -> np.random.Generator:
    return np.random.default_rng([seed, k, *i])


def draw_values(n: int, lo: float, hi: float, sampling: Literal["uniform", "stratified"], bins: int,
                rng: np.random.Generator) -> np.ndarray:
    """n values in [lo, hi]. lo == hi gives a constant."""
    if n <= 0:
        return np.zeros(0)
    if hi <= lo:
        return np.full(n, float(lo))
    if sampling == "uniform":
        return rng.uniform(lo, hi, size=n)
    bins = max(1, int(bins))
    bin_idx = rng.permutation(np.arange(n) % bins)
    width = (hi - lo) / bins
    return lo + (bin_idx + rng.uniform(0.0, 1.0, size=n)) * width


def bin_of(values: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    """Bin index (0..bins-1) of each value in [lo, hi]; values outside are clipped."""
    if hi <= lo:
        return np.zeros(len(values), dtype=int)
    idx = np.floor((np.asarray(values, dtype=float) - lo) / (hi - lo) * bins).astype(int)
    return np.clip(idx, 0, bins - 1)
