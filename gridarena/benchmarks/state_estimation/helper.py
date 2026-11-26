"""Helper function for state estimation benchmarks"""

import hashlib
import random
import uuid
from typing import Any, Dict, Iterator, List, Literal, Tuple

from gridarena.benchmarks.general_noise import Difficulty as NoiseDifficulty
from gridarena.benchmarks.general_noise import (
    corrupt_measurement_record as _corrupt_record,
)
from gridarena.benchmarks.general_noise import get_profile as _get_profile


def _all_grid_ids(cursor) -> List[str]:
    cursor.execute("""SELECT DISTINCT grid_id FROM "Measurements" """)
    return [r[0] for r in cursor.fetchall()]


def _new_unique_mask(cursor, table: str, col: str) -> str:
    """Create an 10-hex masked id unique within the given table/col."""

    while True:
        m = uuid.uuid4().hex[:10]
        cursor.execute(f"SELECT 1 FROM {table} WHERE {col} = %s", (m,))
        if cursor.fetchone() is None:
            return m


def ensure_grid_masks(cursor):
    """Ensure all grids have the unique mask identifier."""

    for gid in _all_grid_ids(cursor):
        cursor.execute(
            """SELECT masked_id FROM GridMask \
                       WHERE grid_id = %s""",
            (gid,),
        )
        if cursor.fetchone() is None:
            masked = _new_unique_mask(cursor, table="GridMask", col="masked_id")
            cursor.execute(
                """INSERT INTO GridMask (grid_id, masked_id) \
                    VALUES (%s, %s)""",
                (gid, masked),
            )


def ensure_all_node_masks(cursor):
    """Ensure NodeMask exists for every (grid_id, node_id)."""

    cursor.execute(
        """SELECT DISTINCT grid_id, node_id, phase \
                   FROM "Measurements" """
    )
    for gid, nid, phase in cursor.fetchall():
        cursor.execute(
            """
            SELECT masked_node_id
            FROM NodeMask
            WHERE grid_id = %s AND node_id = %s AND phase = %s
            """,
            (gid, nid, phase),
        )
        if cursor.fetchone() is None:
            masked = _new_unique_node_mask(cursor, gid)
            cursor.execute(
                """
                INSERT INTO NodeMask (grid_id, node_id, phase, masked_node_id)
                VALUES (%s, %s, %s, %s)
                """,
                (gid, nid, phase, masked),
            )


def _new_unique_node_mask(cursor, grid_id: str) -> str:
    """Create an 10-hex masked node id unique within a grid."""

    while True:
        m = uuid.uuid4().hex[:10]
        cursor.execute(
            """
            SELECT 1
            FROM NodeMask
            WHERE grid_id = %s AND masked_node_id = %s
            """,
            (grid_id, m),
        )
        if cursor.fetchone() is None:
            return m


def iter_grids_with_masks(cursor) -> Iterator[Tuple[str, str]]:
    """Query database for mapping grid_id <-> masked_id."""

    cursor.execute("""SELECT grid_id, masked_id FROM GridMask""")
    for gid, mid in cursor.fetchall():
        yield gid, mid


ObservabilityDifficulty = Literal["clean", "easy", "medium", "hard"]

# Observability is defined by NON-observable fraction ranges.
_OBS_PROFILES: Dict[str, Dict[str, float]] = {
    "clean": {"min_non_obs_frac": 0.0, "max_non_obs_frac": 0.0, "unknown_history_hours": 0.0},
    "easy": {"min_non_obs_frac": 0.0, "max_non_obs_frac": 1.0 / 3.0, "unknown_history_hours": 1.0},
    "medium": {
        "min_non_obs_frac": 1.0 / 3.0,
        "max_non_obs_frac": 2.0 / 3.0,
        "unknown_history_hours": 6.0,
    },
    "hard": {
        "min_non_obs_frac": 2.0 / 3.0,
        "max_non_obs_frac": 0.999,
        "unknown_history_hours": 12.0,
    },
}


def get_noise_profile(difficulty: NoiseDifficulty) -> Dict[str, float]:
    # Kept for backward-compatibility with your router.
    return _get_profile(difficulty)


def get_observability_profile(difficulty: ObservabilityDifficulty) -> Dict[str, float]:
    if difficulty not in _OBS_PROFILES:
        raise ValueError(f"Unsupported observability difficulty: {difficulty}")
    return dict(_OBS_PROFILES[difficulty])


def corrupt_measurement_like_record(
    record: Dict[str, Any],
    rng: random.Random,
    profile: Dict[str, float],
) -> Dict[str, Any]:
    # Backward-compatible wrapper name used internally in this module.
    return _corrupt_record(record, rng, profile)


def corrupt_training_dataset(
    grids: Dict[str, Dict[str, List[Dict[str, Any]]]],
    noise_profile: Dict[str, float],
    difficulty: str = "clean",
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    rngs: Dict[tuple, random.Random] = {}
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for mgid, nodes in grids.items():
        out[mgid] = {}
        for mnid, series in nodes.items():
            seed = hashlib.sha256(f"{mgid}:{mnid}:{difficulty}:training".encode()).hexdigest()
            rng_key = (mgid, mnid, "training")
            if rng_key not in rngs:
                rngs[rng_key] = random.Random(seed)
            out[mgid][mnid] = [
                corrupt_measurement_like_record(rec, rngs[rng_key], noise_profile) for rec in series
            ]
    return out


def _pick_unknown_count(
    n_nodes: int,
    observability: ObservabilityDifficulty,
    rng: random.Random,
) -> int:
    """
    Picks number of NON-observable nodes (unknown at target timestamp) according to:
      clean:  0 unknown
      easy:   [0, 1/3)
      medium: [1/3, 2/3)
      hard:   [2/3, ~1)
    For easy/medium/hard, enforces at least 1 unknown and 1 known when n_nodes >= 2.
    """
    obs = get_observability_profile(observability)

    if observability == "clean":
        return 0

    min_f = float(obs["min_non_obs_frac"])
    max_f = float(obs["max_non_obs_frac"])

    f = rng.uniform(min_f, max_f)  # in [min_f, max_f)
    n_unknown = int(round(f * n_nodes))

    if n_nodes >= 2:
        n_unknown = max(1, min(n_unknown, n_nodes - 1))
    else:
        n_unknown = 0

    return n_unknown
