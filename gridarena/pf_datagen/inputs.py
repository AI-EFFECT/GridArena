"""What a PF data-generation run reads from the database, and its frozen form.

A run reads the database only at its start: the MV grid's topology, the cable
catalogue (axis 5 swaps) and the historical series, which the scenario builder
turns into one complex load vector per sample. `prepare_inputs` does exactly
that part of the job; everything after it (pf_datagen/job.py::execute) works on
a `RunInputs` alone, so the same inputs can be solved inside GridArena or,
saved with `save_inputs`, by the standalone package (pf_datagen/package.py)
on any machine.

The scenario builder draws the series assignment from the run's main random
generator, which the variants and admittance perturbation keep drawing from
afterwards, so the generator's state after the builder is part of the inputs:
restored from it, a run elsewhere gives the same rows as in GridArena.

Files (in an `inputs/` directory):
- grid.json       the topology (pf_datagen/grid_loader.py::MVGridTopology)
- catalogue.json  the database's cable types (pf_datagen/parameters.py::CableType)
- scenarios.npz   timestamps (str) and S (complex, samples x nodes, in
                  node_id_to_index order, W / VAr, before the load moves behind
                  the transformers)
- manifest.json   seed, generator state, and the request fields the scenarios
                  were built from (they can't change without rebuilding them)
- generation.json / generation_<k>.npz   only when DER are generated
                  (der.source not "grid"): each generation database's values
                  at each scenario (pf_datagen/der_profiles.py::GenerationProfile)
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

import numpy as np

from gridarena.pf_datagen.grid_loader import MVBranch, MVDER, MVGridTopology, MVShunt
from gridarena.pf_datagen.parameters import CableType

INPUTS_FORMAT = 1

# Request fields the stored scenarios depend on: the series, their assignment,
# sizing, reactive power, noise and timestamps. Editing them needs new inputs.
LOCKED_FIELDS = (
    "historical_database_id", "scenario_count", "load_noise_sigma", "reassignment_period_timesteps",
    "load_shape", "timeline", "timeline_start", "load_aggregation_s", "seed", "der.generation",
)
# The part of der.generation the generation profiles are read with ("der.generation" above),
# plus the databases the groups follow (and whether each needs a reference maximum).
_LOCKED_DER_GENERATION = ("generation_start", "keep_time_of_day", "timezone", "reference_year")


@dataclass
class RunInputs:
    topology: MVGridTopology
    catalogue: List[CableType]          # database types; uploaded ones are added at solve time
    scenarios: List[dict]               # {"timestamp": str, "S": complex ndarray}
    seed: int
    rng: np.random.Generator            # the main generator, as the scenario builder left it
    generation: Optional[dict] = None   # database_id -> der_profiles.GenerationProfile (generated DER)


def prepare_inputs(mv_grid_id: str, request, seed: int, db) -> RunInputs:
    """Read the grid, the catalogue and the historical series from the
    database and build the run's load scenarios -- and, when generated DER
    follow a generation database, its values at each scenario."""
    from gridarena.pf_datagen.grid_loader import get_mv_grid_topology
    from gridarena.pf_datagen.parameters import load_catalogue
    from gridarena.pf_datagen.sampling import STREAM_LOAD_SHAPE, STREAM_WINDOW, stream
    from gridarena.pf_datagen.scenario_builder import build_scenarios

    rng = np.random.default_rng(seed)
    der = request.der
    conn, cursor = db.get_db_connection()
    try:
        topology = get_mv_grid_topology(mv_grid_id, cursor)
        catalogue = load_catalogue(cursor)
        # End the read transaction now: building the scenarios can take long
        # on a large historical database, and an open transaction that has
        # read the MV tables would make anything needing a lock on them wait.
        conn.rollback()
        # A generation database with a finer step than the load's: a window
        # then steps at the finest generation interval, the load interpolated --
        # unless the load is aggregated: then the run steps at its intervals and
        # a finer generation is averaged over them.
        agg_s = request.load_aggregation_s
        finer_step_s = None
        if der.database_ids and not agg_s:
            from gridarena.pf_datagen.der_profiles import generation_step_s
            finer_step_s = min(generation_step_s(d, der.generation.timezone, cursor) for d in der.database_ids)
        scenarios = build_scenarios(
            mv_grid_id, topology.node_id_to_index, request.scenario_count,
            request.load_noise_sigma, request.historical_database_id,
            request.reassignment_period_timesteps, rng, cursor,
            load_shape=request.load_shape, point_kva=point_kva(topology), branches=topology.branches,
            shape_rng=stream(seed, STREAM_LOAD_SHAPE), timeline=request.timeline,
            timeline_start=request.timeline_start, window_rng=stream(seed, STREAM_WINDOW),
            finer_step_s=finer_step_s, aggregation_s=agg_s,
        )
        generation = None
        if der.database_ids:
            from gridarena.pf_datagen.der_profiles import build_generation_profile
            stamps = [s["timestamp"] for s in scenarios]
            generation = {d: build_generation_profile(der.generation, d, der.aggregated(d), stamps,
                                                      request.timezone, cursor, average_s=agg_s)
                          for d in der.database_ids}
    finally:
        conn.close()
    return RunInputs(topology, catalogue, scenarios, seed, rng, generation)


def point_kva(topology: MVGridTopology) -> dict:
    """Size of each connection point (rating sizing): its active transformers' kVA."""
    out = {}
    for b in topology.branches:
        if b.kind == "transformer" and b.in_service:
            out[b.from_node_id] = out.get(b.from_node_id, 0.0) + (b.rated_kva or 0.0)
    return out


def locked_values(request) -> dict:
    data = request.model_dump(mode="json")
    out = {k: data.get(k) for k in LOCKED_FIELDS if "." not in k}
    der = request.der
    if der.database_ids:
        gen = data["der"]["generation"]
        out["der.generation"] = {**{k: gen.get(k) for k in _LOCKED_DER_GENERATION},
                                 "databases": [[d, der.aggregated(d)] for d in der.database_ids]}
    else:
        out["der.generation"] = None
    return out


# ── Saving / loading ──

def _complex(z):
    return None if z is None else [z.real, z.imag]


def topology_to_dict(t: MVGridTopology) -> dict:
    d = asdict(t)
    for b in d["branches"]:
        b["z_ohm"] = _complex(b["z_ohm"])
    return d


def topology_from_dict(d: dict) -> MVGridTopology:
    branches = []
    for b in d["branches"]:
        b = dict(b)
        if b.get("z_ohm") is not None:
            b["z_ohm"] = complex(*b["z_ohm"])
        branches.append(MVBranch(**b))
    ders = []
    for x in d.get("ders", []):
        x = dict(x)
        x["qv_curve"], x["pv_curve"] = tuple(x["qv_curve"]), tuple(x["pv_curve"])
        ders.append(MVDER(**x))
    return MVGridTopology(
        mv_grid_id=d["mv_grid_id"],
        nominal_voltage_kv=d["nominal_voltage_kv"],
        substation_node_ids=list(d["substation_node_ids"]),
        node_id_to_index={k: int(v) for k, v in d["node_id_to_index"].items()},
        branches=branches,
        node_vn_kv=dict(d.get("node_vn_kv", {})),
        load_transfer={cp: [tuple(t) for t in ts] for cp, ts in d.get("load_transfer", {}).items()},
        shunts=[MVShunt(**s) for s in d.get("shunts", [])],
        ders=ders,
        node_coords={k: tuple(v) for k, v in d.get("node_coords", {}).items()},
    )


def save_inputs(inputs: RunInputs, request, directory) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grid.json", "w") as f:
        json.dump(topology_to_dict(inputs.topology), f, indent=1)
    with open(out / "catalogue.json", "w") as f:
        json.dump([asdict(c) for c in inputs.catalogue], f, indent=1)
    n_nodes = len(inputs.topology.node_id_to_index)
    s = (np.stack([sc["S"] for sc in inputs.scenarios]) if inputs.scenarios
         else np.zeros((0, n_nodes), dtype=complex))
    np.savez_compressed(out / "scenarios.npz", timestamps=np.array([sc["timestamp"] for sc in inputs.scenarios]),
                        S=s.astype(np.complex128))
    with open(out / "manifest.json", "w") as f:
        json.dump({
            "format": INPUTS_FORMAT,
            "mv_grid_id": inputs.topology.mv_grid_id,
            "seed": inputs.seed,
            "rng_state": inputs.rng.bit_generator.state,
            "n_scenarios": len(inputs.scenarios),
            "n_nodes": n_nodes,
            "locked": locked_values(request),
        }, f, indent=1)
    if inputs.generation:
        from gridarena.pf_datagen.der_profiles import profiles_to_files
        profiles_to_files(inputs.generation, out)


def load_manifest(directory) -> dict:
    with open(Path(directory) / "manifest.json") as f:
        manifest = json.load(f)
    if manifest.get("format") != INPUTS_FORMAT:
        raise ValueError(f"Unsupported inputs format {manifest.get('format')!r} (expected {INPUTS_FORMAT}).")
    return manifest


def load_inputs(directory) -> RunInputs:
    d = Path(directory)
    manifest = load_manifest(d)
    with open(d / "grid.json") as f:
        topology = topology_from_dict(json.load(f))
    with open(d / "catalogue.json") as f:
        names = {fl.name for fl in fields(CableType)}
        catalogue = [CableType(**{k: v for k, v in c.items() if k in names}) for c in json.load(f)]
    with np.load(d / "scenarios.npz") as z:
        scenarios = [{"timestamp": str(ts), "S": z["S"][i].copy()} for i, ts in enumerate(z["timestamps"])]
    rng = np.random.default_rng()
    rng.bit_generator.state = manifest["rng_state"]
    from gridarena.pf_datagen.der_profiles import profiles_from_files
    return RunInputs(topology, catalogue, scenarios, int(manifest["seed"]), rng, profiles_from_files(d))
