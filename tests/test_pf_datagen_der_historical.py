"""Generated DER: plant groups following historical generation databases
(pf_datagen/der_profiles.py, der_sampling.py)."""

from datetime import datetime, timedelta, timezone

import importlib

import numpy as np
import pytest

from gridarena.pf_datagen import der_profiles as dp
from gridarena.pf_datagen.der_sampling import DERSampler
from gridarena.pf_datagen.grid_loader import MVBranch, MVGridTopology
from gridarena.pf_datagen.schemas import DERConfig, DERGenerationConfig, PFDataGenRequest
from gridarena.pf_datagen import scenario_builder
from gridarena.pf_datagen.inputs import locked_values
from gridarena.pf_datagen.timeline import aggregate_timeline, interpolated_window, interval_start

DAY = 86400.0


def _profile(agg, n_series=1, step_s=900.0, t0=1_700_000_000.0, ratings=None, ref_max=None, database_id="gen",
             data_type="pv"):
    agg = np.asarray(agg, dtype=float)
    values = np.vstack([agg / n_series] * n_series)
    return dp.GenerationProfile(
        database_id=database_id, data_type=data_type, series_ids=[f"s{k}" for k in range(n_series)], values_w=values,
        ratings_w=np.asarray(ratings if ratings is not None else values.max(axis=1)), reference_year=2024,
        reference_max_w=ref_max if ref_max is not None else float(agg.max()),
        gen_times=t0 + step_s * np.arange(len(agg)), step_s=step_s, shift_s=0.0, n_fallback=0)


def _day_profile(n_days=3, step_s=900.0, peak=40e6):
    """A clear-sky-like aggregate: zero at night, a bell at midday."""
    t = np.arange(int(n_days * DAY / step_s)) * step_s
    hour = (t % DAY) / 3600.0
    return np.where((hour > 6) & (hour < 20), peak * np.sin(np.pi * (hour - 6) / 14) ** 2, 0.0)


# ── The farm model (pv_farms_synthetic_generation.md) ──

def test_ar1_chunked_equals_continuous():
    whole, _ = dp.ar1(np.random.default_rng(5), 1000, 0.85, 0.08)
    rng = np.random.default_rng(5)
    parts, state = [], None
    for n in (1, 299, 300, 400):
        chunk, state = dp.ar1(rng, n, 0.85, 0.08, state)
        parts.append(chunk)
    assert np.allclose(np.concatenate(parts), whole)


def test_ar1_has_the_stationary_std_and_lag_one_correlation():
    eps, _ = dp.ar1(np.random.default_rng(1), 200_000, 0.85, 0.08)
    assert eps.std() == pytest.approx(0.08, rel=0.03)
    assert np.corrcoef(eps[:-1], eps[1:])[0, 1] == pytest.approx(0.85, abs=0.01)


def test_farm_shares_bounded_zero_at_night_reproducible_and_distinct():
    agg = _day_profile()
    prof = _profile(agg)
    cfg = DERGenerationConfig()
    shares, params = dp.farm_shares(prof, range(4), cfg, seed=11)
    assert shares.shape == (4, len(agg))
    assert shares.min() >= 0.0 and shares.max() <= 1.0                     # power within [0, capacity]
    assert np.all(shares[:, agg == 0] == 0.0)                              # zero when the base profile is zero
    again, _ = dp.farm_shares(prof, range(4), cfg, seed=11)
    assert np.array_equal(shares, again)                                   # same config and seed
    assert not np.allclose(shares[0], shares[1])                           # distinct farms
    assert all(0.9 <= p.performance <= 1.0 for p in params)
    assert len({p.performance for p in params}) == 4
    # The shared pattern survives: every farm peaks around the aggregate's peak.
    day = slice(0, int(DAY / 900))
    peak = np.argmax(agg[day])
    assert all(abs(np.argmax(s[day]) - peak) <= 8 for s in shares)
    # Plant k doesn't depend on which others are drawn with it.
    two, _ = dp.farm_shares(prof, [3, 1], cfg, seed=11)
    assert np.array_equal(two, shares[[3, 1]])


def test_farm_shares_use_one_reference_maximum():
    agg = _day_profile()
    small = dp.farm_shares(_profile(agg, ref_max=agg.max() * 2), [0], DERGenerationConfig(sigma=0.0), 3)[0]
    full = dp.farm_shares(_profile(agg), [0], DERGenerationConfig(sigma=0.0), 3)[0]
    assert np.allclose(small * 2, full)                                    # no per-day rescaling


def test_rho_is_per_15_minutes():
    prof = _profile(_day_profile(step_s=60.0), step_s=60.0)
    _, params = dp.farm_shares(prof, [0], DERGenerationConfig(), 1)
    assert params[0].rho_step == pytest.approx(0.85 ** (60 / 900))


# ── Time: alignment, interpolation, gaps ──

def test_window_shift_keeps_the_time_of_day():
    scen = np.array([0.0, 60.0, 120.0]) + 1_700_000_000.0                 # 2023-11-14 22:13:20 UTC
    start = 1_700_000_000.0 + 400 * DAY + 5 * 3600                         # 400 days and 5 h later
    assert dp.window_shift_s(scen, start, True) == 400 * DAY
    assert dp.window_shift_s(scen, start, False) == 400 * DAY + 5 * 3600
    assert dp.window_shift_s(scen, None, True) == 0.0


def test_locate_interpolates_between_steps():
    grid = np.arange(10) * 900.0
    j0, j1, w, used = dp.locate(np.array([900.0, 1200.0, 1800.0]), grid, np.ones(10, bool), 900.0)
    assert list(j0) == [1, 1, 2] and list(j1) == [1, 2, 2]
    assert w[1] == pytest.approx(1 / 3) and np.array_equal(used, [900.0, 1200.0, 1800.0])


def test_locate_borrows_the_same_clock_time_from_the_nearest_day_for_a_gap():
    grid = np.arange(0, 5 * DAY, 900.0)
    usable = np.ones(len(grid), bool)
    usable[(grid >= 2 * DAY) & (grid < 3 * DAY)] = False                   # day 2 missing
    t = 2 * DAY + 12 * 3600
    j0, j1, w, used = dp.locate(np.array([t]), grid, usable, 900.0)
    assert used[0] == t - DAY and grid[j0[0]] == t - DAY                   # earlier day first, same clock time


def test_locate_refuses_to_read_a_gap_as_zero():
    grid = np.arange(0, DAY, 900.0)
    with pytest.raises(ValueError, match="No generation data"):
        dp.locate(np.array([100 * DAY]), grid, np.ones(len(grid), bool), 900.0, max_days=5)


def test_interpolated_window_steps_finer_than_the_load():
    load = [datetime(2024, 1, 1) + timedelta(minutes=15 * k) for k in range(20)]
    ts, brackets = interpolated_window(load, 10, 60.0, start=datetime(2024, 1, 1, 1))
    assert ts[0] == datetime(2024, 1, 1, 1) and ts[1] - ts[0] == timedelta(minutes=1)
    t0, t1, w = brackets[ts[7]]
    assert (t0, t1) == (datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 1, 15)) and w == pytest.approx(7 / 15)
    with pytest.raises(ValueError, match="gap"):
        interpolated_window(load[:5] + load[10:], 60, 60.0, start=datetime(2024, 1, 1, 0, 30))


# ── The sampler ──

_NODES = ["SUB", "A", "B", "C", "D"]


def _topo():
    branches = [MVBranch(f"L{i}", _NODES[i - 1], _NODES[i], "c", 1000.0, 0.2, 0.1, 300.0) for i in range(1, 5)]
    t = MVGridTopology("g", 20.0, ["SUB"], {n: i for i, n in enumerate(_NODES)}, branches)
    t.node_coords = {n: (41.15, -8.61) for n in _NODES}
    return t


def _sampler(der, generation, n=40, p_max=4e6):
    cfg = PFDataGenRequest(historical_database_id="h", der=der).der
    return DERSampler(cfg, _topo(), _NODES, set(_NODES), p_max, n, 9, {"A": 400.0}, generation=generation)


def _group(**kw):
    return {"database_id": "gen", "count": 1, "capacity_mw": 1.0, **kw}


def test_aggregated_groups_follow_their_plant_share():
    prof = _profile(_day_profile(n_days=1)[:40])
    s = _sampler({"source": "replace", "layout": "per_run",
                  "groups": [_group(count=2, capacity_mw=0.5), _group(capacity_mw=2.0)]}, {"gen": prof})
    first = s.sample(0, "2024-06-01 00:00:00", {})
    assert [d.der_id for d in first.ders] == ["GEN-PV-01", "GEN-PV-02", "GEN-PV-03"]
    assert [d.s_rated_kva for d in first.ders] == [500.0, 500.0, 2000.0]
    assert [p.farm for p in s.plants] == [0, 1, 2]                          # numbered across groups
    shares, _ = dp.farm_shares(prof, range(3), DERGenerationConfig(), 9)
    for i in range(40):
        smp = s.sample(i, "2024-06-01 00:00:00", {})
        assert [d.node_id for d in smp.ders] == [d.node_id for d in first.ders]     # per_run: one placement
        for k, d in enumerate(smp.ders):
            assert smp.p_available[d.der_id] == pytest.approx(d.s_rated_kva * 1e3 * shares[k, i])
            assert 0.0 <= smp.p_available[d.der_id] <= d.s_rated_kva * 1e3
    # The groups fix the capacity, so the penetration is the same in every sample.
    assert first.capacity_w == 3e6 and first.penetration == pytest.approx(3e6 / 4e6)
    assert s.penetration == pytest.approx(0.75)


def test_add_counts_the_grids_own_der_in_the_installed_capacity():
    from gridarena.pf_datagen.grid_loader import MVDER
    topo = _topo()
    topo.ders = [MVDER("OWN", "B", "pv", 1000.0, 1000.0, "solar", 1.0, "fixed_pf")]
    cfg = PFDataGenRequest(historical_database_id="h", der={"source": "add", "groups": [_group()]}).der
    s = DERSampler(cfg, topo, _NODES, set(_NODES), 4e6, 5, 9, {}, generation={"gen": _profile(np.ones(5))})
    smp = s.sample(0, "t", {"OWN": 10.0})
    assert [d.der_id for d in smp.ders] == ["OWN", "GEN-PV-01"]
    assert smp.capacity_w == pytest.approx(2e6) and smp.penetration == pytest.approx(0.5)


def test_groups_can_follow_different_databases_and_controls():
    pv = _profile(_day_profile(n_days=1)[:40])
    wind = _profile(np.full(40, 3e9), database_id="wind_db", data_type="wind")
    s = _sampler({"source": "replace", "groups": [
        _group(count=3, capacity_mw=0.2, control="q_v"),
        {"database_id": "wind_db", "count": 2, "capacity_mw": 1.5, "control": "voltage", "placement": "feeder_end"},
    ]}, {"gen": pv, "wind_db": wind})
    for i in (0, 7, 20):
        smp = s.sample(i, "t", {})
        assert [d.der_id for d in smp.ders] == ["GEN-PV-01", "GEN-PV-02", "GEN-PV-03", "GEN-WIND-01", "GEN-WIND-02"]
        assert [d.control_mode for d in smp.ders] == ["q_v"] * 3 + ["voltage"] * 2
        assert all(0.99 <= d.v_set_pu <= 1.03 for d in smp.ders[3:])
        assert smp.location == "random+feeder_end"
    # Wind plants follow the wind aggregate (flat here): never above their capacity, never zero.
    wind_av = [s.sample(i, "t", {}).p_available["GEN-WIND-01"] for i in range(40)]
    assert max(wind_av) <= 1.5e6 and min(wind_av) > 0


def test_per_sample_layout_moves_the_plants_but_keeps_their_sizes():
    s = _sampler({"source": "replace", "layout": "per_sample", "groups": [_group(count=4, capacity_mw=0.3)]},
                 {"gen": _profile(np.ones(40))})
    layouts = {tuple(d.node_id for d in s.sample(i, "t", {}).ders) for i in range(20)}
    assert len(layouts) > 1
    assert all(sum(d.s_rated_kva for d in s.sample(i, "t", {}).ders) == pytest.approx(1200.0) for i in range(5))


def test_plants_never_go_behind_an_mv_lv_transformer():
    # A, B and C have 100 kVA transformers; their LV-side buses are nodes of the solved network too.
    topo = _topo()
    nodes = _NODES + ["TA_LV", "TB_LV", "TC_LV"]
    topo.node_id_to_index = {n: i for i, n in enumerate(nodes)}
    topo.branches += [MVBranch(f"T{n}", n, f"T{n}_LV", None, 0.0, 0.0, 0.0, 2.9, kind="transformer", rated_kva=100.0)
                      for n in "ABC"]
    for placement in ("random", "load", "clustered", "feeder_end"):
        cfg = PFDataGenRequest(historical_database_id="h", der={"source": "replace", "layout": "per_sample",
                                                                "groups": [_group(count=5, placement=placement)]}).der
        s = DERSampler(cfg, topo, nodes, set(nodes), 4e6, 60, 9, {"A": 100.0, "B": 100.0, "C": 100.0},
                       generation={"gen": _profile(np.ones(60))})
        assert not any(n.endswith("_LV") for n in s.eligible)
        placed = {d.node_id for i in range(60) for d in s.sample(i, "t", {}).ders}
        assert placed and not any(n.endswith("_LV") for n in placed), placement
        if placement == "load":
            assert placed <= {"A", "B", "C"}                       # weighted by the MV node's transformer kVA


def test_feeder_end_placement_favours_the_far_nodes():
    s = _sampler({"source": "replace", "groups": [_group(count=1, placement="feeder_end")]},
                 {"gen": _profile(np.ones(400))}, n=400)
    nodes = [s.sample(i, "t", {}).ders[0].node_id for i in range(400)]
    assert nodes.count("D") > nodes.count("A") * 2


def test_mixed_control_draws_from_the_types_mix():
    s = _sampler({"source": "replace", "groups": [_group(count=50, capacity_mw=0.1)]}, {"gen": _profile(np.ones(5))},
                 n=5)
    modes = {d.control_mode for d in s.sample(0, "t", {}).ders}
    assert modes <= {"q_v", "fixed_pf", "p_v"} and len(modes) > 1          # the PV mix
    hydro = _profile(np.ones(5), database_id="h", data_type="hydro")
    s = _sampler({"source": "replace", "groups": [{"database_id": "h", "count": 30, "capacity_mw": 0.1}]},
                 {"h": hydro}, n=5)
    assert {d.control_mode for d in s.sample(0, "t", {}).ders} <= {"fixed_pf", "voltage"}


def test_individual_groups_take_the_series_as_recorded():
    agg = _day_profile(n_days=1)[:40]
    prof = _profile(agg, n_series=3, data_type="wind")
    prof.values_w[1] *= 3.0
    prof.values_w[2] = 0.0                                                  # never produces: left out
    prof.ratings_w = prof.values_w.max(axis=1)
    s = _sampler({"source": "add", "groups": [{"database_id": "gen", "interpretation": "individual",
                                               "control": "fixed_pf"}]}, {"gen": prof})
    for i in (0, 25, 30):
        smp = s.sample(i, "2024-06-01 00:00:00", {})
        assert [d.der_type for d in smp.ders] == ["wind", "wind"]
        assert [smp.p_available[d.der_id] for d in smp.ders] == pytest.approx(list(prof.values_w[:2, i]))
        assert [d.s_rated_kva * 1e3 for d in smp.ders] == pytest.approx(list(prof.ratings_w[:2]))
    assert [p.series for p in s.plants] == [0, 1]


def test_each_run_draws_its_own_number_of_plants_in_the_range():
    from gridarena.pf_datagen.der_sampling import draw_counts
    from gridarena.pf_datagen.package import grid_seeds
    cfg = PFDataGenRequest(historical_database_id="h", der={"source": "replace", "groups": [
        {"database_id": "gen", "count_min": 2, "count_max": 9, "capacity_mw": 0.5},
        {"database_id": "gen", "interpretation": "individual"},
        _group(count=3)]}).der
    # A group of 40 grids: each grid's seed draws its own count, within the range and reproducibly.
    counts = [draw_counts(cfg.groups, s) for s in grid_seeds(123, 40)]
    first = [c[0] for c in counts]
    assert min(first) >= 2 and max(first) <= 9 and len(set(first)) > 3
    assert all(c[1] is None and c[2] == 3 for c in counts)                  # individual: the data; fixed: 3
    assert draw_counts(cfg.groups, 77) == draw_counts(cfg.groups, 77)
    # The groups are drawn independently (group g has its own stream).
    alone = PFDataGenRequest(historical_database_id="h", der={"source": "replace", "groups": [
        {"database_id": "gen", "count_min": 2, "count_max": 9, "capacity_mw": 0.5}]}).der
    assert [draw_counts(alone.groups, s)[0] for s in grid_seeds(123, 40)] == first
    # The sampler builds that many plants, and the capacity follows.
    s = DERSampler(cfg, _topo(), _NODES, set(_NODES), 4e6, 5, 77, {},
                   generation={"gen": _profile(np.ones(5), n_series=2)})
    n0 = s.counts[0]
    assert sum(1 for p in s.plants if p.group == 0) == n0 and sum(1 for p in s.plants if p.group == 2) == 3
    assert s.generated_capacity_w == pytest.approx(n0 * 0.5e6 + 2 * 0.5 + 3 * 1e6)


def test_a_group_can_draw_no_plants():
    s = _sampler({"source": "replace", "groups": [_group(count_min=0, count_max=0)]}, {"gen": _profile(np.ones(40))})
    smp = s.sample(0, "t", {})
    assert smp.ders == [] and smp.capacity_w == 0 and smp.location is None and smp.n_generated == 0


def test_group_and_config_validation():
    with pytest.raises(ValueError, match="der.groups"):
        DERConfig(source="add")
    assert DERConfig().groups == []                                         # the grid's own DER need none
    with pytest.raises(ValueError, match="capacity_mw"):
        DERConfig(source="add", groups=[{"database_id": "g", "count": 2}])
    with pytest.raises(ValueError, match="count_min"):
        DERConfig(source="add", groups=[{"database_id": "g", "count_min": 5, "count_max": 2, "capacity_mw": 1}])
    fixed = DERConfig(source="add", groups=[{"database_id": "g", "count": 4, "capacity_mw": 1}]).groups[0]
    assert (fixed.count_min, fixed.count_max) == (4, 4)                     # `count`: a fixed number
    DERConfig(source="add", groups=[{"database_id": "g", "interpretation": "individual"}])
    # A generating request in the earlier format says what replaced it.
    for old in ({"historical": {"database_id": "g"}}, {"penetration_min": 0.1, "penetration_max": 1}):
        with pytest.raises(ValueError, match="der.groups"):
            DERConfig(source="replace", **old)
    # Grid-only configs saved with the earlier defaults still load.
    assert DERConfig(source="grid", penetration_min=0, units_max=10, locations={"uniform": 1}).source == "grid"
    with pytest.raises(ValueError, match="generation profiles"):
        _sampler({"source": "add", "groups": [_group()]}, None)


def test_layout_follows_the_timeline_unless_set():
    def req(timeline="spread", **der):
        return PFDataGenRequest(historical_database_id="h", timeline=timeline,
                                der={"source": "add", "groups": [_group()], **der})
    assert req("window").der.layout == "per_run"
    assert req().der.layout == "per_sample"
    assert req("window", layout="per_sample").der.layout == "per_sample"
    assert PFDataGenRequest(historical_database_id="h").der.layout == "per_sample"


def test_database_ids_and_aggregated():
    cfg = DERConfig(source="add", groups=[_group(), {"database_id": "w", "interpretation": "individual"},
                                          _group(count=3)])
    assert cfg.database_ids == ["gen", "w"]
    assert cfg.aggregated("gen") and not cfg.aggregated("w")
    assert DERConfig(source="grid", groups=[_group()]).database_ids == []   # grid: nothing generated, nothing read


# ── Set-points: transformer taps within +/-N steps ──

def test_transformer_taps_within_n_steps_of_nominal():
    from gridarena.pf_datagen.schemas import SetpointsConfig
    from gridarena.pf_datagen.setpoints import SetpointSampler

    trafos = [MVBranch(f"T{k}", "A", f"T{k}_LV", None, 0.0, 0.0, 0.0, 0.0, kind="transformer", tap_min=-2,
                       tap_max=2, tap_pos=0, tap_step_pct=2.5) for k in range(30)]
    trafos.append(MVBranch("TP", "A", "TP_LV", None, 0.0, 0.0, 0.0, 0.0, kind="transformer", tap_min=2, tap_max=5,
                           tap_pos=3, tap_step_pct=2.5))
    full = SetpointSampler(SetpointsConfig(transformer_taps_resample=True), 1.0, ["SUB"], 10, 3)
    one = SetpointSampler(SetpointsConfig(transformer_taps_resample=True, transformer_tap_steps=1), 1.0,
                          ["SUB"], 10, 3)
    full_taps = [v for i in range(10) for v in full.transformer_taps(i, trafos).values()]
    taps = [one.transformer_taps(i, trafos) for i in range(10)]
    assert {v for t in taps for k, v in t.items() if k != "TP"} == {-1, 0, 1}
    assert {t["TP"] for t in taps} == {2}                                    # range beside 0: its nearest end
    assert min(full_taps) == -2 and max(full_taps) == 5


# ── Inputs: stored, reloaded, locked ──

def test_profiles_are_saved_with_the_inputs(tmp_path):
    profs = {"gen": _profile(_day_profile(n_days=1)[:10], n_series=2),
             "w": _profile(np.arange(10.0) + 1, database_id="w", data_type="wind")}
    dp.profiles_to_files(profs, tmp_path)
    back = dp.profiles_from_files(tmp_path)
    assert list(back) == ["gen", "w"]
    for k in profs:
        assert back[k].series_ids == profs[k].series_ids and np.array_equal(back[k].values_w, profs[k].values_w)
        assert np.array_equal(back[k].gen_times, profs[k].gen_times)
        assert back[k].reference_max_w == profs[k].reference_max_w and back[k].data_type == profs[k].data_type
    assert dp.profiles_from_files(tmp_path / "nothing_here") is None


def test_generation_settings_are_locked_in_a_package():
    from gridarena.pf_datagen.inputs import LOCKED_FIELDS, locked_values
    base = {"historical_database_id": "h"}
    req = PFDataGenRequest(**base, der={"source": "add", "groups": [
        _group(), {"database_id": "w", "interpretation": "individual"}], "generation": {"rho": 0.5}})
    assert "der.generation" in LOCKED_FIELDS
    locked = locked_values(req)["der.generation"]
    assert locked["databases"] == [["gen", True], ["w", False]] and locked["timezone"] == "UTC"
    assert "rho" not in locked                                              # a solve-side setting, editable
    assert locked_values(PFDataGenRequest(**base))["der.generation"] is None
    # Groups left over with the grid's own DER only don't read (or lock) a generation database.
    grid_only = PFDataGenRequest(**base, der={"source": "grid", "groups": [_group()]})
    assert locked_values(grid_only)["der.generation"] is None


# ── Load aggregation (load_aggregation_s) ──

def test_aggregate_timeline_makes_clock_aligned_intervals_and_drops_gaps():
    t0 = datetime(2024, 1, 1, 0, 0)
    ts = [t0 + timedelta(minutes=k) for k in range(60)]
    ts = [t for t in ts if not (32 <= t.minute < 40)] + [t0 + timedelta(minutes=k) for k in range(60, 66)]
    starts, members = aggregate_timeline(ts, 900.0)
    # 00:30 keeps 7 of 15 minutes (< half: a gap); 01:00 only 6 of 15.
    assert starts == [t0, t0 + timedelta(minutes=15), t0 + timedelta(minutes=45)]
    assert len(members[t0]) == 15 and members[t0][0] == t0
    assert interval_start(datetime(2024, 1, 1, 17, 53), 900) == datetime(2024, 1, 1, 17, 45)
    assert interval_start("2024-01-01 17:53:00", 3600) == datetime(2024, 1, 1, 17, 0)
    with pytest.raises(ValueError, match="finer than"):
        aggregate_timeline([t0 + timedelta(minutes=15 * k) for k in range(8)], 300.0)


def _aggregated_scenarios(count, timeline="spread", start=None, aggregation_s=900.0):
    """build_scenarios on a mocked 1-min Feeder database whose one series reads P = minute + 1 (no Q)."""
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=k) for k in range(60)]
    values = {("s", t): (float(k + 1), None) for k, t in enumerate(ts)}
    fetched = []

    def fetch(db_id, series, stamps, cursor):
        fetched.append(list(stamps))
        return {k: v for k, v in values.items() if k[1] in set(stamps)}
    from unittest.mock import MagicMock, patch
    with patch.object(scenario_builder, "get_historical_source_type", return_value="Feeder"),             patch.object(scenario_builder, "get_historical_timeline", return_value=ts),             patch.object(scenario_builder, "get_all_mv_connection_points", return_value=["P1"]),             patch.object(scenario_builder, "get_historical_series_ids", return_value=["s"]),             patch.object(scenario_builder, "fetch_series_values", side_effect=fetch):
        sc = scenario_builder.build_scenarios(
            "mv1", {"SUB": 0, "P1": 1}, count, 0.0, "db", 100, np.random.default_rng(1), MagicMock(),
            point_kva={"P1": 100.0}, shape_rng=np.random.default_rng(2), timeline=timeline, timeline_start=start,
            window_rng=np.random.default_rng(3), aggregation_s=aggregation_s)
    return sc, fetched


def test_aggregated_load_is_the_mean_of_each_interval():
    sc, fetched = _aggregated_scenarios(4)
    assert [s["timestamp"] for s in sc] == [f"2024-01-01 00:{m:02d}:00" for m in (0, 15, 30, 45)]
    assert len(fetched[0]) == 60                                     # every minute read, then averaged
    p = np.array([s["S"][1].real for s in sc])
    means = np.array([8.0, 23.0, 38.0, 53.0])                        # mean of minutes 1-15, 16-30, ...
    assert p / p[0] == pytest.approx(means / means[0])              # rating sizing keeps the shape
    assert all(s["S"][1].imag > 0 for s in sc)                       # Q from the drawn power factor


def test_an_aggregated_window_starts_at_the_interval_holding_its_start():
    from unittest.mock import patch
    ranges = []

    def interval_means(db_id, series, lo, hi, step_s, cursor):
        ranges.append((lo, hi, step_s))
        return {("s", datetime(2024, 1, 1, 0, m)): (float(m), None) for m in (0, 15, 30, 45)}
    with patch.object(scenario_builder, "fetch_interval_means", side_effect=interval_means):
        sc, _ = _aggregated_scenarios(2, "window", datetime(2024, 1, 1, 0, 20))
    assert [s["timestamp"] for s in sc] == ["2024-01-01 00:15:00", "2024-01-01 00:30:00"]
    # A window reads its records as one range, averaged in the database.
    assert ranges == [(datetime(2024, 1, 1, 0, 15), datetime(2024, 1, 1, 0, 44), 900.0)]
    assert sc[1]["S"][1].real / sc[0]["S"][1].real == pytest.approx(30 / 15)


def _window_level(scale_over, n=100):
    """A 10-step window at the end of a mocked Feeder database whose series reads 1 W in its
    first half and 3 W in its second: the point's P at each step, and the reference reads."""
    from unittest.mock import MagicMock, patch
    from gridarena.pf_datagen.schemas import LoadShapeConfig
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=k) for k in range(n)]
    values = {("s", t): (1.0 if k < n // 2 else 3.0, None) for k, t in enumerate(ts)}
    reads = []
    scenario_builder._REFERENCE_MEANS.clear()

    def fetch(db_id, series, stamps, cursor):
        reads.append(len(stamps))
        return {k: v for k, v in values.items() if k[1] in set(stamps)}
    with patch.object(scenario_builder, "get_historical_source_type", return_value="Feeder"),             patch.object(scenario_builder, "get_historical_timeline", return_value=ts),             patch.object(scenario_builder, "get_all_mv_connection_points", return_value=["P1"]),             patch.object(scenario_builder, "get_historical_series_ids", return_value=["s"]),             patch.object(scenario_builder, "fetch_series_values", side_effect=fetch):
        sc = scenario_builder.build_scenarios(
            "mv1", {"SUB": 0, "P1": 1}, 10, 0.0, "db", 100, np.random.default_rng(1), MagicMock(),
            load_shape=LoadShapeConfig(utilisation=0.5, scale_over=scale_over), point_kva={"P1": 100.0},
            shape_rng=np.random.default_rng(2), timeline="window", timeline_start=ts[-10], window_rng=None)
    return np.array([s["S"][1].real for s in sc]), reads


def test_rating_scale_over_the_database_keeps_a_windows_level():
    size = 0.5 * 100e3                                     # utilisation x kVA: the mean over the database
    p, reads = _window_level("database")
    assert p == pytest.approx(size * 3 / 2)                # the busy half: 1.5 x the database average
    assert reads == [10, 100]                              # the window, then the reference sample
    p, reads = _window_level("run")
    assert p == pytest.approx(size)                        # the old behaviour: every window averages the size
    assert reads == [10]


def test_reference_means_are_read_once_per_database():
    from unittest.mock import MagicMock, patch
    scenario_builder._REFERENCE_MEANS.clear()
    ts = [datetime(2024, 1, 1) + timedelta(minutes=k) for k in range(10)]
    vals = {(s, t): (-2.0 if s == "a" else 4.0, None) for s in "ab" for t in ts}
    with patch.object(scenario_builder, "fetch_series_values", return_value=vals) as fetch:
        m1 = scenario_builder.reference_means("db", ts, ["a", "b"], MagicMock())
        m2 = scenario_builder.reference_means("db", ts, ["b", "a"], MagicMock())
    assert m1 == m2 == {"a": 2.0, "b": 4.0} and fetch.call_count == 1


def test_peak_sizing_bounds_the_load_and_the_export_by_the_rating():
    from unittest.mock import MagicMock, patch
    from gridarena.pf_datagen.load_shape import series_scales
    from gridarena.pf_datagen.schemas import LoadShapeConfig
    # A feeder that mostly draws 1 W but exports 6 W one step in 100, and peaks at 3 W one step in 100.
    p = np.ones(1000)
    p[::100] = -6.0
    p[50::100] = 3.0
    ts = [datetime(2024, 1, 1) + timedelta(minutes=k) for k in range(len(p))]
    values = {("s", t): (float(v), None) for t, v in zip(ts, p)}
    assert series_scales(values, ["s"], ts, "mean")["s"] == pytest.approx(np.abs(p).mean())
    assert series_scales(values, ["s"], ts, "peak")["s"] == pytest.approx(np.percentile(np.abs(p), 99))

    def build(size_by):
        with patch.object(scenario_builder, "get_historical_source_type", return_value="Feeder"), \
                patch.object(scenario_builder, "get_historical_timeline", return_value=ts), \
                patch.object(scenario_builder, "get_all_mv_connection_points", return_value=["P1"]), \
                patch.object(scenario_builder, "get_historical_series_ids", return_value=["s"]), \
                patch.object(scenario_builder, "fetch_series_values",
                             side_effect=lambda d, s, stamps, c: {k: v for k, v in values.items() if k[1] in set(stamps)}):
            sc = scenario_builder.build_scenarios(
                "mv1", {"SUB": 0, "P1": 1}, len(ts), 0.0, "db", 10_000, np.random.default_rng(1), MagicMock(),
                load_shape=LoadShapeConfig(utilisation=0.8, size_by=size_by), point_kva={"P1": 100.0},
                shape_rng=np.random.default_rng(2))
        return np.array([s["S"][1].real for s in sc]) / 100e3          # P / kVA
    mean_based, peak_based = build("mean"), build("peak")
    assert np.abs(mean_based).mean() == pytest.approx(0.8)                     # the average is utilisation
    assert mean_based.min() < -4.0                                             # ...and the export runs away
    assert np.percentile(np.abs(peak_based), 99) == pytest.approx(0.8)        # the P99 is utilisation
    assert (np.abs(peak_based) <= 0.8 + 1e-9).mean() >= 0.99
    # Both divisors come from one read of the reference sample.
    scenario_builder._REFERENCE_MEANS.clear()
    with patch.object(scenario_builder, "fetch_series_values", return_value=values) as fetch:
        a = scenario_builder.reference_means("db", ts, ["s"], MagicMock(), "mean")
        b = scenario_builder.reference_means("db", ts, ["s"], MagicMock(), "peak")
    assert fetch.call_count == 1 and b["s"] > a["s"]
    assert PFDataGenRequest(historical_database_id="h").load_shape.size_by == "mean"


def test_feeder_counts_follow_the_kva_bands():
    from gridarena.pf_datagen.load_shape import draw_feeder_counts
    from gridarena.pf_datagen.schemas import FeederBand
    bands = [FeederBand(max_kva=100, min=1, max=2), FeederBand(max_kva=630, min=4, max=8)]
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = draw_feeder_counts(["A", "B", "C", "D"], {"A": 100.0, "B": 630.0, "C": 2000.0}, bands, 40, rng)
        assert 1 <= n["A"] <= 2 and 4 <= n["B"] <= 8 and 4 <= n["C"] <= 8   # above the last band: the last
        assert 4 <= n["D"] <= 8                                              # no kVA: the others' average (910)
    assert draw_feeder_counts(["B"], {"B": 630.0}, bands, 3, rng) == {"B": 3}  # capped at the database's series


def _combined(load_shape, kva, series, ts, values, connected=False):
    from unittest.mock import MagicMock, patch
    points = list(kva)
    scenario_builder._REFERENCE_MEANS.clear()
    with patch.object(scenario_builder, "get_historical_source_type", return_value="Feeder"), \
            patch.object(scenario_builder, "get_historical_timeline", return_value=ts), \
            patch.object(scenario_builder, "get_all_mv_connection_points", return_value=points), \
            patch.object(scenario_builder, "get_connected_load_points",
                         return_value=[{"node_id": p, "lv_grid_id": None} for p in points]), \
            patch.object(scenario_builder, "get_historical_series_ids", return_value=series), \
            patch.object(scenario_builder, "fetch_series_values",
                         side_effect=lambda d, s, stamps, c: {k: v for k, v in values.items() if k[1] in set(stamps)}):
        sc = scenario_builder.build_scenarios(
            "mv1", {"SUB": 0, **{p: i + 1 for i, p in enumerate(points)}}, len(ts), 0.0, "db", 10_000,
            np.random.default_rng(int(round(load_shape.utilisation * 1000))), MagicMock(), load_shape=load_shape, point_kva=kva,
            shape_rng=np.random.default_rng(2))
    return np.array([[s["S"][i + 1].real for i in range(len(points))] for s in sc])


def test_by_kva_sums_distinct_feeders_per_point():
    from gridarena.pf_datagen.schemas import FeederBand, LoadShapeConfig
    # Series k reads 10^k W, so a point's sum spells which series it took: a 2 would be a repeat.
    series = [f"s{k}" for k in range(6)]
    ts = [datetime(2024, 1, 1) + timedelta(minutes=m) for m in range(3)]
    values = {(s, t): (float(10 ** k), None) for k, s in enumerate(series) for t in ts}
    bands = [FeederBand(max_kva=100, min=2, max=2), FeederBand(max_kva=1000, min=5, max=5)]
    for u in np.linspace(0.1, 1.0, 20):                  # utilisation only seeds the draw here
        P = _combined(LoadShapeConfig(point_sizing="series", feeders_per_point="by_kva", feeder_bands=bands,
                                      utilisation=float(u)), {"P1": 100.0, "P2": 630.0}, series, ts, values)
        digits = [str(int(round(v))).zfill(6) for v in P[0]]
        assert set("".join(digits)) <= {"0", "1"}                            # no feeder twice in a point
        assert [d.count("1") for d in digits] == [2, 5]                     # the band's count


def test_by_kva_sizes_the_sum_by_its_own_peak():
    from gridarena.pf_datagen.schemas import LoadShapeConfig
    rng = np.random.default_rng(5)
    series = [f"s{k}" for k in range(8)]
    ts = [datetime(2024, 1, 1) + timedelta(minutes=m) for m in range(400)]
    raw = {s: rng.uniform(0, 50e3, len(ts)) * (1 + k) for k, s in enumerate(series)}
    values = {(s, t): (float(raw[s][j]), None) for s in series for j, t in enumerate(ts)}
    for scale_over in ("database", "run"):
        ls = LoadShapeConfig(utilisation=0.8, size_by="peak", scale_over=scale_over, feeders_per_point="by_kva")
        P = _combined(ls, {"P1": 630.0, "P2": 100.0}, series, ts, values)
        # 400 timestamps: the database reference sample is all of them, so both give the exact P99.
        assert np.percentile(np.abs(P[:, 0]), 99) == pytest.approx(0.8 * 630e3)
        assert np.percentile(np.abs(P[:, 1]), 99) == pytest.approx(0.8 * 100e3)
    assert PFDataGenRequest(historical_database_id="h").load_shape.feeders_per_point == "one"


def test_feeder_bands_must_increase():
    from pydantic import ValidationError
    from gridarena.pf_datagen.schemas import LoadShapeConfig
    with pytest.raises(ValidationError):
        LoadShapeConfig(feeder_bands=[{"max_kva": 630, "min": 1, "max": 2}, {"max_kva": 100, "min": 1, "max": 2}])
    with pytest.raises(ValidationError):
        LoadShapeConfig(feeder_bands=[{"max_kva": 630, "min": 3, "max": 2}])


def test_reference_timestamps_are_a_fixed_sample():
    from gridarena.pf_datagen.load_shape import reference_timestamps
    ts = list(range(100_000))
    a = reference_timestamps(ts, 2000)
    assert a == reference_timestamps(ts, 2000) and len(a) == 2000 and a == sorted(a)
    assert len({t % 96 for t in a}) == 96                  # every quarter-hour of a 15-min day, no aliasing
    assert reference_timestamps(ts[:50], 2000) == ts[:50]


def test_load_aggregation_validation_and_locking():
    base = {"historical_database_id": "h"}
    assert PFDataGenRequest(**base).load_aggregation_s is None
    assert PFDataGenRequest(**base, load_aggregation_s=900).load_aggregation_s == 900
    for bad in (30, 700, 100000):
        with pytest.raises(ValueError):
            PFDataGenRequest(**base, load_aggregation_s=bad)
    assert locked_values(PFDataGenRequest(**base, load_aggregation_s=900))["load_aggregation_s"] == 900


# ── Against the database: a generation database in UTC, a load in Lisbon time ──

GEN_DB = "test_der_hist_pv"


@pytest.fixture
def generation_db():
    import gridarena.database as db
    from gridarena.schemas import HistoricalDatabaseUpload
    start = datetime(2024, 6, 1)
    times = [start + timedelta(minutes=15 * k) for k in range(4 * 24 * 3)]  # 3 days, 15 min, UTC
    times = [t for t in times if not (t.date() == datetime(2024, 6, 2).date())]   # day 2 missing
    stamps = [t.isoformat() for t in times]
    conn, cur = db.get_db_connection()
    try:
        db.create_historical_data_tables(cur)
        cur.execute('DELETE FROM "HistoricalDatabase" WHERE database_id = %s', (GEN_DB,))
        conn.commit()
        db.insert_historical_data(HistoricalDatabaseUpload(
            database_id=GEN_DB, data_type="pv", timestamps=stamps,
            historicals=[{"series_id": "agg", "grid_level": "MV",
                          "values": [{"datetime": s, "power_active": 1000.0 * (t.hour * 60 + t.minute)}
                                     for s, t in zip(stamps, times)]}]), conn, cur)
        yield db
        cur.execute('DELETE FROM "HistoricalDatabase" WHERE database_id = %s', (GEN_DB,))
        conn.commit()
    finally:
        conn.close()


def test_build_profile_converts_clocks_interpolates_and_fills_gaps_from_another_day(generation_db):
    conn, cur = generation_db.get_db_connection()
    try:
        cfg = DERGenerationConfig(reference_year=2024)

        def build(c, stamps, aggregated=True):
            return dp.build_generation_profile(c, GEN_DB, aggregated, stamps, "Europe/Lisbon", cur)
        # Lisbon summer time is UTC+1: 13:07 local = 12:07 UTC, between 12:00 and 12:15.
        prof = build(cfg, ["2024-06-01 13:07:00", "2024-06-01 13:08:00"])
        assert prof.step_s == 900.0 and prof.n_fallback == 0
        assert prof.values_w[0] == pytest.approx([1000.0 * (12 * 60 + 7), 1000.0 * (12 * 60 + 8)])
        assert prof.reference_max_w == pytest.approx(1000.0 * (23 * 60 + 45))
        # A load day falling on the missing day 2 reads day 1 (nearest, earlier first) at the same clock time.
        gap = build(cfg, ["2024-06-02 13:07:00"])
        assert gap.n_fallback == 1 and gap.values_w[0, 0] == pytest.approx(1000.0 * (12 * 60 + 7))
        # A window matched to another start keeps the time of day (shift rounded to whole days).
        moved = build(cfg.model_copy(update={"generation_start": datetime(2024, 6, 3, 5, 0)}),
                      ["2024-01-10 13:07:00"])
        # Lisbon winter = UTC: 13:07 local -> 13:07 UTC, read on 3 June.
        assert moved.shift_s % DAY == 0 and moved.values_w[0, 0] == pytest.approx(1000.0 * (13 * 60 + 7))
        assert datetime.fromtimestamp(moved.gen_times[0], tz=timezone.utc).date() == datetime(2024, 6, 3).date()
        # Only an aggregated reading needs (and gets) the reference maximum.
        individual = build(cfg, ["2024-06-01 13:07:00"], aggregated=False)
        assert individual.reference_max_w is None
        assert individual.ratings_w[0] == pytest.approx(1000.0 * (23 * 60 + 45))
    finally:
        conn.close()


def test_aggregated_load_averages_a_finer_generation_over_each_interval(generation_db):
    conn, cur = generation_db.get_db_connection()
    try:
        cfg = DERGenerationConfig(reference_year=2024)

        def build(stamps, average_s):
            return dp.build_generation_profile(cfg, GEN_DB, True, stamps, "Europe/Lisbon", cur, average_s=average_s)
        # 13:00 Lisbon (summer) = 12:00 UTC: an hour averages the points 12:00, 12:15, 12:30 and 12:45.
        hourly = build(["2024-06-01 13:00:00", "2024-06-01 14:00:00"], 3600.0)
        assert hourly.values_w[0] == pytest.approx([1000.0 * 742.5, 1000.0 * 802.5])
        assert hourly.info["points_averaged"] == 4 and hourly.n_fallback == 0
        assert datetime.fromtimestamp(hourly.gen_times[0], tz=timezone.utc).hour == 12
        # The same step: read on the point itself. A coarser one: still interpolated at the start.
        same = build(["2024-06-01 13:15:00"], 900.0)
        assert same.values_w[0, 0] == pytest.approx(1000.0 * 735) and "points_averaged" not in same.info
        coarse = build(["2024-06-01 13:05:00"], 300.0)
        assert coarse.values_w[0, 0] == pytest.approx(1000.0 * 725)
        # An interval on the missing day borrows day 1, point by point.
        gap = build(["2024-06-02 13:00:00"], 3600.0)
        assert gap.n_fallback == 1 and gap.values_w[0, 0] == pytest.approx(1000.0 * 742.5)
    finally:
        conn.close()


def test_preview_puts_the_shifted_generation_on_the_loads_clock(generation_db):
    conn, cur = generation_db.get_db_connection()
    try:
        out = dp.generation_preview(None, [GEN_DB], datetime(2023, 6, 1), datetime(2024, 6, 1, 6, 0), True, "UTC",
                                    "Europe/Lisbon", 1.0, cur)
        assert out["load"] is None and out["shift_days"] == 366.0          # whole days (2024 is a leap year)
        g = out["generation"][0]
        assert g["data_type"] == "pv" and g["step_s"] == 900.0 and max(g["values"]) == pytest.approx(1.0)
        # 12:00 UTC on 1 June 2024 is 13:00 on the load's (Lisbon summer) clock, a year earlier.
        k = g["times"].index("2023-06-01T13:00:00")
        assert g["values"][k] * g["max_w"] == pytest.approx(1000.0 * 12 * 60)
        with pytest.raises(ValueError, match="load database"):
            dp.generation_preview(None, [GEN_DB], None, None, True, "UTC", "Europe/Lisbon", 1.0, cur)
    finally:
        conn.close()


def test_a_load_database_is_not_a_generation_source(generation_db):
    conn, cur = generation_db.get_db_connection()
    try:
        cur.execute('UPDATE "HistoricalDatabase" SET data_type = %s WHERE database_id = %s', ("load", GEN_DB))
        with pytest.raises(ValueError, match="holds load"):
            dp.generation_database(GEN_DB, cur)
        conn.rollback()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _fresh_reference_means():
    """The shape divisors are cached per process: start every test without them."""
    importlib.import_module("gridarena.pf_datagen.scenario_builder")._REFERENCE_MEANS.clear()
