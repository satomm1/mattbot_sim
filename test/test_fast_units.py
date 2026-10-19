"""Unit tests for the fast simulator's own parts (mattbot_sim/fast); no ROS needed."""

import math
import os

import numpy as np
import pytest

pytest.importorskip("mattbot_sim.fast.imports")
from mattbot_sim.fast import imports  # noqa: E402,F401

nav_utils = pytest.importorskip("navigation_utils")
from mattbot_sim.fast.fleet import FleetLedger  # noqa: E402
from mattbot_sim.fast.mapper import Mapper  # noqa: E402
from mattbot_sim.fast.navigator import reference_time  # noqa: E402
from mattbot_sim.fast.params import FastParams  # noqa: E402
from mattbot_sim.fast.sweep import aggregate, flatten  # noqa: E402
from mattbot_sim.fast.worldview import FleetWorld  # noqa: E402
from mattbot_sim.perception import Detection  # noqa: E402
from mattbot_sim.world import GridMap, SimObject  # noqa: E402


def test_reference_time_matches_the_tracker():
    from navigation_utils.trackers import TrajectoryTracker

    tr = TrajectoryTracker(2, 2, 2.3, 2.3, soft_start_sec=1.5)
    for t in (0.0, 0.4, 1.0, 1.5, 3.0, 40.0):
        assert reference_time(t, 1.5) == pytest.approx(tr._reference_time(t))


def test_params_update_converts_types_and_ignores_unknown():
    p = FastParams().update({"observe_detour": "true", "observe_min_absent_frames": "6.0", "speed": "4"})
    assert p.observe_detour is True and p.observe_min_absent_frames == 6
    assert FastParams().update({"observe": False}).observe is False


def det(cls, x, y, w=0.4):
    return Detection(cls, x, y, w, 0.9, 0, 0, 0, 0)


def mapper():
    return Mapper((100, 100), (0.0, 0.0), 0.05)


def test_mapper_confirms_on_the_seventh_detection_and_blocks_out():
    m = mapper()
    for k in range(6):
        assert m.frame(float(k), "TRACK", [det("chair", 2.0, 2.0)]) == []
    assert m.frame(6.0, "TRACK", [det("chair", 2.0, 2.0)]) == [("chair", 2.0, 2.0, 0.4)]
    assert m.is_blocked(2.0, 2.0) and not m.is_blocked(3.0, 3.0)
    assert m.frame(7.0, "TRACK", [det("chair", 2.05, 2.0)]) == []  # within a width of a confirmed object


def test_mapper_ignores_unstable_modes_and_classes():
    m = mapper()
    for k in range(10):
        m.frame(float(k), "PARK_HEADING", [det("chair", 2.0, 2.0)])
        m.frame(float(k), "TRACK", [det("person", 1.0, 1.0)])
    assert m.confirmed == [] and all(p[6] != "person" for p in m.proposals)


def test_mapper_blockout_expires_only_while_idle_then_reconfirms():
    m = mapper()
    for k in range(7):
        m.frame(float(k), "TRACK", [det("cone", 2.0, 2.0)])
    m.expire(100.0, "TRACK")
    assert m.is_blocked(2.0, 2.0)  # not IDLE: kept
    m.expire(100.0, "IDLE")
    assert not m.is_blocked(2.0, 2.0) and m.confirmed == []
    # Confirmed again (which refreshes the ledger belief). As in the mapper, the leftover proposals from the
    # first confirmation (every non-confirming match appended one) are still there, so it is immediate.
    assert m.frame(101.0, "IDLE", [det("cone", 2.0, 2.0)]) == [("cone", 2.0, 2.0, 0.4)]


def test_fleet_ledger_merges_removes_and_decays():
    L = FleetLedger(hold_s=20.0, decay_s=40.0)
    a = L.confirm(1, 0.0, "chair", 1.0, 1.0, 0.4)
    assert L.confirm(2, 10.0, "chair", 1.3, 1.0, 0.4) == a  # same object, another robot
    (b,) = L.beliefs(30.0)
    assert b.object_id == a and b.belief == 1.0 and b.x == pytest.approx(1.15)  # last seen at 10 s
    assert L.beliefs(50.0)[0].belief == pytest.approx(0.5)
    assert L.remove_near(2, 60.0, "chair", 1.1, 1.0, 0.4) == a and L.beliefs(61.0) == []
    assert L.remove_near(2, 61.0, "chair", 1.1, 1.0, 0.4) is None


def test_other_robots_block_line_of_sight_but_not_the_observer():
    grid = GridMap(np.zeros((100, 100), dtype=np.int16), 0.1, 0.0, 0.0)
    target = SimObject("t", "cone", 8.0, 5.0, 0.3)
    w = FleetWorld(grid, [target])
    w.robot_discs = {1: (2.0, 5.0, 0.2), 2: (5.0, 5.0, 0.2)}
    w.observer = 1
    assert not w.line_of_sight(2.0, 5.0, target)  # robot 2 is in between
    w.robot_discs[2] = (5.0, 7.0, 0.2)
    assert w.line_of_sight(2.0, 5.0, target)  # the observer's own disc does not count


def test_flatten_and_aggregate():
    flat = flatten({"a": 1, "b": {"c": 2.5, "d": [1, 2]}, "e": True, "f": "x"})
    assert flat == {"a": 1, "b.c": 2.5, "b.d": 2, "e": 1, "f": "x"}
    rows = [{"n": 1, "seed": s, "robot_id": 9, "v": float(s)} for s in (0, 2)]
    (agg,) = aggregate(rows, ["n"])
    assert agg["runs"] == 2 and agg["v.mean"] == 1.0 and agg["v.std"] == 1.0


def test_gif_and_png(tmp_path):
    import os

    from mattbot_sim.fast import imports as fast_imports

    if not os.path.isdir(fast_imports.MAP_JSON_DIR):
        pytest.skip("mattbot_mcl/map_json not found")
    from PIL import Image

    from mattbot_sim.fast import viz
    from mattbot_sim.fast.sim import FastSim

    sim = FastSim(os.path.join(os.path.dirname(__file__), "..", "scenarios", "remove_one.yaml"), duration=60)
    sim.frames, sim.frame_every = [], 5.0
    sim.run()
    assert len(sim.frames) == 12  # t = 0, 5, ..., 55
    gif, png = str(tmp_path / "run.gif"), str(tmp_path / "run.png")
    viz.render_gif(sim, sim.frames, gif)
    viz.render_png(sim, sim.frames, png)
    assert Image.open(gif).n_frames == 12 and os.path.getsize(png) > 1000


# ---------- Scout decisions ----------


def test_scout_params_and_idle_gate():
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    p = FastParams().update({"observe_scout": "true", "observe_scout_fixed_s": "12", "observe_scout_min_idle_s": 10})
    assert p.observe_scout is True and p.observe_scout_fixed_s == 12.0 and p.observe_scout_min_idle_s == 10.0
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "scout_present.yaml")
    sim = FastSim(path, seed=0, duration=30.0)
    sim.run()
    s = sim.summaries()[next(iter(sim.robots))]
    # The robot reaches its idle pose after a few seconds: idle decisions only after 10 s of IDLE, no GO
    # yet (the cone's belief is still about 1 at 30 s)
    assert 0 < s["scout_decisions"] <= (30.0 - 10.0) / 5.0
    assert s["scout_go_decisions"] == 0
    r = next(iter(sim.robots.values()))
    d = r.planner.evaluate_scout((30.0, 18.95, 3.14159), [], sim.now, True)
    assert d == {"idle": True, "go": False, "candidates": []}  # no objects: no candidates


def _mode_trace(sim, until_s, on_step=None):
    """[(t, mode)] of the first robot's mode changes while stepping sim to until_s."""
    r = next(iter(sim.robots.values()))
    trace = [(sim.now, r.nav.mode)]
    sim.orchestrator.on_start(sim)
    while sim.now < until_s:
        if on_step is not None:
            on_step(sim, r)
        sim.step()
        if r.nav.mode != trace[-1][1]:
            trace.append((sim.now, r.nav.mode))
    return r, trace


def _contains_in_order(modes, expected):
    it = iter(modes)
    return all(any(m == e for m in it) for e in expected)


def test_endpoint_checks_at_goal_then_at_start(tmp_path):
    import yaml
    from mattbot_sim.fast.sim import FastSim

    # endpoint_removed without the removal, and a long wait at the east end: the cone (seen only from there) is
    # checked on arrival (AT_GOAL), and again before leaving once that check's cooldown and hold are over (AT_START)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "endpoint_removed.yaml")
    with open(path) as f:
        sc = yaml.safe_load(f)
    sc["events"], sc["expect"] = [], {}
    sc["waypoints"][0]["pause_s"] = 90.0
    variant = tmp_path / "endpoint_wait.yaml"
    variant.write_text(yaml.safe_dump(sc))
    sim = FastSim(str(variant), seed=0, duration=200.0)
    r, trace = _mode_trace(sim, 200.0)
    modes = [m for _t, m in trace]
    # Goal check: park, look, then the final heading turn
    assert _contains_in_order(modes, ["TRACK", "PARK_POSE", "OBSERVE_TURN", "OBSERVE_DWELL", "PARK_HEADING", "IDLE"])
    # Start check: look before leaving the east end, then ALIGN and TRACK
    i = max(k for k, (t, m) in enumerate(trace) if m == "OBSERVE_DWELL")
    assert [m for _t, m in trace[i - 2:i + 3]] == ["IDLE", "OBSERVE_TURN", "OBSERVE_DWELL", "ALIGN", "TRACK"]
    s = r.scorer.summary(sim.now)
    assert s["observation_stops_by_kind"]["AT_GOAL"]["ENDED"] >= 1
    assert s["observation_stops_by_kind"]["AT_START"]["ENDED"] == 1


def test_new_goal_during_a_goal_check_aborts_it():
    from mattbot_sim.fast.sim import FastSim

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "endpoint_removed.yaml")
    sim = FastSim(path, seed=0, duration=120.0)
    sent = []

    def on_step(sim, r):
        if not sent and r.nav.mode == "OBSERVE_DWELL":
            sent.append(sim.now)
            r.nav.set_goal(30.0, 18.95, 0.0)

    r, trace = _mode_trace(sim, 90.0, on_step)
    assert sent
    after = [m for t, m in trace if t >= sent[0]]
    assert "PARK_HEADING" not in after and "TRACK" in after  # left for the new goal without finishing the park
    s = r.scorer.summary(sim.now)
    assert s["observation_stops_by_kind"]["AT_GOAL"]["ABORTED"] == 1


def _detour_sim(**params):
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios", "detour_present.yaml")
    sim = FastSim(path, params=FastParams().update(params) if params else None, seed=0, duration=300.0)
    sim.orchestrator.on_start(sim)
    return sim, next(iter(sim.robots.values()))


def _step_until(sim, cond, limit_s):
    while sim.now < limit_s and not cond():
        sim.step()
    return cond()


def test_pending_detour_yields_and_comes_back_when_the_winner_withdraws():
    from dds_utils.intents import Intent

    sim, r = _detour_sim()
    assert _step_until(sim, lambda: any(e["detour"] for e in r.nav.stops), 300.0)
    entry = next(e for e in r.nav.stops if e["detour"])
    oid = entry["stop"].object_ids[0]
    # Robot 50 announces an earlier check of the same object: the pending detour is marked, not announced
    sim.intents.announce(50, [Intent(oid, 50, 0, sim.now + 2.0, sim.now + 120.0)], sim.now)
    assert _step_until(sim, lambda: entry.get("yielded") == 50, sim.now + 5.0)
    # Robot 50 withdraws it (an empty set) before the leave point: the detour is back and is driven
    sim.intents.announce(50, [], sim.now)
    assert _step_until(sim, lambda: entry.get("yielded") is None, sim.now + 5.0)
    assert _step_until(sim, lambda: r.nav.detour is not None, sim.now + 120.0)
    assert r.scorer.summary(sim.now)["detours_yielded"] == 0


def test_active_detour_yields_mid_route_and_resumes_the_goal():
    from dds_utils.intents import Intent

    sim, r = _detour_sim(observe_intent_commit_s=0.0)  # the detour is shorter than the default commit window
    assert _step_until(sim, lambda: r.nav.detour is not None and r.nav.mode == "TRACK", 300.0)
    oid = r.nav.detour["stop"].object_ids[0]
    goal = r.nav.goal
    sim.intents.announce(50, [Intent(oid, 50, 0, sim.now, sim.now + 60.0, committed=True)], sim.now)
    assert _step_until(sim, lambda: r.nav.detour is None, sim.now + 5.0)
    assert r.nav.goal == goal and r.nav.yielded == {oid: 50}
    assert r.scorer.summary(sim.now)["detours_yielded"] == 1
    # The yielded object is not offered again while robot 50 still plans the check
    assert oid in r.nav.live_yielded()
