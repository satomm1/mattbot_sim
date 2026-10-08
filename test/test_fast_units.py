"""Unit tests for the fast simulator's own parts (mattbot_sim/fast); no ROS needed."""

import math

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
