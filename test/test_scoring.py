"""Unit tests for mattbot_sim/scoring.py and scenario.py (no ROS needed)."""

import os

import pytest

from mattbot_sim.scenario import load_scenario, parse_event
from mattbot_sim.scoring import RunScorer

SCENARIOS = os.path.join(os.path.dirname(__file__), "..", "scenarios")


def placement(oid, cls, x, y, t0, t1=None):
    return {"object_id": oid, "class_name": cls, "x": x, "y": y, "width": 0.5, "t_start": t0, "t_end": t1}


def test_correct_removal_with_latency():
    s = RunScorer()
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0)])
    s.update_ledger(10.0, [("obj-a", "chair", 1.2, 1.0)])
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0, 50.0)])
    s.update_ledger(80.0, [])
    out = s.summary(90.0)
    assert out["correct_removals"] == 1 and out["false_removals"] == 0
    assert out["removal_latency_s"]["mean"] == pytest.approx(30.0)
    assert out["blockouts_still_present"] == ["obj-a"]
    s.update_blockouts(85.0, lambda x, y: False)
    assert s.summary(90.0)["blockout_clear_after_s"] == [pytest.approx(5.0)]


def test_false_removal_and_stale():
    s = RunScorer()
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0), placement("k", "cone", 5.0, 5.0, 0.0, 20.0)])
    s.update_ledger(10.0, [("a", "chair", 1.0, 1.0), ("b", "cone", 5.0, 5.0)])
    s.update_ledger(30.0, [("b", "cone", 5.0, 5.0)])
    out = s.summary(60.0)
    assert out["false_removals"] == 1
    assert out["stale_ledger_objects"] == [{"object_id": "b", "class_name": "cone", "gone_for_s": 40.0}]


def test_moved_object_counts_as_removed_at_old_spot():
    s = RunScorer()
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0, 20.0), placement("c", "chair", 4.0, 1.0, 20.0)])
    s.update_ledger(10.0, [("a", "chair", 1.0, 1.0)])
    s.update_ledger(40.0, [("b", "chair", 4.0, 1.0)])
    out = s.summary(50.0)
    assert out["correct_removals"] == 1 and out["spurious_ledger_objects"] == 0


def test_spurious_and_result_verdicts():
    s = RunScorer()
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0, 20.0)])
    s.update_ledger(10.0, [("a", "chair", 1.0, 1.0), ("z", "chair", 9.0, 9.0)])
    s.add_result(30.0, "a", "chair", "ABSENT", "clear", window_end=30.0)
    s.add_result(31.0, "a", "chair", "PRESENT", "seen", window_end=31.0)
    s.add_result(32.0, "a", "chair", "INCONCLUSIVE", "occluded", window_end=32.0)
    out = s.summary(40.0)
    assert out["spurious_ledger_objects"] == 1
    assert out["observation_outcomes"] == {"PRESENT": 1, "ABSENT": 1, "INCONCLUSIVE": 1}
    assert out["wrong_observation_outcomes"] == 1


def test_result_spanning_a_change_is_not_scored():
    s = RunScorer()
    s.set_ground_truth([placement("c", "chair", 1.0, 1.0, 0.0, 20.0)])
    s.update_ledger(10.0, [("a", "chair", 1.0, 1.0)])
    s.add_result(22.0, "a", "chair", "PRESENT", "2 frames", window_start=18.0, window_end=22.0)
    assert s.results[-1]["verdict"].startswith("n/a")
    assert s.summary(30.0)["wrong_observation_outcomes"] == 0


def test_parse_event_validation():
    assert parse_event({"remove": "c"}) == {"remove": "c"}
    assert parse_event({"at": 3, "move": "c", "x": 1, "y": 2}) == {"at": 3.0, "move": "c", "x": 1.0, "y": 2.0}
    with pytest.raises(ValueError):
        parse_event({"remove": "c", "restore": "c"})
    with pytest.raises(ValueError):
        parse_event({"move": "c", "x": 1})


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(SCENARIOS) if f.endswith(".yaml")))
def test_shipped_scenarios_load(name):
    sc = load_scenario(os.path.join(SCENARIOS, name))
    assert sc.objects and sc.waypoints and sc.events
    assert {e.get("remove") or e.get("move") for e in sc.events} <= {o.object_id for o in sc.objects}
