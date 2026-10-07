"""Unit tests for mattbot_sim/scoring.py and scenario.py (no ROS needed)."""

import os

import pytest

from mattbot_sim.scenario import load_scenario, parse_event
from mattbot_sim.scoring import RateTracker, RunScorer, check_expectations, kind_name

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
    assert sc.objects and sc.waypoints
    assert sc.events or sc.expect  # something to score
    ids = {o.object_id for o in sc.objects}
    assert {e.get("remove") or e.get("move") for e in sc.events} <= ids
    assert set(sc.known_from_peer) <= ids and set(sc.detour) <= ids
    # Every expectation names a real summary field (values do not matter here)
    checks = check_expectations(RunScorer().summary(0.0), sc.expect)
    assert not [c for c in checks if c.get("error")]


def test_scenario_detour_fields(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "map: m\nstart: {x: 0, y: 0}\npeer_id: 7\n"
        "objects:\n  - {id: a, class: cone, x: 1, y: 2, known_from_peer: true, detour: expected}\n"
        "  - {id: b, class: cone, x: 3, y: 4}\n"
        "expect: {min_detour_stops: 1, false_removals: 0}\n")
    sc = load_scenario(str(path))
    assert sc.peer_id == 7 and sc.known_from_peer == ["a"] and sc.detour == {"a": "expected"}
    assert sc.expect == {"min_detour_stops": 1.0, "false_removals": 0.0}
    path.write_text("map: m\nstart: {x: 0, y: 0}\nobjects:\n  - {id: a, class: cone, x: 1, y: 2, detour: maybe}\n")
    with pytest.raises(ValueError):
        load_scenario(str(path))
    path.write_text("map: m\nstart: {x: 0, y: 0}\n")
    assert load_scenario(str(path)).peer_id == 98


def test_stops_and_outcomes_by_kind():
    s = RunScorer()
    s.set_ground_truth([placement("c", "cone", 1.0, 1.0, 0.0)])
    s.update_ledger(1.0, [("a", "cone", 1.0, 1.0)])
    line = s.add_stop_event("STARTED", "DETOUR", "a", 10.0, robot_xy=(3.0, 4.0), distance=3.2)
    assert line.startswith("DETOUR stop STARTED for a at 3.2 m from (3.00, 4.00)")
    s.add_stop_event("ENDED", "DETOUR", "a", 14.0)
    s.add_result(14.5, "a", "cone", "PRESENT", "20 matching frames", window_end=14.0)
    s.add_stop_event("STARTED", "OPPORTUNISTIC", "a", 50.0)
    s.add_stop_event("ABORTED", "OPPORTUNISTIC", "a", 51.0)
    out = s.summary(60.0)
    assert out["observation_stops"] == {"STARTED": 2, "ENDED": 1, "ABORTED": 1}
    assert out["observation_stops_by_kind"]["DETOUR"] == {"STARTED": 1, "ENDED": 1, "ABORTED": 0}
    assert out["observation_stops_by_kind"]["OPPORTUNISTIC"]["ABORTED"] == 1
    assert out["detour_stops"] == 1
    assert out["observation_outcomes_by_kind"]["DETOUR"]["PRESENT"] == 1
    assert out["observation_outcomes_by_kind"]["OPPORTUNISTIC"]["PRESENT"] == 0
    assert s.results[0]["kind"] == "DETOUR"
    assert kind_name(1) == "DETOUR" and kind_name(0) == "OPPORTUNISTIC" and kind_name(7) == "7"


def test_detour_lifecycle_from_navigator_log():
    s = RunScorer()
    # The navigator's own messages (localize_and_navigate.py), with its log prefix
    assert s.add_nav_log("[Navigator] Detour to (26.10, 21.50) to check 98-1-1 (est. 24 s)")
    assert s.add_nav_log("[Navigator] Reached detour viewpoint")
    assert s.add_nav_log("[Navigator] Detour done; replanning to the goal")
    assert s.add_nav_log("[Navigator] Detour to check 98-1-1 abandoned (took too long); replanning to the goal")
    assert s.add_nav_log("[Navigator] Detour cancelled (new goal)")
    assert s.add_nav_log("[Navigator] Planning Succeeded") is None
    s.add_runner_event("resend goal 1 2.2 m short")
    s.add_runner_event("give_up goal 1 2.2 m short")
    out = s.summary(0.0)
    assert (out["detours_started"], out["detours_reached"], out["detours_done"]) == (1, 1, 1)
    assert (out["detours_abandoned"], out["detours_cancelled"]) == (1, 1)
    assert out["detour_abandon_reasons"] == ["abandoned: took too long", "cancelled: new goal"]
    assert (out["goal_resends"], out["goals_given_up"]) == (1, 1)


def test_check_expectations():
    summary = {"detour_stops": 2, "false_removals": 0, "stale_ledger_objects": [{"object_id": "a"}],
               "observation_outcomes_by_kind": {"DETOUR": {"ABSENT": 1}}, "removal_latency_s": None}
    checks = {c["key"]: c for c in check_expectations(summary, {
        "min_detour_stops": 1, "max_detour_stops": 1, "false_removals": 0, "stale_ledger_objects": 0,
        "min_observation_outcomes_by_kind.DETOUR.ABSENT": 1, "removal_latency_s": 0, "nope": 1})}
    assert checks["min_detour_stops"]["ok"] and not checks["max_detour_stops"]["ok"]
    assert checks["false_removals"]["ok"]
    assert not checks["stale_ledger_objects"]["ok"] and checks["stale_ledger_objects"]["actual"] == 1  # list -> count
    assert checks["min_observation_outcomes_by_kind.DETOUR.ABSENT"]["ok"]
    assert not checks["removal_latency_s"]["ok"]  # None is never a pass
    assert not checks["nope"]["ok"] and "error" in checks["nope"]


def test_rate_tracker():
    r = RateTracker(10.0, window_s=5.0)
    assert r.ok() is None and r.summary()["mean_hz"] is None
    for k in range(101):  # 10 Hz for 10 s
        r.tick(100.0 + 0.1 * k)
    assert r.summary()["mean_hz"] == pytest.approx(10.0) and r.ok()
    slow = RateTracker(10.0, window_s=5.0)
    for k in range(51):  # 5 Hz: the node did not keep up
        slow.tick(100.0 + 0.2 * k)
    assert slow.ok() is False and slow.summary()["min_window_hz"] == pytest.approx(5.0)


def test_fleet_scenario_selects_robot(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(
        "map: m\nfleet_start_s: 45\nloop: true\nexpect: {false_removals: 0, goal_resends: 0}\n"
        "launch_args: {observe_detour: true, speed: 2}\n"
        "objects:\n  - {id: a, class: cone, x: 1, y: 2}\n"
        "robots:\n"
        "  - {id: 91, start: {x: 0, y: 0}, waypoints: [{x: 5, y: 0, pause_s: 9}, {x: 0, y: 0}],\n"
        "     expect: {goal_resends: 2}}\n"
        "  - {id: 92, start: {x: 9, y: 0}, loop: false, waypoints: [{x: 12, y: 0}]}\n")
    first = load_scenario(str(path))
    assert first.is_fleet and first.robot_id == 91 and first.robot_ids == [91, 92]
    assert first.fleet_start_s == 45.0 and first.waypoint_pauses == [9.0, None]
    assert first.expect == {"false_removals": 0.0, "goal_resends": 2.0}  # robot keys win
    assert first.launch_args == {"observe_detour": "true", "speed": "2"}
    second = load_scenario(str(path), 92)
    assert (second.start.x, second.loop, len(second.waypoints)) == (9.0, False, 1)
    assert second.expect == {"false_removals": 0.0, "goal_resends": 0.0}
    with pytest.raises(ValueError):
        load_scenario(str(path), 7)
    assert load_scenario(str(path), None).robot_id == 91
    # A single-robot scenario ignores the robot id
    single = os.path.join(SCENARIOS, "remove_one.yaml")
    assert not load_scenario(single, 91).is_fleet


def test_fleet_scenario_validation(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text("map: m\nstart: {x: 0, y: 0}\nrobots:\n  - {id: 1, start: {x: 0, y: 0}}\n")
    with pytest.raises(ValueError):  # start belongs to each robot
        load_scenario(str(path))
    path.write_text("map: m\nrobots:\n  - {id: 1, start: {x: 0, y: 0}}\n  - {id: 1, start: {x: 1, y: 0}}\n")
    with pytest.raises(ValueError):  # duplicate ids
        load_scenario(str(path))
