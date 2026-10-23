"""Experiments (mattbot_sim/experiment): instances, task orchestration, odometry, scout execution and the report."""

import json
import math
import os

import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")
SPEC = os.path.join(PKG, "experiments", "pilot.yaml")

pytest.importorskip("scipy")
from mattbot_sim.fast import imports  # noqa: E402,F401

if not os.path.isdir(imports.MAP_JSON_DIR):
    pytest.skip("mattbot_mcl/map_json not found", allow_module_level=True)


@pytest.fixture(scope="module")
def setup():
    from mattbot_sim.experiment.generate import setup as setup_spec
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(SPEC)
    ctx, landmarks, cands, conn = setup_spec(spec)  # raises if the planner cannot reach a landmark
    return spec, ctx, landmarks, cands, conn


def test_instances_are_seeded_and_shared(setup):
    from mattbot_sim.experiment.generate import make_instance

    spec, _ctx, landmarks, cands, conn = setup
    a, b, c = (make_instance(spec, s, cands, landmarks, conn) for s in (3, 3, 4))
    assert a == b and a != c  # the same instance for every condition of a seed
    assert len(a["tasks"]) == spec.tasks["count"] and len(a["obstacles"]) == spec.obstacles["count"]
    times = [e["at"] for e in a["events"]]
    assert times == sorted(times)  # FastSim pops events in time order
    for t1, t2 in zip(a["tasks"], a["tasks"][1:]):
        assert t1["release_s"] <= t2["release_s"] and t1["landmark"] != t2["landmark"]
    obs = a["obstacles"]
    for i, o in enumerate(obs):
        assert o["I_o"] >= spec.obstacles["min_importance_m"] and o["remove_s"] > o["appear_s"]
        for p in obs[i + 1:]:
            assert math.hypot(o["x"] - p["x"], o["y"] - p["y"]) >= spec.obstacles["min_separation_m"]


def test_candidates_never_cut_a_landmark_off(setup):
    spec, ctx, landmarks, cands, conn = setup
    rng = np.random.default_rng(0)
    for k in rng.choice(len(cands), size=min(25, len(cands)), replace=False):
        c = cands[int(k)]
        assert conn.keeps_connected(c["x"], c["y"], spec.obstacles["width_m"])
    # All of an instance's obstacles together keep every landmark connected (each alone is not enough)
    from mattbot_sim.experiment.generate import make_instance

    for seed in range(5):
        inst = make_instance(spec, seed, cands, landmarks, conn)
        assert conn.keeps_connected_all([(o["x"], o["y"], o["width"]) for o in inst["obstacles"]])
    # A footprint over a dead-end room's only door cuts its landmark off
    x, y, _th = landmarks["room_south"]
    assert not conn.keeps_connected(x, y + 3.0, 2.0)


def _tiny_scenario(tmp_path, events=(), robots=((91, 28.0, 18.95, 0.0),)):
    d = {"name": "tiny", "map": "current_map", "duration": 600, "objects": [], "events": list(events),
         "robots": [{"id": i, "start": {"x": x, "y": y, "theta": th}, "loop": False} for i, x, y, th in robots]}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(d))
    return str(path)


def test_task_orchestrator_completes_tasks_and_odometer_counts_distance(tmp_path):
    from mattbot_sim.experiment.orchestrator import TaskOrchestrator
    from mattbot_sim.fast.sim import FastSim

    tasks = [{"id": 0, "release_s": 0.0, "landmark": "a", "x": 40.0, "y": 18.95, "theta": 0.0},
             {"id": 1, "release_s": 5.0, "landmark": "b", "x": 30.0, "y": 18.95, "theta": 3.14159}]
    orch = TaskOrchestrator(tasks, dwell_s=2.0)
    sim = FastSim(_tiny_scenario(tmp_path), seed=0, orchestrator=orch, duration=600)
    sim.run()
    assert [t["status"] for t in orch.tasks] == ["done", "done"]
    assert sim.done and sim.now < 600  # ended when the work was done
    c = next(iter(sim.robots.values())).nav.counters
    # 28 -> 40 -> 30 along the corridor: about 22 m (plus parking)
    assert 20.0 < c["dist_total_m"] < 26.0 and c["dist_task_m"] == pytest.approx(c["dist_total_m"], abs=0.5)


def test_scout_drives_checks_returns_and_a_task_preempts_it(tmp_path):
    from mattbot_sim.experiment.orchestrator import TaskOrchestrator
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    # scout_present: cone_9 hidden in the passage north of the corridor, known from a peer, robot idle nearby
    path = os.path.join(PKG, "scenarios", "scout_present.yaml")
    params = FastParams().update({"observe_scout": True, "observe_scout_execute": True})
    sim = FastSim(path, params=params, seed=0, duration=200)
    sim.run()
    s = next(iter(sim.summaries().values()))
    assert s["scouts_started"] >= 1 and s["scouts_reached"] >= 1
    c = next(iter(sim.robots.values())).nav.counters
    assert c.get("dist_scout_m", 0) > 2.0 and c.get("dist_return_m", 0) > 2.0  # out and back

    # The cone is taken away: the scout reports ABSENT and the ledger drops it
    events = [{"at": 1.0, "remove": "cone_9"}]
    with open(path) as f:
        sc = yaml.safe_load(f)
    sc["events"], sc["expect"] = events, {}
    variant = tmp_path / "scout_removed.yaml"
    variant.write_text(yaml.safe_dump(sc))
    sim = FastSim(str(variant), params=FastParams().update({"observe_scout": True, "observe_scout_execute": True}),
                  seed=0, duration=200)
    sim.run()
    s = next(iter(sim.summaries().values()))
    assert s["observation_outcomes_by_kind"]["SCOUT"]["ABSENT"] == 1 and s["correct_removals"] == 1

    # A task released while the robot is out on a scout cancels it
    sim = FastSim(path, params=FastParams().update({"observe_scout": True, "observe_scout_execute": True}),
                  seed=0, duration=200)
    sim.orchestrator.on_start(sim)
    r = next(iter(sim.robots.values()))
    while sim.now < 200 and r.nav.scout is None:
        sim.step()
    assert r.nav.scout is not None
    orch = TaskOrchestrator([{"id": 0, "release_s": 0.0, "landmark": "x", "x": 40.0, "y": 18.95, "theta": 0.0}])
    sim.orchestrator = orch
    orch.on_start(sim)
    sim.step()
    assert r.nav.scout is None and r.nav.goal[:2] == (40.0, 18.95)
    assert r.scorer.summary(sim.now)["scouts_cancelled"] == 1


def test_report_handles_censored_latencies():
    from mattbot_sim.experiment.report import kaplan_meier, km_median, mean_ci

    xs, ys = kaplan_meier([10, 20, 30, 40], [False, True, False, True])
    assert ys[0] == 1.0 and ys[-1] == pytest.approx(0.75 * (1 - 1 / 2))
    assert km_median([10, 20, 30, 40], [False, False, True, True]) == 20
    assert km_median([10, 20, 30], [False, True, True]) == float("inf")  # 2/3 still in the ledger at the end
    m, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and lo <= m <= hi


# ---------- Typed obstacles, horizon, oracle (experiments/rechecking) ----------

MAIN = os.path.join(PKG, "experiments", "rechecking", "main.yaml")


def test_spec_extends_and_horizon_tasks():
    from mattbot_sim.experiment.spec import load_spec

    main, fleet1, busy = (load_spec(os.path.join(PKG, "experiments", "rechecking", n)) for n in
                          ("main.yaml", "fleet1.yaml", "busy.yaml"))
    assert list(fleet1.conditions) == list(main.conditions) and fleet1.common == main.common
    assert len(fleet1.robots) == 1 and fleet1.landmarks == main.landmarks and fleet1.end_at_horizon
    assert fleet1.obstacle_types == main.obstacle_types  # merged key by key, not replaced
    assert main.release_rate_per_min == pytest.approx(0.17 * 3) and fleet1.release_rate_per_min == pytest.approx(0.17)
    assert busy.release_rate_per_min == pytest.approx(0.33 * 3) and busy.tasks["dwell_s"] == main.tasks["dwell_s"]
    pilot = load_spec(SPEC)  # flat obstacle keys: one blocking type
    assert list(pilot.obstacle_types) == ["blocking"] and not pilot.end_at_horizon


@pytest.fixture(scope="module")
def typed():
    from mattbot_sim.experiment.generate import setup as setup_spec
    from mattbot_sim.experiment.spec import load_spec

    spec = load_spec(MAIN)
    ctx, landmarks, cands, conn = setup_spec(spec)
    return spec, ctx, landmarks, cands, conn


def test_typed_candidates_meet_their_definitions(typed):
    spec, ctx, _landmarks, cands, conn = typed
    by = {t: [c for c in cands if c["type"] == t] for t in ("blocking", "on_route", "visible")}
    assert len(by["blocking"]) > 50 and len(by["on_route"]) > 50 and len(by["visible"]) > 20
    o = spec.obstacles
    rng = np.random.default_rng(1)
    for typ, cs in by.items():
        width = spec.obstacle_types[typ]["width_m"]
        for k in rng.choice(len(cs), size=min(15, len(cs)), replace=False):
            c = cs[int(k)]
            assert c["width"] == width and conn.keeps_connected(c["x"], c["y"], width)
            if typ == "blocking":
                assert c["blocked"] == 1
            elif typ == "on_route":
                assert c["blocked"] == 0 and c["I_o"] == 0.0 and c["route_share"] >= o["route_min_trip_share"]
                assert c["off_centre_m"] > 0.0  # beside the centre line, against the wall
            else:
                assert c["route_in_view_m"] >= 1.0


def test_typed_instances(typed):
    from mattbot_sim.experiment.generate import make_instance

    spec, _ctx, landmarks, cands, conn = typed
    for seed in range(4):
        inst = make_instance(spec, seed, cands, landmarks, conn)
        types = [ob["type"] for ob in inst["obstacles"]]
        assert types == ["blocking"] * 2 + ["on_route"] * 2 + ["visible"] * 2
        for ob in inst["obstacles"]:
            cfg = spec.obstacle_types[ob["type"]]
            assert ob["width"] == cfg["width_m"] and ob["class"] in cfg["classes"]
            assert ob["I_o"] >= cfg["min_importance_m"]
            assert cfg["max_importance_m"] is None or ob["I_o"] <= cfg["max_importance_m"]
            assert ob["remove_s"] <= spec.obstacles["appear_s"][1] + spec.obstacles["lifetime_s"][1]
        assert conn.keeps_connected_all([(ob["x"], ob["y"], ob["width"]) for ob in inst["obstacles"]])
        rel = [t["release_s"] for t in inst["tasks"]]
        assert rel and max(rel) <= spec.tasks["until_s"]
        assert 0.5 * spec.release_rate_per_min * 60 < len(rel) < 1.6 * spec.release_rate_per_min * 60  # ~Poisson


def test_oracle_drops_an_obstacle_when_it_is_removed(tmp_path):
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    events = [{"at": 1.0, "add": {"id": "c", "class": "chair", "x": 31.0, "y": 18.3, "width": 0.5}},
              {"at": 30.0, "remove": "c"}]
    path = _tiny_scenario(tmp_path, events)
    for oracle in (False, True):
        sim = FastSim(path, params=FastParams().update({"observe": False, "oracle_removals": oracle}), seed=0,
                      duration=60)
        sim.run()
        scorer = next(iter(sim.robots.values())).scorer
        assert [a["true_object"] for a in scorer.added] == ["c"]  # seen from the start pose
        removals = [r for r in scorer.removals if r["correct"]]
        if oracle:
            assert len(removals) == 1 and removals[0]["t"] == pytest.approx(30.0, abs=1.01)
            assert sim.removal_log[-1][2] == "ORACLE"
        else:
            assert not removals


def test_end_at_horizon_runs_the_full_duration(tmp_path):
    from mattbot_sim.experiment.orchestrator import TaskOrchestrator
    from mattbot_sim.fast.sim import FastSim

    tasks = [{"id": 0, "release_s": 0.0, "landmark": "a", "x": 34.0, "y": 18.95, "theta": 0.0}]
    for horizon in (False, True):
        orch = TaskOrchestrator(tasks, dwell_s=2.0, end_at_horizon=horizon)
        sim = FastSim(_tiny_scenario(tmp_path), seed=0, orchestrator=orch, duration=120)
        sim.run()
        assert orch.tasks[0]["status"] == "done"
        assert (sim.now >= 119.9) == horizon
        c = next(iter(sim.robots.values())).nav.counters
        assert c["time_task_s"] > 5.0 and c["time_task_s"] + c.get("time_other_s", 0.0) == pytest.approx(sim.now, abs=0.2)


def test_scout_looks_only_from_its_viewpoint(tmp_path):
    """near_goal is 0.35 m, but a viewpoint is only guaranteed within LOS_POSE_TOLERANCE_M: the navigator parks
    on the viewpoint (observe_viewpoint_tol_m) before it looks."""
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    # scout_present with cone_9 taken away at once: it stays due, so the scout really looks (and finds it gone)
    with open(os.path.join(PKG, "scenarios", "scout_present.yaml")) as f:
        sc = yaml.safe_load(f)
    sc["events"], sc["expect"] = [{"at": 1.0, "remove": "cone_9"}], {}
    path = tmp_path / "scout_removed.yaml"
    path.write_text(yaml.safe_dump(sc))
    sim = FastSim(str(path), params=FastParams().update({"observe_scout": True, "observe_scout_execute": True}),
                  seed=0, duration=200)
    nav = next(iter(sim.robots.values())).nav
    offsets, original = [], sim.observation_event

    def record(robot, name, kind, object_id, target, *a, **k):
        if name == "STARTED" and nav.scout is not None:
            s = nav.scout["entry"]["stop"]
            offsets.append(math.hypot(robot.pose.x - s.x, robot.pose.y - s.y))
        return original(robot, name, kind, object_id, target, *a, **k)

    sim.observation_event = record
    sim.run()
    assert offsets and max(offsets) <= sim.params.observe_viewpoint_tol_m + 0.06  # (park_pose_thresh 0.05)
    assert next(iter(sim.summaries().values()))["observation_outcomes_by_kind"]["SCOUT"]["ABSENT"] == 1


def test_evaluator_map_check_uses_the_planners_rule():
    from mattbot_sim.fast.checker import Evaluator
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.perception import CameraModel
    from observation_eval.evidence import CameraPose, Target

    blocking = np.zeros((60, 100), dtype=bool)
    blocking[30:, 40] = True  # wall from y = 3.0 up at x = 4.0..4.1 (0.1 m cells)
    ev = Evaluator(FastParams(), CameraModel(), blocking, (0.0, 0.0), 0.1)
    target = Target("chair", 5.0, 3.0, 1.0)
    assert not ev.map_visible(CameraPose(2.05, 3.05, 0.0, 0.2), target)  # the corner hides most of it
    assert ev.map_visible(CameraPose(2.05, 1.55, 0.0, 0.2), target)  # from lower down: in view
    assert Evaluator(FastParams(), CameraModel()).map_visible(CameraPose(2.05, 3.05, 0.0, 0.2), target)  # no map


def test_evaluator_other_robots_occlude():
    """Another robot in front of the object, or around the camera (robots pass through each other), never gives a
    clear frame: the real evaluator sees it in depth (depth_occluded)."""
    from mattbot_sim.fast.checker import Evaluator
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.perception import CameraModel
    from observation_eval.evidence import ABSENT, UNCERTAIN

    def verdict(robots):
        ev = Evaluator(FastParams(), CameraModel())
        ev.start("o1", (3.0, 0.0), 0.0, [])
        ev.frame([], (0.0, 0.0, 0.0, 0.2), [], robots)
        return ev.windows["o1"].frames[-1]

    assert verdict([]).verdict == ABSENT  # nothing seen where the object was: clear
    assert verdict([(1.5, 0.0, 0.2)]).verdict == UNCERTAIN  # a robot between camera and object
    assert verdict([(0.05, 0.0, 0.2)]).verdict == UNCERTAIN  # the camera inside another robot's disc
    assert verdict([(0.0, 2.0, 0.2)]).verdict == ABSENT  # a robot off to the side does not hide it


class _FakeNav:
    def __init__(self, can_plan):
        self.mode, self.counters, self.can_plan, self.goal = "IDLE", {"plan_failures": 0}, can_plan, None

    def stop(self):
        pass


class _FakeRobot:
    def __init__(self, rid, x, can_plan):
        self.id, self.nav, self.idle_since = rid, _FakeNav(can_plan), 0.0
        self.pose = type("P", (), {"x": x, "y": 0.0})()


class _FakeSim:
    """Just enough of FastSim for TaskOrchestrator: robots that reach goals instantly or cannot plan."""

    def __init__(self, robots):
        self.robots = {r.id: r for r in robots}
        self.now = self.elapsed = 0.0
        self.done = False
        self.events = []

    def set_goal(self, rid, x, y, theta):
        r = self.robots[rid]
        if r.nav.can_plan:
            r.pose.x, r.pose.y = x, y
        else:
            r.nav.counters["plan_failures"] += 1  # "Planning failed; clearing the goal"
        r.idle_since = self.now

    def runner_event(self, robot, text):
        self.events.append((robot.id, text))

    def tick(self, orch, seconds):
        for _ in range(int(seconds * 10)):
            orch.on_tick(self)
            self.now = self.elapsed = self.now + 0.1


def test_orchestrator_hands_a_stuck_robots_task_to_another():
    from mattbot_sim.experiment.orchestrator import TaskOrchestrator

    task = {"id": 0, "release_s": 0.0, "landmark": "a", "x": 10.0, "y": 0.0, "theta": 0.0}
    # Robot 1 (nearest) cannot plan; robot 2 can: the task is handed over and done, robot 1 rests
    sim = _FakeSim([_FakeRobot(1, 9.0, False), _FakeRobot(2, 0.0, True)])
    orch = TaskOrchestrator([task], dwell_s=0.0, stuck_backoff_s=120.0)
    orch.on_start(sim)
    sim.tick(orch, 60)
    t = orch.tasks[0]
    assert t["status"] == "done" and t["robot"] == 2 and t["handoffs"] == 1 and orch.stuck_events == 1
    assert not orch.available(sim, 1) and orch.state[1]["stuck_until"] > sim.now
    # Nobody can plan: handed off max_handoffs times, then it fails
    sim = _FakeSim([_FakeRobot(1, 9.0, False), _FakeRobot(2, 0.0, False)])
    orch = TaskOrchestrator([dict(task)], dwell_s=0.0, stuck_backoff_s=5.0, max_handoffs=2)
    orch.on_start(sim)
    sim.tick(orch, 200)
    assert orch.tasks[0]["status"] == "failed" and orch.tasks[0]["handoffs"] == 2
    # A robot that plans but stops short (not stuck) fails the task as before
    sim = _FakeSim([_FakeRobot(1, 9.0, True)])
    sim.set_goal = lambda rid, x, y, th: setattr(sim.robots[rid], "idle_since", sim.now)  # never gets there
    orch = TaskOrchestrator([dict(task)], dwell_s=0.0)
    orch.on_start(sim)
    sim.tick(orch, 60)
    assert orch.tasks[0]["status"] == "failed" and orch.tasks[0]["handoffs"] == 0


def test_detour_deadline_covers_the_planned_drive():
    """KNOWN_ISSUES 12.9: the deadline was 2 x the detour's extra cost + 20 s, too short for a detour that leaves
    the path well before its viewpoint; it now also covers the planned drive there, the align pause and the looks."""
    from mattbot_sim.fast.params import FastParams
    from mattbot_sim.fast.sim import FastSim

    path = os.path.join(PKG, "scenarios", "detour_present.yaml")
    sim = FastSim(path, params=FastParams().update({"observe_detour": True}), seed=0, duration=300)
    nav = next(iter(sim.robots.values())).nav
    seen, original = [], nav.start_detour

    def record():
        original()
        if nav.detour is not None and nav.mode == "ALIGN":
            seen.append((sim.now, nav.detour["deadline"], nav.duration, len(nav.detour["stop"].object_ids)))

    nav.start_detour = record
    sim.run()
    p = sim.params
    assert seen
    for now, deadline, duration, n in seen:
        needed = duration + p.post_align_pause_s + p.observe_dwell_s * n
        assert deadline >= now + p.observe_detour_timeout_factor * needed + 20.0 - 1e-6


def test_runner_saves_each_run_and_resumes(tmp_path, monkeypatch):
    """runs/<condition>_seed<k>.json is written as each run finishes; running again skips those (an interrupted
    experiment resumes), and a directory made from another spec is refused."""
    import shutil
    from collections import OrderedDict

    from mattbot_sim.experiment import runner

    spec_path = tmp_path / "pilot.yaml"
    shutil.copy(SPEC, spec_path)
    shutil.copy(os.path.join(PKG, "experiments", "landmarks_current_map.yaml"), tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda spec, out, seeds, cache: ({s: ({"seed": s}, "x.yaml") for s in seeds},))
    calls = []

    def fake_run_one(job):
        _path, condition, seed = job[:3]
        calls.append((condition, seed))
        if len(calls) == 3 and fail[0]:
            raise RuntimeError("interrupted")
        return OrderedDict(condition=condition, seed=seed, x=1), [], []

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    out = tmp_path / "out"
    fail = [True]
    with pytest.raises(RuntimeError):
        runner.run(str(spec_path), str(out), conditions=["none", "detour"], seeds=[0, 1], jobs=1)
    assert sorted(os.listdir(out / "runs")) == ["detour_seed0.json", "none_seed0.json"]  # the two that finished
    fail[0] = False
    calls.clear()
    results = runner.run(str(spec_path), str(out), conditions=["none", "detour"], seeds=[0, 1], jobs=1)
    assert sorted(calls) == [("detour", 1), ("none", 1)] and len(results) == 4  # only the missing runs
    assert len(list(csv_rows(out / "runs.csv"))) == 4
    prov = json.load(open(out / "provenance.json"))
    assert len(prov["starts"]) == 2 and "mattbot_sim" in prov["starts"][0]["git"]
    spec_path.write_text(spec_path.read_text().replace("seeds: 5", "seeds: 6"))
    with pytest.raises(RuntimeError, match="different spec"):
        runner.run(str(spec_path), str(out), conditions=["none"], seeds=[0], jobs=1)


def csv_rows(path):
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))
