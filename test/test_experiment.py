"""Experiments (mattbot_sim/experiment): instances, task orchestration, odometry, scout execution and the report."""

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
